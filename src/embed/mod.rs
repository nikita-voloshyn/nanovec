//! Server-side text embedding via candle.
//!
//! Hosts a pre-trained `sentence-transformers/all-MiniLM-L6-v2` BERT model
//! (6 layers, 384-dim hidden) and turns text into L2-normalized 384-dim
//! vectors via mean-pooling over the last hidden state with attention mask.
//!
//! The model identity is hardcoded for the whole project (YAGNI: no model
//! configuration knob until a second concrete use case appears).

use std::fmt;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

/// Hardcoded embedding dimension for the chosen MiniLM model.
const EMBEDDING_DIM: usize = 384;

/// Canonical HuggingFace identifier for the embedding model. Pinned to the
/// `sentence-transformers/all-MiniLM-L6-v2` checkpoint and exposed publicly so
/// callers (e.g. `stats` MCP tool) can report which model is loaded without
/// duplicating the string.
pub const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Loaded text embedder. Cheap to share via `Arc` — `embed(&self, ...)` is
/// read-only after [`Embedder::load`].
pub struct Embedder {
    tokenizer: Tokenizer,
    model: BertModel,
    device: Device,
}

/// Errors emitted by the embedding pipeline.
///
/// Exhaustive enum per CLAUDE.md rule "no `Box<dyn Error>` in public APIs".
/// String payloads carry the upstream error text (candle / tokenizers /
/// hf-hub) so the caller has actionable diagnostics without us re-exporting
/// foreign error types.
#[derive(Debug)]
pub enum EmbedError {
    /// HF Hub download failed (network, auth, missing file).
    DownloadFailed(String),
    /// Tokenizer could not encode the input text.
    TokenizeFailed(String),
    /// Model weights or config could not be loaded into candle.
    ModelLoadFailed(String),
    /// Forward pass failed inside candle.
    InferenceFailed(String),
}

impl fmt::Display for EmbedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedError::DownloadFailed(msg) => {
                write!(f, "embedder download failed: {msg}")
            }
            EmbedError::TokenizeFailed(msg) => {
                write!(f, "embedder tokenization failed: {msg}")
            }
            EmbedError::ModelLoadFailed(msg) => {
                write!(f, "embedder model load failed: {msg}")
            }
            EmbedError::InferenceFailed(msg) => {
                write!(f, "embedder inference failed: {msg}")
            }
        }
    }
}

impl std::error::Error for EmbedError {}

impl Embedder {
    /// Load the embedder. Synchronous: downloads weights on first run, hits
    /// the HF cache on subsequent runs. Caller is responsible for executing
    /// this off the async reactor (e.g. `tokio::task::spawn_blocking`).
    pub fn load() -> Result<Self, EmbedError> {
        let device = Device::Cpu;

        let api = Api::new().map_err(|e| EmbedError::DownloadFailed(e.to_string()))?;
        let repo = api.model(MODEL_NAME.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| EmbedError::DownloadFailed(format!("config.json: {e}")))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| EmbedError::DownloadFailed(format!("tokenizer.json: {e}")))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| EmbedError::DownloadFailed(format!("model.safetensors: {e}")))?;

        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbedError::ModelLoadFailed(format!("read config.json: {e}")))?;
        let config: Config = serde_json::from_str(&config_json)
            .map_err(|e| EmbedError::ModelLoadFailed(format!("parse config.json: {e}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::ModelLoadFailed(format!("tokenizer load: {e}")))?;

        // SAFETY: `from_mmaped_safetensors` memory-maps `weights_path` read-only.
        // We never mutate the file (the OS enforces RO mmap), and the resulting
        // mmap lives inside the returned `BertModel` for its entire lifetime —
        // tensor views into it remain valid as long as `self.model` exists.
        // The path comes from hf-hub's local cache; concurrent writers would be
        // a cache poisoning bug outside this module's contract.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| EmbedError::ModelLoadFailed(format!("safetensors mmap: {e}")))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| EmbedError::ModelLoadFailed(format!("BertModel::load: {e}")))?;

        Ok(Self {
            tokenizer,
            model,
            device,
        })
    }

    /// Embedding output dimension. Fixed at 384 for MiniLM-L6-v2.
    pub fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Canonical HuggingFace identifier of the loaded model.
    pub fn model_name(&self) -> &'static str {
        MODEL_NAME
    }

    /// Encode `text` into an L2-normalized 384-dimensional vector.
    ///
    /// Pipeline: tokenize (with `[CLS]`/`[SEP]`) → `BertModel::forward`
    /// → mean-pool over the sequence dimension weighted by the attention
    /// mask → L2-normalize.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbedError::TokenizeFailed(e.to_string()))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let seq_len = ids.len();

        let input_ids = Tensor::from_slice(ids, (1, seq_len), &self.device)
            .map_err(|e| EmbedError::InferenceFailed(format!("input_ids tensor: {e}")))?;
        let attention_mask = Tensor::from_slice(mask, (1, seq_len), &self.device)
            .map_err(|e| EmbedError::InferenceFailed(format!("attention_mask tensor: {e}")))?;
        let token_type_ids = input_ids
            .zeros_like()
            .map_err(|e| EmbedError::InferenceFailed(format!("token_type_ids zeros: {e}")))?;

        // BertModel::forward signature in candle-transformers 0.10:
        //   forward(&self, input_ids: &Tensor, token_type_ids: &Tensor,
        //           attention_mask: Option<&Tensor>) -> Result<Tensor>
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| EmbedError::InferenceFailed(format!("bert forward: {e}")))?;

        // hidden: [1, seq_len, 384]. Mean-pool with attention mask.
        let mask_f32 = attention_mask
            .to_dtype(DType::F32)
            .and_then(|t| t.unsqueeze(2))
            .map_err(|e| EmbedError::InferenceFailed(format!("mask cast/unsqueeze: {e}")))?;

        let masked = hidden
            .broadcast_mul(&mask_f32)
            .map_err(|e| EmbedError::InferenceFailed(format!("apply mask: {e}")))?;

        let summed = masked
            .sum(1)
            .map_err(|e| EmbedError::InferenceFailed(format!("sum over seq: {e}")))?;
        let counts = mask_f32
            .sum(1)
            .map_err(|e| EmbedError::InferenceFailed(format!("sum mask: {e}")))?;

        let pooled = summed
            .broadcast_div(&counts)
            .map_err(|e| EmbedError::InferenceFailed(format!("mean pool: {e}")))?;

        // L2-normalize along the embedding dimension. `pooled`: [1, 384].
        let norm = pooled
            .sqr()
            .and_then(|t| t.sum_keepdim(1))
            .and_then(|t| t.sqrt())
            .map_err(|e| EmbedError::InferenceFailed(format!("norm compute: {e}")))?;
        let normalized = pooled
            .broadcast_div(&norm)
            .map_err(|e| EmbedError::InferenceFailed(format!("normalize: {e}")))?;

        let vec = normalized
            .squeeze(0)
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| EmbedError::InferenceFailed(format!("to_vec1: {e}")))?;

        if vec.len() != EMBEDDING_DIM {
            return Err(EmbedError::InferenceFailed(format!(
                "unexpected output dim: {} (want {EMBEDDING_DIM})",
                vec.len(),
            )));
        }

        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use proptest::prelude::*;

    /// Shared embedder across all tests in this module so the model loads
    /// (and downloads) only once per `cargo test` run.
    static EMBEDDER: Lazy<Embedder> =
        Lazy::new(|| Embedder::load().expect("test embedder must load"));

    /// Cosine similarity for already-L2-normalized vectors == dot product.
    fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    #[test]
    fn embedder_loads_and_embeds_to_384_dim() {
        let v = EMBEDDER.embed("hello world").expect("embed");
        assert_eq!(v.len(), 384);
        assert_eq!(EMBEDDER.dimension(), 384);
    }

    #[test]
    fn model_name_const_matches_canonical_repo() {
        assert_eq!(MODEL_NAME, "sentence-transformers/all-MiniLM-L6-v2");
    }

    #[test]
    fn model_name_accessor_returns_const() {
        assert_eq!(EMBEDDER.model_name(), MODEL_NAME);
    }

    #[test]
    fn embedding_is_deterministic() {
        let a = EMBEDDER.embed("the quick brown fox").expect("embed a");
        let b = EMBEDDER.embed("the quick brown fox").expect("embed b");
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "non-determinism at index {i}: {x} vs {y}",
            );
        }
    }

    #[test]
    fn semantic_ordering_simple() {
        let dog = EMBEDDER.embed("dog").expect("embed dog");
        let puppy = EMBEDDER.embed("puppy").expect("embed puppy");
        let spaceship = EMBEDDER.embed("spaceship").expect("embed spaceship");

        let dog_puppy = cos_sim(&dog, &puppy);
        let dog_spaceship = cos_sim(&dog, &spaceship);

        assert!(
            dog_puppy > dog_spaceship,
            "expected cos_sim(dog,puppy)={dog_puppy} > cos_sim(dog,spaceship)={dog_spaceship}",
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            // Each case runs a full BERT forward pass; keep the case count
            // modest so `cargo test` finishes in reasonable time.
            cases: 16,
            .. ProptestConfig::default()
        })]

        #[test]
        fn embedding_is_l2_normalized(text in "[a-zA-Z0-9 .,!?]{1,200}") {
            let v = EMBEDDER.embed(&text).expect("embed");
            prop_assert_eq!(v.len(), 384);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            prop_assert!(
                (norm - 1.0).abs() < 1e-3,
                "‖embed(text)‖₂ = {} not within 1e-3 of 1.0 (text: {:?})",
                norm,
                text,
            );
        }

        #[test]
        fn embedding_dim_always_384(text in "[a-zA-Z0-9 .,!?]{1,200}") {
            let v = EMBEDDER.embed(&text).expect("embed");
            prop_assert_eq!(v.len(), 384);
        }
    }
}
