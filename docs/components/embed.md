# Embedder

## Purpose

The Embedder module (`src/embed/mod.rs`) loads the `sentence-transformers/all-MiniLM-L6-v2`
BERT model via candle (HuggingFace's pure-Rust deep-learning runtime) and converts arbitrary
text strings into 384-dimensional L2-normalized vectors. It is the server-side embedding
layer that lets MCP clients submit plain text instead of pre-computed floats.

The model identity is hardcoded for the whole project (YAGNI: no configuration knob until a
second model is needed in production).

## Public API

```rust
// src/embed/mod.rs

/// Canonical HuggingFace identifier of the loaded model. Public so callers
/// (e.g. the `stats` MCP tool) can report it without duplicating the string.
pub const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";

pub struct Embedder { /* private: tokenizer, BertModel, device */ }

impl Embedder {
    /// Synchronous. Downloads ~90 MB of model weights on first run; uses
    /// mmap'ed cache on subsequent runs (~30 ms warm start).
    /// Must be called off the async reactor — use tokio::task::spawn_blocking.
    pub fn load() -> Result<Self, EmbedError>;

    /// Always 384 for the hardcoded all-MiniLM-L6-v2 model.
    pub fn dimension(&self) -> usize;

    /// Returns `MODEL_NAME` — the canonical HuggingFace identifier.
    pub fn model_name(&self) -> &'static str;

    /// Encode text into a 384-dim L2-normalized vector. Read-only after load();
    /// safe to call concurrently from multiple threads.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}

#[derive(Debug)]
pub enum EmbedError {
    DownloadFailed(String),    // HF Hub network / auth failure
    TokenizeFailed(String),    // tokenizer could not encode the input
    ModelLoadFailed(String),   // config.json / safetensors failed to load
    InferenceFailed(String),   // candle forward pass error
}
```

All `EmbedError` variants implement `Display` and `std::error::Error`. No `Box<dyn Error>`
is used.

## Embedding Pipeline

```
text (String)
  └─► Tokenizer::encode(text, add_special_tokens=true)
        → token_ids: &[u32]      (includes [CLS] and [SEP])
        → attention_mask: &[u32]

  └─► BertModel::forward(input_ids, token_type_ids=zeros, attention_mask)
        → hidden: Tensor [1, seq_len, 384]

  └─► Mean pooling (attention-mask weighted):
        masked    = hidden * broadcast(mask_f32.unsqueeze(2))
        summed    = masked.sum(dim=1)      → [1, 384]
        counts    = mask_f32.sum(dim=1)    → [1, 1]
        pooled    = summed / counts        → [1, 384]

  └─► L2 normalization:
        norm       = sqrt(sum(pooled^2, dim=1, keepdim=true))
        normalized = pooled / norm         → [1, 384]

  └─► squeeze(0).to_vec1::<f32>()         → Vec<f32>(384)
```

The mean-pool step weights each token's hidden state by whether it is a real token
(`mask=1`) or padding (`mask=0`), so variable-length inputs produce a consistent
384-dim summary.

## Why This Model

| Property | Value |
|----------|-------|
| Model ID | `sentence-transformers/all-MiniLM-L6-v2` |
| Layers | 6 (MiniLM variant of BERT-base) |
| Hidden / output dim | 384 |
| Weights file | `model.safetensors` (~90 MB download) |
| Pre-training objective | Sentence similarity (MS MARCO, NLI, etc.) |
| Normalization | L2-normalized output by convention in sentence-transformers |
| Runtime | pure-Rust via candle — no Python, no ONNX, no C++ |

384-dim is a good size/quality trade-off: small enough for fast brute-force KNN at
typical agent working-memory scales (hundreds to thousands of documents), large enough
to retain semantic nuance. Because the model normalizes to the unit sphere, cosine
distance between any two embeddings is in `[0, 2]` and conceptually clean.

## Concurrency

`Embedder` is `Send + Sync`. After `load()` completes, all fields are read-only:
- `tokenizer` — `tokenizers::Tokenizer` holds its vocabulary in memory; `encode` is
  a shared-reference method.
- `model` — `candle_transformers::models::bert::BertModel`; candle tensor operations on
  CPU are side-effect-free for the model weights.
- `device` — `candle_core::Device::Cpu`; CPU device has no mutable runtime state.

In the MCP server, `Embedder` is held under `Arc<Embedder>` inside `NanoVecState`. Tool
handlers that need to embed text use a **two-phase lock pattern** to avoid serializing
concurrent BERT forward passes behind the state Mutex:

```
Phase 1 (locked): clone Arc<Embedder> from NanoVecState → release state lock
Phase 2 (unlocked): call embedder.embed(text) → BERT forward runs without any lock held
Phase 3 (locked): re-acquire state lock → insert vector + record
```

This ensures that two simultaneous `index_document` requests can overlap their BERT
forward passes while still atomically inserting into `VectorStore` and `RecordStore`.

## Network and Caching

On first run, `Embedder::load()` fetches three files from HuggingFace Hub via `hf-hub`:

| File | Size (approx.) |
|------|----------------|
| `config.json` | ~1 KB |
| `tokenizer.json` | ~230 KB |
| `model.safetensors` | ~90 MB |

Downloaded artifacts are stored in `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/`.
Subsequent calls to `load()` mmap the safetensors file directly from cache (~30 ms, no network
required).

The server logs its loading state to stderr so the binary does not appear hung during cold start:

```
INFO nanovec::server: loading embedder (sentence-transformers/all-MiniLM-L6-v2, ~90MB on first run)
INFO nanovec::server: embedder loaded dimension=384
```

If `HF_HOME` or `HUGGINGFACE_HUB_CACHE` is set, `hf-hub` respects those environment
variables for the cache location.

## Safety

`load()` contains exactly one `unsafe` block:

```rust
// SAFETY: `from_mmaped_safetensors` memory-maps `weights_path` read-only.
// We never mutate the file (the OS enforces RO mmap), and the resulting
// mmap lives inside the returned `BertModel` for its entire lifetime —
// tensor views into it remain valid as long as `self.model` exists.
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
        ...
};
```

The mmap is read-only and its lifetime is tied to `BertModel` (which is owned by
`Embedder`). No other `unsafe` blocks exist in this module.

## Performance Characteristics

| Operation | Observed time |
|-----------|---------------|
| `load()` cold (network + mmap) | 10-30 s (network-bound) |
| `load()` warm (HF cache, mmap) | ~30 ms |
| `embed()` single text, CPU | ~5-50 ms depending on text length |
| `dimension()` | O(1) — constant return |

No SIMD acceleration is applied to the BERT forward pass in Phase 2; candle uses
standard scalar operations on CPU. Phase 3 will add SIMD to the NanoVec distance
functions (not the BERT forward pass itself).

## Model Limitations

`all-MiniLM-L6-v2` was trained predominantly on English data. Empirical use
on real corpora has surfaced two practical limits worth flagging at the API
boundary:

1. **Weak on non-English and mixed-language text.** Russian, Ukrainian, and
   other non-Latin-script queries produce noticeably flatter score
   distributions than English. In one observed case, a Russian query
   ("полная бизнес воронка...") missed its target chunk by 4 ranks; the same
   query reformulated in English with corpus-aligned terms moved the target
   to rank 1 with a 0.247 cosine-distance gap to the runner-up. This is a
   model limitation, not a bug in NanoVec's search.

2. **Lexical sensitivity.** The 384-dim embedding has limited capacity to
   distinguish semantically close phrasings when the corpus uses unusual or
   technical vocabulary. Common content words (e.g. "Camera Permission",
   "Calibration") dominate the signal and can pull queries toward chunks
   that share surface vocabulary but diverge in topic.

### Recommended mitigations (in order of ROI)

| Mitigation | What to do | Cost |
|------------|------------|------|
| Breadcrumbs on chunks | Prepend the section path to each chunk text before indexing (`"Section 2.2 Funnel: ..."`) | 0 — keep the same chunks |
| Query rewriting | Have an LLM reformulate the user query into corpus-aligned vocabulary before calling `search_document` | 1 extra LLM call per query |
| Hierarchical chunking | Split heavy sections into sub-sections; keep the parent section name as a breadcrumb | 2-3× chunk count |

For workloads where non-English content is dominant, the right fix is a
multilingual embedding model (e5-multilingual, bge-m3). Replacing the model is
out of scope for Phase 2.5; tracked as a future phase.

## Dependencies

- `candle-core` 0.10 — tensor operations, `Device`, `DType`
- `candle-nn` 0.10 — `VarBuilder`
- `candle-transformers` 0.10 — `BertModel`, `Config`
- `tokenizers` 0.21 — `Tokenizer`, `Encoding`
- `hf-hub` 0.3 (feature `tokio`) — HuggingFace Hub download/cache API
- `serde_json` — parsing `config.json`

No other NanoVec modules depend on Embedder except `src/mcp/mod.rs` (via `Arc<Embedder>`)
and `src/server/mod.rs` (startup wiring).

## Test Coverage

8 tests in `src/embed/mod.rs` under `#[cfg(test)]`:

Tests share a single `static EMBEDDER: Lazy<Embedder>` (via `once_cell`) so the model
loads and downloads at most once per `cargo test` run.

| Test | Type | Invariant |
|------|------|-----------|
| `embedder_loads_and_embeds_to_384_dim` | unit | `embed("hello world").len() == 384`; `dimension() == 384` |
| `embedding_is_deterministic` | unit | Two identical inputs produce bit-identical output (delta < 1e-6 per component) |
| `semantic_ordering_simple` | unit | `cos_sim("dog","puppy") > cos_sim("dog","spaceship")` |
| `model_name_const_matches_canonical_repo` | unit | `MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"` |
| `model_name_accessor_returns_const` | unit | `EMBEDDER.model_name() == MODEL_NAME` |
| `embedding_is_l2_normalized` | proptest (16 cases) | `‖embed(text)‖₂ ≈ 1.0 ± 1e-3` for random alphanumeric texts |
| `embedding_dim_always_384` | proptest (16 cases) | `embed(text).len() == 384` for random alphanumeric texts |

The proptest case count is capped at 16 (down from the default 256) because each case
runs a full BERT forward pass.
