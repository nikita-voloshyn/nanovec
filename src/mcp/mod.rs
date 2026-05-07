pub mod tools;

use std::sync::{Arc, Mutex};

use rmcp::handler::server::router::Router;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_router, ServerHandler};

use crate::distance::Metric;
use crate::embed::Embedder;
use crate::index::brute::{self, BruteForce};
use crate::store::record::RecordStore;
use crate::store::VectorStore;

use self::tools::{
    DeleteParams, IndexDocumentParams, IndexVectorParams, SearchDocumentParams, SearchParams,
};

/// Shared mutable state for the NanoVec MCP server.
///
/// The `store` is initialized at construction time with the embedder's
/// dimension; the dim is locked for the lifetime of the process. The
/// `embedder` is held read-only via `Arc` so handlers can call `embed`
/// without taking the state lock.
pub struct NanoVecState {
    store: VectorStore,
    records: RecordStore,
    metric: Metric,
    embedder: Arc<Embedder>,
}

impl NanoVecState {
    pub fn new(metric: Metric, embedder: Arc<Embedder>) -> Self {
        Self {
            store: VectorStore::new(embedder.dimension()),
            records: RecordStore::new(),
            metric,
            embedder,
        }
    }
}

/// The MCP server handler. Holds shared state behind `Arc<Mutex<>>`.
pub struct NanoVecServer {
    state: Arc<Mutex<NanoVecState>>,
}

impl NanoVecServer {
    pub fn new(metric: Metric, embedder: Arc<Embedder>) -> Self {
        Self {
            state: Arc::new(Mutex::new(NanoVecState::new(metric, embedder))),
        }
    }

    /// Build the Router that wires up all tool routes and the server handler.
    pub fn into_router(self) -> Router<Self> {
        let tool_routes = Self::tool_router();
        Router::new(self).with_tools(tool_routes)
    }
}

/// Parse metadata from an optional JSON value into a Vec of key-value pairs.
fn parse_metadata(value: Option<serde_json::Value>) -> Vec<(String, String)> {
    match value {
        Some(serde_json::Value::Object(map)) => map
            .into_iter()
            .map(|(k, v)| {
                let val = match v {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                };
                (k, val)
            })
            .collect(),
        _ => vec![],
    }
}

/// Parse a metric string into a Metric enum, falling back to a default.
fn parse_metric(s: &Option<String>, default: Metric) -> Result<Metric, String> {
    match s {
        None => Ok(default),
        Some(m) => match m.as_str() {
            "euclidean" => Ok(Metric::Euclidean),
            "cosine" => Ok(Metric::Cosine),
            "dot" => Ok(Metric::DotProduct),
            other => Err(format!("unknown metric: {other}")),
        },
    }
}

#[tool_router]
impl NanoVecServer {
    #[tool(
        name = "index_vector",
        description = "Index a document with its embedding vector and optional metadata"
    )]
    fn index_vector(
        &self,
        Parameters(params): Parameters<IndexVectorParams>,
    ) -> Result<String, String> {
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        if params.vector.is_empty() {
            return Err("vector must not be empty".to_string());
        }

        let dim = params.vector.len();
        let expected = state.store.dimension();
        if dim != expected {
            return Err(format!(
                "vector dim mismatch: expected {expected} (locked by embedder), got {dim}"
            ));
        }

        let offset = state
            .store
            .insert(&params.vector)
            .map_err(|e| format!("insert error: {e}"))?;

        let metadata = parse_metadata(params.metadata);
        let id = state.records.insert(params.text, metadata, offset);

        tracing::debug!(id, offset, "indexed vector");
        Ok(serde_json::json!({ "id": id }).to_string())
    }

    #[tool(
        name = "index_document",
        description = "Index a text document — server computes the embedding via the configured model"
    )]
    fn index_document(
        &self,
        Parameters(params): Parameters<IndexDocumentParams>,
    ) -> Result<String, String> {
        if params.text.trim().is_empty() {
            return Err("text must not be empty".to_string());
        }

        // Two-phase locking per Design Decision D5: clone the `Arc<Embedder>`
        // under the state Mutex, then drop the lock before running the BERT
        // forward pass. Holding the lock across embedding would serialize
        // every concurrent request and defeat the purpose of `Arc<Embedder>`.
        let embedder = {
            let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
            Arc::clone(&state.embedder)
        };

        let vector = embedder
            .embed(&params.text)
            .map_err(|e| format!("embed error: {e}"))?;

        // Re-acquire the lock for the store + records mutation.
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        let offset = state
            .store
            .insert(&vector)
            .map_err(|e| format!("insert error: {e}"))?;

        let metadata = parse_metadata(params.metadata);
        let id = state.records.insert(params.text, metadata, offset);

        tracing::debug!(id, offset, "indexed document");
        Ok(serde_json::json!({ "id": id }).to_string())
    }

    #[tool(
        name = "search",
        description = "Search for the k nearest vectors to a query vector"
    )]
    fn search(&self, Parameters(params): Parameters<SearchParams>) -> Result<String, String> {
        let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        let metric = parse_metric(&params.metric, state.metric)?;

        let results = BruteForce::search(
            &state.store,
            &state.records,
            &params.vector,
            params.k,
            metric,
        );

        let json_results: Vec<serde_json::Value> = results
            .into_iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "score": r.score,
                    "text": r.text,
                })
            })
            .collect();

        Ok(serde_json::json!(json_results).to_string())
    }

    #[tool(
        name = "search_document",
        description = "Semantic search by query text — server computes the query embedding. Default metric is cosine (well-suited for the L2-normalized embedded path)."
    )]
    fn search_document(
        &self,
        Parameters(params): Parameters<SearchDocumentParams>,
    ) -> Result<String, String> {
        if params.query.trim().is_empty() {
            return Err("query must not be empty".to_string());
        }

        // Two-phase locking per Design Decision D5: clone the `Arc<Embedder>`
        // under the state Mutex, then drop the lock before running the BERT
        // forward pass.
        let embedder = {
            let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
            Arc::clone(&state.embedder)
        };

        let query_vec = embedder
            .embed(&params.query)
            .map_err(|e| format!("embed error: {e}"))?;

        // Re-acquire the lock for the KNN scan. Default metric for the embedded
        // path is Cosine (D3): the all-MiniLM model emits L2-normalized vectors,
        // so cosine == 1 - dot, but cosine has the cleanest [0, 2] semantics.
        let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
        let metric = parse_metric(&params.metric, Metric::Cosine)?;

        let results =
            BruteForce::search(&state.store, &state.records, &query_vec, params.k, metric);

        let json_results: Vec<serde_json::Value> = results
            .into_iter()
            .map(|r| {
                // Convert metadata Vec<(K, V)> into a JSON object so the wire
                // shape mirrors how clients pass metadata into index_document.
                let metadata: serde_json::Map<String, serde_json::Value> = r
                    .metadata
                    .into_iter()
                    .map(|(k, v)| (k, serde_json::Value::String(v)))
                    .collect();
                serde_json::json!({
                    "id": r.id,
                    "score": r.score,
                    "text": r.text,
                    "metadata": metadata,
                })
            })
            .collect();

        Ok(serde_json::json!(json_results).to_string())
    }

    #[tool(name = "delete", description = "Delete a document by its ID")]
    fn delete(&self, Parameters(params): Parameters<DeleteParams>) -> Result<String, String> {
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        let NanoVecState {
            store,
            records,
            metric: _,
            embedder: _,
        } = &mut *state;

        brute::delete(store, records, params.id).map_err(|e| format!("{e}"))?;

        tracing::debug!(id = params.id, "deleted vector");
        Ok(serde_json::json!({ "success": true }).to_string())
    }

    #[tool(
        name = "clear",
        description = "Remove all indexed documents. Resets the ID counter so the next inserted document gets id=0. Dimension lock is preserved."
    )]
    fn clear(&self) -> Result<String, String> {
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
        let deleted = state.records.count();
        state.store.clear();
        state.records.clear();
        tracing::info!(count = deleted, "cleared store");
        Ok(serde_json::json!({ "deleted": deleted }).to_string())
    }

    #[tool(name = "stats", description = "Get database statistics")]
    fn stats(&self) -> Result<String, String> {
        let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        // Dimension is locked at server start by the embedder, so it is always
        // present. We keep the field name `dimension` for backward compat with
        // Phase 1 clients.
        let dimension = state.store.dimension();
        let count = state.records.count();
        let metric = format!("{:?}", state.metric);

        Ok(serde_json::json!({
            "count": count,
            "dimension": dimension,
            "metric": metric,
        })
        .to_string())
    }
}

impl ServerHandler for NanoVecServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("nanovec", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "NanoVec is an in-memory vector database for ephemeral AI agent working memory."
                    .to_string(),
            )
    }
}
