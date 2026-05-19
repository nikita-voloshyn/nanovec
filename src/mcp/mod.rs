pub mod tools;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rmcp::handler::server::router::Router;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_router, ServerHandler};

use crate::distance::Metric;
use crate::embed::Embedder;
use crate::index::brute::{self, BruteForce};
use crate::store::collections::{Collection, CollectionMap, DEFAULT_COLLECTION};

use self::tools::{
    ClearParams, CreateCollectionParams, DeleteParams, DropCollectionParams, IndexDocumentParams,
    IndexVectorParams, SearchDocumentParams, SearchParams,
};

/// Shared mutable state for the NanoVec MCP server.
///
/// Holds a [`CollectionMap`] containing one or more named collections. The
/// `"default"` collection is materialized lazily — on the first legacy call
/// that omits a `collection` argument — using the embedder's dimension as the
/// dimension lock. This preserves the Phase 1/2/2.5 wire shape: existing
/// clients keep working without code changes.
///
/// The `embedder` is held read-only via `Arc` so handlers can drop the state
/// `Mutex` before running the (slow) BERT forward pass.
///
/// ### Lock discipline
///
/// One `Arc<Mutex<NanoVecState>>` guards all collections (YAGNI per CLAUDE.md
/// — per-collection locks add complexity that no current workload justifies).
/// The pre-existing two-phase locking pattern for `index_document` /
/// `search_document` is preserved.
///
/// ### `drop_collection("default")`
///
/// Allowed. The next legacy call simply re-materializes a fresh `"default"`
/// collection, keeping semantics simple (no special-case error to teach
/// callers).
pub struct NanoVecState {
    collections: CollectionMap,
    metric: Metric,
    embedder: Arc<Embedder>,
}

impl NanoVecState {
    pub fn new(metric: Metric, embedder: Arc<Embedder>) -> Self {
        Self {
            collections: CollectionMap::new(),
            metric,
            embedder,
        }
    }

    /// Resolve a `collection` argument to a mutable collection reference,
    /// auto-creating `"default"` with the embedder dimension when the caller
    /// passes `None`. Named collections must already exist (use
    /// `create_collection` first) — this method does NOT auto-create them.
    fn resolve_mut(&mut self, name: Option<&str>) -> Result<&mut Collection, String> {
        let dim = self.embedder.dimension();
        match name {
            None => Ok(self.collections.get_or_create(DEFAULT_COLLECTION, dim)),
            Some(n) if n == DEFAULT_COLLECTION => {
                Ok(self.collections.get_or_create(DEFAULT_COLLECTION, dim))
            }
            Some(n) => self
                .collections
                .get_mut(n)
                .ok_or_else(|| format!("collection not found: {n}")),
        }
    }

    /// Read-only counterpart of `resolve_mut`. Returns `None` (the caller
    /// decides whether to error or return empty) when the default collection
    /// has not been materialized yet — searches against a never-touched
    /// store should return an empty result rather than allocate.
    fn resolve(&self, name: Option<&str>) -> Result<Option<&Collection>, String> {
        let name = name.unwrap_or(DEFAULT_COLLECTION);
        match self.collections.get(name) {
            Some(c) => Ok(Some(c)),
            None if name == DEFAULT_COLLECTION => Ok(None),
            None => Err(format!("collection not found: {name}")),
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

/// Convert an MCP wire-format filter (`HashMap<String, String>`) into the
/// `Vec<(String, String)>` shape consumed by [`BruteForce::search`].
fn filter_to_pairs(filter: Option<HashMap<String, String>>) -> Option<Vec<(String, String)>> {
    filter.map(|m| m.into_iter().collect())
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

/// Render a Metric as the lowercase wire-format token used in tool params
/// (`"euclidean"`, `"cosine"`, `"dot"`). Inverse of `parse_metric`.
fn metric_token(metric: Metric) -> &'static str {
    match metric {
        Metric::Euclidean => "euclidean",
        Metric::Cosine => "cosine",
        Metric::DotProduct => "dot",
    }
}

/// Normalize a raw distance into a "higher = closer" similarity score for
/// downstream UX. Mapping is metric-specific:
///   - Cosine distance ∈ [0, 2] → similarity = 1 - d ∈ [-1, 1]
///   - Euclidean distance ∈ [0, ∞) → similarity = 1 / (1 + d) ∈ (0, 1]
///   - Dot product (stored as `-dot`) → similarity = -d (raw dot product, unbounded)
///
/// Cosine and Euclidean stay well-behaved for ranking. Dot-product similarity
/// is unbounded by design — clients comparing dot scores should rely on
/// ordering, not absolute magnitude.
fn similarity_for(metric: Metric, distance: f32) -> f32 {
    match metric {
        Metric::Cosine => 1.0 - distance,
        Metric::Euclidean => 1.0 / (1.0 + distance),
        Metric::DotProduct => -distance,
    }
}

#[tool_router]
impl NanoVecServer {
    #[tool(
        name = "index_vector",
        description = "Index a document with its embedding vector and optional metadata. Optional `collection` defaults to `default`."
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
        let collection = state.resolve_mut(params.collection.as_deref())?;
        let expected = collection.store.dimension();
        if dim != expected {
            return Err(format!(
                "vector dim mismatch in collection {}: expected {expected}, got {dim}",
                collection.name
            ));
        }

        let offset = collection
            .store
            .insert(&params.vector)
            .map_err(|e| format!("insert error: {e}"))?;

        let metadata = parse_metadata(params.metadata);
        let id = collection.records.insert(params.text, metadata, offset);

        tracing::debug!(id, offset, collection = %collection.name, "indexed vector");
        Ok(serde_json::json!({ "id": id }).to_string())
    }

    #[tool(
        name = "index_document",
        description = "Index a text document — server computes the embedding via the configured model. Optional `collection` defaults to `default`; must have dimension 384."
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

        let collection = state.resolve_mut(params.collection.as_deref())?;
        let expected = collection.store.dimension();
        if vector.len() != expected {
            return Err(format!(
                "dimension mismatch: collection {} is locked at {expected} but the embedder emits {}-dim vectors",
                collection.name,
                vector.len()
            ));
        }

        let offset = collection
            .store
            .insert(&vector)
            .map_err(|e| format!("insert error: {e}"))?;

        let metadata = parse_metadata(params.metadata);
        let id = collection.records.insert(params.text, metadata, offset);

        tracing::debug!(id, offset, collection = %collection.name, "indexed document");
        Ok(serde_json::json!({ "id": id }).to_string())
    }

    #[tool(
        name = "search",
        description = "Search for the k nearest vectors to a query vector. Optional `filter` (object) restricts results to records whose metadata contains ALL of the listed key=value pairs. Optional `collection` defaults to `default`."
    )]
    fn search(&self, Parameters(params): Parameters<SearchParams>) -> Result<String, String> {
        let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        let metric = parse_metric(&params.metric, state.metric)?;
        let filter_pairs = filter_to_pairs(params.filter);

        let collection = state.resolve(params.collection.as_deref())?;
        let Some(collection) = collection else {
            // Default collection never materialized — return empty results
            // rather than an error, matching the Phase 1 zero-store contract.
            return Ok(serde_json::json!([]).to_string());
        };

        let results = BruteForce::search(
            &collection.store,
            &collection.records,
            &params.vector,
            params.k,
            metric,
            filter_pairs.as_deref(),
        );

        let json_results: Vec<serde_json::Value> = results
            .into_iter()
            .map(|r| {
                let similarity = similarity_for(metric, r.score);
                let metadata: serde_json::Map<String, serde_json::Value> = r
                    .metadata
                    .into_iter()
                    .map(|(k, v)| (k, serde_json::Value::String(v)))
                    .collect();
                serde_json::json!({
                    "id": r.id,
                    "text": r.text,
                    "metadata": metadata,
                    "distance": r.score,
                    "similarity": similarity,
                    // Backward-compat alias for `distance` — kept so Phase 1/2
                    // clients reading `score` continue to work unchanged.
                    "score": r.score,
                })
            })
            .collect();

        Ok(serde_json::json!(json_results).to_string())
    }

    #[tool(
        name = "search_document",
        description = "Semantic search by query text — server computes the query embedding. Default metric is cosine (well-suited for the L2-normalized embedded path). Supports `filter` (metadata AND) and `collection` (defaults to `default`)."
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
        let filter_pairs = filter_to_pairs(params.filter);

        let collection = state.resolve(params.collection.as_deref())?;
        let Some(collection) = collection else {
            return Ok(serde_json::json!([]).to_string());
        };

        let results = BruteForce::search(
            &collection.store,
            &collection.records,
            &query_vec,
            params.k,
            metric,
            filter_pairs.as_deref(),
        );

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
                let similarity = similarity_for(metric, r.score);
                serde_json::json!({
                    "id": r.id,
                    "text": r.text,
                    "metadata": metadata,
                    "distance": r.score,
                    "similarity": similarity,
                    // Backward-compat alias for `distance`.
                    "score": r.score,
                })
            })
            .collect();

        Ok(serde_json::json!(json_results).to_string())
    }

    #[tool(
        name = "delete",
        description = "Delete a document by its ID. Optional `collection` defaults to `default`."
    )]
    fn delete(&self, Parameters(params): Parameters<DeleteParams>) -> Result<String, String> {
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        let collection = state.resolve_mut(params.collection.as_deref())?;
        brute::delete(&mut collection.store, &mut collection.records, params.id)
            .map_err(|e| format!("{e}"))?;

        tracing::debug!(id = params.id, collection = %collection.name, "deleted vector");
        Ok(serde_json::json!({ "success": true }).to_string())
    }

    #[tool(
        name = "clear",
        description = "Remove all indexed documents from one collection (defaults to `default`). Resets the ID counter; the collection itself is preserved (use `drop_collection` to remove it entirely)."
    )]
    fn clear(&self, Parameters(params): Parameters<ClearParams>) -> Result<String, String> {
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
        let collection = state.resolve_mut(params.collection.as_deref())?;
        let deleted = collection.records.count();
        collection.store.clear();
        collection.records.clear();
        tracing::info!(count = deleted, collection = %collection.name, "cleared collection");
        Ok(serde_json::json!({ "deleted": deleted }).to_string())
    }

    #[tool(
        name = "create_collection",
        description = "Create a new named collection with the given vector dimension."
    )]
    fn create_collection(
        &self,
        Parameters(params): Parameters<CreateCollectionParams>,
    ) -> Result<String, String> {
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
        if params.dimension == 0 {
            return Err("dimension must be > 0".to_string());
        }
        state
            .collections
            .create(params.name.clone(), params.dimension)
            .map_err(|e| format!("{e}"))?;
        tracing::info!(name = %params.name, dimension = params.dimension, "created collection");
        Ok(serde_json::json!({
            "created": params.name,
            "dimension": params.dimension,
        })
        .to_string())
    }

    #[tool(
        name = "list_collections",
        description = "List all collections with their per-collection record counts and dimensions."
    )]
    fn list_collections(&self) -> Result<String, String> {
        let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
        let collections: Vec<serde_json::Value> = state
            .collections
            .iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "count": c.count(),
                    "dimension": c.dimension(),
                })
            })
            .collect();
        Ok(serde_json::json!({ "collections": collections }).to_string())
    }

    #[tool(
        name = "drop_collection",
        description = "Remove a collection (and all of its data) from the server. Dropping `default` is allowed — the next legacy call will re-create it."
    )]
    fn drop_collection(
        &self,
        Parameters(params): Parameters<DropCollectionParams>,
    ) -> Result<String, String> {
        let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
        state
            .collections
            .drop(&params.name)
            .map_err(|e| format!("{e}"))?;
        tracing::info!(name = %params.name, "dropped collection");
        Ok(serde_json::json!({ "dropped": params.name }).to_string())
    }

    #[tool(
        name = "stats",
        description = "Get database statistics: aggregate record count, per-collection breakdown, default metrics, and embedder identity."
    )]
    fn stats(&self) -> Result<String, String> {
        let state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;

        // Per-tool defaults: raw-vector path uses the server-wide metric
        // (Phase 1 contract), document path is hardcoded to cosine because
        // the embedder emits L2-normalized vectors and cosine reads cleanest
        // to API consumers.
        let raw_vector_default = metric_token(state.metric);
        let document_default = metric_token(Metric::Cosine);

        // Embedder identity exposed so clients can confirm which model is
        // loaded without reading the binary's logs.
        let embedder_model = state.embedder.model_name();
        let embedder_dim = state.embedder.dimension();
        let metric_label = format!("{:?}", state.metric);

        // Per-collection breakdown (sorted by name via CollectionMap::iter).
        let collections_json: Vec<serde_json::Value> = state
            .collections
            .iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.name,
                    "count": c.count(),
                    "dimension": c.dimension(),
                })
            })
            .collect();

        // Aggregate record count across every collection.
        let total_count: usize = state.collections.iter().map(|c| c.count()).sum();

        // Backward-compat top-level fields:
        //   - `count` mirrors the default collection's record count (or 0 if
        //     the default has not been materialized) so pre-Phase-4 clients
        //     keep seeing the same number.
        //   - `dimension` mirrors the default collection's dimension (or 0).
        let default_collection = state.collections.get(DEFAULT_COLLECTION);
        let legacy_count = default_collection.map(|c| c.count()).unwrap_or(0);
        let legacy_dim = default_collection.map(|c| c.dimension()).unwrap_or(0);

        Ok(serde_json::json!({
            "count": legacy_count,
            "total_count": total_count,
            "dimension": legacy_dim,
            "default_metric": {
                "raw_vector": raw_vector_default,
                "document": document_default,
            },
            "embedder": {
                "model": embedder_model,
                "dim": embedder_dim,
            },
            // Phase 4 expansion: per-collection breakdown.
            "collections": collections_json,
            // Backward-compat alias — kept so Phase 1/2 clients reading
            // `metric` continue to work. New clients should prefer
            // `default_metric.raw_vector`.
            "metric": metric_label,
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
