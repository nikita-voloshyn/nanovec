pub mod tools;

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::handler::server::router::Router;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{tool, tool_router, ServerHandler};

use crate::distance::Metric;
use crate::embed::Embedder;
use crate::index::brute::{self, BruteForce};
use crate::store::database::NanoVecDatabase;

use self::tools::{
    ClearParams, CreateCollectionParams, DeleteParams, DropCollectionParams, IndexDocumentParams,
    IndexVectorParams, SearchDocumentParams, SearchParams,
};

/// The MCP server handler. Phase 6: holds an `Arc<NanoVecDatabase>` with
/// two-level RwLocks — outer (collection map) and inner (per-collection).
/// The previous global `Mutex<NanoVecState>` is gone; readers no longer
/// block each other.
pub struct NanoVecServer {
    db: Arc<NanoVecDatabase>,
}

impl NanoVecServer {
    pub fn new(metric: Metric, embedder: Arc<Embedder>) -> Self {
        Self {
            db: Arc::new(NanoVecDatabase::new(metric, embedder)),
        }
    }

    /// Constructor with an explicit, already-built database. Useful for tests
    /// that want to pre-populate collections or set a custom memory limit.
    pub fn from_database(db: Arc<NanoVecDatabase>) -> Self {
        Self { db }
    }

    /// Expose the database handle. Used by SSE transport (Task 3) and by
    /// integration tests that drive the database directly.
    pub fn database(&self) -> Arc<NanoVecDatabase> {
        Arc::clone(&self.db)
    }

    /// Build the Router that wires up all tool routes and the server handler.
    pub fn into_router(self) -> Router<Self> {
        let tool_routes = Self::tool_router();
        Router::new(self).with_tools(tool_routes)
    }
}

/// Resolve `(collection, connection_id)` to the target collection name.
///
/// - explicit `collection` wins, regardless of `connection_id`.
/// - else if `connection_id` is `Some("alice")` → `"_conn_alice"`
///   (auto-created and pinned per Phase 6 T4).
/// - else `None` → caller falls back to `"default"` via `db.resolve(None)`.
fn resolve_collection_name(
    collection: Option<String>,
    connection_id: Option<String>,
) -> Option<String> {
    if let Some(c) = collection {
        return Some(c);
    }
    connection_id.map(|id| format!("_conn_{id}"))
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
/// downstream UX.
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
        if params.vector.is_empty() {
            return Err("vector must not be empty".to_string());
        }

        let target = resolve_collection_name(params.collection, params.connection_id);
        let guard = self
            .db
            .resolve(target.as_deref())
            .map_err(|e| e.to_string())?;

        let mut coll = guard.inner.write();
        let dim = params.vector.len();
        let expected = coll.store.dimension();
        if dim != expected {
            return Err(format!(
                "vector dim mismatch in collection {}: expected {expected}, got {dim}",
                coll.name
            ));
        }

        let offset = coll
            .store
            .insert(&params.vector)
            .map_err(|e| format!("insert error: {e}"))?;

        let metadata = parse_metadata(params.metadata);
        let id = coll.records.insert(params.text, metadata, offset);

        tracing::debug!(id, offset, collection = %coll.name, "indexed vector");
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

        // The embedder is held in an Arc on the database and is read-only.
        // No locking needed to call embed() — multiple concurrent index_document
        // calls run BERT forward in parallel.
        let embedder = self
            .db
            .embedder
            .as_ref()
            .ok_or_else(|| "embedder not loaded (raw-vector mode)".to_string())?;
        let vector = embedder
            .embed(&params.text)
            .map_err(|e| format!("embed error: {e}"))?;

        // Resolve target collection, then take its inner write-lock for the
        // store/records mutation.
        let target = resolve_collection_name(params.collection, params.connection_id);
        let guard = self
            .db
            .resolve(target.as_deref())
            .map_err(|e| e.to_string())?;
        let mut coll = guard.inner.write();

        let expected = coll.store.dimension();
        if vector.len() != expected {
            return Err(format!(
                "dimension mismatch: collection {} is locked at {expected} but the embedder emits {}-dim vectors",
                coll.name,
                vector.len()
            ));
        }

        let offset = coll
            .store
            .insert(&vector)
            .map_err(|e| format!("insert error: {e}"))?;

        let metadata = parse_metadata(params.metadata);
        let id = coll.records.insert(params.text, metadata, offset);

        tracing::debug!(id, offset, collection = %coll.name, "indexed document");
        Ok(serde_json::json!({ "id": id }).to_string())
    }

    #[tool(
        name = "search",
        description = "Search for the k nearest vectors to a query vector. Optional `filter` (object) restricts results to records whose metadata contains ALL of the listed key=value pairs. Optional `collection` defaults to `default`."
    )]
    fn search(&self, Parameters(params): Parameters<SearchParams>) -> Result<String, String> {
        let metric = parse_metric(&params.metric, self.db.metric)?;
        let filter_pairs = filter_to_pairs(params.filter);

        let target = resolve_collection_name(params.collection, params.connection_id);
        let guard_opt = self
            .db
            .resolve_readonly(target.as_deref())
            .map_err(|e| e.to_string())?;
        let Some(guard) = guard_opt else {
            return Ok(serde_json::json!([]).to_string());
        };

        // Inner read-lock: many concurrent searches on the same collection
        // never block each other.
        let coll = guard.inner.read();
        let results = BruteForce::search(
            &coll.store,
            &coll.records,
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

        let embedder = self
            .db
            .embedder
            .as_ref()
            .ok_or_else(|| "embedder not loaded (raw-vector mode)".to_string())?;
        let query_vec = embedder
            .embed(&params.query)
            .map_err(|e| format!("embed error: {e}"))?;

        let metric = parse_metric(&params.metric, Metric::Cosine)?;
        let filter_pairs = filter_to_pairs(params.filter);

        let target = resolve_collection_name(params.collection, params.connection_id);
        let guard_opt = self
            .db
            .resolve_readonly(target.as_deref())
            .map_err(|e| e.to_string())?;
        let Some(guard) = guard_opt else {
            return Ok(serde_json::json!([]).to_string());
        };

        let coll = guard.inner.read();
        let results = BruteForce::search(
            &coll.store,
            &coll.records,
            &query_vec,
            params.k,
            metric,
            filter_pairs.as_deref(),
        );

        let json_results: Vec<serde_json::Value> = results
            .into_iter()
            .map(|r| {
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
        let target = resolve_collection_name(params.collection, params.connection_id);
        let guard = self
            .db
            .resolve(target.as_deref())
            .map_err(|e| e.to_string())?;
        let mut coll = guard.inner.write();
        let coll_name = coll.name.clone();
        let coll_ref = &mut *coll;
        brute::delete(&mut coll_ref.store, &mut coll_ref.records, params.id)
            .map_err(|e| format!("{e}"))?;
        tracing::debug!(id = params.id, collection = %coll_name, "deleted vector");
        Ok(serde_json::json!({ "success": true }).to_string())
    }

    #[tool(
        name = "clear",
        description = "Remove all indexed documents from one collection (defaults to `default`). Resets the ID counter; the collection itself is preserved (use `drop_collection` to remove it entirely)."
    )]
    fn clear(&self, Parameters(params): Parameters<ClearParams>) -> Result<String, String> {
        let target = resolve_collection_name(params.collection, params.connection_id);
        let guard = self
            .db
            .resolve(target.as_deref())
            .map_err(|e| e.to_string())?;
        let mut coll = guard.inner.write();
        let deleted = coll.records.count();
        coll.store.clear();
        coll.records.clear();
        tracing::info!(count = deleted, collection = %coll.name, "cleared collection");
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
        if params.dimension == 0 {
            return Err("dimension must be > 0".to_string());
        }
        self.db
            .create_collection(params.name.clone(), params.dimension)
            .map_err(|e| e.to_string())?;
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
        let snapshots = self.db.snapshot();
        let collections: Vec<serde_json::Value> = snapshots
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "count": s.count,
                    "dimension": s.dimension,
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
        self.db
            .drop_collection(&params.name)
            .map_err(|e| e.to_string())?;
        tracing::info!(name = %params.name, "dropped collection");
        Ok(serde_json::json!({ "dropped": params.name }).to_string())
    }

    #[tool(
        name = "stats",
        description = "Get database statistics: aggregate record count, per-collection breakdown, default metrics, memory budget, and embedder identity."
    )]
    fn stats(&self) -> Result<String, String> {
        let raw_vector_default = metric_token(self.db.metric);
        let document_default = metric_token(Metric::Cosine);

        let (embedder_model, embedder_dim) = match self.db.embedder.as_ref() {
            Some(e) => (e.model_name(), e.dimension()),
            None => ("(none)", self.db.auto_dim()),
        };
        let metric_label = format!("{:?}", self.db.metric);

        let snapshots = self.db.snapshot();
        let used_bytes = self.db.recompute_used_bytes();
        let limit_bytes = self.db.limit_bytes();
        let memory_pct = if limit_bytes > 0 {
            (used_bytes as f64 / limit_bytes as f64) * 100.0
        } else {
            0.0
        };

        let collections_json: Vec<serde_json::Value> = snapshots
            .iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "count": s.count,
                    "dimension": s.dimension,
                    "approx_bytes": s.approx_bytes,
                    "pinned": s.pinned,
                })
            })
            .collect();

        let total_count: usize = snapshots.iter().map(|s| s.count).sum();

        // Backward-compat: top-level `count` / `dimension` mirror the default
        // collection (or 0 if not materialized).
        let default_snap = snapshots.iter().find(|s| s.name == "default");
        let legacy_count = default_snap.map(|s| s.count).unwrap_or(0);
        let legacy_dim = default_snap.map(|s| s.dimension).unwrap_or(0);

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
            "collections": collections_json,
            "memory": {
                "used_bytes": used_bytes,
                "limit_bytes": limit_bytes,
                "used_pct": memory_pct,
                "collections_count": snapshots.len(),
            },
            "metric": metric_label,
        })
        .to_string())
    }

    #[tool(
        name = "memory",
        description = "Memory breakdown: per-collection approx-bytes + LRU tick, sorted hot-first. Useful for diagnosing eviction behavior."
    )]
    fn memory(&self) -> Result<String, String> {
        let mut snapshots = self.db.snapshot();
        // Sort: hot first (highest tick), pinned interleaved by tick.
        snapshots.sort_by(|a, b| b.last_accessed_tick.cmp(&a.last_accessed_tick));

        let used_bytes = self.db.recompute_used_bytes();
        let limit_bytes = self.db.limit_bytes();

        let json: Vec<serde_json::Value> = snapshots
            .into_iter()
            .map(|s| {
                serde_json::json!({
                    "name": s.name,
                    "count": s.count,
                    "approx_bytes": s.approx_bytes,
                    "last_accessed_tick": s.last_accessed_tick,
                    "pinned": s.pinned,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "used_bytes": used_bytes,
            "limit_bytes": limit_bytes,
            "collections": json,
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
