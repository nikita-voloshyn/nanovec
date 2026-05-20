use std::collections::HashMap;

use rmcp::schemars;
use serde::Deserialize;

/// Parameters for the `index_vector` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct IndexVectorParams {
    /// The text content to associate with this vector.
    pub text: String,
    /// The embedding vector as a list of floats.
    pub vector: Vec<f32>,
    /// Optional key-value metadata pairs.
    pub metadata: Option<serde_json::Value>,
    /// Optional collection name. Defaults to the connection-scoped collection
    /// (`_conn_<connection_id>`) if `connection_id` is provided, otherwise
    /// `"default"`.
    pub collection: Option<String>,
    /// Optional connection identifier. When set and `collection` is absent,
    /// the call operates on `_conn_<connection_id>` — an auto-created pinned
    /// collection isolated per logical client.
    pub connection_id: Option<String>,
}

/// Parameters for the `index_document` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct IndexDocumentParams {
    /// The text to index. Server computes the embedding.
    pub text: String,
    /// Optional key-value metadata pairs.
    pub metadata: Option<serde_json::Value>,
    /// Optional collection name. Defaults to the connection-scoped collection
    /// when `connection_id` is set, otherwise `"default"`.
    pub collection: Option<String>,
    /// Optional connection identifier (see [`IndexVectorParams`]).
    pub connection_id: Option<String>,
}

/// Parameters for the `search` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct SearchParams {
    /// The query embedding vector.
    pub vector: Vec<f32>,
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Distance metric: "euclidean", "cosine", or "dot". Defaults to the server default.
    pub metric: Option<String>,
    /// Optional metadata filter — only records whose metadata contains
    /// **all** of these `(key, value)` pairs are considered.
    pub filter: Option<HashMap<String, String>>,
    /// Optional collection name. Falls back to the connection-scoped
    /// collection (`_conn_<connection_id>`) or `"default"`.
    pub collection: Option<String>,
    /// Optional connection identifier (see [`IndexVectorParams`]).
    pub connection_id: Option<String>,
}

/// Parameters for the `search_document` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct SearchDocumentParams {
    /// The query text. Server computes the embedding and runs KNN.
    pub query: String,
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Distance metric: "euclidean", "cosine", or "dot". Defaults to "cosine"
    /// for the embedded path (model output is L2-normalized so cosine is
    /// equivalent to dot, but cosine reads more clearly to the API consumer).
    pub metric: Option<String>,
    /// Optional metadata filter — only records whose metadata contains
    /// **all** of these `(key, value)` pairs are considered.
    pub filter: Option<HashMap<String, String>>,
    /// Optional collection name. Falls back to the connection-scoped
    /// collection (`_conn_<connection_id>`) or `"default"`.
    pub collection: Option<String>,
    /// Optional connection identifier (see [`IndexVectorParams`]).
    pub connection_id: Option<String>,
}

/// Parameters for the `delete` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct DeleteParams {
    /// The ID of the document to delete.
    pub id: u64,
    /// Optional collection name. Falls back to the connection-scoped
    /// collection or `"default"`.
    pub collection: Option<String>,
    /// Optional connection identifier (see [`IndexVectorParams`]).
    pub connection_id: Option<String>,
}

/// Parameters for the `clear` tool.
#[derive(Deserialize, schemars::JsonSchema, Default)]
pub struct ClearParams {
    /// Optional collection name. Falls back to the connection-scoped
    /// collection or `"default"`. Clears one collection only.
    pub collection: Option<String>,
    /// Optional connection identifier (see [`IndexVectorParams`]).
    pub connection_id: Option<String>,
}

/// Parameters for the `create_collection` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct CreateCollectionParams {
    /// The name of the collection to create.
    pub name: String,
    /// The vector dimension this collection is locked to.
    pub dimension: usize,
}

/// Parameters for the `drop_collection` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct DropCollectionParams {
    /// The name of the collection to drop.
    pub name: String,
}
