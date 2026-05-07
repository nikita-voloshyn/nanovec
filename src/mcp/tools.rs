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
}

/// Parameters for the `index_document` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct IndexDocumentParams {
    /// The text to index. Server computes the embedding.
    pub text: String,
    /// Optional key-value metadata pairs.
    pub metadata: Option<serde_json::Value>,
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
}

/// Parameters for the `delete` tool.
#[derive(Deserialize, schemars::JsonSchema)]
pub struct DeleteParams {
    /// The ID of the document to delete.
    pub id: u64,
}
