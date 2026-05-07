use std::sync::Arc;

use rmcp::transport::io::stdio;
use rmcp::ServiceExt;

use crate::distance::Metric;
use crate::embed::Embedder;
use crate::mcp::NanoVecServer;

/// Start the NanoVec MCP server on stdio transport.
///
/// Loads the embedder synchronously off the async reactor (via
/// `tokio::task::spawn_blocking`) before opening the stdio transport. On a
/// cold cache the model download is ~90MB and may take 10-30 s, so we log the
/// loading state to stderr to avoid the appearance of a hung process.
///
/// Blocks until the client disconnects.
pub async fn run() -> anyhow::Result<()> {
    tracing::info!("loading embedder (sentence-transformers/all-MiniLM-L6-v2, ~90MB on first run)");
    let embedder = tokio::task::spawn_blocking(Embedder::load)
        .await
        .map_err(|e| anyhow::anyhow!("embedder load task panicked: {e}"))?
        .map_err(|e| anyhow::anyhow!("embedder load failed: {e}"))?;
    let dim = embedder.dimension();
    tracing::info!(dimension = dim, "embedder loaded");

    let server = NanoVecServer::new(Metric::Euclidean, Arc::new(embedder));
    let router = server.into_router();

    tracing::info!("nanovec MCP server starting on stdio");

    let service = router.serve(stdio()).await?;
    service.waiting().await?;

    tracing::info!("nanovec MCP server shut down");
    Ok(())
}
