//! Server entry point — selects the transport based on `NANOVEC_SSE_ADDR`
//! and hands off to either [`crate::transport::stdio`] or
//! [`crate::transport::http`].

use std::sync::Arc;

use crate::distance::Metric;
use crate::embed::Embedder;
use crate::store::database::NanoVecDatabase;
use crate::transport::{self, TransportMode};

/// Start the NanoVec MCP server. The transport is decided by
/// `NANOVEC_SSE_ADDR`: unset → stdio, `host:port` → streamable HTTP.
///
/// The embedder is loaded synchronously off the async reactor before opening
/// any transport — on a cold cache the ~90 MB model download takes 10–30 s,
/// and we want to fail-fast (and log) rather than handshake first and stall
/// the first tool call.
///
/// Blocks until the transport terminates.
pub async fn run() -> anyhow::Result<()> {
    tracing::info!("loading embedder (sentence-transformers/all-MiniLM-L6-v2, ~90MB on first run)");
    let embedder = tokio::task::spawn_blocking(Embedder::load)
        .await
        .map_err(|e| anyhow::anyhow!("embedder load task panicked: {e}"))?
        .map_err(|e| anyhow::anyhow!("embedder load failed: {e}"))?;
    let dim = embedder.dimension();
    tracing::info!(dimension = dim, "embedder loaded");

    let db = Arc::new(NanoVecDatabase::new(Metric::Euclidean, Arc::new(embedder)));

    match transport::select_from_env() {
        TransportMode::Stdio => transport::stdio::run(db).await,
        TransportMode::Http(addr) => transport::http::run(db, addr).await,
    }
}
