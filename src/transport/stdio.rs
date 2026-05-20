//! stdio transport — the Phase 1 path, factored out of `src/server/mod.rs`.

use std::sync::Arc;

use rmcp::transport::io::stdio;
use rmcp::ServiceExt;

use crate::mcp::NanoVecServer;
use crate::store::database::NanoVecDatabase;

/// Run the MCP server on stdio. Blocks until the client closes the pipe.
pub async fn run(db: Arc<NanoVecDatabase>) -> anyhow::Result<()> {
    let server = NanoVecServer::from_database(db);
    let router = server.into_router();

    tracing::info!("nanovec MCP server starting on stdio");
    let service = router.serve(stdio()).await?;
    service.waiting().await?;
    tracing::info!("nanovec MCP server shut down");
    Ok(())
}
