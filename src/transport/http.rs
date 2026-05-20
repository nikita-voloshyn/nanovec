//! Streamable HTTP transport (Phase 6 T3).
//!
//! Wraps an `rmcp::StreamableHttpService` in an `axum::Router` mounted at
//! `/mcp`. The `service_factory` is invoked per session and produces a
//! fresh `Router<NanoVecServer>` — but all sessions share the same
//! `Arc<NanoVecDatabase>`, so writes from one session are immediately
//! visible to readers in another.
//!
//! NanoVec has no authentication: the HTTP server is intended for local
//! development and trusted networks. Production deployment requires an
//! external reverse proxy with auth.

use std::net::SocketAddr;
use std::sync::Arc;

use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio_util::sync::CancellationToken;

use crate::mcp::NanoVecServer;
use crate::store::database::NanoVecDatabase;

/// Run the MCP server over streamable HTTP on `addr`. Listens until the
/// returned cancellation token is cancelled (or the OS sends SIGTERM, which
/// terminates the whole process and trips `tokio::signal::ctrl_c`).
pub async fn run(db: Arc<NanoVecDatabase>, addr: SocketAddr) -> anyhow::Result<()> {
    let ct = CancellationToken::new();
    run_with_cancel(db, addr, ct).await
}

/// Same as [`run`] but lets callers (e.g. integration tests) supply a
/// pre-built cancellation token so they can shut the server down cleanly.
pub async fn run_with_cancel(
    db: Arc<NanoVecDatabase>,
    addr: SocketAddr,
    ct: CancellationToken,
) -> anyhow::Result<()> {
    let config = StreamableHttpServerConfig::default().with_cancellation_token(ct.clone());

    let db_for_factory = Arc::clone(&db);
    let service: StreamableHttpService<_, LocalSessionManager> = StreamableHttpService::new(
        move || {
            let server = NanoVecServer::from_database(Arc::clone(&db_for_factory));
            Ok(server.into_router())
        },
        Arc::new(LocalSessionManager::default()),
        config,
    );

    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let actual = listener.local_addr()?;
    tracing::info!(addr = %actual, "nanovec MCP server starting on streamable HTTP at /mcp");

    let shutdown = ct.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.cancelled_owned().await })
        .await?;

    tracing::info!("nanovec MCP HTTP server shut down");
    Ok(())
}

/// Convenience for tests: bind on port 0 and return the resolved address +
/// a join handle. The caller cancels via the returned `CancellationToken`.
///
/// Stateless + JSON response mode — every POST returns a single
/// `application/json` body with the JSON-RPC reply. Avoids SSE-stream
/// timing in tests; session affinity is not needed because all state lives
/// in the shared `Arc<NanoVecDatabase>`.
pub async fn spawn_for_tests(
    db: Arc<NanoVecDatabase>,
) -> anyhow::Result<(SocketAddr, CancellationToken, tokio::task::JoinHandle<()>)> {
    let ct = CancellationToken::new();
    let config = StreamableHttpServerConfig::default()
        .with_cancellation_token(ct.clone())
        .with_stateful_mode(false)
        .with_json_response(true);

    let db_for_factory = Arc::clone(&db);
    let service: StreamableHttpService<_, LocalSessionManager> = StreamableHttpService::new(
        move || {
            let server = NanoVecServer::from_database(Arc::clone(&db_for_factory));
            Ok(server.into_router())
        },
        Arc::new(LocalSessionManager::default()),
        config,
    );

    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;

    let shutdown = ct.clone();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move { shutdown.cancelled_owned().await })
            .await;
    });

    Ok((addr, ct, handle))
}
