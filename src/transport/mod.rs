//! Phase 6 T3 — transport selection.
//!
//! NanoVec supports two transports:
//! - `stdio` (default) — for local Claude Code, MCP CLI tools, child-process
//!   integrations. Works as a single bidirectional pipe.
//! - `streamable HTTP` (opt-in via `NANOVEC_SSE_ADDR`) — for network-attached
//!   agents, multi-client scenarios. Built on top of `axum` and the rmcp
//!   `StreamableHttpService`. Endpoint path is `/mcp`.
//!
//! Selection:
//! - `NANOVEC_SSE_ADDR` unset       -> stdio only (Phase 1–5 behavior).
//! - `NANOVEC_SSE_ADDR=host:port`   -> HTTP only on `host:port`.
//!
//! Running both transports simultaneously is not currently supported. The
//! HTTP server is local-only by default: NanoVec has no authentication, so
//! exposing it on `0.0.0.0` is the operator's responsibility.

use std::net::SocketAddr;

pub mod http;
pub mod stdio;

/// Where the server should listen.
#[derive(Debug, Clone)]
pub enum TransportMode {
    Stdio,
    Http(SocketAddr),
}

/// Parse `NANOVEC_SSE_ADDR` from the environment. Returns `Stdio` when unset
/// or invalid (with a warning log).
pub fn select_from_env() -> TransportMode {
    match std::env::var("NANOVEC_SSE_ADDR").ok() {
        None => TransportMode::Stdio,
        Some(raw) => match raw.parse::<SocketAddr>() {
            Ok(addr) => TransportMode::Http(addr),
            Err(e) => {
                tracing::warn!(
                    addr = %raw,
                    error = %e,
                    "NANOVEC_SSE_ADDR is not a valid socket address — falling back to stdio"
                );
                TransportMode::Stdio
            }
        },
    }
}
