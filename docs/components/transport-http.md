# Streamable HTTP Transport (Phase 6)

## Purpose

`src/transport/http.rs` exposes NanoVec over HTTP as an alternative to the
default stdio transport. Built on `rmcp::StreamableHttpService` (the
official MCP streamable-HTTP spec) wrapped in an `axum::Router` mounted at
`/mcp`. Used when:

- Multiple MCP clients need to share one running server (per-connection
  isolation via `connection_id`).
- The agent runs in a different process/container than NanoVec.
- The client cannot drive a stdio process directly.

The HTTP transport is **opt-in** — the binary keeps stdio as default to
match the Phase 1–5 contract. NanoVec has no authentication; deployment to
untrusted networks requires an external reverse proxy.

## Public API

```rust
// src/transport/mod.rs

pub enum TransportMode {
    Stdio,
    Http(SocketAddr),
}

pub fn select_from_env() -> TransportMode;

// src/transport/http.rs

pub async fn run(db: Arc<NanoVecDatabase>, addr: SocketAddr) -> anyhow::Result<()>;

pub async fn run_with_cancel(
    db: Arc<NanoVecDatabase>,
    addr: SocketAddr,
    ct: CancellationToken,
) -> anyhow::Result<()>;

// Test helper — binds on 127.0.0.1:0, returns resolved address.
pub async fn spawn_for_tests(
    db: Arc<NanoVecDatabase>,
) -> anyhow::Result<(SocketAddr, CancellationToken, tokio::task::JoinHandle<()>)>;
```

## Environment variables

| Var | Effect |
|-----|--------|
| `NANOVEC_SSE_ADDR=host:port` | Run streamable HTTP on `host:port` (skip stdio). |
| `NANOVEC_SSE_ADDR` unset | Stdio transport (default). |
| Invalid address | Logs a warning, falls back to stdio. |
| `NANOVEC_MEMORY_LIMIT_MB` | Inherited from the database layer; works for both transports. |

## Internal Design

### Service factory

```rust
StreamableHttpService::new(
    move || Ok(NanoVecServer::from_database(Arc::clone(&db)).into_router()),
    Arc::new(LocalSessionManager::default()),
    config,
)
```

The factory is invoked **per session** and produces a fresh
`Router<NanoVecServer>`. All sessions share the same
`Arc<NanoVecDatabase>` — so writes from one client are immediately visible
to readers in another. The `Router` is rmcp's tool-dispatch container, not
axum's.

### Stateful vs stateless

`run` / `run_with_cancel` use the default `StreamableHttpServerConfig`,
which is **stateful** (SSE-style with `mcp-session-id` headers). The test
helper `spawn_for_tests` overrides this to **stateless + JSON response**:

```rust
StreamableHttpServerConfig::default()
    .with_cancellation_token(ct.clone())
    .with_stateful_mode(false)
    .with_json_response(true);
```

This avoids the SSE-stream timing complexity in tests; every POST returns
a single `application/json` body. Session affinity isn't needed because
all state lives in the shared `Arc<NanoVecDatabase>`.

### Endpoint shape

```
POST /mcp
Content-Type: application/json
Accept: application/json, text/event-stream

{"jsonrpc": "2.0", "id": 1, "method": "initialize", ...}
```

Subsequent tool calls follow standard MCP JSON-RPC `tools/call` shape.

### Graceful shutdown

`run_with_cancel` accepts a `CancellationToken`; axum's
`with_graceful_shutdown` waits for it to fire. The test helper returns
the token so callers can cancel cleanly without sending SIGTERM.

## Connection isolation (Phase 6 T4)

Every tool param struct gained a `connection_id: Option<String>` field. The
MCP handler resolves the target collection as follows:

```
resolve_collection_name(collection, connection_id):
    if collection is Some -> use it
    else if connection_id is Some("alice") -> "_conn_alice"
    else -> None (falls back to "default")
```

`_conn_*` collections are pinned at the database layer (never evicted by
LRU) and auto-created on first reference. This gives multi-tenant
isolation without requiring the client to manage collection lifecycles.

**Trust model:** `connection_id` is client-supplied and unauthenticated. A
malicious HTTP client can spoof another's id and read its data. This is
intentional for the local-only deployment model; production deployment
needs an external auth layer.

## Usage Example

```bash
# Server side.
NANOVEC_SSE_ADDR=127.0.0.1:8421 cargo run --release

# Client side (curl).
curl -X POST http://127.0.0.1:8421/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-11-25",
      "capabilities": {},
      "clientInfo": {"name": "curl-test", "version": "1.0"}
    }
  }'

# Tool call with connection_id (multi-tenant).
curl -X POST http://127.0.0.1:8421/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "index_document",
      "arguments": {
        "text": "hello world",
        "connection_id": "alice"
      }
    }
  }'
```

## Performance Notes

- Stateless + JSON mode: one HTTP round-trip per request, no streaming
  overhead.
- Stateful mode: each request goes over an SSE-framed response stream with
  priming events; clients must read the full body to get the JSON-RPC
  result.
- Dependencies: axum + tower + hyper add ~150 transitive crates to the
  binary. The stdio-only build path is unaffected (those crates aren't
  pulled in when the `transport-streamable-http-server` rmcp feature is
  disabled — but in NanoVec it's always on, so the binary always ships
  with HTTP capability).

## Test Coverage

| Test | Verifies |
|------|----------|
| `two_http_clients_isolated_by_connection_id` (`tests/integration/http_e2e.rs`) | Real reqwest clients with different `connection_id`s see isolated `_conn_*` collections. Alice's search returns only `alice-*`, Bob's only `bob-*`. |
| `http_stats_tool_returns_memory_breakdown` (`tests/integration/http_e2e.rs`) | `stats` and `memory` tools work end-to-end over HTTP and return the Phase 6 budget fields. |
| `rebuild_index_builds_hnsw_and_search_uses_it` (`tests/integration/rebuild_index.rs`) | Exercises the `rebuild_index` tool over HTTP — builds HNSW, verifies post-build search matches brute-force baseline. |
| `rebuild_index_uses_hnsw_metric_only_for_cosine` (`tests/integration/rebuild_index.rs`) | Metric dispatch: HNSW under cosine, fallback to brute for euclidean. |
