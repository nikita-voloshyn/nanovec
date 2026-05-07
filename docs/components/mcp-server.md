# MCP Server

## Purpose

The MCP server exposes NanoVec's vector database operations to AI agent clients over the Model Context Protocol. It is the only external interface: no REST API, no gRPC, no custom TCP protocol. All client interaction goes through stdio using newline-delimited JSON-RPC 2.0 framing provided by the `rmcp` crate (version 1.3.0).

## Architecture Overview

```
main.rs
  └── server::run()                   -- async entry point
        └── NanoVecServer::new()      -- construct server with default metric
              └── NanoVecServer::into_router()  -- wire tool routes
                    └── Router::serve(stdio())  -- start stdio transport
```

Shared mutable state is held in `Arc<Mutex<NanoVecState>>` and accessed from each tool handler.

## State Management

```rust
// src/mcp/mod.rs

pub struct NanoVecState {
    store: VectorStore,          // initialized at startup with dim=384
    records: RecordStore,
    metric: Metric,              // server-wide default metric (Euclidean)
    embedder: Arc<Embedder>,     // read-only after load; shared via Arc
}
```

`store` is initialized at server startup (not lazily) with `VectorStore::new(384)`.
The dimension is locked to 384 by the embedder for the lifetime of the process.
`index_vector` calls with any other dimension are rejected immediately with an
error message that makes the lock explicit: `"vector dim mismatch: expected 384 (locked by embedder), got N"`.

`NanoVecState` is wrapped in `Arc<Mutex<NanoVecState>>` inside `NanoVecServer`, making it safe to share across the async runtime.

## Default Metric Divergence

The server has two different default metrics depending on the tool:

| Tool | Metric when `metric` is omitted | Reason |
|------|---------------------------------|--------|
| `index_vector`, `search` | `Euclidean` | Phase 1 raw-vector contract; preserved for backward compatibility |
| `index_document`, `search_document` | `Cosine` | `all-MiniLM-L6-v2` emits L2-normalized vectors; cosine distance on the unit sphere is the canonical choice and produces scores in `[0, 2]` |

Both tools still accept an explicit `metric` override (`"euclidean"`, `"cosine"`, or
`"dot"`) to select a non-default metric.

## Server Startup

`server::run()` loads the embedder **before** opening the stdio transport:

```rust
tracing::info!("loading embedder (sentence-transformers/all-MiniLM-L6-v2, ~90MB on first run)");
let embedder = tokio::task::spawn_blocking(Embedder::load).await??;
tracing::info!(dimension = dim, "embedder loaded");
```

On a cold cache (first run), this step downloads ~90 MB from HuggingFace Hub and may
take 10-30 seconds. The binary logs its progress to stderr; if the process appears hung
during this period, it is performing the download. On subsequent runs the model is
mmap'ed from `~/.cache/huggingface/hub/` and loads in ~30 ms.

If the embedder fails to load (network error, corrupt cache), the server exits with a
descriptive error before opening stdio — a client will never connect to a server
without a working embedder.

## Transport

The server uses `rmcp`'s stdio transport:

- **stdin**: incoming JSON-RPC 2.0 requests, newline-delimited.
- **stdout**: outgoing JSON-RPC 2.0 responses, newline-delimited.
- **stderr**: all log output (tracing subscriber writes exclusively to stderr).

The stdout/stderr separation is enforced in `main.rs`:

```rust
tracing_subscriber::fmt()
    .with_writer(std::io::stderr)
    .init();
```

This ensures that tracing output never contaminates the JSON-RPC stream.

## Tool Definitions

The server registers six tools using `rmcp`'s `#[tool_router]` and `#[tool]` macros.

### `index_vector`

Index a document with its embedding vector and optional metadata.

**Input schema:**

```json
{
  "text":     "<string>  -- text content to associate with this vector",
  "vector":   "[f32]     -- embedding vector (all floats)",
  "metadata": "{ ... }   -- optional JSON object; values coerced to strings"
}
```

**Behavior:**

1. Validates `vector` is non-empty.
2. Validates `vector.len() == 384` (dim is locked at startup by the embedder).
3. Inserts the vector into VectorStore, records the offset.
4. Inserts a record into RecordStore with the assigned offset.
5. Returns the assigned ID.

**Output (success):**

```json
{ "id": 42 }
```

**Output (error):** JSON-RPC error string. Since Phase 2, `index_vector` now validates
`vector.len() == 384` (locked by the embedder at startup) and returns
`"vector dim mismatch: expected 384 (locked by embedder), got N"` for any other length.

### `index_document`

Index a plain-text document. The server computes the embedding — no vector required from
the client.

**Input schema:**

```json
{
  "text":     "<string>  -- text to embed and store (must not be blank)",
  "metadata": "{ ... }   -- optional JSON object; values coerced to strings"
}
```

**Behavior:**

1. Rejects blank text (`text.trim().is_empty()`) with `"text must not be empty"`.
2. **Two-phase lock:** clones `Arc<Embedder>` under the state Mutex, drops the lock,
   runs `embedder.embed(text)` (BERT forward pass, no lock held), re-acquires the lock.
3. Inserts the 384-dim vector into VectorStore.
4. Inserts a record into RecordStore.
5. Returns the assigned ID.

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "tools/call",
  "params": {
    "name": "index_document",
    "arguments": {
      "text": "meeting with engineering team on friday at 3pm",
      "metadata": { "category": "calendar" }
    }
  }
}
```

**Example response:**

```json
{ "id": 0 }
```

**Error cases:**

| Condition | Error message |
|-----------|---------------|
| `text` is blank or whitespace-only | `"text must not be empty"` |
| Embedder inference failure | `"embed error: embedder inference failed: ..."` |

### `search`

Search for the K nearest vectors to a query vector.

**Input schema:**

```json
{
  "vector": "[f32]              -- query embedding vector",
  "k":      "<usize>            -- number of nearest neighbors",
  "metric": "<string|null>      -- 'euclidean', 'cosine', or 'dot'; defaults to server default"
}
```

**Metric values:** `"euclidean"`, `"cosine"`, `"dot"`. Omit or set to `null` to use the server-wide default (Euclidean at startup).

**Behavior:**

- If the store is uninitialized (no documents indexed yet), returns `[]`.
- Otherwise runs `BruteForce::search` with the chosen metric.
- Results are sorted ascending by score (closest first).

**Output (success):**

```json
[
  { "id": 7, "score": 0.12, "text": "most similar document" },
  { "id": 3, "score": 0.45, "text": "second most similar" }
]
```

Note: the `search` tool output does not include `metadata`. Use `search_document` if
metadata in results is needed.

### `search_document`

Semantic search by natural-language query. The server computes the query embedding.
Default metric is Cosine (see "Default Metric Divergence" above).

**Input schema:**

```json
{
  "query":  "<string>          -- query text; server embeds it (must not be blank)",
  "k":      "<usize>           -- number of nearest neighbors",
  "metric": "<string|null>     -- 'euclidean', 'cosine', or 'dot'; defaults to 'cosine'"
}
```

**Behavior:**

1. Rejects blank query with `"query must not be empty"`.
2. **Two-phase lock:** clones `Arc<Embedder>` under the state Mutex, drops the lock,
   runs `embedder.embed(query)`, re-acquires the lock.
3. Runs `BruteForce::search` with the resulting 384-dim query vector.
4. Returns results with `id`, `score`, `text`, and `metadata` (as JSON object).

**Example interaction (calendar / shopping / incident demo):**

Index three documents:
```json
index_document("meeting with engineering team on friday at 3pm", {"category": "calendar"}) → {"id": 0}
index_document("buy milk and bread on the way home", {"category": "shopping"})             → {"id": 1}
index_document("git commit hash a3f9b2 broke the deploy pipeline", {"category": "incident"}) → {"id": 2}
```

Query for meetings:
```json
{
  "jsonrpc": "2.0",
  "id": 20,
  "method": "tools/call",
  "params": {
    "name": "search_document",
    "arguments": { "query": "what meetings do I have this week?", "k": 3 }
  }
}
```

Response (scores vary; cosine range for L2-normalized 384-dim vectors is `[0.0, 2.0]`):
```json
[
  { "id": 0, "score": 0.42, "text": "meeting with engineering team on friday at 3pm", "metadata": { "category": "calendar" } },
  { "id": 2, "score": 0.89, "text": "git commit hash a3f9b2 broke the deploy pipeline", "metadata": { "category": "incident" } },
  { "id": 1, "score": 1.07, "text": "buy milk and bread on the way home", "metadata": { "category": "shopping" } }
]
```

The calendar document always ranks first for meeting-related queries; scores near `0.42`
indicate strong semantic match, scores near `1.07` indicate weak match (the shopping
document shares almost no semantic content with a meeting query).

**Error cases:**

| Condition | Error message |
|-----------|---------------|
| `query` is blank or whitespace-only | `"query must not be empty"` |
| Embedder inference failure | `"embed error: embedder inference failed: ..."` |
| Unknown metric string | `"unknown metric: <value>"` |

### `delete`

Delete a document by its ID.

**Input schema:**

```json
{
  "id": "<u64>  -- ID returned by index_vector"
}
```

**Behavior:** Calls `brute::delete`, which performs a coordinated swap-remove on both VectorStore and RecordStore. See the [BruteForce documentation](brute-force.md) for the offset-fixup protocol.

**Output (success):**

```json
{ "success": true }
```

**Output (error):** JSON-RPC error string, e.g., `"record not found: 999"`.

### `stats`

Get current database statistics. Takes no parameters.

**Output:**

```json
{
  "count":     12,
  "dimension": 384,
  "metric":    "Euclidean"
}
```

`dimension` is always `384` in Phase 2 and later — the store is initialized at startup
with the embedder's dimension and never changes. The field name is preserved for
backward compatibility with Phase 1 clients that checked for `dimension: null`.

## Server Info

The server advertises itself via `ServerHandler::get_info()`:

- **Name:** `"nanovec"`
- **Version:** taken from `CARGO_PKG_VERSION` at compile time
- **Capabilities:** tools enabled
- **Instructions:** `"NanoVec is an in-memory vector database for ephemeral AI agent working memory."`

## rmcp Integration

| Concern | How handled |
|---------|-------------|
| Tool registration | `#[tool_router]` macro on `impl NanoVecServer` |
| Tool declaration | `#[tool(name = "...", description = "...")]` on each method |
| Parameter deserialization | `Parameters<T>` extractor; `T` implements `Deserialize + JsonSchema` |
| Routing | `Router::new(self).with_tools(tool_routes)` |
| Transport | `rmcp::transport::io::stdio()` |
| Lifecycle | `service.waiting().await` blocks until client disconnects |

Parameter structs are in `src/mcp/tools.rs` and derive `serde::Deserialize` and `schemars::JsonSchema` so rmcp can generate and validate input schemas automatically.

## Metric Parsing

All tools that accept a `metric` parameter share the same string-to-enum mapping:

| String value | Metric |
|--------------|--------|
| `"euclidean"` | `Metric::Euclidean` |
| `"cosine"` | `Metric::Cosine` |
| `"dot"` | `Metric::DotProduct` |
| omitted / `null` | tool-specific default (see "Default Metric Divergence") |

Unknown strings return an error: `"unknown metric: <value>"`.

## File Map

| File | Role |
|------|------|
| `src/main.rs` | Binary entry point; configures tracing to stderr, calls `server::run()` |
| `src/server/mod.rs` | `run()` -- loads embedder, constructs server, starts stdio transport, awaits shutdown |
| `src/mcp/mod.rs` | `NanoVecState`, `NanoVecServer`, all six tool implementations, metric/metadata parsers |
| `src/mcp/tools.rs` | `IndexVectorParams`, `IndexDocumentParams`, `SearchParams`, `SearchDocumentParams`, `DeleteParams` structs |

## Dependencies

- `rmcp` 1.3.0 -- MCP protocol implementation, stdio transport, tool macros
- `serde` / `serde_json` -- parameter deserialization and JSON response serialization
- `schemars` -- JSON Schema generation for tool input schemas
- `tokio` -- async runtime
- `tracing` / `tracing-subscriber` -- structured logging to stderr
- `crate::distance` -- `Metric`, `distance_fn`
- `crate::embed` -- `Embedder`, `EmbedError` (see [embed.md](embed.md))
- `crate::index::brute` -- `BruteForce::search`, `delete`
- `crate::store` -- `VectorStore`
- `crate::store::record` -- `RecordStore`

## Test Coverage

2 integration tests covering the full JSON-RPC flow (files in `tests/integration/`):

| Test | File | What it covers |
|------|------|----------------|
| `test_mcp_full_flow` | `mcp_stdio.rs` | `index_vector` (384-dim unit vector), `search`, `delete`, `stats` — raw-vector path; verifies dimension is locked at 384 throughout |
| `test_semantic_search_end_to_end` | `mcp_embedding.rs` | `index_document` (3 documents with metadata), `search_document` (3 semantic queries with correct top-1 assertions), `delete`, `stats`, post-delete search — full embedded path |

Both tests spawn the real `nanovec` binary as a child process and communicate via
newline-delimited JSON-RPC 2.0 over stdio. The embedding test requires HuggingFace
model cache (`~/.cache/huggingface/`) or internet access to download the model on
first run.
