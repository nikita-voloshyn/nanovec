# NanoVec

[Українською →](README.uk.md)

> Lightweight, zero-dependency, in-memory vector database — exposed as an MCP server for ephemeral AI-agent working memory.

NanoVec is a from-scratch Rust implementation of a vector database for AI agents that need short-lived semantic memory. SIMD-accelerated distance, a self-built HNSW index, two transports (stdio + streamable HTTP), and a single external interface: the Model Context Protocol.

**Status:** all seven planned phases complete. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

| Phase | What it added |
|-------|---------------|
| 1 — MVP | VectorStore (SoA), RecordStore, scalar distance, brute-force KNN, MCP stdio |
| 2 — Embeddings | candle + `all-MiniLM-L6-v2` (384-dim), `index_document` / `search_document` |
| 2.5 — Ergonomics | `clear`, expanded `stats`, `distance` / `similarity` fields |
| 3 — SIMD | NEON (aarch64) + AVX2 (x86_64) distance, runtime dispatch, property tests |
| 4 — Collections | Named collections, metadata filtering, multi-tenant in one process |
| 5 — KD-Tree | Spatial index for low-dim, geometric pruning |
| 6 — Multi-agent | Two-level `parking_lot::RwLock`, memory budget, LRU eviction, streamable HTTP, `connection_id` |
| 7 — Approximate | HNSW from scratch (Malkov & Yashunin Algorithm 4 heuristic), `rebuild_index` MCP tool |

## What you get

- **In-process vector store.** Flat `Vec<f32>` SoA layout, fixed dimension, O(1) insert / get / swap-remove.
- **Server-side embeddings.** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized) via [`candle`](https://github.com/huggingface/candle) — pure Rust, CPU-only, no Python or ONNX.
- **SIMD distance.** Cosine / Euclidean / dot product, dispatched at runtime to NEON (Apple Silicon) or AVX2+FMA (x86_64), with property tests against a scalar reference.
- **Three indexes.** Brute-force (any metric, default), KD-Tree (low-dim), HNSW (cosine, opt-in via `rebuild_index`).
- **Multi-agent.** `parking_lot::RwLock` per collection so readers don't block each other; soft memory budget with per-collection LRU eviction.
- **Two transports.** stdio (default, local) and streamable HTTP (axum-backed, opt-in via `NANOVEC_SSE_ADDR`).
- **Multi-tenant.** Each MCP tool accepts `connection_id`; auto-creates pinned `_conn_<id>` collections that are excluded from eviction.
- **Ephemeral by design.** No persistence, no WAL — data lives in RAM and dies with the process.
- **Zero external indexing libraries.** No FAISS, HNSWLIB, Annoy, or Qdrant. All algorithms written from first principles.

## Measured performance

All numbers from `cargo bench` and `cargo test --release` on Apple Silicon (M-series, NEON).

### SIMD distance @ 384-dim

| Op | Scalar | NEON | Speedup |
|----|------:|----:|--------:|
| `dot_product` | 296 ns | 32 ns | **9.3×** |
| `euclidean` | 305 ns | 33 ns | 9.2× |
| `cosine` | 414 ns | 95 ns | 4.4× (3 reductions cap the gain) |

### KD-Tree vs brute-force, k=10

| N × dim | Brute | KD-Tree | Speedup |
|---------|------:|--------:|--------:|
| 1k × 8-dim | 6.3 µs | 1.7 µs | 3.7× |
| 10k × 8-dim | 44.5 µs | 1.5 µs | 29.7× |
| 1k × 384-dim | 42.6 µs | 17.7 µs | 2.4× |
| 10k × 384-dim | 448 µs | 63.5 µs | **7.1×** (curse of dimensionality tempers the gain but doesn't kill it) |

### Phase 6 concurrency (5k × 384-dim corpus, k=10)

| Scenario | 1 thread | 8 threads | Scaling |
|----------|---------:|----------:|--------:|
| Same collection (readers) | 1 922 q/s | 13 389 q/s | **6.97×** |
| 8 collections, 8 threads | 11 648 q/s | 71 920 q/s | 6.17× |

### Phase 7 HNSW (10k real MiniLM embeddings, 30 semantic queries, k=10)

Brute-force baseline: 822 µs p50.

| ef_search | p50 search | Speedup vs brute | Recall@10 |
|----------:|-----------:|----------------:|----------:|
| 10 | 43 µs | **19.1×** | 0.850 |
| 25 | 58 µs | 14.2× | **0.990** (sweet spot) |
| 50 (default) | 92 µs | 8.9× | 0.993 |
| 100 | 156 µs | 5.3× | 1.000 |

Reproduce: `cargo bench --bench simd_vs_scalar --bench kdtree --bench hnsw_sweep` + `cargo test --release -- --ignored --nocapture --test-threads=1 hnsw_stress`.

## Quick start

### Build

```bash
git clone https://github.com/nikita-voloshyn/nanovec
cd nanovec
cargo build --release
```

The binary is `target/release/nanovec`.

### Connect to Claude Code (stdio)

Add the binary to your MCP client config. Example for Claude Code (`.mcp.json` at the repo root):

```json
{
  "mcpServers": {
    "nanovec": {
      "command": "/absolute/path/to/nanovec/target/release/nanovec"
    }
  }
}
```

On first run NanoVec downloads the embedding model (~90 MB) from HuggingFace Hub into `~/.cache/huggingface/hub/`. Cold start ~10–30 s; warm start ~30 ms.

### Connect over HTTP

```bash
NANOVEC_SSE_ADDR=127.0.0.1:8421 cargo run --release
```

The streamable HTTP endpoint mounts at `/mcp`. The server has no authentication — keep it on localhost or behind a reverse proxy with auth.

### Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `NANOVEC_SSE_ADDR` | unset | If set to `host:port`, runs streamable HTTP instead of stdio. |
| `NANOVEC_MEMORY_LIMIT_MB` | 512 | Soft per-process memory cap; triggers LRU eviction of unpinned collections. |
| `RUST_LOG` | warn | Standard `tracing-subscriber` filter. `info`, `debug`, etc. |

All logs go to stderr; stdout is reserved for MCP JSON-RPC traffic (stdio mode).

## MCP tools

| Tool | Input | Output |
|------|-------|--------|
| `index_document` | `text`, optional `metadata`, optional `collection` or `connection_id` | `{ id }` |
| `search_document` | `query`, `k`, optional `metric` / `filter` / `collection` / `connection_id` | `[{ id, text, metadata, distance, similarity, score }]` |
| `index_vector` | `text`, `vector`, optional `metadata` / `collection` / `connection_id` | `{ id }` |
| `search` | `vector`, `k`, optional `metric` / `filter` / `collection` / `connection_id` | `[...]` |
| `delete` | `id`, optional `collection` / `connection_id` | `{ success }` |
| `clear` | optional `collection` / `connection_id` | `{ deleted }` |
| `create_collection` | `name`, `dimension` | `{ created, dimension }` |
| `list_collections` | — | `{ collections: [{ name, count, dimension }] }` |
| `drop_collection` | `name` | `{ dropped }` |
| `stats` | — | `{ count, total_count, dimension, default_metric, embedder, collections, memory: { used_bytes, limit_bytes, used_pct, collections_count }, metric }` |
| `memory` | — | `{ used_bytes, limit_bytes, collections: [{ name, count, approx_bytes, last_accessed_tick, pinned }] }` |
| `rebuild_index` | optional `collection` / `connection_id`, `kind` (`"hnsw"` or `"none"`), optional `m` / `ef_construction` / `ef_search` | `{ kind, collection, nodes, m, ef_construction, ef_search, build_ms, memory_bytes }` |

Default metric: `euclidean` for the raw-vector path, `cosine` for the document path. Override with `"euclidean"`, `"cosine"`, or `"dot"`.

Search results carry `distance` (raw, lower = closer) and `similarity` (metric-aware, higher = closer). `score` is a backward-compat alias for `distance`.

### Multi-tenant via `connection_id`

When a tool is called with `connection_id: "alice"` and no explicit `collection`, the call lands in `_conn_alice` (auto-created on first use, pinned in memory). Two clients with different `connection_id`s never see each other's data unless they share an explicit `collection` name.

### HNSW workflow

```jsonc
// Build an HNSW index for the current collection.
{"name": "rebuild_index", "arguments": {"kind": "hnsw", "m": 16, "ef_search": 25}}
// → {"kind":"hnsw","nodes":10000,"build_ms":2046,"memory_bytes":620000, ...}

// Subsequent search calls automatically use HNSW when metric=cosine.
{"name": "search_document", "arguments": {"query": "...", "k": 10}}

// Any mutation auto-invalidates the index — call rebuild_index again
// to rebuild, or "kind": "none" to drop it explicitly.
```

## Architecture

```
src/
  store/       VectorStore (flat SoA), RecordStore (text + metadata + ID map)
  store/       database.rs   NanoVecDatabase: two-level RwLock + budget + LRU
  distance/    Scalar Euclidean / cosine / dot
  simd/        NEON + AVX2 + scalar fallback, runtime dispatch
  heap/        BoundedMaxHeap for top-K selection
  index/       brute.rs (linear scan), kdtree.rs (geometric pruning), hnsw.rs (Phase 7)
  embed/       candle + all-MiniLM-L6-v2 loader
  mcp/         rmcp tool definitions, server router
  transport/   stdio.rs + http.rs (axum-backed) + select_from_env()
  server/      Async entry point, embedder warmup, transport dispatch
```

Per-component reference: [`docs/components/`](docs/components/).

## Documentation

- [`docs/README.md`](docs/README.md) — bilingual documentation index
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phase plan
- [`docs/PRESENTATION-PL.html`](docs/PRESENTATION-PL.html) — 15-slide project summary (Polish)
- [`docs/PROJECT-STATUS.html`](docs/PROJECT-STATUS.html) — project status (Russian)
- [`docs/MULTILINGUAL-BENCHMARK.md`](docs/MULTILINGUAL-BENCHMARK.md) — ScootGo FAQ recall@5 (EN/PL/UK)
- [`docs/components/`](docs/components/) — per-module reference (database, transport-http, hnsw, …)
- [`docs/plans/`](docs/plans/) — phase plans and dispatch reports

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Full benchmarks (a few minutes):

```bash
cargo bench --bench distance --bench simd_vs_scalar --bench kdtree --bench hnsw_sweep
```

Slow real-corpus tests (requires HuggingFace cache, ~7 min):

```bash
cargo test --release --test integration -- --ignored --nocapture --test-threads=1
```

NanoVec follows a TDD-first / YAGNI workflow. The agent and skill setup that drives day-to-day development is documented in [`CLAUDE.md`](CLAUDE.md).

## License

MIT.
