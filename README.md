# NanoVec

[Українською →](README.uk.md)

> Lightweight, zero-dependency, in-memory vector database — exposed as an MCP server for ephemeral AI-agent working memory.

NanoVec is a from-scratch Rust implementation of a vector database for AI agents that need short-lived semantic memory. It targets sub-millisecond similarity search, a tiny static binary, and a single external interface: the Model Context Protocol over stdio.

**Status:** Phases 1 (MVP), 2 (server-side embeddings), and 2.5 (search ergonomics — `clear`, expanded `stats`, similarity field) are complete. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for what comes next.

## What you get

- **In-process vector store.** Flat `Vec<f32>` Structure-of-Arrays layout, fixed dimension, O(1) insert / get / swap-remove.
- **Server-side embeddings.** Bundled `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized) via [`candle`](https://github.com/huggingface/candle) — pure Rust, CPU-only, no Python or ONNX.
- **Brute-force KNN** with Euclidean, cosine, and dot-product metrics. KD-Tree and SIMD acceleration are planned for later phases.
- **MCP over stdio.** Six tools: `index_vector`, `search`, `index_document`, `search_document`, `delete`, `stats`.
- **Ephemeral by design.** No persistence, no WAL — data lives in RAM and dies with the process.
- **Zero indexing dependencies.** No FAISS, no HNSWLIB. All algorithms are written from first principles.

## Quick start

### Build

```bash
git clone https://github.com/nikita-voloshyn/nanovec
cd nanovec
cargo build --release
```

The binary is `target/release/nanovec`.

### Connect to Claude Code (or any MCP client)

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

On first run NanoVec downloads the embedding model (~90 MB) from HuggingFace Hub into `~/.cache/huggingface/hub/`. Cold start takes 10–30 s; subsequent runs warm-start in ~30 ms.

All logs go to stderr; stdout is reserved for MCP JSON-RPC traffic.

## MCP tools

| Tool | Input | Output |
|------|-------|--------|
| `index_document` | `text`, optional `metadata` | `{ "id": u64 }` |
| `search_document` | `query`, `k`, optional `metric` | `[{ id, text, metadata, distance, similarity, score }]` |
| `index_vector` | `text`, `vector` (length 384), optional `metadata` | `{ "id": u64 }` |
| `search` | `vector` (length 384), `k`, optional `metric` | `[{ id, text, metadata, distance, similarity, score }]` |
| `delete` | `id` | `{ "success": bool }` |
| `clear` | — | `{ "deleted": usize }` |
| `stats` | — | `{ count, dimension, default_metric, embedder, metric }` |

Default metric is `euclidean` for the raw-vector path and `cosine` for the document path. Override with `"euclidean"`, `"cosine"`, or `"dot"`.

Search results carry both `distance` (raw, lower = closer) and `similarity` (metric-aware, higher = closer). The `score` field is a backward-compat alias for `distance`. `clear` resets the ID counter so the next inserted document gets `id: 0`.

The vector dimension is locked to 384 once the embedder loads. `index_vector` calls with any other dimension are rejected.

## Architecture

```
src/
  store/       VectorStore (flat SoA), RecordStore (text + metadata + ID map)
  distance/    Scalar Euclidean, cosine, dot
  heap/        Bounded max-heap for top-K selection
  index/       Brute-force KNN
  embed/       candle + all-MiniLM-L6-v2 loader
  mcp/         rmcp tool definitions, server router
  server/      Async entry point, embedder warmup, stdio wiring
```

Per-component reference: [`docs/components/`](docs/components/).

## Documentation

- [`docs/README.md`](docs/README.md) — bilingual documentation index
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phase plan
- [`docs/PHASE-1-PRESENTATION.md`](docs/PHASE-1-PRESENTATION.md) — MVP summary
- [`docs/PHASE-2-PRESENTATION.md`](docs/PHASE-2-PRESENTATION.md) — embeddings summary
- [`docs/components/`](docs/components/) — per-module reference
- [`docs/plans/`](docs/plans/) — phase plans and dispatch reports
- [`docs/TEST-CASES.md`](docs/TEST-CASES.md) — test catalogue

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

NanoVec follows a TDD-first / YAGNI workflow. The agent and skill setup that drives day-to-day development is documented in [`CLAUDE.md`](CLAUDE.md).

## License

MIT.
