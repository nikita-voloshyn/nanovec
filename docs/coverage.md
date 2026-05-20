# NanoVec Documentation Coverage

## Phase 1 (Complete)

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| VectorStore | `src/store/mod.rs` | `docs/components/vector-store.md` | documented |
| RecordStore | `src/store/record.rs` | `docs/components/record-store.md` | documented |
| Distance Functions | `src/distance/{mod,euclidean,cosine,dot}.rs` | `docs/components/distance.md` | documented |
| BoundedMaxHeap + BruteForce | `src/heap/mod.rs`, `src/index/brute.rs` | `docs/components/brute-force.md` | documented |
| MCP Server | `src/mcp/`, `src/server/`, `src/main.rs` | `docs/components/mcp-server.md` | documented |

## Phase 2 (Complete — Vectors → Text)

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| Embedder (candle + all-MiniLM-L6-v2, 384-dim, L2-normalized) | `src/embed/mod.rs` | `docs/components/embed.md` | documented |
| `index_document` / `search_document` MCP tools | `src/mcp/tools.rs`, `src/mcp/mod.rs` | `docs/components/mcp-server.md` | documented |

## Phase 2.5 (Complete — Search Ergonomics)

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| `VectorStore::clear()` / `RecordStore::clear()` | `src/store/mod.rs`, `src/store/record.rs` | `docs/components/vector-store.md`, `docs/components/record-store.md` | inline in API tables; behavior documented in mcp-server.md `clear` section |
| `Embedder::MODEL_NAME` + `model_name()` accessor | `src/embed/mod.rs` | `docs/components/embed.md` | documented |
| `clear` MCP tool | `src/mcp/mod.rs` | `docs/components/mcp-server.md` | documented |
| `stats` expansion (`default_metric`, `embedder` blocks) | `src/mcp/mod.rs` | `docs/components/mcp-server.md` | documented |
| `distance` + `similarity` fields on search results | `src/mcp/mod.rs` | `docs/components/mcp-server.md` ("Score Fields" section) | documented |
| Model limitations note (English-trained, weak on non-English) | `src/embed/mod.rs` | `docs/components/embed.md` ("Model Limitations" section) | documented |

## Phase 3 (Complete — Scalar → SIMD)

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| SIMD Distance (AVX2/NEON/portable_simd) | `src/simd/` | `docs/components/distance.md` | documented |

## Phase 4 (Complete — Pile → Organized)

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| Collections (multi-namespace) | `src/store/collections.rs` | inline in mcp-server.md | documented |
| Metadata Filtering | `src/index/brute.rs` (`filter` param on `search`) | `docs/components/brute-force.md` | documented |

## Phase 5 (Complete — O(n) → O(log n))

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| KD-Tree Index | `src/index/kdtree.rs` | `docs/components/kdtree.md` | documented |
| `BoundedMaxHeap::peek_worst` (pruning support) | `src/heap/mod.rs` | `docs/components/kdtree.md` ("Heap Extension" section) | documented |
| `should_use_kdtree` heuristic | `src/index/kdtree.rs` | `docs/components/kdtree.md` ("When to use vs BruteForce" section) | documented |

## Phase 6 (Complete — Single → Multi-agent)

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| NanoVecDatabase (two-level RwLock, budget, LRU) | `src/store/database.rs` | `docs/components/database.md` | documented |
| Memory budget + per-collection LRU eviction | `src/store/database.rs` | `docs/components/database.md` ("Budget" + "LRU tick" sections) | documented |
| Pinned collections (`default`, `_conn_*`) | `src/store/database.rs` | `docs/components/database.md` ("Pinned collections" section) | documented |
| Streamable HTTP transport (axum + rmcp) | `src/transport/http.rs`, `src/transport/mod.rs` | `docs/components/transport-http.md` | documented |
| `connection_id` field on every tool param | `src/mcp/tools.rs`, `src/mcp/mod.rs` | `docs/components/transport-http.md` ("Connection isolation" section) | documented |
| `memory` MCP tool + extended `stats` (memory block) | `src/mcp/mod.rs` | `docs/components/database.md` + `docs/components/mcp-server.md` | documented |

## Phase 7 (Complete — Exact → Approximate)

| Component | Source | Doc file | Status |
|-----------|--------|----------|--------|
| HNSW from scratch (Malkov & Yashunin Algorithm 4 heuristic) | `src/index/hnsw.rs` | `docs/components/hnsw.md` | documented |
| `Collection.hnsw: Option<Hnsw>` + auto-invalidation on mutation | `src/store/collections.rs`, `src/mcp/mod.rs` | `docs/components/hnsw.md` ("Index lifecycle" section) | documented |
| `rebuild_index` MCP tool | `src/mcp/mod.rs`, `src/mcp/tools.rs` | `docs/components/hnsw.md` ("MCP tool" section) | documented |
| HNSW search dispatch (cosine only; fallback to brute on metric mismatch) | `src/mcp/mod.rs` | `docs/components/hnsw.md` ("Trade-offs" section) | documented |
| Real-corpus recall sweep | `tests/integration/hnsw_stress.rs`, `benches/hnsw_sweep.rs` | `docs/components/hnsw.md` ("Measured Performance" section) | documented |

## Coverage Notes

- All Phase 1–7 public modules have corresponding component documentation files.
- Doc comments (`///`) exist on all public items.
- Benchmark numbers from `cargo bench` and `cargo test ... --ignored` runs on Apple Silicon (NEON) are captured inline in the Phase 3 (SIMD), Phase 5 (KD-Tree), Phase 6 (concurrency), and Phase 7 (HNSW recall sweep) docs.
- Integration tests in `tests/integration/` cover: MCP stdio (4), concurrency + LRU (6), streamable HTTP e2e (2), HNSW recall + filter (4), rebuild_index (2), real-corpus stress (2, `#[ignore]`). Total 18 integration + 104 unit tests passing.
- Real-corpus tests (`hnsw_real_corpus`, `hnsw_stress`) require the HuggingFace model cache (90 MB cold) and take 10s–8m to run, so they're `#[ignore]`'d. Numbers from those runs feed into `docs/components/hnsw.md`.
