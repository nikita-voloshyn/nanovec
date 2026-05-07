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

## Phase 3 (Next — Scalar → SIMD)

| Component | Planned source | Status |
|-----------|---------------|--------|
| SIMD Distance (AVX2/NEON/portable_simd) | `src/simd/` | not started |

## Phase 4 (Planned — Pile → Organized)

| Component | Planned source | Status |
|-----------|---------------|--------|
| Collections (multi-namespace) | `src/store/` | not started |
| Metadata Filtering | `src/index/` | not started |

## Phase 5 (Planned — O(n) → O(log n))

| Component | Planned source | Status |
|-----------|---------------|--------|
| KD-Tree Index | `src/index/kdtree.rs` | not started |

## Coverage Notes

- All Phase 1 and Phase 2 public modules have corresponding component documentation files.
- Doc comments (`///`) exist on all public items in Phase 1 and Phase 2 source.
- No benchmark results are available yet; performance claims in current docs describe algorithmic complexity only. Benchmark evidence will be added after Phase 3 SIMD work and `cargo bench` runs are recorded.
- Integration tests in `tests/` exercise the MCP server end-to-end; both tests are summarized in `docs/components/mcp-server.md`.
- Phase 2 Embedder tests require HuggingFace model cache or internet access; see `docs/components/embed.md` for caching details.
