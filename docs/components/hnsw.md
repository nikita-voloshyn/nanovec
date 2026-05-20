# HNSW Index (Phase 7)

## Purpose

`Hnsw` (`src/index/hnsw.rs`) implements an approximate nearest-neighbor
index from scratch following Malkov & Yashunin's "Efficient and robust
approximate nearest neighbor search using Hierarchical Navigable Small
World graphs" (arXiv:1603.09320). It complements `BruteForce` and `KdTree`:

- `BruteForce` — `O(n)`, exact, any metric. Best up to ~10k vectors.
- `KdTree` — geometric pruning, exact, Euclidean only. Useful for low-dim
  (≤ 50). Still wins at 384-dim but with a much smaller margin (~7×).
- `Hnsw` — `O(log n)`-ish, approximate, cosine (today). Wins from ~10k
  vectors upward; the bigger the corpus, the bigger the gap.

HNSW is **opt-in**: collections start in brute-force mode. The MCP
`rebuild_index` tool builds the graph on demand. Any subsequent mutation
to the underlying store automatically invalidates the index.

## Public API

```rust
// src/index/hnsw.rs

pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub m_l: f64,
}

impl HnswParams {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self;
    // Default: M=16, ef_c=200, ef_s=50, m_l = 1 / ln(16).
}

pub struct Hnsw { /* private */ }

impl Hnsw {
    pub fn new(dimension: usize, params: HnswParams) -> Self;
    pub fn with_seed(dimension: usize, params: HnswParams, seed: u64) -> Self;
    pub fn build(store: &VectorStore, params: HnswParams) -> Self;

    pub fn insert(&mut self, id: u32, store: &VectorStore);
    pub fn search(
        &self,
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        filter: Option<&[(String, String)]>,
    ) -> Vec<SearchResult>;

    pub fn params(&self) -> &HnswParams;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn dimension(&self) -> usize;
    pub fn approx_bytes(&self) -> u64;
}
```

### MCP tool

```
rebuild_index(
  collection?: string,
  connection_id?: string,
  kind?: "hnsw" | "none",        // default "hnsw"
  m?: usize,                      // default 16
  ef_construction?: usize,        // default 200
  ef_search?: usize,              // default 50
) -> {
  kind, collection, nodes, m, ef_construction, ef_search,
  build_ms, memory_bytes
}
```

## Internal Design

### Graph layout

```
layers: Vec<Vec<Vec<u32>>>     // layers[layer][node] -> neighbor ids
node_levels: Vec<u8>            // node_levels[i] = top layer of node i
entry_point: Option<u32>        // current top-layer entry
```

- `u32` node ids are sufficient (4G vectors per index ≫ realistic
  workloads).
- A node at level `L` participates in layers `0..=L`. Layers beyond a node's
  level hold an empty neighbor list for it.
- The outer index of `layers` is the layer; the middle index is the global
  node id — flat, cache-friendly traversal.

### Layer assignment

```rust
fn sample_level(&mut self) -> u8 {
    let u = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
    let lvl = (-u.ln() * self.params.m_l).floor() as i64;
    lvl.clamp(0, u8::MAX as i64) as u8
}
```

`m_l = 1 / ln(M)` per the paper. With `M=16`:
- `P(level = 0)` ≈ `1 - exp(-ln 16)` = `1 - 1/16` ≈ **93.75%**.
- `P(level = k)` ≈ `(1/16)^k × (1 - 1/16)`.

Verified by the `build_assigns_geometric_levels` unit test (0.88..0.98
range over 1000 samples).

### Search

```
search(query, k):
    ep = entry_point
    for layer in (1..=top_layer).rev():
        ep = search_layer(query, [ep], ef=1, layer).first()       // greedy
    candidates = search_layer(query, [ep], ef=ef_search, layer=0) // beam
    return top-k from candidates (with optional filter)
```

`search_layer` is Algorithm 2 in the paper: a beam search with a min-heap
of candidates (next to expand) and a max-heap of best-`ef` results.

Distance is `simd::cosine(query, stored_vector)` — HNSW inherits Phase 3's
NEON/AVX2 acceleration automatically.

### Insert

```
insert(id, store):
    level = sample_level()
    ep = entry_point or (id is first -> set entry_point and return)

    // Phase A: greedy descent above `level` with ef=1.
    for layer in (level+1..=entry_level).rev():
        ep = search_layer(query, [ep], 1, layer).first().id

    // Phase B: beam search + connect on every layer at or below `level`.
    for layer in (0..=level.min(entry_level)).rev():
        candidates = search_layer(query, [ep,...], ef_construction, layer)
        selected   = select_neighbors_heuristic(candidates, M)
        for n in selected:
            link id <-> n on this layer
            if neighbor_count(n, layer) > cap:
                prune_neighbors(n, layer, cap)
        next_ep = selected
```

### Heuristic neighbor selection (Algorithm 4)

The "simple" variant picks top-M by distance and works passably. The
**heuristic** variant from Algorithm 4 picks M *diverse* neighbors:

```
for cand in candidates (ascending by dist to query):
    if some selected `s` has `dist(s, cand) < dist(cand, query)`:
        skip cand   // redundant — `s` already covers this direction
    else:
        add cand
```

This is what lets recall hit ~99% at `M=16`. Without it, the same `M`
produces ~85–90% recall.

### Index lifecycle

```
Collection
├── hnsw: Option<Hnsw>           // None by default
└── invalidate_index() -> ()     // called on every mutation
```

Phase 6 T4 wired `rebuild_index` into the MCP layer. The handler calls
`Hnsw::build(&coll.store, params)` and stores the result on the
collection. Every subsequent `index_*`, `delete`, and `clear` invalidates
the index. Search uses HNSW iff `coll.hnsw.is_some()` AND the requested
metric is cosine; otherwise it falls back to brute-force.

### Trade-offs

- **Cosine-only.** HNSW is built against `simd::cosine`. Search with
  `metric=euclidean` falls back to brute-force even when an HNSW exists.
  Extending to other metrics requires either a per-metric graph or a
  parameterised `distance()`.
- **Single-threaded build.** Sequential `insert` per node. 10k × 384-dim
  builds in ~2s; 1M would take ~3.5 minutes. Parallel build is plan
  Phase 7 follow-up.
- **No persistence.** Graph is reconstructed from `VectorStore` on every
  `rebuild_index` call. Matches the "ephemeral by design" project rule.
- **Filter is post-applied.** `search` retrieves `ef_search` candidates,
  then applies the metadata filter. For highly-selective filters this
  may return fewer than `k` results; pre-filter inside the graph is
  hard to do correctly under HNSW's beam structure.

## Measured Performance

### Synthetic corpus (10k × 384-dim sine-based, L2-normalised)

| M | ef_c | ef_s | Build (ms) | p50 search (µs) | Speedup | Recall@10 |
|---|------|------|-----------:|----------------:|--------:|----------:|
| 8 | 100 | 30 | 535 | 15 | 54.9× | 1.000 |
| 16 | 200 | 50 | 982 | 19 | 43.4× | 1.000 |
| 32 | 200 | 50 | 881 | 17 | 48.5× | 1.000 |
| 32 | 400 | 200 | 1806 | 62 | 13.3× | 1.000 |

Brute-force baseline: 824 µs p50.

### Real corpus (10k MiniLM embeddings, 30 semantic queries)

| ef_search | p50 (µs) | Speedup | Recall@10 |
|----------:|---------:|--------:|----------:|
| 10 | 43 | 19.1× | 0.850 |
| 25 | 58 | 14.2× | **0.990** |
| 50 (default) | 92 | 8.9× | 0.993 |
| 100 | 156 | 5.3× | 1.000 |
| 200 | 210 | 3.9× | 1.000 |

Brute-force baseline: 822 µs p50. Memory ~625 KB for 10k graph.

Source: `benches/hnsw_sweep.rs` + `tests/integration/hnsw_stress.rs`.

## Error Modes

The current implementation panics on:
- `insert` with non-sequential id (must be `0..N` in order).
- `dist_to` on an out-of-range id.

These are programming errors in `Hnsw::build`'s caller, not user-facing
runtime conditions, so panics are acceptable. The MCP layer never
constructs `Hnsw` manually — it always goes through `build`.

## Test Coverage

| Test | Location | Verifies |
|------|----------|----------|
| `empty_graph_search_returns_no_results` | `src/index/hnsw.rs` | Empty graph returns empty vec |
| `single_node_search_returns_itself` | `src/index/hnsw.rs` | Self-distance is 0 |
| `recall_on_500_random_vectors_matches_brute_force` | `src/index/hnsw.rs` | recall@10 ≥ 0.85 (actual 1.0) |
| `build_assigns_geometric_levels` | `src/index/hnsw.rs` | ~94% nodes at layer 0 |
| `neighbor_caps_respected_on_layer_0` | `src/index/hnsw.rs` | No layer-0 node has > 2M neighbors |
| `recall_on_2k_384_dim_corpus_meets_target` | `tests/integration/hnsw_recall.rs` | recall@10 ≥ 0.85 on 2k corpus |
| `recall_improves_with_higher_ef_search` | `tests/integration/hnsw_recall.rs` | Monotonic recall vs ef_search |
| `hnsw_results_match_simd_cosine` | `tests/integration/hnsw_recall.rs` | Reported scores match `simd::cosine` ±1e-5 |
| `hnsw_search_with_filter_returns_only_matching` | `tests/integration/hnsw_recall.rs` | Post-filter correctness |
| `rebuild_index_builds_hnsw_and_search_uses_it` | `tests/integration/rebuild_index.rs` | End-to-end via MCP |
| `rebuild_index_uses_hnsw_metric_only_for_cosine` | `tests/integration/rebuild_index.rs` | Metric fallback works |
| `recall_on_real_minilm_embeddings_at_k10` (`#[ignore]`) | `tests/integration/hnsw_real_corpus.rs` | recall@10 = 1.000 on 200 real embeddings |
| `hnsw_stress_10k_real_embeddings` (`#[ignore]`) | `tests/integration/hnsw_stress.rs` | ef_search sweep on 10k real embeddings |
