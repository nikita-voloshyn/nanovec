# KD-Tree Index

## Purpose

`KdTree` (`src/index/kdtree.rs`) provides sub-linear top-K nearest-neighbor search under the Euclidean metric. It complements — but does not replace — `BruteForce`. The tree is the right choice when:

- The corpus is large enough that brute-force `O(n)` becomes noticeable.
- The vector dimension is low enough that axis-aligned geometric pruning still does meaningful work.

For high-dimensional embeddings (e.g. the 384-dim MiniLM vectors used in production) brute-force is usually still the better default; see "When to use vs BruteForce" below.

## Public API

```rust
// src/index/kdtree.rs

pub struct KdTree { /* private */ }

impl KdTree {
    pub fn build(store: &VectorStore, records: &RecordStore) -> Self;
    pub fn search(
        &self,
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        filter: Option<&[(String, String)]>,
    ) -> Vec<SearchResult>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn dimension(&self) -> usize;
}

pub fn should_use_kdtree(n: usize, dim: usize) -> bool;
```

`SearchResult` is re-used from `brute.rs` — the tree and brute-force return identical result shapes so callers can swap implementations transparently.

## Algorithm

### Construction

```
build(store, records):
    items = [(record.id, record.offset) for record in records]
    build_subtree(items, depth=0)

build_subtree(items, depth):
    if items is empty: return NONE
    axis = depth % dimension
    select_nth_unstable_by(items, mid=len/2, key=|i| store[i.offset][axis])
    push KdNode { record_id, offset, axis, left=NONE, right=NONE }
    left  = build_subtree(items[..mid], depth+1)
    right = build_subtree(items[mid+1..], depth+1)
    patch children
```

Construction is `O(n log n)`: each level partitions all `n` items in linear time via `slice::select_nth_unstable_by`, and there are `log n` levels.

The tree is stored as a flat `Vec<KdNode>` rather than `Box<Node>` recursion. Each node holds two child indices (`u32`); `u32::MAX` represents a missing child. This keeps nodes small (24 bytes including alignment padding) and avoids heap allocation per node, which is friendlier to the cache prefetcher during traversal.

### Search with Geometric Pruning

```
search(query, k):
    heap = BoundedMaxHeap(k)
    recurse(root)

recurse(node):
    if node is NONE: return
    score = euclidean(query, store[node.offset])
    if filter passes: heap.push(score, node.record_id)
    delta = query[node.axis] - store[node.offset][node.axis]
    near, far = (left, right) if delta < 0 else (right, left)
    recurse(near)                              // always
    if heap.len() < k OR |delta| < heap.peek_worst():
        recurse(far)                           // pruning gate
```

Why this is correct: `|delta|` is the perpendicular distance from the query to the splitting hyperplane. Every point in the `far` subtree is on the opposite side of that hyperplane, so its distance to the query is **at least** `|delta|`. If the current worst result is closer than that, the `far` subtree cannot improve the answer and we skip it. The bound is `|delta|` (not `delta^2`) because the heap holds sqrt-Euclidean distances.

### Why Euclidean-only

Geometric pruning relies on the fact that the splitting hyperplane is perpendicular to one axis and that `|q[axis] - p[axis]|` is a lower bound on the L2 distance from `q` to any point on the other side. This bound does **not** hold for cosine similarity or dot product: those metrics measure angle / projection, and axis-aligned splits give no useful bound. Callers wanting cosine/dot semantics must use `BruteForce` (or normalize their vectors to unit length and treat L2 distance on unit vectors as a cosine proxy).

### Why immutable

The tree references `RecordStore` IDs and `VectorStore` offsets directly. Any mutation to those stores (insert, delete, swap-remove) can invalidate either the IDs that nodes point to or the offsets they read from. Rather than build a partial-update protocol that gets the staleness window wrong, the tree is treated as a build-from-scratch artifact: rebuild after a batch of writes settles.

There is therefore no `insert` or `delete` on `KdTree`.

## Filtering Semantics

When a non-empty `filter` is supplied, the tree:

1. Still descends as if doing **unfiltered** KNN — the geometry of the tree is unchanged.
2. Only pushes records that match the filter into the result heap.
3. Derives its pruning radius from the heap, which only contains accepted points.

This means a highly selective filter does not produce a pruning speedup — the tree still visits roughly the same set of nodes it would visit without the filter. This is a v1 limitation. The reason we do not "prune by filtered-out points" is correctness: if a filtered-out point sits closer than every accepted point, using its distance as the pruning radius would cut off subtrees that contain accepted matches. The result would be silently wrong (missing a true top-K result), which is unacceptable.

For highly selective filters, prefer `BruteForce::search(..., filter)` — it pays the full scan cost but does not also pay the tree-recursion overhead.

## When to use vs BruteForce

`should_use_kdtree(n, dim)` returns `true` when `n >= 1024 && dim <= 64`.

| Regime | Recommended |
|--------|-------------|
| n < 1024 | BruteForce — tree overhead exceeds scan cost |
| dim > 64 | BruteForce — curse of dimensionality kills axis-aligned pruning |
| filter is highly selective | BruteForce — tree pruning radius does not benefit |
| Cosine / DotProduct metric | BruteForce — pruning is geometrically invalid |
| Otherwise (large low-dim Euclidean corpus) | KdTree |

The 1024 break-even is a rough number from typical literature; in our own benches at dim=8, K=10 the tree already wins by ~3.6x at N=1k and ~26x at N=10k (see "Benchmarks" below). The 64-dim cap is conservative: empirical surveys put the practical ceiling somewhere between 20 and 50 dims depending on the data distribution, after which the fraction of subtrees that can be pruned drops toward zero.

For v1 the MCP layer does **not** auto-select between tree and brute-force — `KdTree` is exposed as a library type for callers that know they want it. Auto-selection is a candidate for Phase 5.5.

## Heap Extension

KD-Tree pruning needs `O(1)` access to the current worst (largest) score in the result heap. `BoundedMaxHeap` gained a small companion method to support this:

```rust
impl BoundedMaxHeap {
    pub fn peek_worst(&self) -> Option<f32>;
}
```

It returns the largest score currently in the heap, or `None` if the heap is empty. Implementation is a one-line `BinaryHeap::peek` map.

## Error Handling

`KdTree` itself does not produce errors — `build` always succeeds (an empty store yields an empty tree), and `search` returns an empty `Vec` for the defensive cases (`k == 0`, empty tree, dimension mismatch). It does not need a dedicated error enum because there is no operation whose failure mode is interesting to the caller.

## Performance Characteristics

| Operation | Complexity |
|-----------|-----------|
| `build` | `O(n log n)` time, `O(n)` space |
| `search` (low dim) | `O(log n)` average, `O(n)` worst-case |
| `search` (high dim) | tends toward `O(n)` |
| `peek_worst` | `O(1)` |

## Benchmarks

Captured on Apple Silicon, criterion `--quick` mode. Numbers are median time per `search` call (10k corpus search reads back top-K=10).

| Configuration | Brute (µs) | KdTree (µs) | Speedup |
|---|---|---|---|
| dim=8,  N=1k,  K=10 | 5.77 | 1.61 | 3.6x |
| dim=8,  N=10k, K=10 | 38.26 | 1.46 | 26x |
| dim=384, N=1k, K=10 | 40.40 | 16.63 | 2.4x (corpus has structure) |
| dim=384, N=10k, K=10 | 412.5 | 56.5 | 7.3x (corpus has structure) |

The dim=384 results are flattering to the tree because the synthetic bench corpus (`make_vec(i, dim) = sin(0.01337 * (31*i + j))`) is highly clustered along each axis. On real random unit-norm 384-dim data we expect the tree to perform **worse** than brute force in this regime — which is why `should_use_kdtree(_, 384)` returns `false`. The bench documents the effect; do not generalize from it.

Construction time at dim=8:

| N | KdTree::build (median) |
|---|---|
| 1,000 | 144 µs |
| 10,000 | 2.68 ms |

That puts the amortized search cost (build + many queries) firmly in favor of the tree for any workload with more than a handful of queries per corpus snapshot.

## Test Coverage

10 unit tests in `src/index/kdtree.rs`:

- `build_on_empty_store_returns_empty_tree`
- `build_on_one_vector_returns_single_node`
- `search_5_2d_vectors_top3_matches_expected`
- `search_empty_query_returns_empty`
- `search_k_zero_returns_empty`
- `filter_excludes_records_in_kdtree_search`
- `should_use_kdtree_heuristic`
- `kdtree_matches_brute_force_2d` (proptest, 64 cases)
- `kdtree_matches_brute_force_4d` (proptest, 64 cases)
- `kdtree_matches_brute_force_8d` (proptest, 64 cases)

The three proptests are the load-bearing correctness check: they generate random 2D/4D/8D corpora (1..=100 points), random queries, and random `k` in `1..=20`, then assert that `KdTree::search(...).iter().map(|r| r.id)` equals `BruteForce::search(..., Metric::Euclidean).iter().map(|r| r.id)`. Tie-breaking is determined by the shared `BoundedMaxHeap` ordering rule (score asc, id asc), so the comparison can be done by exact ID-list equality rather than score-set equality.

2 additional unit tests cover the new `BoundedMaxHeap::peek_worst` method in `src/heap/mod.rs`.

## Dependencies

- `crate::distance::euclidean` — sqrt-Euclidean SIMD-accelerated distance
- `crate::heap::BoundedMaxHeap` — top-K result accumulator (with new `peek_worst`)
- `crate::index::brute::SearchResult` — shared result type
- `crate::store::record::{RecordStore, VectorRecord}` — record lookup + filter helper
- `crate::store::VectorStore` — flat SoA vector data
