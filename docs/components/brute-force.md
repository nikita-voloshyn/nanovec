# BoundedMaxHeap and BruteForce Search

## Purpose

This document covers two closely related components:

- **BoundedMaxHeap** (`src/heap/mod.rs`) -- a bounded priority queue that efficiently tracks the K best (lowest-score) results seen so far during a scan.
- **BruteForce** (`src/index/brute.rs`) -- the Phase 1 search index that performs a full linear scan over VectorStore, using BoundedMaxHeap to accumulate top-K results, and a coordinated `delete` function that keeps VectorStore and RecordStore in sync.

## BoundedMaxHeap

### Purpose

Keep the K items with the smallest scores out of an arbitrarily large stream of `(score, id)` pairs. This is the "min-K" selection problem, solved by maintaining a bounded max-heap: the largest score sits at the top and can be evicted in O(log K) when a better (smaller) score arrives.

### Public API

```rust
// src/heap/mod.rs

pub struct BoundedMaxHeap { /* private */ }

impl BoundedMaxHeap {
    pub fn new(capacity: usize) -> Self;
    pub fn push(&mut self, score: f32, id: u64);
    pub fn into_sorted_vec(self) -> Vec<(f32, u64)>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

### Algorithm

Internally, `BoundedMaxHeap` wraps `std::collections::BinaryHeap<OrderedItem>` (which is a max-heap by default). `OrderedItem` orders items by `score` descending, then by `id` ascending as a tiebreaker.

Push logic:

```
if heap.len() < capacity:
    heap.push(item)
else if score < heap.peek().score:
    heap.pop()           // evict current worst
    heap.push(item)      // insert new better item
else:
    discard              // item is worse than current worst
```

`into_sorted_vec` drains all items and sorts them ascending by score (best-first), so callers receive results ordered from closest to furthest.

A capacity of 0 silently discards all pushes and returns an empty vec.

### Usage Example

```rust
use nanovec::heap::BoundedMaxHeap;

let mut heap = BoundedMaxHeap::new(3);
heap.push(10.0, 10);
heap.push(5.0, 5);
heap.push(1.0, 1);
heap.push(3.0, 3);
heap.push(0.5, 0);

let results = heap.into_sorted_vec();
// [(0.5, 0), (1.0, 1), (3.0, 3)]  -- top 3 smallest scores
```

### Performance Characteristics

| Operation | Time complexity |
|-----------|----------------|
| `push` | O(log K) |
| `into_sorted_vec` | O(K log K) |
| Space | O(K) |

## BruteForce Search

### Purpose

Perform exact K-nearest-neighbor (KNN) search by scanning every vector in VectorStore. This is the correct and complete Phase 1 index. Phase 2 will add a KD-Tree for sub-linear search; BruteForce will remain as the reference implementation and fallback.

### Public API

```rust
// src/index/brute.rs

pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub text: String,
}

#[derive(Debug)]
pub enum DeleteError {
    NotFound(u64),
}

pub struct BruteForce;

impl BruteForce {
    pub fn search(
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        metric: Metric,
    ) -> Vec<SearchResult>;
}

pub fn delete(
    store: &mut VectorStore,
    records: &mut RecordStore,
    id: u64,
) -> Result<(), DeleteError>;
```

### Search Algorithm

```
1. If k == 0 or store is empty, return [].
2. Obtain a distance function for the chosen metric.
3. Create BoundedMaxHeap(k).
4. For each record in RecordStore:
     a. Fetch its vector from VectorStore using record.offset.
     b. Compute distance(query, vector).
     c. Push (score, record.id) onto the heap.
5. Drain the heap (ascending score order).
6. Map each (score, id) -> SearchResult by looking up text in RecordStore.
7. Return Vec<SearchResult>.
```

`SearchResult` contains `id`, `score`, and `text`. Metadata is not included in search results in Phase 1.

### Complexity

O(n * d + n * log(k)) where:
- n = number of stored vectors
- d = vector dimension
- k = number of results requested

The n*d term dominates for typical d >> log(k). No approximation is made; all n vectors are scored.

### Delete: Coordinated Swap-Remove

Deleting a record requires both stores to remain consistent. The `delete` function orchestrates the following steps:

```
1. Look up record by id in RecordStore; get its offset.
2. Call VectorStore::swap_remove(offset).
   - Moves the last vector into `offset`.
   - The old last vector was at index (old_count - 1) == store.count() after removal.
3. Call RecordStore::remove(id) to drop the record.
4. If offset != old_last_index (a swap occurred):
   - Find the record whose .offset == old_last_index (linear scan of RecordStore).
   - Call RecordStore::update_offset(moved_id, offset) to fix it.
5. Return Ok(()) or DeleteError::NotFound(id).
```

Step 4 involves a linear scan of RecordStore (O(n)) to find the moved record. This is acceptable for Phase 1. Phase 2 may add a reverse mapping (offset -> id) to bring this to O(1).

### Error Type

```rust
pub enum DeleteError {
    NotFound(u64),  // No record with the given ID exists
}
```

Implements `Display` and `std::error::Error`. No `Box<dyn Error>` used.

### Usage Example

```rust
use nanovec::store::VectorStore;
use nanovec::store::record::RecordStore;
use nanovec::distance::Metric;
use nanovec::index::brute::{BruteForce, delete};

let mut vs = VectorStore::new(2);
let mut rs = RecordStore::new();

let off = vs.insert(&[1.0, 0.0]).unwrap();
let id  = rs.insert("hello".to_string(), vec![], off);

let results = BruteForce::search(&vs, &rs, &[0.0, 0.0], 1, Metric::Euclidean);
assert_eq!(results[0].text, "hello");
assert!((results[0].score - 1.0).abs() < 1e-6);

delete(&mut vs, &mut rs, id).unwrap();
assert_eq!(vs.count(), 0);
```

## Phase 2 Preview

Phase 2 will introduce a KD-Tree index (`src/index/kdtree.rs`) for O(log n) average-case search. BruteForce will remain as:

- The reference implementation for correctness testing.
- The automatic fallback for low-cardinality stores where tree overhead exceeds scan cost.

## Dependencies

- `crate::distance` -- `Metric` enum and `distance_fn`
- `crate::heap` -- `BoundedMaxHeap`
- `crate::store` -- `VectorStore`
- `crate::store::record` -- `RecordStore`

## Test Coverage

5 unit tests in `src/heap/mod.rs`:

- `push_more_than_capacity_keeps_k` -- heap never exceeds capacity
- `into_sorted_vec_returns_ascending_order` -- result ordering
- `keeps_top_3_smallest_scores` -- correct items selected
- `push_same_score_keeps_k_items` -- tie handling
- `capacity_zero_returns_empty` -- edge case

6 unit tests in `src/index/brute.rs`:

- `search_5_vectors_k3_returns_3_closest` -- top-3 correctness with Euclidean
- `search_empty_store_returns_empty` -- empty store edge case
- `search_k_greater_than_count_returns_all` -- k > n returns all vectors
- `delete_existing_id_decreases_count` -- count and searchability after delete
- `delete_nonexistent_id_returns_not_found` -- DeleteError::NotFound
- `delete_first_vector_swap_remove_remaining_searchable` -- offset fixup after swap
