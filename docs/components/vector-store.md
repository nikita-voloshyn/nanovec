# VectorStore

## Purpose

VectorStore is the primary raw-vector storage layer. It holds all embedding vectors in a single flat `Vec<f32>` using a Structure of Arrays (SoA) layout, enabling contiguous cache-line access during distance computation scans. It knows nothing about text, IDs, or metadata -- those concerns belong to RecordStore. The dimension is fixed at construction and validated on every insert.

## Public API

```rust
// src/store/mod.rs

pub enum StoreError {
    DimensionMismatch { expected: usize, got: usize },
    InvalidOffset(usize),
}

pub struct VectorStore { /* private fields */ }

impl VectorStore {
    pub fn new(dimension: usize) -> Self;
    pub fn insert(&mut self, vector: &[f32]) -> Result<usize, StoreError>;
    pub fn get(&self, index: usize) -> Option<&[f32]>;
    pub fn swap_remove(&mut self, index: usize) -> Result<usize, StoreError>;
    pub fn count(&self) -> usize;
    pub fn dimension(&self) -> usize;
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f32])>;
}
```

### Method details

| Method | Description | Return |
|--------|-------------|--------|
| `new(dimension)` | Create an empty store locked to `dimension`-length vectors | `VectorStore` |
| `insert(vector)` | Append a vector; returns its 0-based logical index (offset) | `Ok(usize)` or `DimensionMismatch` |
| `get(index)` | Borrow the vector slice at logical index | `Some(&[f32])` or `None` |
| `swap_remove(index)` | O(1) delete by swapping with last element, then truncating | `Ok(index)` or `InvalidOffset` |
| `count()` | Number of stored vectors | `usize` |
| `dimension()` | Fixed dimension of each vector | `usize` |
| `iter()` | Enumerate all vectors as `(index, &[f32])` | Iterator |

## Internal Design

### Memory layout

Vectors are stored contiguously in a single `Vec<f32>`. For a store with `n` vectors of dimension `d`:

```
index:   [   0   |   1   |   2   |   ...  ]
floats:  [v0_d0 v0_d1 ... v0_dD | v1_d0 v1_d1 ... v1_dD | ...]
```

Accessing vector `i` is an O(1) slice: `&vectors[i*dim .. (i+1)*dim]`.

### Swap-remove for O(1) delete

Removing from the middle of a packed array without gaps requires moving data. The swap-remove strategy avoids an O(n) shift by:

1. Copying the last vector's floats into the removed slot (`copy_within`).
2. Truncating the backing `Vec<f32>` to drop the last (now-duplicate) slot.

When the removed element is already the last one, no copy is needed -- the method just truncates. The caller (RecordStore / brute::delete) is responsible for updating any offset references that pointed to the old last position.

### Dimension invariant

`insert` enforces `vector.len() == self.dimension` and returns `DimensionMismatch` on violation. The dimension cannot change after construction.

## Error Types

```rust
pub enum StoreError {
    /// Attempted insert with wrong number of floats.
    DimensionMismatch { expected: usize, got: usize },
    /// Passed an index >= count() to get() or swap_remove().
    InvalidOffset(usize),
}
```

Both variants implement `std::fmt::Display` and `std::error::Error`. No `Box<dyn Error>` is used.

## Usage Example

```rust
use nanovec::store::{VectorStore, StoreError};

let mut store = VectorStore::new(3);

let offset_a = store.insert(&[0.1, 0.2, 0.3]).unwrap(); // -> 0
let offset_b = store.insert(&[0.4, 0.5, 0.6]).unwrap(); // -> 1

assert_eq!(store.count(), 2);
assert_eq!(store.get(offset_a), Some(&[0.1f32, 0.2, 0.3][..]));

// Swap-remove index 0: index 1's vector moves to slot 0
store.swap_remove(offset_a).unwrap();
assert_eq!(store.count(), 1);
assert_eq!(store.get(0), Some(&[0.4f32, 0.5, 0.6][..]));

// Wrong dimension
let err = store.insert(&[1.0, 2.0]);
assert!(matches!(err, Err(StoreError::DimensionMismatch { expected: 3, got: 2 })));
```

## Performance Characteristics

| Operation | Time complexity |
|-----------|----------------|
| `insert` | O(1) amortized (Vec growth) |
| `get` | O(1) |
| `swap_remove` | O(d) -- one `copy_within` of `d` floats |
| `iter` (full scan) | O(n * d) |
| Space | O(n * d) f32 values |

The flat layout means a full linear scan for brute-force search accesses memory sequentially, which is cache-friendly. No benchmark numbers are available for Phase 1; SIMD acceleration is planned for Phase 2.

## Dependencies

- No other NanoVec modules. VectorStore is a leaf component.

## Test Coverage

7 unit tests in `src/store/mod.rs` under `#[cfg(test)]`:

- `insert_returns_correct_offset` -- offset sequence is 0, 1, 2...
- `get_after_insert_returns_same_vector` -- round-trip correctness
- `insert_wrong_dimension_errors` -- DimensionMismatch variant
- `count_grows_after_insert` -- count tracking
- `swap_remove_decreases_count` -- count after removal, moved vector at correct slot
- `swap_remove_last_element_no_swap` -- no-copy path when removing last element
- `swap_remove_invalid_offset_errors` -- InvalidOffset variant
- `iter_yields_all_vectors` -- full iteration order
