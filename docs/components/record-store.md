# RecordStore

## Purpose

RecordStore manages the metadata side of the database: text content, key-value metadata, and the mapping between logical IDs and the physical offsets used by VectorStore. It is deliberately separate from VectorStore so that the hot distance-computation path never touches string data.

## Public API

```rust
// src/store/record.rs

#[derive(Debug, Clone)]
pub struct VectorRecord {
    pub id: u64,
    pub offset: usize,
    pub text: String,
    pub metadata: Vec<(String, String)>,
}

pub struct RecordStore { /* private fields */ }

impl RecordStore {
    pub fn new() -> Self;
    pub fn insert(
        &mut self,
        text: String,
        metadata: Vec<(String, String)>,
        offset: usize,
    ) -> u64;
    pub fn get(&self, id: u64) -> Option<&VectorRecord>;
    pub fn remove(&mut self, id: u64) -> Option<VectorRecord>;
    pub fn update_offset(&mut self, id: u64, new_offset: usize);
    pub fn count(&self) -> usize;
    pub fn iter(&self) -> impl Iterator<Item = &VectorRecord>;
}
```

### VectorRecord fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `u64` | Stable, unique, monotonically-increasing identifier |
| `offset` | `usize` | Current logical index in VectorStore (may change after swap-remove) |
| `text` | `String` | Text content associated with the vector |
| `metadata` | `Vec<(String, String)>` | Arbitrary key-value pairs |

### Method details

| Method | Description | Return |
|--------|-------------|--------|
| `new()` | Empty store, next ID starts at 0 | `RecordStore` |
| `insert(text, metadata, offset)` | Create a record, assign next available ID | assigned `u64` ID |
| `get(id)` | Borrow record by ID | `Some(&VectorRecord)` or `None` |
| `remove(id)` | Swap-remove the record, returns it | `Some(VectorRecord)` or `None` |
| `update_offset(id, new_offset)` | Update the stored offset after VectorStore reorg | `()` |
| `count()` | Number of live records | `usize` |
| `iter()` | Iterate all records in insertion order (affected by swaps) | Iterator |

`RecordStore` implements `Default` (delegates to `new()`).

## Internal Design

### ID scheme

IDs are assigned from a `next_id: u64` counter that only ever increments. After a `remove`, the freed ID is never reused. IDs are therefore unique for the lifetime of the process.

### Internal data structures

```
records: Vec<VectorRecord>       -- dense storage, swap-removed on delete
id_to_idx: HashMap<u64, usize>   -- O(1) ID -> Vec index lookup
```

`get(id)` resolves in two steps: `id_to_idx` lookup (O(1)), then a direct Vec index (O(1)).

### Swap-remove for O(1) delete

`remove(id)` uses `Vec::swap_remove` to avoid O(n) shifting:

1. Look up `idx = id_to_idx[id]`, remove the entry.
2. Call `records.swap_remove(idx)` -- this moves the last element to `idx`.
3. If the removed record was not the last, update `id_to_idx[moved_record.id] = idx`.

The `offset` field inside each `VectorRecord` tracks where in VectorStore the corresponding float data lives. When VectorStore performs a swap-remove, the float data at the old-last position moves to the freed slot; the caller must then call `update_offset` to keep the two stores synchronized.

## Usage Example

```rust
use nanovec::store::record::RecordStore;

let mut records = RecordStore::new();

// Insert records alongside VectorStore offsets
let id0 = records.insert(
    "first document".to_string(),
    vec![("source".to_string(), "web".to_string())],
    0, // VectorStore offset
);
let id1 = records.insert("second document".to_string(), vec![], 1);

assert_eq!(records.count(), 2);
assert_eq!(records.get(id0).unwrap().text, "first document");
assert_eq!(records.get(id0).unwrap().offset, 0);

// IDs are always increasing
assert!(id1 > id0);

// Remove id0; id1's Vec index shifts to 0 internally
records.remove(id0);
assert!(records.get(id0).is_none());
assert_eq!(records.count(), 1);

// After VectorStore::swap_remove moves id1's data to offset 0
records.update_offset(id1, 0);
assert_eq!(records.get(id1).unwrap().offset, 0);
```

## Performance Characteristics

| Operation | Time complexity |
|-----------|----------------|
| `insert` | O(1) amortized |
| `get` | O(1) |
| `remove` | O(1) amortized |
| `update_offset` | O(1) |
| `iter` | O(n) |
| Space | O(n) records + O(n) HashMap entries |

## Dependencies

- No other NanoVec modules. RecordStore is a leaf component.

## Test Coverage

6 unit tests in `src/store/record.rs` under `#[cfg(test)]`:

- `insert_get_roundtrip` -- text, offset, and metadata preserved
- `remove_returns_none_on_get` -- record absent after removal
- `update_offset_changes_stored_offset` -- offset mutation verified
- `count_tracks_insertions_and_removals` -- count accuracy
- `iter_yields_all_records` -- all records reachable via iteration
- `ids_are_unique_and_incrementing` -- ID monotonicity, no reuse after remove
