# NanoVecDatabase (Phase 6)

## Purpose

`NanoVecDatabase` (`src/store/database.rs`) is the multi-tenant container that
holds all collections behind a two-level `parking_lot::RwLock` scheme, plus a
soft memory budget with LRU eviction. It replaced the Phase 1–5
`Arc<Mutex<NanoVecState>>` so that concurrent search and per-collection writes
no longer serialize through one global lock.

Three responsibilities:
1. **Concurrency.** Outer `RwLock<HashMap<name, Arc<CollectionGuard>>>` for
   collection lookup; inner `RwLock<Collection>` per collection for the
   actual store/records access.
2. **Budget.** Approximate `used_bytes` tracker, hard `limit_bytes` cap from
   `NANOVEC_MEMORY_LIMIT_MB` (default 512 MiB).
3. **Eviction.** Per-collection `last_accessed` tick; `evict_until_under_budget`
   drops the coldest **unpinned** collection until usage fits.

## Public API

```rust
// src/store/database.rs

pub const DEFAULT_MEMORY_LIMIT_MB: u64 = 512;

pub struct NanoVecDatabase { /* private */ }

impl NanoVecDatabase {
    // Constructors
    pub fn new(metric: Metric, embedder: Arc<Embedder>) -> Self;
    pub fn with_limit_mb(metric: Metric, embedder: Arc<Embedder>, limit_mb: u64) -> Self;
    pub fn with_dimension(metric: Metric, dim: usize, limit_mb: u64) -> Self; // tests

    // Collection lookup
    pub fn resolve(&self, name: Option<&str>) -> Result<Arc<CollectionGuard>, DbError>;
    pub fn resolve_readonly(&self, name: Option<&str>) -> Result<Option<Arc<CollectionGuard>>, DbError>;
    pub fn create_collection(&self, name: String, dimension: usize) -> Result<(), DbError>;
    pub fn drop_collection(&self, name: &str) -> Result<(), DbError>;
    pub fn snapshot(&self) -> Vec<CollectionSnapshot>;

    // Budget / eviction
    pub fn limit_bytes(&self) -> u64;
    pub fn used_bytes(&self) -> u64;
    pub fn recompute_used_bytes(&self) -> u64;
    pub fn evict_until_under_budget(&self) -> Vec<String>;

    // Metadata
    pub fn auto_dim(&self) -> usize;
    pub fn next_tick(&self) -> u64;
    pub fn collections_count(&self) -> usize;
}

pub struct CollectionGuard {
    pub inner: parking_lot::RwLock<Collection>,
    // last_accessed, pinned hidden behind methods
}

impl CollectionGuard {
    pub fn touch(&self, tick: u64);
    pub fn last_accessed_tick(&self) -> u64;
    pub fn is_pinned(&self) -> bool;
    pub fn approx_bytes(&self) -> u64;
}

pub struct CollectionSnapshot {
    pub name: String,
    pub count: usize,
    pub dimension: usize,
    pub approx_bytes: u64,
    pub last_accessed_tick: u64,
    pub pinned: bool,
}

pub enum DbError {
    CollectionNotFound(String),
    BudgetExceeded { used: u64, limit: u64 },
    Collection(CollectionError),
}
```

## Internal Design

### Lock topology

```
NanoVecDatabase
└── collections: RwLock<HashMap<String, Arc<CollectionGuard>>>   ← outer
                                          │
                                          └── inner: RwLock<Collection>  ← inner
```

- **Outer lock.** Write-locked only for `create_collection` / `drop_collection`
  / eviction (rare). Read-locked for the common path: collection lookup. The
  `Arc<CollectionGuard>` clone makes the lookup `O(1)` without holding the
  outer lock for the duration of the work.
- **Inner lock.** Per-collection. Readers (search) share; writers
  (`index_*`, `delete`, `clear`) are exclusive but only block on the *same*
  collection. Two clients indexing into different collections never contend.

### Pinned collections

`is_pinned_name(&str) -> bool` returns true for `"default"` and any name
starting with `"_conn_"`. Pinned collections:
- Auto-create on first access (default + connection-scoped).
- Are excluded from `evict_until_under_budget`.

User-created collections (`create_collection`) are unpinned — they're the
eviction candidates when the budget is exceeded.

### LRU tick

`tick_counter: AtomicU64` is incremented on every successful `resolve` /
`resolve_readonly`. The current value is written into the
`CollectionGuard::last_accessed` atom. Eviction selects
`min last_accessed_tick` across unpinned collections.

The tick is approximate — it doesn't bump on every distance computation,
only on whole tool calls — which is the right granularity since LRU operates
at the collection level, not the record level.

### Memory accounting

`approx_bytes(&self) -> u64` per collection sums:
- `dimension * 4 * count` (vector bytes)
- Sum of `record.text.len()` (text bytes)

Ignores HashMap / Vec capacity overhead — D7 in the Phase 6 plan: accuracy
is ±10–20%, sufficient for a soft budget where the limit is
operator-chosen.

### Optional embedder

`embedder: Option<Arc<Embedder>>` instead of mandatory `Arc<Embedder>` so
the database can be constructed in raw-vector-only mode (e.g.
`with_dimension` for concurrency tests that don't want to wait for the BERT
weights to load). When `None`, the `index_document` / `search_document`
tools return `"embedder not loaded (raw-vector mode)"`.

## Error Types

| Variant | When |
|---------|------|
| `CollectionNotFound(name)` | `resolve` / `resolve_readonly` on a name that doesn't exist and isn't auto-creatable. |
| `BudgetExceeded { used, limit }` | Reserved for future use (today eviction handles overflow). |
| `Collection(CollectionError)` | Wrapper for `AlreadyExists` / `NotFound` from `CollectionMap`. |

## Usage Example

```rust
use std::sync::Arc;
use nanovec::distance::Metric;
use nanovec::store::database::NanoVecDatabase;

// Test setup — no embedder.
let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, 384, 256));

// Auto-create "default" via resolve(None).
let guard = db.resolve(None).unwrap();
{
    let mut coll = guard.inner.write();
    let off = coll.store.insert(&vec![0.5_f32; 384]).unwrap();
    coll.records.insert("hello".into(), vec![], off);
}

// Connection-scoped (auto-create, pinned).
let alice = db.resolve(Some("_conn_alice")).unwrap();
// ...

// User-created (evictable).
db.create_collection("docs".into(), 384).unwrap();

// Eviction.
let evicted = db.evict_until_under_budget();
```

## Performance

| Operation | Cost |
|-----------|------|
| `resolve` (existing) | One outer read-lock, hashmap lookup, atomic touch — `O(1)` |
| `resolve` (auto-create pinned) | Outer write-lock + insert |
| `create_collection` | Outer write-lock |
| `drop_collection` | Outer write-lock |
| `evict_until_under_budget` | `O(C)` per loop iter to find min tick (C = collection count) |

Measured concurrency (Apple M-series, 5k × 384-dim corpus, k=10):

| Scenario | 1 thread | 8 threads | Scaling |
|----------|---------:|----------:|--------:|
| Same collection (readers) | 1 922 q/s | 13 389 q/s | 6.97× |
| 8 collections, 8 threads | 11 648 q/s | 71 920 q/s | 6.17× |

Write isolation: with a 20 ms write-lock held on collection A, a reader on
collection B completed 148 searches.

## Test Coverage

| Test | Location | Verifies |
|------|----------|----------|
| `smoke_pinned_detection` | `src/store/database.rs` | `_conn_*` and `default` are pinned, others aren't |
| `parallel_search_scales_on_same_collection` | `tests/integration/concurrency.rs` | ≥ 2.0× scaling (actual 6.97×) |
| `parallel_search_scales_across_collections` | `tests/integration/concurrency.rs` | ≥ 3.0× scaling (actual 6.17×) |
| `writes_to_different_collections_are_independent` | `tests/integration/concurrency.rs` | B-reads progress during A-write |
| `lru_eviction_drops_coldest_unpinned_collection` | `tests/integration/concurrency.rs` | Coldest non-pinned collection evicted first |
| `pinned_default_collection_survives_eviction` | `tests/integration/concurrency.rs` | `default` never evicted |
| `budget_accounting_tracks_inserts` | `tests/integration/concurrency.rs` | `approx_bytes` within ±5% of theoretical |
