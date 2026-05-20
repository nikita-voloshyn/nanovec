//! Phase 6 — the database layer.
//!
//! `NanoVecDatabase` replaces the single `Arc<Mutex<NanoVecState>>` used in
//! Phases 1–5 with a two-level locking scheme:
//!
//! - Outer `RwLock<HashMap<String, Arc<CollectionGuard>>>` — taken in **read**
//!   for the typical search/insert path (collection lookup) and in **write**
//!   only for `create_collection` / `drop_collection`.
//! - Inner `RwLock<Collection>` — per-collection. Search takes a reader; the
//!   `index_*` / `delete` / `clear` paths take a writer. Two clients indexing
//!   into **different** collections never contend.
//!
//! Additionally tracks per-collection memory usage + LRU tick for the
//! eviction policy (env: `NANOVEC_MEMORY_LIMIT_MB`).
//!
//! Lock discipline (must hold):
//!   1. Never hold the outer write lock while taking an inner lock.
//!   2. When acquiring multiple inner collection locks, do so in lexicographic
//!      name order. Today no handler needs this; the rule is here for the
//!      future.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::distance::Metric;
use crate::embed::Embedder;
use crate::store::collections::{Collection, CollectionError, DEFAULT_COLLECTION};

/// Default memory budget when `NANOVEC_MEMORY_LIMIT_MB` is unset (512 MiB).
pub const DEFAULT_MEMORY_LIMIT_MB: u64 = 512;

/// Wrapper that bundles a `Collection` (under an inner RwLock) with an
/// access-tick counter for LRU eviction. The tick is updated by callers
/// (typically the MCP layer) after every successful read or write.
pub struct CollectionGuard {
    pub inner: RwLock<Collection>,
    last_accessed: AtomicU64,
    /// `true` for `default` and any collection whose name matches the
    /// `_conn_*` connection-scoped pattern. Pinned collections are never
    /// chosen by the LRU eviction policy.
    pinned: bool,
}

impl CollectionGuard {
    pub fn new(collection: Collection, pinned: bool, initial_tick: u64) -> Self {
        Self {
            inner: RwLock::new(collection),
            last_accessed: AtomicU64::new(initial_tick),
            pinned,
        }
    }

    /// Bump the LRU tick. Called by handlers after touching the collection.
    pub fn touch(&self, tick: u64) {
        self.last_accessed.store(tick, Ordering::Relaxed);
    }

    pub fn last_accessed_tick(&self) -> u64 {
        self.last_accessed.load(Ordering::Relaxed)
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Approximate byte size of the collection's contents (vectors + record
    /// text). HashMap / Vec overhead is intentionally ignored — D7 in the
    /// Phase 6 plan: accuracy is ±10–20%, sufficient for a soft budget.
    pub fn approx_bytes(&self) -> u64 {
        let c = self.inner.read();
        let vector_bytes = (c.dimension() * 4 * c.count()) as u64;
        let text_bytes: u64 = c.records.iter().map(|r| r.text.len() as u64).sum();
        vector_bytes + text_bytes
    }
}

/// A collection is "pinned" (never evicted) iff its name is `default` or it
/// matches the `_conn_*` connection-scoped pattern from Phase 6 T4.
fn is_pinned_name(name: &str) -> bool {
    name == DEFAULT_COLLECTION || name.starts_with("_conn_")
}

/// Errors returned by [`NanoVecDatabase`] operations.
#[derive(Debug)]
pub enum DbError {
    /// A named collection was requested but does not exist (and cannot be
    /// auto-created — that path is reserved for `default` and `_conn_*`).
    CollectionNotFound(String),
    /// Memory budget exceeded and the collection cannot grow (pinned, or
    /// `evict_on_overflow=false`).
    BudgetExceeded { used: u64, limit: u64 },
    /// Underlying collection-map error (e.g. duplicate name).
    Collection(CollectionError),
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::CollectionNotFound(name) => write!(f, "collection not found: {name}"),
            DbError::BudgetExceeded { used, limit } => {
                write!(f, "memory budget exceeded: used {used} B, limit {limit} B")
            }
            DbError::Collection(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DbError {}

impl From<CollectionError> for DbError {
    fn from(e: CollectionError) -> Self {
        DbError::Collection(e)
    }
}

/// The database: a multi-tenant container of collections behind a two-level
/// RwLock, plus a global memory budget and a monotonic tick counter used by
/// the LRU eviction policy.
pub struct NanoVecDatabase {
    /// Outer map. Write-locked only for create/drop; read-locked for lookup.
    collections: RwLock<HashMap<String, Arc<CollectionGuard>>>,
    /// Monotonic access counter for LRU eviction. Incremented on every
    /// successful read/write touch via `next_tick()`.
    tick_counter: AtomicU64,
    /// Cumulative approx-bytes tracker. Refreshed lazily; precise value is
    /// computed by `recompute_used_bytes()`.
    used_bytes: AtomicU64,
    /// Hard memory cap. From `NANOVEC_MEMORY_LIMIT_MB` env, default 512 MiB.
    limit_bytes: u64,
    /// Vector dimension for auto-created `default` / `_conn_*` collections.
    /// Mirrors the embedder dimension when one is loaded; in raw-vector tests
    /// the embedder may be absent and the dim is set explicitly.
    auto_dim: usize,
    /// The default distance metric. Used by tools that do not specify one.
    pub metric: Metric,
    /// The embedder, if loaded. `None` for tests that exercise only the
    /// raw-vector path. `embed()` callers must `expect()` this Option.
    pub embedder: Option<Arc<Embedder>>,
}

impl NanoVecDatabase {
    /// Construct a database with the limit drawn from `NANOVEC_MEMORY_LIMIT_MB`
    /// (default: 512 MiB). The embedder must already be loaded.
    pub fn new(metric: Metric, embedder: Arc<Embedder>) -> Self {
        let limit_mb = std::env::var("NANOVEC_MEMORY_LIMIT_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MEMORY_LIMIT_MB);
        Self::with_limit_mb(metric, embedder, limit_mb)
    }

    /// Explicit-limit constructor for tests that still want a real embedder.
    pub fn with_limit_mb(metric: Metric, embedder: Arc<Embedder>, limit_mb: u64) -> Self {
        let auto_dim = embedder.dimension();
        Self {
            collections: RwLock::new(HashMap::new()),
            tick_counter: AtomicU64::new(0),
            used_bytes: AtomicU64::new(0),
            limit_bytes: limit_mb.saturating_mul(1024 * 1024),
            auto_dim,
            metric,
            embedder: Some(embedder),
        }
    }

    /// Raw-vector-only constructor: skip the embedder entirely. The auto-dim
    /// for `default` / `_conn_*` is set from the caller. Calls to
    /// `index_document` / `search_document` will panic because `embedder` is
    /// `None`; raw-vector paths and concurrency tests work unchanged.
    pub fn with_dimension(metric: Metric, dim: usize, limit_mb: u64) -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            tick_counter: AtomicU64::new(0),
            used_bytes: AtomicU64::new(0),
            limit_bytes: limit_mb.saturating_mul(1024 * 1024),
            auto_dim: dim,
            metric,
            embedder: None,
        }
    }

    /// Dimension used when auto-creating `default` / `_conn_*` collections.
    pub fn auto_dim(&self) -> usize {
        self.auto_dim
    }

    /// Total number of collections (including pinned ones). O(1).
    pub fn collections_count(&self) -> usize {
        self.collections.read().len()
    }

    /// Allocate the next LRU tick.
    pub fn next_tick(&self) -> u64 {
        self.tick_counter.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Current limit in bytes (config snapshot — never changes after start).
    pub fn limit_bytes(&self) -> u64 {
        self.limit_bytes
    }

    /// Last computed approx-bytes used. Call `recompute_used_bytes()` to
    /// refresh the cached value to the actual sum.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    /// Walk every collection and sum approx-bytes. Linear in collection count;
    /// callers should call this sparingly (after writes, before reporting via
    /// `stats`).
    pub fn recompute_used_bytes(&self) -> u64 {
        let map = self.collections.read();
        let total: u64 = map.values().map(|g| g.approx_bytes()).sum();
        self.used_bytes.store(total, Ordering::Relaxed);
        total
    }

    /// Look up a collection by name.
    /// - `None` or `Some("default")` -> the default collection (auto-created
    ///   on first access using the embedder dimension; pinned).
    /// - `Some(name)` matching `_conn_*` -> auto-created (pinned).
    /// - `Some(other)` -> must already exist via `create_collection`.
    pub fn resolve(&self, name: Option<&str>) -> Result<Arc<CollectionGuard>, DbError> {
        let resolved_name = name.unwrap_or(DEFAULT_COLLECTION).to_string();

        // Fast path: collection already exists. Outer read lock only.
        {
            let map = self.collections.read();
            if let Some(guard) = map.get(&resolved_name) {
                let tick = self.next_tick();
                guard.touch(tick);
                return Ok(Arc::clone(guard));
            }
        }

        // Auto-create path (default + connection scoped only). Take outer write.
        if is_pinned_name(&resolved_name) {
            let mut map = self.collections.write();
            // Re-check under write — another writer might have just created it.
            if let Some(guard) = map.get(&resolved_name) {
                let tick = self.next_tick();
                guard.touch(tick);
                return Ok(Arc::clone(guard));
            }
            let dim = self.auto_dim;
            let tick = self.next_tick();
            let guard = Arc::new(CollectionGuard::new(
                Collection::new_with_dim(resolved_name.clone(), dim),
                true,
                tick,
            ));
            map.insert(resolved_name, Arc::clone(&guard));
            return Ok(guard);
        }

        Err(DbError::CollectionNotFound(resolved_name))
    }

    /// Read-only lookup. Unlike `resolve`, this never auto-creates: missing
    /// `default` returns `Ok(None)` so search tools can short-circuit with an
    /// empty result. Missing named collection returns `Err`.
    pub fn resolve_readonly(
        &self,
        name: Option<&str>,
    ) -> Result<Option<Arc<CollectionGuard>>, DbError> {
        let resolved_name = name.unwrap_or(DEFAULT_COLLECTION).to_string();
        let map = self.collections.read();
        if let Some(guard) = map.get(&resolved_name) {
            let tick = self.next_tick();
            guard.touch(tick);
            return Ok(Some(Arc::clone(guard)));
        }
        if is_pinned_name(&resolved_name) {
            Ok(None)
        } else {
            Err(DbError::CollectionNotFound(resolved_name))
        }
    }

    /// Create a new (unpinned, eviction-eligible) collection.
    pub fn create_collection(&self, name: String, dimension: usize) -> Result<(), DbError> {
        let mut map = self.collections.write();
        if map.contains_key(&name) {
            return Err(DbError::Collection(CollectionError::AlreadyExists(name)));
        }
        let pinned = is_pinned_name(&name);
        let tick = self.next_tick();
        let guard = Arc::new(CollectionGuard::new(
            Collection::new_with_dim(name.clone(), dimension),
            pinned,
            tick,
        ));
        map.insert(name, guard);
        Ok(())
    }

    /// Drop a collection (any name — including `default`, which can later be
    /// re-materialized on next legacy access).
    pub fn drop_collection(&self, name: &str) -> Result<(), DbError> {
        let mut map = self.collections.write();
        if map.remove(name).is_none() {
            return Err(DbError::Collection(CollectionError::NotFound(
                name.to_string(),
            )));
        }
        Ok(())
    }

    /// Iterate names + descriptors of every collection. Order is sorted by
    /// name to keep `list_collections` / `stats` stable.
    pub fn snapshot(&self) -> Vec<CollectionSnapshot> {
        let map = self.collections.read();
        let mut names: Vec<&String> = map.keys().collect();
        names.sort_unstable();
        names
            .into_iter()
            .map(|name| {
                let guard = &map[name];
                let inner = guard.inner.read();
                CollectionSnapshot {
                    name: inner.name.clone(),
                    count: inner.count(),
                    dimension: inner.dimension(),
                    approx_bytes: {
                        let vector_bytes = (inner.dimension() * 4 * inner.count()) as u64;
                        let text_bytes: u64 =
                            inner.records.iter().map(|r| r.text.len() as u64).sum();
                        vector_bytes + text_bytes
                    },
                    last_accessed_tick: guard.last_accessed_tick(),
                    pinned: guard.is_pinned(),
                }
            })
            .collect()
    }

    /// LRU eviction. Drops the least-recently-accessed **unpinned** collection
    /// until `used_bytes < limit_bytes`. Returns the names of evicted
    /// collections (for logging by the caller).
    pub fn evict_until_under_budget(&self) -> Vec<String> {
        let mut evicted = Vec::new();
        loop {
            // Refresh used-bytes snapshot.
            let used = self.recompute_used_bytes();
            if used <= self.limit_bytes {
                break;
            }

            // Pick the unpinned collection with the smallest last_accessed_tick.
            let victim = {
                let map = self.collections.read();
                map.iter()
                    .filter(|(_, g)| !g.is_pinned())
                    .min_by_key(|(_, g)| g.last_accessed_tick())
                    .map(|(name, _)| name.clone())
            };

            match victim {
                Some(name) => {
                    let mut map = self.collections.write();
                    map.remove(&name);
                    evicted.push(name);
                }
                None => break, // Nothing left to evict (only pinned collections remain).
            }
        }
        evicted
    }
}

/// Descriptor used by `list_collections` / `stats` / `memory`.
#[derive(Debug, Clone)]
pub struct CollectionSnapshot {
    pub name: String,
    pub count: usize,
    pub dimension: usize,
    pub approx_bytes: u64,
    pub last_accessed_tick: u64,
    pub pinned: bool,
}

// `Collection::new` was private to `collections.rs` before Phase 6; expose a
// pub-crate constructor so the database can build collections directly.
impl Collection {
    pub(crate) fn new_with_dim(name: String, dimension: usize) -> Self {
        // Re-use the existing private path. Implemented here via the public
        // surface (`VectorStore::new`, `RecordStore::new`) to avoid touching
        // visibility on the original `Collection::new`.
        use crate::store::record::RecordStore;
        use crate::store::VectorStore;
        Collection {
            name,
            store: VectorStore::new(dimension),
            records: RecordStore::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Lightweight unit-tests of pure helpers. Behavioral tests that require an
    // Embedder live in tests/concurrency.rs.
    #[test]
    fn smoke_pinned_detection() {
        assert!(is_pinned_name("default"));
        assert!(is_pinned_name("_conn_alice"));
        assert!(!is_pinned_name("docs"));
        assert!(!is_pinned_name("anything-else"));
    }
}
