//! Multi-namespace storage: a flat map of named [`Collection`]s, each holding
//! its own [`VectorStore`] + [`RecordStore`] and dimension lock.
//!
//! Per the Phase 4 contract, collections are independent — the dimension lock
//! belongs to each collection, not to the server. Existing tools that do not
//! pass a `collection` parameter operate on a lazily-created `"default"`
//! collection (see `src/mcp/mod.rs`).

use std::collections::HashMap;

use super::record::RecordStore;
use super::VectorStore;

/// Reserved collection name used by legacy callers that omit the `collection`
/// parameter. The MCP layer lazily creates this collection on first access.
pub const DEFAULT_COLLECTION: &str = "default";

/// A single named namespace: store + records + the dimension shared by both.
///
/// `dimension` is duplicated from `store.dimension()` for ergonomic read
/// access without borrowing `store`. The two are kept in sync because the
/// store is dimension-locked at construction.
pub struct Collection {
    pub name: String,
    pub store: VectorStore,
    pub records: RecordStore,
}

impl Collection {
    fn new(name: String, dimension: usize) -> Self {
        Self {
            name,
            store: VectorStore::new(dimension),
            records: RecordStore::new(),
        }
    }

    /// Dimension this collection is locked to.
    pub fn dimension(&self) -> usize {
        self.store.dimension()
    }

    /// Number of records in this collection.
    pub fn count(&self) -> usize {
        self.records.count()
    }
}

/// Errors returned by [`CollectionMap`] operations.
#[derive(Debug)]
pub enum CollectionError {
    /// A collection with this name already exists.
    AlreadyExists(String),
    /// No collection with this name was found.
    NotFound(String),
}

impl std::fmt::Display for CollectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollectionError::AlreadyExists(name) => {
                write!(f, "collection already exists: {name}")
            }
            CollectionError::NotFound(name) => write!(f, "collection not found: {name}"),
        }
    }
}

impl std::error::Error for CollectionError {}

/// Map of named collections. All multi-namespace state lives here so the MCP
/// layer can keep its single `Arc<Mutex<...>>` locking discipline (per the
/// YAGNI rule: do not split into per-collection locks until proven necessary).
pub struct CollectionMap {
    collections: HashMap<String, Collection>,
}

impl CollectionMap {
    /// Empty map. The MCP layer is responsible for lazily creating the
    /// `"default"` collection on first legacy access.
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }

    /// Create a new collection with the given dimension. Errors if a
    /// collection with the same name already exists.
    pub fn create(&mut self, name: String, dimension: usize) -> Result<(), CollectionError> {
        if self.collections.contains_key(&name) {
            return Err(CollectionError::AlreadyExists(name));
        }
        self.collections
            .insert(name.clone(), Collection::new(name, dimension));
        Ok(())
    }

    /// Immutable lookup by name.
    pub fn get(&self, name: &str) -> Option<&Collection> {
        self.collections.get(name)
    }

    /// Mutable lookup by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Collection> {
        self.collections.get_mut(name)
    }

    /// Names of all collections, sorted lexicographically. Stable output is
    /// useful for `list_collections` so clients can rely on ordering without
    /// having to sort themselves.
    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.collections.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Drop a collection by name. Errors if it does not exist.
    pub fn drop(&mut self, name: &str) -> Result<(), CollectionError> {
        if self.collections.remove(name).is_none() {
            return Err(CollectionError::NotFound(name.to_string()));
        }
        Ok(())
    }

    /// Total number of collections.
    pub fn count(&self) -> usize {
        self.collections.len()
    }

    /// Iterate over all collections in sorted-by-name order. Mirrors `list()`
    /// so `list_collections` and `stats` produce consistent ordering.
    pub fn iter(&self) -> impl Iterator<Item = &Collection> {
        let mut names: Vec<&String> = self.collections.keys().collect();
        names.sort_unstable();
        names.into_iter().map(move |n| &self.collections[n])
    }

    /// Ensure a collection with the given name exists. If absent, it is
    /// created with the supplied default dimension. Returns a mutable
    /// reference to the (now-extant) collection.
    ///
    /// This is the bridge used by legacy MCP calls that omit `collection`:
    /// the `"default"` collection is materialized on first access using the
    /// embedder dimension as the default lock.
    pub fn get_or_create(&mut self, name: &str, default_dimension: usize) -> &mut Collection {
        if !self.collections.contains_key(name) {
            self.collections.insert(
                name.to_string(),
                Collection::new(name.to_string(), default_dimension),
            );
        }
        self.collections.get_mut(name).expect("just inserted")
    }
}

impl Default for CollectionMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_then_get_returns_collection() {
        let mut map = CollectionMap::new();
        map.create("docs".to_string(), 8).unwrap();
        let c = map.get("docs").expect("collection should exist");
        assert_eq!(c.name, "docs");
        assert_eq!(c.dimension(), 8);
        assert_eq!(c.count(), 0);
    }

    #[test]
    fn create_duplicate_errors() {
        let mut map = CollectionMap::new();
        map.create("docs".to_string(), 8).unwrap();
        let err = map.create("docs".to_string(), 8).unwrap_err();
        match err {
            CollectionError::AlreadyExists(name) => assert_eq!(name, "docs"),
            other => panic!("expected AlreadyExists, got {other}"),
        }
    }

    #[test]
    fn drop_missing_errors() {
        let mut map = CollectionMap::new();
        let err = map.drop("missing").unwrap_err();
        match err {
            CollectionError::NotFound(name) => assert_eq!(name, "missing"),
            other => panic!("expected NotFound, got {other}"),
        }
    }

    #[test]
    fn drop_removes_collection() {
        let mut map = CollectionMap::new();
        map.create("docs".to_string(), 4).unwrap();
        assert_eq!(map.count(), 1);
        map.drop("docs").unwrap();
        assert_eq!(map.count(), 0);
        assert!(map.get("docs").is_none());
    }

    #[test]
    fn list_returns_sorted_names() {
        let mut map = CollectionMap::new();
        map.create("zeta".to_string(), 2).unwrap();
        map.create("alpha".to_string(), 2).unwrap();
        map.create("mu".to_string(), 2).unwrap();
        assert_eq!(map.list(), vec!["alpha", "mu", "zeta"]);
    }

    #[test]
    fn get_or_create_creates_when_missing() {
        let mut map = CollectionMap::new();
        let c = map.get_or_create("default", 16);
        assert_eq!(c.name, "default");
        assert_eq!(c.dimension(), 16);
        assert_eq!(map.count(), 1);
    }

    #[test]
    fn get_or_create_returns_existing_without_redim() {
        let mut map = CollectionMap::new();
        map.create("docs".to_string(), 8).unwrap();
        // The default_dimension is ignored when the collection already exists.
        let c = map.get_or_create("docs", 999);
        assert_eq!(c.dimension(), 8);
    }

    #[test]
    fn count_tracks_create_and_drop() {
        let mut map = CollectionMap::new();
        assert_eq!(map.count(), 0);
        map.create("a".to_string(), 2).unwrap();
        map.create("b".to_string(), 2).unwrap();
        assert_eq!(map.count(), 2);
        map.drop("a").unwrap();
        assert_eq!(map.count(), 1);
    }

    #[test]
    fn get_mut_allows_record_insert() {
        let mut map = CollectionMap::new();
        map.create("docs".to_string(), 2).unwrap();
        let c = map.get_mut("docs").unwrap();
        let offset = c.store.insert(&[1.0, 2.0]).unwrap();
        c.records.insert("hi".to_string(), vec![], offset);
        assert_eq!(c.count(), 1);
    }
}
