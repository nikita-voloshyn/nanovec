use std::collections::HashMap;

/// A single stored document with its vector offset and metadata.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    pub id: u64,
    pub offset: usize,
    pub text: String,
    pub metadata: Vec<(String, String)>,
}

/// Manages records and provides O(1) lookup by ID.
pub struct RecordStore {
    records: Vec<VectorRecord>,
    id_to_idx: HashMap<u64, usize>,
    next_id: u64,
}

impl RecordStore {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            id_to_idx: HashMap::new(),
            next_id: 0,
        }
    }

    /// Insert a new record. Returns the assigned ID.
    pub fn insert(&mut self, text: String, metadata: Vec<(String, String)>, offset: usize) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let idx = self.records.len();
        self.records.push(VectorRecord {
            id,
            offset,
            text,
            metadata,
        });
        self.id_to_idx.insert(id, idx);
        id
    }

    /// Look up a record by ID.
    pub fn get(&self, id: u64) -> Option<&VectorRecord> {
        self.id_to_idx.get(&id).map(|&idx| &self.records[idx])
    }

    /// Remove a record by ID. Returns the removed record, or None if not found.
    /// Uses swap-remove on the internal Vec to stay O(1); updates id_to_idx accordingly.
    pub fn remove(&mut self, id: u64) -> Option<VectorRecord> {
        let idx = self.id_to_idx.remove(&id)?;
        let record = self.records.swap_remove(idx);
        // If the removed record was not the last, the record that was at the end
        // has now moved to `idx`. Update its entry in id_to_idx.
        if idx < self.records.len() {
            let moved_id = self.records[idx].id;
            self.id_to_idx.insert(moved_id, idx);
        }
        Some(record)
    }

    /// Update the stored offset for a record (called after VectorStore::swap_remove moves a vector).
    pub fn update_offset(&mut self, id: u64, new_offset: usize) {
        if let Some(&idx) = self.id_to_idx.get(&id) {
            self.records[idx].offset = new_offset;
        }
    }

    pub fn count(&self) -> usize {
        self.records.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &VectorRecord> {
        self.records.iter()
    }
}

impl Default for RecordStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_roundtrip() {
        let mut store = RecordStore::new();
        let id = store.insert(
            "hello world".to_string(),
            vec![("key".to_string(), "value".to_string())],
            0,
        );
        let record = store.get(id).expect("record should exist");
        assert_eq!(record.id, id);
        assert_eq!(record.offset, 0);
        assert_eq!(record.text, "hello world");
        assert_eq!(
            record.metadata,
            vec![("key".to_string(), "value".to_string())]
        );
    }

    #[test]
    fn remove_returns_none_on_get() {
        let mut store = RecordStore::new();
        let id = store.insert("text".to_string(), vec![], 0);
        let removed = store.remove(id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, id);
        assert!(store.get(id).is_none());
    }

    #[test]
    fn update_offset_changes_stored_offset() {
        let mut store = RecordStore::new();
        let id = store.insert("text".to_string(), vec![], 10);
        store.update_offset(id, 42);
        let record = store.get(id).unwrap();
        assert_eq!(record.offset, 42);
    }

    #[test]
    fn count_tracks_insertions_and_removals() {
        let mut store = RecordStore::new();
        assert_eq!(store.count(), 0);
        let id1 = store.insert("a".to_string(), vec![], 0);
        let _id2 = store.insert("b".to_string(), vec![], 4);
        assert_eq!(store.count(), 2);
        store.remove(id1);
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn iter_yields_all_records() {
        let mut store = RecordStore::new();
        store.insert("a".to_string(), vec![], 0);
        store.insert("b".to_string(), vec![], 4);
        store.insert("c".to_string(), vec![], 8);
        let texts: Vec<&str> = store.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts.len(), 3);
        assert!(texts.contains(&"a"));
        assert!(texts.contains(&"b"));
        assert!(texts.contains(&"c"));
    }

    #[test]
    fn ids_are_unique_and_incrementing() {
        let mut store = RecordStore::new();
        let id1 = store.insert("a".to_string(), vec![], 0);
        let id2 = store.insert("b".to_string(), vec![], 4);
        let id3 = store.insert("c".to_string(), vec![], 8);
        assert!(id1 < id2);
        assert!(id2 < id3);
        // After removing and inserting, ID should still increment
        store.remove(id2);
        let id4 = store.insert("d".to_string(), vec![], 4);
        assert!(id4 > id3);
    }
}
