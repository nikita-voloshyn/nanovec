use crate::distance::{distance_fn, Metric};
use crate::heap::BoundedMaxHeap;
use crate::store::record::{RecordStore, VectorRecord};
use crate::store::VectorStore;

/// Returns true iff `record.metadata` contains every `(key, value)` pair in
/// `filter`. The check is a linear scan because metadata is stored as a small
/// `Vec<(String, String)>` — typically <10 entries per record — so building a
/// hash set per record would cost more than the linear walk it would replace.
fn record_matches_filter(record: &VectorRecord, filter: &[(String, String)]) -> bool {
    filter.iter().all(|needle| {
        record
            .metadata
            .iter()
            .any(|entry| entry.0 == needle.0 && entry.1 == needle.1)
    })
}

/// A single search result containing the record ID, distance score, text, and metadata.
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub text: String,
    pub metadata: Vec<(String, String)>,
}

/// Errors that can occur during delete operations.
#[derive(Debug)]
pub enum DeleteError {
    /// The requested ID was not found in the record store.
    NotFound(u64),
}

impl std::fmt::Display for DeleteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeleteError::NotFound(id) => write!(f, "record not found: {id}"),
        }
    }
}

impl std::error::Error for DeleteError {}

/// Brute-force linear scan search.
pub struct BruteForce;

impl BruteForce {
    /// Linear scan KNN search. Returns up to `k` results sorted by score
    /// ascending (closest first).
    ///
    /// If `filter` is `Some(pairs)`, only records whose metadata contains
    /// **every** `(key, value)` pair are considered (logical AND over the
    /// supplied predicates). Filtering happens **before** the distance
    /// computation so excluded records never touch the SIMD path. `None`
    /// disables filtering (legacy behavior).
    pub fn search(
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        metric: Metric,
        filter: Option<&[(String, String)]>,
    ) -> Vec<SearchResult> {
        if k == 0 || store.count() == 0 {
            return vec![];
        }
        let dist = distance_fn(metric);
        let mut heap = BoundedMaxHeap::new(k);
        for record in records.iter() {
            if let Some(pairs) = filter {
                if !record_matches_filter(record, pairs) {
                    continue;
                }
            }
            if let Some(vec) = store.get(record.offset) {
                let score = dist.compute(query, vec);
                heap.push(score, record.id);
            }
        }
        heap.into_sorted_vec()
            .into_iter()
            .filter_map(|(score, id)| {
                records.get(id).map(|r| SearchResult {
                    id,
                    score,
                    text: r.text.clone(),
                    metadata: r.metadata.clone(),
                })
            })
            .collect()
    }
}

/// Coordinated delete: removes from both VectorStore and RecordStore.
/// Uses swap_remove to stay O(1) amortized.
pub fn delete(
    store: &mut VectorStore,
    records: &mut RecordStore,
    id: u64,
) -> Result<(), DeleteError> {
    let record = records.get(id).ok_or(DeleteError::NotFound(id))?;
    let offset = record.offset;

    // Remove from vector store (swap-removes the vector at offset).
    store
        .swap_remove(offset)
        .map_err(|_| DeleteError::NotFound(id))?;

    // Remove the record.
    records.remove(id);

    // If the removed vector was not the last one, the last vector was moved
    // into `offset`. We need to find the record that pointed to the old last
    // position and update it.
    //
    // After swap_remove, store.count() == old_count - 1.
    // The old last vector lived at index (old_count - 1) == store.count().
    let old_last_index = store.count();
    if old_last_index != offset {
        // Find the record whose offset still points to old_last_index.
        let moved_id = records.iter().find_map(|r| {
            if r.offset == old_last_index {
                Some(r.id)
            } else {
                None
            }
        });
        if let Some(mid) = moved_id {
            records.update_offset(mid, offset);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a store with the given 2D vectors and text labels.
    fn build_stores(data: &[(&[f32], &str)]) -> (VectorStore, RecordStore) {
        let dim = data.first().map(|(v, _)| v.len()).unwrap_or(0);
        let mut vs = VectorStore::new(dim);
        let mut rs = RecordStore::new();
        for (vec, text) in data {
            let offset = vs.insert(vec).unwrap();
            rs.insert(text.to_string(), vec![], offset);
        }
        (vs, rs)
    }

    #[test]
    fn search_5_vectors_k3_returns_3_closest() {
        let (vs, rs) = build_stores(&[
            (&[0.0, 0.0], "origin"),
            (&[1.0, 0.0], "close"),
            (&[2.0, 0.0], "mid"),
            (&[10.0, 0.0], "far"),
            (&[0.5, 0.0], "closest"),
        ]);
        let query = &[0.0, 0.0];
        let results = BruteForce::search(&vs, &rs, query, 3, Metric::Euclidean, None);
        assert_eq!(results.len(), 3);
        // Ascending by distance: origin (0), closest (0.5), close (1.0)
        assert_eq!(results[0].text, "origin");
        assert_eq!(results[1].text, "closest");
        assert_eq!(results[2].text, "close");
    }

    #[test]
    fn search_empty_store_returns_empty() {
        let vs = VectorStore::new(2);
        let rs = RecordStore::new();
        let results = BruteForce::search(&vs, &rs, &[1.0, 2.0], 5, Metric::Euclidean, None);
        assert!(results.is_empty());
    }

    #[test]
    fn search_k_greater_than_count_returns_all() {
        let (vs, rs) = build_stores(&[(&[1.0, 0.0], "a"), (&[2.0, 0.0], "b")]);
        let results = BruteForce::search(&vs, &rs, &[0.0, 0.0], 10, Metric::Euclidean, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn delete_existing_id_decreases_count() {
        let (mut vs, mut rs) =
            build_stores(&[(&[1.0, 0.0], "a"), (&[2.0, 0.0], "b"), (&[3.0, 0.0], "c")]);
        assert_eq!(rs.count(), 3);
        delete(&mut vs, &mut rs, 0).unwrap();
        assert_eq!(rs.count(), 2);
        assert_eq!(vs.count(), 2);
        // Search should not find "a"
        let results = BruteForce::search(&vs, &rs, &[1.0, 0.0], 10, Metric::Euclidean, None);
        assert_eq!(results.len(), 2);
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        assert!(!texts.contains(&"a"));
    }

    #[test]
    fn delete_nonexistent_id_returns_not_found() {
        let (mut vs, mut rs) = build_stores(&[(&[1.0, 0.0], "a")]);
        let err = delete(&mut vs, &mut rs, 999).unwrap_err();
        match err {
            DeleteError::NotFound(id) => assert_eq!(id, 999),
        }
    }

    #[test]
    fn search_returns_metadata_from_records() {
        let mut vs = VectorStore::new(2);
        let mut rs = RecordStore::new();
        let offset = vs.insert(&[1.0, 0.0]).unwrap();
        rs.insert(
            "doc".to_string(),
            vec![("source".to_string(), "test".to_string())],
            offset,
        );

        let results = BruteForce::search(&vs, &rs, &[1.0, 0.0], 1, Metric::Euclidean, None);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].metadata,
            vec![("source".to_string(), "test".to_string())]
        );
    }

    #[test]
    fn delete_first_vector_swap_remove_remaining_searchable() {
        let (mut vs, mut rs) = build_stores(&[
            (&[0.0, 0.0], "first"),
            (&[1.0, 0.0], "second"),
            (&[2.0, 0.0], "third"),
        ]);
        delete(&mut vs, &mut rs, 0).unwrap(); // removes "first"
        assert_eq!(vs.count(), 2);
        let results = BruteForce::search(&vs, &rs, &[0.0, 0.0], 10, Metric::Euclidean, None);
        assert_eq!(results.len(), 2);
        // "second" is at distance 1.0, "third" at distance 2.0
        assert_eq!(results[0].text, "second");
        assert_eq!(results[1].text, "third");
    }

    /// Helper for filter tests: build stores with metadata.
    #[allow(clippy::type_complexity)]
    fn build_stores_with_meta(
        data: &[(&[f32], &str, &[(&str, &str)])],
    ) -> (VectorStore, RecordStore) {
        let dim = data.first().map(|(v, _, _)| v.len()).unwrap_or(0);
        let mut vs = VectorStore::new(dim);
        let mut rs = RecordStore::new();
        for (vec, text, meta) in data {
            let offset = vs.insert(vec).unwrap();
            let metadata: Vec<(String, String)> = meta
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            rs.insert(text.to_string(), metadata, offset);
        }
        (vs, rs)
    }

    #[test]
    fn filter_matches_subset_of_records() {
        let (vs, rs) = build_stores_with_meta(&[
            (&[0.0, 0.0], "alpha", &[("lang", "en")]),
            (&[1.0, 0.0], "beta", &[("lang", "pl")]),
            (&[2.0, 0.0], "gamma", &[("lang", "en")]),
        ]);
        let filter = vec![("lang".to_string(), "en".to_string())];
        let results = BruteForce::search(
            &vs,
            &rs,
            &[0.0, 0.0],
            10,
            Metric::Euclidean,
            Some(filter.as_slice()),
        );
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(results.len(), 2, "expected 2 en-tagged results");
        assert!(texts.contains(&"alpha"));
        assert!(texts.contains(&"gamma"));
        assert!(!texts.contains(&"beta"));
    }

    #[test]
    fn filter_excluding_everything_returns_empty() {
        let (vs, rs) = build_stores_with_meta(&[
            (&[0.0, 0.0], "alpha", &[("lang", "en")]),
            (&[1.0, 0.0], "beta", &[("lang", "pl")]),
        ]);
        let filter = vec![("lang".to_string(), "ja".to_string())];
        let results = BruteForce::search(
            &vs,
            &rs,
            &[0.0, 0.0],
            10,
            Metric::Euclidean,
            Some(filter.as_slice()),
        );
        assert!(results.is_empty(), "filter excludes everything");
    }

    #[test]
    fn filter_with_multiple_pairs_requires_all_to_match() {
        let (vs, rs) = build_stores_with_meta(&[
            (&[0.0, 0.0], "alpha", &[("lang", "en"), ("topic", "math")]),
            (&[1.0, 0.0], "beta", &[("lang", "en"), ("topic", "history")]),
            (&[2.0, 0.0], "gamma", &[("lang", "pl"), ("topic", "math")]),
        ]);
        let filter = vec![
            ("lang".to_string(), "en".to_string()),
            ("topic".to_string(), "math".to_string()),
        ];
        let results = BruteForce::search(
            &vs,
            &rs,
            &[0.0, 0.0],
            10,
            Metric::Euclidean,
            Some(filter.as_slice()),
        );
        assert_eq!(results.len(), 1, "only alpha has both tags");
        assert_eq!(results[0].text, "alpha");
    }

    #[test]
    fn filter_none_matches_existing_behavior() {
        let (vs, rs) = build_stores_with_meta(&[
            (&[0.0, 0.0], "alpha", &[("lang", "en")]),
            (&[1.0, 0.0], "beta", &[("lang", "pl")]),
        ]);
        let with_none = BruteForce::search(&vs, &rs, &[0.0, 0.0], 10, Metric::Euclidean, None);
        assert_eq!(with_none.len(), 2);
    }

    #[test]
    fn filter_empty_slice_matches_all_records() {
        // An empty filter list means "no predicates" — every record matches.
        let (vs, rs) = build_stores_with_meta(&[
            (&[0.0, 0.0], "alpha", &[("lang", "en")]),
            (&[1.0, 0.0], "beta", &[]),
        ]);
        let filter: Vec<(String, String)> = vec![];
        let results = BruteForce::search(
            &vs,
            &rs,
            &[0.0, 0.0],
            10,
            Metric::Euclidean,
            Some(filter.as_slice()),
        );
        assert_eq!(results.len(), 2);
    }
}
