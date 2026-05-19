//! Static KD-Tree spatial index for sub-linear Euclidean KNN search.
//!
//! Construction is O(n log n) using `select_nth_unstable_by` (median-of-medians
//! style nth-element). The tree is stored as a flat `Vec<KdNode>` rather than
//! `Box<Node>` recursion to keep nodes cache-friendly and avoid pointer chasing.
//!
//! Limitations:
//!   - Euclidean only. Cosine/DotProduct callers must use `BruteForce`.
//!   - Immutable after build. Inserts/deletes require a full rebuild.
//!   - High dimensions degenerate to linear scan (curse of dimensionality).
//!     See [`should_use_kdtree`] for the recommended threshold.

use crate::distance::euclidean;
use crate::heap::BoundedMaxHeap;
use crate::index::brute::SearchResult;
use crate::store::record::{RecordStore, VectorRecord};
use crate::store::VectorStore;

const NONE: u32 = u32::MAX;

/// A single node in the flat KD-Tree array.
///
/// The splitting hyperplane is implicit: it is perpendicular to `axis` and
/// passes through the point at `offset` (the splitting record).
struct KdNode {
    /// RecordStore ID for the splitting point (lets us look up metadata/text).
    record_id: u64,
    /// VectorStore offset for the splitting point's vector data.
    offset: usize,
    /// Splitting axis: `0..dimension`.
    axis: u8,
    /// Index into `KdTree::nodes` for the left child, or `NONE`.
    left: u32,
    /// Index into `KdTree::nodes` for the right child, or `NONE`.
    right: u32,
}

/// A static, balanced KD-Tree over a single `VectorStore` + `RecordStore`.
///
/// The tree is **immutable after construction**. Mutating the underlying
/// `VectorStore` or `RecordStore` (insert, delete) invalidates the tree;
/// callers must rebuild from scratch.
pub struct KdTree {
    nodes: Vec<KdNode>,
    dimension: usize,
}

impl KdTree {
    /// Build a balanced KD-Tree from a `VectorStore` + `RecordStore`.
    ///
    /// Construction picks the splitting axis cyclically (`axis = depth %
    /// dimension`). At each level the median along the current axis is chosen
    /// as the splitting point via `select_nth_unstable_by` — O(n) per level,
    /// O(n log n) total. Nodes are appended to `self.nodes` in DFS order; the
    /// returned index from each recursive call is the root of that subtree.
    pub fn build(store: &VectorStore, records: &RecordStore) -> Self {
        let dimension = store.dimension();
        // Collect (record_id, offset) for every record. We do not touch vector
        // data during this collection step — that comes later when we sort by
        // axis-projected coordinates.
        let mut items: Vec<(u64, usize)> = records.iter().map(|r| (r.id, r.offset)).collect();

        let mut tree = KdTree {
            nodes: Vec::with_capacity(items.len()),
            dimension,
        };

        if !items.is_empty() && dimension > 0 {
            tree.build_subtree(store, &mut items, 0);
        }
        tree
    }

    /// Recursively build the subtree spanning `items`. Returns the node index
    /// of this subtree's root (or `NONE` if `items` is empty).
    fn build_subtree(
        &mut self,
        store: &VectorStore,
        items: &mut [(u64, usize)],
        depth: usize,
    ) -> u32 {
        if items.is_empty() {
            return NONE;
        }
        let axis = (depth % self.dimension) as u8;
        let axis_idx = axis as usize;
        let mid = items.len() / 2;

        // Median by axis-projected coordinate, O(n) average.
        items.select_nth_unstable_by(mid, |a, b| {
            let va = store.get(a.1).expect("offset must be valid")[axis_idx];
            let vb = store.get(b.1).expect("offset must be valid")[axis_idx];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Reserve our node slot *before* recursing so children get indices
        // greater than this node (DFS order).
        let (record_id, offset) = items[mid];
        let my_index = self.nodes.len() as u32;
        self.nodes.push(KdNode {
            record_id,
            offset,
            axis,
            left: NONE,
            right: NONE,
        });

        // Split into left = items[..mid], right = items[mid+1..]
        let (left_slice, right_slice_inc_mid) = items.split_at_mut(mid);
        // right_slice_inc_mid[0] is the median we already stored; skip it.
        let right_slice = &mut right_slice_inc_mid[1..];

        let left_child = self.build_subtree(store, left_slice, depth + 1);
        let right_child = self.build_subtree(store, right_slice, depth + 1);

        // Patch children indices on our node.
        self.nodes[my_index as usize].left = left_child;
        self.nodes[my_index as usize].right = right_child;
        my_index
    }

    /// Top-K nearest neighbors under Euclidean distance with geometric pruning.
    ///
    /// Filtering semantics: when a non-empty `filter` is supplied, only records
    /// matching the filter are pushed into the result heap, but the tree still
    /// recurses **as if doing unfiltered KNN** — i.e. the pruning radius is
    /// derived from accepted-into-heap candidates only. This means filtering
    /// inside the tree is primarily a "limit the returned set" mechanism, not
    /// a pruning speedup. For highly selective filters, `BruteForce::search`
    /// may be cheaper because it avoids the tree traversal overhead. This
    /// limitation is acceptable for v1 because it preserves correctness: a
    /// more aggressive scheme that pruned by filtered-out points would miss
    /// matches whenever the closest filter-matching record was geometrically
    /// far from filtered-out neighbors.
    pub fn search(
        &self,
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        filter: Option<&[(String, String)]>,
    ) -> Vec<SearchResult> {
        if k == 0 || self.nodes.is_empty() || query.len() != self.dimension {
            return vec![];
        }

        let mut heap = BoundedMaxHeap::new(k);
        self.search_recursive(store, records, query, k, filter, 0, &mut heap);

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

    /// Recursive descent. `node_idx` is the current subtree root.
    #[allow(clippy::too_many_arguments)]
    fn search_recursive(
        &self,
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        filter: Option<&[(String, String)]>,
        node_idx: u32,
        heap: &mut BoundedMaxHeap,
    ) {
        if node_idx == NONE {
            return;
        }
        let node = &self.nodes[node_idx as usize];

        // Score this node's point. The vector must exist (build guarantees it).
        if let Some(vec) = store.get(node.offset) {
            let score = euclidean(query, vec);
            // Apply filter at heap-push time only — see method docstring.
            let push_allowed = match filter {
                None => true,
                Some(pairs) => match records.get(node.record_id) {
                    Some(rec) => record_matches_filter(rec, pairs),
                    None => false,
                },
            };
            if push_allowed {
                heap.push(score, node.record_id);
            }
        }

        // Decide which child is "near" by comparing the query coordinate to the
        // splitting plane.
        let axis = node.axis as usize;
        let split_value = store
            .get(node.offset)
            .map(|v| v[axis])
            .expect("offset must be valid");
        let delta = query[axis] - split_value;
        let (near, far) = if delta < 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        // Descend into the near child first.
        self.search_recursive(store, records, query, k, filter, near, heap);

        // Visit the far child only if the splitting plane is closer than the
        // current worst (top-of-heap) score, or if the heap is not yet full.
        // Pruning bound is `|delta|` because we use sqrt-Euclidean distance.
        let bound = delta.abs();
        let must_visit_far = if heap.len() < k {
            true
        } else {
            match heap.peek_worst() {
                Some(worst) => bound < worst,
                None => true,
            }
        };
        if must_visit_far {
            self.search_recursive(store, records, query, k, filter, far, heap);
        }
    }

    /// Number of stored nodes (== number of indexed vectors).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the tree has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The vector dimension the tree was built for.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Same predicate as `brute::record_matches_filter`. Duplicated to keep the
/// brute-force module's helper private.
fn record_matches_filter(record: &VectorRecord, filter: &[(String, String)]) -> bool {
    filter.iter().all(|needle| {
        record
            .metadata
            .iter()
            .any(|entry| entry.0 == needle.0 && entry.1 == needle.1)
    })
}

/// Heuristic for KD-Tree vs BruteForce selection.
///
/// Returns `true` when a KD-Tree is likely to outperform brute force, i.e.
/// the corpus is "big enough" for tree overhead to pay off **and** the
/// dimension is low enough that geometric pruning is still effective.
///
/// Defaults: `n >= 1024 && dim <= 64`. The 1024 break-even is a rough number
/// from typical KD-tree literature; 64 dimensions is a conservative cap
/// because curse-of-dimensionality causes axis pruning to fail almost
/// completely once `dim` exceeds a few dozen (every node ends up visited).
pub fn should_use_kdtree(n: usize, dim: usize) -> bool {
    n >= 1024 && dim <= 64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::index::brute::BruteForce;
    use proptest::prelude::*;

    /// Helper: build a store with the given vectors and text labels.
    fn build_stores(data: &[(&[f32], &str)]) -> (VectorStore, RecordStore) {
        let dim = data.first().map(|(v, _)| v.len()).unwrap_or(2);
        let mut vs = VectorStore::new(dim);
        let mut rs = RecordStore::new();
        for (vec, text) in data {
            let offset = vs.insert(vec).unwrap();
            rs.insert(text.to_string(), vec![], offset);
        }
        (vs, rs)
    }

    #[test]
    fn build_on_empty_store_returns_empty_tree() {
        let vs = VectorStore::new(2);
        let rs = RecordStore::new();
        let tree = KdTree::build(&vs, &rs);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.dimension(), 2);
    }

    #[test]
    fn build_on_one_vector_returns_single_node() {
        let (vs, rs) = build_stores(&[(&[1.0, 2.0], "only")]);
        let tree = KdTree::build(&vs, &rs);
        assert_eq!(tree.len(), 1);
        let results = tree.search(&vs, &rs, &[0.0, 0.0], 5, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "only");
    }

    #[test]
    fn search_5_2d_vectors_top3_matches_expected() {
        let (vs, rs) = build_stores(&[
            (&[0.0, 0.0], "origin"),
            (&[1.0, 0.0], "close"),
            (&[2.0, 0.0], "mid"),
            (&[10.0, 0.0], "far"),
            (&[0.5, 0.0], "closest"),
        ]);
        let tree = KdTree::build(&vs, &rs);
        let results = tree.search(&vs, &rs, &[0.0, 0.0], 3, None);
        assert_eq!(results.len(), 3);
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["origin", "closest", "close"]);
    }

    #[test]
    fn search_empty_query_returns_empty() {
        let (vs, rs) = build_stores(&[(&[1.0, 2.0], "only")]);
        let tree = KdTree::build(&vs, &rs);
        // Wrong-dimension query yields empty results (defensive).
        let results = tree.search(&vs, &rs, &[1.0], 3, None);
        assert!(results.is_empty());
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let (vs, rs) = build_stores(&[(&[1.0, 2.0], "only")]);
        let tree = KdTree::build(&vs, &rs);
        let results = tree.search(&vs, &rs, &[0.0, 0.0], 0, None);
        assert!(results.is_empty());
    }

    #[test]
    fn filter_excludes_records_in_kdtree_search() {
        let mut vs = VectorStore::new(2);
        let mut rs = RecordStore::new();
        #[allow(clippy::type_complexity)]
        let coords: [(&[f32], &str, &[(&str, &str)]); 4] = [
            (&[0.0, 0.0], "alpha", &[("lang", "en")]),
            (&[1.0, 0.0], "beta", &[("lang", "pl")]),
            (&[2.0, 0.0], "gamma", &[("lang", "en")]),
            (&[3.0, 0.0], "delta", &[("lang", "pl")]),
        ];
        for (v, t, meta) in coords.iter() {
            let off = vs.insert(v).unwrap();
            let m: Vec<(String, String)> = meta
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect();
            rs.insert((*t).to_string(), m, off);
        }
        let tree = KdTree::build(&vs, &rs);
        let filter = vec![("lang".to_string(), "en".to_string())];
        let results = tree.search(&vs, &rs, &[0.0, 0.0], 10, Some(filter.as_slice()));
        let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(results.len(), 2);
        assert!(texts.contains(&"alpha"));
        assert!(texts.contains(&"gamma"));
        assert!(!texts.contains(&"beta"));
        assert!(!texts.contains(&"delta"));

        // Cross-check filtered results equal BruteForce with the same filter.
        let brute = BruteForce::search(
            &vs,
            &rs,
            &[0.0, 0.0],
            10,
            Metric::Euclidean,
            Some(filter.as_slice()),
        );
        let brute_ids: Vec<u64> = brute.iter().map(|r| r.id).collect();
        let tree_ids: Vec<u64> = results.iter().map(|r| r.id).collect();
        assert_eq!(tree_ids, brute_ids);
    }

    #[test]
    fn should_use_kdtree_heuristic() {
        assert!(should_use_kdtree(1024, 8));
        assert!(should_use_kdtree(2048, 64));
        assert!(!should_use_kdtree(512, 8));
        assert!(!should_use_kdtree(2048, 128));
        assert!(!should_use_kdtree(0, 0));
    }

    // --- proptest: KdTree must match BruteForce on all queries. ---

    fn vec_strategy(dim: usize, n: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        prop::collection::vec(prop::collection::vec(-100.0f32..100.0f32, dim), n..=n)
    }

    fn run_match_case(dim: usize, corpus: Vec<Vec<f32>>, query: Vec<f32>, k: usize) {
        let mut vs = VectorStore::new(dim);
        let mut rs = RecordStore::new();
        for (i, v) in corpus.iter().enumerate() {
            let off = vs.insert(v).unwrap();
            rs.insert(format!("doc-{i}"), vec![], off);
        }
        let tree = KdTree::build(&vs, &rs);
        let tree_results = tree.search(&vs, &rs, &query, k, None);
        let brute_results = BruteForce::search(&vs, &rs, &query, k, Metric::Euclidean, None);

        let tree_ids: Vec<u64> = tree_results.iter().map(|r| r.id).collect();
        let brute_ids: Vec<u64> = brute_results.iter().map(|r| r.id).collect();
        assert_eq!(
            tree_ids, brute_ids,
            "tree vs brute id mismatch for dim={dim} k={k}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn kdtree_matches_brute_force_2d(
            corpus in vec_strategy(2, 50),
            query in prop::collection::vec(-100.0f32..100.0f32, 2),
            k in 1usize..=20,
        ) {
            run_match_case(2, corpus, query, k);
        }

        #[test]
        fn kdtree_matches_brute_force_4d(
            corpus in vec_strategy(4, 80),
            query in prop::collection::vec(-100.0f32..100.0f32, 4),
            k in 1usize..=20,
        ) {
            run_match_case(4, corpus, query, k);
        }

        #[test]
        fn kdtree_matches_brute_force_8d(
            corpus in vec_strategy(8, 100),
            query in prop::collection::vec(-100.0f32..100.0f32, 8),
            k in 1usize..=20,
        ) {
            run_match_case(8, corpus, query, k);
        }
    }
}
