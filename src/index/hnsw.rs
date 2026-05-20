//! Phase 7 — Hierarchical Navigable Small World graph (HNSW).
//!
//! Approximate nearest-neighbor index. Builds a multi-layer proximity graph
//! over the corpus: the top layers hold a sparse subset of points with
//! long-range links (fast coarse navigation), the bottom layer holds every
//! point with short-range links (local refinement). Search descends the
//! hierarchy greedily, then runs a beam search on layer 0.
//!
//! Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs",
//! arXiv:1603.09320 (Algorithms 1–4).
//!
//! Implementation notes:
//! - Graph: `Vec<Vec<Vec<u32>>>` indexed `[layer][node] -> Vec<neighbor>`.
//!   Layers beyond a node's level have empty neighbor lists; the outer index
//!   is the global node id.
//! - Distance: calls `simd::cosine` directly so HNSW automatically inherits
//!   the AVX2/NEON acceleration from Phase 3.
//! - Neighbor selection: heuristic from Algorithm 4 of the paper (not the
//!   simple "top-M" — it picks geometrically diverse neighbors, which is
//!   what makes recall actually hit ~0.95 at M=16).
//! - RNG: `SmallRng` with a fixed seed for reproducible tests.
//! - Distance metric: cosine only for now (the embedded path's natural
//!   default). Other metrics can be added by parameterising `distance()`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::heap::BoundedMaxHeap;
use crate::index::brute::SearchResult;
use crate::simd;
use crate::store::record::RecordStore;
use crate::store::VectorStore;

/// HNSW build / search parameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Maximum number of neighbors per node on each layer above layer 0.
    /// Layer 0 uses `m_max0 = 2 * m` (paper recommendation).
    pub m: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// Beam width during search. Higher → better recall, slower.
    pub ef_search: usize,
    /// Multiplier for layer assignment: probability of node landing on
    /// layer `l` is `exp(-l / m_l)`. The paper recommends `1 / ln(M)`.
    pub m_l: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            ef_construction: 200,
            ef_search: 50,
            m_l: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswParams {
    /// Convenience: paper recommends `m_l = 1 / ln(M)` — keep them in sync.
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        Self {
            m,
            ef_construction,
            ef_search,
            m_l: 1.0 / (m as f64).ln(),
        }
    }
}

/// Hierarchical Navigable Small World graph.
pub struct Hnsw {
    params: HnswParams,
    /// `layers[l][node]` -> neighbor ids on layer `l`. Layer `node_levels[i]`
    /// is the highest layer node `i` participates in.
    layers: Vec<Vec<Vec<u32>>>,
    node_levels: Vec<u8>,
    entry_point: Option<u32>,
    rng: SmallRng,
    /// Snapshot of the vector dimension at construction; rejects mismatched
    /// query slices in `search`.
    dimension: usize,
}

/// Float wrapper that orders by distance (smaller = better) and breaks ties
/// deterministically by id. Used for the BinaryHeap priority queues.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DistanceOrdered {
    dist: f32,
    id: u32,
}

impl Eq for DistanceOrdered {}
impl Ord for DistanceOrdered {
    fn cmp(&self, other: &Self) -> Ordering {
        // Smaller distance is "less"; tie-break by id ascending for determinism.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}
impl PartialOrd for DistanceOrdered {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reverse ordering — max-heap of *closest* by using BinaryHeap as a min-heap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct DistanceReversed(DistanceOrdered);
impl Eq for DistanceReversed {}
impl Ord for DistanceReversed {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}
impl PartialOrd for DistanceReversed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hnsw {
    /// New empty index. Pre-allocates capacity for `capacity` nodes — caller
    /// will then add them via `insert()` or use `build()` for the bulk path.
    pub fn new(dimension: usize, params: HnswParams) -> Self {
        Self::with_seed(dimension, params, 0xC0FFEE_u64)
    }

    /// Same as `new` but with an explicit RNG seed for reproducible tests.
    pub fn with_seed(dimension: usize, params: HnswParams, seed: u64) -> Self {
        Self {
            params,
            layers: vec![Vec::new()], // start with one (empty) layer
            node_levels: Vec::new(),
            entry_point: None,
            rng: SmallRng::seed_from_u64(seed),
            dimension,
        }
    }

    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    pub fn len(&self) -> usize {
        self.node_levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.node_levels.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Approximate memory footprint in bytes — sum of layer neighbor lists +
    /// node levels vector. Excludes Vec overhead.
    pub fn approx_bytes(&self) -> u64 {
        let neigh: usize = self
            .layers
            .iter()
            .flat_map(|layer| layer.iter().map(|n| n.len() * std::mem::size_of::<u32>()))
            .sum();
        let level_bytes = self.node_levels.len(); // u8 per node
        (neigh + level_bytes) as u64
    }

    /// Build an HNSW index over an existing `VectorStore`. Iterates the store
    /// in insertion order and inserts each vector into the graph.
    pub fn build(store: &VectorStore, params: HnswParams) -> Self {
        let mut hnsw = Self::new(store.dimension(), params);
        let n = store.count();
        for (offset, _vec) in store.iter() {
            hnsw.insert(offset as u32, store);
            // Coarse progress log every 10k for big builds.
            if (offset + 1) % 10_000 == 0 {
                tracing::info!(inserted = offset + 1, total = n, "hnsw build progress");
            }
        }
        hnsw
    }

    /// Sample a level using the geometric distribution from the paper:
    /// `level = floor(-ln(uniform(0, 1)) * m_l)`.
    fn sample_level(&mut self) -> u8 {
        let u: f64 = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
        let lvl = (-u.ln() * self.params.m_l).floor() as i64;
        lvl.clamp(0, u8::MAX as i64) as u8
    }

    /// Distance between a query slice and a stored vector by node id.
    #[inline]
    fn dist_to(&self, query: &[f32], id: u32, store: &VectorStore) -> f32 {
        let vec = store.get(id as usize).expect("node id out of range");
        simd::cosine(query, vec)
    }

    /// Algorithm 2 (paper): search a single layer with a fixed-size beam.
    /// Returns up to `ef` ids sorted ascending by distance.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        store: &VectorStore,
    ) -> Vec<DistanceOrdered> {
        let mut visited: Vec<bool> = vec![false; self.node_levels.len()];
        // candidate: min-heap (closest at top, pop next to expand).
        let mut candidates: BinaryHeap<DistanceReversed> = BinaryHeap::new();
        // result: max-heap (furthest of the `ef` best — pop to evict).
        let mut result: BinaryHeap<DistanceOrdered> = BinaryHeap::new();

        for &ep in entry_points {
            let d = self.dist_to(query, ep, store);
            let entry = DistanceOrdered { dist: d, id: ep };
            candidates.push(DistanceReversed(entry));
            result.push(entry);
            visited[ep as usize] = true;
        }

        while let Some(DistanceReversed(c)) = candidates.pop() {
            // If the closest candidate is further than the worst kept result,
            // we can stop — no improvement possible.
            let worst_kept = result.peek().copied().expect("result nonempty");
            if c.dist > worst_kept.dist {
                break;
            }
            // Expand neighbors of `c` at this layer.
            let neighbors = &self.layers[layer][c.id as usize];
            for &n in neighbors {
                let n_idx = n as usize;
                if visited[n_idx] {
                    continue;
                }
                visited[n_idx] = true;
                let d = self.dist_to(query, n, store);
                let candidate = DistanceOrdered { dist: d, id: n };
                let worst = result.peek().copied().expect("nonempty");
                if d < worst.dist || result.len() < ef {
                    candidates.push(DistanceReversed(candidate));
                    result.push(candidate);
                    if result.len() > ef {
                        result.pop(); // pops worst (BinaryHeap is max-heap)
                    }
                }
            }
        }

        let mut out: Vec<DistanceOrdered> = result.into_sorted_vec();
        // Canonicalise: ascending by distance. `into_sorted_vec` returns
        // ascending of our `Ord` (which orders by distance) — re-sort to be
        // robust to future Ord changes.
        out.sort();
        out
    }

    /// Algorithm 4 (paper): select up to `m` neighbors from `candidates`
    /// using the diversity heuristic. Skips a candidate if any already-selected
    /// neighbor is closer to it than the query is — that means the candidate is
    /// redundant given an existing pick.
    fn select_neighbors_heuristic(
        &self,
        candidates: &[DistanceOrdered],
        m: usize,
        layer: usize,
        store: &VectorStore,
        node_being_inserted_query: &[f32],
    ) -> Vec<u32> {
        // candidates are already sorted ascending by distance to the query.
        let _ = layer; // unused for now; could be used for layer-specific tweaks.
        let mut selected: Vec<u32> = Vec::with_capacity(m);
        for cand in candidates {
            if selected.len() >= m {
                break;
            }
            // Heuristic: include only if no already-selected neighbor is closer
            // to `cand` than `cand` is to the query.
            let cand_vec = store.get(cand.id as usize).expect("cand id valid");
            let cand_to_query = cand.dist; // already computed
            let mut keep = true;
            for &s in &selected {
                let s_vec = store.get(s as usize).expect("selected id valid");
                let s_to_cand = simd::cosine(s_vec, cand_vec);
                if s_to_cand < cand_to_query {
                    keep = false;
                    break;
                }
            }
            if keep {
                selected.push(cand.id);
            }
            // (Note: we keep `node_being_inserted_query` so the signature
            // matches an alternative variant; today the heuristic operates on
            // the cand→query distance which is in `cand.dist`.)
            let _ = node_being_inserted_query;
        }
        selected
    }

    /// Insert node `id` from `store` into the graph.
    pub fn insert(&mut self, id: u32, store: &VectorStore) {
        debug_assert_eq!(
            id as usize,
            self.node_levels.len(),
            "ids must be sequential 0..N"
        );

        let level = self.sample_level();
        self.node_levels.push(level);

        // Ensure layer arrays exist up to `level`.
        while self.layers.len() <= level as usize {
            self.layers.push(Vec::new());
        }
        // Grow every layer's nodes slot to include this id.
        for layer in self.layers.iter_mut() {
            while layer.len() <= id as usize {
                layer.push(Vec::new());
            }
        }

        let query = store.get(id as usize).expect("inserted id has a vector");

        // First node ever inserted — becomes the entry point with whatever level it has.
        let Some(mut ep) = self.entry_point else {
            self.entry_point = Some(id);
            return;
        };

        let entry_level = self.node_levels[ep as usize];

        // Phase A: greedy descent from entry_level down to level + 1, ef=1.
        let top_search = entry_level.max(level) as usize;
        if entry_level > level {
            for layer in (level as usize + 1..=top_search).rev() {
                let found = self.search_layer(query, &[ep], 1, layer, store);
                if let Some(closest) = found.first() {
                    ep = closest.id;
                }
            }
        }

        // Phase B: at each layer from min(level, entry_level) down to 0,
        // beam-search with ef_construction and connect.
        let m_max0 = 2 * self.params.m;
        let mut entry_points = vec![ep];
        for layer in (0..=level.min(entry_level) as usize).rev() {
            let candidates = self.search_layer(
                query,
                &entry_points,
                self.params.ef_construction,
                layer,
                store,
            );
            // Paper uses M neighbors during insert on all layers; layer 0's
            // higher cap (m_max0 = 2*M) is enforced via pruning below.
            let selected = self.select_neighbors_heuristic(
                &candidates,
                self.params.m,
                layer,
                store,
                query,
            );

            // Connect `id` to selected neighbors.
            for &neigh in &selected {
                self.layers[layer][id as usize].push(neigh);
                self.layers[layer][neigh as usize].push(id);
                // Prune the neighbor if it exceeds the per-layer cap.
                let cap = if layer == 0 { m_max0 } else { self.params.m };
                if self.layers[layer][neigh as usize].len() > cap {
                    self.prune_neighbors(neigh, layer, cap, store);
                }
            }

            // Next layer's entry points = `selected` (paper-recommended).
            if !selected.is_empty() {
                entry_points = selected;
            }
        }

        // Possibly promote entry point if our level is higher than the current one.
        if level > self.node_levels[self.entry_point.unwrap() as usize] {
            self.entry_point = Some(id);
        }
    }

    /// Re-run heuristic neighbor selection on a node whose connection list
    /// has exceeded the per-layer cap.
    fn prune_neighbors(&mut self, node: u32, layer: usize, cap: usize, store: &VectorStore) {
        let node_vec = store.get(node as usize).expect("node id valid");
        // Compute distances to every current neighbor.
        let mut candidates: Vec<DistanceOrdered> = self.layers[layer][node as usize]
            .iter()
            .map(|&n| DistanceOrdered {
                dist: self.dist_to(node_vec, n, store),
                id: n,
            })
            .collect();
        candidates.sort();
        // Re-select with heuristic, capping at `cap`.
        let kept = self.select_neighbors_heuristic(&candidates, cap, layer, store, node_vec);
        self.layers[layer][node as usize] = kept;
    }

    /// Approximate KNN search. Returns up to `k` results sorted by distance
    /// ascending. Filter, if supplied, restricts results to records whose
    /// metadata matches **all** `(key, value)` pairs — applied post-search
    /// (we over-fetch by `ef_search` candidates first, then filter).
    pub fn search(
        &self,
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        filter: Option<&[(String, String)]>,
    ) -> Vec<SearchResult> {
        if k == 0 || self.is_empty() {
            return vec![];
        }
        let Some(mut ep) = self.entry_point else {
            return vec![];
        };

        // Greedy descent from the top layer down to layer 1 with ef=1.
        let top_layer = self.layers.len().saturating_sub(1);
        for layer in (1..=top_layer).rev() {
            let found = self.search_layer(query, &[ep], 1, layer, store);
            if let Some(closest) = found.first() {
                ep = closest.id;
            }
        }

        // Beam search on layer 0 with ef_search.
        let ef = self.params.ef_search.max(k);
        let mut hits = self.search_layer(query, &[ep], ef, 0, store);

        // Apply post-filter (over-fetched candidates first).
        if let Some(pairs) = filter {
            hits.retain(|h| {
                records
                    .get(h.id as u64)
                    .map(|r| {
                        pairs.iter().all(|needle| {
                            r.metadata
                                .iter()
                                .any(|e| e.0 == needle.0 && e.1 == needle.1)
                        })
                    })
                    .unwrap_or(false)
            });
        }

        // Use BoundedMaxHeap to keep top-k as the unified return shape.
        let mut heap = BoundedMaxHeap::new(k);
        for h in hits {
            heap.push(h.dist, h.id as u64);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::record::RecordStore;
    use crate::store::VectorStore;

    fn make_vec(seed: usize, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| (((seed * 31 + i) as f32) * 0.013_37).sin())
            .collect();
        // L2-normalise so cosine distance is bounded in [0, 2].
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn build_corpus(n: usize, dim: usize) -> (VectorStore, RecordStore) {
        let mut vs = VectorStore::new(dim);
        let mut rs = RecordStore::new();
        for i in 0..n {
            let v = make_vec(i, dim);
            let off = vs.insert(&v).unwrap();
            rs.insert(format!("doc-{i}"), vec![], off);
        }
        (vs, rs)
    }

    fn brute_topk(store: &VectorStore, query: &[f32], k: usize) -> Vec<u32> {
        use crate::heap::BoundedMaxHeap;
        let mut heap = BoundedMaxHeap::new(k);
        for (off, v) in store.iter() {
            let d = simd::cosine(query, v);
            heap.push(d, off as u64);
        }
        heap.into_sorted_vec()
            .into_iter()
            .map(|(_, id)| id as u32)
            .collect()
    }

    #[test]
    fn empty_graph_search_returns_no_results() {
        let store = VectorStore::new(8);
        let records = RecordStore::new();
        let hnsw = Hnsw::new(8, HnswParams::default());
        let q = make_vec(0, 8);
        assert!(hnsw.search(&store, &records, &q, 5, None).is_empty());
    }

    #[test]
    fn single_node_search_returns_itself() {
        let (vs, rs) = build_corpus(1, 8);
        let hnsw = Hnsw::build(&vs, HnswParams::new(8, 50, 20));
        let q = make_vec(0, 8);
        let hits = hnsw.search(&vs, &rs, &q, 5, None);
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].score < 1e-3,
            "self-distance should be ~0, got {}",
            hits[0].score
        );
    }

    #[test]
    fn recall_on_500_random_vectors_matches_brute_force() {
        let dim = 64;
        let n = 500;
        let (vs, rs) = build_corpus(n, dim);
        let hnsw = Hnsw::build(&vs, HnswParams::new(16, 200, 50));

        let mut total_hits = 0;
        let trials = 30;
        let k = 10;
        for q_seed in 1000..1000 + trials {
            let q = make_vec(q_seed, dim);
            let truth = brute_topk(&vs, &q, k);
            let hits = hnsw.search(&vs, &rs, &q, k, None);
            let hit_ids: Vec<u32> = hits.iter().map(|h| h.id as u32).collect();
            let intersect = truth.iter().filter(|t| hit_ids.contains(t)).count();
            total_hits += intersect;
        }
        let recall = total_hits as f64 / (trials * k) as f64;
        println!("[hnsw_recall_500] recall@{k} = {recall:.3}");
        assert!(
            recall >= 0.85,
            "recall@10 = {recall:.3}, expected >= 0.85 on 500x{dim}-dim corpus"
        );
    }

    #[test]
    fn build_assigns_geometric_levels() {
        let (vs, _) = build_corpus(1000, 16);
        let hnsw = Hnsw::build(&vs, HnswParams::default());
        // With M=16, m_l = 1/ln(16) ≈ 0.361; expected P(level=0) = 1 - exp(-ln(16))
        // = 1 - 1/16 ≈ 0.9375. Bound generously to absorb sample variance.
        let l0 = hnsw.node_levels.iter().filter(|&&l| l == 0).count();
        let frac0 = l0 as f64 / 1000.0;
        println!("[hnsw_levels] fraction at level 0: {frac0:.3} (target ~0.94)");
        assert!(
            (0.88..0.98).contains(&frac0),
            "expected ~94% at level 0, got {frac0:.3}"
        );
        // Plus a sanity check: top layer should be small but nonzero.
        let max_lvl = *hnsw.node_levels.iter().max().unwrap();
        assert!(
            (1..=8).contains(&max_lvl),
            "expected max level in [1, 8], got {max_lvl}"
        );
    }

    #[test]
    fn neighbor_caps_respected_on_layer_0() {
        let (vs, _) = build_corpus(500, 16);
        let params = HnswParams::new(8, 100, 50);
        let hnsw = Hnsw::build(&vs, params);
        let cap_l0 = 2 * params.m;
        for (id, neighbors) in hnsw.layers[0].iter().enumerate() {
            assert!(
                neighbors.len() <= cap_l0,
                "node {id} on layer 0 has {} neighbors, cap is {cap_l0}",
                neighbors.len()
            );
        }
    }
}
