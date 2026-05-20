//! Phase 7 — HNSW recall + correctness, including property-based tests on a
//! larger 384-dim (embedding-shaped) corpus.

use nanovec::heap::BoundedMaxHeap;
use nanovec::index::{Hnsw, HnswParams};
use nanovec::simd;
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

fn make_vec(seed: usize, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|i| (((seed * 31 + i) as f32) * 0.013_37).sin())
        .collect();
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
    let mut heap = BoundedMaxHeap::new(k);
    for (off, v) in store.iter() {
        heap.push(simd::cosine(query, v), off as u64);
    }
    heap.into_sorted_vec()
        .into_iter()
        .map(|(_, id)| id as u32)
        .collect()
}

fn measure_recall(
    hnsw: &Hnsw,
    store: &VectorStore,
    records: &RecordStore,
    k: usize,
    n_queries: usize,
    query_seed_offset: usize,
) -> f64 {
    let mut hits = 0;
    for q in 0..n_queries {
        let query = make_vec(query_seed_offset + q, store.dimension());
        let truth = brute_topk(store, &query, k);
        let hnsw_hits = hnsw.search(store, records, &query, k, None);
        let ids: Vec<u32> = hnsw_hits.iter().map(|h| h.id as u32).collect();
        let overlap = truth.iter().filter(|t| ids.contains(t)).count();
        hits += overlap;
    }
    hits as f64 / (n_queries * k) as f64
}

#[test]
fn recall_on_2k_384_dim_corpus_meets_target() {
    // 2k vectors at embedding-shaped 384-dim. With M=16/ef_c=200/ef_s=50
    // (default), expect recall@10 ≥ 0.85.
    let (vs, rs) = build_corpus(2_000, 384);
    let hnsw = Hnsw::build(&vs, HnswParams::default());
    let recall = measure_recall(&hnsw, &vs, &rs, 10, 50, 10_000);
    println!("[hnsw_recall_2k_384] recall@10 = {recall:.3} (M=16, ef_c=200, ef_s=50)");
    assert!(recall >= 0.85, "recall@10 = {recall:.3}, target >= 0.85");
}

#[test]
fn recall_improves_with_higher_ef_search() {
    let (vs, rs) = build_corpus(1_000, 128);

    let lo = Hnsw::build(&vs, HnswParams::new(16, 200, 10));
    let hi = Hnsw::build(&vs, HnswParams::new(16, 200, 100));

    let r_lo = measure_recall(&lo, &vs, &rs, 10, 30, 5_000);
    let r_hi = measure_recall(&hi, &vs, &rs, 10, 30, 5_000);
    println!("[hnsw_ef_sensitivity] ef=10 -> {r_lo:.3}, ef=100 -> {r_hi:.3}");

    assert!(
        r_hi + 0.005 >= r_lo,
        "higher ef_search should not be worse: ef=10 gave {r_lo}, ef=100 gave {r_hi}"
    );
}

#[test]
fn hnsw_results_match_simd_cosine() {
    // The HNSW search must use the same SIMD distance as the brute baseline.
    // Verifies the reported `score` is plausible (cosine distance in [0, 2]).
    let (vs, rs) = build_corpus(200, 64);
    let hnsw = Hnsw::build(&vs, HnswParams::new(8, 80, 30));
    let q = make_vec(99_999, 64);
    let hits = hnsw.search(&vs, &rs, &q, 5, None);
    for h in &hits {
        assert!(
            (0.0..=2.001).contains(&h.score),
            "cosine distance out of range: {}",
            h.score
        );
        let stored = vs.get(h.id as usize).unwrap();
        let recomputed = simd::cosine(&q, stored);
        approx::assert_relative_eq!(h.score, recomputed, max_relative = 1e-5);
    }
}

#[test]
fn hnsw_search_with_filter_returns_only_matching() {
    let mut vs = VectorStore::new(8);
    let mut rs = RecordStore::new();
    for i in 0..50 {
        let off = vs.insert(&make_vec(i, 8)).unwrap();
        let bucket = if i % 2 == 0 { "even" } else { "odd" };
        rs.insert(
            format!("doc-{i}"),
            vec![("bucket".to_string(), bucket.to_string())],
            off,
        );
    }
    let hnsw = Hnsw::build(&vs, HnswParams::new(8, 80, 40));
    let q = make_vec(1234, 8);
    let filter = vec![("bucket".to_string(), "even".to_string())];
    let hits = hnsw.search(&vs, &rs, &q, 10, Some(&filter));
    assert!(!hits.is_empty());
    for h in &hits {
        let id: usize = h.text.strip_prefix("doc-").unwrap().parse().unwrap();
        assert!(id.is_multiple_of(2), "filter leaked odd doc {}", h.text);
    }
}
