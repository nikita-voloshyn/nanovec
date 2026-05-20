//! NanoVec HNSW vs `instant-distance` HNSW — apples-to-apples comparison.
//!
//! ## Why not Qdrant directly?
//!
//! `qdrant-client` is a gRPC client that connects to a separate Qdrant
//! server process. A side-by-side benchmark through gRPC measures network
//! round-trip + server-side dispatch, not HNSW implementation quality —
//! the comparison would be dominated by the in-process vs out-of-process
//! gap (typically 20–50× depending on payload size). To compare *the index*
//! we need both implementations running in-process on the same data, same
//! CPU, same allocator.
//!
//! `instant-distance` (https://github.com/InstantDomain/instant-distance) is
//! a pure-Rust HNSW that implements the same Malkov & Yashunin algorithm
//! NanoVec does. Both are from-scratch, both use Algorithm 4 heuristic
//! neighbor selection, both run in-process. That makes for a fair test of
//! NanoVec's HNSW implementation against an established public reference.
//!
//! Numbers from this run go into the Phase 7 report.

use std::time::Instant;

use instant_distance::{Builder, Point, Search};
use nanovec::heap::BoundedMaxHeap;
use nanovec::index::{Hnsw, HnswParams};
use nanovec::simd;
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

const DIM: usize = 384;
const N: usize = 10_000;
const N_QUERIES: usize = 200;
const K: usize = 10;

#[derive(Clone)]
struct CosPoint(Vec<f32>);

impl Point for CosPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Match NanoVec exactly: use the same SIMD cosine kernel.
        simd::cosine(&self.0, &other.0)
    }
}

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

fn median(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() {
    println!("# NanoVec HNSW vs instant-distance HNSW");
    println!("# N={N}, dim={DIM}, k={K}, {N_QUERIES} queries, metric=cosine");
    println!();

    // -------- Build shared corpus --------
    println!("Building corpus...");
    let mut vs = VectorStore::new(DIM);
    let mut rs = RecordStore::new();
    let mut points: Vec<CosPoint> = Vec::with_capacity(N);
    let mut values: Vec<u64> = Vec::with_capacity(N);
    for i in 0..N {
        let v = make_vec(i, DIM);
        let off = vs.insert(&v).unwrap();
        rs.insert(format!("doc-{i}"), vec![], off);
        points.push(CosPoint(v));
        values.push(i as u64);
    }

    // Queries shared by both engines.
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|i| make_vec(1_000_000 + i, DIM))
        .collect();

    // -------- Brute-force ground truth + baseline latency --------
    println!("Computing brute-force ground truth...");
    let mut brute_lats: Vec<u128> = Vec::with_capacity(N_QUERIES);
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(N_QUERIES);
    for q in &queries {
        let t = Instant::now();
        let topk = brute_topk(&vs, q, K);
        brute_lats.push(t.elapsed().as_micros());
        truth.push(topk);
    }
    let brute_p50 = median(brute_lats);
    println!("brute-force baseline: p50 search = {brute_p50} µs\n");

    // -------- NanoVec HNSW --------
    let params = HnswParams::new(16, 200, 50);
    let t = Instant::now();
    let nanovec_hnsw = Hnsw::build(&vs, params);
    let nanovec_build_ms = t.elapsed().as_millis();
    let nanovec_bytes = nanovec_hnsw.approx_bytes();

    let mut nv_lats: Vec<u128> = Vec::with_capacity(N_QUERIES);
    let mut nv_hits = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let t = Instant::now();
        let res = nanovec_hnsw.search(&vs, &rs, q, K, None);
        nv_lats.push(t.elapsed().as_micros());
        let ids: Vec<u32> = res.iter().map(|h| h.id as u32).collect();
        nv_hits += truth[i].iter().filter(|t| ids.contains(t)).count();
    }
    let nv_p50 = median(nv_lats);
    let nv_recall = nv_hits as f64 / (N_QUERIES * K) as f64;

    // -------- instant-distance HNSW --------
    let t = Instant::now();
    let id_index = Builder::default()
        .ef_construction(200)
        .ef_search(50)
        .seed(0xC0FFEE)
        .build(points.clone(), values.clone());
    let id_build_ms = t.elapsed().as_millis();

    let mut id_lats: Vec<u128> = Vec::with_capacity(N_QUERIES);
    let mut id_hits = 0usize;
    let mut search = Search::default();
    for (i, q) in queries.iter().enumerate() {
        let qp = CosPoint(q.clone());
        let t = Instant::now();
        let results: Vec<u64> = id_index
            .search(&qp, &mut search)
            .take(K)
            .map(|item| *item.value)
            .collect();
        id_lats.push(t.elapsed().as_micros());
        // instant-distance's ids are our `value` payload (u64 doc index).
        let result_ids: Vec<u32> = results.iter().map(|&v| v as u32).collect();
        id_hits += truth[i].iter().filter(|t| result_ids.contains(t)).count();
    }
    let id_p50 = median(id_lats);
    let id_recall = id_hits as f64 / (N_QUERIES * K) as f64;

    // -------- Report --------
    println!("{}", "-".repeat(78));
    println!(
        "{:>20} | {:>10} {:>10} {:>10} {:>10}",
        "Implementation", "build_ms", "p50_us", "speedup", "recall@10"
    );
    println!("{}", "-".repeat(78));
    println!(
        "{:>20} | {:>10} {:>10} {:>9.1}× | {:>10.3}",
        "brute-force (ref)", "-", brute_p50, 1.0, 1.000
    );
    println!(
        "{:>20} | {:>10} {:>10} {:>9.1}× | {:>10.3}    mem={} KB",
        "NanoVec HNSW",
        nanovec_build_ms,
        nv_p50,
        brute_p50 as f64 / nv_p50.max(1) as f64,
        nv_recall,
        nanovec_bytes / 1024
    );
    println!(
        "{:>20} | {:>10} {:>10} {:>9.1}× | {:>10.3}",
        "instant-distance",
        id_build_ms,
        id_p50,
        brute_p50 as f64 / id_p50.max(1) as f64,
        id_recall
    );
    println!();
    println!("Done.");
}
