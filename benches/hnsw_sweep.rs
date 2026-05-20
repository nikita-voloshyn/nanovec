//! Phase 7 — HNSW recall × latency × memory sweep.
//!
//! Not a criterion bench: this is a one-shot measurement that produces a
//! human-readable table on stdout, suitable for pasting into a report. We
//! cycle through a small grid of (M, ef_construction, ef_search) on a
//! 10k × 384-dim synthetic Gaussian-ish corpus and report build_time,
//! search latency p50, memory_bytes, and recall@10 vs brute-force ground truth.
//!
//! Usage:
//!   cargo run --release --bench hnsw_sweep
//!
//! Numbers are recorded by the user when documenting Phase 7 results.

use std::time::Instant;

use nanovec::heap::BoundedMaxHeap;
use nanovec::index::{Hnsw, HnswParams};
use nanovec::simd;
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

const DIM: usize = 384;
const N: usize = 10_000;
const N_QUERIES: usize = 200;
const K: usize = 10;

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

fn median(mut values: Vec<u128>) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() {
    println!("# HNSW sweep — Phase 7 measurement");
    println!("# corpus: N={N}, dim={DIM} (L2-normalised synthetic vectors)");
    println!("# queries: {N_QUERIES}, k = {K}, metric = cosine");
    println!();

    println!("Building corpus...");
    let (vs, rs) = build_corpus(N, DIM);

    // Precompute brute-force ground truth.
    println!("Computing brute-force ground truth + median latency...");
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|i| make_vec(1_000_000 + i, DIM))
        .collect();

    let mut brute_lats = Vec::with_capacity(N_QUERIES);
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(N_QUERIES);
    for q in &queries {
        let t0 = Instant::now();
        let topk = brute_topk(&vs, q, K);
        brute_lats.push(t0.elapsed().as_micros());
        truth.push(topk);
    }
    let brute_p50_us = median(brute_lats);
    println!("brute-force baseline: p50 search = {brute_p50_us} µs\n");

    println!(
        "{:>3} {:>5} {:>5} | {:>10} {:>10} {:>9} | {:>10}",
        "M", "ef_c", "ef_s", "build_ms", "p50_us", "speedup", "recall@10"
    );
    println!("{}", "-".repeat(82));

    let configs = vec![
        (8, 100, 30),
        (8, 100, 100),
        (16, 100, 30),
        (16, 200, 50),
        (16, 200, 100),
        (16, 200, 200),
        (32, 200, 50),
        (32, 400, 100),
        (32, 400, 200),
    ];

    for (m, ef_c, ef_s) in configs {
        let params = HnswParams::new(m, ef_c, ef_s);
        let t0 = Instant::now();
        let hnsw = Hnsw::build(&vs, params);
        let build_ms = t0.elapsed().as_millis();

        let mem_bytes = hnsw.approx_bytes();

        // Search latencies + recall.
        let mut lats = Vec::with_capacity(N_QUERIES);
        let mut hits = 0;
        for (i, q) in queries.iter().enumerate() {
            let t0 = Instant::now();
            let res = hnsw.search(&vs, &rs, q, K, None);
            lats.push(t0.elapsed().as_micros());
            let ids: Vec<u32> = res.iter().map(|h| h.id as u32).collect();
            hits += truth[i].iter().filter(|t| ids.contains(t)).count();
        }
        let p50_us = median(lats);
        let speedup = brute_p50_us as f64 / p50_us.max(1) as f64;
        let recall = hits as f64 / (N_QUERIES * K) as f64;

        println!(
            "{:>3} {:>5} {:>5} | {:>10} {:>10} {:>8.1}× | {:>10.3}    mem={} KB",
            m,
            ef_c,
            ef_s,
            build_ms,
            p50_us,
            speedup,
            recall,
            mem_bytes / 1024
        );
    }

    println!();
    println!("Done.");
}
