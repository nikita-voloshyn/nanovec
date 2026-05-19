//! KD-Tree vs BruteForce KNN bench (criterion).
//!
//! Axes:
//!   - dim = 8  -> KD-Tree should win significantly
//!   - dim = 384 -> KD-Tree should NOT be used (curse of dimensionality:
//!     axis-aligned pruning bounds become loose enough that nearly every
//!     subtree is visited, so the tree degenerates to a worse-than-linear
//!     scan due to recursion overhead). The bench documents this empirically.
//!
//!   - N = 1_000, 10_000
//!   - K = 10
//!
//! A separate group benchmarks KdTree construction time alone (dim = 8).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nanovec::distance::Metric;
use nanovec::index::brute::BruteForce;
use nanovec::index::kdtree::KdTree;
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

const CORPUS_SIZES: &[usize] = &[1_000, 10_000];
const K: usize = 10;

fn make_vec(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| (((seed * 31 + i) as f32) * 0.013_37).sin())
        .collect()
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

fn bench_search_dim(c: &mut Criterion, dim: usize) {
    let mut g = c.benchmark_group(format!("search_dim={dim}_k={K}"));
    for &n in CORPUS_SIZES {
        let (vs, rs) = build_corpus(n, dim);
        let tree = KdTree::build(&vs, &rs);
        let query = make_vec(n + 1, dim);

        g.bench_with_input(BenchmarkId::new("brute", n), &n, |bencher, _| {
            bencher.iter(|| {
                BruteForce::search(
                    black_box(&vs),
                    black_box(&rs),
                    black_box(&query),
                    K,
                    Metric::Euclidean,
                    None,
                )
            });
        });

        g.bench_with_input(BenchmarkId::new("kdtree", n), &n, |bencher, _| {
            bencher
                .iter(|| tree.search(black_box(&vs), black_box(&rs), black_box(&query), K, None));
        });
    }
    g.finish();
}

fn bench_build_kdtree(c: &mut Criterion) {
    let dim = 8;
    let mut g = c.benchmark_group(format!("kdtree_build_dim={dim}"));
    for &n in CORPUS_SIZES {
        let (vs, rs) = build_corpus(n, dim);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| KdTree::build(black_box(&vs), black_box(&rs)));
        });
    }
    g.finish();
}

fn bench_all(c: &mut Criterion) {
    bench_search_dim(c, 8);
    bench_search_dim(c, 384);
    bench_build_kdtree(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
