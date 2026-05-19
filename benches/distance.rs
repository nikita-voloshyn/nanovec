//! Scalar distance baseline (criterion).
//!
//! Establishes the Phase 3 SIMD baseline: throughput of euclidean, cosine,
//! and dot-product on representative embedding dimensions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nanovec::distance::{cosine, dot_product, euclidean};

const DIMS: &[usize] = &[64, 128, 384, 768, 1536];

fn make_pair(dim: usize) -> (Vec<f32>, Vec<f32>) {
    // Deterministic pseudo-random fill — same seed across all dims so the only
    // varying axis is length, not data distribution.
    let mut a = Vec::with_capacity(dim);
    let mut b = Vec::with_capacity(dim);
    for i in 0..dim {
        a.push((i as f32 * 0.013_37).sin());
        b.push((i as f32 * 0.024_71).cos());
    }
    (a, b)
}

fn bench_distance(c: &mut Criterion) {
    for &dim in DIMS {
        let (a, b) = make_pair(dim);

        let mut g = c.benchmark_group("euclidean");
        g.throughput(Throughput::Elements(dim as u64));
        g.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| euclidean(black_box(&a), black_box(&b)));
        });
        g.finish();

        let mut g = c.benchmark_group("cosine");
        g.throughput(Throughput::Elements(dim as u64));
        g.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| cosine(black_box(&a), black_box(&b)));
        });
        g.finish();

        let mut g = c.benchmark_group("dot_product");
        g.throughput(Throughput::Elements(dim as u64));
        g.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            bencher.iter(|| dot_product(black_box(&a), black_box(&b)));
        });
        g.finish();
    }
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
