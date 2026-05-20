//! Direct SIMD-vs-scalar comparison to capture the real speedup ratio on the
//! current host (NEON on Apple Silicon, AVX2 on x86_64).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nanovec::simd as dispatched;
use nanovec::simd::scalar;

const DIMS: &[usize] = &[128, 384, 768, 1536];

fn make_pair(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(dim);
    let mut b = Vec::with_capacity(dim);
    for i in 0..dim {
        a.push((i as f32 * 0.013_37).sin());
        b.push((i as f32 * 0.024_71).cos());
    }
    (a, b)
}

fn bench(c: &mut Criterion) {
    for &dim in DIMS {
        let (a, b) = make_pair(dim);

        // ----- dot_product -----
        let mut g = c.benchmark_group(format!("dot_product/dim={dim}"));
        g.bench_function(BenchmarkId::new("scalar", dim), |bn| {
            bn.iter(|| scalar::dot_product(black_box(&a), black_box(&b)))
        });
        g.bench_function(BenchmarkId::new("simd", dim), |bn| {
            bn.iter(|| dispatched::dot_product(black_box(&a), black_box(&b)))
        });
        g.finish();

        // ----- euclidean -----
        let mut g = c.benchmark_group(format!("euclidean/dim={dim}"));
        g.bench_function(BenchmarkId::new("scalar", dim), |bn| {
            bn.iter(|| scalar::euclidean(black_box(&a), black_box(&b)))
        });
        g.bench_function(BenchmarkId::new("simd", dim), |bn| {
            bn.iter(|| dispatched::euclidean(black_box(&a), black_box(&b)))
        });
        g.finish();

        // ----- cosine -----
        let mut g = c.benchmark_group(format!("cosine/dim={dim}"));
        g.bench_function(BenchmarkId::new("scalar", dim), |bn| {
            bn.iter(|| scalar::cosine(black_box(&a), black_box(&b)))
        });
        g.bench_function(BenchmarkId::new("simd", dim), |bn| {
            bn.iter(|| dispatched::cosine(black_box(&a), black_box(&b)))
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
