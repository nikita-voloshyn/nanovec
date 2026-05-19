//! BoundedMaxHeap micro-benchmark (criterion).
//!
//! Simulates the top-K eviction pattern inside a KNN scan: N pushes, capacity K.
//! Two regimes:
//!   - random:    scores drawn pseudo-randomly (typical case)
//!   - adversarial: scores monotonically decreasing (every push displaces the top)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nanovec::heap::BoundedMaxHeap;

const NS: &[usize] = &[1_000, 10_000];
const KS: &[usize] = &[1, 10, 50];

fn bench_heap_random(c: &mut Criterion) {
    let mut g = c.benchmark_group("heap_random");
    for &n in NS {
        let scores: Vec<f32> = (0..n).map(|i| ((i * 2654435761) & 0xffff) as f32).collect();
        g.throughput(Throughput::Elements(n as u64));
        for &k in KS {
            g.bench_with_input(
                BenchmarkId::new(format!("n={n}"), format!("k={k}")),
                &k,
                |bencher, &k| {
                    bencher.iter(|| {
                        let mut h = BoundedMaxHeap::new(k);
                        for (i, &s) in scores.iter().enumerate() {
                            h.push(black_box(s), i as u64);
                        }
                        h.into_sorted_vec()
                    });
                },
            );
        }
    }
    g.finish();
}

fn bench_heap_adversarial(c: &mut Criterion) {
    let mut g = c.benchmark_group("heap_adversarial");
    for &n in NS {
        // Strictly decreasing scores: every push beats the current max → forces
        // pop + push every iteration after the heap fills up.
        let scores: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        g.throughput(Throughput::Elements(n as u64));
        for &k in KS {
            g.bench_with_input(
                BenchmarkId::new(format!("n={n}"), format!("k={k}")),
                &k,
                |bencher, &k| {
                    bencher.iter(|| {
                        let mut h = BoundedMaxHeap::new(k);
                        for (i, &s) in scores.iter().enumerate() {
                            h.push(black_box(s), i as u64);
                        }
                        h.into_sorted_vec()
                    });
                },
            );
        }
    }
    g.finish();
}

criterion_group!(benches, bench_heap_random, bench_heap_adversarial);
criterion_main!(benches);
