//! Brute-force KNN search bench (criterion).
//!
//! Measures end-to-end top-K search throughput on the scalar baseline so we
//! can later compare Phase 3 SIMD acceleration.
//!
//! Axes:
//!   - N (corpus size):  100, 1k, 10k
//!   - K (top results):  1, 10, 50
//!   - metric:           Euclidean, Cosine, DotProduct
//!
//! Dimension is pinned to 384 (the MiniLM-L6-v2 embedding size used in prod).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nanovec::distance::Metric;
use nanovec::index::brute::BruteForce;
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

const DIM: usize = 384;
const CORPUS_SIZES: &[usize] = &[100, 1_000, 10_000];
const KS: &[usize] = &[1, 10, 50];

fn make_vec(seed: usize) -> Vec<f32> {
    (0..DIM)
        .map(|i| (((seed * 31 + i) as f32) * 0.013_37).sin())
        .collect()
}

fn build_corpus(n: usize) -> (VectorStore, RecordStore) {
    let mut vs = VectorStore::new(DIM);
    let mut rs = RecordStore::new();
    for i in 0..n {
        let v = make_vec(i);
        let off = vs.insert(&v).unwrap();
        rs.insert(format!("doc-{i}"), vec![], off);
    }
    (vs, rs)
}

fn bench_search(c: &mut Criterion) {
    let metrics = [
        ("euclidean", Metric::Euclidean),
        ("cosine", Metric::Cosine),
        ("dot", Metric::DotProduct),
    ];

    for (name, metric) in metrics {
        let mut g = c.benchmark_group(format!("search_{name}"));
        for &n in CORPUS_SIZES {
            let (vs, rs) = build_corpus(n);
            let query = make_vec(n + 1);
            g.throughput(Throughput::Elements(n as u64));
            for &k in KS {
                g.bench_with_input(
                    BenchmarkId::new(format!("n={n}"), format!("k={k}")),
                    &k,
                    |bencher, &k| {
                        bencher.iter(|| {
                            BruteForce::search(
                                black_box(&vs),
                                black_box(&rs),
                                black_box(&query),
                                k,
                                metric,
                                None,
                            )
                        });
                    },
                );
            }
        }
        g.finish();
    }
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
