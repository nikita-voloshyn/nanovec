//! VectorStore + RecordStore micro-benchmarks (criterion).
//!
//! Captures baseline throughput of:
//!   - VectorStore::insert        (append to flat SoA Vec<f32>)
//!   - VectorStore::get           (offset lookup, slice return)
//!   - RecordStore::insert        (Vec push + HashMap entry)
//!   - RecordStore::get           (HashMap lookup → Vec index)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

const DIMS: &[usize] = &[128, 384, 768];

fn bench_vector_store_insert(c: &mut Criterion) {
    let mut g = c.benchmark_group("vector_store_insert");
    for &dim in DIMS {
        let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        g.throughput(Throughput::Bytes((dim * std::mem::size_of::<f32>()) as u64));
        g.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            // Fresh store each iteration so we measure steady-state append
            // without amortizing over a single Vec growth curve.
            bencher.iter_batched(
                || VectorStore::new(dim),
                |mut store| {
                    store.insert(black_box(&v)).unwrap();
                    store
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_vector_store_get(c: &mut Criterion) {
    let mut g = c.benchmark_group("vector_store_get");
    const N: usize = 10_000;
    for &dim in DIMS {
        let mut store = VectorStore::new(dim);
        for i in 0..N {
            let v: Vec<f32> = (0..dim).map(|j| (i + j) as f32 * 0.1).collect();
            store.insert(&v).unwrap();
        }
        g.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bencher, _| {
            let mut idx = 0usize;
            bencher.iter(|| {
                let v = store.get(black_box(idx)).unwrap();
                idx = (idx + 1) % N;
                v.len()
            });
        });
    }
    g.finish();
}

fn bench_record_store_insert(c: &mut Criterion) {
    c.bench_function("record_store_insert", |bencher| {
        bencher.iter_batched(
            RecordStore::new,
            |mut rs| {
                rs.insert(
                    black_box("doc".to_string()),
                    black_box(vec![]),
                    black_box(0),
                );
                rs
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_record_store_get(c: &mut Criterion) {
    let mut rs = RecordStore::new();
    const N: u64 = 10_000;
    for i in 0..N {
        rs.insert(format!("doc-{i}"), vec![], i as usize);
    }
    c.bench_function("record_store_get", |bencher| {
        let mut id: u64 = 0;
        bencher.iter(|| {
            let r = rs.get(black_box(id));
            id = (id + 1) % N;
            r.map(|r| r.offset)
        });
    });
}

criterion_group!(
    benches,
    bench_vector_store_insert,
    bench_vector_store_get,
    bench_record_store_insert,
    bench_record_store_get,
);
criterion_main!(benches);
