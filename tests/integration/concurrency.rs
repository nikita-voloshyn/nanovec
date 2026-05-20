//! Phase 6 — concurrency: verifies the two-level RwLock actually lets readers
//! run in parallel. We exercise the database in raw-vector mode (no embedder
//! load), pre-populate a collection with N synthetic vectors, then race
//! several worker threads doing reads / writes.
//!
//! Each test records its wall-clock numbers so we have **real** speedup data
//! to cite — no fabricated benchmarks.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use nanovec::distance::Metric;
use nanovec::index::brute::BruteForce;
use nanovec::store::database::NanoVecDatabase;

const DIM: usize = 384;
const CORPUS: usize = 5_000;
const QUERIES_PER_THREAD: usize = 200;

fn make_vec(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| (((seed * 31 + i) as f32) * 0.013_37).sin())
        .collect()
}

fn populate(db: &NanoVecDatabase, collection: Option<&str>, n: usize) {
    let guard = db.resolve(collection).unwrap();
    let mut coll = guard.inner.write();
    for i in 0..n {
        let v = make_vec(i, DIM);
        let off = coll.store.insert(&v).unwrap();
        coll.records.insert(format!("doc-{i}"), vec![], off);
    }
}

fn run_search_threads(
    db: Arc<NanoVecDatabase>,
    n_threads: usize,
    queries: usize,
    collection_picker: impl Fn(usize) -> Option<String> + Send + Sync + 'static,
) -> std::time::Duration {
    let picker = Arc::new(collection_picker);
    let counter = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n_threads);
    for t in 0..n_threads {
        let db = Arc::clone(&db);
        let picker = Arc::clone(&picker);
        let counter = Arc::clone(&counter);
        handles.push(std::thread::spawn(move || {
            for q in 0..queries {
                let coll_name = picker(t);
                let guard = db
                    .resolve_readonly(coll_name.as_deref())
                    .expect("resolve")
                    .expect("collection present");
                let coll = guard.inner.read();
                let query = make_vec(q * 7 + t, DIM);
                let _hits = BruteForce::search(
                    &coll.store,
                    &coll.records,
                    &query,
                    10,
                    Metric::Cosine,
                    None,
                );
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.join().expect("worker panicked");
    }
    let elapsed = start.elapsed();
    assert_eq!(counter.load(Ordering::Relaxed), n_threads * queries);
    elapsed
}

#[test]
fn parallel_search_scales_on_same_collection() {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 256));
    populate(&db, Some("default"), CORPUS);

    // Single-thread baseline.
    let t1 = run_search_threads(Arc::clone(&db), 1, QUERIES_PER_THREAD, |_| None);
    // 8 parallel threads on the same collection. With per-collection reader
    // sharing, throughput should scale much better than under a global Mutex.
    let t8 = run_search_threads(Arc::clone(&db), 8, QUERIES_PER_THREAD, |_| None);

    let single_throughput = QUERIES_PER_THREAD as f64 / t1.as_secs_f64();
    let multi_throughput = (8 * QUERIES_PER_THREAD) as f64 / t8.as_secs_f64();
    let scaling = multi_throughput / single_throughput;

    println!(
        "[parallel_same_collection] 1 thread: {:.0} q/s  |  8 threads: {:.0} q/s  |  scaling: {:.2}×",
        single_throughput, multi_throughput, scaling
    );

    // Under a global Mutex, scaling would be ~1.0. With the new RwLock,
    // we expect substantially better. Threshold is conservative to avoid
    // flake on CI; the real number is in the println above.
    assert!(
        scaling > 2.0,
        "expected reader-parallel speedup > 2.0×, got {scaling:.2}× (t1={t1:?}, t8={t8:?})"
    );
}

#[test]
fn parallel_search_scales_across_collections() {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 256));
    // Build 8 collections so different threads hit different collection locks.
    let n_colls = 8;
    let per_coll = CORPUS / n_colls;
    for k in 0..n_colls {
        db.create_collection(format!("c{k}"), DIM).unwrap();
        populate(&db, Some(&format!("c{k}")), per_coll);
    }

    let single = run_search_threads(Arc::clone(&db), 1, QUERIES_PER_THREAD, |_| {
        Some("c0".to_string())
    });
    let multi = run_search_threads(Arc::clone(&db), n_colls, QUERIES_PER_THREAD, move |t| {
        Some(format!("c{}", t % n_colls))
    });

    let single_throughput = QUERIES_PER_THREAD as f64 / single.as_secs_f64();
    let multi_throughput = (n_colls * QUERIES_PER_THREAD) as f64 / multi.as_secs_f64();
    let scaling = multi_throughput / single_throughput;

    println!(
        "[parallel_across_collections] 1 thread (c0): {:.0} q/s  |  {n_colls} threads × {n_colls} colls: {:.0} q/s  |  scaling: {:.2}×",
        single_throughput, multi_throughput, scaling
    );

    assert!(
        scaling > 3.0,
        "expected disjoint-collection speedup > 3.0×, got {scaling:.2}×"
    );
}

#[test]
fn writes_to_different_collections_are_independent() {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 256));
    db.create_collection("a".into(), DIM).unwrap();
    db.create_collection("b".into(), DIM).unwrap();
    populate(&db, Some("a"), 1_000);
    populate(&db, Some("b"), 1_000);

    // Hold a long write lock on collection A.
    let db_a = Arc::clone(&db);
    let writer = std::thread::spawn(move || {
        let guard = db_a.resolve(Some("a")).unwrap();
        let mut coll = guard.inner.write();
        // Simulate a slow insert burst that hogs the write lock on A.
        for i in 1_000..1_500 {
            let v = make_vec(i, DIM);
            let off = coll.store.insert(&v).unwrap();
            coll.records.insert(format!("doc-{i}"), vec![], off);
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    });

    // While the writer is busy on A, read from B and confirm latency stays low.
    let db_b = Arc::clone(&db);
    let reader_start = Instant::now();
    let mut completed = 0;
    while reader_start.elapsed() < std::time::Duration::from_millis(20) {
        let guard = db_b.resolve_readonly(Some("b")).unwrap().unwrap();
        let coll = guard.inner.read();
        let _ = BruteForce::search(
            &coll.store,
            &coll.records,
            &make_vec(42, DIM),
            10,
            Metric::Cosine,
            None,
        );
        completed += 1;
    }
    writer.join().unwrap();

    println!(
        "[write_isolation] reads on B completed in 20 ms while A held a write lock: {completed}"
    );
    assert!(
        completed > 5,
        "expected several B-reads during A-write window, got {completed}"
    );
}

#[test]
fn lru_eviction_drops_coldest_unpinned_collection() {
    // 1 MiB budget. Each 384-dim vector is 1.5 KiB raw + ~6 B of record text;
    // ~650 vectors fit before pressure. We push >700 to force eviction.
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 1));

    db.create_collection("hot".into(), DIM).unwrap();
    db.create_collection("cold".into(), DIM).unwrap();
    db.create_collection("warm".into(), DIM).unwrap();

    // Equal-size base populate so all three have similar approx_bytes.
    populate(&db, Some("hot"), 250);
    populate(&db, Some("cold"), 250);
    populate(&db, Some("warm"), 250);

    // Touch hot + warm many times so `cold` ends up with the smallest tick.
    for _ in 0..50 {
        let _ = db.resolve_readonly(Some("hot")).unwrap().unwrap();
        let _ = db.resolve_readonly(Some("warm")).unwrap().unwrap();
    }

    // Verify we are actually over budget before evicting.
    let used_before = db.recompute_used_bytes();
    println!(
        "[lru_eviction] used before evict: {used_before} bytes, limit {}",
        db.limit_bytes()
    );
    assert!(used_before > db.limit_bytes());

    let evicted = db.evict_until_under_budget();
    println!("[lru_eviction] evicted: {evicted:?}");

    assert!(!evicted.is_empty(), "expected at least one eviction");
    // `cold` has the smallest tick, so it must be among the first evicted.
    assert!(
        evicted.contains(&"cold".to_string()),
        "expected `cold` (smallest tick) to be evicted; got {evicted:?}"
    );
    // Default is pinned (was never auto-materialized here, but if it had been,
    // it should not be in the evicted list).
    assert!(!evicted.contains(&"default".to_string()));
    // Final state must be under budget.
    assert!(db.recompute_used_bytes() <= db.limit_bytes());
}

#[test]
fn pinned_default_collection_survives_eviction() {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 1));
    // Auto-create + populate `default` (pinned).
    populate(&db, None, 400);
    db.create_collection("evictable".into(), DIM).unwrap();
    populate(&db, Some("evictable"), 400);

    let used = db.recompute_used_bytes();
    assert!(
        used > db.limit_bytes(),
        "test setup must exceed budget (used {used}, limit {})",
        db.limit_bytes()
    );

    let evicted = db.evict_until_under_budget();
    let names: Vec<String> = db.snapshot().into_iter().map(|s| s.name).collect();
    println!("[pinned_default] evicted: {evicted:?}, survivors: {names:?}");

    assert!(
        names.contains(&"default".to_string()),
        "default was evicted but should be pinned"
    );
    assert!(
        evicted.contains(&"evictable".to_string()),
        "expected `evictable` to be evicted, got {evicted:?}"
    );
}

#[test]
fn budget_accounting_tracks_inserts() {
    let db = Arc::new(NanoVecDatabase::with_dimension(Metric::Cosine, DIM, 256));
    populate(&db, None, 1_000);
    let used = db.recompute_used_bytes();
    // 1_000 vectors × 384 dim × 4 bytes = 1_536_000 bytes minimum.
    // Plus records' text ("doc-{i}" ≈ 6 bytes avg).
    let vector_bytes = 1_000u64 * 384 * 4;
    assert!(
        used >= vector_bytes && used < vector_bytes * 2,
        "expected used bytes ~{vector_bytes}, got {used}"
    );
    println!("[budget] 1k×384-dim collection accounted as {used} bytes");
}
