//! Phase 7 follow-up — big real-corpus HNSW stress test.
//!
//! The 200-sentence test (`hnsw_real_corpus.rs`) hit recall@10 = 1.000
//! everywhere, which doesn't actually exercise the approximate-vs-exact
//! tradeoff that HNSW is supposed to demonstrate. To see meaningful recall
//! numbers we need a corpus where (a) the embedder has enough room to map
//! sentences into a dense, partially-overlapping space, and (b) `ef_search`
//! values that are deliberately too small force HNSW to miss neighbors.
//!
//! This test:
//!   1. Generates ~10k diverse English sentences by combining real
//!      templates × subject/verb/object/context vocabulary. Each sentence
//!      is unique; the embeddings span a realistic semantic spread (not
//!      just paraphrases of one cluster).
//!   2. Embeds all of them with the production `all-MiniLM-L6-v2`. This is
//!      the expensive step (~7 minutes on Apple Silicon CPU).
//!   3. For 30 hand-picked query strings, computes brute-force ground truth
//!      and HNSW recall at multiple `ef_search` values.
//!   4. Prints a recall vs ef_search × latency table.
//!
//! Marked `#[ignore]` — run explicitly:
//!
//!   cargo test --release --test integration hnsw_stress -- --ignored --nocapture --test-threads=1
//!
//! Numbers from this run are pasted into the project presentation.

use std::time::Instant;

use nanovec::embed::Embedder;
use nanovec::heap::BoundedMaxHeap;
use nanovec::index::{Hnsw, HnswParams};
use nanovec::simd;
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

// Vocabulary — each axis intentionally diverse so the combined sentences
// span topics, not just paraphrases.

const SUBJECTS: &[&str] = &[
    "The engineer",
    "A curious child",
    "Our team",
    "The traveler",
    "An elderly couple",
    "The musician",
    "A young scientist",
    "The chef",
    "A weather forecaster",
    "The investor",
];

const VERBS: &[&str] = &[
    "discovered",
    "tested",
    "rebuilt",
    "rejected",
    "celebrated",
    "documented",
    "investigated",
    "released",
    "criticized",
    "perfected",
];

const OBJECTS: &[&str] = &[
    "a new algorithm",
    "the prototype design",
    "a critical bug fix",
    "the quarterly report",
    "a fresh recipe",
    "an experimental theory",
    "the security audit",
    "a complex puzzle",
    "the budget proposal",
    "an obscure manuscript",
];

const CONTEXTS: &[&str] = &[
    "during the early morning hours",
    "right after lunch",
    "throughout the long evening",
    "while a thunderstorm raged outside",
    "between two scheduled meetings",
    "before the holiday weekend",
    "after the major announcement",
    "in the middle of the conference",
    "shortly before the deadline",
    "as the autumn leaves fell",
];

const EXTRAS: &[&str] = &[
    "and shared the findings with colleagues.",
    "without losing focus on the larger goal.",
    "despite repeated objections from management.",
    "to the surprise of every observer.",
    "with help from an unexpected partner.",
    "after months of careful preparation.",
    "while the system gracefully recovered.",
    "because the previous attempt had failed.",
    "even though resources were severely limited.",
    "as the team rallied around a shared vision.",
];

/// 30 free-form query strings, each loosely associated with one of the
/// vocabulary axes (and intentionally NOT identical to any generated
/// sentence — we want semantic, not exact, retrieval).
const QUERIES: &[&str] = &[
    "engineering breakthroughs and prototype testing",
    "scientific discoveries by young researchers",
    "musical performance and creative inspiration",
    "the chef preparing a new dish",
    "weather forecasting during a thunderstorm",
    "investing decisions and quarterly reports",
    "children solving puzzles after school",
    "team rebuilding a complex system",
    "ancient manuscripts discovered by archaeologists",
    "elderly travelers exploring foreign cities",
    "security audits and critical fixes",
    "managing budgets across departments",
    "celebrating milestones after long preparation",
    "graceful system recovery after failure",
    "documentary research on lost civilizations",
    "the autumn weather and falling leaves",
    "investigations published despite objections",
    "deadlines and last minute changes",
    "experimental theories of physics",
    "morning routines of busy professionals",
    "writing technical documentation thoroughly",
    "evening discussions between two colleagues",
    "preparing for the long holiday weekend",
    "rebuilding trust after a misunderstanding",
    "celebrating an unexpected partnership win",
    "scientific announcements that surprised observers",
    "the obscure manuscript found in a library",
    "conferences with packed schedules",
    "criticizing a flawed proposal in public",
    "perfecting a recipe over many tries",
];

fn generate_corpus() -> Vec<String> {
    // SUBJECTS × VERBS × OBJECTS × CONTEXTS × EXTRAS = 10 ^ 5 = 100,000.
    // We dedupe + cap at 10k for runtime sanity.
    let mut texts: Vec<String> = Vec::with_capacity(10_000);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    'outer: for s in SUBJECTS {
        for v in VERBS {
            for o in OBJECTS {
                for c in CONTEXTS {
                    for e in EXTRAS {
                        let t = format!("{s} {v} {o} {c}, {e}");
                        if seen.insert(t.clone()) {
                            texts.push(t);
                            if texts.len() >= 10_000 {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }
    texts
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
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

#[test]
#[ignore = "10k MiniLM embeddings (~7 minutes); run with --ignored --nocapture"]
fn hnsw_stress_10k_real_embeddings() {
    println!("\n[hnsw_stress] generating corpus...");
    let corpus = generate_corpus();
    println!("[hnsw_stress] corpus size: {}", corpus.len());

    println!("[hnsw_stress] loading embedder...");
    let embedder = Embedder::load().expect("embedder must load");
    let dim = embedder.dimension();
    assert_eq!(dim, 384);

    println!("[hnsw_stress] embedding {} sentences...", corpus.len());
    let t0 = Instant::now();
    let mut vs = VectorStore::new(dim);
    let mut rs = RecordStore::new();
    for (i, text) in corpus.iter().enumerate() {
        let v = embedder.embed(text).expect("embed");
        let off = vs.insert(&v).unwrap();
        rs.insert(text.clone(), vec![], off);
        if (i + 1) % 500 == 0 {
            let pct = (i + 1) as f64 / corpus.len() as f64 * 100.0;
            let elapsed = t0.elapsed().as_secs();
            println!(
                "  ...{}/{}  ({pct:>5.1}%, {elapsed}s elapsed)",
                i + 1,
                corpus.len()
            );
        }
    }
    let embed_secs = t0.elapsed().as_secs_f64();
    println!(
        "[hnsw_stress] corpus embedded in {embed_secs:.1}s ({:.1} ms/doc avg)",
        embed_secs * 1000.0 / corpus.len() as f64
    );

    // Embed queries once — we'll reuse the vectors across the sweep.
    println!("[hnsw_stress] embedding {} queries...", QUERIES.len());
    let query_vecs: Vec<Vec<f32>> = QUERIES
        .iter()
        .map(|q| embedder.embed(q).expect("embed query"))
        .collect();

    // Brute-force ground truth + baseline latency.
    println!("[hnsw_stress] computing brute-force ground truth (k=10)...");
    let k = 10;
    let mut brute_lats: Vec<u128> = Vec::with_capacity(QUERIES.len());
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(QUERIES.len());
    for q in &query_vecs {
        let t = Instant::now();
        let topk = brute_topk(&vs, q, k);
        brute_lats.push(t.elapsed().as_micros());
        truth.push(topk);
    }
    let brute_p50 = median(brute_lats);
    println!("[hnsw_stress] brute-force baseline: p50 search = {brute_p50} µs");

    // Build HNSW once with default M / ef_construction, then sweep ef_search.
    // Re-building per ef_search would be ~30s wasted — ef_search is a query-
    // time parameter on existing graphs, but our `Hnsw` stores it at build,
    // so we mutate `params.ef_search` via the public API by rebuilding the
    // graph each time. The trade-off: builds are O(N · ef_construction).
    // Since ef_construction is fixed, every sweep iteration takes roughly
    // the same build time.
    let m = 16;
    let ef_c = 200;
    let ef_search_grid: &[usize] = &[10, 25, 50, 100, 200];

    println!();
    println!(
        "{:>10} | {:>10} {:>10} {:>10} | {:>10}",
        "ef_search", "build_ms", "p50_us", "speedup", "recall@10"
    );
    println!("{}", "-".repeat(64));

    for &ef_s in ef_search_grid {
        let params = HnswParams::new(m, ef_c, ef_s);
        let t = Instant::now();
        let hnsw = Hnsw::build(&vs, params);
        let build_ms = t.elapsed().as_millis();

        let mut lats: Vec<u128> = Vec::with_capacity(QUERIES.len());
        let mut hits = 0usize;
        for (i, q) in query_vecs.iter().enumerate() {
            let t = Instant::now();
            let res = hnsw.search(&vs, &rs, q, k, None);
            lats.push(t.elapsed().as_micros());
            let ids: Vec<u32> = res.iter().map(|h| h.id as u32).collect();
            hits += truth[i].iter().filter(|t| ids.contains(t)).count();
        }
        let p50 = median(lats);
        let speedup = brute_p50 as f64 / p50.max(1) as f64;
        let recall = hits as f64 / (QUERIES.len() * k) as f64;

        println!(
            "{:>10} | {:>10} {:>10} {:>9.1}× | {:>10.3}",
            ef_s, build_ms, p50, speedup, recall
        );
    }

    println!();
    println!(
        "[hnsw_stress] memory of last graph would be ~{} KB",
        (m * 4 * corpus.len() / 1024)
    );
    println!("[hnsw_stress] done — copy the table above into the report.");
}
