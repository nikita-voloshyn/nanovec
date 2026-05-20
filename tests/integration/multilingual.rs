//! Verification of the multilingual embedder swap.
//!
//! Phase 2 used `all-MiniLM-L6-v2` (English-only); the ScootGo FAQ
//! benchmark in `docs/MULTILINGUAL-BENCHMARK.md` recorded 0% cross-lingual
//! recall — PL/UK queries did not surface semantically equivalent EN
//! answers. The Phase 7 follow-up swaps the default model to
//! `paraphrase-multilingual-MiniLM-L12-v2`.
//!
//! This test embeds the same English sentence + its PL/UK translations,
//! then asserts that cross-lingual cosine similarity is meaningfully
//! higher than random. It's the smallest possible smoke test that proves
//! the new model actually delivers what the old one couldn't.
//!
//! Marked `#[ignore]` — first run downloads ~120 MB and embeds 7
//! sentences (~3-5 seconds). Run explicitly:
//!
//!   cargo test --release --test integration multilingual -- --ignored --nocapture

use nanovec::embed::Embedder;
use nanovec::simd;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    // Vectors are L2-normalised by the embedder; cosine_similarity = dot.
    1.0 - simd::cosine(a, b)
}

#[test]
#[ignore = "downloads ~120 MB multilingual model on first run"]
fn cross_lingual_alignment_is_non_zero() {
    let embedder = Embedder::load().expect("embedder must load");
    println!("[multilingual] model = {}", embedder.model_name());

    // Three semantic groups — each carries the same idea in EN/PL/UK.
    let groups: [(&str, [&str; 3]); 4] = [
        (
            "speed",
            [
                "How fast does the scooter go?",
                "Jak szybko jeździ hulajnoga?",
                "Як швидко їздить самокат?",
            ],
        ),
        (
            "payment",
            [
                "How do I pay for the ride?",
                "Jak zapłacić za przejazd?",
                "Як заплатити за поїздку?",
            ],
        ),
        (
            "helmet",
            [
                "Do I need to wear a helmet?",
                "Czy muszę nosić kask?",
                "Чи потрібно носити шолом?",
            ],
        ),
        (
            "battery",
            [
                "How long does the battery last?",
                "Jak długo trzyma bateria?",
                "Як довго тримає батарея?",
            ],
        ),
    ];

    println!();
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10}",
        "topic", "EN<->PL", "EN<->UK", "PL<->UK", "min"
    );
    println!("{}", "-".repeat(56));

    let mut all_intra_sims: Vec<f32> = Vec::new();
    let mut all_inter_sims: Vec<f32> = Vec::new();

    let group_vecs: Vec<(String, [Vec<f32>; 3])> = groups
        .iter()
        .map(|(name, [e, p, u])| {
            let ev = embedder.embed(e).expect("embed en");
            let pv = embedder.embed(p).expect("embed pl");
            let uv = embedder.embed(u).expect("embed uk");
            (name.to_string(), [ev, pv, uv])
        })
        .collect();

    for (topic, [en, pl, uk]) in &group_vecs {
        let en_pl = cosine_sim(en, pl);
        let en_uk = cosine_sim(en, uk);
        let pl_uk = cosine_sim(pl, uk);
        let intra_min = en_pl.min(en_uk).min(pl_uk);
        println!("{topic:>10} {en_pl:>10.3} {en_uk:>10.3} {pl_uk:>10.3} {intra_min:>10.3}");
        all_intra_sims.push(en_pl);
        all_intra_sims.push(en_uk);
        all_intra_sims.push(pl_uk);
    }

    // Inter-topic baseline: random pairing across different topics — this is
    // the "noise floor" the intra-topic similarities should beat.
    for i in 0..group_vecs.len() {
        for j in 0..group_vecs.len() {
            if i == j {
                continue;
            }
            for li in 0..3 {
                for lj in 0..3 {
                    let sim = cosine_sim(&group_vecs[i].1[li], &group_vecs[j].1[lj]);
                    all_inter_sims.push(sim);
                }
            }
        }
    }

    let intra_avg = all_intra_sims.iter().sum::<f32>() / all_intra_sims.len() as f32;
    let inter_avg = all_inter_sims.iter().sum::<f32>() / all_inter_sims.len() as f32;
    let intra_min = all_intra_sims.iter().copied().fold(f32::INFINITY, f32::min);

    println!();
    println!("[multilingual] avg intra-topic cross-lingual similarity: {intra_avg:.3}");
    println!("[multilingual] avg inter-topic random-pair similarity:   {inter_avg:.3}");
    println!("[multilingual] worst intra-topic pair similarity:        {intra_min:.3}");

    // The Phase 2 model gave ~0 cross-lingual signal. The multilingual model
    // should at least clear inter-topic noise by a meaningful margin.
    assert!(
        intra_avg > inter_avg + 0.10,
        "intra-topic ({intra_avg:.3}) should exceed inter-topic ({inter_avg:.3}) by > 0.10"
    );
    assert!(
        intra_min > 0.20,
        "worst intra-topic pair is {intra_min:.3} — below 0.20 means multilingual transfer failed"
    );
}
