//! Phase 7 follow-up — HNSW recall on **real embeddings** (all-MiniLM-L6-v2).
//!
//! The headline Phase 7 sweep hit recall@10 = 1.000 on synthetic sinusoidal
//! vectors. That's plausible-but-suspicious: the sin/cos corpus has hidden
//! structure that HNSW is essentially guaranteed to recover. To produce
//! honest recall numbers we embed real English sentences with the same
//! MiniLM the production server uses, then compare HNSW against brute-force
//! ground truth.
//!
//! Marked `#[ignore]` by default because it downloads the 90 MB model on
//! first run and embeds 200 sentences (~10-20s on CPU). Run explicitly:
//!
//!   cargo test --release --test integration hnsw_real_corpus -- --ignored --nocapture

use nanovec::embed::Embedder;
use nanovec::heap::BoundedMaxHeap;
use nanovec::index::{Hnsw, HnswParams};
use nanovec::simd;
use nanovec::store::record::RecordStore;
use nanovec::store::VectorStore;

/// 200 short English sentences covering everyday topics. Picked so the
/// embedder produces a realistic spread of vectors (not just paraphrases
/// of one cluster).
const CORPUS: &[&str] = &[
    "The dog barks loudly at the mailman every morning.",
    "Cats prefer napping in sunny spots near the window.",
    "She brewed a strong pot of coffee before sunrise.",
    "He prefers green tea over coffee on cold winter days.",
    "The marathon runner trained for six months straight.",
    "Local bakeries open early to sell fresh croissants.",
    "Engineers debugged the production outage all night.",
    "A new bug was filed for the payment processing API.",
    "Quarterly revenue exceeded analyst expectations.",
    "Renewable energy adoption is accelerating worldwide.",
    "Solar panels are now cheaper than coal in many regions.",
    "Wind turbines were installed along the rocky coast.",
    "The cat chased a red laser pointer across the floor.",
    "Puppies need consistent training and lots of patience.",
    "Hiking the alpine trail takes about eight hours.",
    "She mapped the mountain ridge with a small drone.",
    "Astronomers spotted a new exoplanet last week.",
    "The telescope captured rare gamma ray bursts.",
    "Black holes warp spacetime around their event horizon.",
    "Quantum computers struggle with decoherence noise.",
    "The chef tasted the soup and added more salt.",
    "Sourdough bread requires a long fermentation process.",
    "Pizza dough should rest overnight for the best texture.",
    "He grilled the salmon with lemon and rosemary.",
    "Vegan recipes are popular among health-conscious millennials.",
    "Investors rushed to buy the new tech IPO at opening.",
    "Stock markets opened with a sharp decline this morning.",
    "Bond yields rose sharply after the inflation report.",
    "Cryptocurrencies experienced wild swings last quarter.",
    "Bitcoin reached a new all time high last weekend.",
    "Machine learning models predict house prices reasonably well.",
    "Neural networks excel at pattern recognition in images.",
    "Transformer models revolutionized natural language processing.",
    "GPT-style language models can generate coherent essays.",
    "Reinforcement learning trains agents through trial and error.",
    "Football fans cheered when the team scored a late goal.",
    "Basketball games end with a flurry of three pointers.",
    "The soccer match went into extra time and penalties.",
    "Tennis players practice serves for hours every day.",
    "Golf requires precision and an even temperament.",
    "Hospitals expanded their emergency room capacity.",
    "Doctors recommend daily exercise for cardiovascular health.",
    "Vaccines reduced childhood mortality dramatically.",
    "Medical research uncovered new treatments for diabetes.",
    "Surgery for the broken arm went smoothly yesterday.",
    "Climate change is altering global weather patterns.",
    "Hurricanes are becoming more intense due to warming oceans.",
    "Wildfires devastated entire neighborhoods this summer.",
    "Droughts threaten crop yields across the continent.",
    "Glaciers are retreating faster than scientists predicted.",
    "Streaming services produced more original shows last year.",
    "Movie theaters struggled to fill seats after the pandemic.",
    "The film won an Oscar for best cinematography.",
    "Television writers walked off in a major strike.",
    "Indie game developers released a surprise hit on Steam.",
    "Kids love watching cartoons on Saturday mornings.",
    "Reading novels improves vocabulary and empathy.",
    "Libraries host weekly book clubs for retirees.",
    "Booksellers reported strong demand for biographies.",
    "Poetry slams attract diverse audiences in big cities.",
    "Trains depart from the central station every fifteen minutes.",
    "Highways were jammed because of the holiday weekend.",
    "Electric vehicles outsold gasoline cars in Norway.",
    "Bicycle infrastructure expanded throughout the downtown core.",
    "Airlines reduced fuel consumption with newer aircraft.",
    "Construction crews paved the new bridge overnight.",
    "Architects designed a sustainable office building.",
    "Skyscrapers in the financial district reach above the clouds.",
    "Renovations to the old theater preserved its facade.",
    "Heritage homes need careful restoration work.",
    "Universities reopened for the fall semester yesterday.",
    "Online courses gained popularity during the lockdown.",
    "Researchers published a paper on novel materials.",
    "PhD students presented posters at the conference.",
    "Math olympiads attract talented teenagers worldwide.",
    "The startup raised seed funding from a top VC.",
    "Series B rounds dried up across the SaaS sector.",
    "Founders pitched their idea to angel investors.",
    "Acquisitions in the AI space accelerated last month.",
    "Layoffs swept through several tech giants.",
    "Spices like turmeric and cumin define Indian curries.",
    "Sushi chefs train for years before serving customers.",
    "Mexican street tacos use simple but fresh ingredients.",
    "Italian pasta is best with very few ingredients.",
    "French pastries require patience and butter quality.",
    "Concerts returned to packed venues after the pandemic.",
    "Vinyl records sold more copies than CDs last year.",
    "Jazz bars host live music every Friday night.",
    "Symphony orchestras rely on philanthropic support.",
    "Rock festivals attract fans from many countries.",
    "Yoga improves flexibility and reduces stress levels.",
    "Meditation apps gained millions of new subscribers.",
    "Therapists are reporting longer waiting lists than ever.",
    "Mindfulness practices help with anxiety management.",
    "Sleep hygiene affects mood and cognitive performance.",
    "Recycling rates increased after the new policy.",
    "Plastic pollution chokes the world's oceans.",
    "Compost piles convert food scraps into rich soil.",
    "Cities encourage residents to reduce single use plastics.",
    "Reusable shopping bags became mandatory in stores.",
    "Coding bootcamps train career switchers in twelve weeks.",
    "Open source projects depend on volunteer maintainers.",
    "Linux distributions vary in package managers and defaults.",
    "Containers let developers ship reproducible environments.",
    "Kubernetes manages workloads across many machines.",
    "Photography hobbyists capture wildlife at dawn.",
    "Painters experiment with new pigments and media.",
    "Sculptors carved marble blocks in the Renaissance.",
    "Modern art divides critics and casual viewers.",
    "Museums display priceless artifacts from ancient civilizations.",
    "Backpackers explored the rainforest with local guides.",
    "Hostels offer affordable beds for budget travelers.",
    "Cruises sail the Mediterranean during summer months.",
    "Safari tours take visitors close to wild lions.",
    "Camping under the stars feels deeply restorative.",
    "Politicians debated the new tax bill for hours.",
    "Voters head to the polls on the first Tuesday.",
    "Polls suggest a close race in the swing states.",
    "Court rulings reshape policy across many states.",
    "Diplomats negotiated a ceasefire after months of talks.",
    "Insurance premiums rose after the recent disaster.",
    "Mortgage rates climbed for the third week running.",
    "Banks tightened lending standards last quarter.",
    "Credit scores affect access to affordable loans.",
    "Personal finance bloggers share budgeting tips daily.",
    "Beekeepers worry about declining honeybee populations.",
    "Pollinators are essential for many crop species.",
    "Butterflies migrate thousands of miles each year.",
    "Honey production dropped after the long winter.",
    "Wildflowers bloomed across the open meadow.",
    "Coral reefs are bleaching faster than ever before.",
    "Marine biologists track whale migration patterns.",
    "Plastic pollution harms sea turtles and dolphins.",
    "Submarines explored the deepest ocean trenches.",
    "Octopuses are remarkably clever problem solvers.",
    "Robots performed precise surgery in the operating room.",
    "Self driving cars are still mastering edge cases.",
    "Drones inspect bridges that humans cannot reach safely.",
    "Industrial robots assemble cars on the factory floor.",
    "Humanoid robots learned to balance on uneven terrain.",
    "Programmers debate tabs versus spaces forever.",
    "Code reviews catch defects before they reach production.",
    "Static analysis tools find subtle memory bugs.",
    "Continuous integration runs tests on every commit.",
    "Pair programming helps junior engineers learn faster.",
    "Stargazing requires a clear night and dark skies.",
    "Constellations are easier to spot far from city lights.",
    "Eclipses fascinate observers in their narrow paths.",
    "Comets streak across the predawn sky.",
    "Meteor showers peak in mid August each year.",
    "Coffee shops are noisy in the late morning rush.",
    "Tea ceremonies follow centuries of refined tradition.",
    "Hot chocolate is comforting on snowy afternoons.",
    "Smoothies blend fruit, yogurt, and a little ice.",
    "Bubble tea shops expanded across many neighborhoods.",
    "Marathons attract amateur runners from many backgrounds.",
    "Triathletes train in three disciplines simultaneously.",
    "Cyclists tackle steep mountain stages in famous races.",
    "Swimmers practice flip turns to save crucial seconds.",
    "Climbers ascend sheer rock faces with rope teams.",
    "Spring gardens burst with tulips and daffodils.",
    "Tomatoes grow best in warm sunny conditions.",
    "Composting kitchen scraps reduces household waste.",
    "Indoor plants improve air quality in apartments.",
    "Bonsai trees require patient and careful pruning.",
    "Cleaning gutters in autumn prevents winter damage.",
    "Repairing leaky pipes saves water and money.",
    "Insulating attics improves home heating efficiency.",
    "Replacing old windows lowers energy bills significantly.",
    "Painting walls a light color brightens small rooms.",
    "Babies start crawling around six or seven months.",
    "Toddlers love repeating new words endlessly.",
    "Teenagers crave independence and peer connection.",
    "Parents juggle work, school, and family activities.",
    "Grandparents share stories from a different era.",
    "Foreign languages take years of consistent practice.",
    "Bilingual children switch effortlessly between languages.",
    "Subtitles help language learners with idioms.",
    "Pronunciation requires both listening and mimicking.",
    "Translation apps still miss cultural nuance often.",
    "Birds migrate south as the days grow shorter.",
    "Robins return as the first sign of spring.",
    "Owls hunt mice silently after sunset.",
    "Hummingbirds hover near tubular flowers each summer.",
    "Penguins huddle together to survive the antarctic winter.",
    "Magic tricks rely on misdirection and practice.",
    "Card games span every culture and era.",
    "Board games returned to popularity during the pandemic.",
    "Puzzles relax the mind on quiet afternoons.",
    "Chess masters analyze classical openings for years.",
    "Wedding planners coordinate every detail of the day.",
    "Anniversaries bring couples back to their favorite spots.",
    "Birthdays become more reflective with passing years.",
    "Funerals gather distant relatives in shared mourning.",
    "Graduation ceremonies celebrate years of hard work.",
    "Volcanoes erupt with little warning sometimes.",
    "Earthquakes shake major cities along fault lines.",
    "Tsunamis travel across oceans in just hours.",
    "Floods overwhelm city drains during heavy rains.",
    "Tornadoes tear through small towns in spring.",
];

/// Short query strings unrelated to any single corpus sentence — used to
/// probe semantic search rather than exact-match retrieval.
const QUERIES: &[&str] = &[
    "machine learning and neural networks",
    "coffee shop in the morning",
    "preparing italian food at home",
    "stargazing at night",
    "marathon training routine",
    "renewable energy investment",
    "wildlife in the ocean",
    "language learning advice",
    "raising children",
    "natural disasters",
    "open source software",
    "movies and television",
    "investing in stocks",
    "robotics and automation",
    "pets and animals",
    "musical instruments and concerts",
    "books and libraries",
    "art and painting",
    "travel and tourism",
    "global politics",
];

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

#[test]
#[ignore = "downloads 90MB model + embeds 200 sentences; run with --ignored"]
fn recall_on_real_minilm_embeddings_at_k10() {
    println!("\n[hnsw_real_corpus] loading embedder...");
    let embedder = Embedder::load().expect("embedder must load (network needed on cold cache)");
    let dim = embedder.dimension();
    assert_eq!(dim, 384);

    println!(
        "[hnsw_real_corpus] embedding {} corpus sentences...",
        CORPUS.len()
    );
    let t0 = std::time::Instant::now();
    let mut vs = VectorStore::new(dim);
    let mut rs = RecordStore::new();
    for (i, text) in CORPUS.iter().enumerate() {
        let v = embedder.embed(text).expect("embed");
        let off = vs.insert(&v).unwrap();
        rs.insert(text.to_string(), vec![], off);
        if (i + 1) % 50 == 0 {
            println!("  ...{}/{}", i + 1, CORPUS.len());
        }
    }
    let embed_ms = t0.elapsed().as_millis();
    println!(
        "[hnsw_real_corpus] corpus embedded in {embed_ms} ms ({:.1} ms/doc)",
        embed_ms as f64 / CORPUS.len() as f64
    );

    // Build HNSW with the default Phase 7 params.
    let t0 = std::time::Instant::now();
    let hnsw = Hnsw::build(&vs, HnswParams::default());
    let build_ms = t0.elapsed().as_millis();
    println!("[hnsw_real_corpus] HNSW built in {build_ms} ms");

    // For each query, compute brute-force ground truth + HNSW result, then
    // measure recall@10 overlap.
    let k = 10;
    let mut total_hits = 0usize;
    let mut total_brute_us = 0u128;
    let mut total_hnsw_us = 0u128;
    for q_text in QUERIES {
        let q_vec = embedder.embed(q_text).expect("embed query");

        let t = std::time::Instant::now();
        let truth = brute_topk(&vs, &q_vec, k);
        total_brute_us += t.elapsed().as_micros();

        let t = std::time::Instant::now();
        let hits = hnsw.search(&vs, &rs, &q_vec, k, None);
        total_hnsw_us += t.elapsed().as_micros();
        let ids: Vec<u32> = hits.iter().map(|h| h.id as u32).collect();
        let overlap = truth.iter().filter(|t| ids.contains(t)).count();

        let pct = (overlap as f64 / k as f64) * 100.0;
        // Top-1 sanity-print so we see *what* HNSW retrieves on real text.
        let top1 = hits.first().map(|h| h.text.as_str()).unwrap_or("(none)");
        println!("  recall@{k} = {overlap}/{k} ({pct:>5.1}%)  query={q_text:?}  top1={top1:?}");
        total_hits += overlap;
    }

    let recall = total_hits as f64 / (QUERIES.len() * k) as f64;
    let avg_brute_us = total_brute_us as f64 / QUERIES.len() as f64;
    let avg_hnsw_us = total_hnsw_us as f64 / QUERIES.len() as f64;
    let speedup = avg_brute_us / avg_hnsw_us.max(1.0);

    println!();
    println!(
        "[hnsw_real_corpus] N={} corpus, {} queries, k={k}, dim=384",
        CORPUS.len(),
        QUERIES.len()
    );
    println!(
        "[hnsw_real_corpus] overall recall@10 = {recall:.3}  ({total_hits}/{})",
        QUERIES.len() * k
    );
    println!(
        "[hnsw_real_corpus] avg search latency: brute={avg_brute_us:.0} µs  hnsw={avg_hnsw_us:.0} µs  ({speedup:.1}× speedup)"
    );

    // On 200 short sentences and the default ef_search=50, recall should be
    // very high (>= 0.9). 200 is small enough that brute is fast too — the
    // value of this test is the *recall number*, not the latency win.
    assert!(
        recall >= 0.85,
        "recall@10 on real MiniLM embeddings = {recall:.3}, expected >= 0.85"
    );
}
