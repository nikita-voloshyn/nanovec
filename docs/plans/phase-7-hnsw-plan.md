# Plan: Phase 7 — Exact → Approximate (HNSW)

## Summary

Реализовать **HNSW** (Hierarchical Navigable Small World) с нуля — третий тип индекса
NanoVec после brute-force и KD-Tree. HNSW даёт `O(log n)` поиск с recall > 0.95 на
произвольной размерности, включая 384-dim embeddings, где KD-Tree бесполезен из-за
curse of dimensionality, а brute-force деградирует при `n > 100k`.

Используем существующие SIMD distance (`cosine_avx2` / `cosine_neon`) — граф строится
поверх той же distance metric, что и brute-force. Auto-dispatch выбирает индекс
по `(dim, n)`: brute-force для маленьких или низкоразмерных, KD-Tree для `dim < 50`,
HNSW для `n > 10k & dim > 50`.

Цель: **recall@10 ≥ 0.95** при **p50 latency < 1 ms** для 1M × 384-dim. Чёткий
бенчмарк recall vs latency vs memory на разных `(M, ef)` параметрах.

После Phase 7: NanoVec покрывает все сценарии от 100 до 10⁶ векторов с одним
интерфейсом — `Index::search(&query, k)`. Stretch-фаза: только если есть реальный
use-case с > 100k embeddings.

---

## Design Decisions (требуют подтверждения)

| # | Решение | Rationale |
|---|---------|-----------|
| D1 | **HNSW реализуется с нуля**, без `hnsw_rs` / `instant-distance` / других crate. | Project rule: zero external indexing libraries. Это тот же принцип, что не использовать FAISS/HNSWLIB на Phase 1. Pedagogical + полный контроль над malloc-pattern. |
| D2 | **Граф = `Vec<Vec<Vec<u32>>>`** (слой → нода → соседи как `u32`-индексы записи в `VectorStore`). | `u32` достаточен (max 4G векторов в одной кэлекции — на порядки больше реальных нужд). Плоский layout с тремя уровнями: дёшево по памяти, кэш-френдли при traversal. |
| D3 | **Distance переиспользуется из `simd::dispatch`**. HNSW не имеет собственной distance — вызывает `simd::cosine(a, b)`. | DRY + автоматический выигрыш от SIMD. Альтернатива (передача `fn` указателя) даёт лишний indirect call в hot loop. |
| D4 | **Уровни — geometric с `m_L = 1 / ln(M)`** (классическая формула из оригинального paper Malkov & Yashunin). | Это математически обосновано: ожидаемое число уровней ≈ `log(N)`, что даёт асимптотику `O(log N)`. Никаких «улучшений» без paper-evidence. |
| D5 | **Параметры по умолчанию: `M=16`, `ef_construction=200`, `ef_search=50`.** Конфигурируются через MCP tool params + env vars. | Эти числа — индустриальный стандарт (FAISS, Qdrant, Weaviate), дают recall > 0.95 на типичных embeddings. Не угадываем. |
| D6 | **Build = O(N · ef_construction · log N)**, single-threaded на Phase 7. Параллельный build — отложен. | Параллельный build требует lock-free skip-list или per-level locks; это ~1500 дополнительных строк. Сначала correctness, потом скорость. Для 1M × 384-dim single-threaded build ~5 минут — приемлемо для one-shot indexing. |
| D7 | **Persistence графа — отсутствует.** Граф пересоздаётся при каждом старте от `VectorStore`. | Соответствует «ephemeral by design». Snapshot/restore — Phase 8+ (если когда-либо). |
| D8 | **Auto-dispatch выбирает индекс на основе `(dim, n)`** при первом search, кешируется в `Collection::active_index`. Перестроение — explicit через новый tool `rebuild_index`. | Не строим HNSW автоматически (5 минут на 1M) — пользователь должен явно решить. По умолчанию остаётся brute-force; HNSW активируется через `rebuild_index(collection, "hnsw")`. |
| D9 | **Heuristic для neighbor selection — `select_neighbors_heuristic`** из paper (не simple). | `select_neighbors_simple` (top-M) даёт меньший recall при том же M; heuristic учитывает «разнообразие» соседей. Реализация на ~50 строк больше, но recall +5–10 п.п. |

---

## Tasks

### Task 1 — `Index` trait + `IndexKind` enum
- **Agent:** core
- **Files:** `src/index/mod.rs`, `src/store/collections.rs`
- **Depends on:** none
- **Acceptance:**
  - Существующие `BruteForce` и `KdTree` реализуют общий trait `Index`
  - `IndexKind::{BruteForce, KdTree, Hnsw { m, ef_construction, ef_search }}` сериализуется в JSON через `serde`
  - Все существующие тесты brute/kdtree зелёные после refactoring'а
  - Failing test `hnsw_index_kind_roundtrips_through_serde` для будущей реализации

**API:**
```rust
pub trait Index: Send + Sync {
    fn search(&self, query: &[f32], k: usize, store: &VectorStore) -> Vec<SearchHit>;
    fn build(&mut self, store: &VectorStore) -> Result<(), IndexError>;
    fn kind(&self) -> IndexKind;
    fn approx_bytes(&self) -> u64;
}

pub enum IndexKind {
    BruteForce,
    KdTree,
    Hnsw { m: usize, ef_construction: usize, ef_search: usize },
}
```

`Collection` получает `active_index: Box<dyn Index>` (вместо текущей prямой brute-force-логики в search).

---

### Task 2 — HNSW data structure scaffold (failing tests)
- **Agent:** core
- **Files:** `src/index/hnsw.rs` (new), `src/index/mod.rs`
- **Depends on:** Task 1
- **Acceptance:** `cargo build` чистый; `cargo test --lib hnsw` запускается с N failing tests:
  - `hnsw_empty_graph_returns_no_results`
  - `hnsw_single_node_search_returns_self_at_distance_zero`
  - `hnsw_inserts_assign_correct_levels` (распределение уровней геометрическое)
  - `hnsw_search_matches_brute_force_within_recall_target` (recall@10 ≥ 0.95 на 10k random vectors)

**Структуры:**
```rust
pub struct Hnsw {
    params: HnswParams,                    // M, ef_construction, ef_search, m_L
    layers: Vec<Vec<Vec<u32>>>,            // [layer][node] -> Vec<neighbor_id>
    node_levels: Vec<u8>,                  // assigned level per node id
    entry_point: Option<u32>,              // top-layer entry
    rng: SmallRng,                          // reproducible for tests
}

pub struct HnswParams { pub m: usize, pub ef_construction: usize, pub ef_search: usize, pub m_l: f64 }
```

**Зависимости в `Cargo.toml`:**
```toml
rand = { version = "0.8", default-features = false, features = ["small_rng"] }
```

---

### Task 3 — Layer assignment + entry point logic
- **Agent:** core
- **Files:** `src/index/hnsw.rs`
- **Depends on:** Task 2
- **Acceptance:** unit тесты зелёные:
  - `level_is_geometric` — 10k samples, χ²-test pass для распределения `floor(-ln(uniform) * m_L)`
  - `entry_point_is_highest_level` — после построения, `entry_point` указывает на ноду с max level (или одну из них)
  - `entry_point_updates_when_higher_level_inserted` — insert ноды level > current entry → entry обновляется

---

### Task 4 — Search: greedy descent + ef-search beam
- **Agent:** core
- **Files:** `src/index/hnsw.rs`
- **Depends on:** Task 3
- **Acceptance:**
  - `search_layer(query, entry, ef)` возвращает top-`ef` ближайших на указанном слое (приоритетная очередь candidates + visited set)
  - `search(query, k)`:
    1. Greedy descent от entry на верхнем слое до layer 1 (ef=1)
    2. Beam search на layer 0 (ef=ef_search)
    3. Возврат top-k из beam
  - Тест `search_recall_on_10k_random` — recall@10 ≥ 0.95 vs brute-force baseline, по 100 random queries

**Hot-path:** `distance(query, candidate)` вызывает `simd::cosine` напрямую — никаких boxed traits в loop.

---

### Task 5 — Insert: neighbor selection (heuristic) + bidirectional linking
- **Agent:** core
- **Files:** `src/index/hnsw.rs`
- **Depends on:** Task 4
- **Acceptance:** все тесты зелёные:
  - `insert_single` — после `hnsw.insert(0)` граф имеет 1 ноду, неприсоединена ни к чему
  - `insert_n_random` — после `hnsw.insert(0..1000)` граф консистентен: каждая нода имеет соседей на каждом своём уровне, `len(neighbors) ≤ M` (или `M_max = 2*M` на layer 0)
  - `select_neighbors_heuristic` — Algorithm 4 из paper (Malkov & Yashunin, 2018), не simple top-M
  - `bidirectional_pruning` — после insert обе стороны связи присутствуют, при превышении M_max применяется heuristic pruning

**Reference:** Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", arXiv:1603.09320, Algorithms 1–4.

---

### Task 6 — `build()` from existing `VectorStore` + `rebuild_index` MCP tool
- **Agent:** core + mcp
- **Files:** `src/index/hnsw.rs`, `src/mcp/tools.rs`
- **Depends on:** Task 5
- **Acceptance:**
  - `Hnsw::build(&store)` — пересоздаёт граф из всех векторов в store; в порядке insertion order; вызывает `insert` для каждого
  - Прогресс-логирование каждые 10k нод через `tracing::info!`
  - MCP tool `rebuild_index(collection, kind, params?)`:
    - `kind = "brute" | "kdtree" | "hnsw"`
    - `params` для HNSW: `{m?: usize, ef_construction?: usize, ef_search?: usize}`
    - Возвращает `{built_at_ms, memory_bytes, recall_estimate}` (recall measured на 100 sampled queries vs brute-force)
  - Integration test через MCP — после `rebuild_index("test", "hnsw")` поиск возвращает ту же top-1 что и brute-force на 95% случаев

---

### Task 7 — Auto-dispatch (selection) + `Collection::active_index`
- **Agent:** core
- **Files:** `src/store/collections.rs`, `src/index/mod.rs`
- **Depends on:** Task 6
- **Acceptance:**
  - `Collection::auto_select_index()` runs after each insert и принимает решение:
    - `n < 1000` → BruteForce (всегда)
    - `dim < 50` → KdTree
    - `n >= 10000 && dim >= 50` → suggest HNSW (но не строит автоматически — пишет `tracing::info!` с recommendation)
  - НЕ перестраивает HNSW автоматически — только если пользователь явно вызвал `rebuild_index`
  - Тест `auto_select_logs_hnsw_recommendation_above_threshold`

**Поведение по умолчанию остаётся consistent с Phase 5:** пользователь не получит сюрприза в виде 5-минутного build'а. HNSW — opt-in.

---

### Task 8 — SIMD distance reuse verification
- **Agent:** simd
- **Files:** `src/index/hnsw.rs`, `benches/hnsw.rs` (new)
- **Depends on:** Task 4
- **Acceptance:**
  - Проверить через `cargo asm nanovec::index::hnsw::search_layer` что distance вызовы инлайнятся в `cosine_avx2` / `cosine_neon` (а не в scalar fallback)
  - Если inlining не происходит — добавить `#[inline]` на distance wrapper'е
  - Criterion bench `hnsw_search_simd_vs_scalar` — search через SIMD distance vs forced scalar (`cargo bench --bench hnsw -- --features force-scalar`) — ожидаемый speedup ≥ 4× (HNSW touches fewer vectors, но distance still hot)

---

### Task 9 — Recall vs Latency vs Memory benchmark
- **Agent:** testing
- **Files:** `benches/hnsw_quality.rs` (new), `docs/plans/phase-7-hnsw-report.md` (new)
- **Depends on:** Task 6
- **Acceptance:** sweep benchmark на 100k синтетических 384-dim векторов (Gaussian random):
  - `M ∈ {8, 16, 32}` × `ef_construction ∈ {100, 200, 400}` × `ef_search ∈ {10, 50, 100, 200}` = 36 точек
  - Для каждой точки измерить: build_time, memory_bytes, recall@10, recall@100, p50/p99 search latency
  - Результаты в таблице в `phase-7-hnsw-report.md`
  - Pareto-frontier identified: `recall vs latency` график (как ASCII или ссылка на PNG)
  - Подтверждение target: существует (M, ef) такая что recall@10 ≥ 0.95 и p50 < 1 ms

---

### Task 10 — Side-by-side comparison vs Qdrant in-memory
- **Agent:** testing
- **Files:** `benches/compare_qdrant.rs` (new), `docs/plans/phase-7-hnsw-report.md`
- **Depends on:** Task 9
- **Acceptance:**
  - Spawn Qdrant in-memory mode (`qdrant_client` crate, в `[dev-dependencies]` с feature flag `--features qdrant-compare`)
  - Загрузить одинаковый корпус: 100k × 384-dim, M=16, ef=200
  - Измерить: build_time, search p50/p99, memory_bytes, recall vs ground-truth (brute-force)
  - Результаты в виде таблицы в `phase-7-hnsw-report.md` — никаких маркетинговых заявлений, только числа

**Принципиально не зависимо от MCP** — этот бенчмарк только про индекс vs индекс на одинаковых данных.

---

### Task 11 — Property-based tests
- **Agent:** testing
- **Files:** `src/index/hnsw.rs` (proptests в `#[cfg(test)] mod tests`)
- **Depends on:** Task 6
- **Acceptance:** proptest проверки:
  - `prop_hnsw_recall_above_threshold` — на random vectors любой dimension ∈ [16, 768], n ∈ [100, 10000], HNSW recall@10 ≥ 0.85 (более мягкий чем prod target, чтобы не флакало)
  - `prop_hnsw_consistent_under_insertion_order` — два графа, построенные insertions [v0..vN] и shuffle([v0..vN]), оба дают recall ≥ 0.85
  - `prop_hnsw_distance_matches_simd` — для пары случайных векторов, distance внутри HNSW search ≡ `simd::cosine(a, b)` ± 1e-5
  - `prop_hnsw_empty_after_clear` — после clear() граф пустой, search возвращает []

---

### Task 12 — Documentation
- **Agent:** docs
- **Files:** `docs/components/hnsw.md` (new), `docs/components/index-dispatch.md` (new), `docs/coverage.md`, `docs/ROADMAP.md`, `CHANGELOG.md`, `README.md`, `docs/plans/phase-7-hnsw-report.md`
- **Depends on:** Task 1–11
- **Acceptance:**
  - `hnsw.md`: алгоритм (с ASCII-диаграммой слоёв), параметры M/ef, accept-tradeoffs, ссылка на paper
  - `index-dispatch.md`: дерево решений «когда какой индекс», c числами из Task 9
  - `phase-7-hnsw-report.md`: финальная таблица из Task 9 + 10, deviations, follow-ups
  - `ROADMAP.md`: Phase 7 → ✅ Complete
  - `README.md`: новая секция «Index types» с decision table

---

## Final Validation

```bash
cargo fmt --check && \
cargo clippy --all-targets --all-features -- -D warnings && \
cargo test --all-features && \
cargo bench --bench hnsw_quality && \
cargo bench --bench compare_qdrant --features qdrant-compare
```

**Critical pass criteria:**
1. Найдена (M, ef) комбинация дающая recall@10 ≥ 0.95 при p50 < 1 ms на 1M × 384-dim
2. NanoVec HNSW в пределах 2× от Qdrant по latency на одинаковом сценарии (без cluster mode)
3. Все proptest свойства зелёные на CI

---

## Out of Scope (явно отложено)

- **Parallel build** — single-threaded build достаточен для первой версии (D6)
- **Persistence графа** — нарушает ephemeral принцип
- **Quantization (scalar / PQ / binary)** — Phase 8 если будет нужда уменьшить memory footprint
- **Filtering во время HNSW search** — текущая семантика: post-filter после top-k retrieval; pre-filter (как у Qdrant payload index) — отложено
- **GPU acceleration** — CPU-only по дизайну, GPU выводит из «zero deps» и «small binary»
- **Dynamic delete с tombstones** — текущая реализация поддерживает delete через swap-remove + rebuild; in-place delete с tombstone marking — Phase 8

---

## Risks

| Risk | Mitigation |
|------|------------|
| Build на 1M × 384 занимает > 10 минут | Прогресс-логирование; пользователь видит что не висит. Параллельный build — Phase 8 если станет реальной болью |
| Recall ниже target на каких-то edge-corpus | Sweep benchmark (Task 9) находит работающие параметры; если 0.95 недостижим — публикуем честно 0.90 как floor |
| Qdrant comparison показывает 5× отставание | Принимаем — NanoVec не позиционируется как «лучше Qdrant» (см. слайд 14 презентации). Цель — «в той же лиге» |
| HNSW memory > brute-force в 4× | Документируем в `index-dispatch.md`; auto-dispatch не активирует HNSW автоматически — opt-in |
