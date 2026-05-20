# Plan: Phase 6 — Single → Multi-agent

## Summary

Превратить NanoVec из «один процесс, один клиент, stdio» в сервер, обслуживающий
**нескольких клиентов параллельно** с изоляцией кэлекций, контролируемым бюджетом
памяти и LRU-эвикцией. Текущий `Arc<Mutex<NanoVecState>>` заменяется на
`Arc<RwLock<NanoVecDatabase>>` с per-collection блокировками: search (95% нагрузки)
становится reader-параллельным, write остаётся exclusive только в рамках одной
кэлекции. Дополнительно — SSE-транспорт на `axum` рядом со stdio для сетевого
доступа, connection manager с per-client default collection, и hard memory budget
с LRU-эвикцией per-collection.

После Phase 6: два и более MCP-клиентов могут работать одновременно (например,
параллельные subagent'ы в Claude Code), каждый в своей кэлекции, без блокирующих
друг друга lock'ов. Память не растёт безгранично — превышение лимита триггерит
эвикцию наименее используемых записей.

---

## Design Decisions (требуют подтверждения)

| # | Решение | Rationale |
|---|---------|-----------|
| D1 | **`RwLock` на уровне `CollectionMap`, не на отдельной `Collection`.** Структура: `Arc<RwLock<CollectionMap>>` + внутри `Collection` — `RwLock<Collection>` через `DashMap` или `parking_lot::RwLock`. | `CollectionMap` нужно блокировать только при `create_collection` / `drop_collection`. Остальные операции (search, insert, delete внутри одной кэлекции) идут через per-collection lock. Двухуровневая схема: write на map = редкое событие, write на collection = типичная операция. |
| D2 | **`parking_lot::RwLock` вместо `tokio::sync::RwLock`.** | Read-heavy нагрузка; `parking_lot` быстрее и не требует `.await` (handler уже внутри `spawn_blocking`-style контекста rmcp). `tokio::RwLock` справедлив (FIFO), но даёт лишний overhead на типичных коротких critical section. |
| D3 | **SSE-транспорт через `rmcp` + `axum`**, оба транспорта живут параллельно. | rmcp 1.3+ поддерживает SSE «из коробки» через `transport-sse-server`. Stdio остаётся default для локального dev; SSE включается env-флагом `NANOVEC_SSE_ADDR=0.0.0.0:8421`. |
| D4 | **Connection manager — stateless по дизайну.** Per-client `default_collection` хранится в JSON-RPC `meta.connection_id`, который клиент передаёт в каждом запросе; сервер не хранит сессии. | Cessии = state, state = bug-magnet (timeout, eviction, leak). MCP уже stateless по контракту; per-client default — просто параметр, не «сессия». |
| D5 | **Memory budget — глобальный, не per-collection.** Hard cap `NANOVEC_MEMORY_LIMIT_MB` (default: 512 MB). Эвикция выбирает кэлекцию с самым старым `last_accessed`. | Per-collection бюджеты требуют конфигурации, которой клиент не хочет писать. Глобальный лимит + per-collection LRU даёт честное «горячие кэлекции выживают». |
| D6 | **LRU = approximate, не точный.** Каждая `Collection` хранит `AtomicU64 last_accessed_tick` (монотонный счётчик); эвикция выбирает min. Внутри кэлекции эвикции записей нет — выбрасываем целиком кэлекцию. | Точный LRU per-record требует двусвязного списка под lock'ом → contention. Per-collection eviction — простое и предсказуемое поведение: «холодная кэлекция выбрасывается целиком». Альтернатива (per-record) — отложена до Phase 7+. |
| D7 | **Memory accounting — приближённое, не точное.** Считаем только основной cost: `dim * 4 bytes * count` (векторы) + `text.len()` (записи). Игнорируем overhead HashMap, Vec capacity, padding. | Точный sizeof невозможен в Rust без unstable features. Приближение достаточное: ошибка ~10–20%, лимит всё равно «мягкий» (можно превысить на одну операцию). |

---

## Tasks

### Task 1 — Refactor: `NanoVecState` → `NanoVecDatabase` with `RwLock`
- **Agent:** core
- **Files:** `src/mcp/mod.rs`, `src/store/collections.rs`, `src/store/mod.rs`
- **Depends on:** none
- **Acceptance:**
  - Все существующие тесты (`cargo test --all-features`) зелёные после рефакторинга — поведение не меняется, только лок
  - `NanoVecServer` держит `Arc<NanoVecDatabase>` (без внешнего `Mutex`)
  - `NanoVecDatabase` содержит `RwLock<CollectionMap>` + общий `Arc<Embedder>` + `Metric`
  - Каждая `Collection` обёрнута в `RwLock<CollectionInner>` (внутренний тип, наружу публикуем тонкие read/write guards)
  - Добавляется failing test `concurrent_searches_do_not_block_each_other`: два потока запускают search одновременно, оба должны успеть за < 2× одиночного времени (sanity-check, что reader-lock работает)

**API после рефакторинга:**
```rust
pub struct NanoVecDatabase {
    collections: RwLock<HashMap<String, Arc<RwLock<CollectionInner>>>>,
    embedder: Arc<Embedder>,
    metric: Metric,
    budget: MemoryBudget, // Task 5
}

impl NanoVecDatabase {
    pub fn read_collection(&self, name: &str) -> Result<Arc<RwLock<CollectionInner>>, DbError>;
    pub fn write_collection(&self, name: &str) -> Result<Arc<RwLock<CollectionInner>>, DbError>;
    pub fn create_collection(&self, name: String) -> Result<(), DbError>;
    pub fn drop_collection(&self, name: &str) -> Result<(), DbError>;
}
```

`parking_lot` добавляется в `[dependencies]`: `parking_lot = "0.12"`.

---

### Task 2 — Per-collection lock granularity tests (proptest)
- **Agent:** testing
- **Files:** `tests/concurrency.rs` (new)
- **Depends on:** Task 1
- **Acceptance:** все тесты зелёные:
  - `parallel_search_on_same_collection` — N=8 потоков по 1000 запросов к одной кэлекции, результат каждого идентичен sequential baseline
  - `parallel_writes_to_different_collections_are_independent` — поток A пишет в кэлекцию X, поток B пишет в Y; ни один не блокируется > чем на длину write на той же кэлекции
  - `write_blocks_only_target_collection` — поток A держит write-lock на X в `sleep(50ms)`; поток B успевает сделать search на Y за < 5 ms (independence check)
  - `proptest`: при любой последовательности (insert / search / delete) сериализованной в один поток vs распределённой по N потокам — финальное состояние идентично

---

### Task 3 — SSE transport via `rmcp` + `axum`
- **Agent:** mcp
- **Files:** `Cargo.toml`, `src/server/mod.rs`, `src/transport/mod.rs` (new), `src/transport/sse.rs` (new)
- **Depends on:** Task 1
- **Acceptance:**
  - `NANOVEC_SSE_ADDR=127.0.0.1:8421 cargo run` поднимает SSE endpoint на указанном адресе
  - Параллельно stdio продолжает работать (default режим без env var)
  - Integration-тест `tests/sse_smoke.rs`: spawn server в background → `reqwest` коннект к `/sse` → ping/pong через `initialize` JSON-RPC → graceful shutdown

**Зависимости в `Cargo.toml`:**
```toml
rmcp = { version = "1.3", features = ["server", "transport-io", "transport-sse-server", "macros"] }
axum = "0.7"
tower = "0.5"
```

**Структура:**
```rust
// src/transport/mod.rs
pub enum TransportMode { Stdio, Sse(SocketAddr), Both(SocketAddr) }

pub fn select_from_env() -> TransportMode { /* читает NANOVEC_SSE_ADDR */ }

// src/server/mod.rs
pub async fn run() -> anyhow::Result<()> {
    let db = Arc::new(NanoVecDatabase::load().await?);
    match transport::select_from_env() {
        TransportMode::Stdio => run_stdio(db).await,
        TransportMode::Sse(addr) => run_sse(db, addr).await,
        TransportMode::Both(addr) => tokio::try_join!(run_stdio(db.clone()), run_sse(db, addr)).map(|_| ()),
    }
}
```

---

### Task 4 — Connection manager: per-client default collection
- **Agent:** mcp
- **Files:** `src/mcp/mod.rs`, `src/mcp/tools.rs`
- **Depends on:** Task 1, Task 3
- **Acceptance:**
  - Все MCP-tools принимают optional `connection_id: Option<String>` через `meta`-поле JSON-RPC
  - Если клиент передаёт `connection_id="alice"` и `collection=None` → дефолт = `"_conn_alice"` (auto-create по dim embedder'а)
  - Если `connection_id=None` → дефолт = `"default"` (текущее поведение)
  - `tests/connection_isolation.rs`: два клиента `alice` и `bob` индексируют одинаковый текст без явной кэлекции → их данные изолированы (search от `alice` не возвращает записи `bob`)

**Implementation note:** `connection_id` идёт в `_meta` payload'а JSON-RPC (стандартное поле MCP для transport-level metadata). Не путать с TCP-сессией: `connection_id` — это просто строка от клиента, сервер ей доверяет.

---

### Task 5 — Memory budget + accounting
- **Agent:** core
- **Files:** `src/store/budget.rs` (new), `src/store/collections.rs`, `src/mcp/mod.rs`
- **Depends on:** Task 1
- **Acceptance:** все тесты зелёные:
  - `budget_tracks_insertions` — после N инсертов в кэлекцию dim=384 → `db.memory_bytes() == N * 384 * 4 + sum(text.len())` ± 5%
  - `budget_exceeded_returns_error_without_eviction` — при `evict_on_overflow=false` и attempt insert beyond limit → `Err(DbError::BudgetExceeded)`, состояние не меняется
  - `budget_includes_records_and_vectors` — отдельный аккаунтинг для `VectorStore` и `RecordStore`, оба учтены в `total_bytes()`

**API:**
```rust
pub struct MemoryBudget {
    pub limit_bytes: u64,           // hard cap from NANOVEC_MEMORY_LIMIT_MB
    pub used_bytes: AtomicU64,      // current usage
    pub evict_on_overflow: bool,    // default: true (Task 6 включает eviction)
}

impl Collection {
    pub fn approx_bytes(&self) -> u64 {
        (self.dimension() * 4 * self.count()) as u64
            + self.records.iter().map(|r| r.text.len() as u64).sum::<u64>()
    }
}
```

Env-var: `NANOVEC_MEMORY_LIMIT_MB` (default `512`).

---

### Task 6 — LRU eviction per-collection
- **Agent:** core
- **Files:** `src/store/collections.rs`, `src/store/budget.rs`, `src/mcp/mod.rs`
- **Depends on:** Task 5
- **Acceptance:**
  - Каждая `CollectionInner` имеет `last_accessed_tick: AtomicU64` (увеличивается на каждый search/insert/delete относительно глобального `Database::tick_counter: AtomicU64`)
  - `db.evict_until_under_budget()` выбирает кэлекцию с min `last_accessed_tick` и дропает её целиком; повторяет пока `used_bytes < limit_bytes`
  - Кэлекция `"default"` и кэлекции с `connection_id`-префиксом `"_conn_*"` **не эвиктятся** — это «pinned»; только явно созданные через `create_collection` эвиктятся
  - `tests/eviction.rs`: лимит 1 MB; создать 3 кэлекции (A, B, C) по 400 KB каждая; обратиться к A; insert в C на 500 KB → B (самая «холодная») должна быть выброшена; A и C остаются

**Поведение при попытке insert в pinned кэлекцию с превышением:** `Err(DbError::BudgetExceeded { collection })` — pinned кэлекции имеют приоритет, но не безграничны.

---

### Task 7 — `stats` tool обновление + новый `memory` tool
- **Agent:** mcp
- **Files:** `src/mcp/tools.rs`
- **Depends on:** Task 5, Task 6
- **Acceptance:**
  - `stats` теперь возвращает `memory_bytes`, `memory_limit_bytes`, `memory_pct`, `collections_count`, `eviction_events` per последний час
  - Новый tool `memory` возвращает breakdown per collection: `[{name, count, approx_bytes, last_accessed_tick}]`, отсортированный по `last_accessed_tick` (горячие сверху)
  - Integration test через MCP: `stats` после серии операций даёт ожидаемые числа в пределах ±10%

---

### Task 8 — Load test + benchmark
- **Agent:** testing
- **Files:** `benches/concurrency.rs` (new)
- **Depends on:** Task 1, Task 3, Task 4
- **Acceptance:** criterion benchmark с тремя сценариями:
  - `search_single_thread_baseline` — 10k × 384-dim, brute-force, 1 поток (reference)
  - `search_8_threads_same_collection` — те же 10k, 8 потоков параллельно → throughput target ≥ 6× baseline (reader-lock overhead < 25%)
  - `search_8_threads_8_collections` — 8 кэлекций × 1.25k each, 8 потоков один-к-одному → throughput target ≥ 7.5× baseline (idealный case)
  - Числа фиксируются в `phase-6-multi-agent-report.md` (создаётся в Task 10)

---

### Task 9 — End-to-end SSE integration test
- **Agent:** testing
- **Files:** `tests/sse_integration.rs` (new)
- **Depends on:** Task 3, Task 4, Task 6
- **Acceptance:** один integration-тест spawn'ит реальный server (subprocess), коннектится по SSE, выполняет полный цикл:
  - `initialize` → `index_document(text="hello", connection_id="alice")` × 100
  - Параллельно второй клиент `bob` индексирует свои 100 записей
  - `search_document(query="hello", connection_id="alice")` → возвращает только alice-записи
  - `stats` → видит обе connection-кэлекции
  - Graceful shutdown через SIGTERM, restart → state пустой (ephemeral verified)

---

### Task 10 — Documentation
- **Agent:** docs
- **Files:** `docs/components/multi-agent.md` (new), `docs/components/sse-transport.md` (new), `docs/coverage.md`, `docs/ROADMAP.md`, `CHANGELOG.md`, `README.md`, `docs/plans/phase-6-multi-agent-report.md` (new)
- **Depends on:** Task 1–9 (все code-задачи)
- **Acceptance:**
  - `multi-agent.md`: схема двухуровневого lock'а, диаграмма потоков, описание budget/eviction
  - `sse-transport.md`: env vars, `/sse` endpoint, пример curl, ограничения (no auth — local-only по дизайну)
  - `phase-6-multi-agent-report.md`: финальные числа из Task 8, deviations from plan, follow-ups
  - `ROADMAP.md`: Phase 6 → ✅ Complete
  - `CHANGELOG.md`: новая секция `## [Unreleased] — Phase 6`

---

## Final Validation

```bash
cargo fmt --check && \
cargo clippy --all-targets --all-features -- -D warnings && \
cargo test --all-features && \
cargo bench --bench concurrency
```

Все шаги зелёные. Числа из Task 8 (≥ 6× и ≥ 7.5×) подтверждены.

---

## Out of Scope (явно отложено)

- **Per-record LRU eviction** — Phase 7+ (требует двусвязного списка / weak-ref графа)
- **Authentication / TLS для SSE** — local-only по дизайну, перенесено в потенциальную Phase 8 (production-hardening)
- **Persistence** — нарушает «ephemeral by design» из CLAUDE.md
- **Cluster mode / sharding** — это Qdrant'овский use-case, NanoVec остаётся single-node
- **WebSocket transport** — SSE покрывает все нужные сценарии (half-duplex), WS отложен как YAGNI
