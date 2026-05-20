# NanoVec

[English →](README.md)

> Легкий in-memory векторний DB без зовнішніх залежностей — у вигляді MCP-сервера для ефемерної робочої пам'яті AI-агентів.

NanoVec — реалізація векторної бази даних на чистому Rust з нуля. SIMD-прискорена дистанція, власноруч написаний HNSW, два транспорти (stdio + streamable HTTP), єдиний зовнішній інтерфейс — Model Context Protocol.

**Статус:** усі сім запланованих фаз завершені. Дивись [`docs/ROADMAP.md`](docs/ROADMAP.md).

| Фаза | Що додала |
|------|-----------|
| 1 — MVP | VectorStore (SoA), RecordStore, скалярна дистанція, brute-force KNN, MCP stdio |
| 2 — Embeddings | candle + `all-MiniLM-L6-v2` (384-dim), `index_document` / `search_document` |
| 2.5 — Ергономіка | `clear`, розширений `stats`, поля `distance` / `similarity` |
| 3 — SIMD | NEON (aarch64) + AVX2 (x86_64) дистанція, runtime dispatch, property-тести |
| 4 — Collections | Іменовані колекції, фільтрація за метаданими, multi-tenant в одному процесі |
| 5 — KD-Tree | Просторовий індекс для низької розмірності |
| 6 — Multi-agent | Дворівневий `parking_lot::RwLock`, бюджет пам'яті, LRU eviction, streamable HTTP, `connection_id` |
| 7 — Approximate | HNSW з нуля (евристика Algorithm 4), MCP-інструмент `rebuild_index` |

## Що всередині

- **In-process векторне сховище.** Плоский `Vec<f32>` SoA, фіксована розмірність, O(1) insert / get / swap-remove.
- **Серверний ембеддинг.** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-нормалізована) через [`candle`](https://github.com/huggingface/candle) — pure Rust, CPU-only.
- **SIMD дистанція.** Cosine / Euclidean / dot, runtime dispatch на NEON (Apple Silicon) чи AVX2+FMA (x86_64), з property-тестами проти скалярного референсу.
- **Три індекси.** Brute-force (будь-яка метрика, default), KD-Tree (низька розмірність), HNSW (cosine, opt-in через `rebuild_index`).
- **Multi-agent.** `RwLock` per-collection: читачі не блокують один одного; soft memory budget з per-collection LRU eviction.
- **Два транспорти.** stdio (default, локальний) і streamable HTTP (axum, opt-in через `NANOVEC_SSE_ADDR`).
- **Multi-tenant.** Кожен MCP-інструмент приймає `connection_id`; авто-створює pinned `_conn_<id>` колекції що виключені з eviction.
- **Ефемерний за дизайном.** Без персистенції, без WAL — дані живуть у RAM і зникають разом із процесом.
- **Нуль зовнішніх індексаційних бібліотек.** Без FAISS, без HNSWLIB, без Annoy чи Qdrant.

## Виміряна продуктивність

Усі числа з `cargo bench` і `cargo test --release` на Apple Silicon (M-series, NEON).

### SIMD дистанція @ 384-dim

| Операція | Scalar | NEON | Прискорення |
|----------|------:|----:|------------:|
| `dot_product` | 296 ns | 32 ns | **9.3×** |
| `euclidean` | 305 ns | 33 ns | 9.2× |
| `cosine` | 414 ns | 95 ns | 4.4× (3 редукції обмежують виграш) |

### KD-Tree vs brute-force, k=10

| N × dim | Brute | KD-Tree | Прискорення |
|---------|------:|--------:|------------:|
| 1k × 8-dim | 6.3 µs | 1.7 µs | 3.7× |
| 10k × 8-dim | 44.5 µs | 1.5 µs | 29.7× |
| 10k × 384-dim | 448 µs | 63.5 µs | **7.1×** (curse of dimensionality пригнічує, але не вбиває) |

### Phase 6 — concurrency (5k × 384-dim, k=10)

| Сценарій | 1 потік | 8 потоків | Scaling |
|----------|--------:|----------:|--------:|
| Та сама колекція (читачі) | 1 922 q/s | 13 389 q/s | **6.97×** |
| 8 колекцій × 8 потоків | 11 648 q/s | 71 920 q/s | 6.17× |

### Phase 7 — HNSW (10k реальних MiniLM embeddings, 30 семантичних запитів)

Baseline brute-force: 822 µs p50.

| ef_search | p50 search | Прискорення | Recall@10 |
|----------:|-----------:|------------:|----------:|
| 10 | 43 µs | **19.1×** | 0.850 |
| 25 | 58 µs | 14.2× | **0.990** (sweet spot) |
| 50 (default) | 92 µs | 8.9× | 0.993 |
| 100 | 156 µs | 5.3× | 1.000 |

Відтворити: `cargo bench --bench simd_vs_scalar --bench kdtree --bench hnsw_sweep` + `cargo test --release -- --ignored --nocapture --test-threads=1 hnsw_stress`.

## Швидкий старт

### Збірка

```bash
git clone https://github.com/nikita-voloshyn/nanovec
cd nanovec
cargo build --release
```

Бінарник: `target/release/nanovec`.

### Підключити до Claude Code (stdio)

Додай бінарник у MCP-конфіг клієнта. Приклад для Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "nanovec": {
      "command": "/absolute/path/to/nanovec/target/release/nanovec"
    }
  }
}
```

При першому запуску NanoVec завантажить модель (~90 MB) із HuggingFace Hub до `~/.cache/huggingface/hub/`. Холодний старт ~10–30 с; теплий ~30 мс.

### Підключити через HTTP

```bash
NANOVEC_SSE_ADDR=127.0.0.1:8421 cargo run --release
```

Streamable HTTP endpoint змонтований на `/mcp`. Сервер без автентифікації — тримай на localhost або за reverse proxy з авторизацією.

### Змінні середовища

| Змінна | Default | Ефект |
|--------|---------|-------|
| `NANOVEC_SSE_ADDR` | unset | Якщо вказано `host:port`, запускає streamable HTTP замість stdio. |
| `NANOVEC_MEMORY_LIMIT_MB` | 512 | Soft межа пам'яті на процес; спричиняє LRU eviction некріплених колекцій. |
| `RUST_LOG` | warn | Стандартний фільтр `tracing-subscriber`. |

Усі логи йдуть у stderr; stdout зарезервовано для MCP JSON-RPC (у stdio-режимі).

## MCP-інструменти

| Інструмент | Вхід | Вихід |
|------------|------|-------|
| `index_document` | `text`, опц. `metadata`, опц. `collection` чи `connection_id` | `{ id }` |
| `search_document` | `query`, `k`, опц. `metric` / `filter` / `collection` / `connection_id` | `[{ id, text, metadata, distance, similarity, score }]` |
| `index_vector` | `text`, `vector`, опц. `metadata` / `collection` / `connection_id` | `{ id }` |
| `search` | `vector`, `k`, опц. `metric` / `filter` / `collection` / `connection_id` | `[...]` |
| `delete` | `id`, опц. `collection` / `connection_id` | `{ success }` |
| `clear` | опц. `collection` / `connection_id` | `{ deleted }` |
| `create_collection` | `name`, `dimension` | `{ created, dimension }` |
| `list_collections` | — | `{ collections: [...] }` |
| `drop_collection` | `name` | `{ dropped }` |
| `stats` | — | агрегати + `memory` блок (used/limit/pct) |
| `memory` | — | per-collection breakdown відсортований hot-first |
| `rebuild_index` | опц. `collection` / `connection_id`, `kind` (`"hnsw"` чи `"none"`), опц. `m` / `ef_construction` / `ef_search` | `{ kind, nodes, build_ms, memory_bytes, ... }` |

Метрика за замовчуванням: `euclidean` для raw-vector шляху, `cosine` для document шляху. Override через `"euclidean"`, `"cosine"`, `"dot"`.

Результати містять `distance` (raw, менше = ближче) і `similarity` (адаптована до метрики, більше = ближче). `score` — backward-compat alias до `distance`.

### Multi-tenant через `connection_id`

Виклик інструмента з `connection_id: "alice"` і без явного `collection` потрапляє у `_conn_alice` (auto-створюється при першому використанні, pinned). Два клієнти з різними `connection_id` не бачать дані один одного, якщо не вказують спільний `collection`.

### Робота з HNSW

```jsonc
// Побудувати HNSW для поточної колекції.
{"name": "rebuild_index", "arguments": {"kind": "hnsw", "m": 16, "ef_search": 25}}
// → {"kind":"hnsw","nodes":10000,"build_ms":2046,...}

// Подальші search/search_document автоматично використовують HNSW при metric=cosine.
{"name": "search_document", "arguments": {"query": "...", "k": 10}}

// Будь-яка мутація invalidує індекс — викличи rebuild_index знову,
// або "kind": "none" щоб явно скинути.
```

## Архітектура

```
src/
  store/       VectorStore (flat SoA), RecordStore (text + metadata + ID map)
  store/       database.rs   NanoVecDatabase: двохрівневий RwLock + бюджет + LRU
  distance/    Скалярні Евклід / cosine / dot
  simd/        NEON + AVX2 + scalar fallback, runtime dispatch
  heap/        BoundedMaxHeap для top-K
  index/       brute.rs (linear scan), kdtree.rs, hnsw.rs (Phase 7)
  embed/       candle + all-MiniLM-L6-v2 loader
  mcp/         rmcp tool definitions, server router
  transport/   stdio.rs + http.rs (axum) + select_from_env()
  server/      Async entry point, прогрів ембеддера, transport dispatch
```

Покомпонентна довідка: [`docs/components/`](docs/components/).

## Документація

- [`docs/README.md`](docs/README.md) — двомовний індекс документації
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — план фаз
- [`docs/PRESENTATION-PL.html`](docs/PRESENTATION-PL.html) — підсумкова презентація польською (15 слайдів)
- [`docs/PROJECT-STATUS.html`](docs/PROJECT-STATUS.html) — статус проєкту російською
- [`docs/MULTILINGUAL-BENCHMARK.md`](docs/MULTILINGUAL-BENCHMARK.md) — ScootGo FAQ recall@5 (EN/PL/UK)
- [`docs/components/`](docs/components/) — описи компонентів (database, transport-http, hnsw, …)
- [`docs/plans/`](docs/plans/) — плани фаз і dispatch-звіти

## Розробка

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Повні бенчмарки (кілька хвилин):

```bash
cargo bench --bench distance --bench simd_vs_scalar --bench kdtree --bench hnsw_sweep
```

Повільні real-corpus тести (потрібен HuggingFace cache, ~7 хв):

```bash
cargo test --release --test integration -- --ignored --nocapture --test-threads=1
```

NanoVec слідує робочому процесу TDD-first / YAGNI. Налаштування агентів і скілів задокументоване у [`CLAUDE.md`](CLAUDE.md).

## Ліцензія

MIT.
