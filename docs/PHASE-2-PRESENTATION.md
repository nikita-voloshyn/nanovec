# NanoVec — Phase 2: Vectors → Text

## Що це

Phase 2 додає серверний embedding до NanoVec: агент тепер пише звичайний текст —
сервер сам перетворює його у вектор і зберігає. Пошук теж приймає текст.

```
// До (Phase 1): клієнт передає вектор
index_vector(text: "зустріч в п'ятницю", vector: [0.12, -0.07, ...384 floats...])

// Після (Phase 2): клієнт передає текст
index_document(text: "зустріч в п'ятницю", metadata: {"category": "calendar"})
search_document(query: "які зустрічі заплановані?", k: 3)
```

---

## Що реалізовано в Phase 2

### 1. Embedder (`src/embed/mod.rs`)

Серверний embedding на базі **candle** (HuggingFace pure-Rust runtime) і моделі
**`sentence-transformers/all-MiniLM-L6-v2`** (6 шарів BERT, 384-dim).

| Властивість | Значення |
|-------------|----------|
| Модель | `sentence-transformers/all-MiniLM-L6-v2` |
| Розмірність | 384 |
| Нормалізація | L2 (одиничний вектор) |
| Розмір ваг | ~90 MB (`model.safetensors`) |
| Кеш | `~/.cache/huggingface/hub/` |
| Холодний старт | 10-30 с (завантаження) |
| Теплий старт | ~30 мс (mmap із кешу) |
| Runtime | pure Rust, без Python / ONNX |

**Pipeline `embed(text)`:**

```
tokenize (CLS/SEP) → BertModel::forward → mean-pool (attention mask) → L2-normalize → Vec<f32>(384)
```

**API:**

```rust
pub struct Embedder { /* private */ }
impl Embedder {
    pub fn load() -> Result<Self, EmbedError>;     // sync; use spawn_blocking
    pub fn dimension(&self) -> usize;              // 384
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}
```

### 2. Нові MCP-інструменти (`src/mcp/`)

Два нові інструменти поверх існуючих чотирьох:

| Інструмент | Вхід | Вихід |
|------------|------|-------|
| `index_document` | `text`, `metadata?` | `{"id": N}` |
| `search_document` | `query`, `k`, `metric?` (default Cosine) | `[{id, score, text, metadata}]` |

Старі `index_vector` і `search` збережені для сумісності. Усього тепер 6 інструментів:
`index_vector`, `index_document`, `search`, `search_document`, `delete`, `stats`.

### 3. Dim-lock при старті

`VectorStore` ініціалізується при запуску сервера з dim=384 (а не при першому insert).
`index_vector` тепер вимагає dim=384; інші розмірності → помилка.

### 4. Two-phase lock pattern

`index_document` та `search_document` використовують двофазне блокування:

```
Phase 1: взяти Mutex, клонувати Arc<Embedder>, відпустити Mutex
Phase 2: embed(text) — BERT forward без будь-якого блокування
Phase 3: взяти Mutex знову, вставити вектор і запис
```

Це дозволяє паралельним запитам виконувати BERT forward одночасно, не серіалізуючись
на одному мютексі.

---

## Архітектурні рішення

| Рішення | Чому |
|---------|------|
| D1: Розширення API, не заміна | `index_vector` / `search` залишаються; не ламаємо Phase 1 контракт |
| D2: Dim lock = 384 при старті | Fail-fast при завантаженні embedder'а; передбачуваніший, ніж lazy-lock |
| D3: Default metric = Cosine для embedded path | all-MiniLM-L6-v2 дає L2-normalized вектори; cosine семантично прозоріший; server-wide default залишається Euclidean |
| D4: Eager load в `server::run()` через `spawn_blocking` | Fail-fast: якщо HF Hub недоступний — помилка до MCP handshake |
| D5: `Arc<Embedder>` поза Mutex | `embed(&self)` — read-only; Mutex тримає тільки `VectorStore + RecordStore` |

**Примітка про розходження default metric:**

Серверний default = Euclidean (Phase 1 сумісність); `search_document` default = Cosine.
Обидва інструменти приймають явний `metric` override.

---

## Тести

### Покрытие по модулям (Phase 2)

| Модуль | Тестів | Тип |
|--------|--------|-----|
| `embed::Embedder` | 3 unit + 2 proptest (16 cases each) | unit + proptest |
| `mcp` unit (index_document, search_document handlers) | входять до 54 lib | unit |
| **MCP integration (`test_mcp_full_flow`)** | 1 | integration |
| **Semantic integration (`test_semantic_search_end_to_end`)** | 1 | integration |
| **Разом** | **54 lib + 2 integration = 56** | |

Quality gate після Phase 2: `cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features` — зелений, 0 failed, 0 ignored.

### Ключовий тест: `test_semantic_search_end_to_end`

`tests/integration/mcp_embedding.rs`. Запускає реальний бінарник, індексує 3 тексти
з різних семантичних кластерів, перевіряє top-1 для трьох запитів:

| Запит | Очікуваний top-1 |
|-------|-----------------|
| `"what meetings do I have this week?"` | calendar document (id=0) |
| `"groceries to pick up"` | shopping document (id=1) |
| `"what broke production?"` | incident document (id=2) |

Тест також перевіряє, що `metadata` передається правильно у відповіді `search_document`,
і що після `delete(id=1)` продуктовий документ зникає з результатів.

Proptest-властивості для Embedder:
- `embedding_is_l2_normalized` — `‖embed(text)‖₂ ≈ 1.0 ± 1e-3` для 16 random inputs
- `embedding_dim_always_384` — `embed(text).len() == 384` для 16 random inputs

---

## Запуск

```bash
# Усі тести (включно з інтеграційними — потрібен HF cache або інтернет)
cargo test --all-features

# Тільки semantic integration test
cargo test test_semantic_search_end_to_end --test integration

# Quality gate
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features

# Запуск сервера
cargo run
```

**Перший запуск** завантажує ~90 MB з HuggingFace Hub. Під час завантаження сервер
пише в stderr:

```
INFO nanovec::server: loading embedder (sentence-transformers/all-MiniLM-L6-v2, ~90MB on first run)
INFO nanovec::server: embedder loaded dimension=384
INFO nanovec::server: nanovec MCP server starting on stdio
```

Після завантаження (10-30 с cold / ~30 мс warm) сервер готовий до JSON-RPC.

### Sample JSON-RPC interaction

```jsonc
// 1. Initialize
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"myagent","version":"1.0"}}}
← {"jsonrpc":"2.0","id":1,"result":{...}}

// 2. Index a document
→ {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"index_document","arguments":{"text":"buy oat milk and sourdough bread","metadata":{"tag":"errand"}}}}
← {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\"id\":0}"}]}}

// 3. Search by text
→ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_document","arguments":{"query":"grocery shopping list","k":1}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"[{\"id\":0,\"score\":0.47,\"text\":\"buy oat milk and sourdough bread\",\"metadata\":{\"tag\":\"errand\"}}]"}]}}
```

---

## Структура нового коду

```
src/
  embed/
    mod.rs              # Embedder: load, dimension, embed + EmbedError
  mcp/
    mod.rs              # + index_document, search_document handlers
                        # + two-phase lock pattern
                        # + NanoVecState::embedder field
    tools.rs            # + IndexDocumentParams, SearchDocumentParams
  server/
    mod.rs              # + embedder load via spawn_blocking before stdio
  index/
    brute.rs            # SearchResult::metadata field added (Phase 2 Task 5a)
tests/
  integration/
    mcp_embedding.rs    # new: semantic search end-to-end test
    mcp_stdio.rs        # updated: 384-dim unit vector (was 4-dim toy vector)
```

---

## Відомі follow-up (не виправлені в Phase 2)

1. **ANSI escape codes в stderr.** `tracing-subscriber` за замовчуванням кольоровий.
   При підключенні через `.mcp.json` escape-коди потрапляють в лог-стрім. Фікс:
   `with_ansi(false)` у `src/main.rs`. Відкладено до post-Phase-2 polish.

2. **Повільний перший release build.** `candle` + весь transformer dependency tree —
   ~5 хвилин з нуля. Наступні incremental builds: ~5-10 с. Це one-time cost.

3. **CI strategy для HF cache.** Інтеграційні тести потребують або мережу при першому
   запуску, або прогрітий `~/.cache/huggingface/`. Стратегія для CI (cache artifact,
   мережевий доступ або mock) виходить за рамки Phase 2.

---

## Що далі — Phase 3

> **Scalar → SIMD**: прискорення обчислення відстаней в 8-16x

**До:** скалярний цикл — `a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum()`
**Після:** AVX2 (x86_64) / NEON (ARM) — 8 float'ів за такт

Deliverables Phase 3:
- AVX2 реалізації L2, cosine, dot product
- NEON реалізації для Apple Silicon
- Portable SIMD fallback (`std::simd`, nightly)
- Runtime dispatch: `is_x86_feature_detected!("avx2")`
- Criterion benchmarks: scalar vs SIMD на 384-dim векторах
- Proptest: SIMD output == scalar output ± epsilon
