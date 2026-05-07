# Plan: Phase 2 — Vectors → Text

## Summary

Добавить серверный embedding к NanoVec: новый модуль `src/embed/` на базе **candle**
(HuggingFace pure-Rust runtime) с моделью **`sentence-transformers/all-MiniLM-L6-v2`**
(384-dim, L2-normalized). Два новых MCP-инструмента — `index_document(text, metadata?)`
и `search_document(query_text, k, metric?)` — клиент работает с текстом, сервер сам
считает эмбеддинги. Старые `index_vector` / `search` остаются (raw vector path),
но теперь требуют согласованности с dim=384 после загрузки embedder'а.

После Phase 2: Claude Code пишет в NanoVec фразы естественным языком и ищет
семантически — без клиентского embedding.

---

## Design Decisions (требуют подтверждения)

| # | Решение | Rationale |
|---|---------|-----------|
| D1 | **Расширение API, не замена.** Добавляем `index_document` / `search_document`; `index_vector` / `search` остаются. | Не ломаем интеграционный контракт Phase 1 без необходимости. ROADMAP явно сохраняет `index_vector` для совместимости. |
| D2 | **Dimension lock = 384 при загрузке embedder'а.** `VectorStore` инициализируется на старте, `index_vector` с dim≠384 → ошибка. | Предсказуемее, чем lazy-lock. Цена: Phase 1 integration test должен использовать dim=384 либо переключиться на `index_document`. |
| D3 | **Default metric = Cosine** для embedded путей. | all-MiniLM-L6-v2 даёт L2-normalized output → cosine ≡ dot product, но cosine семантически прозрачнее. Server-wide default остаётся Euclidean (для raw-векторов); новые tools fallback'ятся в Cosine. |
| D4 | **Eager load embedder'а в `server::run()`** до старта stdio transport, через `tokio::task::spawn_blocking`. | Fail-fast: если HF Hub недоступен и кэша нет — выходим с понятной ошибкой до handshake. |
| D5 | **Embedder = `Arc<Embedder>` в state.** `embed(&self, &str)` принимает `&self`, без блокировки. | candle `BertModel` — Send+Sync; embed read-only после load. Mutex держим только вокруг `VectorStore` + `RecordStore`. |

---

## Tasks

### Task 1 — Cargo deps + Embedder scaffold
- **Agent:** core
- **Files:** `Cargo.toml`, `src/embed/mod.rs` (new), `src/lib.rs`
- **Depends on:** none
- **Acceptance:**
  - `cargo build` чистый
  - `cargo clippy --all-targets --all-features -- -D warnings` чистый
  - В `src/embed/mod.rs` присутствует failing test `embedder_loads_and_embeds_to_384_dim` (`#[ignore]` пока не реализован, но компилируется)

**Дополнения к `[dependencies]`** (версии — последние стабильные на crates.io, агент проверяет перед commit):
- `candle-core` (≥ 0.8)
- `candle-nn` (≥ 0.8)
- `candle-transformers` (≥ 0.8)
- `tokenizers` (≥ 0.20)
- `hf-hub` (≥ 0.3, feature `tokio`)

`src/lib.rs` получает `pub mod embed;`.

---

### Task 2 — Embedder implementation
- **Agent:** core
- **Files:** `src/embed/mod.rs`
- **Depends on:** Task 1
- **Acceptance:** все тесты модуля проходят:
  - `embedder_loads_and_embeds_to_384_dim` — `embed("hello world")` возвращает `Vec<f32>` длины 384
  - `embedding_is_l2_normalized` — `‖v‖₂ ≈ 1.0 ± 1e-3` (proptest на нескольких текстах)
  - `embedding_is_deterministic` — два вызова с одинаковым текстом → идентичный вектор (bitwise или ε=1e-6)
  - `semantic_ordering_simple` — `cos_sim("dog","puppy") > cos_sim("dog","spaceship")`

**API:**
```rust
pub struct Embedder { /* tokenizer, BertModel, device */ }

impl Embedder {
    pub fn load() -> Result<Self, EmbedError>;        // sync; downloads model on first run
    pub fn dimension(&self) -> usize;                 // 384
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}

#[derive(Debug)]
pub enum EmbedError {
    DownloadFailed(String),
    TokenizeFailed(String),
    ModelLoadFailed(String),
    InferenceFailed(String),
}
```

**Pipeline `embed()`**: tokenize → `BertModel::forward` → mean-pool по seq dim
с маской внимания → L2-normalize → `Vec<f32>` длины 384.

**Тестовая инфраструктура:** общий `Lazy<Embedder>` (через `once_cell` или
`std::sync::OnceLock`) внутри `mod tests`, чтобы все тесты делили одну загрузку
модели и не качали её многократно.

---

### Task 3 — Wire Embedder into state + server startup
- **Agent:** mcp
- **Files:** `src/mcp/mod.rs`, `src/server/mod.rs`
- **Depends on:** Task 2
- **Acceptance:**
  - `cargo run` логирует в stderr `loaded embedder, dim=384` и поднимает stdio как раньше
  - `cargo test --test integration` (Phase 1 flow, обновлённый под dim=384) зелёный

**Изменения:**
- `NanoVecState` получает поле `embedder: Arc<Embedder>` и инициализируется со
  store, заполненным `Some(VectorStore::new(384))` (вместо `None`).
- `NanoVecServer::new(metric: Metric, embedder: Arc<Embedder>)`.
- `server::run()`:
  ```
  let embedder = tokio::task::spawn_blocking(Embedder::load).await??;
  let server = NanoVecServer::new(Metric::Euclidean, Arc::new(embedder));
  ```
- При load failure — `anyhow::Error` с описанием (download / tokenizer / model).

**Влияние на существующие tools:**
- `index_vector` теперь валидирует `vector.len() == 384` (а не «первый вызов задаёт
  dim»). Сообщение об ошибке делает явным: «server dim is 384 (locked by embedder)».
- `delete` больше не сбрасывает `store` в `None` при `count == 0` — store
  остаётся с зафиксированной размерностью.
- `stats` возвращает `dimension: 384` всегда.

---

### Task 4 — `index_document` MCP tool
- **Agent:** mcp
- **Files:** `src/mcp/tools.rs`, `src/mcp/mod.rs`
- **Depends on:** Task 3
- **Acceptance:**
  - Unit-тест на handler: непустой текст → `{"id": N}`; пустой → понятная ошибка
  - Поведение совпадает с `index_vector` после embedding (тот же id-сompare)

**Параметры:**
```rust
#[derive(Deserialize, schemars::JsonSchema)]
pub struct IndexDocumentParams {
    pub text: String,
    pub metadata: Option<serde_json::Value>,
}
```

**Handler-flow:** lock state → `state.embedder.embed(&params.text)?` →
`store.insert(&vector)?` → `records.insert(text, metadata, offset)` → JSON `{id}`.

---

### Task 5a — Extend `SearchResult` with `metadata` field
- **Agent:** core
- **Files:** `src/index/brute.rs`
- **Depends on:** Task 1 (нужен собранный проект; от embedder'а не зависит)
- **Acceptance:**
  - `SearchResult` имеет `metadata: Vec<(String, String)>`, заполнено из `VectorRecord::metadata` в `BruteForce::search`
  - Существующие тесты `index::brute::tests::*` остаются зелёными (там, где они конструируют `SearchResult` напрямую — добавляется `metadata: vec![]`)

**Изменения:**
```rust
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
    pub text: String,
    pub metadata: Vec<(String, String)>,  // NEW
}
```
В `BruteForce::search` — при сборке `SearchResult` из `VectorRecord` добавить `metadata: r.metadata.clone()`.

---

### Task 5b — `search_document` MCP tool
- **Agent:** mcp
- **Files:** `src/mcp/tools.rs`, `src/mcp/mod.rs`
- **Depends on:** Task 3, Task 5a
- **Acceptance:**
  - Unit-тест handler'а: проиндексировать 3 текста, `search_document(близкий запрос, k=1)` → правильный top-1
  - Возвращает массив с `id`, `score`, `text`, `metadata`

**Параметры:**
```rust
#[derive(Deserialize, schemars::JsonSchema)]
pub struct SearchDocumentParams {
    pub query: String,
    pub k: usize,
    pub metric: Option<String>,  // default → Cosine для embedded path
}
```

**Handler-flow:** `state.embedder.embed(&params.query)?` (без блокировки — embedder под `Arc`) → lock state → `BruteForce::search(store, records, &q_vec, k, metric)` → JSON array с `id` / `score` / `text` / `metadata`.

---

### Task 6 — Tests + Proptest baseline
- **Agent:** testing
- **Files:** `src/embed/mod.rs` (proptest внутри `mod tests`), `tests/integration/mcp_embedding.rs` (new), обновление `tests/integration/mcp_stdio.rs`
- **Depends on:** Task 4, Task 5b
- **Acceptance:** `cargo test --all-features` полностью зелёный.

**Proptest (`src/embed/mod.rs`):**
```rust
proptest! {
    #[test]
    fn embedding_norm_is_unit(text in "\\PC{1,200}") {
        let v = SHARED_EMBEDDER.embed(&text).unwrap();
        let norm = v.iter().map(|x| x*x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-3);
    }
    #[test]
    fn embedding_dim_always_384(text in "\\PC{1,200}") {
        prop_assert_eq!(SHARED_EMBEDDER.embed(&text).unwrap().len(), 384);
    }
}
```

**Integration test (`tests/integration/mcp_embedding.rs`):**
1. start `nanovec` бинарник
2. `index_document("dogs are loyal companions", {})` → id1
3. `index_document("cats are independent", {})` → id2
4. `index_document("rockets reach orbit", {})` → id3
5. `search_document("faithful canine friend", k=1)` → top.id == id1
6. `delete(id1)` → стало 2
7. `stats` → `count=2, dimension=384`

**Обновление существующего `mcp_stdio.rs`:** заменить toy-вектор размерности 4
на 384-dim вектор (можно сгенерировать `vec![1.0/384_f32.sqrt(); 384]`) либо
переписать сценарий через `index_document`.

**Network policy:** оба интеграционных теста требуют либо HF cache, либо доступ
в интернет на первый запуск. Помечаем модулем-уровневым `#[cfg(feature = "online-tests")]`
или env-флагом `NANOVEC_RUN_NETWORK_TESTS=1` — финальное решение принимает testing agent
на этапе реализации.

---

### Task 7 — Documentation
- **Agent:** docs
- **Files:** `docs/components/embed.md` (new), `docs/components/mcp-server.md` (update), `docs/coverage.md` (update)
- **Depends on:** Task 6
- **Acceptance:**
  - `docs/components/embed.md` описывает Embedder API, pipeline, выбор модели
  - `docs/components/mcp-server.md` дополнен секциями `index_document` / `search_document`
  - `docs/coverage.md`: Phase 2 → status `documented`
  - (опц.) `docs/PHASE-2-PRESENTATION.md` по образцу Phase 1

---

## Execution Order

```
Task 1 (deps + scaffold)
  └── Task 2 (Embedder impl)
        └── Task 3 (state + startup)
              ├── Task 4  (index_document, mcp)         ─┐
              └── Task 5a (SearchResult.metadata, core) ─┤
                                                          └── Task 5b (search_document, mcp)
                                                                └── Task 6 (Tests)
                                                                      └── Task 7 (Docs)
```

Group D (Task 4 ‖ Task 5a) — параллельно: разные агенты, разные файлы.
Group E (Task 5b) — последовательно после 4 и 5a (общие файлы с Task 4 в `src/mcp/`).

---

## Risks

### R1 — candle / tokenizers / hf-hub минорные версии — MEDIUM
Crates активно развиваются, API между 0.x ломается. **Митигация:** core agent
проверяет crates.io перед Task 1, фиксирует `Cargo.lock`. Если `BertModel` API
изменился по сравнению с candle examples — следовать актуальному `candle-examples/examples/bert`.

### R2 — Network на CI / в тестах — MEDIUM
Первый прогон качает ~90 MB с HF Hub. **Митигация:** опциональный фичефлаг
для интеграционных тестов; локально модель кэшируется в `~/.cache/huggingface/`
после первого запуска. README/CHANGELOG описывает требование.

### R3 — Время старта сервера — LOW
~0.5-1 сек на CPU-init модели. **Митигация:** документируем; для дальнейшей
оптимизации (Phase 3+) — ленивая загрузка или mmap'нутый safetensors.

### R4 — Phase 1 integration test ломается из-за dim-lock — LOW (известный)
**Митигация:** Task 6 явно покрывает обновление `mcp_stdio.rs`.

### R5 — Mutex contention на `embed` — LOW
`embedder.embed()` вызывается **до** взятия Mutex на state — он `&self`, не
требует блокировки. Mutex держим только вокруг `VectorStore + RecordStore`,
как и раньше.

### R6 — Двойственность default metric — LOW
Server-wide default = Euclidean (Phase 1), но новые `*_document` tools fallback'ятся
в Cosine. Задокументировать в `docs/components/mcp-server.md`, чтобы не было
сюрпризов при `metric: null` в новых вызовах.

---

## Acceptance for the whole phase

После всех задач:
1. `cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features` — зелёный
2. `cargo run` стартует, грузит embedder, открывает stdio
3. End-to-end сценарий через MCP: индексировать 3 фразы → семантический поиск возвращает корректный top-1 по смыслу
4. `docs/coverage.md` помечает Phase 2 как documented
5. `docs/ROADMAP.md` строка `Phase 2 — Vectors → Text` обновлена в `✅ Complete`

---

## Next step

План одобрен. Dispatch schedule зафиксирован в
[`phase-2-embeddings-dispatch.md`](./phase-2-embeddings-dispatch.md).
Дальше — `/execute` для пошагового выполнения по schedule.
