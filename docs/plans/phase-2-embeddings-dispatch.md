# Dispatch: Phase 2 — Vectors → Text

Source plan: [`phase-2-embeddings-plan.md`](./phase-2-embeddings-plan.md).

## Schedule

| Order | Task | Agent | Skill | Parallel Group | Status |
|-------|------|-------|-------|----------------|--------|
| 1 | Cargo deps + Embedder scaffold | core | — | A | done |
| 2 | Embedder implementation (`load`, `embed`, errors) | core | — | B | done |
| 3 | Wire Embedder into NanoVecState + `server::run` startup | mcp | — | C | done |
| 4 | `index_document` MCP tool | mcp | — | D (parallel) | done |
| 5a | Extend `SearchResult` with `metadata` (core boundary) | core | — | D (parallel) | done |
| 5b | `search_document` MCP tool | mcp | — | E | done |
| 6 | Tests + Proptest baseline (unit + 2 integration) | testing | — | F | done |
| 6.5 | Quality Gate (`fmt + clippy + test`) | — | `/check` | F | done |
| 7 | Documentation | docs | `/docs` | G | done |

**Group D (Tasks 4 + 5a):** запускаются параллельно через
`superpowers:dispatching-parallel-agents` — разные домены и непересекающиеся
файлы. Task 5b ждёт завершения обоих.

---

## Boundary Note: Task 5 Split

В исходном плане Task 5 описывал и расширение `SearchResult` (поле `metadata`),
и handler `search_document`. Это нарушает domain rules CLAUDE.md:

- `src/index/brute.rs` — **core domain**
- `src/mcp/tools.rs`, `src/mcp/mod.rs` — **mcp domain**

mcp agent не может править production-код в core domain. Поэтому Task 5
разбит на:

- **Task 5a (core).** Добавить `pub metadata: Vec<(String, String)>` в `SearchResult`,
  заполнить из `VectorRecord::metadata` в `BruteForce::search`. Файл: `src/index/brute.rs`.
  Тривиальное изменение (~5 строк + обновить существующие тесты, чтобы они компилировались).
- **Task 5b (mcp).** Handler `search_document`, который читает обновлённое поле и
  кладёт в JSON-ответ. Файлы: `src/mcp/tools.rs`, `src/mcp/mod.rs`. Зависит от Task 5a + Task 3.

**Замечание для plan:** добавить эту разбивку в `phase-2-embeddings-plan.md` как
sub-tasks под Task 5.

---

## Agent Validation

### Task 1 — core ✅
- `Cargo.toml`, `rust-toolchain.toml`, `src/lib.rs`, `src/embed/mod.rs` — project-level + новый модуль с алгоритмической ответственностью
- Прецедент: Phase 1 dispatch назначал Cargo.toml + src/lib.rs core'у
- Запрещено core: SIMD intrinsics, MCP, benchmarks — ничто из этого не затронуто ✅

### Task 2 — core ✅
- `src/embed/mod.rs` — embed-модуль = data structure + алгоритмический pipeline (tokenize, mean-pool, L2-norm)
- candle инициализирует библиотечные SIMD под капотом, но **наш код не пишет intrinsics** — boundary соблюдён
- Нет MCP, нет benchmarks ✅

### Task 3 — mcp ✅
- `src/mcp/mod.rs`, `src/server/mod.rs` — оба явно в mcp domain
- Wiring + startup, нет векторной математики, нет SIMD ✅

### Task 4 — mcp ✅
- `src/mcp/tools.rs`, `src/mcp/mod.rs` — mcp domain
- Handler вызывает `Embedder::embed` и `VectorStore::insert` — потребляет API, не реализует ✅

### Task 5a — core ✅
- `src/index/brute.rs` — core domain
- Нет SIMD, нет MCP, нет benches ✅

### Task 5b — mcp ✅
- `src/mcp/tools.rs`, `src/mcp/mod.rs` — mcp domain
- Зависит от Task 5a (поле `SearchResult::metadata` уже существует) и Task 3 (embedder в state)
- Handler читает поля и форматирует JSON ✅

### Task 6 — testing ✅ (с boundary note)
- `tests/integration/mcp_embedding.rs` (new) — testing domain (`tests/`)
- Обновление `tests/integration/mcp_stdio.rs` — testing domain
- Proptest в `src/embed/mod.rs` — **только** внутри `#[cfg(test)] mod tests` (см. Phase 1 precedent: testing редактирует тестовые секции в production source)
- Нет правок production-кода вне `#[cfg(test)]` блоков ✅

### Task 7 — docs ✅
- `docs/components/embed.md`, `docs/components/mcp-server.md`, `docs/coverage.md`
- (опц.) `docs/PHASE-2-PRESENTATION.md`
- Не касается `src/` ✅

---

## Risks

### R1 — File overlap Task 4 / Task 5b — RESOLVED
**Проблема:** оба правят `src/mcp/tools.rs` и `src/mcp/mod.rs` — параллельный запуск
двух mcp subagents даст merge conflict.
**Митигация:** Task 5b ставится в Group E (после 4). В Group D параллелятся **только**
Task 4 (mcp) и Task 5a (core) — разные файлы, разные агенты, безопасно.

### R2 — candle / tokenizers / hf-hub minor version drift — MEDIUM
Унаследовано из плана (R1 plan). **Митигация:** core agent на Task 1 проверяет crates.io,
смотрит `candle-examples/examples/bert` для актуального API и фиксирует Cargo.lock.

### R3 — Network на CI / в `cargo test` — MEDIUM
Унаследовано из плана (R2 plan). **Митигация:** testing agent на Task 6 принимает
финальное решение (env-флаг `NANOVEC_RUN_NETWORK_TESTS=1` или `#[ignore]`) и
документирует в README. Quality Gate (6.5) запускается **с включённым флагом**, чтобы
интеграция действительно прогналась минимум раз.

### R4 — Слой Embedder под Mutex — LOW (плановое D5)
`embed(&self, ...)` не требует mut state. Mcp agent на Task 3 кладёт `Arc<Embedder>`
вне `Mutex<VectorStore + RecordStore>`. **Контроль:** Code-review этого момента
обязателен в Task 4 / 5b — не должно быть `state.lock()` вокруг embed-вызова.

### R5 — Phase 1 integration test ломается на dim=384 — LOW (плановое D2)
`tests/integration/mcp_stdio.rs` использует toy-вектор. Task 6 обновляет его
(384-dim seed-вектор либо переход на `index_document`).

### R6 — Время `cargo test` из-за загрузки модели — LOW
Каждый proptest-кейс не должен заново грузить embedder. **Митигация:** в `mod tests`
у `src/embed/mod.rs` использовать `OnceLock<Embedder>` или эквивалент. Testing
agent ответственен за это в Task 6.

---

## Execution Notes

### Group D parallel запуск
Task 4 (mcp) и Task 5a (core) — два subagent'а одновременно. Файловые зоны не пересекаются:
- Task 4 → `src/mcp/tools.rs`, `src/mcp/mod.rs`
- Task 5a → `src/index/brute.rs`

После завершения обоих — Group E (Task 5b) последовательно.

### TDD порядок внутри каждой задачи
1. Failing test (`cargo test` → red)
2. Минимальная реализация → green
3. Refactor
4. Следующий тест

Особенно строго для Task 2 (Embedder) — proptest на L2-норму и определённость
пишутся **до** реализации `embed()`.

### Quality Gate (Task 6.5)
Обязательный прогон после Task 6:
```bash
cargo fmt --check && \
cargo clippy --all-targets --all-features -- -D warnings && \
cargo test --all-features
```
Все три должны быть зелёными перед стартом Task 7.

### Docs последовательность (Task 7)
1. Создать `docs/components/embed.md`
2. Обновить `docs/components/mcp-server.md` секциями `index_document` / `search_document`
3. Обновить `docs/coverage.md`: Phase 2 → status `documented`
4. Обновить `docs/ROADMAP.md`: Phase 2 строка → ✅ Complete
5. (опц.) `docs/PHASE-2-PRESENTATION.md`

---

## Plan Update Required

После approval этого dispatch — добавить в `phase-2-embeddings-plan.md` явную
разбивку Task 5 → 5a + 5b (см. Boundary Note выше). Это держит план и dispatch
синхронизированными.
