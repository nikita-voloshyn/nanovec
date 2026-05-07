# Dispatch: Phase 1 MVP — Zero → Working

## Schedule

| Order | Task | Agent | Skill | Parallel Group | Status |
|-------|------|-------|-------|---------------|--------|
| 1 | Project Scaffold | core | — | A | done |
| 2 | VectorStore | core | — | B (parallel) | done |
| 3 | RecordStore | core | — | B (parallel) | done |
| 4 | Scalar Distance Functions | core | — | B (parallel) | done |
| 5 | BoundedMaxHeap + Brute-Force KNN + Delete | core | — | C | done |
| 6 | MCP Server (stdio transport) | mcp | — | D | done |
| 7 | Tests & Proptest Baseline | testing | — | E | done |
| 7.5 | Quality Gate | — | `/check` | E | done |
| 8 | Documentation | docs | `/docs` | F | done |

**Group B (Tasks 2, 3, 4):** запускаются параллельно через `superpowers:dispatching-parallel-agents`
после завершения Group A.

---

## Agent Validation

### Task 1 — core ✅
- `Cargo.toml`, `rust-toolchain.toml`, `src/lib.rs` — project-level файлы
- Нет пересечений с запрещёнными доменами
- Нет unsafe кода

### Task 2 — core ✅
- `src/store/mod.rs` — явно в domain core (`src/store/`)
- TDD: тесты пишутся **до** реализации (#[cfg(test)] mod tests внутри файла)
- Нет unsafe, нет SIMD, нет MCP

### Task 3 — core ✅
- `src/store/record.rs` — явно в domain core
- Нет пересечений
- Нет unsafe кода

### Task 4 — core ✅
- `src/distance/mod.rs`, `euclidean.rs`, `cosine.rs`, `dot.rs` — domain core
- Только scalar реализации (SIMD — Phase 2, domain simd)
- Нет unsafe кода

### Task 5 — core ✅
- `src/heap/mod.rs`, `src/index/mod.rs`, `src/index/brute.rs` — domain core
- Нет SIMD (brute-force = scalar loops)
- Нет unsafe кода

### Task 6 — mcp ✅
- `src/mcp/mod.rs`, `src/mcp/tools.rs`, `src/server/mod.rs`, `src/main.rs`
- Все файлы в domain mcp (`src/mcp/`, `src/server/`)
- `src/main.rs` — entry point сервера, mcp domain owns it
- `Arc<Mutex<...>>` — нет unsafe (std примитивы)
- Запрещено: вектор матем., SIMD, бенчмарки ✅

### Task 7 — testing ✅ (с boundary note)
- `tests/integration/mcp_tools.rs` — явно в domain testing (`tests/`)
- Proptest в `src/distance/` — **boundary note:** testing добавляет только в `#[cfg(test)] mod tests` секции внутри существующих core файлов. Не модифицирует production код.
- Согласно CLAUDE.md: "Unit tests go in `#[cfg(test)] mod tests` within the source file" — это зона testing agent.

### Task 8 — docs ✅
- `docs/components/*.md`, `docs/coverage.md` — явно в domain docs
- Не трогает `src/`

---

## Risks

### Risk 1: rmcp API версия (Task 6) — MEDIUM
**Проблема:** rmcp crate активно развивается; API в документации (06-MCP-INTEGRATION.md) написан раньше и может устареть.
**Митигация:** mcp agent обязан проверить `crates.io/crates/rmcp` и актуальную документацию перед реализацией. Использовать последнюю стабильную версию.

### Risk 2: Parallel Tasks 2, 3, 4 — LOW
**Проблема:** Если выполнять последовательно в одном контексте — потеря времени. Если параллельно субагентами — конфликты в `src/lib.rs` (объявление модулей).
**Митигация:** Dispatch agent обновляет `src/lib.rs` отдельно после завершения Group B, или core agent включает все объявления модулей в Task 1 scaffold.
**Рекомендация:** добавить в Task 1 все `mod` объявления сразу.

### Risk 3: NanoVecState dimension initialization (Task 6) — LOW
**Проблема:** VectorStore требует dimension при создании, но в MCP сервер dimension неизвестен до первого `index_vector` вызова.
**Митигация:** `Option<VectorStore>` в NanoVecState; инициализируется при первом index_vector, последующие вызовы валидируют размерность.

---

## Execution Notes

### Group B параллельный запуск
При выполнении Tasks 2, 3, 4 одновременно — каждый subagent работает в своём файле:
- Task 2 → только `src/store/mod.rs`
- Task 3 → только `src/store/record.rs`
- Task 4 → только `src/distance/{mod,euclidean,cosine,dot}.rs`

Конфликтов нет. `src/lib.rs` обновляется после через dispatch.

### TDD порядок внутри каждой задачи
1. Написать failing test (`cargo test` → red)
2. Написать минимальную реализацию → green
3. Refactor
4. Следующий тест

### Quality Gate (Task 7.5)
После Task 7 — обязательный прогон:
```bash
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features
```
Все три команды должны пройти чисто.
