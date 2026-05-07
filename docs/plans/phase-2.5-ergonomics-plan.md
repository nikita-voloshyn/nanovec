# Plan: Phase 2.5 — Search Ergonomics & UX Fixes

## Summary

Закрыть 4 UX-проблемы NanoVec, выявленные при индексации Montu MVP Metrics
Framework (19 чанков по разделам) и серии семантических запросов. Добавить
bulk-операцию `clear`, расширить `stats` per-tool default metrics и embedder
info, нормализовать score (добавить distance + similarity), задокументировать
ограничения MiniLM-L6-v2 на не-английских языках.

Это не полноценная фаза — это инкремент Phase 2.5: ~200 LoC кода + ~100 LoC
тестов + ~80 LoC документации, без новых компонентов и без изменения архитектуры.

## Test Findings That Drove This Plan

| Симптом | Причина | Что делаем |
|---------|---------|------------|
| Очистка БД = 24 ручных `delete` | Нет bulk-операции | Task 3: `clear` MCP tool |
| `stats` пишет `metric: Euclidean`, но `search_document` считает cosine | Server-wide default ≠ per-tool default | Task 4: новая структура stats |
| `score` возвращается как distance (меньше = ближе), это нигде в API не объяснено | Терминологическая дыра | Task 5: добавить `distance` + `similarity` |
| MiniLM-L6-v2 на русском промахивается на 5 позиций | Известное ограничение модели | Task 7: документация ограничений |

## Out of Scope (явно вынесено)

- Замена embedding-модели → отдельная фаза (e5-multilingual / bge-m3)
- Server-side query rewriting → отдельная фаза (требует LLM или synonym-dict)
- Hybrid search (BM25 + dense) и rerank → отдельная фаза
- Bulk `index` → выигрыш над параллельными `index_document` минимальный

## Design Decisions

| # | Решение | Rationale |
|---|---------|-----------|
| D1 | `clear` reset'ит `next_id` в 0 | Полная очистка состояния — ID-счётчик часть состояния. После clear ID-пространство начинается заново. |
| D2 | `score` остаётся в wire-формате как alias для `distance` | Backward compatibility: существующие клиенты, читающие `score`, не ломаются. |
| D3 | `similarity` для dot product = `-distance` (= raw dot product) | Раз `compute` возвращает `-dot_product`, обратное преобразование = `-distance` = исходный dot product. Не нормализуется в [0,1] — это семантика метрики. |
| D4 | Top-level `metric` в stats остаётся как alias для `default_metric.raw_vector` | Backward compatibility: phase-1/2 клиенты, читающие `metric`, продолжают работать. |
| D5 | Similarity вычисляется на MCP-сериализации, не в `BruteForce::SearchResult` | `SearchResult` остаётся простым (id, score, text, metadata). Конверсия — презентационная задача, знаем метрику в месте вызова. |
| D6 | `Embedder::MODEL_NAME` — публичная константа | Stats нужен canonical model identifier; константа честнее, чем хардкод в mcp/mod.rs. |

## Tasks

### Task 1: VectorStore::clear() and RecordStore::clear()
- **Agent:** core
- **Files:** `src/store/mod.rs`, `src/store/record.rs`
- **Depends on:** none
- **Acceptance:**
  - `cargo test --lib store::tests::clear_resets_count` — VectorStore: count=0, dimension не меняется, следующий insert даёт offset=0.
  - `cargo test --lib store::record::tests::clear_resets_state` — RecordStore: count=0, next_id=0, следующий insert даёт id=0.

**Детали:**
```rust
// src/store/mod.rs
impl VectorStore {
    /// Remove all vectors. Dimension lock is preserved.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.count = 0;
    }
}

// src/store/record.rs
impl RecordStore {
    /// Remove all records and reset the ID counter to 0.
    pub fn clear(&mut self) {
        self.records.clear();
        self.id_to_idx.clear();
        self.next_id = 0;
    }
}
```

TDD-цикл: написать падающие тесты (`clear_resets_count`, `clear_resets_state`,
`insert_after_clear_starts_from_zero`) → имплементировать → green.

---

### Task 2: Embedder model_name accessor
- **Agent:** core
- **Files:** `src/embed/mod.rs`
- **Depends on:** none
- **Acceptance:**
  - `pub const MODEL_NAME: &str = "sentence-transformers/all-MiniLM-L6-v2";` экспортирован.
  - `Embedder::model_name(&self) -> &'static str` возвращает `MODEL_NAME`.
  - Существующий `dimension()` остаётся.
  - Unit-тест: `Embedder::MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"`.

**Детали:** хардкод имени модели уже есть внутри `Embedder::load()` — просто
продвинуть его в `pub const` на уровне модуля и добавить геттер.

---

### Task 3: `clear` MCP tool
- **Agent:** mcp
- **Files:** `src/mcp/mod.rs`, `src/mcp/tools.rs` (без params struct — пустой ввод)
- **Depends on:** Task 1
- **Acceptance:**
  - Новый `#[tool(name = "clear", description = "Remove all indexed documents")]` handler.
  - Без параметров (или `Parameters<()>`).
  - Возвращает `{"deleted": N}` — N это count *до* clear.
  - Логирует `tracing::info!(count = N, "cleared store")`.
  - Integration test: index 3 docs → `clear` → stats.count == 0 → index 1 doc → id == 0.

**Детали:**
```rust
#[tool(name = "clear", description = "Remove all indexed documents")]
fn clear(&self) -> Result<String, String> {
    let mut state = self.state.lock().map_err(|e| format!("lock error: {e}"))?;
    let deleted = state.records.count();
    state.store.clear();
    state.records.clear();
    tracing::info!(count = deleted, "cleared store");
    Ok(serde_json::json!({ "deleted": deleted }).to_string())
}
```

---

### Task 4: Stats restructure
- **Agent:** mcp
- **Files:** `src/mcp/mod.rs`
- **Depends on:** Task 2
- **Acceptance:** `stats` возвращает:
  ```json
  {
    "count": 0,
    "dimension": 384,
    "default_metric": {
      "raw_vector": "euclidean",
      "document": "cosine"
    },
    "embedder": {
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "dim": 384
    },
    "metric": "Euclidean"
  }
  ```
  - `metric` (top-level) остаётся как alias для backward compat (D4).
  - `default_metric.raw_vector` всегда отражает `state.metric` (server-wide).
  - `default_metric.document` хардкод-строка `"cosine"` (соответствует поведению `search_document`).

**Детали:** добавить вычисление новых полей перед `serde_json::json!()`,
использовать `Embedder::MODEL_NAME` через `state.embedder.model_name()`.

---

### Task 5: Score normalization in search/search_document
- **Agent:** mcp
- **Files:** `src/mcp/mod.rs`
- **Depends on:** none (но физически после Task 3, 4 — общий файл)
- **Acceptance:** Каждый результат `search` и `search_document` теперь содержит:
  ```json
  {
    "id": 0,
    "text": "...",
    "metadata": {...},
    "score": 0.311,
    "distance": 0.311,
    "similarity": 0.689
  }
  ```
  - `score` = `distance` = raw output of `Distance::compute` (backcompat alias, D2).
  - `similarity` вычисляется по метрике, использованной в этом вызове:
    - **Cosine:** `1.0 - distance`
    - **Euclidean:** `1.0 / (1.0 + distance)`
    - **DotProduct:** `-distance` (D3)
  - Для `search` — берём `metric` из `parse_metric(...)`.
  - Для `search_document` — берём `metric` после `parse_metric`.

**Детали:**
```rust
fn similarity_for(metric: Metric, distance: f32) -> f32 {
    match metric {
        Metric::Cosine => 1.0 - distance,
        Metric::Euclidean => 1.0 / (1.0 + distance),
        Metric::DotProduct => -distance,
    }
}
```
Вызывается в обоих `search` и `search_document` при сериализации.

---

### Task 6: Integration tests for new MCP surface
- **Agent:** testing
- **Files:** `tests/integration/mcp_phase25.rs` (новый файл) + регистрация в `tests/integration/main.rs`
- **Depends on:** Tasks 3, 4, 5
- **Acceptance:**
  - `test_clear_resets_state` — index 3 vectors → `clear` → stats.count == 0 → index 1 → id == 0
  - `test_stats_shape` — `stats` отдаёт все ключи: `count`, `dimension`, `default_metric.raw_vector`, `default_metric.document`, `embedder.model`, `embedder.dim`, `metric` (alias)
  - `test_search_returns_similarity_cosine` — `search_document` с cosine: для exact match similarity ≈ 1.0
  - `test_search_returns_similarity_euclidean` — `search` с metric="euclidean": similarity = 1/(1+distance)
  - Тесты используют raw vectors (не требуют embedder cold start) для скорости там, где можно

**Детали:** следовать pattern из `tests/integration/mcp_stdio.rs` — spawn binary, send JSON-RPC over stdio, assert на response shape.

---

### Task 7: Documentation updates
- **Agent:** docs
- **Files:**
  - `docs/components/embed.md` (новый раздел "Model Limitations")
  - `docs/components/mcp-server.md` (обновить tool reference + stats shape + search response shape)
  - `README.md` (tools table + brief mention of similarity field)
  - `README.uk.md` (то же)
  - `docs/coverage.md` (новая секция "Phase 2.5 (Complete — Ergonomics)")
  - `CLAUDE.md` (обновить статус: "Phase 2.5 complete, Phase 3 next")
- **Depends on:** Tasks 1–5
- **Acceptance:**
  - **embed.md**: новый раздел с явным указанием — модель trained на английском, multilingual capability ограничен, рекомендация: для real-world не-английского контента использовать query preprocessing или дождаться замены модели в Phase X.
  - **mcp-server.md**: добавить `clear` в tool reference; обновить stats shape; добавить описание `distance` / `similarity` / `score` в search response.
  - **README.md / README.uk.md**: в таблице MCP tools добавить строку `clear`. Под таблицей — короткий note: "Search responses include both `distance` (raw, lower = closer) and `similarity` (normalized, higher = closer)."
  - **coverage.md**: новая секция "Phase 2.5 (Complete — Ergonomics)" со списком: clear tool, stats restructure, score normalization, model-limitations doc.
  - **CLAUDE.md**: одну строку статуса.

## Sequencing

```
Task 1 (store::clear)  ─┐
                        ├→ Task 3 (clear tool) ─┐
Task 2 (embedder name) ─┴→ Task 4 (stats)       ├→ Task 6 (tests) → Task 7 (docs)
                           Task 5 (similarity) ─┘
```

- **Tasks 1 и 2 параллельно** — разные модули (`src/store/`, `src/embed/`), нет зависимостей.
- **Tasks 3, 4, 5 — последовательно** — все трогают `src/mcp/mod.rs`. Каждый отдельный коммит.
- **Task 6** — после полной MCP-поверхности. End-to-end тесты как gate перед документацией.
- **Task 7** — финальный, читает финальный код для точности.

## Quality Gate

После всех tasks:
```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Все три должны быть green до /assign → /execute завершения.

## Estimated Effort

| Task | LoC | Time est. |
|------|-----|-----------|
| 1. store::clear | ~30 + 4 tests | 15 min |
| 2. embedder name | ~10 + 1 test | 10 min |
| 3. clear tool | ~20 | 15 min |
| 4. stats restructure | ~30 | 20 min |
| 5. similarity normalization | ~25 | 15 min |
| 6. integration tests | ~120 | 40 min |
| 7. docs | ~80 | 30 min |
| **Total** | **~315 LoC** | **~2.5 h** |

Не полноценная фаза — это targeted UX patch.

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Стары MCP-клиенты ломаются на новом stats shape | низкая | Top-level `metric` остаётся (D4); все существующие поля сохранены |
| `score` исчезает из search response → ломаются клиенты | низкая | `score` остаётся как alias для `distance` (D2) |
| Integration tests флейкают из-за embedder cold start | средняя | Использовать `index_vector` (raw path) в тестах clear/stats — не требует embedder load |
| `similarity` для dot product не в [0,1] вызывает confusion | средняя | Документировать в mcp-server.md: "DotProduct similarity is unbounded — use cosine for normalized scores" |

## Next Steps After This Plan

1. `/assign` — мапить задачи на агентов с дедлайнами и порядком
2. `/execute` — выполнять taskwise, коммитить каждый task отдельным коммитом
3. После задач 1–7: `/check` (full quality pipeline) + `/docs` (coverage audit)
4. Финальный коммит `feat: phase 2.5 — search ergonomics & UX fixes` (squash или merge)
