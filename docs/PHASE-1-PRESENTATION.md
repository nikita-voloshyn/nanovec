# NanoVec — Phase 1: Zero → Working

## Что это

NanoVec — лёгкая in-memory векторная база данных на Rust без внешних зависимостей,
реализованная как MCP-сервер. Предназначена для эфемерной рабочей памяти AI-агентов:
агент индексирует эмбеддинги, делает семантический поиск и удаляет ненужные записи
прямо через инструменты Claude Code.

---

## Что реализовано в Phase 1

### 1. VectorStore (`src/store/mod.rs`)

Хранилище векторов на основе **Structure of Arrays (SoA)** — плоский `Vec<f32>`.

- Размерность фиксируется при первом `insert`, все последующие векторы валидируются
- `insert(vector)` → offset
- `get(offset)` → `&[f32]`
- `swap_remove(offset)` — O(1) удаление через замену последним элементом
- `iter()` → итерация по всем векторам
- Собственный тип ошибок: `StoreError::DimensionMismatch`, `StoreError::InvalidOffset`

### 2. RecordStore (`src/store/record.rs`)

Хранилище метаданных, отдельное от векторов.

- Каждая запись: `id: u64`, `text: String`, `metadata: Option<Value>`, `offset: usize`
- `HashMap<u64, usize>` для O(1) поиска записи по ID
- `swap_fix(moved_id, new_offset)` — синхронизация offsets после swap-remove

### 3. Функции расстояния (`src/distance/`)

Три скалярные метрики, каждая в отдельном файле:

| Метрика | Файл | Формула |
|---------|------|---------|
| Euclidean (L2) | `euclidean.rs` | `√Σ(aᵢ - bᵢ)²` |
| Cosine | `cosine.rs` | `1 - (a·b) / (‖a‖·‖b‖)` |
| Dot Product | `dot.rs` | `-Σ(aᵢ·bᵢ)` (negated for min-heap) |

Единый диспатч через `distance_fn(metric: &str)` → `fn(&[f32], &[f32]) -> f32`.

### 4. BoundedMaxHeap (`src/heap/mod.rs`)

Max-heap фиксированного размера для top-K результатов.

- `push(id, score)` — если score меньше максимума в куче, вытесняет его
- `into_sorted_vec()` → результаты в порядке возрастания score (ближайшие первые)
- Сложность: O(log K) на push, O(K log K) для финальной сортировки

### 5. Brute-Force KNN (`src/index/brute.rs`)

Линейный скан по всем векторам.

- `insert(text, vector, metadata)` → `u64` ID
- `search(query, k, metric)` → `Vec<SearchResult>`
- `delete(id)` → `Result<(), IndexError>` — swap-remove + синхронизация RecordStore
- `count()`, `dimension()` — статистика

### 6. MCP-сервер (`src/mcp/`, `src/server/`, `src/main.rs`)

Полный MCP-сервер на [rmcp](https://crates.io/crates/rmcp) с **stdio транспортом**.

Четыре инструмента:

| Инструмент | Параметры | Возвращает |
|------------|-----------|-----------|
| `index_vector` | `text`, `vector`, `metadata?` | `{"id": N}` |
| `search` | `vector`, `k`, `metric?` | `[{id, score, text, metadata}]` |
| `delete` | `id` | `{"success": true}` |
| `stats` | — | `{"count", "dimension", "metric"}` |

Протокол: JSON-RPC 2.0 (newline-delimited). Состояние защищено `Arc<Mutex<>>`.

---

## Архитектурные решения

| Решение | Почему |
|---------|--------|
| SoA layout для векторов | Векторные данные лежат contiguous в памяти → cache-friendly при линейном скане |
| Метаданные отдельно от векторов | На hot path вычислений расстояния метаданные не нужны |
| swap-remove для удаления | O(1) вместо O(n) сдвига; ID никогда не переиспользуются |
| `dimension: null` до первого insert | Клиент не обязан знать размерность заранее |
| Все ошибки — exhaustive enum | Нет `Box<dyn Error>` в публичных API |

---

## Тесты

### Покрытие по модулям

| Модуль | Unit тестов | Тип |
|--------|-------------|-----|
| `store::VectorStore` | 9 | unit |
| `store::RecordStore` | 6 | unit |
| `distance::euclidean` | 7 | unit + proptest |
| `distance::cosine` | 7 | unit + proptest |
| `distance::dot` | 6 | unit + proptest |
| `distance` (dispatch) | 3 | unit |
| `heap::BoundedMaxHeap` | 5 | unit |
| `index::BruteForce` | 6 | unit |
| MCP full flow | 1 | integration |
| **Итого** | **50** | |

**49 тестов пасятся**: 48 unit + 1 integration (0 failed).

### Unit тесты — по группам

#### VectorStore
```
store::tests::insert_returns_correct_offset
store::tests::get_after_insert_returns_same_vector
store::tests::count_grows_after_insert
store::tests::insert_wrong_dimension_errors
store::tests::iter_yields_all_vectors
store::tests::swap_remove_decreases_count
store::tests::swap_remove_last_element_no_swap
store::tests::swap_remove_invalid_offset_errors
```

#### RecordStore
```
store::record::tests::insert_get_roundtrip
store::record::tests::ids_are_unique_and_incrementing
store::record::tests::count_tracks_insertions_and_removals
store::record::tests::remove_returns_none_on_get
store::record::tests::update_offset_changes_stored_offset
store::record::tests::iter_yields_all_records
```

#### Distance functions
```
distance::euclidean::tests::euclidean_identical_vectors     → 0.0
distance::euclidean::tests::euclidean_3_4_triangle          → 5.0
distance::euclidean::tests::euclidean_symmetric
distance::euclidean::tests::euclidean_non_negative
distance::euclidean::tests::euclidean_identity
distance::euclidean::tests::euclidean_single_dimension
distance::euclidean::tests::euclidean_dimension_mismatch_panics

distance::cosine::tests::cosine_identical_vectors           → 0.0
distance::cosine::tests::cosine_orthogonal_vectors          → 1.0
distance::cosine::tests::cosine_opposite_vectors            → 2.0
distance::cosine::tests::cosine_zero_vector_returns_one
distance::cosine::tests::cosine_range                       → [0, 2]
distance::cosine::tests::cosine_identity_is_zero
distance::cosine::tests::cosine_dimension_mismatch_panics

distance::dot::tests::dot_product_simple
distance::dot::tests::dot_product_basic
distance::dot::tests::dot_product_orthogonal                → 0.0
distance::dot::tests::dot_product_symmetric
distance::dot::tests::dot_product_zero_vector               → 0.0
distance::dot::tests::dot_product_dimension_mismatch_panics

distance::tests::distance_fn_euclidean
distance::tests::distance_fn_cosine
distance::tests::distance_fn_dot_product
```

#### BoundedMaxHeap
```
heap::tests::keeps_top_3_smallest_scores
heap::tests::push_more_than_capacity_keeps_k
heap::tests::push_same_score_keeps_k_items
heap::tests::into_sorted_vec_returns_ascending_order
heap::tests::capacity_zero_returns_empty
```

#### BruteForce KNN
```
index::brute::tests::search_empty_store_returns_empty
index::brute::tests::search_5_vectors_k3_returns_3_closest
index::brute::tests::search_k_greater_than_count_returns_all
index::brute::tests::delete_existing_id_decreases_count
index::brute::tests::delete_nonexistent_id_returns_not_found
index::brute::tests::delete_first_vector_swap_remove_remaining_searchable
```

### Интеграционный тест MCP (`tests/integration/mcp_stdio.rs`)

Тест поднимает реальный бинарник `nanovec` как дочерний процесс и общается с ним
через JSON-RPC 2.0 по stdio:

```
test_mcp_full_flow:
  1. initialize                     → проверяет id ответа
  2. notifications/initialized      → (no response)
  3. tools/call index_vector        → проверяет { "id": N }
  4. tools/call search (k=1)        → проверяет id + text совпадают
  5. tools/call delete              → успех
  6. tools/call stats               → count == 0
```

---

## Запуск

```bash
# Все тесты
cargo test --all-features

# Только интеграционный
cargo test test_mcp_full_flow --test integration

# Quality gate (fmt + clippy + test)
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features

# Запуск сервера напрямую
cargo run
```

---

## Структура кода

```
src/
  lib.rs                  # Объявления модулей
  main.rs                 # Точка входа сервера
  store/
    mod.rs                # VectorStore — плоский Vec<f32>, SoA layout
    record.rs             # RecordStore — метаданные + HashMap<u64, usize>
  distance/
    mod.rs                # Диспатч по имени метрики
    euclidean.rs          # L2 расстояние
    cosine.rs             # Косинусное расстояние
    dot.rs                # Dot product (negated)
  heap/
    mod.rs                # BoundedMaxHeap для top-K
  index/
    mod.rs                # (placeholder)
    brute.rs              # Brute-force KNN, delete
  mcp/
    mod.rs                # MCP сервер, обработчики инструментов
    tools.rs              # Параметры инструментов (serde + JsonSchema)
  server/
    mod.rs                # Запуск rmcp stdio сервера
tests/
  integration/
    main.rs
    mcp_stdio.rs          # End-to-end тест через реальный процесс
```

**~1300 строк кода**, 0 внешних зависимостей для векторной математики.

---

## Что дальше — Phase 2

> **Vectors → Text**: серверный embedding через candle (HuggingFace pure Rust) и модель
> all-MiniLM-L6-v2 (384-dim). Новый MCP-инструмент `index_document(text, metadata)` —
> сервер сам считает вектор; `search(query_text, k)` принимает запрос текстом.
