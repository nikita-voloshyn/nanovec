# Plan: Phase 1 MVP — Zero → Working

## Summary

Построить минимальный рабочий MCP-сервер NanoVec с нуля. После фазы 1 Claude Code
может подключиться к NanoVec через stdio, проиндексировать векторы, выполнить
семантический поиск и получить статистику. Код на Rust: VectorStore (SoA),
RecordStore, скалярные функции расстояния, brute-force KNN, MCP-инструменты.

---

## Tasks

### Task 1: Project Scaffold
- **Agent:** core
- **Files:** `Cargo.toml`, `rust-toolchain.toml`, `src/lib.rs`, `.gitignore` (update)
- **Depends on:** none
- **Acceptance:** `cargo build` проходит без ошибок; `cargo fmt --check` и `cargo clippy` чистые

**Детали:**
- Cargo.toml: binary crate `nanovec`, edition 2021
- Зависимости: `rmcp`, `tokio` (full), `serde`, `serde_json`, `tracing`, `tracing-subscriber`
- Dev-зависимости: `proptest`, `approx`
- rust-toolchain.toml: stable (nightly только для portable_simd — Phase 2)
- Feature gates: пустые пока, готовим под `simd` в Phase 2

---

### Task 2: VectorStore
- **Agent:** core
- **Files:** `src/store/mod.rs`
- **Depends on:** Task 1
- **Acceptance:** unit tests в файле проходят (`cargo test store`)

**API:**
```rust
pub struct VectorStore { /* flat Vec<f32>, dimension: usize, count: usize */ }

impl VectorStore {
    pub fn new(dimension: usize) -> Self
    pub fn insert(&mut self, vector: &[f32]) -> Result<usize, StoreError>  // возвращает offset
    pub fn get(&self, offset: usize) -> Option<&[f32]>
    pub fn swap_remove(&mut self, offset: usize) -> Result<usize, StoreError>  // возвращает новый offset для перемещённого вектора
    pub fn count(&self) -> usize
    pub fn dimension(&self) -> usize
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f32])>
}

pub enum StoreError { DimensionMismatch { expected: usize, got: usize }, InvalidOffset(usize) }
```

**Тесты (TDD — написать до реализации):**
- insert возвращает корректный offset
- get после insert возвращает тот же вектор
- insert с неверной размерностью → DimensionMismatch
- count растёт после insert
- swap_remove уменьшает count

---

### Task 3: RecordStore
- **Agent:** core
- **Files:** `src/store/record.rs`
- **Depends on:** Task 1
- **Acceptance:** unit tests в файле проходят (`cargo test record`)

**API:**
```rust
pub struct VectorRecord {
    pub id: u64,
    pub offset: usize,       // позиция в VectorStore
    pub text: String,        // исходный текст
    pub metadata: Vec<(String, String)>,  // key-value теги
}

pub struct RecordStore {
    records: Vec<VectorRecord>,
    id_to_idx: HashMap<u64, usize>,
    next_id: u64,
}

impl RecordStore {
    pub fn new() -> Self
    pub fn insert(&mut self, text: String, metadata: Vec<(String, String)>, offset: usize) -> u64
    pub fn get(&self, id: u64) -> Option<&VectorRecord>
    pub fn remove(&mut self, id: u64) -> Option<VectorRecord>
    pub fn update_offset(&mut self, id: u64, new_offset: usize)  // для swap_remove
    pub fn count(&self) -> usize
    pub fn iter(&self) -> impl Iterator<Item = &VectorRecord>
}

pub enum RecordError { NotFound(u64) }
```

**Тесты:**
- insert → get возвращает тот же record
- remove → get возвращает None
- update_offset корректно обновляет offset в HashMap

---

### Task 4: Scalar Distance Functions
- **Agent:** core
- **Files:** `src/distance/mod.rs`, `src/distance/euclidean.rs`, `src/distance/cosine.rs`, `src/distance/dot.rs`
- **Depends on:** Task 1
- **Acceptance:** unit tests с известными векторами проходят; функции возвращают верные значения с точностью 1e-6

**API:**
```rust
// src/distance/mod.rs
pub trait Distance: Send + Sync {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

pub enum Metric { Euclidean, Cosine, DotProduct }

pub fn distance_fn(metric: Metric) -> Box<dyn Distance>;

// Каждый модуль экспортирует:
pub fn euclidean(a: &[f32], b: &[f32]) -> f32;  // L2 distance
pub fn cosine(a: &[f32], b: &[f32]) -> f32;      // 1 - cosine_similarity
pub fn dot_product(a: &[f32], b: &[f32]) -> f32; // raw dot product (выше = ближе)
```

**Тесты с known values:**
- euclidean([0,0], [3,4]) == 5.0
- cosine([1,0,0], [1,0,0]) == 0.0 (identical)
- cosine([1,0,0], [0,1,0]) == 1.0 (orthogonal)
- dot_product([1,2,3], [4,5,6]) == 32.0

---

### Task 5: BoundedMaxHeap + Brute-Force KNN + Delete
- **Agent:** core
- **Files:** `src/heap/mod.rs`, `src/index/mod.rs`, `src/index/brute.rs`
- **Depends on:** Task 2, 3, 4
- **Acceptance:** KNN возвращает корректный top-K; delete корректно убирает вектор

**API:**
```rust
// src/heap/mod.rs
pub struct BoundedMaxHeap { capacity: usize, items: Vec<(f32, u64)> }
impl BoundedMaxHeap {
    pub fn new(capacity: usize) -> Self
    pub fn push(&mut self, score: f32, id: u64)  // вытесняет худший если полон
    pub fn into_sorted_vec(self) -> Vec<(f32, u64)>  // ascending по score
}

// src/index/brute.rs
pub struct BruteForce;
impl BruteForce {
    pub fn search(
        store: &VectorStore,
        records: &RecordStore,
        query: &[f32],
        k: usize,
        metric: Metric,
    ) -> Vec<SearchResult>
}

pub struct SearchResult { pub id: u64, pub score: f32, pub text: String }

// delete: координируется через VectorStore::swap_remove + RecordStore::remove + RecordStore::update_offset
pub fn delete(store: &mut VectorStore, records: &mut RecordStore, id: u64) -> Result<(), DeleteError>
pub enum DeleteError { NotFound(u64) }
```

**Тесты:**
- Вставить 5 векторов, найти 3 ближайших к query → правильный порядок
- Delete существующего вектора → count уменьшается, search не находит его
- Delete несуществующего → DeleteError::NotFound
- BoundedMaxHeap: push K+1 элементов → размер остаётся K, лучшие K сохранены

---

### Task 6: MCP Server (stdio transport)
- **Agent:** mcp
- **Files:** `src/mcp/mod.rs`, `src/mcp/tools.rs`, `src/server/mod.rs`, `src/main.rs`
- **Depends on:** Task 2, 3, 4, 5
- **Acceptance:** `echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | cargo run` возвращает список инструментов

**MCP Tools:**

```
index_vector(text: string, vector: number[], metadata?: object) → { id: number }
search(vector: number[], k: number, metric?: "euclidean"|"cosine"|"dot") → [{ id, score, text }]
delete(id: number) → { success: bool }
stats() → { count: number, dimension: number, metric: string }
```

**Правила реализации:**
- Весь logging → stderr (CRITICAL: stdout только для JSON-RPC)
- Глобальный state через `Arc<Mutex<NanoVecState>>`
- NanoVecState содержит VectorStore + RecordStore + дефолтный Metric
- Dimension определяется первым вызовом `index_vector`; последующие валидируются
- rmcp: использовать текущий стабильный API (проверить crates.io перед имплементацией)

---

### Task 7: Tests & Proptest Baseline
- **Agent:** testing
- **Files:** `tests/integration/mcp_tools.rs`, добавление proptest в unit-тесты distance/
- **Depends on:** Task 4, 5, 6
- **Acceptance:** `cargo test --all-features` полностью зелёный

**Proptest (добавить в src/distance/):**
```rust
proptest! {
    #[test]
    fn euclidean_non_negative(a: Vec<f32>, b: Vec<f32>) { ... }
    #[test]
    fn cosine_range(a: Vec<f32>, b: Vec<f32>) { /* 0.0..=2.0 */ }
    #[test]
    fn dot_product_symmetric(a: Vec<f32>, b: Vec<f32>) { /* a·b == b·a */ }
}
```

**Integration test (MCP stdio):**
- Запустить бинарник, отправить JSON-RPC через stdin, проверить ответ через stdout
- Сценарий: index → search → delete → stats

---

### Task 8: Documentation
- **Agent:** docs
- **Files:** `docs/components/vector-store.md`, `docs/components/record-store.md`, `docs/components/distance.md`, `docs/components/brute-force.md`, `docs/components/mcp-server.md`, `docs/coverage.md`
- **Depends on:** Task 1–7
- **Acceptance:** `docs/coverage.md` показывает Phase 1 как documented

---

## Execution Order

```
Task 1 (scaffold)
  ├── Task 2 (VectorStore)  ──┐
  ├── Task 3 (RecordStore)  ──┤
  └── Task 4 (Distance)     ──┤
                               ├── Task 5 (BruteForce + Heap)
                               └── Task 6 (MCP server) ← зависит от Task 5
                                       └── Task 7 (Tests)
                                               └── Task 8 (Docs)
```

Tasks 2, 3, 4 могут выполняться **параллельно** после Task 1.
Tasks 5 и 6 выполняются **последовательно** в указанном порядке.
