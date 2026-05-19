# NanoVec — Roadmap

Каждая фаза = одна чёткая трансформация с measurable результатом.

---

## Phase 1 — Zero → Working
> *Claude може використовувати NanoVec як інструмент пам'яті прямо зараз*

**До:** ничего нет
**После:** подключается к Claude Code через MCP, индексирует векторы, делает semantic search

Deliverables:
- `VectorStore` — flat `Vec<f32>`, SoA layout, dimension locked at init
- `RecordStore` — `Vec<VectorRecord>` + `HashMap<u64, usize>`
- Scalar distance functions: L2, cosine, dot product
- Brute-force KNN search (linear scan)
- Delete via swap-remove
- MCP server (stdio transport): `index_vector`, `search`, `delete`, `stats`
- Unit tests + proptest baseline

---

## Phase 2 — Vectors → Text
> *Агент пише текст — NanoVec сам робить ембеддинг і зберігає*

**До:** клієнт передає вектор `[0.1, 0.3, ...]`
**Після:** клієнт передає текст `"зустріч з командою в п'ятницю"` — сервер ембеддить сам

```
// До
index_vector(vector: [...768 floats...], metadata: {...})

// Після
index_document(text: "зустріч з командою в п'ятницю", metadata: {...})
search(query: "які зустрічі планувались?", k: 5)
```

Deliverables:
- **candle** (HuggingFace, pure Rust) — embedding runtime, без ONNX/Python
- Модель: `all-MiniLM-L6-v2` (384-dim), завантажується з HuggingFace Hub при першому запуску
- Новий MCP tool: `index_document(text, metadata)` → embed → store вектор + raw text
- `search(query_text, k)` → embed query → brute-force KNN → повертає текст + score
- `index_vector` залишається для сумісності (raw вектори)
- Math crates дозволені: `ndarray`, `faer` тощо
- Тести: embedding roundtrip, semantic similarity ordering

---

## Phase 3 — Scalar → SIMD
> *Distance computation прискорюється в 8-16x з доказами*

**До:** скалярний цикл — `a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum()`
**Після:** AVX2 (x86_64) / NEON (ARM) — 8 float'ів за такт

Deliverables:
- AVX2 реалізації: L2, cosine, dot product
- NEON реалізації для Apple Silicon
- Portable SIMD fallback (`std::simd`, nightly)
- Runtime dispatch: `is_x86_feature_detected!("avx2")`
- Criterion benchmarks: scalar vs SIMD на 384-dim векторах
- Proptest: SIMD output == scalar output ± epsilon

---

## Phase 4 — Pile → Organized
> *Агент може структурувати пам'ять, а не звалювати все в одну купу*

**До:** один global store, search по всьому
**Після:** іменовані collections + metadata filtering

```
// До
search(query="мої задачі", k=5)

// Після
search(query="мої задачі", k=5, collection="work", filter="date > 2024-01")
```

Deliverables:
- Named collections (multi-tenant одного процесу)
- Metadata filtering (прості предикати: eq, gt, lt, contains)
- Batch `index_many` + `search_many`
- MCP tools: `create_collection`, `list_collections`, `drop_collection`

---

## Phase 5 — O(n) → O(log n)
> *Перший індекс: пошук перестає бути лінійним скануванням*

**До:** кожен query = повний прохід по всіх векторах
**Після:** KD-Tree для низьких розмірностей

Deliverables:
- KD-Tree: побудова O(n log n), KNN з geometric pruning
- Auto-select: brute-force якщо dim > 50, KD-Tree якщо dim < 50
- Benchmark по розмірностях: 8 / 16 / 32 / 64 / 128 / 384 dim
- Чесний аналіз curse of dimensionality з числами
- MCP tool: `build_index`, `index_stats`

---

## Phase 6 — Single → Multi-agent
> *Кілька агентів працюють одночасно з ізольованими колекціями*

**До:** один процес, один клієнт, stdio
**Після:** `Arc<RwLock<>>` + SSE transport — паралельні читачі, безпечні записи

Deliverables:
- `Arc<RwLock<VectorDatabase>>` — many readers, exclusive writer
- SSE transport (axum) поряд зі stdio
- Connection management
- Memory budget + базова eviction policy (LRU)
- Load test: N одночасних search запитів

---

## Phase 7 — Exact → Approximate (stretch)
> *384-dim embeddings отримують production-grade індекс*

**До:** brute-force при 384 dim — O(n)
**Після:** HNSW — O(log n) при будь-якій розмірності

Deliverables:
- HNSW з нуля: граф побудови, greedy search, layer selection
- Параметри: `M` (connections), `ef_construction`, `ef_search`
- Recall vs latency tradeoff benchmark
- Порівняння з Qdrant (in-memory mode) на однаковому залізі
- MCP: прозорий fallback brute-force → KD-Tree → HNSW по розмірності

---

## Status

| Phase | Status |
|-------|--------|
| Phase 1 — Zero → Working | ✅ Complete |
| Phase 2 — Vectors → Text | ✅ Complete |
| Phase 3 — Scalar → SIMD | ✅ Complete |
| Phase 4 — Pile → Organized | ✅ Complete |
| Phase 5 — O(n) → O(log n) | ✅ Complete |
| Phase 6 — Single → Multi-agent | ⬜ Next |
| Phase 7 — Exact → Approximate | ⬜ Stretch |
