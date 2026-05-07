# NanoVec

[English →](README.md)

> Легкий in-memory векторний DB без зовнішніх залежностей — у вигляді MCP-сервера для ефемерної робочої пам'яті AI-агентів.

NanoVec — реалізація векторної бази даних на чистому Rust, написана з нуля для AI-агентів, яким потрібна короткочасна семантична пам'ять. Цілі: семантичний пошук за сабмілісекунду, мініатюрний статичний бінарник і єдиний зовнішній інтерфейс — Model Context Protocol через stdio.

**Статус:** Phase 1 (MVP MCP-сервер) і Phase 2 (серверні ембеддинги) завершені. Що далі — у [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Що всередині

- **In-process векторне сховище.** Плоский `Vec<f32>` у Structure-of-Arrays розкладці, фіксована розмірність, O(1) insert / get / swap-remove.
- **Серверний ембеддинг.** Вбудована модель `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-нормалізована) через [`candle`](https://github.com/huggingface/candle) — чистий Rust, тільки CPU, без Python чи ONNX.
- **Brute-force KNN** з метриками Евкліда, косинусу та dot-product. KD-Tree і SIMD-прискорення — у наступних фазах.
- **MCP через stdio.** Шість інструментів: `index_vector`, `search`, `index_document`, `search_document`, `delete`, `stats`.
- **Ефемерний за дизайном.** Без персистенції й WAL — дані живуть у RAM і зникають разом із процесом.
- **Нуль зовнішніх індексаційних бібліотек.** Без FAISS, без HNSWLIB. Усі алгоритми написані від першопринципів.

## Швидкий старт

### Збірка

```bash
git clone https://github.com/nikita-voloshyn/nanovec
cd nanovec
cargo build --release
```

Бінарник — `target/release/nanovec`.

### Підключення до Claude Code (або будь-якого MCP-клієнта)

Додай бінарник у конфіг MCP-клієнта. Приклад для Claude Code (`.mcp.json` у корені репо):

```json
{
  "mcpServers": {
    "nanovec": {
      "command": "/absolute/path/to/nanovec/target/release/nanovec"
    }
  }
}
```

Під час першого запуску NanoVec завантажує модель ембеддингів (~90 МБ) з HuggingFace Hub у `~/.cache/huggingface/hub/`. Холодний старт — 10–30 с; наступні запуски прогріваються за ~30 мс.

Усі логи йдуть у stderr; stdout зарезервований під MCP JSON-RPC.

## MCP-інструменти

| Інструмент | Вхід | Вихід |
|------------|------|-------|
| `index_document` | `text`, опційно `metadata` | `{ "id": u64 }` |
| `search_document` | `query`, `k`, опційно `metric` | `[{ id, text, metadata, score }]` |
| `index_vector` | `text`, `vector` (довжина 384), опційно `metadata` | `{ "id": u64 }` |
| `search` | `vector` (довжина 384), `k`, опційно `metric` | `[{ id, text, metadata, score }]` |
| `delete` | `id` | `{ "deleted": bool }` |
| `stats` | — | `{ "count": usize, "dimension": usize }` |

За замовчуванням метрика — `euclidean` для raw-vector шляху і `cosine` для document шляху. Перевизначається через `"euclidean"`, `"cosine"` або `"dot"`.

Розмірність вектора фіксується на 384 у момент завантаження ембеддера. Виклики `index_vector` з іншою розмірністю відхиляються.

## Архітектура

```
src/
  store/       VectorStore (плоский SoA), RecordStore (текст + metadata + ID-мапа)
  distance/    Скалярні Евклід, косинус, dot
  heap/        Обмежений max-heap для top-K
  index/       Brute-force KNN
  embed/       candle + завантаження all-MiniLM-L6-v2
  mcp/         rmcp визначення інструментів, роутер сервера
  server/      Async-точка входу, прогрів ембеддера, stdio
```

Деталі по кожному модулю — у [`docs/components/`](docs/components/).

## Документація

- [`docs/README.md`](docs/README.md) — двомовний індекс документації
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — план фаз
- [`docs/PHASE-1-PRESENTATION.md`](docs/PHASE-1-PRESENTATION.md) — підсумок MVP
- [`docs/PHASE-2-PRESENTATION.md`](docs/PHASE-2-PRESENTATION.md) — підсумок ембеддингів
- [`docs/components/`](docs/components/) — довідник по модулях
- [`docs/plans/`](docs/plans/) — плани фаз і диспатч-звіти
- [`docs/TEST-CASES.md`](docs/TEST-CASES.md) — каталог тестів

## Розробка

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

NanoVec використовує TDD-first / YAGNI-підхід. Налаштування агентів і skill'ів, що керують щоденною розробкою, — у [`CLAUDE.md`](CLAUDE.md).

## Ліцензія

MIT.
