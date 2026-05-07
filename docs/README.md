# NanoVec — Documentation Index / Індекс документації

[← Project README](../README.md) · [← Корінь репо](../README.uk.md)

This directory holds NanoVec's design, planning, and per-component documentation. The table below is the canonical entry point in both English and Ukrainian.

Цей каталог містить дизайн, плани й покомпонентну документацію NanoVec. Таблиця нижче — канонічна точка входу англійською та українською.

---

## Top-level / Верхній рівень

| File / Файл | English | Українською |
|-------------|---------|-------------|
| [`ROADMAP.md`](ROADMAP.md) | Phase plan: scope, deliverables, success criteria for each milestone. | План фаз: обсяг, deliverables і критерії успіху для кожної віхи. |
| [`PHASE-1-PRESENTATION.md`](PHASE-1-PRESENTATION.md) | Walk-through of the Phase 1 MVP: VectorStore, RecordStore, scalar distance, brute-force KNN, MCP stdio server. | Огляд Phase 1 MVP: VectorStore, RecordStore, скалярні дистанції, brute-force KNN, MCP-сервер на stdio. |
| [`PHASE-2-PRESENTATION.md`](PHASE-2-PRESENTATION.md) | Walk-through of Phase 2: candle-based embedder and the document tools (`index_document`, `search_document`). | Огляд Phase 2: ембеддер на candle і document-інструменти (`index_document`, `search_document`). |
| [`TEST-CASES.md`](TEST-CASES.md) | End-to-end test scenarios with hand-crafted vectors and expected ranking. | Сценарії end-to-end тестів із прикладними векторами й очікуваним ранжуванням. |
| [`coverage.md`](coverage.md) | Documentation coverage map — which modules have component docs. | Карта покриття документацією — які модулі мають компонентні описи. |
| [`approaches-reference.md`](approaches-reference.md) | Reference for the development approaches the project supports (TDD-first, YAGNI/KISS, …). | Довідник із підходів розробки, що підтримує проєкт (TDD-first, YAGNI/KISS, …). |

## Per-component / Покомпонентно

Files in [`components/`](components/). All component docs are written in English and follow the same template: Purpose, Public API, Internal Design, Error Types, Usage Example, Performance, Test Coverage.

Файли в [`components/`](components/). Усі описи компонентів написані англійською й слідують одному шаблону: Purpose, Public API, Internal Design, Error Types, Usage Example, Performance, Test Coverage.

| File / Файл | English | Українською |
|-------------|---------|-------------|
| [`components/vector-store.md`](components/vector-store.md) | Flat SoA `Vec<f32>` store, dimension lock, swap-remove semantics. | Плоске SoA-сховище `Vec<f32>`, фіксація розмірності, семантика swap-remove. |
| [`components/record-store.md`](components/record-store.md) | Text + metadata + ID mapping; offset synchronization after swap-remove. | Текст + метадані + ID-мапа; синхронізація offset'ів після swap-remove. |
| [`components/distance.md`](components/distance.md) | Scalar Euclidean, cosine, dot product; runtime metric dispatch. | Скалярні Евклід, косинус, dot product; рантайм-вибір метрики. |
| [`components/brute-force.md`](components/brute-force.md) | Bounded max-heap and brute-force KNN linear scan. | Обмежений max-heap і brute-force KNN лінійним скануванням. |
| [`components/embed.md`](components/embed.md) | candle loader for `all-MiniLM-L6-v2`, mean-pool + L2-normalize pipeline. | Завантажувач `all-MiniLM-L6-v2` на candle, pipeline mean-pool + L2-нормалізації. |
| [`components/mcp-server.md`](components/mcp-server.md) | rmcp tool definitions, stdio transport, embedder warmup, dimension lock. | Визначення інструментів rmcp, stdio-транспорт, прогрів ембеддера, фіксація розмірності. |

## Plans / Плани

Files in [`plans/`](plans/). Each phase has a plan (decomposition into tasks) and a dispatch report (which agent did what and when).

Файли в [`plans/`](plans/). Кожна фаза має plan (декомпозицію на задачі) і dispatch-звіт (хто що і коли робив).

| File / Файл | English | Українською |
|-------------|---------|-------------|
| [`plans/phase-1-mvp-plan.md`](plans/phase-1-mvp-plan.md) | Phase 1 task decomposition with agent assignments and acceptance criteria. | Декомпозиція задач Phase 1 з призначеними агентами та критеріями прийомки. |
| [`plans/phase-1-mvp-dispatch.md`](plans/phase-1-mvp-dispatch.md) | Phase 1 execution log: timing, ordering, handoffs between agents. | Журнал виконання Phase 1: тайминг, порядок, передачі між агентами. |
| [`plans/phase-2-embeddings-plan.md`](plans/phase-2-embeddings-plan.md) | Phase 2 task decomposition with design decisions for candle integration. | Декомпозиція задач Phase 2 з дизайн-рішеннями для інтеграції candle. |
| [`plans/phase-2-embeddings-dispatch.md`](plans/phase-2-embeddings-dispatch.md) | Phase 2 execution log. | Журнал виконання Phase 2. |
| [`plans/phase-2-embeddings-report.md`](plans/phase-2-embeddings-report.md) | Phase 2 wrap-up: acceptance results, deviations from plan, follow-ups. | Підсумок Phase 2: результати прийомки, відхилення від плану, follow-up'и. |

## Conventions / Конвенції

- **Per-component docs are English-only** to keep the technical reference single-sourced and avoid translation drift on API tables.
- **Plans and presentations may be bilingual or in the language used during the planning session** — they are historical artefacts, not living references.
- **READMEs are bilingual.** Project root: [`../README.md`](../README.md) (en) and [`../README.uk.md`](../README.uk.md) (uk).

---

- **Покомпонентні описи — лише англійською**, щоб технічний довідник мав одне джерело правди й API-таблиці не розходилися між мовами.
- **Плани й презентації можуть бути двомовними або мовою сесії планування** — це історичні артефакти, не живі довідники.
- **READMEs — двомовні.** Корінь репозиторію: [`../README.md`](../README.md) (en) та [`../README.uk.md`](../README.uk.md) (uk).
