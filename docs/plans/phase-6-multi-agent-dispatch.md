# Dispatch: Phase 6 — Single → Multi-agent

Source plan: [`phase-6-multi-agent-plan.md`](./phase-6-multi-agent-plan.md)

## Domain Routing Validation

Cross-domain risks identified during routing:

| Plan task | Concern | Resolution |
|-----------|---------|------------|
| T1 (refactor RwLock) | Plan lists both `src/store/*` (core domain) and `src/mcp/mod.rs` (mcp domain) | **Split into T1a (core) + T1b (mcp).** Database lifts out of `mcp/mod.rs` into new `src/store/database.rs` (core), then `NanoVecServer` is rewired in `src/mcp/mod.rs` (mcp). |
| T5 (memory budget) | Plan touches `src/store/budget.rs` + `src/mcp/mod.rs` | **Split into T5a (core) + T5b (mcp wire-up).** Wire-up is small enough to merge into T7 (mcp stats update) instead — done that way to keep agent count down. |
| T6 (LRU eviction) | Plan touches `src/mcp/mod.rs` | Same as T5 — wire-up merged into T7. T6 itself is pure core. |
| T8 (benchmark) | `benches/` is testing domain, but bench uses `NanoVecDatabase` API | OK — testing agent imports public API, does not touch `src/`. |

All other plan tasks map cleanly to a single domain.

---

## Schedule

| Order | Task | Agent | Skill | Depends on | Status |
|-------|------|-------|-------|------------|--------|
| 1 | T1a — Extract `NanoVecDatabase` into `src/store/database.rs` with two-level `RwLock` (CollectionMap + per-Collection) | core | — | none | pending |
| 2 | T1b — Rewire `NanoVecServer` to hold `Arc<NanoVecDatabase>` (drop external `Mutex`) | mcp | — | 1 | pending |
| 3 | T2 — Concurrency proptests (`tests/concurrency.rs`): parallel search isolation, write blocks only target collection | testing | `/check` | 2 | pending |
| 4 | T3 — SSE transport via `rmcp` + `axum`; `src/transport/` module; env `NANOVEC_SSE_ADDR` | mcp | — | 2 | pending |
| 5 | T5a — `MemoryBudget` + `approx_bytes()` on `Collection`; env `NANOVEC_MEMORY_LIMIT_MB` | core | — | 2 | pending |
| 6 | T4 — Connection manager: per-client default collection through `_meta.connection_id` | mcp | — | 2, 4 | pending |
| 7 | T6 — LRU eviction per-collection (atomic `last_accessed_tick`; pinned: `default`, `_conn_*`) | core | — | 5 | pending |
| 8 | T7 — Update `stats` tool + new `memory` tool exposing budget + per-collection breakdown | mcp | — | 5, 7 | pending |
| 9 | T8 — Criterion benchmark `benches/concurrency.rs` (single vs 8-thread same/different collections) | testing | `/bench` | 4, 6 | pending |
| 10 | T9 — End-to-end SSE integration test (`tests/sse_integration.rs`) with two clients | testing | `/check` | 4, 6, 7 | pending |
| 11 | T10 — Docs: `docs/components/multi-agent.md`, `sse-transport.md`, `phase-6-multi-agent-report.md`, ROADMAP, CHANGELOG | docs | `/docs` | 3, 9, 10 | pending |

---

## Parallelizable Groups

After Task 2 (mcp rewire) is complete, three independent tracks open up:

```
                    ┌─► [3] T2  testing  (concurrency proptests)
[1] T1a ─► [2] T1b ─┼─► [4] T3  mcp      (SSE transport)
   core      mcp    └─► [5] T5a core     (memory budget)
                            │
                            ▼
                       [7] T6 core (LRU eviction)
                            │
[4] T3 ─┐                   │
        ▼                   │
   [6] T4 mcp (conn mgr) ◄──┘
        │
        ▼
   [8] T7 mcp (stats/memory tool)
        │
        ▼
   [9] T8 testing (bench)  ──┐
                              ├─► [11] T10 docs (final)
  [10] T9 testing (SSE e2e) ──┘
```

**Recommended parallel dispatch (after T1b lands):**
- Track A (mcp): T3 → T4 → T7
- Track B (core): T5a → T6 → (feeds T7 + T9)
- Track C (testing): T2 (independent verification of T1a/T1b)

Tracks A and B can run concurrently; T7 syncs them.

---

## Skill Invocations

| Step | Skill | When | Purpose |
|------|-------|------|---------|
| After Task 2 | `/check` | Sanity after lock refactor | Confirm `fmt + clippy + test` zero-warnings before opening parallel tracks |
| After Task 7 | `/check` | Sanity gate before benchmark | Same — full pipeline must pass before measuring perf |
| Step 9 (T8) | `/bench` | Capture concurrency numbers | Record numbers cited in plan: ≥ 6× and ≥ 7.5× throughput |
| Step 10 (T9) | `/check` | Integration test gate | SSE e2e must pass full pipeline |
| Step 11 (T10) | `/docs` | Documentation coverage | Update `docs/coverage.md` for new components |

---

## Risks & Notes

| Risk | Mitigation |
|------|------------|
| `parking_lot::RwLock` is non-async — handler must run inside `spawn_blocking` to avoid blocking tokio runtime | rmcp already routes tool handlers through a worker pool; verify on T1b that no `await` happens while holding a `parking_lot` guard. Add clippy `await_holding_lock` to deny list. |
| Per-collection lock granularity could deadlock if a handler acquires two collection locks (e.g. cross-collection search) | Document lock ordering: collections must be locked in lexicographic-name order. Add a `debug_assert!` helper in T1a. |
| SSE transport (T3) brings ~150 transitive deps (axum, tower, hyper) — bloats binary | Acceptable: SSE is opt-in via env var. Verify with `cargo bloat` after T3 that stdio-only build path is unaffected. Document in `sse-transport.md` that production deployments stay on stdio unless network access is required. |
| `_meta.connection_id` is client-trusted — a malicious client can read another's data by spoofing the ID | Document explicitly in `multi-agent.md` that this is local-only (same trust boundary as stdio). Authentication is Phase 8 (out of scope per plan). |
| Memory accounting is approximate (D7) — could under-count by 10–20%, causing OOM before eviction triggers | Default `NANOVEC_MEMORY_LIMIT_MB=512` is 4× typical agent workload; the safety margin absorbs the inaccuracy. Document in `multi-agent.md`. |

No nightly-only features. No new `unsafe` blocks expected (all locking via `parking_lot`).

---

## Ready for Execution

Once approved, invoke `/execute docs/plans/phase-6-multi-agent-dispatch.md` to start Task 1.
