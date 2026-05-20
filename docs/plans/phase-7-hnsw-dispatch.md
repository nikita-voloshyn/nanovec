# Dispatch: Phase 7 — Exact → Approximate (HNSW)

Source plan: [`phase-7-hnsw-plan.md`](./phase-7-hnsw-plan.md)

## Domain Routing Validation

Cross-domain risks identified during routing:

| Plan task | Concern | Resolution |
|-----------|---------|------------|
| T6 (build + `rebuild_index` tool) | Plan lists both `src/index/hnsw.rs` (core) and `src/mcp/tools.rs` (mcp) | **Split into T6a (core: `Hnsw::build`) + T6b (mcp: `rebuild_index` tool).** T6b imports public API from core. |
| T8 (SIMD reuse verification) | Plan touches `src/index/hnsw.rs` (core domain) for `#[inline]` annotations, `benches/hnsw.rs` (testing domain), and requires `cargo asm` (simd expertise) | **Split into T8a (simd: `/asm` inspection, read-only), T8b (testing: criterion bench).** If T8a finds missing inlining, file a follow-up T8c (core) to add `#[inline]` — declared contingent in the schedule. |
| T9, T10 (benchmarks emitting into `docs/plans/phase-7-hnsw-report.md`) | Benchmarks are testing domain; report file is docs domain | Testing agent emits numbers as terminal output / CSV in `benches/`; **docs agent writes the report in T12** consuming those numbers. T9 and T10 stay pure-testing. |
| T11 (proptests inside `src/index/hnsw.rs`) | Source file is core's, but `#[cfg(test)] mod tests` blocks are explicitly testing's territory per agent roster | OK — testing agent edits only the `#[cfg(test)]` block. |

All other plan tasks map cleanly to a single domain.

---

## Schedule

| Order | Task | Agent | Skill | Depends on | Status |
|-------|------|-------|-------|------------|--------|
| 1 | T1 — `Index` trait + `IndexKind` enum; refactor `BruteForce` and `KdTree` under trait; `Collection::active_index: Box<dyn Index>` | core | — | none | pending |
| 2 | T2 — HNSW scaffold + failing tests in `src/index/hnsw.rs` (`Hnsw`, `HnswParams`; rand SmallRng dep) | core | — | 1 | pending |
| 3 | T3 — Layer assignment (`m_L = 1 / ln(M)`), entry point logic | core | — | 2 | pending |
| 4 | T4 — Search: greedy descent through layers + ef-search beam on layer 0 | core | — | 3 | pending |
| 5 | T5 — Insert: heuristic neighbor selection (Algorithm 4), bidirectional linking, M_max pruning | core | — | 4 | pending |
| 6 | T6a — `Hnsw::build(&store)` with progress logging | core | — | 5 | pending |
| 7 | T6b — `rebuild_index` MCP tool (`collection`, `kind`, `params?`) returning `{built_at_ms, memory_bytes, recall_estimate}` | mcp | — | 6 | pending |
| 8 | T8a — `/asm` inspection of `hnsw::search_layer`: verify distance calls inline to `cosine_avx2`/`cosine_neon` | simd | `/asm` | 4 | pending |
| 9 | T8c — *(contingent)* Add `#[inline]` to distance wrapper in `src/index/hnsw.rs` if T8a found scalar fallback in hot loop | core | — | 8 | contingent |
| 10 | T7 — Auto-dispatch in `Collection::auto_select_index()`; HNSW = opt-in (log recommendation only) | core | — | 6 | pending |
| 11 | T11 — Property-based tests in `src/index/hnsw.rs` `#[cfg(test)] mod tests` (recall threshold, insertion-order invariance, SIMD parity) | testing | `/check` | 6 | pending |
| 12 | T8b — Criterion bench `benches/hnsw.rs`: HNSW search with SIMD distance vs forced scalar | testing | `/bench` | 4, 9 | pending |
| 13 | T9 — Sweep benchmark `benches/hnsw_quality.rs` — 36 points (M × ef_construction × ef_search), emit recall/latency/memory CSV | testing | `/bench` | 6, 10 | pending |
| 14 | T10 — Side-by-side `benches/compare_qdrant.rs` (`--features qdrant-compare`): 100k × 384-dim, build/search/memory/recall vs Qdrant in-memory | testing | `/bench` | 6 | pending |
| 15 | T12 — Docs: `hnsw.md`, `index-dispatch.md`, `phase-7-hnsw-report.md` (consuming T9 + T10 numbers), ROADMAP, README "Index types" section, CHANGELOG | docs | `/docs` | 11, 12, 13, 14 | pending |

---

## Parallelizable Groups

HNSW core path (T2 → T5) is **strictly sequential** — each step depends on the previous algorithmic primitive. Parallelism opens up after T6a (build is functional):

```
[1] T1 ─► [2] T2 ─► [3] T3 ─► [4] T4 ─► [5] T5 ─► [6] T6a ─┐
   core    core      core      core      core      core    │
                                          │                │
                                          ▼                │
                                  [8] T8a simd /asm ──┐    │
                                          │           │    │
                                          ▼           │    │
                                  [9] T8c core (cont) │    │
                                          │           │    │
                                          ▼           │    │
                                  [12] T8b testing  ──┘    │
                                                           │
                                                           ├─► [7]  T6b mcp (rebuild_index tool)
                                                           ├─► [10] T7  core (auto-dispatch)
                                                           ├─► [11] T11 testing (proptests)
                                                           ├─► [13] T9  testing (sweep bench)
                                                           └─► [14] T10 testing (Qdrant compare)
                                                                          │
                                                                          ▼
                                                                  [15] T12 docs (final)
```

**Recommended parallel dispatch after T6a (build is done):**
- Track A (mcp): T6b
- Track B (core): T7 (auto-dispatch)
- Track C (testing): T11 → T9 → T10 sequentially (or T9 || T10 if hardware allows)
- Track D (simd → core → testing): T8a → T8c (if needed) → T8b

All four tracks converge at T12 (docs).

---

## Skill Invocations

| Step | Skill | When | Purpose |
|------|-------|------|---------|
| Step 8 (T8a) | `/asm` | After search lands | Verify `cosine_avx2`/`cosine_neon` inlines into `search_layer` hot loop |
| After Step 9 (T8c) | `/check` | Sanity gate post-inlining-fix | Full `fmt + clippy + test` pipeline; ensures inlining change didn't regress |
| Step 11 (T11) | `/check` | Proptest gate | proptests must run as part of `cargo test --all-features` |
| Step 12 (T8b) | `/bench` | SIMD speedup numbers | Capture HNSW-search-with-SIMD vs forced-scalar ratio (plan target ≥ 4×) |
| Step 13 (T9) | `/bench` | Recall/latency sweep | Find (M, ef) point hitting recall@10 ≥ 0.95 + p50 < 1 ms |
| Step 14 (T10) | `/bench` | Side-by-side vs Qdrant | Numbers feed report; no marketing claims, raw measurements |
| Step 15 (T12) | `/docs` | Documentation coverage | Update `docs/coverage.md`; mark Phase 7 complete in `ROADMAP.md` |

---

## Risks & Notes

| Risk | Mitigation |
|------|------------|
| Strict sequential dependency T1 → T2 → … → T5 (5 tasks, single core agent) is the critical path for everything else | Accepted — this is inherent to HNSW algorithmic build-up. Schedule reflects ~5 core-agent sessions before parallelism opens. |
| T8c is contingent on T8a findings — could be a no-op or a small but mandatory edit | Marked **contingent** in the schedule. If T8a reports clean inlining, T8c is skipped and T8b proceeds immediately. |
| Qdrant comparison (T10) brings `qdrant_client` + tonic + heavy transitive deps | Gated behind `--features qdrant-compare`. Default `cargo test` and `cargo bench` builds do not pull it in. Document in `Cargo.toml` and `compare_qdrant.rs`. |
| `cargo asm` requires `cargo-show-asm` installed (`cargo install cargo-show-asm`) | Document prerequisite in T8a brief; simd agent will note in report if tool not available locally and fall back to `cargo rustc --release -- --emit=asm`. |
| HNSW `unsafe` blocks for performance (raw pointer access during graph traversal) | Per CLAUDE.md rule: any `unsafe` requires `// SAFETY:` comment. Tasks T4 + T5 explicitly call this out. Default to safe code; introduce `unsafe` only with bench evidence. |
| Recall target 0.95 may not be reachable on synthetic Gaussian corpus from T9 | Plan Risks section already addresses: if 0.95 is unreachable, publish honest floor (e.g. 0.90). T9 acceptance includes Pareto frontier — not a single recall target. |
| Property test on dimension ∈ [16, 768] × n ∈ [100, 10000] is slow under proptest's default 256 cases | Reduce to 32 cases for the big-corpus property, keep 256 for small-corpus properties. Add `#![proptest(cases = 32)]` annotation on heavy props. |
| `Box<dyn Index>` in `Collection::active_index` introduces vtable indirection in search hot path | Acceptable: dispatch overhead is amortized over `O(log n)` distance calls. Verify in T8a inspection. If problematic, switch to enum-dispatch (`enum ActiveIndex { Brute(BruteForce), Kd(KdTree), Hnsw(Hnsw) }`) — note as follow-up. |

**Nightly features:** none required. **New `unsafe`:** possible in T4/T5 hot loops — must carry `// SAFETY:` annotations and survive proptest in T11.

---

## Ready for Execution

Once approved, invoke `/execute docs/plans/phase-7-hnsw-dispatch.md` to start Task 1.

**Estimated cost:** ~15 sequential agent sessions on the critical path (T1 → T2 → T3 → T4 → T5 → T6a → fan-out → T12). With parallel execution of the post-T6a fan-out, wall time ≈ 8–10 sessions if multiple agents dispatch concurrently.
