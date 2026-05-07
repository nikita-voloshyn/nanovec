# Dispatch: Phase 2.5 — Search Ergonomics & UX Fixes

**Plan:** [`phase-2.5-ergonomics-plan.md`](phase-2.5-ergonomics-plan.md)
**Report:** [`phase-2.5-ergonomics-report.md`](phase-2.5-ergonomics-report.md)
**Created:** 2026-05-07
**Status:** complete

## Agent Validation

| Task | Files Touched | Owning Agent | Validation |
|------|---------------|--------------|------------|
| 1 | `src/store/mod.rs`, `src/store/record.rs` | core | OK — core owns `src/store/` |
| 2 | `src/embed/mod.rs` | core | OK — `src/embed/` is core domain by Phase 2 precedent (data/tensor, not protocol) |
| 3 | `src/mcp/mod.rs`, `src/mcp/tools.rs` | mcp | OK — mcp owns `src/mcp/` |
| 4 | `src/mcp/mod.rs` | mcp | OK |
| 5 | `src/mcp/mod.rs` | mcp | OK — `similarity_for` helper is presentation-layer arithmetic, not distance computation. The boundary is preserved: distance functions (forbidden territory) live in `src/distance/`; this helper transforms an already-computed scalar. |
| 6 | `tests/integration/mcp_phase25.rs` (new), `tests/integration/main.rs` | testing | OK — testing owns `tests/` |
| 7 | `docs/**`, `README.md`, `README.uk.md`, `CLAUDE.md` | docs | OK with clarification — docs agent's effective scope includes top-level project documentation files (`README*.md`, `CLAUDE.md`), not just `docs/`. Source code in `src/` remains forbidden. |

No cross-domain tasks needing splitting. No agent reaches into another's territory.

## Schedule

| Order | Task | Agent | Skill | Files | Depends on | Status |
|-------|------|-------|-------|-------|------------|--------|
| 1 | VectorStore::clear() + RecordStore::clear() | core | — | `src/store/mod.rs`, `src/store/record.rs` | — | done |
| 2 | Embedder::MODEL_NAME + model_name() | core | — | `src/embed/mod.rs` | — | done |
| 2.5 | Quality gate after core work | testing | `/check` | — | 1, 2 | done |
| 3 | clear MCP tool | mcp | — | `src/mcp/mod.rs`, `src/mcp/tools.rs` | 1 | done |
| 4 | stats restructure | mcp | — | `src/mcp/mod.rs` | 2 | done |
| 5 | similarity normalization in search/search_document | mcp | — | `src/mcp/mod.rs` | — (file-level: after 3, 4) | done |
| 5.5 | Quality gate after MCP work | testing | `/check` | — | 3, 4, 5 | done |
| 6 | Integration tests (clear, stats shape, similarity) | testing | — | `tests/integration/mcp_phase25.rs`, `tests/integration/main.rs` | 3, 4, 5 | done |
| 6.5 | Full quality pipeline | testing | `/check` | — | 6 | done |
| 7 | Documentation updates | docs | `/docs` | `docs/components/embed.md`, `docs/components/mcp-server.md`, `README.md`, `README.uk.md`, `docs/coverage.md`, `CLAUDE.md` | 1–5 | done |
| 7.5 | Final verification | testing | `/check` | — | 7 | done |

Tasks 1 and 2 may execute in parallel (different modules, no shared state, no merge conflict).
Tasks 3, 4, 5 must execute sequentially (all touch `src/mcp/mod.rs`) — one commit each.

## Skill Rationale

| Skill | When | Why |
|-------|------|-----|
| `/check` after Task 2 (core gate) | core domain done, before mcp begins | catches breakage early; mcp will import from store/embed |
| `/check` after Task 5 (mcp gate) | mcp domain done, before testing | confirms wire format compiles + clippy clean before integration tests are written against it |
| `/check` after Task 6 | full code + tests done | Quality Gates from CLAUDE.md require all-green before docs |
| `/docs` in Task 7 | docs phase | audits coverage map, surfaces undocumented items |
| `/check` after Task 7 | post-docs | ensures no doc-test regressions, fmt drift |

No `/bench` (no perf-sensitive work — pure UX).
No `/asm` (no SIMD changes).
No `/audit` (no security-sensitive changes).

## Commit Sequence

Each task = one commit. Conventional Commits with domain scope:

| Order | Commit Message |
|-------|----------------|
| 1 | `feat(store): add clear() to VectorStore and RecordStore` |
| 2 | `feat(embed): expose MODEL_NAME constant and model_name() accessor` |
| 3 | `feat(mcp): add clear tool for bulk reset` |
| 4 | `feat(mcp): expand stats with per-tool default metrics and embedder info` |
| 5 | `feat(mcp): add similarity field to search responses` |
| 6 | `test(integration): cover clear, stats shape, and similarity normalization` |
| 7 | `docs: model limitations, MCP surface updates, bilingual README refresh` |

After all 7 commits, no squash — each commit stands alone, `git log` reads as the implementation story.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Tasks 3+4+5 all touch `src/mcp/mod.rs` — risk of textual merge conflicts if executed out of order or in parallel | Force sequential execution (3 → 4 → 5). Each commits before the next starts. Plan has been written so each task touches a different method (`clear`, `stats`, `search`/`search_document`), keeping diffs disjoint. |
| R2 | Task 6 integration tests may require embedder cold start (~10–30 s on first run) | Use `index_vector` (raw path) for clear/stats tests — bypasses embedder. Use `index_document` only for similarity tests where the document path is meaningful. Reuse existing test pattern from `mcp_stdio.rs` which already handles this. |
| R3 | Two-phase locking pattern in `search_document` (Phase 2 D5) must be preserved when adding similarity | Task 5 only modifies the post-KNN serialization block; the lock acquisition pattern stays untouched. Code review checkpoint at Task 5. |
| R4 | `similarity_for` helper introduces math into `src/mcp/` — could be argued to belong in `src/distance/` | Conscious design decision (D5 in plan): similarity is presentation, distance is computation. Helper is private to mcp module. Documented in mcp-server.md. |
| R5 | Backward-compat aliases (`score`, top-level `metric`) bloat the wire format | Acceptable cost for non-breaking rollout. Aliases removable in a future major version. Plan documents them as D2/D4. |
| R6 | `next_id = 0` reset after clear could cause ID confusion if a client cached old IDs | Documented behavior (D1 in plan). Client must re-query after clear. Note this in mcp-server.md tool reference. |
| R7 | `cargo test` run time may grow with new integration tests | Testing agent uses `index_vector` not `index_document` where possible — keeps embedder out of hot tests. Acceptable trade-off. |

No circular dependencies. No nightly features required (toolchain stays `stable`). No new `unsafe` code.

## Acceptance for `/execute` Completion

All of the following must hold:

1. All 7 tasks committed with conventional commits scoped by domain.
2. `cargo fmt --check` clean.
3. `cargo clippy --all-targets --all-features -- -D warnings` clean.
4. `cargo test --all-features` all green (including new integration tests).
5. `docs/coverage.md` shows Phase 2.5 section with all sub-items marked documented.
6. README.md and README.uk.md tools tables include `clear` and mention similarity field.
7. CLAUDE.md status line updated to "Phase 2.5 complete".
8. No backward-compatibility break: existing Phase 1/2 clients reading `score` field or top-level `metric` field continue to work.

## Next Step

Awaiting developer approval. On `/execute`:
- Dispatch agent walks the schedule top to bottom.
- Each task gets a fresh agent invocation in its domain.
- After every checkpoint (`/check`), failures halt the run for triage.
- Final commit hash reported back, branch ready for push.
