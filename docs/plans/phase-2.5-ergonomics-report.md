# Report: Phase 2.5 — Search Ergonomics & UX Fixes

**Plan:** [`phase-2.5-ergonomics-plan.md`](phase-2.5-ergonomics-plan.md)
**Dispatch:** [`phase-2.5-ergonomics-dispatch.md`](phase-2.5-ergonomics-dispatch.md)
**Status:** complete
**Date:** 2026-05-07

## Summary

- **Tasks completed:** 7/7
- **Commits:** 7 (one per task, conventional commits scoped by domain)
- **Tests added:** 7 unit (`store`/`store::record` clear tests + `embed` model_name tests) + 1 integration test with 6 assertion sections
- **Files changed:** 13 (4 source, 1 test, 8 docs/READMEs)
- **Backward compatibility:** preserved — `score` alias on search results, top-level `metric` alias on stats
- **Quality gates:** all green (`fmt`, `clippy`, 63 unit tests, 3 integration tests)

## Commits

```
7270ddf docs: model limitations, MCP surface updates, bilingual README refresh
78a4409 test(integration): cover clear, stats shape, and similarity normalization
7bb2cb9 feat(mcp): add similarity field to search responses
dad5dc8 feat(mcp): expand stats with per-tool defaults and embedder info
ffef7c1 feat(mcp): add clear tool for bulk reset
f1cac3c feat(embed): expose MODEL_NAME constant and model_name() accessor
4ce0d8e feat(store): add clear() to VectorStore and RecordStore
```

Each commit stands alone; no squash. `git log` reads as the implementation
story from foundations (store) up to documentation.

## Changes by File

### Source

| File | What changed |
|------|--------------|
| `src/store/mod.rs` | `+VectorStore::clear()` (preserves dimension lock) + 4 unit tests |
| `src/store/record.rs` | `+RecordStore::clear()` (resets `next_id` to 0) + 3 unit tests |
| `src/embed/mod.rs` | Promoted private `MODEL_REPO_ID` to public `MODEL_NAME` const, added `Embedder::model_name()` accessor + 2 unit tests |
| `src/mcp/mod.rs` | `+clear` tool, restructured `stats` (`default_metric`, `embedder` blocks; `metric` alias preserved), `+distance` and `+similarity` fields on `search` and `search_document` (`score` alias preserved), `+similarity_for()` and `+metric_token()` helpers; `search` now also returns `metadata` for parity with `search_document` |

### Tests

| File | What changed |
|------|--------------|
| `tests/integration/mcp_phase25.rs` | New: 6-section single-binary E2E test (`clear` deleted count, idempotent on empty, `stats` new shape with all fields, `next_id` reset to 0 after `clear`, `distance`/`similarity`/`score` alias correctness, similarity formula verification for euclidean and cosine) |
| `tests/integration/main.rs` | Registered `mcp_phase25` module |

### Documentation

| File | What changed |
|------|--------------|
| `docs/components/embed.md` | New "Model Limitations" section with mitigation guidance; `MODEL_NAME` and `model_name()` added to public API; test count 6 → 8 |
| `docs/components/mcp-server.md` | Added `clear` tool section, restructured `stats` output table, new "Score Fields (Phase 2.5)" section explaining `distance`/`similarity`/`score`; tool count 6 → 7; integration test count 2 → 3 |
| `docs/coverage.md` | New "Phase 2.5 (Complete — Search Ergonomics)" section listing all six deliverables |
| `README.md` | Tools table expanded with `clear` and updated search response shape; status line and similarity note added |
| `README.uk.md` | Same as English README, fully translated |
| `CLAUDE.md` | Status line updated to "Phases 1, 2, and 2.5 complete; Phase 3 next" |

## Test Results

```
$ cargo test --all-features
test result: ok. 63 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out (lib)
test result: ok.  3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out (integration)

cargo fmt --check                                           : pass
cargo clippy --all-targets --all-features -- -D warnings   : pass
```

Phase 1 baseline (`test_mcp_full_flow`) and Phase 2 baseline
(`test_semantic_search_end_to_end`) both still pass — backward compatibility
verified at the wire-format level.

## Behavior Summary

### `clear` MCP tool
- No parameters
- Returns `{ "deleted": N }` where `N` is the pre-clear record count
- Resets `next_id` to 0; the next `index_*` call gets `id: 0`
- Idempotent on empty store (returns `{ "deleted": 0 }`)
- Dimension lock preserved across clear

### Expanded `stats`
```json
{
  "count": 12,
  "dimension": 384,
  "default_metric": { "raw_vector": "euclidean", "document": "cosine" },
  "embedder":       { "model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384 },
  "metric": "Euclidean"
}
```
Top-level `metric` is a backward-compat alias for `default_metric.raw_vector`,
rendered as the `Debug` form of the `Metric` enum.

### Score normalization on search
Each search result now carries:
- `distance` — raw output of the configured metric (lower = closer)
- `similarity` — metric-aware normalization (higher = closer):
  - cosine: `1 - distance` ∈ `[-1, 1]`
  - euclidean: `1 / (1 + distance)` ∈ `(0, 1]`
  - dot product: `-distance` (= raw dot product, unbounded)
- `score` — backward-compat alias for `distance`
- `metadata` — also added to `search` for parity with `search_document`

### Model limitations documented
`docs/components/embed.md` now explicitly flags the two practical limits of
MiniLM-L6-v2 surfaced during the Montu MVP test: weak performance on
non-English content and lexical sensitivity in the 384-dim space. Recommended
mitigations (breadcrumbs, query rewriting, hierarchical chunking) are listed
in priority order. Replacement of the model with a multilingual variant is
explicitly deferred to a future phase.

## Acceptance Checklist

- [x] All 7 tasks committed with conventional commits scoped by domain
- [x] `cargo fmt --check` clean
- [x] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [x] `cargo test --all-features` all green (63 unit + 3 integration)
- [x] `docs/coverage.md` Phase 2.5 section present with all six items marked documented
- [x] README.md and README.uk.md tools tables include `clear` and mention `similarity`
- [x] CLAUDE.md status reflects Phase 2.5 complete
- [x] No backward-compatibility break: `score` field and top-level `metric` field still present and correct

## Risks Resolved

| # | Risk from dispatch | Outcome |
|---|--------------------|---------|
| R1 | Tasks 3+4+5 textual conflicts on `src/mcp/mod.rs` | Resolved — sequential commits, each touched a different method, zero merge churn |
| R2 | Embedder cold start in tests | Resolved — `mcp_phase25.rs` uses `index_vector` for all clear/stats/similarity assertions; embedder load is a single one-time startup cost |
| R3 | Two-phase locking pattern preservation | Verified — Task 5 only modified post-KNN serialization; lock acquisition unchanged |
| R4 | `similarity_for` math in `src/mcp/` | Accepted as conscious design (D5); helper is private to the module and documented in mcp-server.md |
| R5 | Wire format bloat from aliases | Accepted; aliases are removable in a future major version |
| R6 | `next_id = 0` reset surprises clients | Documented in mcp-server.md `clear` section: "any previously-returned ID is stale" |
| R7 | `cargo test` runtime growth | Acceptable: `mcp_phase25` runs in ~10 s warm (single binary spawn) |

## Follow-ups (out of Phase 2.5 scope)

- **Multilingual embedding model.** Replace MiniLM-L6-v2 with `e5-multilingual` or `bge-m3` for non-English workloads. Will require a Phase X kickoff because the embedding dim changes (384 → 1024 typical) and dimension-locked storage needs to handle the migration.
- **Server-side query rewriting.** Could be a Phase 3.5 addition: a small LLM call before `search_document` to align the query with corpus vocabulary. ROI demonstrated in the Montu test (0.68 → 0.31 cosine distance gap).
- **Bulk index tool.** Skipped here as the parallel-call workaround works adequately. Reconsider if real workloads show contention on the `Arc<Mutex<NanoVecState>>` from many concurrent `index_document` calls.

## Next Phase

Per the roadmap and the updated `CLAUDE.md`, the next phase is **Phase 3 —
Scalar → SIMD**: AVX2 (x86_64) and NEON (aarch64) implementations of the
three distance metrics with property-based correctness verification against
the existing scalar references.
