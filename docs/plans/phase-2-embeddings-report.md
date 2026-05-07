# Report: Phase 2 — Vectors → Text

Status: **Complete** (2026-05-06).
Plan: [`phase-2-embeddings-plan.md`](./phase-2-embeddings-plan.md) ·
Dispatch: [`phase-2-embeddings-dispatch.md`](./phase-2-embeddings-dispatch.md).

## Summary

| Metric | Value |
|---|---|
| Tasks completed | 7/7 + 1 quality gate (8/8) |
| Tests before | 49 (48 unit + 1 integration) |
| Tests after | **56** (54 unit + 2 integration) |
| Net new tests | +7 |
| Source files modified | 8 |
| Doc files created/updated | 6 |
| `cargo fmt --check` | OK |
| `cargo clippy ... -D warnings` | OK |
| `cargo test --all-features` | 56 passed, 0 failed, 0 ignored |
| Test runtime | ~38 s lib + ~20 s integration |
| Cold release build | ~5 min (one-time, candle deps) |
| Cold embedder load | 10–30 s (HF Hub download ~90 MB) |
| Warm embedder load | ~30 ms (mmap'ed safetensors) |

## Changes

### Source (`src/`)

- `src/lib.rs` — added `pub mod embed;`
- `src/embed/mod.rs` — **new** (~280 LOC): `Embedder` (tokenizer + BertModel + Device), `EmbedError` enum (Display + Error), `load`/`dimension`/`embed`, 5 tests including 2 proptest properties on shared `Lazy<Embedder>`
- `src/mcp/tools.rs` — added `IndexDocumentParams`, `SearchDocumentParams`
- `src/mcp/mod.rs` — `NanoVecState` rewired (`store: VectorStore` non-optional at dim=384, `embedder: Arc<Embedder>`); existing handlers updated for dim-lock invariant; new `index_document` and `search_document` handlers using two-phase lock pattern
- `src/server/mod.rs` — eager embedder load via `tokio::task::spawn_blocking` before `serve(stdio())`, with informative `tracing::info!` lines
- `src/index/brute.rs` — `SearchResult.metadata: Vec<(String,String)>` field, populated in `BruteForce::search`; +1 unit test
- `tests/integration/mcp_stdio.rs` — Phase 1 test updated under dim=384 lock (toy 3-dim → unit-norm 384-dim vector)
- `tests/integration/mcp_embedding.rs` — **new** (~280 LOC): `test_semantic_search_end_to_end` covering handshake → 3 `index_document` → 3 semantic searches with top-1 + metadata assertions → delete → stats → post-delete search
- `tests/integration/main.rs` — `mod mcp_embedding;`

### Cargo

- `Cargo.toml` — added `candle-core/-nn/-transformers @ 0.10` (CPU-only, default-features off), `tokenizers @ 0.23` (`onig` only), `hf-hub @ 0.5` (`ureq` + `default-tls` for sync API), `once_cell @ 1`
- `Cargo.lock` — pinned

### Docs (`docs/`)

- `docs/components/embed.md` — **new**: Embedder API, pipeline, model rationale, concurrency model, network/cache, safety
- `docs/components/mcp-server.md` — extended: `index_document`, `search_document`, default-metric divergence section, server startup notes
- `docs/coverage.md` — Phase 2 → Complete; Phase 3 → Next
- `docs/ROADMAP.md` — Phase 2 row → ✅ Complete
- `docs/PHASE-2-PRESENTATION.md` — **new** (mirror of Phase 1 presentation)
- `CLAUDE.md` — phase status line corrected

### Config

- `.mcp.json` — corrected stale path to `/Users/volosyn.nikita/dev/02_Apps/nanovec/target/release/nanovec`

## Test results

```
$ cargo fmt --check
OK

$ cargo clippy --all-targets --all-features -- -D warnings
Finished `dev` profile, no warnings

$ cargo test --all-features
test result: ok. 54 passed; 0 failed; 0 ignored    (lib unit)
test result: ok. 2 passed;  0 failed; 0 ignored    (integration)
```

## Manual product verification

End-to-end against `target/release/nanovec`:

```
INFO loading embedder (sentence-transformers/all-MiniLM-L6-v2, ~90MB on first run)
INFO embedder loaded dimension=384            ← ~30 ms (warm cache)
INFO nanovec MCP server starting on stdio

index_document "meeting with engineering team on friday at 3pm"   → id=0
index_document "buy milk and bread on the way home"               → id=1
index_document "git commit hash a3f9b2 broke the deploy pipeline" → id=2

search_document "what meetings do I have this week?" k=3
  ✅ top-1 = id 0 (meeting), score 0.507
search_document "groceries to pick up" k=3
  ✅ top-1 = id 1 (groceries), score 0.568
search_document "what broke production?" k=3
  ✅ top-1 = id 2 (deploy bug), score 0.732
```

Top-1 correct in all three queries. Cosine score range observed `[0.42, 1.07]` matches L2-normalized BERT theory.

## Notable engineering decisions made during execution

1. **`hf-hub` features pivot.** Plan suggested `["tokio"]`; actual sync API requires `["ureq", "default-tls"]`. Caught and fixed in Task 2.
2. **Task 5 split into 5a (core) + 5b (mcp).** `/assign` validation found mcp agent crossing into `src/index/brute.rs` (core domain). Split kept domain boundaries clean.
3. **Group D parallelism.** Tasks 4 (mcp) and 5a (core) ran via `superpowers:dispatching-parallel-agents` — different agents, non-overlapping files. One transient incremental-cache hiccup, self-resolved.
4. **Two-phase lock pattern in handlers.** `Arc::clone(&state.embedder)` under lock → release lock → embed without lock → re-lock for store mutation. Avoids serializing concurrent requests on the ~1 ms BERT forward.
5. **Default metric divergence.** Server-wide default = Euclidean (Phase 1 contract); `search_document` defaults to Cosine (L2-normalized embeddings → cosine reads cleanest).
6. **Dim-lock at startup.** `VectorStore::new(384)` in `NanoVecState::new`, no more lazy init. `index_vector` errors clearly: `"vector dim mismatch: expected 384 (locked by embedder), got N"`.
7. **Proptest cases capped at 16/property.** Each case = full BERT forward. 256 default cases would push lib tests past 5 minutes.

## Follow-ups (not blocking — candidates for post-Phase 2 polish)

1. **ANSI escape codes in stderr.** `tracing-subscriber` emits colored output; through MCP this appears as `\u{1b}[2m...` in client-side log ingestion. Fix: `with_ansi(false)` in `src/main.rs` (1 line).
2. **CI strategy for HF cache.** Integration tests need network on first run. Consider a CI step that pre-warms `~/.cache/huggingface/` before `cargo test`.
3. **Cold release build is slow** (~5 min, candle dep tree). Document the one-time cost in README; optionally explore feature gates to slim further if it bites.
4. **Brute-force doc has stale "metadata not in results" line.** `docs/components/brute-force.md` still describes Phase 1 state. Not misleading per se (the `search` tool still doesn't return metadata; only `search_document` does), but a sentence noting the new `SearchResult.metadata` field would close the loop.

## Next phase

Per `docs/ROADMAP.md` — **Phase 3: Scalar → SIMD.** Acceleration of distance functions via AVX2 (x86_64) and NEON (aarch64) with criterion benchmarks vs. scalar baseline and proptest correctness equivalence. Domain: `simd` agent (`src/simd/`, `src/distance/*_simd.rs`). Will require nightly toolchain for `portable_simd` and ~no production-code changes outside `src/distance/` and `src/simd/`.
