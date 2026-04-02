---
name: phase
description: "Phase-level feature executor for NanoVec. Use to start, track, or complete a development phase (MVP, SIMD, Indexing, MCP). Triggers: phase, milestone, phase status, start phase."
---

# Phase: Milestone Executor

## NanoVec Phases

| Phase | Name | Key Deliverables | Status |
|-------|------|-----------------|--------|
| 0 | Planning & Design | Technical specs, architecture docs | Complete |
| 1 | MVP Implementation | VectorStore, RecordStore, scalar distance, brute-force search, basic tests | Next |
| 2 | SIMD Optimization | AVX2/NEON implementations, portable_simd, benchmarks, proptest | Planned |
| 3 | Spatial Indexing | KD-Tree construction, KNN search with pruning, index vs brute-force logic | Planned |
| 4 | MCP Integration | rmcp server, tool definitions, stdio transport, embedding API | Planned |

## Steps

### start -- Begin a new phase

1. **Identify phase.** Determine which phase to start based on developer input or the next uncompleted phase.

2. **Read phase requirements.** Extract deliverables from the phase table above and from the project vision docs (`01-VISION.md`).

3. **Decompose into features.** Break the phase into features, each suitable for a `/plan` invocation:
   - Phase 1: VectorStore, RecordStore, scalar distances, brute-force search
   - Phase 2: AVX2 dot product, AVX2 L2 distance, NEON implementations, portable_simd, benchmarks
   - Phase 3: KD-Tree build, KNN search, BoundedMaxHeap, index selection logic
   - Phase 4: rmcp server, tool definitions, stdio transport, integration tests

4. **Create phase plan.** Save to `docs/plans/phase-<N>-plan.md` with all features listed.

5. **Execute features sequentially.** For each feature, run `/plan` then `/assign` then `/execute`.

### status -- Check phase progress

1. Read `docs/plans/phase-<N>-*.md` files to determine completed vs pending tasks.
2. Run `cargo test --all-features` to verify current state.
3. Report completion percentage and remaining work.

### complete -- Finalize a phase

1. Run full quality pipeline (`/check`).
2. Run benchmarks if applicable (`/bench`).
3. Update documentation (`/docs`).
4. Generate changelog (`/changelog`).
5. Mark phase as complete in the phase table.
