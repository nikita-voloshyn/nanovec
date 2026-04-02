---
name: plan
description: "Decompose a feature into tasks with domain agent assignments. Use when starting a new feature, phase, or multi-step implementation. Triggers: plan, decompose, break down, feature planning."
---

# Plan: Feature Decomposition

## Steps

1. **Gather context.** Read the feature request and identify which NanoVec modules are affected (store, index, distance, simd, mcp, server).

2. **Read current state.** Check the existing code in affected modules to understand what exists and what needs to change:
   ```bash
   find src/ -name '*.rs' -type f | head -30
   cargo test --all-features 2>&1 | tail -5
   ```

3. **Decompose into tasks.** Break the feature into atomic, testable tasks. Each task must:
   - Have a single domain agent owner (core, simd, mcp, testing, docs)
   - Be completable independently (or have explicit dependencies)
   - Include acceptance criteria (what test must pass)
   - Follow TDD: specify the test before the implementation

4. **Assign agents.** Map each task to a domain agent based on the routing table:
   - VectorStore, RecordStore, KD-Tree, distance -> **core**
   - AVX2, NEON, portable_simd, assembly -> **simd**
   - MCP tools, rmcp, transport, JSON-RPC -> **mcp**
   - Tests, proptest, benchmarks, coverage -> **testing**
   - Documentation, CHANGELOG, API docs -> **docs**

5. **Sequence tasks.** Order tasks by dependencies. Common patterns:
   - Scalar implementation (core) before SIMD acceleration (simd)
   - Data structure (core) before MCP tool (mcp)
   - Implementation before tests (unless TDD, then reverse)
   - All code before documentation (docs)

6. **Write the plan.** Save to `docs/plans/<feature-slug>-plan.md` using this format:
   ```markdown
   # Plan: <Feature Name>

   ## Summary
   <one paragraph>

   ## Tasks

   ### Task 1: <title>
   - **Agent:** <agent name>
   - **Files:** <files to create or modify>
   - **Depends on:** none | Task N
   - **Acceptance:** <test or verification command>

   ### Task 2: ...
   ```

7. **Present for approval.** Show the plan to the developer and wait for confirmation before execution.
