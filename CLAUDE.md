# NanoVec -- Development Instructions

## Project Overview

NanoVec is a lightweight, zero-dependency, pure in-memory vector database engineered from scratch in Rust, designed as an MCP server for ephemeral AI agent working memory. It targets sub-millisecond semantic search with SIMD-accelerated distance computations, KD-Tree spatial indexing, and native Model Context Protocol integration.

**Repository:** https://github.com/nikita-voloshyn/nanovec
**Language:** Rust (stable + nightly for portable_simd)
**Build system:** Cargo
**Phase:** Phase 1 complete (MVP MCP server), Phase 2 next (server-side embeddings via candle)

## Architecture Rules

1. **Zero external indexing libraries.** No FAISS, HNSWLIB, Annoy, or any third-party vector search crate. All algorithms are implemented from first principles.
2. **Structure of Arrays (SoA) layout.** VectorStore uses a flat `Vec<f32>` for contiguous cache-line access. Metadata lives in a separate RecordStore. Never mix vector data with string metadata in the same struct during distance computation paths.
3. **SIMD safety boundary.** All `unsafe` blocks containing SIMD intrinsics must have a `// SAFETY:` comment explaining the invariant. Portable SIMD (`std::simd`) is preferred over raw intrinsics where possible.
4. **Scalar fallback always exists.** Every SIMD-accelerated function must have a scalar reference implementation. Property-based tests (proptest) verify SIMD output matches scalar output within floating-point tolerance.
5. **MCP is the only external interface.** No REST API, no gRPC, no custom TCP protocol. All client interaction goes through rmcp (stdio or SSE transport).
6. **Ephemeral by design.** No persistence layer, no WAL, no snapshots. Data lives only in RAM and is destroyed on process exit.
7. **Cache-line alignment.** Hot data structures must be `#[repr(align(64))]` where beneficial. Prefetch hints are used for sequential vector scans.
8. **Error types are exhaustive enums.** No `Box<dyn Error>` in public APIs. Each module defines its own error enum.

## Build and Quality Commands

```bash
# Format check (must pass before commit)
cargo fmt --check

# Lint (treat all warnings as errors)
cargo clippy --all-targets --all-features -- -D warnings

# Run all tests
cargo test --all-features

# Run benchmarks
cargo bench

# Inspect generated assembly for a function
cargo asm nanovec::simd::dot_product_avx2

# Security audit
cargo audit

# Full quality pipeline
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features
```

## Agent Roster

| Agent | Domain | Model | Owns | Forbidden |
|-------|--------|-------|------|-----------|
| **dispatch** | Task orchestration | sonnet | Routing, plan execution | Implementation code |
| **core** | VectorStore, RecordStore, KD-Tree, distance functions | opus | `src/store/`, `src/index/`, `src/distance/` | SIMD intrinsics, MCP, benchmarks |
| **simd** | AVX2/NEON, portable_simd, assembly inspection | opus | `src/simd/`, `src/distance/*_simd.rs` | Core data structures, MCP, CI |
| **mcp** | rmcp server, tool definitions, transports | opus | `src/mcp/`, `src/server/`, `src/transport/` | Vector math, SIMD, benchmarks |
| **testing** | Unit tests, proptest, criterion/divan, approx | sonnet | `tests/`, `benches/`, `src/**/tests.rs` | Production code outside test modules |
| **docs** | Architecture docs, API docs, CHANGELOG | sonnet | `docs/`, `CHANGELOG.md` | Any `src/` implementation code |

## Skill Index

| Skill | Trigger | Description |
|-------|---------|-------------|
| `/plan` | "plan", "decompose", "break down" | Decompose a feature into tasks with domain agent assignments |
| `/assign` | "assign", "dispatch", "route" | Assign agents and skills to plan tasks |
| `/execute` | "execute", "run plan", "implement" | Execute an approved dispatch plan task by task |
| `/check` | "check", "quality", "ci" | Full quality pipeline: fmt, clippy, test |
| `/bench` | "benchmark", "perf", "simd vs scalar" | Run criterion/divan benchmarks, compare implementations |
| `/asm` | "assembly", "asm", "codegen" | Inspect generated assembly via cargo-show-asm |
| `/audit` | "audit", "security", "sanitizer" | cargo audit, cargo deny, sanitizer runs |
| `/container` | "docker", "container", "deploy" | Multi-arch Docker build for static binary |
| `/changelog` | "changelog", "session summary" | Generate structured changelog from git diff |
| `/phase` | "phase", "milestone" | Phase-level feature executor |
| `/deploy-check` | "deploy check", "release ready" | Pre-deployment audit checklist |
| `/docs` | "docs coverage", "document" | Audit and update documentation coverage |
| `/setup-approach` | "change approach", "switch methodology" | Reconfigure development approach |

## Development Workflow

### Starting a Feature

1. Run `/plan` with the feature description to decompose it into tasks
2. Run `/assign` to map tasks to domain agents
3. Run `/execute` to implement the approved plan task by task
4. Run `/check` to verify quality before committing
5. Run `/docs` to update documentation coverage

### Quality Gates

Every change must pass before merge:
1. `cargo fmt --check` -- consistent formatting
2. `cargo clippy --all-targets --all-features -- -D warnings` -- zero warnings
3. `cargo test --all-features` -- all tests green
4. Property-based tests pass (proptest) -- SIMD correctness
5. No `unsafe` blocks without `// SAFETY:` comments

### Commit Convention

Use conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `perf:`, `chore:`

Scope by domain: `feat(simd):`, `fix(mcp):`, `test(store):`, `docs(arch):`

## Development Approach: TDD-First

### Philosophy

Tests are the specification. Code exists only to make failing tests pass.

### Rules

1. **Red-Green-Refactor.** Write a failing test first. Write the minimum code to make it pass. Refactor only after green. Never skip the red step.
2. **Property-based tests for mathematical correctness.** Every distance function and SIMD implementation must have proptest properties that verify output matches the scalar reference within epsilon tolerance (`approx::assert_relative_eq!`).
3. **Benchmark before and after.** Performance claims require criterion/divan evidence. Record baseline numbers before optimization, measure after.
4. **Coverage is the primary quality signal.** Untested code is unfinished code. Aim for >90% line coverage on core modules (`store`, `index`, `distance`).
5. **Integration tests for MCP.** Every MCP tool must have an integration test that sends a JSON-RPC request through the stdio transport and validates the response schema.
6. **Tests live close to code.** Unit tests go in `#[cfg(test)] mod tests` within the source file. Integration tests go in `tests/`. Benchmarks go in `benches/`.

### TDD Cycle for NanoVec

| Step | Action | Artifact |
|------|--------|----------|
| 1. Specify | Write test expressing desired behavior | `#[test]` or `proptest!` |
| 2. Fail | Run `cargo test` -- confirm red | Terminal output |
| 3. Implement | Write minimum code to pass | `src/` module |
| 4. Pass | Run `cargo test` -- confirm green | Terminal output |
| 5. Refactor | Clean up, extract, rename | Same test still green |
| 6. Benchmark | If performance-sensitive, add criterion bench | `benches/` |

---

### Secondary Approaches

> These approaches supplement the primary approach above. When they conflict with the primary approach, the primary approach takes precedence.

### Development Approach: YAGNI/KISS

### Philosophy

Build only what is needed right now. Complexity is the enemy of correctness.

### Rules

1. **No premature abstractions.** Do not create traits, generics, or extension points until the second concrete use case appears. The first implementation is always concrete.
2. **No speculative features.** If the current phase does not require it, do not build it. Phase 1 needs brute-force search -- do not build KD-Tree in Phase 1.
3. **Refactor on the second similar case.** When you write something similar for the second time, extract. Never on the first time.
4. **Prefer `pub(crate)` over `pub`.** Minimize the public API surface. Internal modules use restricted visibility.
5. **Zero external indexing libraries.** This is a YAGNI rule too: we do not need FAISS because we are building it ourselves, one operation at a time.
6. **Delete dead code immediately.** Unused functions, commented-out blocks, and TODO stubs that are not in the current phase get deleted, not left around.

## File Structure (Planned)

```
nanovec/
  Cargo.toml
  rust-toolchain.toml
  src/
    lib.rs              # Feature gates, module declarations
    store/
      mod.rs            # VectorStore (flat f32 array, SoA)
      record.rs         # RecordStore (metadata, ID mapping)
    distance/
      mod.rs            # Distance trait, metric selection
      euclidean.rs      # Scalar L2 distance
      cosine.rs         # Scalar cosine similarity
      dot.rs            # Scalar dot product
    simd/
      mod.rs            # SIMD dispatch (AVX2/NEON/scalar fallback)
      avx2.rs           # x86_64 AVX2 implementations
      neon.rs           # aarch64 NEON implementations
      portable.rs       # std::simd portable implementations
    index/
      mod.rs            # Index trait, selection logic
      kdtree.rs         # KD-Tree construction and KNN search
      brute.rs          # Brute-force linear scan
    heap/
      mod.rs            # BoundedMaxHeap for top-K results
    mcp/
      mod.rs            # MCP server setup, tool registration
      tools.rs          # Tool definitions (index, search, delete, stats)
      transport.rs      # stdio/SSE transport configuration
    server/
      mod.rs            # Main server entry point
  tests/
    integration/        # End-to-end MCP tests
  benches/
    distance.rs         # Scalar vs SIMD benchmarks
    search.rs           # Brute-force vs KD-Tree benchmarks
  docs/
    plans/              # Feature plans and dispatch reports
    components/         # Per-component documentation
```
