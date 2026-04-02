# Development Approaches Reference

This file contains the full content for all available development approaches. Used by the `/setup-approach` skill to reconfigure the development methodology in CLAUDE.md.

---

## Iterative + Timeboxing

### Development Approach: Iterative + Timeboxing

### Philosophy

Ship working increments on a fixed cadence; time pressure forces scope discipline.

### Rules

1. **Fixed iteration length.** Every iteration is exactly 2 hours of focused work. When the timer ends, commit what is green and stop.
2. **Iteration starts with a goal.** Write a one-sentence goal before starting. If you cannot state it in one sentence, the scope is too large -- split it.
3. **Demo at the end.** Every iteration must produce something demonstrable: a passing test, a benchmark result, a working MCP tool response.
4. **Unfinished work rolls over.** If an iteration ends mid-task, create a `docs/plans/<feature>-rollover.md` describing what remains.
5. **No scope creep within an iteration.** If you discover additional work, add it to the next iteration's backlog. Do not expand the current iteration.
6. **Retrospective every 3 iterations.** Review velocity, quality, and approach. Adjust iteration length or scope granularity if needed.

### Iteration Structure

| Phase | Duration | Action |
|-------|----------|--------|
| Plan | 5 min | State goal, list tasks |
| Implement | 100 min | TDD cycle, write code |
| Verify | 10 min | Run /check, review |
| Demo | 5 min | Show result, commit |

---

## Shape Up

### Development Approach: Shape Up

### Philosophy

Shape the work before building; appetite (time budget) defines scope, not the other way around.

### Rules

1. **Shape before building.** Every feature starts with a pitch document in `docs/plans/<feature>-pitch.md` that defines the problem, appetite, solution sketch, and rabbit holes to avoid.
2. **Fixed appetite.** Assign a time budget (small: 2 hours, medium: 6 hours, large: 2 days). The implementation must fit within the appetite or be re-scoped.
3. **No backlogs.** There is no feature backlog. If a feature is important, it will be pitched again. Unpitched ideas are forgotten intentionally.
4. **Hill chart progress.** Track each task on a hill chart: uphill (figuring out) vs downhill (executing). Report which hill phase each task is in.
5. **Circuit breaker.** If a task is still uphill after 50% of the appetite is consumed, stop and re-evaluate. Either re-scope or kill the feature.
6. **Cooldown between cycles.** After completing a shaped feature, spend 10% of the appetite on cleanup, refactoring, or tech debt.

### Pitch Format

```markdown
# Pitch: <Feature Name>

## Problem
<what is broken or missing>

## Appetite
<small | medium | large>

## Solution
<high-level approach with rough sketches>

## Rabbit Holes
<things to explicitly avoid>

## No-Gos
<out of scope for this appetite>
```

---

## TDD-First

### Development Approach: TDD-First

### Philosophy

Tests are the specification. Code exists only to make failing tests pass.

### Rules

1. **Red-Green-Refactor.** Write a failing test first. Write the minimum code to make it pass. Refactor only after green. Never skip the red step.
2. **Property-based tests for mathematical correctness.** Every distance function and SIMD implementation must have proptest properties that verify output matches the scalar reference within epsilon tolerance (`approx::assert_relative_eq!`).
3. **Benchmark before and after.** Performance claims require criterion/divan evidence. Record baseline numbers before optimization, measure after.
4. **Coverage is the primary quality signal.** Untested code is unfinished code. Aim for >90% line coverage on core modules (`store`, `index`, `distance`).
5. **Integration tests for MCP.** Every MCP tool must have an integration test that sends a JSON-RPC request through the stdio transport and validates the response schema.
6. **Tests live close to code.** Unit tests go in `#[cfg(test)] mod tests` within the source file. Integration tests go in `tests/`. Benchmarks go in `benches/`.

### TDD Cycle

| Step | Action | Artifact |
|------|--------|----------|
| 1. Specify | Write test expressing desired behavior | `#[test]` or `proptest!` |
| 2. Fail | Run `cargo test` -- confirm red | Terminal output |
| 3. Implement | Write minimum code to pass | `src/` module |
| 4. Pass | Run `cargo test` -- confirm green | Terminal output |
| 5. Refactor | Clean up, extract, rename | Same test still green |
| 6. Benchmark | If performance-sensitive, add criterion bench | `benches/` |

---

## Trunk-Based

### Development Approach: Trunk-Based

### Philosophy

Integrate continuously to the main branch; small commits reduce merge pain and enable fast feedback.

### Rules

1. **Commit to main.** All work happens on `main` (or `master`). No long-lived feature branches. Short-lived branches (< 1 day) are acceptable for PRs.
2. **Small commits.** Each commit should be a single logical change: one test, one function, one refactor. Never bundle unrelated changes.
3. **All commits are green.** Every commit must pass `cargo fmt --check && cargo clippy -- -D warnings && cargo test`. No broken commits on main.
4. **Feature flags over branches.** If a feature is not ready for use, gate it behind a Cargo feature flag (`#[cfg(feature = "kdtree")]`) rather than a branch.
5. **Rebase, not merge.** Keep linear history. Use `git rebase` instead of merge commits.
6. **CI is the authority.** If CI is red, stop all other work and fix it. A broken main branch blocks everyone.

### Commit Flow

```
write test -> cargo test (red) -> implement -> cargo test (green) -> cargo fmt -> cargo clippy -> git add -> git commit -> push
```

---

## YAGNI/KISS

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
