---
name: execute
description: "Execute an approved dispatch plan task by task. Use after /assign to implement the planned work. Triggers: execute, run plan, implement, go."
---

# Execute: Plan Implementation

## Steps

1. **Load dispatch schedule.** Read the dispatch file from `docs/plans/<feature-slug>-dispatch.md`.

2. **Verify prerequisites.** Before starting:
   ```bash
   cargo test --all-features 2>&1 | tail -5
   cargo clippy --all-targets --all-features -- -D warnings 2>&1 | tail -5
   ```
   If either fails, fix before proceeding.

3. **Execute tasks in order.** For each task in the dispatch schedule:

   a. **Announce** the current task: agent, files, acceptance criteria.

   b. **Switch context** to the assigned agent's domain rules and constraints.

   c. **TDD cycle** (if implementation task):
      - Write the failing test first
      - Run `cargo test --all-features` to confirm red
      - Implement the minimum code to pass
      - Run `cargo test --all-features` to confirm green
      - Refactor if needed (test must stay green)

   d. **Run quality gate:**
      ```bash
      cargo fmt --check
      cargo clippy --all-targets --all-features -- -D warnings
      cargo test --all-features
      ```

   e. **Update status** in the dispatch file: `pending` -> `done`

4. **Handle failures.** If a task fails:
   - Do not proceed to the next task
   - Diagnose the failure
   - Fix within the current agent's domain
   - Re-run quality gate
   - Only proceed when green

5. **Generate report.** After all tasks complete, write `docs/plans/<feature-slug>-report.md`:
   ```markdown
   # Report: <Feature Name>

   ## Summary
   - Tasks completed: N/N
   - Tests added: <count>
   - Files changed: <count>

   ## Changes
   <list of files created or modified>

   ## Test Results
   <output of cargo test>

   ## Next Steps
   <any follow-up work identified>
   ```

6. **Final verification.**
   ```bash
   cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features
   ```
