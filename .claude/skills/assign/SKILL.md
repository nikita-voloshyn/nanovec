---
name: assign
description: "Assign agents and skills to plan tasks. Use after /plan to create a dispatch schedule. Triggers: assign, dispatch, route, map tasks."
---

# Assign: Agent and Skill Mapping

## Steps

1. **Read the plan.** Load the most recent plan from `docs/plans/` or accept a plan reference from the developer.

2. **Validate agent assignments.** For each task in the plan, verify:
   - The assigned agent owns the target files (check agent Domain sections)
   - The agent is not forbidden from any files the task touches
   - Cross-domain tasks are split into separate subtasks per agent

3. **Assign skills.** For each task, determine if a skill should be invoked:
   - Implementation tasks -> direct agent work (no skill)
   - Quality verification -> `/check`
   - Performance measurement -> `/bench`
   - Assembly inspection -> `/asm`
   - Security audit -> `/audit`
   - Documentation update -> `/docs`

4. **Create dispatch schedule.** Write to `docs/plans/<feature-slug>-dispatch.md`:
   ```markdown
   # Dispatch: <Feature Name>

   ## Schedule

   | Order | Task | Agent | Skill | Status |
   |-------|------|-------|-------|--------|
   | 1 | <title> | core | - | pending |
   | 2 | <title> | testing | /check | pending |
   | 3 | <title> | simd | - | pending |
   | 4 | <title> | testing | /bench | pending |
   | 5 | <title> | docs | /docs | pending |
   ```

5. **Identify risks.** Flag any tasks that:
   - Touch multiple agent domains (needs splitting)
   - Have circular dependencies (needs reordering)
   - Require nightly Rust features (note toolchain requirement)
   - Involve unsafe code (needs SAFETY documentation)

6. **Present for approval.** Show the dispatch schedule and wait for developer confirmation.
