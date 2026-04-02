---
name: setup-approach
description: "Reconfigure the development approach for NanoVec. Use to switch between TDD-First, YAGNI/KISS, Iterative+Timeboxing, Shape Up, or Trunk-Based development. Triggers: change approach, switch methodology, setup approach."
---

# Setup Approach: Development Methodology Configuration

## Steps

1. **Show current approach.** Read the `## Development Approach` section from `CLAUDE.md` and display the current primary and secondary approaches.

2. **Present options.** Show the developer all available approaches from `docs/approaches-reference.md`:
   - TDD-First
   - YAGNI/KISS
   - Iterative + Timeboxing
   - Shape Up
   - Trunk-Based
   - Custom (free text)

3. **Collect selection.** Ask the developer to choose:
   - Primary approach (required)
   - Secondary approach(es) (optional)

4. **Preview changes.** Show what the `## Development Approach` section in CLAUDE.md will look like with the new selection. Use the content from `docs/approaches-reference.md` for the selected approach(es).

5. **Apply changes.** After developer approval:
   - Replace the `## Development Approach` section in `CLAUDE.md` with the new primary approach content
   - If secondary approaches are selected, add them under `### Secondary Approaches` with a precedence note
   - Keep all other CLAUDE.md content unchanged

6. **Confirm.** Show the updated section and confirm the change was applied.
