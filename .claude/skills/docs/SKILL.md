---
name: docs
description: "Audit and update documentation coverage for NanoVec components. Use to check which modules have docs, update component docs, or generate coverage reports. Triggers: docs coverage, document, audit docs, update docs."
---

# Docs: Documentation Coverage

## Operations

### audit -- Check documentation coverage

1. **Scan source modules.** List all public modules in `src/`:
   ```bash
   find src/ -name 'mod.rs' -o -name '*.rs' | grep -v test | sort
   ```

2. **Scan existing docs.** List all component documentation:
   ```bash
   ls docs/components/ 2>/dev/null || echo "No component docs yet"
   ```

3. **Compare.** For each source module, check if a corresponding `docs/components/<name>.md` exists.

4. **Check doc comments.** For each public item, verify `///` doc comments exist:
   ```bash
   grep -rn 'pub fn\|pub struct\|pub enum\|pub trait' src/ --include='*.rs' | head -30
   ```

5. **Generate report.** Write `docs/coverage.md`:
   ```markdown
   # Documentation Coverage

   Last updated: <date>

   | Module | Component Doc | Doc Comments | Status |
   |--------|--------------|-------------|--------|
   | store/mod.rs | docs/components/vector-store.md | yes/no | covered/missing |
   | distance/euclidean.rs | docs/components/distance.md | yes/no | covered/missing |

   **Coverage: X/Y modules (Z%)**
   ```

### update -- Update component documentation

1. **Read source.** Read the target module's source code to extract:
   - Public API (functions, structs, enums, traits)
   - Internal design decisions
   - Performance characteristics
   - Dependencies on other modules

2. **Write component doc.** Create or update `docs/components/<name>.md` following the standard format (see docs agent for template).

3. **Update coverage.** Re-run the audit to update `docs/coverage.md`.

### status -- Quick coverage summary

1. Count documented vs undocumented modules and print a one-line summary.
