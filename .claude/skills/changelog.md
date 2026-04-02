---
name: changelog
description: "Generate a structured session changelog from git diff. Use at the end of a development session to document what changed. Triggers: changelog, session summary, what changed."
---

# Changelog: Session Summary

## Steps

1. **Capture diff.** Get all changes since the last commit (or since a specified ref):
   ```bash
   git diff --stat HEAD
   git diff --name-only HEAD
   ```

2. **Categorize changes.** Group modified files by domain:
   - **Core:** `src/store/`, `src/index/`, `src/distance/`, `src/heap/`
   - **SIMD:** `src/simd/`
   - **MCP:** `src/mcp/`, `src/server/`, `src/transport/`
   - **Tests:** `tests/`, `benches/`, `**/tests.rs`
   - **Docs:** `docs/`, `CHANGELOG.md`, `README.md`
   - **Config:** `Cargo.toml`, `rust-toolchain.toml`, `.github/`

3. **Summarize changes.** For each category, write a brief description of what changed and why.

4. **Generate changelog entry.** Format as:
   ```markdown
   ## [Unreleased] - <date>

   ### Added
   - <new features, modules, functions>

   ### Changed
   - <modifications to existing code>

   ### Fixed
   - <bug fixes>

   ### Performance
   - <optimization changes, benchmark improvements>

   ### Internal
   - <refactoring, CI changes, dependency updates>
   ```

5. **Prepend to CHANGELOG.md.** Add the entry at the top of the changelog file, after the header.

6. **Show summary.** Display the changelog entry for developer review.
