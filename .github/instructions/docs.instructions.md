---
title: Documentation
applyTo: "**/*.md,docs/**"
summary: Absolute URLs, valid and current mermaid, doctested code blocks, STE prose.
---

# Documentation

- crates.io and docs.rs render the README files. Every link to a repo file and every image
  uses an absolute URL (`https://github.com/...`, `https://raw.githubusercontent.com/...`).
  Only in-page anchors can be relative.
- Mermaid diagrams (```` ```mermaid ````) must be valid. `task lint:mermaid` renders them.
  The diagrams must also match the code. Flag diagrams that the change makes stale. Keep
  each diagram to less than about 15 nodes.
- `README.md` runs as doctests through the `readme_doctests` module in `src/lib.rs`.
- Tag each code block with its language. Untagged blocks and `rust` blocks in included
  Markdown run as doctests. Thus they must compile. Use `rust,no_run` or `rust,ignore` only
  if you give the reason.
- Make sure that the badges, the License section and the MSRV statement agree with
  `Cargo.toml`.
- Write the prose in Markdown docs, spec files and review instructions in ASD-STE100
  Simplified Technical English (STE). Do not change quoted text, code and identifiers.
