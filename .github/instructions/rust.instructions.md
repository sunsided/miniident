---
title: Rust code
applyTo: "**/*.rs"
summary: House style: docs on every item, no mod.rs, no separators, blank lines, helper extraction, pedantic clippy, no unsafe, explicit numerics, no panics on input, bounded resources, STE text.
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment. This includes private and `pub(crate)` items. Docs give the intent, invariants,
  units and error/panic conditions (`# Errors`, `# Panics`, `# Safety`). Docs do not repeat
  the name.
- Do not use `mod.rs` files. A module with children is `foo.rs` plus `foo/`. Put
  integration tests under `tests/it/`.
- Do not use separator comments (`// -----`, `// =====`). If a file needs sections, split
  it into submodules.
- Put a blank line after the closing `}` of a block statement if another statement follows.
  Also put a blank line between the logical phases of a function. Do not lump code together.
- Each function has one level of abstraction. Long loops (simulations, game ticks, solvers)
  are short drivers. They call documented step functions. If a block needs a "what" comment,
  extract the block into a function. Name the function after that comment.
- `clippy::pedantic` gives no warnings. Suppress a lint only with a scoped
  `#[expect(lint, reason = "...")]`. Never use `#[allow]`. Never suppress a lint at crate
  level.
- Do not use `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a
  documented reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- Do not use lossy `as` casts. Use `From` to widen. Use `TryFrom` plus an error to narrow.
  Choose the overflow semantics for each operation (`checked_`, `saturating_`, `wrapping_`).
  Do not leave them to the build profile. Do not use float `==` except for exact values.
  Document the NaN behavior and the tolerances.
- On paths that external input can reach, do not use `unwrap`, `panic!`, `todo!` or
  unchecked indexing. At the boundary, change untrusted input into `Result` and into types
  that carry invariants. `expect` messages and `#[expect]` reasons give the invariant that
  makes them safe.
- If work or memory grows with the input, a documented limit bounds it. The code checks the
  limit before it allocates. Recursion has a documented maximum depth that the code
  enforces. If not, replace the recursion with iteration.
- The library is `#![no_std]` and allocates nothing. Do not add `alloc` or `std` use outside
  tests. Dimensions are const generics.
- Numeric code is generic over minikalman's `MatrixDataType`, fixed point included. Do not
  assume IEEE floats in library code: no `f64` literals, no NaN or epsilon tests. Get
  constants through `T::from_f64` or `T::one()`, and state the fallback.
- `update` paths stay `O(N²)`. `assess` paths can be `O(N³)`. Document the cost on each
  public method.
- The repo uses the High-assurance profile. `unwrap`, `panic!`, indexing, unchecked
  arithmetic and `as` are denied. Each `#[expect]` reason states the invariant.
- Mermaid diagrams in doc comments must match the code that they describe. If the structure
  changes, update the diagrams.
- Write doc comments, code comments, `// SAFETY:` comments, `#[expect]` reasons and
  `expect`/`assert!` messages in ASD-STE100 Simplified Technical English (STE). Flag text
  that breaks STE: long sentences, passive voice, -ing forms, or two ideas in one sentence.
