# Repository instructions

This is a Rust project with strict conventions. Path-specific rules are in
`.github/instructions/*.instructions.md`.

- Run the checks through the Taskfile. `task ci` mirrors CI. `task lint`, `task test` and
  `task doc` are the common entry points.
- Every item has a doc comment. This includes private items. Do not use `mod.rs` files
  (use `foo.rs` plus `foo/`). Do not use `// ----` separator comments. Split the file into
  modules instead.
- `clippy::pedantic` must stay clean. Suppress a lint only with
  `#[expect(lint, reason = "...")]` on the smallest item. Never use `#[allow]`.
- Do not use unsafe code unless the crate documents why it needs it.
- Make numeric conversions and overflow explicit (`TryFrom`, `checked_`/`saturating_`/
  `wrapping_`). Code that external input can reach returns errors. It does not panic.
- Write commit subjects in the imperative mood ("Add X", "Fix Y"). Do not use a Conventional
  Commits prefix or the past tense. Keep the subject to 72 characters or fewer. Then add a
  blank line and a body that explains why.
- Use GitHub Flow, as CONTRIBUTING.md states: `main` plus `feature/<topic>`. Never use `fix/`, `chore/`, `deps/` or `copilot/`. Before you work on a
  `copilot/*` branch, rename it to `feature/*`.
- If CONTRIBUTING.md and these instructions do not agree, CONTRIBUTING.md applies.
- release-plz makes the release PRs. Do not bump versions by hand.
- Write docs and comments in ASD-STE100 Simplified Technical English (STE): short sentences,
  active voice, no -ing forms, one idea per sentence.
