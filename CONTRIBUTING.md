# Contributing to miniident

Thank you for your help. This file describes the setup of the repository. It also describes
what a change needs before a merge.

## Setup

1. Install Rust with [rustup](https://rustup.rs/). `Cargo.toml` declares the toolchain and
   MSRV (`rust-version`).
2. Install [Task](https://taskfile.dev/installation/). All commands below use Task. Run
   `task` to list them.
3. Optional tools: if a task needs a tool that is missing, the task tells you.
   - [cargo-nextest](https://nexte.st/) for faster tests;
   - [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) and
     [cargo-hack](https://github.com/taiki-e/cargo-hack);
   - [cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) for coverage;
   - [typos](https://github.com/crate-ci/typos);
   - [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) if you change diagrams;
   - [actionlint](https://github.com/rhysd/actionlint) and [zizmor](https://docs.zizmor.sh/);
   - nightly Rust plus [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) for fuzz tests.
4. Install the git hooks with [prek](https://prek.j178.dev/) (`prek install`), or with
   `pre-commit install`. Fast checks run on commit. On push, clippy and the tests run.

## Everyday commands

| Command | What it does |
|---|---|
| `task test` | Runs the tests (with nextest, if it is installed) and the doctests |
| `task lint` | Runs clippy (pedantic), the separator ban, the README link check, typos and the workflow lints |
| `task fmt` | Formats the code |
| `task doc` | Builds the docs with private items. It treats warnings as errors |
| `task ci` | Runs all CI steps. Run it before you open a PR |
| `task test:prop` | Runs the property tests with 10000 cases |
| `task bench` | Runs the benchmarks |
| `task fuzz:smoke` | Runs each fuzz target for a short time (needs nightly) |
| `task clean` | Removes the build output. It keeps fuzz corpora and crash artifacts |

## Code style

- Every item has a doc comment. This includes private items.
- Do not use `mod.rs`. Use `foo.rs` plus `foo/`.
- Do not use `// ----` separator comments. Split the file into modules instead.
- Keep functions small. Extract helpers and give them descriptive names.
- `clippy::pedantic` must pass. If a lint is wrong in one place, put
  `#[expect(lint, reason = "...")]` on that item. Never use `#[allow]`.
- Do not use unsafe code unless the crate explicitly documents why it needs it.
- The library is `no_std` and allocates nothing. Numeric code is generic over minikalman's
  `MatrixDataType`, so it must also work in fixed point.
- The crate uses the High-assurance lint profile: `unwrap`, `panic!`, indexing, unchecked
  arithmetic and `as` casts are denied. A deviation is an `#[expect]` whose reason states
  the invariant.
- Do not use lossy `as` casts. Do not leave overflow to the build profile. On paths that
  external input can reach, do not panic. Return errors instead.
- README and doc links use absolute URLs, because crates.io renders READMEs outside GitHub.
  Mermaid diagrams are welcome.
- Write docs and comments in [ASD-STE100 Simplified Technical English](https://www.asd-ste100.org)
  (STE): short sentences, active voice, no -ing forms, one idea per sentence.

## Branches

This repository uses **GitHub Flow**:

- `main` is always ready for a release.
- Do the work on `feature/<topic>` branches. Use feature branches for bug fixes too.
- Open a PR into `main`.

Do not use other prefixes (`fix/`, `chore/`, `deps/`, ...). `task lint:branch`, the pre-push
hook and CI check the branch name. Bot branches (`release/*` from release-plz,
`dependabot/*`) are the only exceptions. Rename Copilot's `copilot/*` branches to
`feature/*` before anyone works on them.

## Commits and pull requests

- Write commit subjects in the **imperative mood**, as Git itself does: "Add parser for pair
  lists", "Fix overflow in tile index". Do not write "Added ...", "Adds ..." or `feat: ...`.
- Keep the subject to 72 characters or fewer. Start it with a capital letter. Do not end it
  with a period. Then add a blank line and a body that explains *why*. The `commit-msg` hook
  checks the subject.
- The changelog groups commits by their first verb (Add, Fix, Remove, Deprecate, ...). For
  incompatible changes, write a `BREAKING CHANGE:` line in the body.
- A PR merges with a merge commit, or as a squash. A squash uses the PR title and
  description. Write both with care. GitHub deletes the head branch after the merge.
- Keep PRs focused. CI must be green.
- Do not bump versions or edit `CHANGELOG.md` by hand. The releases are automatic (see
  below).

## Releases

Versions follow [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html), as Cargo
implements it. In `0.x`, a minor bump is a breaking change.

[release-plz](https://release-plz.dev/) does the releases. Each merge to `main`
updates a release PR. A merge of that PR tags the crate and creates the GitHub
Releases. Each tag (`vX.Y.Z`) then starts `publish-crates-io.yml`.
This workflow publishes the crate to crates.io through trusted publishing. It runs in the
protected `crates-io` environment.

## License

If you contribute, you agree that your contributions are dual licensed under MIT OR Apache-2.0, the
same terms as the project. See the License section of the README.
