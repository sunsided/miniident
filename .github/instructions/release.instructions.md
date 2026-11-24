---
title: Crates and releases
applyTo: "**/Cargo.toml,release-plz.toml,.github/workflows/release-plz.yml,**/CHANGELOG.md"
summary: Crate metadata, a version for each crate, deny policy, release-plz and trusted publishing.
---

# Crates and releases

- Versions use SemVer 2.0, as Cargo interprets it. Pre-releases (`-rc.1`) are permitted.
  Build metadata (`+...`) is not permitted.
- In release-plz repos, the release-plz release PRs manage the versions and `CHANGELOG.md`.
  In feature PRs, do not bump versions or edit generated changelog entries by hand.
- Every publishable crate has `description`, `license`, `repository`, `readme`, `keywords`
  (5 or fewer) and `categories` (valid crates.io slugs). Every publishable crate also
  declares `rust-version`. Crates that you do not publish have `publish = false`.
- miniident is a single crate. Its release tags are `vX.Y.Z`.
- New dependencies must pass `cargo deny check`. They must have allowed licenses and no
  advisories. They must not come from git or from an unknown registry. Path dependencies of
  publishable crates have a `version`.
- `rust-version` is a signal, not a contract. You can raise it if this helps a dependency or
  a feature. A raise needs a minor release and a changelog note. Never raise it in a patch
  release.
- release-plz runs in git-only mode. It creates the tags and the GitHub Releases.
  `publish-crates-io.yml` publishes from the tag in the `crates-io` environment. It uses
  trusted publishing (`id-token: write`, no `CARGO_REGISTRY_TOKEN`). Keep this setup.
- The imperative verb in each commit subject sets the changelog group and the bump
  (`release-plz.toml`). In 0.x, an addition is a patch bump and a breaking change is a minor
  bump. A `custom_major_increment_regex` on a 0.x crate is a bug. It moves the crate to
  1.0.0. If a subject hides a feature behind "Update ...", the changelog puts it under
  Changed. Flag it.
