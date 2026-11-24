---
title: CI, Taskfile and hooks
applyTo: ".github/workflows/**,Taskfile.dist.yaml,**/Taskfile.dist.yaml,.pre-commit-config.yaml"
summary: Keep task ci, ci.yml and the hooks in sync. Pin each action and use least privilege.
---

# CI, Taskfile and hooks stay in sync

- The `ci` task in `Taskfile.dist.yaml` mirrors the jobs in `.github/workflows/ci.yml`. If
  a change adds, removes or renames a CI job, or changes its flags, make the same change to
  `task ci` and to the task that it calls. Apply this rule in the other direction too. Flag
  each difference in review.
- Local hooks in `.pre-commit-config.yaml` call Taskfile tasks. The hooks must not contain
  copies of the commands.
- Pin each `uses:` to a full 40-character commit SHA, with a `# vX.Y.Z` comment. Reject tag
  refs and branch refs.
- The workflow sets `permissions: {}` at the top. Each job sets the least privilege that it
  needs. Each `actions/checkout` sets `persist-credentials: false`.
- Do not use `pull_request_target`. Do not put `${{ github.event.* }}` or other untrusted
  expressions directly in `run:`. Pass them through `env:`.
- Do not rename `publish-crates-io.yml` or its `crates-io` environment. The crates.io
  trusted publishing configuration names both. Deploy jobs run in their own tag-restricted
  environment. Each deploy job `needs:` a verification job.
- `scripts/check-branch.sh` is the single definition of the allowed branch names. The
  Taskfile, the hook and the CI `branch` job use it.
- The `ci.yml` jobs are `branch`, `fmt`, `clippy`, `style`, `doc`, `test` (Linux, macOS,
  Windows), `coverage`, `deny`, `msrv` and `fuzz`. `security.yml` (CodeQL, clippy and zizmor
  SARIF, SBOM) and `coverage` are CI-only. They have no step in `task ci`.
- Add the `semver` job after the first release on crates.io. Before that it has no baseline.
- If a new job must pass before a merge, also add it to branch protection. Say this in the
  PR.
