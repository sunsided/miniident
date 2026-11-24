# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.2](https://github.com/sunsided/miniident/releases/tag/v0.1.0-alpha.2) - 2026-11-24

### Added

- Add a "Who this is for" section to the README
- Add lint table with the High-assurance profile
- Add release-plz and tag-driven crates.io publishing
- Add CONTRIBUTING.md
- Add git hooks and Copilot review rules
- Add crate metadata and README badges
- Add property tests, fuzz targets and benchmarks
- Add license texts and REUSE annotations
- Add supply-chain checks and track Cargo.lock
- Add Codecov configuration
- Add Taskfile and CI workflows

### Changed

- Property-test identifiability under both forgetting modes
- Accept minikalman 0.9.1 estimators with directional forgetting
- Name the real remedy for diverging sensitivities
- Propagate output sensitivities of grey-box models
- Compare the covariance with the prior it started from
- Detect indefinite covariances by the smallest scaled eigenvalue
- Treat a negative variance as a broken covariance
- Build fuzz targets for the gnu target in CI
- Document NaN and fixed-point overflow behavior
- Check the MSRV on the library only
- Document the example and drop its unchecked casts
- Split the identifiability verdict into documented steps
- Read the excitation extremes without indexing
- Accumulate information without index loops
- Make PRBS integer arithmetic explicit
- Restructure the Jacobi eigen solver for the High-assurance lints
- Render mermaid on docs.rs and doctest the README
- Build on minikalman and add identifiability assessment
- Initial commit

### Fixed

- Fix NaN covariance being reported as identified

## [0.1.0-alpha.1](https://github.com/sunsided/miniident/releases/tag/v0.1.0-alpha.1) - 2026-11-24

### Added

- Add lint table with the High-assurance profile
- Add release-plz and tag-driven crates.io publishing
- Add CONTRIBUTING.md
- Add git hooks and Copilot review rules
- Add crate metadata and README badges
- Add property tests, fuzz targets and benchmarks
- Add license texts and REUSE annotations
- Add supply-chain checks and track Cargo.lock
- Add Codecov configuration
- Add Taskfile and CI workflows

### Changed

- Detect indefinite covariances by the smallest scaled eigenvalue
- Treat a negative variance as a broken covariance
- Build fuzz targets for the gnu target in CI
- Document NaN and fixed-point overflow behavior
- Check the MSRV on the library only
- Document the example and drop its unchecked casts
- Split the identifiability verdict into documented steps
- Read the excitation extremes without indexing
- Accumulate information without index loops
- Make PRBS integer arithmetic explicit
- Restructure the Jacobi eigen solver for the High-assurance lints
- Render mermaid on docs.rs and doctest the README
- Build on minikalman and add identifiability assessment
- Initial commit

### Fixed

- Fix NaN covariance being reported as identified
