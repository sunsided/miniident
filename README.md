# miniident

[![crates.io](https://img.shields.io/crates/v/miniident.svg)](https://crates.io/crates/miniident)
[![docs.rs](https://img.shields.io/docsrs/miniident)](https://docs.rs/miniident)
[![CI](https://github.com/sunsided/miniident/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/miniident/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/miniident/graph/badge.svg)](https://codecov.io/gh/sunsided/miniident)
[![MSRV](https://img.shields.io/crates/msrv/miniident)](https://crates.io/crates/miniident)
[![License](https://img.shields.io/crates/l/miniident)](https://github.com/sunsided/miniident#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Online identifiability and excitation diagnostics on top of
[minikalman](https://github.com/sunsided/minikalman-rs).

minikalman estimates parameters, with `RecursiveLeastSquares` or in the state of an
augmented EKF/UKF. miniident tells you whether that estimate can be trusted, which
parameters the data cannot pin down, and gives you input that fixes it:

> A self-calibrating system that knows what it doesn't know.

## Design

- **The estimator's covariance is the source of truth.** For RLS,
  `Pₖ₊₁⁻¹ = μₖ·Pₖ⁻¹ + φₖ·φₖᵀ/r`: the windup guard's variable `μₖ` (or, with
  minikalman's `Directional` forgetting, forgetting along `φₖ` only), the measurement
  variance, the prior and every gated sample are already in `P`. miniident reads `P`
  (reconstructed from the UD factors where needed) instead of tracking a second
  information matrix that would drift from it. For augmented-state filters, the
  parameter block of `estimate_covariance()` plays the same role.
- **minikalman's numeric types.** Generic over `MatrixDataType`, so `f32`, `f64` and
  the fixed-point types; `Logarithm` where a logarithm is needed. Same features as
  minikalman (`libm` default, `std`, `micromath`, `fixed`).
- **minikalman's philosophy.** `no_std`, no allocation, no `unsafe`, const-generic
  dimensions, edition 2024, MSRV 1.85.
- **Tolerances in parameter units.** Parameters have different units, so verdicts are
  computed on the tolerance-scaled covariance, where `1` means "exactly at tolerance".

## Contents

| Module            | Contents |
| ----------------- | -------- |
| `identifiability` | `Identifiability::assess_rls`, `assess_block` (augmented EKF/UKF), `assess`: per-parameter σ and σ/tolerance, worst-determined parameter combination, strongest correlated pair, verdict `Identified` / `Correlated` / `Unidentified`; `prior_ratio` / `prior_ratio_rls`: how much of `P` is still the prior `P₀` |
| `excitation`      | `ExcitationMonitor` on the regressors (`for_rls` copies the estimator's μ): `Warmup` / `Insufficient` / `IllConditioned` / `Sufficient`, weakest direction, number of deficient directions |
| `information`     | Exponentially weighted information matrix from regressors or grey-box output sensitivities; log-determinant (D-optimality) |
| `sensitivity`     | `Sensitivity`: propagates `∂y/∂θ` of a grey-box model `x_{k+1} = f(x, u, θ)`, `y = h(x, u, θ)` from its Jacobians; `jacobian` fills them by central differences |
| `prbs`            | Maximal-length PRBS, orders 2–32, bit hold for band limiting, dithering around an operating point |
| `linalg`          | Jacobi symmetric eigen-decomposition (minikalman has Cholesky, but no eigen) and Cholesky log-determinant |

```rust
use miniident::prelude::*;
use minikalman::rls::EstimatorStorage;

let mut storage = EstimatorStorage::<2, f32>::new(0.0);
let mut rls = storage.build(10.0);
rls.set_forgetting_factor(0.98);

// a to ±0.02, b to ±0.05, |ρ| at most 0.95
let identifiability = Identifiability::new([0.02, 0.05], 0.95);

// regressor φ and measurement y, one pair per sample
for (phi, y) in [([0.0_f32, 1.0], 0.5), ([0.5, -1.0], -0.05)] {
    rls.update(&phi, y);

    let report = identifiability.assess_rls(&rls);
    match report.status {
        IdentifiabilityStatus::Identified => { /* hand θ to the controller */ }
        IdentifiabilityStatus::Correlated => { /* early warning: excite soon */ }
        IdentifiabilityStatus::Unidentified => {
            // keep the last trusted θ, excite along report.worst_direction
        }
    }
}
```

## Example

`cargo run --example rls-identifiability` identifies the first-order motor of
minikalman's `rls-adaptive` example through PRBS, rest and PRBS again:

```text
 step  input  a_hat   b_hat   σa/tol  σb/tol  worst  identifiability  excitation
  300  prbs    0.902   0.491    0.06    0.11    0.1  Identified       Sufficient
  350  rest    0.901   0.490    0.10    0.19    0.2  Correlated       Sufficient
  500  rest    0.901   0.495    0.42    0.84    0.9  Correlated       Sufficient
  550  rest    0.902   0.490    0.69    1.38    1.5  Unidentified     IllConditioned
  700  rest    0.908   0.460    2.31    4.62    5.2  Unidentified     Insufficient
  750  prbs    0.902   0.489    0.08    0.15    0.2  Identified       Sufficient
```

At rest only the steady-state relation is observed. The correlation verdict fires
~200 steps before any tolerance is violated, and the worst direction is the
combination the steady state cannot see.

`cargo run --example greybox-sensitivity` does the same for a motor with smooth Coulomb
friction, `v_{k+1} = a·v + b·u − c·tanh(v/ε)`, which has no linear regressor: `Sensitivity`
propagates `∂y/∂θ` through the model's Jacobians into `ExcitationMonitor::update_sensitivity`,
which turns `IllConditioned` at rest along the combination the steady state cannot see.

## Roadmap

| Release | Scope |
| ------- | ----- |
| 0.1 | Identifiability from RLS and augmented-filter covariances, excitation monitor, PRBS |
| 0.2 | Sensitivity propagation for grey-box models `x_{k+1} = f(x, u, θ)`, so `update_sensitivity` can be fed without hand-derived `∂y/∂θ`; identifiability relative to a diagonal prior (data vs. `P₀`) (this draft) |
| 0.3 | Model validation on minikalman's innovations: whiteness over a lag window, input–residual cross-correlation, NIS/NEES consistency |
| 0.4 | Excitation design: D-/E-optimal choice among candidate inputs under amplitude and rate limits; excitation along `worst_direction` |
| later | Recursive IV/RPEM, recursive subspace identification, EM on the RTS smoother |

## Open questions

- **Cost of `assess`.** `O(N³)` Jacobi per call. For larger `N` on small MCUs, tracking
  only the extreme eigenpair by power iteration on `P` would be `O(N²)`.
- **Fixed point.** The eigen-decomposition avoids epsilon tests and squared quotients
  so it runs on `I16F16`/`I32F32`, and the verdict test passes in `I16F16`, but it is
  not yet exercised against a fixed-point RLS run.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](https://github.com/sunsided/miniident/blob/main/LICENSE-APACHE-2.0))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/miniident/blob/main/LICENSE-MIT))

at your option. The examples in `examples/` are public domain
([CC0-1.0](https://github.com/sunsided/miniident/blob/main/LICENSES/CC0-1.0.txt)), so you can
copy them into anything.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
