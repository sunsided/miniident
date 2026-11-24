# miniident

Allocation-free, `no_std` online system identification diagnostics for Rust.
A companion to [`minikalman`](https://github.com/sunsided/minikalman-rs).

`minikalman` estimates. `miniident` tells you whether what it estimates can be
trusted, and helps you produce data that makes it trustworthy:

> A self-calibrating system that knows what it doesn't know.

## Scope

`miniident` sits **above** the estimation layer. It consumes the same
regressors (linear-in-parameters models) or output sensitivities (grey-box
models) that the estimator uses, and never touches the estimator's state.
This keeps the crates decoupled: no shared matrix traits, no version lock-step,
and `miniident` works equally well next to a hand-written RLS.

Design rules, inherited from `minikalman`:

- `no_std`, no allocation, no `unsafe`
- const-generic dimensions, plain arrays at the API boundary
- generic over `f32` / `f64` via `num_traits::Float` (with `libm`)
- O(P²) per sample on the hot path; O(P³) work is explicit and can run at a lower rate

## What is in 0.1 (this draft)

| Module        | Contents                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------- |
| `information` | Exponentially weighted information matrix from regressors or sensitivities; log-det; Cramér–Rao covariance bound; correlation matrix |
| `excitation`  | Persistent excitation monitor: `Warmup` / `Insufficient` / `IllConditioned` / `Sufficient`, weakest parameter direction, number of deficient directions |
| `prbs`        | Maximal-length PRBS, orders 2–32, bit hold for band limiting, offset dithering                    |
| `linalg`      | Cyclic Jacobi symmetric eigen-decomposition, Cholesky log-determinant                             |

```rust
use miniident::excitation::{ExcitationMonitor, ExcitationThresholds, ExcitationStatus};

let mut monitor = ExcitationMonitor::<f32, 3>::new(0.98, ExcitationThresholds::default());

// alongside the estimator update, with the same regressor:
monitor.update(&phi);

// at a lower rate:
let report = monitor.assess();
if report.status != ExcitationStatus::Sufficient {
    // freeze adaptation, inflate process noise only along excited directions,
    // or schedule an excitation manoeuvre along report.weakest_direction
}
```

`cargo run --example excitation` shows an ARX plant losing identifiability
under constant input and regaining it under PRBS.

### Notes on interpretation

- Thresholds apply to the **normalized** information matrix (windowed average
  of `w φ φᵀ`), so they are independent of λ and runtime. They are in units of
  regressor power, so regressor scaling matters: a large DC component in one
  regressor (e.g. an output sitting at its operating point) inflates the
  condition number. Centre or scale regressors where the model allows it.
- Use the estimator's forgetting factor in the monitor, so both describe the
  same data window.
- For grey-box models, `update_sensitivity` gives the local Fisher information.
  It is only as good as the linearization point: identifiability can differ
  between operating points.

## Roadmap

| Release | Scope |
| ------- | ----- |
| 0.1 | Information matrix, excitation monitor, PRBS (this draft) |
| 0.2 | Sensitivity propagation helpers for grey-box models (forward sensitivity of `x_{k+1} = f(x, u, θ)`); per-parameter identifiability report (CRB standard deviations, correlated pairs); multisine with Schroeder phases |
| 0.3 | Residual validation: online whiteness (normalized autocorrelation over a lag window), input–residual cross-correlation, NIS/NEES consistency checks |
| 0.4 | Recursive estimators that `minikalman` does not cover: RPEM, recursive instrumental variables, parameter projection for bounds and stability constraints |
| 0.5 | Experiment design: D-/A-/E-optimal input selection from a candidate set, constrained to amplitude and rate limits; directional excitation along the weakest eigenvector |
| later | Recursive subspace identification (PAST / recursive MOESP), EM on top of the `minikalman` RTS smoother, model order selection |

## Open questions

- **Eigen-decomposition cost.** Jacobi is O(P³) per `assess`. For embedded use
  with larger P, a rank-one eigen-update or tracking only the smallest
  eigenpair by inverse iteration may be worth it.
- **Coupling with minikalman.** Is it worth an optional `minikalman` feature
  that derives the information matrix from the UD-factorized RLS covariance
  instead of tracking it separately? That saves O(P²) per step but ties the
  crates' versions together.
- **Fixed-point.** `minikalman` supports fixed-point arithmetic; this draft is
  float-only. The monitor needs `sqrt` and comparisons only, so a fixed-point
  path is plausible.

## License

MIT OR Apache-2.0
