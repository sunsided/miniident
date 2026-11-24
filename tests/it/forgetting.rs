//! Identifiability under minikalman's two forgetting modes on the same data.

use miniident::identifiability::{Identifiability, IdentifiabilityReport, prior_ratio_rls};
use minikalman::rls::{Directional, EstimatorStorage, Exponential, Forgetting};
use proptest::prelude::*;

/// Number of parameters.
const N: usize = 3;

/// Tolerance of every parameter; the figures compared are ratios, so its value does not matter.
const TOLERANCE: f64 = 0.1;

/// One run: the regressors in order, then a constant regressor repeated `rest` times.
#[derive(Debug, Clone)]
struct Run {
    /// Excitation phase.
    excitation: Vec<[f64; N]>,
    /// Regressor held during the rest phase.
    held: [f64; N],
    /// Length of the rest phase.
    rest: usize,
    /// Forgetting factor `μ`.
    mu: f64,
    /// Measurement variance `r`.
    r: f64,
    /// Prior variance `p0`.
    p0: f64,
}

/// Runs of random excitation followed by a random rest.
fn runs() -> impl Strategy<Value = Run> {
    (
        proptest::collection::vec(proptest::array::uniform3(-1.0..1.0_f64), 10..100),
        proptest::array::uniform3(0.1..1.0_f64),
        0..200_usize,
        0.9..0.999_f64,
        0.01..1.0_f64,
        1.0..1000.0_f64,
    )
        .prop_map(|(excitation, held, rest, mu, r, p0)| Run {
            excitation,
            held,
            rest,
            mu,
            r,
            p0,
        })
}

/// Feeds `run` to an estimator with forgetting mode `F` and the windup guard off, and
/// returns its identifiability report and prior ratios.
fn assess<F: Forgetting>(run: &Run, mu: f64) -> (IdentifiabilityReport<f64, N>, [f64; N]) {
    let mut storage = EstimatorStorage::<N, f64>::new(0.0);
    let mut rls = storage.build(run.p0).with_forgetting::<F>();
    rls.set_forgetting_factor(mu);
    rls.set_measurement_variance(run.r);
    rls.set_trace_bound(None);

    let rest = core::iter::repeat_n(&run.held, run.rest);
    for phi in run.excitation.iter().chain(rest) {
        rls.update(phi, phi.iter().sum());
    }

    let report = Identifiability::new([TOLERANCE; N], 0.95).assess_rls(&rls);
    let prior = prior_ratio_rls(&rls, run.p0).map_or([f64::NAN; N], |p| p.variance_ratio);
    (report, prior)
}

/// `a ≤ b` up to a relative rounding margin.
fn at_most(a: f64, b: f64) -> bool {
    a <= b * (1.0 + 1e-6) + 1e-12
}

proptest! {
    /// Directional forgetting removes at most the information exponential forgetting removes
    /// (`φ·φᵀ/(φᵀ·P·φ) ≤ P⁻¹`), so on the same data no parameter is reported worse
    /// determined, and none is closer to its prior.
    #[test]
    fn directional_never_reports_worse_than_exponential(run in runs()) {
        let (exp, exp_prior) = assess::<Exponential>(&run, run.mu);
        let (dir, dir_prior) = assess::<Directional>(&run, run.mu);

        for i in 0..N {
            prop_assert!(at_most(dir.relative_std_dev[i], exp.relative_std_dev[i]), "σ{i}: {} > {}", dir.relative_std_dev[i], exp.relative_std_dev[i]);
            prop_assert!(at_most(dir_prior[i], exp_prior[i]), "prior ratio {i}: {} > {}", dir_prior[i], exp_prior[i]);
        }
        prop_assert!(at_most(dir.worst_relative_std_dev, exp.worst_relative_std_dev), "worst: {} > {}", dir.worst_relative_std_dev, exp.worst_relative_std_dev);
    }

    /// Without forgetting (`μ = 1`) the two modes are the same estimator.
    #[test]
    fn modes_agree_without_forgetting(run in runs()) {
        let (exp, exp_prior) = assess::<Exponential>(&run, 1.0);
        let (dir, dir_prior) = assess::<Directional>(&run, 1.0);

        prop_assert_eq!(exp.status, dir.status);
        for i in 0..N {
            prop_assert!((exp.relative_std_dev[i] - dir.relative_std_dev[i]).abs() <= 1e-9 * exp.relative_std_dev[i].max(1.0), "σ{i}");
            prop_assert!((exp_prior[i] - dir_prior[i]).abs() <= 1e-9, "prior ratio {i}");
        }
    }
}
