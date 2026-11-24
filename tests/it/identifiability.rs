//! Properties of the identifiability verdict on arbitrary positive-definite covariances.

use crate::strategies::{N, positive_definite, tolerances};
use miniident::identifiability::{Identifiability, IdentifiabilityStatus};
use proptest::prelude::*;

/// Correlation limit the properties use.
const MAX_CORRELATION: f64 = 0.95;

/// Severity rank of a verdict: higher is worse.
fn severity(status: IdentifiabilityStatus) -> u8 {
    match status {
        IdentifiabilityStatus::Identified => 0,
        IdentifiabilityStatus::Correlated => 1,
        IdentifiabilityStatus::Unidentified => 2,
    }
}

/// Builds the criteria, or fails the case if a generated tolerance is not positive.
fn criteria(tolerance: [f64; N]) -> Result<Identifiability<f64, N>, TestCaseError> {
    Identifiability::try_new(tolerance, MAX_CORRELATION)
        .ok_or_else(|| TestCaseError::fail("generated tolerances are positive"))
}

proptest! {
    /// A looser tolerance never makes the verdict worse.
    #[test]
    fn looser_tolerance_never_worsens_the_verdict(
        p in positive_definite(),
        tolerance in tolerances(),
        factor in 1.0..100.0_f64,
    ) {
        let loose = tolerance.map(|t| t * factor);
        let strict = criteria(tolerance)?.assess_array(&p);
        let relaxed = criteria(loose)?.assess_array(&p);

        prop_assert!(severity(relaxed.status) <= severity(strict.status), "{:?} then {:?}", strict.status, relaxed.status);
    }

    /// The worst combination is at least as uncertain as the worst single parameter.
    #[test]
    fn worst_combination_bounds_every_parameter(p in positive_definite(), tolerance in tolerances()) {
        let r = criteria(tolerance)?.assess_array(&p);
        let worst_single = r.relative_std_dev.iter().copied().fold(0.0, f64::max);

        prop_assert!(r.worst_relative_std_dev >= worst_single * (1.0 - 1e-9), "{} < {worst_single}", r.worst_relative_std_dev);
        prop_assert!(r.relative_std_dev[r.least_identified] >= worst_single, "least_identified points at the maximum");
    }

    /// Scaling a parameter's units scales its standard deviation and leaves the relative
    /// figures unchanged.
    #[test]
    fn relative_figures_do_not_depend_on_units(
        p in positive_definite(),
        tolerance in tolerances(),
        scale in proptest::array::uniform4(0.1..10.0_f64),
    ) {
        let mut scaled_p = p;
        for (i, row) in scaled_p.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x *= scale[i] * scale[j];
            }
        }
        let scaled_tolerance: [f64; N] = core::array::from_fn(|i| tolerance[i] * scale[i]);

        let a = criteria(tolerance)?.assess_array(&p);
        let b = criteria(scaled_tolerance)?.assess_array(&scaled_p);

        for (x, y) in a.relative_std_dev.iter().zip(&b.relative_std_dev) {
            prop_assert!((x - y).abs() <= 1e-9 * x.max(1.0), "{x} vs {y}");
        }
        prop_assert!(
            (a.worst_relative_std_dev - b.worst_relative_std_dev).abs() <= 1e-9 * a.worst_relative_std_dev.max(1.0),
            "{} vs {}", a.worst_relative_std_dev, b.worst_relative_std_dev,
        );
    }
}
