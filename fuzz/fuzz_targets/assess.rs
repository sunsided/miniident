//! Fuzzes the identifiability verdict with arbitrary covariances and tolerances.
//!
//! The verdict must never panic, also for NaN, infinite and huge entries. The report must
//! name parameter indices in range, and a covariance with a NaN entry must be unidentified.

#![no_main]

use libfuzzer_sys::fuzz_target;
use miniident::identifiability::{Identifiability, IdentifiabilityStatus};

/// Number of parameters the target uses.
const N: usize = 3;

/// One fuzz case: a covariance, which need not be symmetric, plus the criteria.
#[derive(Debug, arbitrary::Arbitrary)]
struct Case {
    /// The covariance to judge.
    covariance: [[f64; N]; N],
    /// The tolerance per parameter; non-positive values must be rejected by `try_new`.
    tolerance: [f64; N],
    /// The correlation limit.
    max_correlation: f64,
}

fuzz_target!(|case: Case| {
    let Some(criteria) = Identifiability::try_new(case.tolerance, case.max_correlation) else {
        return;
    };
    let report = criteria.assess_array(&case.covariance);

    assert!(
        report.least_identified < N,
        "least_identified is an index below N"
    );
    if case.covariance.iter().flatten().any(|x| x.is_nan()) {
        assert_eq!(
            report.status,
            IdentifiabilityStatus::Unidentified,
            "a NaN covariance is never trusted"
        );
    }
    if let Some((i, j, _)) = report.strongest_correlation {
        assert!(
            i < j && j < N,
            "the correlated pair is ordered and in range"
        );
    }
});
