//! Identifiability of an estimate, read from the estimator's covariance.
//!
//! [`Identifiability`] holds what you require of the estimate: a **tolerance** per parameter,
//! the standard deviation you are willing to accept in that parameter's own units, and the
//! largest acceptable correlation between two parameters. It then judges a covariance:
//!
//! - from a [`RecursiveLeastSquares`] estimator with [`assess_rls`](Identifiability::assess_rls),
//!   which reconstructs `P` from its UD factors;
//! - from the parameter block of an augmented-state EKF or UKF with
//!   [`assess_block`](Identifiability::assess_block), passing
//!   [`estimate_covariance()`](minikalman::extended::ExtendedKalman::estimate_covariance) and
//!   the indices of the parameters in the state;
//! - from any `N` × `N` covariance with [`assess`](Identifiability::assess).
//!
//! ## Scaling
//!
//! Parameters have different units, so eigenvectors of `P` itself mix metres with
//! kilograms. Every quantity here is computed on the tolerance-scaled covariance
//! `P̃ = T⁻¹·P·T⁻¹`, `T = diag(tolerance)`, in which `1` means "exactly at tolerance". The
//! largest eigenvalue of `P̃` belongs to the worst-determined parameter *combination*, which
//! can be far worse than any single parameter: two strongly correlated parameters may each
//! sit within tolerance while their difference is unknown.
//!
//! ## The prior
//!
//! `P` includes the prior `P₀`. A small `p0` makes a parameter look identified before any data
//! arrived. Start from a `p0` well above the squared tolerance, so that only data can bring a
//! parameter within it.
//!
//! [`RecursiveLeastSquares`]: minikalman::rls::RecursiveLeastSquares

use crate::linalg::{abs, sqrt_clamped, symmetric_eigen};
use minikalman::matrix::{Matrix, MatrixData, MatrixDataType, MatrixMut};
use minikalman::rls::RecursiveLeastSquares;

/// Identifiability verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentifiabilityStatus {
    /// Every parameter is within tolerance and no pair is too strongly correlated.
    Identified,
    /// Every parameter is within tolerance, but at least one pair is correlated beyond the
    /// limit: the data hardly tells them apart, and the estimate of their difference rests on
    /// few samples or on the prior.
    Correlated,
    /// At least one parameter is outside its tolerance.
    Unidentified,
}

/// Identifiability of `N` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdentifiabilityReport<T, const N: usize> {
    /// The verdict.
    pub status: IdentifiabilityStatus,
    /// Standard deviation of each parameter, `√Pᵢᵢ`.
    pub std_dev: [T; N],
    /// `std_dev[i] / tolerance[i]`; above `1` is out of tolerance.
    pub relative_std_dev: [T; N],
    /// Index of the parameter with the largest relative standard deviation.
    pub least_identified: usize,
    /// Relative standard deviation of the worst-determined parameter combination, `√λ_max`
    /// of the tolerance-scaled covariance. At least the largest of
    /// [`relative_std_dev`](Self::relative_std_dev).
    pub worst_relative_std_dev: T,
    /// That combination: a unit vector in tolerance-scaled coordinates (sign arbitrary).
    /// Entry `i` weighs `θᵢ/toleranceᵢ`; e.g. `(0.71, 0.71)` means `θ₀/t₀ + θ₁/t₁` is poorly
    /// known.
    pub worst_direction: [T; N],
    /// The most strongly correlated pair `(i, j, ρᵢⱼ)` with `i < j`, or `None` for fewer than
    /// two parameters.
    pub strongest_correlation: Option<(usize, usize, T)>,
}

/// What is required of an estimate of `N` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Identifiability<T, const N: usize> {
    tolerance: [T; N],
    max_correlation: T,
}

impl<T: MatrixDataType, const N: usize> Identifiability<T, N> {
    /// Creates the criteria from the acceptable standard deviation of each parameter, in its
    /// own units, and the largest acceptable `|ρ|` between two parameters (e.g. `0.95`).
    ///
    /// # Panics
    /// If a tolerance is not positive. Use [`Self::try_new`] to avoid panicking.
    pub fn new(tolerance: [T; N], max_correlation: T) -> Self {
        Self::try_new(tolerance, max_correlation).expect("tolerances must be positive")
    }

    /// Creates the criteria, or `None` if a tolerance is not positive.
    pub fn try_new(tolerance: [T; N], max_correlation: T) -> Option<Self> {
        tolerance.iter().all(|&t| t > T::zero()).then_some(Self {
            tolerance,
            max_correlation,
        })
    }

    /// The tolerances.
    pub fn tolerance(&self) -> &[T; N] {
        &self.tolerance
    }

    /// The correlation limit.
    pub fn max_correlation(&self) -> T {
        self.max_correlation
    }

    /// Judges the covariance of a [`RecursiveLeastSquares`] estimator, in either
    /// [`CovarianceForm`](minikalman::rls::CovarianceForm). `O(N³)`, with an `N` × `N`
    /// scratch matrix on the stack.
    pub fn assess_rls<Theta, P, TempF, TempK>(
        &self,
        rls: &RecursiveLeastSquares<N, T, Theta, P, TempF, TempK>,
    ) -> IdentifiabilityReport<T, N>
    where
        Theta: MatrixMut<N, 1, T>,
        P: MatrixMut<N, N, T>,
        TempF: MatrixMut<N, 1, T>,
        TempK: MatrixMut<N, 1, T>,
    {
        let mut p = [[T::zero(); N]; N];
        rls.covariance(&mut MatrixData::new_mut::<N, N, T>(p.as_flattened_mut()));
        self.assess_array(&p)
    }

    /// Judges an `N` × `N` parameter covariance.
    pub fn assess<M: Matrix<N, N, T>>(&self, covariance: &M) -> IdentifiabilityReport<T, N> {
        let mut p = [[T::zero(); N]; N];
        for (i, row) in p.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = covariance.get(i, j);
            }
        }
        self.assess_array(&p)
    }

    /// Judges the parameter block of a larger covariance, e.g. the estimate covariance of an
    /// augmented-state filter whose state is `[x…, θ…]`. `indices[k]` is the state index of
    /// parameter `k`; the report is in the order of `indices`.
    ///
    /// The marginal covariance of the parameters is exactly this block, so the verdict
    /// accounts for the uncertainty of the states they are estimated jointly with.
    ///
    /// # Panics
    /// If an index is not below `S`.
    pub fn assess_block<const S: usize, M: Matrix<S, S, T>>(
        &self,
        covariance: &M,
        indices: &[usize; N],
    ) -> IdentifiabilityReport<T, N> {
        let mut p = [[T::zero(); N]; N];
        for (row, &i) in p.iter_mut().zip(indices) {
            for (x, &j) in row.iter_mut().zip(indices) {
                assert!(i < S && j < S, "parameter index out of range");
                *x = covariance.get(i, j);
            }
        }
        self.assess_array(&p)
    }

    /// Judges a covariance given as an array.
    pub fn assess_array(&self, p: &[[T; N]; N]) -> IdentifiabilityReport<T, N> {
        let zero = T::zero();
        let one = T::one();

        let mut std_dev = [zero; N];
        let mut relative = [zero; N];
        let mut least_identified = 0;
        for i in 0..N {
            std_dev[i] = sqrt_clamped(p[i][i]);
            relative[i] = std_dev[i] / self.tolerance[i];
            if relative[i] > relative[least_identified] {
                least_identified = i;
            }
        }

        let mut scaled = [[zero; N]; N];
        for i in 0..N {
            for j in 0..N {
                scaled[i][j] = p[i][j] / self.tolerance[i] / self.tolerance[j];
            }
        }
        let e = symmetric_eigen(&scaled);
        let (worst_relative_std_dev, worst_direction) = if N == 0 {
            (zero, [zero; N])
        } else {
            (sqrt_clamped(e.values[N - 1]), e.vectors[N - 1])
        };

        let mut strongest_correlation: Option<(usize, usize, T)> = None;
        for i in 0..N {
            for j in (i + 1)..N {
                let d = std_dev[i] * std_dev[j];
                let rho = if d > zero { p[i][j] / d } else { zero };
                if strongest_correlation.is_none_or(|(_, _, r)| abs(rho) > abs(r)) {
                    strongest_correlation = Some((i, j, rho));
                }
            }
        }

        let status = if relative.iter().any(|&r| r > one) {
            IdentifiabilityStatus::Unidentified
        } else if strongest_correlation.is_some_and(|(_, _, r)| abs(r) > self.max_correlation) {
            IdentifiabilityStatus::Correlated
        } else {
            IdentifiabilityStatus::Identified
        };

        IdentifiabilityReport {
            status,
            std_dev,
            relative_std_dev: relative,
            least_identified,
            worst_relative_std_dev,
            worst_direction,
            strongest_correlation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prbs::Prbs;
    use minikalman::rls::{CovarianceForm, EstimatorStorage};

    #[test]
    fn rejects_non_positive_tolerance() {
        assert!(Identifiability::<f64, 2>::try_new([0.1, 0.0], 0.9).is_none());
    }

    #[test]
    fn scaled_covariance_verdicts() {
        let id = Identifiability::<f64, 2>::new([0.1, 2.0], 0.9);

        // σ = (0.05, 1.0): both within tolerance, uncorrelated.
        let r = id.assess_array(&[[0.0025, 0.0], [0.0, 1.0]]);
        assert_eq!(r.status, IdentifiabilityStatus::Identified);
        assert!((r.relative_std_dev[0] - 0.5).abs() < 1e-12);
        assert!((r.relative_std_dev[1] - 0.5).abs() < 1e-12);

        // Same marginals, ρ = −0.99: each within tolerance, but the data barely separates them;
        // the scaled difference is the worst-determined combination.
        let c = -0.99 * 0.05 * 1.0;
        let r = id.assess_array(&[[0.0025, c], [c, 1.0]]);
        assert_eq!(r.status, IdentifiabilityStatus::Correlated);
        assert!((r.strongest_correlation.unwrap().2 + 0.99).abs() < 1e-12);
        assert!(r.worst_relative_std_dev > 0.7);
        let d = r.worst_direction;
        assert!((d[0].abs() - d[1].abs()).abs() < 1e-9);
        assert!(d[0] * d[1] < 0.0);

        // σ₁ = 3 > 2.
        let r = id.assess_array(&[[0.0025, 0.0], [0.0, 9.0]]);
        assert_eq!(r.status, IdentifiabilityStatus::Unidentified);
        assert_eq!(r.least_identified, 1);
    }

    fn run_rls(form: CovarianceForm, excite: bool) -> IdentifiabilityReport<f64, 2> {
        // y = b1·u_{k-1} + b2·u_{k-2}
        let mut storage = EstimatorStorage::<2, f64>::new(0.0);
        let mut rls = storage.build(100.0);
        rls.reset_with_form(form, 100.0);
        rls.set_forgetting_factor(0.98);
        rls.set_measurement_variance(0.01);

        let mut prbs = Prbs::new(8, 1).unwrap();
        let (mut u1, mut u2) = (0.0, 0.0);
        for _ in 0..300 {
            let phi = [u1, u2];
            let y = 0.5 * u1 + 0.2 * u2;
            rls.update(&phi, y);
            u2 = u1;
            u1 = if excite { prbs.next_level(1.0) } else { 1.0 };
        }
        Identifiability::new([0.05, 0.05], 0.9).assess_rls(&rls)
    }

    #[test]
    fn rls_with_prbs_is_identified_in_both_forms() {
        for form in [CovarianceForm::Factored, CovarianceForm::Covariance] {
            let r = run_rls(form, true);
            assert_eq!(r.status, IdentifiabilityStatus::Identified, "{form:?}");
        }
    }

    #[test]
    fn rls_with_constant_input_is_not() {
        let r = run_rls(CovarianceForm::Factored, false);
        assert_eq!(r.status, IdentifiabilityStatus::Unidentified);
        // The difference b1 − b2 is the unknown combination.
        let d = r.worst_direction;
        assert!((d[0] + d[1]).abs() < 1e-3, "{d:?}");
    }

    #[test]
    #[cfg(feature = "fixed")]
    fn fixed_point_verdicts() {
        use minikalman::prelude::I16F16;
        let f = I16F16::from_num;
        let id = Identifiability::new([f(0.1), f(2.0)], f(0.9));

        let r = id.assess_array(&[[f(0.0025), f(0.0)], [f(0.0), f(1.0)]]);
        assert_eq!(r.status, IdentifiabilityStatus::Identified);
        assert!((r.relative_std_dev[0] - f(0.5)).abs() < f(0.01));

        let c = f(-0.99 * 0.05);
        let r = id.assess_array(&[[f(0.0025), c], [c, f(1.0)]]);
        assert_eq!(r.status, IdentifiabilityStatus::Correlated);
        assert!((r.worst_relative_std_dev - f(0.705)).abs() < f(0.01));

        let r = id.assess_array(&[[f(0.0025), f(0.0)], [f(0.0), f(9.0)]]);
        assert_eq!(r.status, IdentifiabilityStatus::Unidentified);
    }

    #[test]
    fn block_matches_full() {
        let p = [
            [1.0, 0.2, 0.0, 0.1],
            [0.2, 2.0, 0.3, 0.0],
            [0.0, 0.3, 0.04, 0.01],
            [0.1, 0.0, 0.01, 0.09],
        ];
        let m = MatrixData::new_ref::<4, 4, f64>(p.as_flattened());
        let id = Identifiability::new([0.5, 0.5], 0.95);
        let block = id.assess_block(&m, &[2, 3]);
        let direct = id.assess_array(&[[0.04, 0.01], [0.01, 0.09]]);
        assert_eq!(block, direct);
    }
}
