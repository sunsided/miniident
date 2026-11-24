//! Persistent excitation monitoring: is the *input* informative?
//!
//! The regressors are persistently exciting when the windowed information matrix is
//! uniformly positive definite, i.e. every parameter direction receives information.
//! [`ExcitationMonitor`] tracks this alongside the estimator and answers:
//!
//! 1. Does every parameter direction receive information? (smallest eigenvalue)
//! 2. Is the problem well conditioned? (condition number)
//! 3. If not, *which* parameter combination is starved? (weakest eigenvector)
//!
//! This is the cause; [`Identifiability`](crate::identifiability::Identifiability) on the
//! estimator's covariance is the effect. With the estimator's forgetting factor, both see the
//! same window and fade at the same rate when the input goes flat. For an early "the input
//! stopped exciting" signal, give the monitor a shorter window (a smaller `μ`) than the
//! estimator.

use crate::information::InformationMatrix;
use minikalman::matrix::{MatrixDataType, MatrixMut};
use minikalman::rls::RecursiveLeastSquares;

/// Thresholds for the excitation verdict, applied to the normalized information matrix (see
/// [`InformationMatrix::normalized`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExcitationThresholds<T> {
    /// Smallest acceptable eigenvalue, in units of regressor power. With regressors of unit
    /// amplitude, `1e-3` means the weakest direction receives at least 0.1 % of unit power.
    pub min_eigenvalue: T,
    /// Largest acceptable condition number `λ_max / λ_min`.
    pub max_condition: T,
    /// Effective samples required before any verdict other than
    /// [`ExcitationStatus::Warmup`].
    pub min_samples: T,
}

impl<T: MatrixDataType> Default for ExcitationThresholds<T> {
    fn default() -> Self {
        Self {
            min_eigenvalue: T::from_f64(1e-3).expect("representable"),
            max_condition: T::from_f64(1e4).expect("representable"),
            min_samples: T::from_f64(10.0).expect("representable"),
        }
    }
}

/// Excitation verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExcitationStatus {
    /// Not enough samples yet to judge.
    Warmup,
    /// At least one parameter direction receives (almost) no information.
    Insufficient,
    /// All directions are excited, but very unevenly: estimates in the weakest direction will
    /// be slow and noise sensitive.
    IllConditioned,
    /// All parameter directions are adequately excited.
    Sufficient,
}

/// Snapshot of the excitation state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExcitationReport<T, const N: usize> {
    /// The verdict.
    pub status: ExcitationStatus,
    /// Smallest eigenvalue of the normalized information matrix.
    pub min_eigenvalue: T,
    /// Largest eigenvalue of the normalized information matrix.
    pub max_eigenvalue: T,
    /// `max_eigenvalue / min_eigenvalue`, or `None` if the minimum is not positive.
    pub condition: Option<T>,
    /// Unit vector of the least excited parameter combination (sign arbitrary).
    pub weakest_direction: [T; N],
    /// How many independent parameter combinations are below `min_eigenvalue`; they are the
    /// first eigenvectors of [`InformationMatrix::eigen`].
    pub deficient_directions: usize,
    /// Effective samples in the window.
    pub effective_samples: T,
}

/// Online persistent-excitation monitor for `N` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExcitationMonitor<T, const N: usize> {
    /// The windowed information of the regressors.
    info: InformationMatrix<T, N>,
    /// The thresholds the verdict applies.
    thresholds: ExcitationThresholds<T>,
}

impl<T: MatrixDataType, const N: usize> ExcitationMonitor<T, N> {
    /// Creates a monitor with forgetting factor `mu`.
    ///
    /// # Panics
    /// If `mu ∉ (0, 1]`.
    pub fn new(mu: T, thresholds: ExcitationThresholds<T>) -> Self {
        Self {
            info: InformationMatrix::new(mu),
            thresholds,
        }
    }

    /// Creates a monitor that judges the same window as `rls`: its forgetting factor `μ`.
    ///
    /// The window matches as long as the estimator's windup guard is inactive; while it
    /// limits forgetting, the estimator remembers longer than the monitor does. That is the
    /// conservative direction for a warning.
    pub fn for_rls<Theta, P, TempF, TempK>(
        rls: &RecursiveLeastSquares<N, T, Theta, P, TempF, TempK>,
        thresholds: ExcitationThresholds<T>,
    ) -> Self
    where
        Theta: MatrixMut<N, 1, T>,
        P: MatrixMut<N, N, T>,
        TempF: MatrixMut<N, 1, T>,
        TempK: MatrixMut<N, 1, T>,
    {
        Self::new(rls.forgetting_factor(), thresholds)
    }

    /// Adds one regressor, the same `φ` passed to the estimator. `O(N²)`.
    pub fn update(&mut self, phi: &[T; N]) {
        self.info.update(phi);
    }

    /// Adds one sample from the output sensitivities of a grey-box model. `O(M·N²)`.
    pub fn update_sensitivity<const M: usize>(
        &mut self,
        sensitivities: &[[T; N]; M],
        inv_noise_variance: &[T; M],
    ) {
        self.info
            .update_sensitivity(sensitivities, inv_noise_variance);
    }

    /// Computes the verdict. `O(N³)`; call it at a lower rate than [`Self::update`] where
    /// that matters.
    #[must_use]
    pub fn assess(&self) -> ExcitationReport<T, N> {
        let samples = self.info.effective_samples();
        let e = self.info.eigen();
        let th = &self.thresholds;

        // The eigenvalues ascend, so the first pair is the weakest direction. N = 0 has none.
        let (min, max, weakest) = match (e.values.first(), e.values.last(), e.vectors.first()) {
            (Some(&min), Some(&max), Some(&weakest)) => (min, max, weakest),
            _ => (T::zero(), T::zero(), [T::zero(); N]),
        };
        let condition = condition_number(min, max);
        let deficient_directions = e
            .values
            .iter()
            .filter(|&&v| !is_at_least(v, th.min_eigenvalue))
            .count();

        let status = if samples < th.min_samples {
            ExcitationStatus::Warmup
        } else if deficient_directions > 0 {
            ExcitationStatus::Insufficient
        } else if condition.is_none_or(|c| c > th.max_condition) {
            ExcitationStatus::IllConditioned
        } else {
            ExcitationStatus::Sufficient
        };

        ExcitationReport {
            status,
            min_eigenvalue: min,
            max_eigenvalue: max,
            condition,
            weakest_direction: weakest,
            deficient_directions,
            effective_samples: samples,
        }
    }

    /// The underlying information matrix.
    pub fn information(&self) -> &InformationMatrix<T, N> {
        &self.info
    }

    /// The thresholds.
    pub fn thresholds(&self) -> &ExcitationThresholds<T> {
        &self.thresholds
    }

    /// Replaces the thresholds, keeping the accumulated information.
    pub fn set_thresholds(&mut self, thresholds: ExcitationThresholds<T>) {
        self.thresholds = thresholds;
    }

    /// Discards all accumulated information.
    pub fn reset(&mut self) {
        self.info.reset();
    }
}

/// `max / min`, or `None` if `min` is not positive or is NaN.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see linalg docs)"
)]
fn condition_number<T: MatrixDataType>(min: T, max: T) -> Option<T> {
    (min > T::zero()).then(|| max / min)
}

/// `value ≥ threshold`, and `false` for NaN, so a NaN eigenvalue counts as deficient.
fn is_at_least<T: MatrixDataType>(value: T, threshold: T) -> bool {
    matches!(
        value.partial_cmp(&threshold),
        Some(core::cmp::Ordering::Greater | core::cmp::Ordering::Equal)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prbs::Prbs;
    use minikalman::rls::EstimatorStorage;

    #[test]
    fn warmup_then_verdict() {
        let mut m = ExcitationMonitor::<f64, 2>::new(0.99, ExcitationThresholds::default());
        m.update(&[1.0, 0.0]);
        assert_eq!(m.assess().status, ExcitationStatus::Warmup);
    }

    #[test]
    fn constant_input_is_insufficient_with_correct_direction() {
        let mut m = ExcitationMonitor::<f64, 2>::new(0.99, ExcitationThresholds::default());
        for _ in 0..200 {
            m.update(&[2.0, 2.0]);
        }
        let r = m.assess();
        assert_eq!(r.status, ExcitationStatus::Insufficient);
        assert_eq!(r.deficient_directions, 1);
        // Only b1 + b2 is identifiable, b1 − b2 is not.
        assert!((r.weakest_direction[0] + r.weakest_direction[1]).abs() < 1e-9);
    }

    #[test]
    fn prbs_is_sufficient() {
        let mut m = ExcitationMonitor::<f32, 3>::new(0.98, ExcitationThresholds::default());
        let mut prbs = Prbs::new(9, 1).unwrap();
        let mut u = [0.0f32; 3];
        for _ in 0..1000 {
            u = [prbs.next_level(1.0), u[0], u[1]];
            m.update(&u);
        }
        let r = m.assess();
        assert_eq!(r.status, ExcitationStatus::Sufficient);
        assert!(r.condition.unwrap() < 3.0);
    }

    #[test]
    fn slow_input_is_ill_conditioned() {
        let th = ExcitationThresholds {
            min_eigenvalue: 1e-6,
            max_condition: 20.0,
            min_samples: 10.0,
        };
        let mut m = ExcitationMonitor::<f64, 2>::new(1.0, th);
        let mut prbs = Prbs::new(7, 50).unwrap();
        let mut prev = 0.0;
        for _ in 0..prbs.period_samples() {
            let u = prbs.next_level(1.0);
            m.update(&[u, prev]);
            prev = u;
        }
        assert_eq!(m.assess().status, ExcitationStatus::IllConditioned);
    }

    #[test]
    fn for_rls_copies_forgetting() {
        let mut storage = EstimatorStorage::<2, f64>::new(0.0);
        let mut rls = storage.build(100.0);
        rls.set_forgetting_factor(0.97);
        let m = ExcitationMonitor::for_rls(&rls, ExcitationThresholds::default());
        assert_eq!(m.information().forgetting_factor(), 0.97);
    }
}
