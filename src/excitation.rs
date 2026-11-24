//! Persistent excitation monitoring.
//!
//! A regressor sequence is persistently exciting over a window when the
//! windowed information matrix is uniformly positive definite: every parameter
//! direction receives information. [`ExcitationMonitor`] tracks this online and
//! answers three questions:
//!
//! 1. Does every parameter direction receive information? (smallest eigenvalue)
//! 2. Is the problem well conditioned? (condition number)
//! 3. If not, *which* parameter combination is starved? (weakest eigenvector)
//!
//! The third answer is the actionable one. For a rover whose drive model has
//! motor gain and friction as parameters, a weakest direction of roughly
//! `(1, 1)/√2` says the data only constrains their difference, which happens
//! when the rover never changes speed.
//!
//! Typical use is to gate estimator behaviour: freeze adaptation, raise
//! process noise only in excited directions, or schedule an excitation
//! manoeuvre when the verdict is not [`ExcitationStatus::Sufficient`].

use crate::information::InformationMatrix;
use num_traits::Float;

/// Thresholds for the excitation verdict, applied to the **normalized**
/// information matrix (see [`InformationMatrix::normalized`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExcitationThresholds<T> {
    /// Smallest acceptable eigenvalue, in units of (weighted) regressor power.
    /// With regressors of unit amplitude, `1e-3` means the weakest direction
    /// receives at least 0.1 % of unit power on average.
    pub min_eigenvalue: T,
    /// Largest acceptable condition number `λ_max / λ_min`.
    pub max_condition: T,
    /// Effective samples required before any verdict other than
    /// [`ExcitationStatus::Warmup`] is given.
    pub min_samples: T,
}

impl<T: Float> Default for ExcitationThresholds<T> {
    fn default() -> Self {
        Self {
            min_eigenvalue: T::from(1e-3).unwrap(),
            max_condition: T::from(1e4).unwrap(),
            min_samples: T::from(10).unwrap(),
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
    /// All directions are excited, but very unevenly; estimates in the
    /// weakest direction will be slow and noise sensitive.
    IllConditioned,
    /// All parameter directions are adequately excited.
    Sufficient,
}

/// Snapshot of the excitation state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExcitationReport<T, const P: usize> {
    /// The verdict.
    pub status: ExcitationStatus,
    /// Smallest eigenvalue of the normalized information matrix.
    pub min_eigenvalue: T,
    /// Largest eigenvalue of the normalized information matrix.
    pub max_eigenvalue: T,
    /// `max_eigenvalue / min_eigenvalue`; infinite if the minimum is not positive.
    pub condition: T,
    /// Unit vector of the least informed parameter combination (sign arbitrary).
    pub weakest_direction: [T; P],
    /// Number of eigenvalues below `min_eigenvalue`: how many independent
    /// parameter combinations are currently unidentifiable. The corresponding
    /// directions are the first eigenvectors of
    /// [`InformationMatrix::eigen`](crate::information::InformationMatrix::eigen).
    pub deficient_directions: usize,
    /// Effective samples in the window.
    pub effective_samples: T,
}

/// Online persistent-excitation monitor for `P` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExcitationMonitor<T, const P: usize> {
    info: InformationMatrix<T, P>,
    thresholds: ExcitationThresholds<T>,
}

impl<T: Float, const P: usize> ExcitationMonitor<T, P> {
    /// Creates a monitor with forgetting factor `lambda`.
    ///
    /// Use the estimator's forgetting factor so that the monitor judges the
    /// same data window the estimator learns from.
    ///
    /// # Panics
    /// If `lambda ∉ (0, 1]`.
    pub fn new(lambda: T, thresholds: ExcitationThresholds<T>) -> Self {
        Self {
            info: InformationMatrix::new(lambda),
            thresholds,
        }
    }

    /// Adds one regressor. O(P²).
    pub fn update(&mut self, phi: &[T; P]) {
        self.info.update(phi);
    }

    /// Adds one sample from output sensitivities (grey-box models). O(M·P²).
    pub fn update_sensitivity<const M: usize>(
        &mut self,
        sensitivities: &[[T; P]; M],
        inv_noise_variance: &[T; M],
    ) {
        self.info
            .update_sensitivity(sensitivities, inv_noise_variance);
    }

    /// Computes the current verdict. O(P³); call it at a lower rate than
    /// [`Self::update`] if P is large or the platform is slow.
    pub fn assess(&self) -> ExcitationReport<T, P> {
        let samples = self.info.effective_samples();
        let e = self.info.eigen();

        let (min, max, weakest) = if P == 0 {
            (T::zero(), T::zero(), [T::zero(); P])
        } else {
            (e.values[0], e.values[P - 1], e.vectors[0])
        };
        let condition = if min > T::zero() {
            max / min
        } else {
            T::infinity()
        };

        let th = &self.thresholds;
        let deficient_directions = e
            .values
            .iter()
            .filter(|&&v| !(v >= th.min_eigenvalue))
            .count();
        let status = if samples < th.min_samples {
            ExcitationStatus::Warmup
        } else if !(min >= th.min_eigenvalue) {
            ExcitationStatus::Insufficient
        } else if condition > th.max_condition {
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
    pub fn information(&self) -> &InformationMatrix<T, P> {
        &self.info
    }

    /// The configured thresholds.
    pub fn thresholds(&self) -> &ExcitationThresholds<T> {
        &self.thresholds
    }

    /// Replaces the thresholds without discarding accumulated information.
    pub fn set_thresholds(&mut self, thresholds: ExcitationThresholds<T>) {
        self.thresholds = thresholds;
    }

    /// Discards all accumulated information.
    pub fn reset(&mut self) {
        self.info.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prbs::Prbs;

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
        // Only b1 + b2 is identifiable; b1 - b2 is not.
        let d = r.weakest_direction;
        assert!((d[0] + d[1]).abs() < 1e-9);
        assert!(r.condition.is_infinite() || r.condition > 1e12);
        assert_eq!(r.deficient_directions, 1);
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
        assert!(r.condition < 3.0);
        assert_eq!(r.deficient_directions, 0);
    }

    #[test]
    fn excitation_decays_after_input_stops() {
        let mut m = ExcitationMonitor::<f64, 2>::new(0.95, ExcitationThresholds::default());
        let mut prbs = Prbs::new(7, 1).unwrap();
        let mut prev = 0.0;
        for _ in 0..200 {
            let u = prbs.next_level(1.0);
            m.update(&[u, prev]);
            prev = u;
        }
        assert_eq!(m.assess().status, ExcitationStatus::Sufficient);
        for _ in 0..400 {
            m.update(&[0.5, 0.5]);
        }
        assert_ne!(m.assess().status, ExcitationStatus::Sufficient);
    }

    #[test]
    fn slow_input_is_ill_conditioned() {
        // Highly correlated neighbouring samples: u_k ≈ u_{k-1}.
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
}
