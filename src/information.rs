//! Exponentially weighted information matrix.
//!
//! For a linear-in-parameters model `y_k = φ_kᵀ θ + e_k` the information
//! accumulated from data is
//!
//! ```text
//! G_k = λ G_{k-1} + w_k φ_k φ_kᵀ
//! ```
//!
//! with forgetting factor `λ ∈ (0, 1]` and sample weight `w_k` (typically
//! `1/σ²`). For nonlinear grey-box models the regressor is replaced by the
//! output sensitivities `S_k = ∂y_k/∂θ` (an `M × P` Jacobian) and the update
//! becomes the Fisher information `G_k = λ G_{k-1} + S_kᵀ R⁻¹ S_k`.
//!
//! Use the **same** λ as the estimator you are monitoring; then `G` describes
//! exactly the information the estimator is working with. For an RLS
//! estimator `G` is the inverse of its covariance up to the noise variance and
//! initial prior, but tracking it separately avoids inverting the estimator's
//! covariance and keeps this crate independent of any estimator's storage.

use crate::linalg::{cholesky_log_det, symmetric_eigen, SymmetricEigen};
use num_traits::Float;

/// Exponentially weighted information matrix for `P` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationMatrix<T, const P: usize> {
    g: [[T; P]; P],
    lambda: T,
    samples: T,
}

impl<T: Float, const P: usize> InformationMatrix<T, P> {
    /// Creates an empty information matrix with forgetting factor `lambda`.
    ///
    /// # Panics
    /// If `lambda` is not in `(0, 1]`. Use [`Self::try_new`] to avoid panicking.
    pub fn new(lambda: T) -> Self {
        Self::try_new(lambda).expect("forgetting factor must be in (0, 1]")
    }

    /// Creates an empty information matrix, or `None` if `lambda ∉ (0, 1]`.
    pub fn try_new(lambda: T) -> Option<Self> {
        if lambda > T::zero() && lambda <= T::one() {
            Some(Self {
                g: [[T::zero(); P]; P],
                lambda,
                samples: T::zero(),
            })
        } else {
            None
        }
    }

    /// Discards all accumulated information.
    pub fn reset(&mut self) {
        self.g = [[T::zero(); P]; P];
        self.samples = T::zero();
    }

    /// The forgetting factor.
    pub fn lambda(&self) -> T {
        self.lambda
    }

    /// Effective number of samples in the window, `Σ λ^k`.
    ///
    /// Saturates at `1/(1-λ)` for `λ < 1`.
    pub fn effective_samples(&self) -> T {
        self.samples
    }

    /// Adds one regressor with unit weight.
    pub fn update(&mut self, phi: &[T; P]) {
        self.update_weighted(phi, T::one());
    }

    /// Adds one regressor with weight `w ≥ 0` (e.g. the inverse measurement
    /// variance, or zero to only apply forgetting for a missing sample).
    pub fn update_weighted(&mut self, phi: &[T; P], w: T) {
        self.decay();
        for i in 0..P {
            let wi = w * phi[i];
            for j in 0..P {
                self.g[i][j] = self.g[i][j] + wi * phi[j];
            }
        }
    }

    /// Adds one sample of an `M`-output model given its output sensitivities.
    ///
    /// `sensitivities[i]` is `∂y_i/∂θ` for output `i` (row `i` of the Jacobian)
    /// and `inv_noise_variance[i]` is `1/σ_i²`, assuming uncorrelated output
    /// noise. One call counts as one sample regardless of `M`.
    pub fn update_sensitivity<const M: usize>(
        &mut self,
        sensitivities: &[[T; P]; M],
        inv_noise_variance: &[T; M],
    ) {
        self.decay();
        for (s, &r) in sensitivities.iter().zip(inv_noise_variance) {
            for i in 0..P {
                let ri = r * s[i];
                for j in 0..P {
                    self.g[i][j] = self.g[i][j] + ri * s[j];
                }
            }
        }
    }

    fn decay(&mut self) {
        for row in self.g.iter_mut() {
            for x in row.iter_mut() {
                *x = *x * self.lambda;
            }
        }
        self.samples = self.samples * self.lambda + T::one();
    }

    /// The raw accumulated information matrix `G`.
    pub fn matrix(&self) -> &[[T; P]; P] {
        &self.g
    }

    /// `G` divided by the effective sample count: the windowed average of
    /// `w φ φᵀ`. Its eigenvalues are in units of (weighted) regressor power and
    /// do not depend on λ or on how long the monitor has been running, which
    /// makes it the right quantity to compare against fixed thresholds.
    pub fn normalized(&self) -> [[T; P]; P] {
        let mut out = [[T::zero(); P]; P];
        if self.samples > T::zero() {
            for i in 0..P {
                for j in 0..P {
                    out[i][j] = self.g[i][j] / self.samples;
                }
            }
        }
        out
    }

    /// Eigen-decomposition of the normalized information matrix.
    pub fn eigen(&self) -> SymmetricEigen<T, P> {
        symmetric_eigen(&self.normalized())
    }

    /// Log-determinant of the raw information matrix (D-optimality measure).
    ///
    /// `None` while some parameter direction has received no information.
    pub fn log_det(&self) -> Option<T> {
        cholesky_log_det(&self.g)
    }

    /// Cramér–Rao lower bound on the parameter covariance, `σ² G⁻¹`.
    ///
    /// For unit-weight updates pass the equation-error variance as
    /// `noise_variance`; for updates already weighted by `1/σ²` pass one.
    /// Returns `None` if the smallest eigenvalue of the raw `G` is at or below
    /// `min_eigenvalue`, since the bound is then meaningless (unbounded in the
    /// unexcited direction).
    pub fn covariance_bound(&self, noise_variance: T, min_eigenvalue: T) -> Option<[[T; P]; P]> {
        let e = symmetric_eigen(&self.g);
        if P == 0 {
            return Some(self.g);
        }
        if !(e.values[0] > min_eigenvalue) || !(e.values[0] > T::zero()) {
            return None;
        }
        let mut cov = [[T::zero(); P]; P];
        for (value, vector) in e.values.iter().zip(&e.vectors) {
            let s = noise_variance / *value;
            for i in 0..P {
                for j in 0..P {
                    cov[i][j] = cov[i][j] + s * vector[i] * vector[j];
                }
            }
        }
        Some(cov)
    }
}

/// Converts a covariance matrix to a correlation matrix.
///
/// Off-diagonal magnitudes close to one flag parameter pairs that the data
/// cannot tell apart, even if each is individually well excited. Rows and
/// columns with zero variance are left at zero.
pub fn correlation<T: Float, const P: usize>(cov: &[[T; P]; P]) -> [[T; P]; P] {
    let mut out = [[T::zero(); P]; P];
    for i in 0..P {
        for j in 0..P {
            let d = (cov[i][i] * cov[j][j]).sqrt();
            if d > T::zero() {
                out[i][j] = cov[i][j] / d;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prbs::Prbs;

    #[test]
    fn rejects_bad_lambda() {
        assert!(InformationMatrix::<f64, 2>::try_new(0.0).is_none());
        assert!(InformationMatrix::<f64, 2>::try_new(1.1).is_none());
        assert!(InformationMatrix::<f64, 2>::try_new(1.0).is_some());
    }

    #[test]
    fn effective_samples_saturate() {
        let mut g = InformationMatrix::<f64, 1>::new(0.9);
        for _ in 0..1000 {
            g.update(&[1.0]);
        }
        assert!((g.effective_samples() - 10.0).abs() < 1e-9);
        assert!((g.normalized()[0][0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn constant_regressor_is_rank_deficient() {
        let mut g = InformationMatrix::<f64, 2>::new(1.0);
        for _ in 0..100 {
            g.update(&[1.0, 1.0]);
        }
        assert!(g.log_det().is_none());
        assert!(g.covariance_bound(1.0, 1e-9).is_none());
    }

    #[test]
    fn prbs_gives_crb_close_to_sigma2_over_n() {
        // φ = [u_k, u_{k-1}] with a full PRBS period: G ≈ N·I.
        let mut prbs = Prbs::new(10, 1).unwrap();
        let mut g = InformationMatrix::<f64, 2>::new(1.0);
        let mut prev = prbs.next_level(1.0);
        let n = prbs.period_samples();
        for _ in 0..n {
            let u = prbs.next_level(1.0);
            g.update(&[u, prev]);
            prev = u;
        }
        let cov = g.covariance_bound(0.25, 1e-9).unwrap();
        let expected = 0.25 / n as f64;
        assert!((cov[0][0] / expected - 1.0).abs() < 1e-2);
        assert!((cov[1][1] / expected - 1.0).abs() < 1e-2);
        let rho = correlation(&cov);
        assert!(rho[0][1].abs() < 1e-2);
        assert!((rho[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sensitivity_update_matches_regressor_update() {
        let mut a = InformationMatrix::<f64, 3>::new(0.95);
        let mut b = InformationMatrix::<f64, 3>::new(0.95);
        let s = [[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]];
        let r = [2.0, 0.5];
        a.update_sensitivity(&s, &r);
        // Equivalent to two weighted rank-one updates with one decay step.
        b.update_weighted(&s[0], r[0]);
        let before = *b.matrix();
        let mut expected = before;
        for i in 0..3 {
            for j in 0..3 {
                expected[i][j] += r[1] * s[1][i] * s[1][j];
            }
        }
        assert_eq!(a.matrix(), &expected);
        assert_eq!(a.effective_samples(), 1.0);
    }
}
