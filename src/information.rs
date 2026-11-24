//! Exponentially weighted information matrix of the data alone.
//!
//! ```text
//! G_k = μ·G_{k-1} + w_k·φ_k·φ_kᵀ                (regressors)
//! G_k = μ·G_{k-1} + Σᵢ S_{k,i}ᵀ·S_{k,i} / σᵢ²   (output sensitivities of a grey-box model)
//! ```
//!
//! This is the data side of the RLS information `P⁻¹`, without the prior, the windup guard
//! and the gate. It is what [`ExcitationMonitor`](crate::excitation::ExcitationMonitor) judges
//! the input by, and what experiment design will compare candidate inputs by. To judge the
//! *estimate*, use [`Identifiability`](crate::identifiability::Identifiability) on the
//! estimator's covariance instead.

use crate::linalg::{SymmetricEigen, cholesky_log_det, symmetric_eigen};
use minikalman::matrix::{Logarithm, MatrixDataType};

/// Exponentially weighted information matrix for `N` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationMatrix<T, const N: usize> {
    g: [[T; N]; N],
    mu: T,
    samples: T,
}

impl<T: MatrixDataType, const N: usize> InformationMatrix<T, N> {
    /// Creates an empty information matrix with forgetting factor `mu`, as in
    /// [`RecursiveLeastSquares::set_forgetting_factor`](minikalman::rls::RecursiveLeastSquares::set_forgetting_factor).
    ///
    /// # Panics
    /// If `mu ∉ (0, 1]`. Use [`Self::try_new`] to avoid panicking.
    pub fn new(mu: T) -> Self {
        Self::try_new(mu).expect("forgetting factor must be in (0, 1]")
    }

    /// Creates an empty information matrix, or `None` if `mu ∉ (0, 1]`.
    pub fn try_new(mu: T) -> Option<Self> {
        (mu > T::zero() && mu <= T::one()).then(|| Self {
            g: [[T::zero(); N]; N],
            mu,
            samples: T::zero(),
        })
    }

    /// Discards all accumulated information.
    pub fn reset(&mut self) {
        self.g = [[T::zero(); N]; N];
        self.samples = T::zero();
    }

    /// The forgetting factor `μ`.
    pub fn forgetting_factor(&self) -> T {
        self.mu
    }

    /// Effective number of samples in the window, `Σ μᵏ`; saturates at `1/(1 − μ)`.
    pub fn effective_samples(&self) -> T {
        self.samples
    }

    /// Adds one regressor with unit weight. `O(N²)`.
    pub fn update(&mut self, phi: &[T; N]) {
        self.update_weighted(phi, T::one());
    }

    /// Adds one regressor with weight `w ≥ 0`, e.g. `1/r`. A weight of zero only forgets,
    /// for a sample that was missing or gated.
    pub fn update_weighted(&mut self, phi: &[T; N], w: T) {
        self.forget();
        for i in 0..N {
            let wi = w * phi[i];
            for j in 0..N {
                self.g[i][j] += wi * phi[j];
            }
        }
    }

    /// Adds one sample of an `M`-output grey-box model from its output sensitivities.
    ///
    /// `sensitivities[i]` is `∂yᵢ/∂θ`, row `i` of the Jacobian, and `inv_noise_variance[i]`
    /// is `1/σᵢ²` for uncorrelated output noise. One call counts as one sample. `O(M·N²)`.
    pub fn update_sensitivity<const M: usize>(
        &mut self,
        sensitivities: &[[T; N]; M],
        inv_noise_variance: &[T; M],
    ) {
        self.forget();
        for (s, &r) in sensitivities.iter().zip(inv_noise_variance) {
            for i in 0..N {
                let ri = r * s[i];
                for j in 0..N {
                    self.g[i][j] += ri * s[j];
                }
            }
        }
    }

    fn forget(&mut self) {
        if self.mu < T::one() {
            for row in self.g.iter_mut() {
                for x in row.iter_mut() {
                    *x = *x * self.mu;
                }
            }
        }
        self.samples = self.samples * self.mu + T::one();
    }

    /// The raw information matrix `G`.
    pub fn matrix(&self) -> &[[T; N]; N] {
        &self.g
    }

    /// `G` divided by the effective sample count: the windowed average of `w·φ·φᵀ`.
    ///
    /// Its eigenvalues are in units of (weighted) regressor power and depend neither on `μ`
    /// nor on how long the window has been filling, so fixed thresholds apply.
    pub fn normalized(&self) -> [[T; N]; N] {
        let mut out = [[T::zero(); N]; N];
        if self.samples > T::zero() {
            for i in 0..N {
                for j in 0..N {
                    out[i][j] = self.g[i][j] / self.samples;
                }
            }
        }
        out
    }

    /// Eigen-decomposition of the normalized information matrix. `O(N³)`.
    pub fn eigen(&self) -> SymmetricEigen<T, N> {
        symmetric_eigen(&self.normalized())
    }

    /// Log-determinant of the raw `G`, the D-optimality measure; `None` while some parameter
    /// direction has received no information.
    pub fn log_det(&self) -> Option<T>
    where
        T: Logarithm,
    {
        cholesky_log_det(&self.g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_forgetting() {
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
    fn constant_regressor_has_no_log_det() {
        let mut g = InformationMatrix::<f64, 2>::new(1.0);
        for _ in 0..100 {
            g.update(&[1.0, 1.0]);
        }
        assert!(g.log_det().is_none());
    }

    #[test]
    fn sensitivity_update_is_weighted_sum() {
        let mut a = InformationMatrix::<f64, 3>::new(0.95);
        let s = [[1.0, -2.0, 0.5], [0.0, 1.0, 3.0]];
        let r = [2.0, 0.5];
        a.update_sensitivity(&s, &r);
        for i in 0..3 {
            for j in 0..3 {
                let expected = r[0] * s[0][i] * s[0][j] + r[1] * s[1][i] * s[1][j];
                assert_eq!(a.matrix()[i][j], expected);
            }
        }
        assert_eq!(a.effective_samples(), 1.0);
    }
}
