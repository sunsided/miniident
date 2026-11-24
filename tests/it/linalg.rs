//! Properties of the Jacobi eigen-decomposition and the Cholesky log-determinant.

use crate::strategies::{MAGNITUDE, N, positive_definite, symmetric};
use miniident::linalg::{cholesky_log_det, symmetric_eigen};
use proptest::prelude::*;

/// Absolute tolerance for eigen identities on matrices with entries up to [`MAGNITUDE`].
///
/// Jacobi is backward stable: the residual is a small multiple of machine epsilon times the
/// matrix norm, and the norm is at most `N·MAGNITUDE`.
const EIGEN_TOLERANCE: f64 = 1e-9 * MAGNITUDE;

proptest! {
    /// Every eigenpair satisfies `A·v = λ·v`.
    #[test]
    fn eigenpairs_satisfy_definition(m in symmetric()) {
        let e = symmetric_eigen(&m);

        for (value, vector) in e.values.iter().zip(&e.vectors) {
            for (row, &vk) in m.iter().zip(vector) {
                let av: f64 = row.iter().zip(vector).map(|(a, v)| a * v).sum();
                prop_assert!((av - value * vk).abs() < EIGEN_TOLERANCE, "A·v = {av}, λ·v = {}", value * vk);
            }
        }
    }

    /// The eigenvectors are orthonormal.
    #[test]
    fn eigenvectors_are_orthonormal(m in symmetric()) {
        let e = symmetric_eigen(&m);

        for (i, a) in e.vectors.iter().enumerate() {
            for (j, b) in e.vectors.iter().enumerate() {
                let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!((dot - expected).abs() < 1e-9, "v{i}·v{j} = {dot}");
            }
        }
    }

    /// The eigenvalues are ascending and sum to the trace.
    #[test]
    fn eigenvalues_are_sorted_and_keep_the_trace(m in symmetric()) {
        let e = symmetric_eigen(&m);
        let trace: f64 = (0..N).map(|i| m[i][i]).sum();
        let sum: f64 = e.values.iter().sum();

        prop_assert!(e.values.windows(2).all(|w| w[0] <= w[1]), "{:?}", e.values);
        prop_assert!((trace - sum).abs() < EIGEN_TOLERANCE, "trace {trace}, Σλ {sum}");
    }

    /// For a positive-definite matrix, the Cholesky log-determinant equals `Σ ln λ`.
    #[test]
    fn log_det_matches_eigenvalues(m in positive_definite()) {
        let e = symmetric_eigen(&m);
        let expected: f64 = e.values.iter().map(|v| v.ln()).sum();
        let log_det = cholesky_log_det(&m);

        prop_assert!(log_det.is_some(), "B·Bᵀ + I is positive definite");
        let log_det = log_det.unwrap_or(f64::NAN);
        prop_assert!((log_det - expected).abs() < 1e-8 * expected.abs().max(1.0), "{log_det} vs {expected}");
    }
}
