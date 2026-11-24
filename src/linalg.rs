//! Small dense symmetric linear algebra on const-generic arrays.
//!
//! minikalman has Cholesky but no eigen-decomposition, which identifiability analysis needs:
//! the eigenvectors are the parameter combinations, the eigenvalues how well each is known.
//! Matrices are row-major `[[T; N]; N]`; every call costs `O(N³)` and allocates nothing.

use minikalman::matrix::{Logarithm, MatrixDataType};

/// Upper bound on Jacobi sweeps. Convergence is quadratic; 5–10 sweeps are typical.
const MAX_SWEEPS: usize = 50;

/// `|x|`.
#[inline]
pub(crate) fn abs<T: MatrixDataType>(x: T) -> T {
    if x < T::zero() { -x } else { x }
}

/// `√max(x, 0)`, so that a variance driven slightly negative by rounding gives zero.
#[inline]
pub(crate) fn sqrt_clamped<T: MatrixDataType>(x: T) -> T {
    if x > T::zero() {
        x.square_root()
    } else {
        T::zero()
    }
}

/// Eigen-decomposition of a symmetric matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricEigen<T, const N: usize> {
    /// Eigenvalues in ascending order.
    pub values: [T; N],
    /// Unit eigenvectors; `vectors[i]` belongs to `values[i]`. Signs are arbitrary.
    pub vectors: [[T; N]; N],
}

/// Computes all eigenvalues and eigenvectors of a symmetric matrix with cyclic Jacobi
/// rotations.
///
/// Symmetry is assumed and not checked. Jacobi is short and accurate for the small
/// eigenvalues of positive semi-definite matrices, which is what identifiability is about.
/// The rotation angle is computed without squaring large quotients, and an off-diagonal
/// entry is dropped once it no longer changes either diagonal entry it couples, so neither
/// test depends on a machine epsilon.
pub fn symmetric_eigen<T: MatrixDataType, const N: usize>(m: &[[T; N]; N]) -> SymmetricEigen<T, N> {
    let zero = T::zero();
    let one = T::one();
    let two = one + one;

    let mut a = *m;
    let mut v = [[zero; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = one;
    }

    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..N {
            for q in (p + 1)..N {
                let apq = a[p][q];
                if apq == zero {
                    continue;
                }
                let (app, aqq) = (a[p][p], a[q][q]);
                let g = abs(apq);
                if abs(app) + g == abs(app) && abs(aqq) + g == abs(aqq) {
                    a[p][q] = zero;
                    a[q][p] = zero;
                    continue;
                }

                // tan φ of the rotation that annihilates a[p][q] (Numerical Recipes §11.1),
                // rewritten for |θ| > 1 so that θ is never squared.
                let theta = (aqq - app) / (two * apq);
                let t = if abs(theta) > one {
                    let r = one / theta;
                    r / (one + (one + r * r).square_root())
                } else {
                    let sign = if theta < zero { -one } else { one };
                    sign / (abs(theta) + (theta * theta + one).square_root())
                };
                if t == zero {
                    a[p][q] = zero;
                    a[q][p] = zero;
                    continue;
                }
                let c = one / (t * t + one).square_root();
                let s = t * c;

                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                for k in 0..N {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                a[p][q] = zero;
                a[q][p] = zero;

                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }

    // Sort ascending (insertion sort on indices; N is small).
    let mut order = [0usize; N];
    for (i, o) in order.iter_mut().enumerate() {
        *o = i;
    }
    for i in 1..N {
        let mut j = i;
        while j > 0 && a[order[j - 1]][order[j - 1]] > a[order[j]][order[j]] {
            order.swap(j - 1, j);
            j -= 1;
        }
    }

    let mut values = [zero; N];
    let mut vectors = [[zero; N]; N];
    for (dst, &src) in order.iter().enumerate() {
        values[dst] = a[src][src];
        for k in 0..N {
            vectors[dst][k] = v[k][src];
        }
    }
    SymmetricEigen { values, vectors }
}

/// Natural log-determinant of a symmetric positive-definite matrix via Cholesky.
///
/// Returns `None` if the matrix is not numerically positive definite.
pub fn cholesky_log_det<T: Logarithm, const N: usize>(m: &[[T; N]; N]) -> Option<T> {
    let mut l = [[T::zero(); N]; N];
    let mut half = T::zero();
    for j in 0..N {
        let mut d = m[j][j];
        for k in 0..j {
            d = d - l[j][k] * l[j][k];
        }
        if !(d > T::zero()) {
            return None;
        }
        let ljj = d.square_root();
        l[j][j] = ljj;
        half = half + ljj.logarithm();
        for i in (j + 1)..N {
            let mut s = m[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k];
            }
            l[i][j] = s / ljj;
        }
    }
    Some(half + half)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec<const N: usize>(m: &[[f64; N]; N], x: &[f64; N]) -> [f64; N] {
        let mut y = [0.0; N];
        for i in 0..N {
            for j in 0..N {
                y[i] += m[i][j] * x[j];
            }
        }
        y
    }

    #[test]
    fn eigen_reconstructs() {
        let m = [
            [4.0, 1.0, -2.0, 2.0],
            [1.0, 2.0, 0.0, 1.0],
            [-2.0, 0.0, 3.0, -2.0],
            [2.0, 1.0, -2.0, -1.0],
        ];
        let e = symmetric_eigen(&m);
        for i in 0..4 {
            let av = matvec(&m, &e.vectors[i]);
            for k in 0..4 {
                assert!((av[k] - e.values[i] * e.vectors[i][k]).abs() < 1e-10);
            }
            for j in 0..4 {
                let dot: f64 = (0..4).map(|k| e.vectors[i][k] * e.vectors[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-10);
            }
        }
        assert!(e.values.windows(2).all(|w| w[0] <= w[1]));
        assert!((e.values.iter().sum::<f64>() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn eigen_of_rank_one_and_extreme_scaling() {
        let e = symmetric_eigen(&[[1.0f64, 1.0], [1.0, 1.0]]);
        assert!(e.values[0].abs() < 1e-12);
        assert!((e.values[1] - 2.0).abs() < 1e-12);
        assert!((e.vectors[0][0] + e.vectors[0][1]).abs() < 1e-12);

        // Covariances span many decades; θ is far outside the range where θ² would be safe
        // in f32.
        let e = symmetric_eigen(&[[1.0e12f32, 1.0], [1.0, 1.0e-6]]);
        assert!((e.values[1] / 1.0e12 - 1.0).abs() < 1e-6);
        assert!(e.values[0] <= 1.0e-6 && e.values[0] > 0.9e-6);
    }

    #[test]
    fn eigen_of_zero_and_diagonal() {
        assert_eq!(symmetric_eigen(&[[0.0f32; 3]; 3]).values, [0.0; 3]);
        assert_eq!(
            symmetric_eigen(&[[3.0, 0.0], [0.0, 1.0]]).values,
            [1.0, 3.0]
        );
    }

    #[test]
    fn log_det_matches() {
        let m = [[4.0, 2.0], [2.0, 3.0]];
        assert!((cholesky_log_det(&m).unwrap() - 8.0f64.ln()).abs() < 1e-12);
        assert!(cholesky_log_det(&[[1.0, 1.0], [1.0, 1.0]]).is_none());
    }
}
