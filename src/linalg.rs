//! Small dense linear algebra on const-generic arrays.
//!
//! Matrices are row-major `[[T; P]; P]`. Everything here targets the small
//! parameter counts typical of online identification (P ≲ 20); each call is
//! O(P³) and allocation-free.

use num_traits::Float;

/// Upper bound on Jacobi sweeps. Convergence is quadratic; 5–10 sweeps are
/// typical even for badly scaled matrices.
const MAX_SWEEPS: usize = 50;

/// Eigen-decomposition of a symmetric matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricEigen<T, const P: usize> {
    /// Eigenvalues in ascending order.
    pub values: [T; P],
    /// Unit eigenvectors; `vectors[i]` belongs to `values[i]`.
    pub vectors: [[T; P]; P],
}

/// Computes all eigenvalues and eigenvectors of a symmetric matrix using cyclic
/// Jacobi rotations.
///
/// Symmetry is assumed, not checked. Jacobi is chosen over QR-based methods
/// because it is short, branch-light, and accurate for small eigenvalues of
/// positive semi-definite matrices, which is exactly what excitation
/// monitoring cares about.
pub fn symmetric_eigen<T: Float, const P: usize>(m: &[[T; P]; P]) -> SymmetricEigen<T, P> {
    let mut a = *m;
    let mut v = [[T::zero(); P]; P];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = T::one();
    }

    let two = T::one() + T::one();
    let tol = T::epsilon() * frobenius_norm(&a);
    let theta_max = T::one() / T::epsilon().sqrt();

    for _ in 0..MAX_SWEEPS {
        let mut off = T::zero();
        for p in 0..P {
            for q in (p + 1)..P {
                off = off + a[p][q] * a[p][q];
            }
        }
        if !(off.sqrt() > tol) {
            break;
        }

        for p in 0..P {
            for q in (p + 1)..P {
                let apq = a[p][q];
                if apq == T::zero() {
                    continue;
                }
                // Rotation angle that annihilates a[p][q] (Numerical Recipes §11.1).
                let theta = (a[q][q] - a[p][p]) / (two * apq);
                let t = if theta.abs() > theta_max {
                    T::one() / (two * theta)
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + T::one()).sqrt())
                };
                let c = T::one() / (t * t + T::one()).sqrt();
                let s = t * c;

                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                for k in 0..P {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                a[p][q] = T::zero();
                a[q][p] = T::zero();

                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }

    // Sort ascending (insertion sort on indices; P is small).
    let mut order = [0usize; P];
    for (i, o) in order.iter_mut().enumerate() {
        *o = i;
    }
    for i in 1..P {
        let mut j = i;
        while j > 0 && a[order[j - 1]][order[j - 1]] > a[order[j]][order[j]] {
            order.swap(j - 1, j);
            j -= 1;
        }
    }

    let mut values = [T::zero(); P];
    let mut vectors = [[T::zero(); P]; P];
    for (dst, &src) in order.iter().enumerate() {
        values[dst] = a[src][src];
        for k in 0..P {
            vectors[dst][k] = v[k][src];
        }
    }
    SymmetricEigen { values, vectors }
}

/// Natural log-determinant of a symmetric positive-definite matrix via Cholesky.
///
/// Returns `None` if the matrix is not numerically positive definite.
pub fn cholesky_log_det<T: Float, const P: usize>(m: &[[T; P]; P]) -> Option<T> {
    let mut l = [[T::zero(); P]; P];
    let mut log_det = T::zero();
    for j in 0..P {
        let mut d = m[j][j];
        for k in 0..j {
            d = d - l[j][k] * l[j][k];
        }
        if !(d > T::zero()) {
            return None;
        }
        let ljj = d.sqrt();
        l[j][j] = ljj;
        log_det = log_det + ljj.ln();
        for i in (j + 1)..P {
            let mut s = m[i][j];
            for k in 0..j {
                s = s - l[i][k] * l[j][k];
            }
            l[i][j] = s / ljj;
        }
    }
    Some(log_det + log_det)
}

/// Frobenius norm of a square matrix.
pub fn frobenius_norm<T: Float, const P: usize>(m: &[[T; P]; P]) -> T {
    let mut s = T::zero();
    for row in m {
        for &x in row {
            s = s + x * x;
        }
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matvec<const P: usize>(m: &[[f64; P]; P], x: &[f64; P]) -> [f64; P] {
        let mut y = [0.0; P];
        for i in 0..P {
            for j in 0..P {
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
        let trace: f64 = e.values.iter().sum();
        assert!((trace - 8.0).abs() < 1e-10);
    }

    #[test]
    fn eigen_of_rank_one() {
        let e = symmetric_eigen(&[[1.0, 1.0], [1.0, 1.0]]);
        assert!(e.values[0].abs() < 1e-12);
        assert!((e.values[1] - 2.0).abs() < 1e-12);
        // Null direction is (1, -1)/√2 up to sign.
        assert!((e.vectors[0][0] + e.vectors[0][1]).abs() < 1e-12);
    }

    #[test]
    fn eigen_of_zero_and_diagonal() {
        let z = symmetric_eigen(&[[0.0f32; 3]; 3]);
        assert_eq!(z.values, [0.0; 3]);
        let d = symmetric_eigen(&[[3.0, 0.0], [0.0, 1.0]]);
        assert_eq!(d.values, [1.0, 3.0]);
    }

    #[test]
    fn log_det_matches() {
        let m = [[4.0, 2.0], [2.0, 3.0]];
        assert!((cholesky_log_det(&m).unwrap() - 8.0f64.ln()).abs() < 1e-12);
        assert!(cholesky_log_det(&[[1.0, 1.0], [1.0, 1.0]]).is_none());
    }
}
