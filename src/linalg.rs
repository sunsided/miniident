//! Small dense symmetric linear algebra on const-generic arrays.
//!
//! minikalman has Cholesky but no eigen-decomposition, which identifiability analysis needs:
//! the eigenvectors are the parameter combinations, the eigenvalues how well each is known.
//! Matrices are row-major `[[T; N]; N]`; every call costs `O(N³)` and allocates nothing.
//!
//! ## Arithmetic
//!
//! [`MatrixDataType`] has no checked operators, so the operators of `T` decide what happens
//! on overflow: floating-point types go to `±∞` or NaN, the fixed-point types behave as in
//! minikalman's own filters. The functions here never panic on their own; NaN input gives NaN
//! output, not an error.

use minikalman::matrix::{Logarithm, MatrixDataType};

/// Upper bound on Jacobi sweeps. Convergence is quadratic; 5–10 sweeps are typical.
const MAX_SWEEPS: usize = 50;

/// `|x|`. NaN stays NaN.
#[inline]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see module docs)"
)]
pub(crate) fn abs<T: MatrixDataType>(x: T) -> T {
    if x < T::zero() { -x } else { x }
}

/// `√max(x, 0)`, so that a variance driven slightly negative by rounding gives zero. NaN
/// gives zero.
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

/// A Jacobi rotation by the angle `φ` in the plane of the coordinates `p < q`.
#[derive(Debug, Clone, Copy)]
struct Rotation<T> {
    /// First coordinate of the plane; below `q`.
    p: usize,
    /// Second coordinate of the plane; below `N`.
    q: usize,
    /// `cos φ`.
    c: T,
    /// `sin φ`.
    s: T,
}

/// Computes all eigenvalues and eigenvectors of a symmetric matrix with cyclic Jacobi
/// rotations.
///
/// Symmetry is assumed and not checked. Jacobi is short and accurate for the small
/// eigenvalues of positive semi-definite matrices, which is what identifiability is about.
/// The rotation angle is computed without squaring large quotients, and an off-diagonal
/// entry is dropped once it no longer changes either diagonal entry it couples, so neither
/// test depends on a machine epsilon. At most [`MAX_SWEEPS`] sweeps run, also for NaN input.
#[must_use]
pub fn symmetric_eigen<T: MatrixDataType, const N: usize>(m: &[[T; N]; N]) -> SymmetricEigen<T, N> {
    let mut a = *m;
    let mut v = identity();

    for _ in 0..MAX_SWEEPS {
        if !sweep(&mut a, &mut v) {
            break;
        }
    }

    sorted_eigen(&a, &v)
}

/// The `N` × `N` identity matrix.
fn identity<T: MatrixDataType, const N: usize>() -> [[T; N]; N] {
    core::array::from_fn(|i| core::array::from_fn(|j| if i == j { T::one() } else { T::zero() }))
}

/// All index pairs `(p, q)` with `p < q < N`, in the cyclic Jacobi order.
fn upper_pairs<const N: usize>() -> impl Iterator<Item = (usize, usize)> {
    (0..N).flat_map(|p| (0..N).filter(move |&q| q > p).map(move |q| (p, q)))
}

/// Does one cyclic sweep over all off-diagonal pairs of `a` and accumulates the rotations
/// in `v`. Returns `true` if at least one rotation was applied.
#[expect(
    clippy::indexing_slicing,
    reason = "p < q < N: upper_pairs yields only such pairs, and a is N × N"
)]
fn sweep<T: MatrixDataType, const N: usize>(a: &mut [[T; N]; N], v: &mut [[T; N]; N]) -> bool {
    let mut rotated = false;

    for (p, q) in upper_pairs::<N>() {
        let Some((c, s)) = rotation(a[p][p], a[q][q], a[p][q]) else {
            a[p][q] = T::zero();
            a[q][p] = T::zero();
            continue;
        };

        let rotation = Rotation { p, q, c, s };
        rotate_columns(a, rotation);
        rotate_rows(a, rotation);
        a[p][q] = T::zero();
        a[q][p] = T::zero();
        rotate_columns(v, rotation);
        rotated = true;
    }

    rotated
}

/// Computes `(cos φ, sin φ)` of the rotation that annihilates `apq`, or `None` when `apq` is
/// zero or too small to change `app` or `aqq`.
///
/// `tan φ` follows Numerical Recipes §11.1, rewritten for `|θ| > 1` so that `θ` is never
/// squared.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see module docs)"
)]
fn rotation<T: MatrixDataType>(app: T, aqq: T, apq: T) -> Option<(T, T)> {
    let zero = T::zero();
    let one = T::one();
    let g = abs(apq);

    if apq == zero || (abs(app) + g == abs(app) && abs(aqq) + g == abs(aqq)) {
        return None;
    }

    let theta = (aqq - app) / ((one + one) * apq);
    let t = if abs(theta) > one {
        let r = one / theta;
        r / (one + (one + r * r).square_root())
    } else {
        let sign = if theta < zero { -one } else { one };
        sign / (abs(theta) + (theta * theta + one).square_root())
    };

    if t == zero {
        return None;
    }

    let c = one / (t * t + one).square_root();

    Some((c, t * c))
}

/// Applies `rotation` to the columns `p` and `q` of `m`: `m ← m·J`.
#[expect(
    clippy::indexing_slicing,
    reason = "p < q < N by the Rotation invariant, and each row has N entries"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see module docs)"
)]
fn rotate_columns<T: MatrixDataType, const N: usize>(m: &mut [[T; N]; N], rotation: Rotation<T>) {
    let Rotation { p, q, c, s } = rotation;

    for row in m {
        let (x, y) = (row[p], row[q]);
        row[p] = c * x - s * y;
        row[q] = s * x + c * y;
    }
}

/// Applies `rotation` to the rows `p` and `q` of `m`: `m ← Jᵀ·m`.
#[expect(
    clippy::indexing_slicing,
    reason = "p < q < N by the Rotation invariant, so row p is in head and row q starts tail"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see module docs)"
)]
fn rotate_rows<T: MatrixDataType, const N: usize>(m: &mut [[T; N]; N], rotation: Rotation<T>) {
    let Rotation { p, q, c, s } = rotation;
    let (head, tail) = m.split_at_mut(q);

    for (x, y) in head[p].iter_mut().zip(tail[0].iter_mut()) {
        let (xp, yq) = (*x, *y);
        *x = c * xp - s * yq;
        *y = s * xp + c * yq;
    }
}

/// Reads the eigenvalues from the diagonal of the converged `a` and the eigenvectors from the
/// columns of `v`, sorted by ascending eigenvalue.
///
/// Selection sort with `<`: it never panics, also when NaN makes the order partial. NaN
/// entries then stay where they are.
#[expect(
    clippy::indexing_slicing,
    reason = "i, j and best come from 0..N, and every array is N or N × N"
)]
fn sorted_eigen<T: MatrixDataType, const N: usize>(
    a: &[[T; N]; N],
    v: &[[T; N]; N],
) -> SymmetricEigen<T, N> {
    let mut values: [T; N] = core::array::from_fn(|i| a[i][i]);
    let mut vectors: [[T; N]; N] = core::array::from_fn(|i| core::array::from_fn(|k| v[k][i]));

    for i in 0..N {
        let mut best = i;
        for j in i..N {
            if values[j] < values[best] {
                best = j;
            }
        }

        values.swap(i, best);
        vectors.swap(i, best);
    }

    SymmetricEigen { values, vectors }
}

/// Natural log-determinant of a symmetric positive-definite matrix via Cholesky.
///
/// Returns `None` if the matrix is not numerically positive definite, or if a pivot is NaN.
#[must_use]
#[expect(
    clippy::indexing_slicing,
    reason = "i, j and k come from 0..N, and every array is N × N"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see module docs)"
)]
pub fn cholesky_log_det<T: Logarithm, const N: usize>(m: &[[T; N]; N]) -> Option<T> {
    let mut l = [[T::zero(); N]; N];
    let mut half = T::zero();

    for j in 0..N {
        let d = m[j][j] - dot(&l[j], &l[j], j);
        if !is_positive(d) {
            return None;
        }

        let ljj = d.square_root();
        l[j][j] = ljj;
        half += ljj.logarithm();
        for i in (0..N).filter(|&i| i > j) {
            l[i][j] = (m[i][j] - dot(&l[i], &l[j], j)) / ljj;
        }
    }

    Some(half + half)
}

/// `Σ_{k < len} a[k]·b[k]`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see module docs)"
)]
fn dot<T: MatrixDataType, const N: usize>(a: &[T; N], b: &[T; N], len: usize) -> T {
    a.iter()
        .zip(b)
        .take(len)
        .fold(T::zero(), |sum, (&x, &y)| sum + x * y)
}

/// `x > 0`, and `false` for NaN.
fn is_positive<T: MatrixDataType>(x: T) -> bool {
    matches!(
        x.partial_cmp(&T::zero()),
        Some(core::cmp::Ordering::Greater)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `m·x` for an `N` × `N` matrix.
    fn matvec<const N: usize>(m: &[[f64; N]; N], x: &[f64; N]) -> [f64; N] {
        core::array::from_fn(|i| m[i].iter().zip(x).map(|(a, b)| a * b).sum())
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
                assert!(
                    (av[k] - e.values[i] * e.vectors[i][k]).abs() < 1e-10,
                    "A·v = λ·v for pair {i}"
                );
            }

            for j in 0..4 {
                let dot: f64 = (0..4).map(|k| e.vectors[i][k] * e.vectors[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-10, "v{i}·v{j} = δ");
            }
        }

        assert!(
            e.values.windows(2).all(|w| w[0] <= w[1]),
            "eigenvalues ascend"
        );
        assert!(
            (e.values.iter().sum::<f64>() - 8.0).abs() < 1e-10,
            "Σλ equals the trace"
        );
    }

    #[test]
    fn eigen_of_rank_one_and_extreme_scaling() {
        let e = symmetric_eigen(&[[1.0f64, 1.0], [1.0, 1.0]]);
        assert!(e.values[0].abs() < 1e-12, "rank one: λ₀ = 0");
        assert!((e.values[1] - 2.0).abs() < 1e-12, "rank one: λ₁ = 2");
        assert!(
            (e.vectors[0][0] + e.vectors[0][1]).abs() < 1e-12,
            "the null vector is (1, −1)/√2"
        );

        // Covariances span many decades; θ is far outside the range where θ² would be safe
        // in f32.
        let e = symmetric_eigen(&[[1.0e12f32, 1.0], [1.0, 1.0e-6]]);
        assert!(
            (e.values[1] / 1.0e12 - 1.0).abs() < 1e-6,
            "large eigenvalue is kept"
        );
        assert!(
            e.values[0] <= 1.0e-6 && e.values[0] > 0.9e-6,
            "small eigenvalue is kept"
        );
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
    fn eigen_of_nan_terminates() {
        let e = symmetric_eigen(&[[f64::NAN, 1.0], [1.0, 2.0]]);
        assert!(
            e.values.iter().any(|v| v.is_nan()),
            "NaN input gives NaN output"
        );
    }

    #[test]
    fn log_det_matches() {
        let m = [[4.0, 2.0], [2.0, 3.0]];
        assert!(
            (cholesky_log_det(&m).unwrap() - 8.0f64.ln()).abs() < 1e-12,
            "ln det = ln 8"
        );
        assert!(
            cholesky_log_det(&[[1.0, 1.0], [1.0, 1.0]]).is_none(),
            "singular"
        );
        assert!(
            cholesky_log_det(&[[f64::NAN, 0.0], [0.0, 1.0]]).is_none(),
            "NaN pivot"
        );
    }
}
