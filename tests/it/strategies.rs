//! Shared proptest strategies: symmetric and positive-definite matrices.

use proptest::prelude::*;

/// Dimension of the matrices the properties use.
pub const N: usize = 4;

/// Largest absolute value of a generated matrix entry.
pub const MAGNITUDE: f64 = 100.0;

/// A symmetric `N` × `N` matrix with entries in `[-MAGNITUDE, MAGNITUDE]`.
///
/// The strategy draws a full matrix and keeps its upper triangle, so the result is exactly
/// symmetric.
pub fn symmetric() -> impl Strategy<Value = [[f64; N]; N]> {
    any_square(MAGNITUDE).prop_map(|m| {
        let mut out = m;

        for (i, row) in out.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                if j < i {
                    *x = m[j][i];
                }
            }
        }

        out
    })
}

/// A symmetric positive-definite `N` × `N` matrix `B·Bᵀ + I`.
///
/// The identity term keeps the smallest eigenvalue at 1 or more, so the logarithm of every
/// eigenvalue is finite and well conditioned.
pub fn positive_definite() -> impl Strategy<Value = [[f64; N]; N]> {
    any_square(10.0).prop_map(|b| {
        let mut out = [[0.0; N]; N];

        for (i, row) in out.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = b[i].iter().zip(&b[j]).map(|(p, q)| p * q).sum();
                if i == j {
                    *x += 1.0;
                }
            }
        }

        out
    })
}

/// A tolerance vector with entries in `[0.01, 10]`.
pub fn tolerances() -> impl Strategy<Value = [f64; N]> {
    proptest::array::uniform4(0.01..10.0)
}

/// An `N` × `N` matrix with entries in `[-magnitude, magnitude]`, not symmetric.
fn any_square(magnitude: f64) -> impl Strategy<Value = [[f64; N]; N]> {
    proptest::array::uniform4(proptest::array::uniform4(-magnitude..=magnitude))
}
