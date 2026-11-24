//! Fuzzes the Jacobi eigen-decomposition with arbitrary symmetric matrices.
//!
//! It must never panic, and it must terminate within its sweep limit for every input. For
//! finite input of moderate size, the eigenvalues must come out ascending.

#![no_main]

use libfuzzer_sys::fuzz_target;
use miniident::linalg::symmetric_eigen;

/// Dimension of the fuzzed matrices.
const N: usize = 4;

/// Largest entry for which the target checks the order of the eigenvalues. Above it,
/// rotations can overflow to infinity, and the order of non-finite values is not defined.
const ORDER_CHECK_LIMIT: f64 = 1e100;

fuzz_target!(|upper: [[f64; N]; N]| {
    let m: [[f64; N]; N] =
        core::array::from_fn(|i| core::array::from_fn(|j| upper[i.min(j)][i.max(j)]));
    let e = symmetric_eigen(&m);

    let moderate = m
        .iter()
        .flatten()
        .all(|x| x.is_finite() && x.abs() < ORDER_CHECK_LIMIT);
    if moderate && e.values.iter().all(|v| v.is_finite()) {
        assert!(
            e.values.windows(2).all(|w| w[0] <= w[1]),
            "eigenvalues ascend: {:?}",
            e.values
        );
    }
});
