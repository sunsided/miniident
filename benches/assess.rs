//! Benchmarks of the `O(N³)` assessment paths and the `O(N²)` per-sample update.
//!
//! `assess` runs at a lower rate than the estimator update, but on a small MCU its cost
//! decides that rate. The benchmarks cover the parameter counts of typical grey-box models.

use criterion::{BenchmarkId, Criterion};
use miniident::identifiability::Identifiability;
use miniident::information::InformationMatrix;
use miniident::linalg::symmetric_eigen;
use std::hint::black_box;

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_eigen::<2>(&mut criterion);
    bench_eigen::<4>(&mut criterion);
    bench_eigen::<8>(&mut criterion);
    bench_assess::<2>(&mut criterion);
    bench_assess::<4>(&mut criterion);
    bench_assess::<8>(&mut criterion);
    bench_update::<4>(&mut criterion);
    bench_update::<8>(&mut criterion);
    criterion.final_summary();
}

/// A dense, symmetric positive-definite test covariance: `1` on the diagonal, `ρ = 0.5`
/// between every pair, scaled by `0.01`.
fn covariance<const N: usize>() -> [[f64; N]; N] {
    core::array::from_fn(|i| core::array::from_fn(|j| if i == j { 0.01 } else { 0.005 }))
}

/// Benchmarks the Jacobi eigen-decomposition of an `N` × `N` covariance.
fn bench_eigen<const N: usize>(criterion: &mut Criterion) {
    let p = covariance::<N>();

    criterion.bench_with_input(BenchmarkId::new("symmetric_eigen", N), &p, |b, p| {
        b.iter(|| symmetric_eigen(black_box(p)));
    });
}

/// Benchmarks one identifiability verdict on an `N` × `N` covariance.
fn bench_assess<const N: usize>(criterion: &mut Criterion) {
    let p = covariance::<N>();
    let criteria = Identifiability::new([0.1; N], 0.95);

    criterion.bench_with_input(BenchmarkId::new("assess_array", N), &p, |b, p| {
        b.iter(|| criteria.assess_array(black_box(p)));
    });
}

/// Benchmarks one information-matrix update with an `N`-element regressor.
fn bench_update<const N: usize>(criterion: &mut Criterion) {
    let phi: [f64; N] = core::array::from_fn(|i| if i % 2 == 0 { 1.0 } else { -0.5 });
    let mut info = InformationMatrix::<f64, N>::new(0.98);

    criterion.bench_with_input(BenchmarkId::new("information_update", N), &phi, |b, phi| {
        b.iter(|| info.update(black_box(phi)));
    });
}
