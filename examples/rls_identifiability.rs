//! Identifiability of a minikalman RLS estimate through excitation, rest and re-excitation.
//!
//! The plant is the first-order motor of minikalman's `rls-adaptive` example,
//! `yₖ = a·yₖ₋₁ + b·uₖ₋₁ + wₖ`, identified as `θ = [a, b]` from `φ = [yₖ₋₁, uₖ₋₁]` with
//! forgetting `μ = 0.98`. The run has three phases:
//!
//! 1. **PRBS** around the operating point: the estimate converges and becomes identified.
//! 2. **Rest**: the input holds the operating point. The regressor is constant, so only the
//!    steady-state relation `y = a·y + b·u` is still observed. Forgetting erodes the rest of
//!    the information at the rate `μ`. The correlation between `â` and `b̂` rises first
//!    (`Correlated`), then the tolerances fail (`Unidentified`) along the combination the
//!    steady state cannot see, and the excitation monitor, which forgets at the same rate,
//!    follows once the remaining information drops below its thresholds.
//! 3. **PRBS** again: both verdicts recover within a few dozen steps.
//!
//! Run with `cargo run --example rls-identifiability`.

use miniident::prelude::*;
use minikalman::prelude::*;
use minikalman::rls::EstimatorStorage;

const A: f64 = 0.9;
const B: f64 = 0.5;
const NOISE_SIGMA: f64 = 0.02;
const FORGETTING: f64 = 0.98;
const P0: f64 = 10.0;
const OPERATING_POINT: f64 = 1.0;

fn main() {
    let mut storage = EstimatorStorage::<2, f64>::new(0.0);
    let mut rls = storage.build(P0);
    rls.set_forgetting_factor(FORGETTING);
    rls.set_measurement_variance(NOISE_SIGMA * NOISE_SIGMA);

    let mut excitation = ExcitationMonitor::for_rls(&rls, ExcitationThresholds::default());
    let identifiability = Identifiability::new([0.02, 0.05], 0.95);

    let mut prbs = Prbs::new(9, 4).unwrap();
    let mut noise = Gaussian::new(42);

    let (mut y, mut u) = (0.0, 0.0);
    println!(
        " step  input  a_hat   b_hat   σa/tol  σb/tol  worst  direction        identifiability  excitation"
    );

    for k in 0..1000 {
        let phi = [y, u];
        y = A * y + B * u + NOISE_SIGMA * noise.sample();

        rls.update(&phi, y);
        excitation.update(&phi);

        let resting = (300..700).contains(&k);
        u = if resting {
            OPERATING_POINT
        } else {
            prbs.next_around(OPERATING_POINT, 0.5)
        };

        if k % 50 == 49 {
            let id = identifiability.assess_rls(&rls);
            let ex = excitation.assess();
            let theta = rls.parameters();
            let d = id.worst_direction;
            println!(
                "{:5}  {}  {:6.3}  {:6.3}  {:6.2}  {:6.2}  {:5.1}  [{:+.2}, {:+.2}]   {:<15}  {:?}",
                k + 1,
                if resting { "rest " } else { "prbs " },
                theta.get_row(0),
                theta.get_row(1),
                id.relative_std_dev[0],
                id.relative_std_dev[1],
                id.worst_relative_std_dev,
                d[0],
                d[1],
                format!("{:?}", id.status),
                ex.status,
            );
        }
    }
}

/// Seeded standard normal samples (xorshift64* and Box–Muller), to keep the example free of
/// dependencies.
struct Gaussian(u64);

impl Gaussian {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn sample(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}
