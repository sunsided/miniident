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

/// True pole `a` of the plant.
const A: f64 = 0.9;
/// True input gain `b` of the plant.
const B: f64 = 0.5;
/// Standard deviation of the process noise `wₖ`.
const NOISE_SIGMA: f64 = 0.02;
/// Forgetting factor `μ` of the estimator and the excitation monitor.
const FORGETTING: f64 = 0.98;
/// Initial covariance `P₀ = p0·I`; well above the squared tolerances.
const P0: f64 = 10.0;
/// Input level the plant rests at, and the center of the PRBS.
const OPERATING_POINT: f64 = 1.0;
/// Number of simulated steps.
const STEPS: u32 = 1000;
/// Steps during which the input rests at the operating point.
const REST: core::ops::RangeInclusive<u32> = 301..=700;
/// A table row is printed every this many steps.
const REPORT_EVERY: u32 = 50;
/// `2⁵³`: the scale that maps 53 random bits to `[0, 1)`.
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;

/// Runs the three phases and prints one table row every [`REPORT_EVERY`] steps.
fn main() {
    let mut storage = EstimatorStorage::<2, f64>::new(0.0);
    let mut rls = storage.build(P0);
    rls.set_forgetting_factor(FORGETTING);
    rls.set_measurement_variance(NOISE_SIGMA * NOISE_SIGMA);

    let mut excitation = ExcitationMonitor::for_rls(&rls, ExcitationThresholds::default());
    let identifiability = Identifiability::new([0.02, 0.05], 0.95);

    let mut prbs = Prbs::new(9, 4).expect("order 9 and hold 4 are valid");
    let mut noise = Gaussian::new(42);

    let (mut y, mut u) = (0.0, 0.0);
    println!(
        " step  input  a_hat   b_hat   σa/tol  σb/tol  worst  direction        identifiability  excitation"
    );

    for step in 1..=STEPS {
        let phi = [y, u];
        y = A * y + B * u + NOISE_SIGMA * noise.sample();

        rls.update(&phi, y);
        excitation.update(&phi);

        let resting = REST.contains(&step);
        u = if resting {
            OPERATING_POINT
        } else {
            prbs.next_around(OPERATING_POINT, 0.5)
        };

        if step % REPORT_EVERY == 0 {
            let id = identifiability.assess_rls(&rls);
            let ex = excitation.assess();
            let theta = rls.parameters();
            let d = id.worst_direction;
            println!(
                "{:5}  {}  {:6.3}  {:6.3}  {:6.2}  {:6.2}  {:5.1}  [{:+.2}, {:+.2}]   {:<15}  {:?}",
                step,
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
    /// Creates the generator; a zero seed becomes 1, because xorshift stays at zero.
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// A uniform sample in `(0, 1)`, never 0, so that its logarithm is finite.
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (top_53_bits(x) + 0.5) / TWO_POW_53
    }

    /// A standard normal sample.
    fn sample(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

/// The 53 high bits of `x` as an `f64`.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "x >> 11 is below 2⁵³, so f64 represents it exactly"
)]
fn top_53_bits(x: u64) -> f64 {
    (x >> 11) as f64
}
