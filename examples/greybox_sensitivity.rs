//! Excitation of a nonlinear grey-box model, judged from propagated output sensitivities.
//!
//! The plant is a motor with smooth Coulomb friction,
//! `vₖ₊₁ = a·vₖ + b·uₖ − c·tanh(vₖ/ε)`, measured as `yₖ = vₖ`, with unknown `θ = [a, b, c]`.
//! The friction term makes it nonlinear in the state, so there is no linear regressor `φ`.
//! Instead, [`Sensitivity`] propagates `∂yₖ/∂θ` through the model's Jacobians, here obtained
//! with [`jacobian`] by central differences, and feeds them to the [`ExcitationMonitor`].
//!
//! The run has three phases: PRBS around the operating point, rest at the operating point,
//! PRBS again. At rest, the sensitivities settle to a constant vector: only one parameter
//! combination is observed, forgetting erodes the information in the other two at the rate
//! `μ`, and the monitor turns `IllConditioned` within ~150 steps. The weakest direction it
//! reports is the combination the steady state cannot see.
//!
//! Run with `cargo run --example greybox-sensitivity`.

use miniident::prelude::*;

/// True pole `a`.
const A: f64 = 0.9;
/// True input gain `b`.
const B: f64 = 0.5;
/// True friction level `c`.
const C: f64 = 0.05;
/// Velocity scale `ε` of the friction's sign transition.
const EPSILON: f64 = 0.1;
/// Standard deviation of the measurement noise; weighs each sensitivity sample by `1/σ²`.
const NOISE_SIGMA: f64 = 0.02;
/// Forgetting factor `μ` of the excitation monitor.
const FORGETTING: f64 = 0.98;
/// Input level the plant rests at, and the center of the PRBS.
const OPERATING_POINT: f64 = 1.0;
/// Number of simulated steps.
const STEPS: u32 = 1000;
/// Steps during which the input rests at the operating point.
const REST: core::ops::RangeInclusive<u32> = 301..=700;
/// A table row is printed every this many steps.
const REPORT_EVERY: u32 = 50;
/// Central-difference steps for the Jacobians over the state and over `θ`.
const DELTA: f64 = 1e-6;

/// The state transition `f(v, u, θ)`.
fn f(v: [f64; 1], u: f64, theta: &[f64; 3]) -> [f64; 1] {
    let [pole, gain, friction] = *theta;
    [pole * v[0] + gain * u - friction * (v[0] / EPSILON).tanh()]
}

/// Runs the three phases and prints one table row every [`REPORT_EVERY`] steps.
fn main() {
    let theta = [A, B, C];
    let mut sensitivity = Sensitivity::<f64, 1, 3>::new();
    let mut excitation = ExcitationMonitor::new(FORGETTING, ExcitationThresholds::default());
    let mut prbs = Prbs::new(9, 4).expect("order 9 and hold 4 are valid");
    let inv_noise_variance = [1.0 / (NOISE_SIGMA * NOISE_SIGMA)];

    // y = v: H_x = [1], H_θ = 0.
    let (h_x, h_theta) = ([[1.0]], [[0.0; 3]]);

    let (mut v, mut u) = ([0.0], OPERATING_POINT);
    println!(
        " step  input  ∂y/∂a    ∂y/∂b    ∂y/∂c    λ_min       weakest direction        excitation"
    );

    for step in 1..=STEPS {
        let dy = sensitivity.output(&h_x, &h_theta);
        excitation.update_sensitivity(&dy, &inv_noise_variance);

        let f_x = jacobian(|x| f(*x, u, &theta), &v, &[DELTA]).expect("positive step");
        let f_theta = jacobian(|th| f(v, u, th), &theta, &[DELTA; 3]).expect("positive steps");
        sensitivity.step(&f_x, &f_theta);
        v = f(v, u, &theta);

        let resting = REST.contains(&step);
        u = if resting {
            OPERATING_POINT
        } else {
            prbs.next_around(OPERATING_POINT, 0.5)
        };

        if step % REPORT_EVERY == 0 {
            let ex = excitation.assess();
            let [sa, sb, sc] = dy[0];
            let w = ex.weakest_direction;
            println!(
                "{step:5}  {}  {sa:+7.3}  {sb:+7.3}  {sc:+7.3}  {:9.2e}  [{:+.2}, {:+.2}, {:+.2}]  {:?}",
                if resting { "rest " } else { "prbs " },
                ex.min_eigenvalue,
                w[0],
                w[1],
                w[2],
                ex.status,
            );
        }
    }
}
