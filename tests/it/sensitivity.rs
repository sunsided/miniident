//! Propagated output sensitivities against re-simulation at perturbed parameters.

use miniident::sensitivity::{Sensitivity, jacobian};
use proptest::prelude::*;

/// Number of simulated steps per case.
const STEPS: usize = 40;

/// Central-difference step on the parameters.
const STEP: f64 = 1e-6;

/// Second-order plant `x₀' = x₁`, `x₁' = a·x₁ + b·x₀ + c·u`, `y = x₀ + x₁`, with `θ = [a, b, c]`.
fn f(state: &[f64; 2], input: f64, theta: &[f64; 3]) -> [f64; 2] {
    let [damping, stiffness, gain] = *theta;
    [
        state[1],
        damping * state[1] + stiffness * state[0] + gain * input,
    ]
}

/// Output `y = x₀ + x₁`; does not depend on `θ`.
fn h(x: &[f64; 2]) -> [f64; 1] {
    [x[0] + x[1]]
}

/// Outputs of the plant from `x = 0` for the input sequence.
fn simulate(theta: &[f64; 3], inputs: &[f64]) -> Vec<f64> {
    let mut x = [0.0; 2];
    inputs
        .iter()
        .map(|&u| {
            let [y] = h(&x);
            x = f(&x, u, theta);
            y
        })
        .collect()
}

proptest! {
    /// `∂y_k/∂θ` from the recursion with numerical Jacobians equals the difference quotient of
    /// whole re-simulated trajectories.
    #[test]
    fn recursion_matches_resimulation(
        a in -0.6..0.6_f64,
        b in -0.3..0.3_f64,
        c in 0.1..2.0_f64,
        inputs in proptest::collection::vec(-1.0..1.0_f64, STEPS),
    ) {
        let theta = [a, b, c];
        let delta_x = [1e-5; 2];
        let delta_theta = [1e-5; 3];
        let fail = || TestCaseError::fail("steps are positive");

        let mut sens = Sensitivity::<f64, 2, 3>::new();
        let mut x = [0.0; 2];
        let mut propagated = Vec::with_capacity(STEPS);
        for &u in &inputs {
            let h_x = jacobian(h, &x, &delta_x).ok_or_else(fail)?;
            propagated.push(sens.output(&h_x, &[[0.0; 3]])[0]);

            let f_x = jacobian(|v| f(v, u, &theta), &x, &delta_x).ok_or_else(fail)?;
            let f_theta = jacobian(|th| f(&x, u, th), &theta, &delta_theta).ok_or_else(fail)?;
            sens.step(&f_x, &f_theta);
            x = f(&x, u, &theta);
        }

        for (i, _) in theta.iter().enumerate() {
            let (mut plus, mut minus) = (theta, theta);
            plus[i] += STEP;
            minus[i] -= STEP;
            let (yp, ym) = (simulate(&plus, &inputs), simulate(&minus, &inputs));
            for (k, s) in propagated.iter().enumerate() {
                let fd = (yp[k] - ym[k]) / (2.0 * STEP);
                prop_assert!((s[i] - fd).abs() < 1e-5 * fd.abs().max(1.0), "θ{i}, k={k}: {} vs {fd}", s[i]);
            }
        }
    }
}
