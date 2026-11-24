//! Output sensitivities of a grey-box model, propagated along a trajectory.
//!
//! For a model with state `x` (`S` entries), input `u` and parameters `θ` (`N` entries),
//!
//! ```text
//! x_{k+1} = f(x_k, u_k, θ)        y_k = h(x_k, u_k, θ)    (M outputs)
//! ```
//!
//! the output sensitivity `∂y_k/∂θ` depends on the whole history, because `x_k` itself
//! depends on `θ`. With the state sensitivity `S_k = ∂x_k/∂θ` (`S` × `N`, `S_0 = 0`):
//!
//! ```text
//! ∂y_k/∂θ = H_x·S_k + H_θ          S_{k+1} = F_x·S_k + F_θ
//! ```
//!
//! where `F_x = ∂f/∂x`, `F_θ = ∂f/∂θ`, `H_x = ∂h/∂x` and `H_θ = ∂h/∂θ` are evaluated at
//! `(x_k, u_k, θ)`. [`Sensitivity`] runs this recursion, so
//! [`InformationMatrix::update_sensitivity`](crate::information::InformationMatrix::update_sensitivity)
//! and
//! [`ExcitationMonitor::update_sensitivity`](crate::excitation::ExcitationMonitor::update_sensitivity)
//! can be fed from the model's Jacobians, which an EKF needs anyway, instead of a hand-derived
//! `∂y/∂θ`. Per step, in this order:
//!
//! 1. `let s = sensitivity.output(&h_x, &h_theta);` at `(x_k, u_k)`,
//! 2. `monitor.update_sensitivity(&s, &inv_noise_variance);`,
//! 3. `sensitivity.step(&f_x, &f_theta);`.
//!
//! Where no analytic Jacobian is at hand, [`jacobian`] fills each block by central differences
//! of `f` or `h` over `x` or `θ`.
//!
//! ## Limits
//!
//! - This is the **open-loop** (output-error) sensitivity of the simulated model: it ignores
//!   the corrections a filter applies to `x`. It measures what the input can reveal about
//!   `θ` through the model, not what a particular filter has learned; for that, judge the
//!   filter's covariance with [`Identifiability`](crate::identifiability::Identifiability).
//! - For an unstable `F_x`, `S_k` grows without bound, which overflows fixed-point types. Call
//!   [`Sensitivity::reset`] where the trajectory restarts.

use minikalman::matrix::MatrixDataType;

/// The state sensitivity `S_k = ∂x_k/∂θ` of a model with `S` states and `N` parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sensitivity<T, const S: usize, const N: usize> {
    /// `∂x_k/∂θ`, row `i` is `∂xᵢ/∂θ`.
    s: [[T; N]; S],
}

impl<T: MatrixDataType, const S: usize, const N: usize> Default for Sensitivity<T, S, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: MatrixDataType, const S: usize, const N: usize> Sensitivity<T, S, N> {
    /// Starts with `S_0 = 0`: an initial state that does not depend on `θ`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            s: [[T::zero(); N]; S],
        }
    }

    /// Restarts at `S = 0`, e.g. where the simulated trajectory is re-initialized.
    pub fn reset(&mut self) {
        self.s = [[T::zero(); N]; S];
    }

    /// The current state sensitivity `S_k = ∂x_k/∂θ`.
    #[must_use]
    pub fn state(&self) -> &[[T; N]; S] {
        &self.s
    }

    /// The output sensitivity `∂y_k/∂θ = H_x·S_k + H_θ`, with the Jacobians of `h` at
    /// `(x_k, u_k, θ)`. Call before [`step`](Self::step). `O(M·S·N)`.
    #[must_use]
    pub fn output<const M: usize>(&self, h_x: &[[T; S]; M], h_theta: &[[T; N]; M]) -> [[T; N]; M] {
        affine(h_x, &self.s, h_theta)
    }

    /// Advances to `S_{k+1} = F_x·S_k + F_θ`, with the Jacobians of `f` at `(x_k, u_k, θ)`.
    /// `O(S²·N)`.
    pub fn step(&mut self, f_x: &[[T; S]; S], f_theta: &[[T; N]; S]) {
        self.s = affine(f_x, &self.s, f_theta);
    }
}

/// `A·X + B` for `A` (`R` × `S`), `X` (`S` × `N`) and `B` (`R` × `N`).
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see linalg docs)"
)]
fn affine<T: MatrixDataType, const R: usize, const S: usize, const N: usize>(
    a: &[[T; S]; R],
    x: &[[T; N]; S],
    b: &[[T; N]; R],
) -> [[T; N]; R] {
    let mut out = *b;
    for (out_row, a_row) in out.iter_mut().zip(a) {
        for (&aij, x_row) in a_row.iter().zip(x) {
            for (o, &xjn) in out_row.iter_mut().zip(x_row) {
                *o += aij * xjn;
            }
        }
    }
    out
}

/// The Jacobian `∂g/∂v` at `at` by central differences, `(g(v + δⱼ·eⱼ) − g(v − δⱼ·eⱼ)) / 2δⱼ`;
/// `None` if a step `delta[j]` is not positive. Calls `g` `2·IN` times.
///
/// Row `i` of the result is `∂gᵢ/∂v`. To get one of the four blocks of [`Sensitivity`], close
/// over the other arguments, e.g. `jacobian(|th| f(&x, &u, th), &theta, &delta)` for `F_θ`.
///
/// The truncation error is `O(δ²)` and the rounding error `O(ε/δ)`; for `f64`, `δ` near
/// `1e-5` times the magnitude of `v` balances the two. In fixed point both are large, so pass
/// analytic Jacobians there.
#[expect(
    clippy::indexing_slicing,
    reason = "j < IN from enumerate over an IN-array; the point copies have IN entries"
)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "MatrixDataType has no checked operators; T defines overflow (see linalg docs)"
)]
pub fn jacobian<T, const IN: usize, const OUT: usize>(
    mut g: impl FnMut(&[T; IN]) -> [T; OUT],
    at: &[T; IN],
    delta: &[T; IN],
) -> Option<[[T; IN]; OUT]>
where
    T: MatrixDataType,
{
    if !delta.iter().all(|&d| d > T::zero()) {
        return None;
    }

    let mut jac = [[T::zero(); IN]; OUT];
    for (j, &d) in delta.iter().enumerate() {
        let (mut plus, mut minus) = (*at, *at);
        plus[j] += d;
        minus[j] -= d;
        let (gp, gm) = (g(&plus), g(&minus));
        for ((row, &p), &m) in jac.iter_mut().zip(&gp).zip(&gm) {
            row[j] = (p - m) / (d + d);
        }
    }
    Some(jac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobian_of_linear_map_is_its_matrix() {
        let a = [[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]];
        let g = |v: &[f64; 3]| a.map(|row| row.iter().zip(v).map(|(x, y)| x * y).sum());
        let j = jacobian(g, &[0.3, -1.0, 2.0], &[0.5, 0.5, 0.5]).unwrap();
        for (jr, ar) in j.iter().zip(&a) {
            for (x, y) in jr.iter().zip(ar) {
                assert!((x - y).abs() < 1e-12, "{j:?}");
            }
        }
    }

    #[test]
    fn jacobian_rejects_non_positive_step() {
        assert!(jacobian(|v: &[f64; 1]| *v, &[0.0], &[0.0]).is_none());
    }

    /// `y_k = x_k`, `x_{k+1} = a·x_k + b·u_k`: compare with re-simulating at perturbed `θ`.
    #[test]
    fn first_order_matches_resimulation() {
        let (a, b) = (0.8, 0.5);
        let mut state = 0.0;
        let input = |k: usize| if (k / 3) % 2 == 0 { 1.0 } else { -1.0 };
        let simulate = |a: f64, b: f64| {
            let mut x = 0.0;
            let mut ys = [0.0; 30];
            for (k, y) in ys.iter_mut().enumerate() {
                *y = x;
                x = a * x + b * input(k);
            }
            ys
        };

        let mut sens = Sensitivity::<f64, 1, 2>::new();
        let step = 1e-6;
        let (ap, am) = (simulate(a + step, b), simulate(a - step, b));
        let (bp, bm) = (simulate(a, b + step), simulate(a, b - step));
        for k in 0..30 {
            let s = sens.output(&[[1.0]], &[[0.0, 0.0]]);
            let fd = [
                (ap[k] - am[k]) / (2.0 * step),
                (bp[k] - bm[k]) / (2.0 * step),
            ];
            assert!((s[0][0] - fd[0]).abs() < 1e-6, "k={k}: {s:?} vs {fd:?}");
            assert!((s[0][1] - fd[1]).abs() < 1e-6, "k={k}: {s:?} vs {fd:?}");
            sens.step(&[[a]], &[[state, input(k)]]);
            state = a * state + b * input(k);
        }
    }
}
