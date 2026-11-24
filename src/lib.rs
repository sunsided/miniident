//! # miniident
//!
//! Online identifiability and excitation diagnostics for
//! [minikalman](https://crates.io/crates/minikalman) estimators.
//!
//! minikalman estimates parameters: with [`RecursiveLeastSquares`], or as part of the state of
//! an augmented EKF or UKF. miniident answers the questions that come after:
//!
//! - **Are the parameters identified?** [`identifiability`] reads the estimator's own
//!   covariance and reports, per parameter and per parameter combination, whether the
//!   uncertainty is within the tolerance you set, and which parameters the data cannot tell
//!   apart. [`prior_ratio`](identifiability::prior_ratio) tells how much of that covariance
//!   is still the prior.
//! - **Why not?** [`excitation`] looks at the regressors themselves and tells whether the
//!   input is persistently exciting, and which parameter combination it starves.
//!   For grey-box models, [`sensitivity`] propagates `∂y/∂θ` from the model's Jacobians, so
//!   the same check works without a linear regressor.
//! - **What fixes it?** [`prbs`] generates maximal-length binary excitation, held for as many
//!   samples as the plant's bandwidth needs.
//!
//! ```mermaid
//! flowchart LR
//!   prbs["prbs: excitation input"] --> plant[plant]
//!   plant -- "φ, y" --> rls["minikalman estimator"]
//!   plant -- "φ" --> excitation["excitation: is the input informative?"]
//!   plant -- "Jacobians" --> sensitivity["sensitivity: ∂y/∂θ"]
//!   sensitivity -- "∂y/∂θ" --> excitation
//!   rls -- "covariance P" --> identifiability["identifiability: is θ trusted?"]
//!   identifiability -- "Unidentified: excite again" --> prbs
//! ```
//!
//! ## Why the estimator's covariance
//!
//! For RLS, `Pₖ₊₁⁻¹ = μₖ·Pₖ⁻¹ + φₖ·φₖᵀ/r`: the inverse covariance *is* the information
//! matrix, including the windup guard's variable forgetting `μₖ`, the measurement variance
//! `r`, the prior `P₀` and every gated sample that was left out. A separately tracked
//! information matrix would drift from what the estimator actually knows, so the verdict
//! comes from `P` itself. For an augmented-state filter, the parameter block of `P` plays the
//! same role.
//!
//! ## Numeric types
//!
//! Everything is generic over minikalman's [`MatrixDataType`], works on const-generic arrays
//! and allocates nothing. Functions that need a logarithm require [`Logarithm`], as in
//! minikalman.
//!
//! [`RecursiveLeastSquares`]: minikalman::rls::RecursiveLeastSquares
//! [`MatrixDataType`]: minikalman::matrix::MatrixDataType
//! [`Logarithm`]: minikalman::matrix::Logarithm

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(test)]
extern crate std;

pub mod excitation;
pub mod identifiability;
pub mod information;
pub mod linalg;
pub mod prbs;
pub mod sensitivity;

pub use minikalman;

/// Doctests for the code examples in README.md (not part of the API).
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

/// The commonly used types.
pub mod prelude {
    pub use crate::excitation::{
        ExcitationMonitor, ExcitationReport, ExcitationStatus, ExcitationThresholds,
    };
    pub use crate::identifiability::{
        Identifiability, IdentifiabilityReport, IdentifiabilityStatus, PriorReport, prior_ratio,
        prior_ratio_rls,
    };
    pub use crate::information::InformationMatrix;
    pub use crate::prbs::Prbs;
    pub use crate::sensitivity::{Sensitivity, jacobian};
}
