//! # miniident
//!
//! Allocation-free, `no_std` building blocks for **online system identification
//! diagnostics**: knowing whether the data you are estimating from actually
//! contains enough information to identify your parameters, and generating
//! inputs that make sure it does.
//!
//! `miniident` deliberately does not estimate anything itself. Estimation
//! (RLS, EKF/UKF, augmented and dual filters, smoothers) is the job of
//! [`minikalman`](https://crates.io/crates/minikalman). `miniident` sits next to
//! the estimator and consumes the same regressors or output sensitivities.
//!
//! ## Modules
//!
//! - [`information`]: exponentially weighted information (Fisher) matrix from
//!   regressors or output sensitivities, log-determinant and Cramér–Rao bound.
//! - [`excitation`]: persistent excitation monitoring with a verdict and the
//!   weakest identifiable parameter direction.
//! - [`prbs`]: maximal-length pseudo-random binary sequences for excitation.
//! - [`linalg`]: the small dense linear algebra the above needs
//!   (Jacobi eigen-decomposition, Cholesky log-determinant).
//!
//! ## Quick start
//!
//! ```
//! use miniident::excitation::{ExcitationMonitor, ExcitationStatus, ExcitationThresholds};
//! use miniident::prbs::Prbs;
//!
//! // Two-tap FIR model: y_k = b1 u_{k-1} + b2 u_{k-2}.
//! let mut monitor = ExcitationMonitor::<f64, 2>::new(0.99, ExcitationThresholds::default());
//! let mut prbs = Prbs::new(7, 1).unwrap();
//!
//! let (mut u1, mut u2) = (0.0, 0.0);
//! for _ in 0..500 {
//!     monitor.update(&[u1, u2]);
//!     u2 = u1;
//!     u1 = prbs.next_level(1.0);
//! }
//! assert_eq!(monitor.assess().status, ExcitationStatus::Sufficient);
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(test)]
extern crate std;

pub mod excitation;
pub mod information;
pub mod linalg;
pub mod prbs;

pub use num_traits::Float;
