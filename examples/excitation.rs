//! Excitation monitoring on a simulated ARX plant
//!
//!     y_k = a·y_{k-1} + b1·u_{k-1} + b2·u_{k-2} + e_k
//!
//! with regressor φ_k = [y_{k-1}, u_{k-1}, u_{k-2}].
//!
//! Phase 1 holds the input constant: once the output settles, all three
//! regressor entries are constant and only one combination of (a, b1, b2) is
//! identifiable. Phase 2 switches to a PRBS dithered around the same operating
//! point and the monitor recovers.
//!
//! Run with `cargo run --example excitation`.

use miniident::excitation::{ExcitationMonitor, ExcitationThresholds};
use miniident::prbs::Prbs;

fn main() {
    let (a, b1, b2) = (0.9, 0.5, 0.2);
    let mut monitor = ExcitationMonitor::<f64, 3>::new(0.98, ExcitationThresholds::default());
    let mut prbs = Prbs::new(9, 2).unwrap();

    let (mut y1, mut u1, mut u2) = (0.0, 0.0, 0.0);
    println!(" step  phase  status          deficient  min_eig    cond      weakest direction");

    for k in 0..1200 {
        let (phase, u) = if k < 600 {
            ("const", 1.0)
        } else {
            ("prbs ", prbs.next_around(1.0, 0.3))
        };

        let phi = [y1, u1, u2];
        let y = a * y1 + b1 * u1 + b2 * u2;
        monitor.update(&phi);

        if k % 100 == 99 {
            let r = monitor.assess();
            let d = r.weakest_direction;
            println!(
                "{:5}  {}  {:<14}  {:>9}  {:9.2e}  {:8.1e}  [{:+.2}, {:+.2}, {:+.2}]",
                k + 1,
                phase,
                format!("{:?}", r.status),
                r.deficient_directions,
                r.min_eigenvalue,
                r.condition,
                d[0],
                d[1],
                d[2]
            );
        }

        (y1, u2, u1) = (y, u1, u);
    }
}
