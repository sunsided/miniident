//! Maximal-length pseudo-random binary sequences.
//!
//! A PRBS of order `n` has period `N = 2ⁿ − 1` bits, is almost perfectly balanced
//! (`2ⁿ⁻¹` ones, `2ⁿ⁻¹ − 1` zeros) and, mapped to `±a`, has a circular autocorrelation of
//! `a²` at lag zero and `−a²/N` at every other lag. It excites all frequencies up to the
//! clock rate almost evenly and is persistently exciting of order `N`. Holding each bit for
//! `hold` samples moves the energy below `f_s / hold`, which is how the excitation is matched
//! to a slow plant (minikalman's `rls-adaptive` example holds its input for 20 steps).
//!
//! The generator is a Galois LFSR. The feedback masks for orders 2–32 are low-weight
//! primitive polynomials; orders up to 20 are verified by exhaustive period checks in the
//! tests. Levels are generic over minikalman's `MatrixDataType`, fixed point included.

use minikalman::matrix::MatrixDataType;

/// Galois feedback masks; index `n − 2` for order `n`. Bit `i` set means the characteristic
/// polynomial contains `x^(i+1)`; the constant term is implied.
const MASKS: [u32; 31] = [
    0x3,
    0x5,
    0x9,
    0x12,
    0x21,
    0x41,
    0xC3,
    0x108,
    0x204,
    0x402,
    0x883,
    0x1013,
    0x2803,
    0x4001,
    0x8805,
    0x0001_0004,
    0x0002_0040,
    0x0004_0013,
    0x0008_0004,
    0x0010_0002,
    0x0020_0001,
    0x0040_0010,
    0x0080_0043,
    0x0100_0004,
    0x0200_0023,
    0x0400_0013,
    0x0800_0004,
    0x1000_0002,
    0x2040_0003,
    0x4000_0004,
    0x8020_0003,
];

/// Smallest supported order.
pub const MIN_ORDER: u8 = 2;
/// Largest supported order.
pub const MAX_ORDER: u8 = 32;

/// Maximal-length pseudo-random binary sequence generator.
///
/// It is `Clone` but not `Copy`: a copied iterator silently restarts the sequence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Prbs {
    /// LFSR state; never zero in the low `order` bits.
    state: u32,
    /// Galois feedback mask of the order, from [`MASKS`].
    mask: u32,
    /// LFSR order, in `MIN_ORDER..=MAX_ORDER`.
    order: u8,
    /// Period in bits, `2^order − 1`: the `order` low bits set.
    period_bits: u32,
    /// Samples each bit is held for; at least 1.
    hold: u32,
    /// Samples left before the next bit; below `hold`.
    remaining: u32,
    /// The bit that is held at the moment.
    current: bool,
}

impl Prbs {
    /// Creates a generator of the given order (2–32) that holds every bit for `hold ≥ 1`
    /// samples. Returns `None` for unsupported arguments. `const`, so it can live in a
    /// `static`.
    #[must_use]
    pub const fn new(order: u8, hold: u32) -> Option<Self> {
        Self::with_seed(order, hold, 1)
    }

    /// Like [`Self::new`] with an explicit seed, which selects the phase of the sequence.
    /// Bits above `order` are ignored; a seed that is zero in the low `order` bits is rejected.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        reason = "u8 to u32 and u8 to usize widen without loss; From is not const"
    )]
    #[expect(
        clippy::indexing_slicing,
        reason = "order is in 2..=32, checked above, so order − 2 < 31 = MASKS.len()"
    )]
    pub const fn with_seed(order: u8, hold: u32, seed: u32) -> Option<Self> {
        if order < MIN_ORDER || order > MAX_ORDER || hold == 0 {
            return None;
        }

        let (Some(unused_bits), Some(index)) =
            (MAX_ORDER.checked_sub(order), order.checked_sub(MIN_ORDER))
        else {
            return None;
        };
        let period_bits = u32::MAX >> unused_bits as u32;
        let state = seed & period_bits;
        if state == 0 {
            return None;
        }

        Some(Self {
            state,
            mask: MASKS[index as usize],
            order,
            period_bits,
            hold,
            remaining: 0,
            current: false,
        })
    }

    /// The LFSR order.
    #[must_use]
    pub const fn order(&self) -> u8 {
        self.order
    }

    /// Samples each bit is held for.
    #[must_use]
    pub const fn hold(&self) -> u32 {
        self.hold
    }

    /// Period in bits, `2ⁿ − 1`.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        reason = "u8 to u32 widens without loss; From is not const"
    )]
    pub const fn period_bits(&self) -> u64 {
        self.period_bits as u64
    }

    /// Period in samples, `(2ⁿ − 1)·hold`.
    ///
    /// # Panics
    ///
    /// Does not panic: the product is below `(2³² − 1)² < 2⁶⁴`.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        reason = "u32 to u64 widens without loss; From is not const"
    )]
    pub const fn period_samples(&self) -> u64 {
        self.period_bits()
            .checked_mul(self.hold as u64)
            .expect("(2³² − 1)·(2³² − 1) is below 2⁶⁴")
    }

    /// Next sample as a bit.
    pub fn next_bit(&mut self) -> bool {
        if let Some(left) = self.remaining.checked_sub(1) {
            self.remaining = left;
        } else {
            self.current = self.shift();
            // hold ≥ 1 by construction, so this does not saturate.
            self.remaining = self.hold.saturating_sub(1);
        }

        self.current
    }

    /// Advances the LFSR by one bit and returns the bit shifted out.
    fn shift(&mut self) -> bool {
        let out = self.state & 1 == 1;
        self.state >>= 1;
        if out {
            self.state ^= self.mask;
        }

        out
    }

    /// Next sample mapped to `+amplitude` or `−amplitude`.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "MatrixDataType has no checked operators; T defines overflow on negation"
    )]
    pub fn next_level<T: MatrixDataType>(&mut self, amplitude: T) -> T {
        if self.next_bit() {
            amplitude
        } else {
            -amplitude
        }
    }

    /// Next sample mapped to `offset ± amplitude`, e.g. a duty cycle dithered around an
    /// operating point.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "MatrixDataType has no checked operators; T defines overflow on addition"
    )]
    pub fn next_around<T: MatrixDataType>(&mut self, offset: T, amplitude: T) -> T {
        offset + self.next_level(amplitude)
    }
}

impl Iterator for Prbs {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        Some(self.next_bit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn rejects_invalid() {
        assert!(Prbs::new(1, 1).is_none());
        assert!(Prbs::new(33, 1).is_none());
        assert!(Prbs::new(8, 0).is_none());
        assert!(Prbs::with_seed(8, 1, 0x100).is_none());
    }

    #[test]
    fn maximal_period_up_to_order_20() {
        for order in MIN_ORDER..=20 {
            let mut p = Prbs::new(order, 1).unwrap();
            let start = p.state;
            let mut n = 0u64;
            loop {
                p.next_bit();
                n += 1;
                if p.state == start {
                    break;
                }
            }
            assert_eq!(n, p.period_bits(), "order {order}");
        }
    }

    #[test]
    fn order_32_does_not_lock_up() {
        let mut p = Prbs::new(32, 1).unwrap();
        for _ in 0..100_000 {
            p.next_bit();
            assert_ne!(p.state, 0);
        }
    }

    #[test]
    fn balance_and_autocorrelation() {
        let mut p = Prbs::new(7, 1).unwrap();
        let n = usize::try_from(p.period_bits()).unwrap();
        let s: Vec<i32> = (0..n).map(|_| if p.next_bit() { 1 } else { -1 }).collect();
        assert_eq!(s.iter().sum::<i32>(), 1);
        for lag in 1..n {
            let r: i32 = (0..n).map(|k| s[k] * s[(k + lag) % n]).sum();
            assert_eq!(r, -1, "lag {lag}");
        }
    }

    #[test]
    fn hold_repeats_bits() {
        let mut a = Prbs::new(5, 1).unwrap();
        let mut b = Prbs::new(5, 3).unwrap();
        for _ in 0..40 {
            let bit = a.next_bit();
            for _ in 0..3 {
                assert_eq!(b.next_bit(), bit);
            }
        }
        assert_eq!(b.period_samples(), 93);
    }
}
