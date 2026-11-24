//! Properties of the PRBS generator for every seed and hold.

use miniident::prbs::Prbs;
use proptest::prelude::*;

/// Largest order the properties enumerate a full period of; `2¹² − 1` bits per case.
const MAX_TESTED_ORDER: u8 = 12;

/// Builds a generator, or rejects the case if the seed is zero in the low `order` bits.
fn generator(order: u8, hold: u32, seed: u32) -> Result<Prbs, TestCaseError> {
    Prbs::with_seed(order, hold, seed)
        .ok_or_else(|| TestCaseError::reject("seed is zero in the low order bits"))
}

proptest! {
    /// Every seed gives a sequence that repeats after exactly one period and holds
    /// `2ⁿ⁻¹` ones per period.
    #[test]
    fn every_seed_has_full_period_and_balance(order in 2..=MAX_TESTED_ORDER, seed in any::<u32>()) {
        let mut prbs = generator(order, 1, seed)?;
        let period = usize::try_from(prbs.period_bits()).map_err(|_| TestCaseError::fail("period fits usize"))?;
        let first: Vec<bool> = prbs.by_ref().take(period).collect();
        let second: Vec<bool> = prbs.by_ref().take(period).collect();
        let ones = first.iter().filter(|&&b| b).count();

        prop_assert_eq!(first, second);
        prop_assert_eq!(ones, period.div_ceil(2));
    }

    /// With `hold = h`, every bit of the `hold = 1` sequence appears `h` times in a row.
    #[test]
    fn hold_repeats_each_bit(order in 2..=MAX_TESTED_ORDER, seed in any::<u32>(), hold in 1..8_u32) {
        let mut single = generator(order, 1, seed)?;
        let mut held = generator(order, hold, seed)?;

        for _ in 0..64 {
            let bit = single.next_bit();
            for _ in 0..hold {
                prop_assert_eq!(held.next_bit(), bit);
            }
        }
    }
}
