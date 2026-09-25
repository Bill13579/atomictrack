// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

#[cfg(feature = "smaller-atomics")]
use core::sync::atomic::AtomicU32;
#[cfg(feature = "smaller-atomics")]
pub type AtomicType = AtomicU32;
#[cfg(feature = "smaller-atomics")]
pub type NumericType = u32;

#[cfg(not(feature = "smaller-atomics"))]
use core::sync::atomic::AtomicU64;
#[cfg(not(feature = "smaller-atomics"))]
pub type AtomicType = AtomicU64;
#[cfg(not(feature = "smaller-atomics"))]
pub type NumericType = u64;

pub const MSB: NumericType = NumericType::MAX - (NumericType::MAX >> 1);

/// bits `0..user_bit_low`:                  number
///
/// bits `user_bit_low..NumericType::BITS`:  custom reserved user bits
///
/// bit  `NumericType::BITS`:                atomictrack internal
pub const fn is_valid_user_bit_low(user_bit_low: u32) -> bool {
    user_bit_low >= 3 && user_bit_low <= NumericType::BITS
}

pub const fn number_mask(user_bit_low: u32) -> NumericType {
    assert!(is_valid_user_bit_low(user_bit_low), "user_bit_low should be within range 3..=NumericType::BITS");
    (1 << (user_bit_low - 1)) - 1
}

pub const fn user_mask(user_bit_low: u32) -> NumericType {
    !number_mask(user_bit_low) & !MSB
}

#[inline]
pub const fn gte_masked(a: NumericType, b: NumericType, mask: NumericType) -> bool {
    a.wrapping_sub(b) & mask <= mask >> 1
}

#[inline]
pub const fn lte_masked(a: NumericType, b: NumericType, mask: NumericType) -> bool {
    gte_masked(b, a, mask)
}

#[inline]
pub const fn max2_masked(a: NumericType, b: NumericType, mask: NumericType) -> NumericType {
    if gte_masked(b, a, mask) { b } else { a }
}

#[inline]
pub const fn min2_masked(a: NumericType, b: NumericType, mask: NumericType) -> NumericType {
    if lte_masked(b, a, mask) { b } else { a }
}

#[inline]
pub const fn max3_masked(a: NumericType, b: NumericType, c: NumericType, mask: NumericType) -> NumericType {
    max2_masked(max2_masked(a, b, mask), c, mask)
}

/// Wrapping-aware distance (a - b) between two numbers with MSB masked out, handling the case where `a` has wrapped past MAX but `b` hasn't.
#[inline]
pub fn dist_msb_masked(a: NumericType, b: NumericType) -> NumericType {
    a.wrapping_sub(b) & !MSB
}

/// Wrapping-aware comparison between two numbers with MSB masked out: is `a` ahead of or equal to `b`?
/// Returns true if a >= b in the circular sense (distance < MAX/**4**).
#[inline]
pub fn gte_msb_masked(a: NumericType, b: NumericType) -> bool {
    // If a == b, distance is 0.
    // If a is "ahead" of b (even across wrap), wrapping_sub gives a small positive number.
    // If a is "behind" b, wrapping_sub gives a huge number (> MAX/4).
    a.wrapping_sub(b) & !MSB <= (NumericType::MAX / 4)
}

/// Wrapping-aware addition of two numbers with MSB masked out.
#[inline]
pub fn add_msb_masked(a: NumericType, b: NumericType) -> NumericType {
    a.wrapping_add(b) & !MSB
}

/// Wrapping-aware comparison between two numbers with MSB masked out: is `a` behind or equal to `b`?
/// Returns true if a <= b in the circular sense (distance < MAX/**4**).
#[inline]
pub fn lte_msb_masked(a: NumericType, b: NumericType) -> bool {
    gte_msb_masked(b, a)
}

/// Wrapping-aware comparison between two numbers with MSB masked out: is `a` strictly ahead of `b`?
#[inline]
pub fn gt_msb_masked(a: NumericType, b: NumericType) -> bool {
    a != b && gte_msb_masked(a, b)
}

/// Wrapping-aware comparison between two numbers with MSB masked out: is `a` strictly behind `b`?
#[inline]
pub fn lt_msb_masked(a: NumericType, b: NumericType) -> bool {
    gt_msb_masked(b, a)
}

/// Three-way wrapping-aware max among three numbers with MSB masked out, assuming that ***all*** the numbers live within a MAX/**4** ***range in total*** basically.
#[inline]
pub fn max3_msb_masked(a: NumericType, b: NumericType, c: NumericType) -> NumericType {
    let ab = if gte_msb_masked(b, a) { b } else { a };
    if gte_msb_masked(c, ab) { c } else { ab }
}

/// Two-way wrapping-aware max among two numbers with MSB masked out, assuming that ***all*** the numbers (not just these two, *all* of them) live within a MAX/**4** ***range in total*** basically.
///
/// The additional requirement is because all values must fit inside a contiguous circular interval narrower than MAX/**4** (or 2^(BITS - 2)) (inclusive) for the max (and min for that matter) to be well-defined. If the values are too far apart, the max is utterly ambiguous. Who *knows* who is ahead and who is behind anymore?
#[inline]
pub fn max2_msb_masked(a: NumericType, b: NumericType) -> NumericType {
    if gte_msb_masked(b, a) { b } else { a }
}

/// `min2` equivalent of `max2_msb_masked`.
#[inline]
pub fn min2_msb_masked(a: NumericType, b: NumericType) -> NumericType {
    if lte_msb_masked(b, a) { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_PUBLIC: NumericType = MSB - 1;

    #[test]
    fn distance_and_addition_wrap_at_the_reserved_bit() {
        assert_eq!(dist_msb_masked(1, MAX_PUBLIC), 2);
        assert_eq!(add_msb_masked(MAX_PUBLIC, 1), 0);
        assert_eq!(add_msb_masked(MAX_PUBLIC - 1, 3), 1);
    }

    #[test]
    fn comparisons_and_extrema_work_across_wrap() {
        let older = MAX_PUBLIC - 1;
        let middle = 0;
        let newer = 1;

        assert!(gte_msb_masked(newer, older));
        assert!(gt_msb_masked(newer, older));
        assert!(lte_msb_masked(older, newer));
        assert!(lt_msb_masked(older, newer));
        assert!(gte_msb_masked(middle, middle));
        assert!(!gt_msb_masked(middle, middle));

        assert_eq!(max2_msb_masked(older, newer), newer);
        assert_eq!(min2_msb_masked(older, newer), older);
        assert_eq!(max3_msb_masked(older, middle, newer), newer);
        assert_eq!(max3_msb_masked(newer, older, middle), newer);
    }

    #[test]
    fn masks_split_the_word_at_the_user_bit_low() {
        assert_eq!(number_mask(NumericType::BITS), !MSB);
        assert_eq!(user_mask(NumericType::BITS), 0);
        assert_eq!(number_mask(NumericType::BITS - 1), MSB / 2 - 1);
        assert_eq!(user_mask(NumericType::BITS - 1), MSB / 2);
        assert_eq!(number_mask(3), 0b11);
        assert_eq!(user_mask(3), !0b11 & !MSB);
        assert!(!is_valid_user_bit_low(2));
        assert!(!is_valid_user_bit_low(NumericType::BITS + 1));
    }

    #[test]
    fn masked_comparisons_match_msb_versions_by_default() {
        let samples = [0, 1, 7, MAX_PUBLIC, MAX_PUBLIC - 1, NumericType::MAX / 4, NumericType::MAX / 4 + 1, MSB / 3];
        for a in samples {
            for b in samples {
                assert_eq!(gte_masked(a, b, !MSB), gte_msb_masked(a, b), "{a} {b}");
                assert_eq!(max2_masked(a, b, !MSB), max2_msb_masked(a, b), "{a} {b}");
                assert_eq!(min2_masked(a, b, !MSB), min2_msb_masked(a, b), "{a} {b}");
            }
        }
    }

    #[test]
    fn masked_comparisons_ignore_bits_above_the_mask_and_wrap_within_it() {
        let counter = number_mask(NumericType::BITS - 1);
        let flag = user_mask(NumericType::BITS - 1);

        assert!(!gte_masked(flag | 1, 5, counter));
        assert!(gte_masked(5, flag | 1, counter));
        assert!(gte_masked(0, counter, counter));
        assert!(!gte_masked(counter, 0, counter));
        assert_eq!(max2_masked(counter, flag, counter), flag);
        assert_eq!(max3_masked(3, flag | 9, counter, counter), flag | 9);
    }

    #[test]
    fn comparison_boundary_excludes_exactly_half_the_ring() {
        let furthest_unambiguous = NumericType::MAX / 4;
        let exactly_half_the_ring = furthest_unambiguous + 1;

        assert!(gte_msb_masked(furthest_unambiguous, 0));
        assert!(!gte_msb_masked(exactly_half_the_ring, 0));
        assert!(!gte_msb_masked(0, exactly_half_the_ring));
    }
}
