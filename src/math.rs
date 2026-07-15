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
