// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo and The Crossbeam Project Developers
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

use core::fmt;
use core::ops::{Deref, DerefMut};

/// Pads and aligns a value to the length of a cache line. Forked from crossbeam-utils.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
#[cfg_attr(
    all(
        not(feature = "no-cache-pad"),
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "arm64ec",
            target_arch = "powerpc64",
        )
    ),
    repr(align(128))
)]
#[cfg_attr(
    all(
        not(feature = "no-cache-pad"),
        any(
            target_arch = "arm",
            target_arch = "mips",
            target_arch = "mips32r6",
            target_arch = "mips64",
            target_arch = "mips64r6",
            target_arch = "sparc",
            target_arch = "hexagon",
        )
    ),
    repr(align(32))
)]
#[cfg_attr(all(not(feature = "no-cache-pad"), target_arch = "m68k"), repr(align(16)))]
#[cfg_attr(all(not(feature = "no-cache-pad"), target_arch = "s390x"), repr(align(256)))]
#[cfg_attr(
    all(
        not(feature = "no-cache-pad"),
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "arm64ec",
            target_arch = "powerpc64",
            target_arch = "arm",
            target_arch = "mips",
            target_arch = "mips32r6",
            target_arch = "mips64",
            target_arch = "mips64r6",
            target_arch = "sparc",
            target_arch = "hexagon",
            target_arch = "m68k",
            target_arch = "s390x",
        ))
    ),
    repr(align(64))
)]
#[cfg_attr(
    feature = "no-cache-pad",
    repr(align(4))
)]
pub struct CachePadded<T> {
    value: T,
}

unsafe impl<T: Send> Send for CachePadded<T> {}
unsafe impl<T: Sync> Sync for CachePadded<T> {}

/// Stride in elements of `T`, such that each element is spread out on its own cache-line,
/// taking up different positions on each cache-line through different cycles through the waiting pool
/// by the stride being coprime with the pool size.
///
/// More specific:
///
/// a ≡ b (mod n) means n | (a - b)
///
/// 17 ≡ 1 (mod 8) means 8 divides 17 - 1 = 16, and yes, 16 / 8 = 2
///
/// a = base + i·s \
/// b = base + j·s
///
/// collision happens when a ≡ b (mod 2^k) where 2^k is the waiting pool size,
/// or when 2^k divides |a - b| = |i·s - j·s| = |(i-j)·s|
///
/// s does not have any factors of 2 by construction, so 2^k is unable to divide it, and while |i-j| is less than 2^k (0 < |i-j| < 2^k), 2^k can never divide |i-j| either.
///
/// There are no collisions until after consecutive 2^k range. □
pub(crate) const fn stride<T>() -> usize {
    align_of::<CachePadded<()>>().div_ceil(size_of::<T>()) | 1
}

impl<T> CachePadded<T> {
    pub const fn new(value: T) -> Self {
        Self { value }
    }

    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: fmt::Debug> fmt::Debug for CachePadded<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachePadded")
            .field("value", &self.value)
            .finish()
    }
}

impl<T> From<T> for CachePadded<T> {
    fn from(t: T) -> Self {
        Self::new(t)
    }
}

impl<T: fmt::Display> fmt::Display for CachePadded<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}
