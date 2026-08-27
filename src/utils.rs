// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

#[macro_export]
macro_rules! env_or_default {
    ($env_name:expr, $default:expr, $ty:ty) => {{
        const STR_VAL: &str = match option_env!($env_name) {
            Some(s) => s,
            None => $default,
        };
        
        // Compile-time string-to-integer parsing
        {
            let bytes = STR_VAL.as_bytes();
            let mut value: $ty = 0;
            let mut i = 0;
            while i < bytes.len() {
                value = value * 10 + (bytes[i] - b'0') as $ty;
                i += 1;
            }
            value
        }
    }};
}

#[cfg(feature = "std")]
extern crate std;

#[macro_export]
macro_rules! spin_for_step {
    ($step:expr) => {{
        let step = $step;

        debug_assert!(step < usize::BITS as usize);

        for _ in 0..(1usize << step) {
            core::hint::spin_loop();
        }
    }};
}

#[cfg(feature = "std")]
#[inline]
pub fn yield_now() {
    std::thread::yield_now();
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn yield_now() {
    spin_for_step!(MAX_SPINS); // Don't go beyond 1 << MAX_SPINS spins.
}

pub(crate) const MAX_SPINS: usize = env_or_default!("ATOMICTRACK_MAX_SPINS", "6", usize);
pub(crate) const MAX_LOOPS_BEFORE_SLEEP: usize = env_or_default!("ATOMICTRACK_MAX_LOOPS_BEFORE_SLEEP", "10", usize);
const _: () = assert!(MAX_SPINS <= MAX_LOOPS_BEFORE_SLEEP, "MAX_SPINS must be <= MAX_LOOPS_BEFORE_SLEEP");
const _: () = assert!(MAX_SPINS < usize::BITS as usize, "MAX_SPINS must be less than the number of bits in usize; on each backoff step we spin for 1 << step times, and 1 << usize::BITS is undefined behavior in Rustlang");

