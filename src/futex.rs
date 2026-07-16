// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org).

//! Small, dependency-free, process-private atomic wait/wake helpers.
//!
//! This module uses [`AtomicU32`] because 32 bits is the widest size supported
//! natively by every backend here. In particular, traditional Linux futexes,
//! FreeBSD's `UMTX_OP_WAIT_UINT_PRIVATE`, and Wasm's `memory.atomic.wait32`
//! all operate on 32-bit words.
//!
//! These functions only block and wake. They do not load or store the value
//! with acquire/release ordering for you. A typical notifier first stores with
//! [`Ordering::Release`] and then calls
//! [`wake_one`] or [`wake_all`]; a waiter normally checks the value with
//! [`Ordering::Acquire`] in a loop.
//!
//! Waits may return spuriously, including when interrupted by a Unix signal, so
//! always re-check the condition. All waits and wakes are process-private.
//!
//! # Platform notes
//!
//! - Windows requires Windows 8 / Server 2012 or newer. Finite nanosecond
//!   timeouts are rounded up to milliseconds and capped at `u32::MAX - 1`
//!   milliseconds (`u32::MAX` means infinite to `WaitOnAddress`).
//! - macOS uses the private `__ulock_wait2` API. This can make an application
//!   unsuitable for App Store distribution. `__ulock_wait2` is available on
//!   macOS 10.15 and newer.
//! - Wasm requires the `atomics` target feature and nightly Rust's
//!   `stdarch_wasm_atomic_wait` feature.

use core::sync::atomic::{AtomicU32, Ordering};

/// Wait while `atomic` equals `expected`.
///
/// Returns immediately if the value is already different. This is a raw,
/// possibly-spurious wait: re-check the condition after it returns.
#[inline]
pub fn wait(atomic: &AtomicU32, expected: u32) {
    if atomic.load(Ordering::Relaxed) != expected {
        return;
    }
    let _ = imp::wait(atomic, expected, None);
}

/// Wait while `atomic` equals `expected`, for at most `timeout_ns` nanoseconds.
///
/// Returns `false` only when the operating system reports that the timeout
/// elapsed. Returns `true` after a wake, a value mismatch, an interruption, or
/// another spurious return. Re-check the atomic value in either case.
///
/// A timeout of zero does not block. On Windows, positive values are rounded
/// up to the next whole millisecond; other platforms retain nanosecond input.
#[inline]
#[must_use]
pub fn wait_timeout(atomic: &AtomicU32, expected: u32, timeout_ns: u64) -> bool {
    if atomic.load(Ordering::Relaxed) != expected {
        return true;
    }
    if timeout_ns == 0 {
        return false;
    }
    imp::wait(atomic, expected, Some(timeout_ns))
}

/// Wake at most one waiter sleeping on `atomic`.
///
/// Returns `Some(number_woken)` on Linux and Wasm. Returns `None` where the
/// platform API does not report a reliable count (Windows, macOS, FreeBSD).
#[inline]
#[must_use]
pub fn wake_one(atomic: &AtomicU32) -> Option<u32> {
    imp::wake(atomic, false)
}

/// Wake all waiters sleeping on `atomic`.
///
/// Returns `Some(number_woken)` on Linux and Wasm. Returns `None` where the
/// platform API does not report a reliable count (Windows, macOS, FreeBSD).
#[inline]
#[must_use]
pub fn wake_all(atomic: &AtomicU32) -> Option<u32> {
    imp::wake(atomic, true)
}

#[cfg(target_os = "windows")]
mod imp {
    use super::AtomicU32;
    use core::ffi::c_void;

    #[cfg_attr(
        target_arch = "x86",
        link(
            name = "api-ms-win-core-synch-l1-2-0",
            kind = "raw-dylib",
            import_name_type = "undecorated"
        )
    )]
    #[cfg_attr(
        not(target_arch = "x86"),
        link(name = "api-ms-win-core-synch-l1-2-0", kind = "raw-dylib")
    )]
    unsafe extern "system" {
        fn WaitOnAddress(
            address: *const c_void,
            compare_address: *const c_void,
            address_size: usize,
            timeout_ms: u32,
        ) -> i32;
        fn WakeByAddressSingle(address: *const c_void);
        fn WakeByAddressAll(address: *const c_void);
    }

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>) -> bool {
        const INFINITE: u32 = u32::MAX;
        let timeout_ms = timeout_ns.map_or(INFINITE, |ns| {
            ns.div_ceil(1_000_000).min((INFINITE - 1) as u64) as u32
        });
        unsafe {
            WaitOnAddress(
                atomic as *const AtomicU32 as *const c_void,
                &expected as *const u32 as *const c_void,
                size_of::<u32>(),
                timeout_ms,
            ) != 0
        }
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool) -> Option<u32> {
        let address = atomic as *const AtomicU32 as *const c_void;
        unsafe {
            if all {
                WakeByAddressAll(address);
            } else {
                WakeByAddressSingle(address);
            }
        }
        None
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::AtomicU32;
    use core::{ffi::c_void, ptr};

    type CLong = isize;

    #[cfg(any(target_arch = "m68k", target_arch = "riscv32"))]
    type TimeUnit = i64;
    #[cfg(not(any(target_arch = "m68k", target_arch = "riscv32")))]
    type TimeUnit = CLong;

    #[repr(C)]
    struct Timespec {
        tv_sec: TimeUnit,
        tv_nsec: TimeUnit,
    }

    const FUTEX_WAIT: i32 = 0;
    const FUTEX_WAKE: i32 = 1;
    const FUTEX_PRIVATE_FLAG: i32 = 128;
    const ETIMEDOUT: i32 = 110;

    #[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
    const SYS_FUTEX: CLong = 202;
    #[cfg(all(target_arch = "x86_64", target_pointer_width = "32"))]
    const SYS_FUTEX: CLong = 0x4000_0000 + 202;
    #[cfg(any(target_arch = "x86", target_arch = "arm"))]
    const SYS_FUTEX: CLong = 240;
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "csky",
        target_arch = "hexagon",
        target_arch = "loongarch64",
        target_arch = "riscv64"
    ))]
    const SYS_FUTEX: CLong = 98;
    #[cfg(any(target_arch = "m68k", target_arch = "riscv32"))]
    const SYS_FUTEX: CLong = 422;
    #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
    const SYS_FUTEX: CLong = 221;
    #[cfg(target_arch = "s390x")]
    const SYS_FUTEX: CLong = 238;
    #[cfg(any(target_arch = "sparc", target_arch = "sparc64"))]
    const SYS_FUTEX: CLong = 142;
    #[cfg(target_arch = "mips")]
    const SYS_FUTEX: CLong = 4_238;
    #[cfg(target_arch = "mips64")]
    const SYS_FUTEX: CLong = 5_194;

    unsafe extern "C" {
        fn syscall(number: CLong, ...) -> CLong;
        fn __errno_location() -> *mut i32;
    }

    fn relative_timespec(ns: u64) -> Timespec {
        let seconds = ns / 1_000_000_000;
        Timespec {
            tv_sec: seconds.min(TimeUnit::MAX as u64) as TimeUnit,
            tv_nsec: (ns % 1_000_000_000) as TimeUnit,
        }
    }

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>) -> bool {
        let timeout = timeout_ns.map(relative_timespec);
        let timeout_ptr = timeout
            .as_ref()
            .map_or(ptr::null(), |value| value as *const Timespec);
        let result = unsafe {
            syscall(
                SYS_FUTEX,
                atomic as *const AtomicU32,
                FUTEX_WAIT | FUTEX_PRIVATE_FLAG,
                expected,
                timeout_ptr,
                ptr::null::<c_void>(),
                0_u32,
            )
        };
        result >= 0 || unsafe { *__errno_location() } != ETIMEDOUT
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool) -> Option<u32> {
        let count = if all { i32::MAX } else { 1 };
        let result = unsafe {
            syscall(
                SYS_FUTEX,
                atomic as *const AtomicU32,
                FUTEX_WAKE | FUTEX_PRIVATE_FLAG,
                count,
            )
        };
        if result >= 0 {
            Some(result as u32)
        } else {
            // A valid AtomicU32 is aligned, mapped, and uses a valid operation;
            // keep the portable Option contract if the kernel still rejects it.
            None
        }
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::AtomicU32;
    use core::ffi::c_void;

    const UL_COMPARE_AND_WAIT: u32 = 1;
    const ULF_WAKE_ALL: u32 = 0x0000_0100;
    const ULF_NO_ERRNO: u32 = 0x0100_0000;
    const ETIMEDOUT: i32 = 60;

    // Private SPI in libSystem. ULF_NO_ERRNO makes failures negative errno
    // values, avoiding platform errno access and preserving no_std support.
    unsafe extern "C" {
        fn __ulock_wait2(
            operation: u32,
            address: *mut c_void,
            value: u64,
            timeout_ns: u64,
            value2: u64,
        ) -> i32;
        fn __ulock_wake(operation: u32, address: *mut c_void, wake_value: u64) -> i32;
    }

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>) -> bool {
        let result = unsafe {
            __ulock_wait2(
                UL_COMPARE_AND_WAIT | ULF_NO_ERRNO,
                atomic as *const AtomicU32 as *mut c_void,
                expected as u64,
                timeout_ns.unwrap_or(0),
                0,
            )
        };
        result != -ETIMEDOUT
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool) -> Option<u32> {
        let operation = UL_COMPARE_AND_WAIT | ULF_NO_ERRNO | if all { ULF_WAKE_ALL } else { 0 };
        unsafe {
            let _ = __ulock_wake(operation, atomic as *const AtomicU32 as *mut c_void, 0);
        }
        None
    }
}

#[cfg(target_os = "freebsd")]
mod imp {
    use super::AtomicU32;
    use core::{ffi::c_void, ptr};

    #[cfg(target_arch = "x86")]
    type TimeT = i32;
    #[cfg(not(target_arch = "x86"))]
    type TimeT = i64;
    type CLong = isize;

    #[repr(C)]
    struct Timespec {
        tv_sec: TimeT,
        tv_nsec: CLong,
    }

    const UMTX_OP_WAIT_UINT_PRIVATE: i32 = 15;
    const UMTX_OP_WAKE_PRIVATE: i32 = 16;
    const ETIMEDOUT: i32 = 60;

    unsafe extern "C" {
        fn _umtx_op(
            object: *mut c_void,
            operation: i32,
            value: usize,
            address: *mut c_void,
            address2: *mut c_void,
        ) -> i32;
        fn __error() -> *mut i32;
    }

    fn relative_timespec(ns: u64) -> Timespec {
        let seconds = ns / 1_000_000_000;
        Timespec {
            tv_sec: seconds.min(TimeT::MAX as u64) as TimeT,
            tv_nsec: (ns % 1_000_000_000) as CLong,
        }
    }

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>) -> bool {
        let timeout = timeout_ns.map(relative_timespec);
        let (size, timeout_ptr) =
            timeout
                .as_ref()
                .map_or((ptr::null_mut(), ptr::null_mut()), |value| {
                    (
                        ptr::without_provenance_mut(size_of::<Timespec>()),
                        value as *const Timespec as *mut c_void,
                    )
                });
        let result = unsafe {
            _umtx_op(
                atomic as *const AtomicU32 as *mut c_void,
                UMTX_OP_WAIT_UINT_PRIVATE,
                expected as usize,
                size,
                timeout_ptr,
            )
        };
        result >= 0 || unsafe { *__error() } != ETIMEDOUT
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool) -> Option<u32> {
        let count = if all { i32::MAX as usize } else { 1 };
        unsafe {
            let _ = _umtx_op(
                atomic as *const AtomicU32 as *mut c_void,
                UMTX_OP_WAKE_PRIVATE,
                count,
                ptr::null_mut(),
                ptr::null_mut(),
            );
        }
        None
    }
}

#[cfg(all(target_family = "wasm", target_feature = "atomics"))]
mod imp {
    use super::AtomicU32;

    #[cfg(target_arch = "wasm32")]
    use core::arch::wasm32 as wasm;
    #[cfg(target_arch = "wasm64")]
    use core::arch::wasm64 as wasm;

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>) -> bool {
        let timeout = timeout_ns
            .and_then(|ns| i64::try_from(ns).ok())
            .unwrap_or(-1);
        unsafe {
            wasm::memory_atomic_wait32(
                atomic as *const AtomicU32 as *mut i32,
                expected as i32,
                timeout,
            ) != 2
        }
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool) -> Option<u32> {
        let count = if all { u32::MAX } else { 1 };
        Some(unsafe { wasm::memory_atomic_notify(atomic as *const AtomicU32 as *mut i32, count) })
    }
}

#[cfg(not(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    target_os = "freebsd",
    all(target_family = "wasm", target_feature = "atomics")
)))]
compile_error!(
    "futex.rs supports Windows, Linux, macOS, FreeBSD, and Wasm with the atomics target feature"
);

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;
    use std::{sync::Arc, thread};

    #[test]
    fn mismatched_wait_returns_immediately() {
        let atomic = AtomicU32::new(7);
        wait(&atomic, 6);
        assert!(wait_timeout(&atomic, 6, 1));
    }

    #[test]
    fn zero_timeout_does_not_block() {
        let atomic = AtomicU32::new(7);
        assert!(!wait_timeout(&atomic, 7, 0));
    }

    #[test]
    fn wake_releases_or_races_with_waiter() {
        let atomic = Arc::new(AtomicU32::new(0));
        let ready = Arc::new(AtomicU32::new(0));
        let worker_atomic = Arc::clone(&atomic);
        let worker_ready = Arc::clone(&ready);

        let worker = thread::spawn(move || {
            worker_ready.store(1, Ordering::Release);
            wait(&worker_atomic, 0);
            assert_eq!(worker_atomic.load(Ordering::Acquire), 1);
        });

        while ready.load(Ordering::Acquire) == 0 {
            core::hint::spin_loop();
        }
        atomic.store(1, Ordering::Release);
        let _ = wake_one(&atomic);
        worker.join().unwrap();
    }
}
