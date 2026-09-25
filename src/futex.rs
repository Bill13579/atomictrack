// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org).

//! Small, dependency-free atomic wait/wake helper module.
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
//! always re-check the condition.
//!
//! When `pinfo` is not zero, it is a packed value consisting of:
//! - `upper bits`: u64 value which uniquely identifies the waiting (atomics) pool
//!   that the atomic belongs to *across processes* (any unique and shared random number will do), and
//! - `lower bits`: u64 value representing the *base address* where the first atomic
//!   in that pool is mapped *within this process*. So `*const AtomicU32 - pinfo_lower_bits = offset of the atomic within the waiting pool it belongs to`.
//!
//! This value, should it be non-zero, will then be used to correlate and enable cross-process waiting
//! on Windows. On every other platform, the exact value of `pinfo` does not matter, but if possible, still try to provide the value as described.
//! Private atomic waits that need not be cross-process can have `pinfo` set to `0`, which will do a regular private futex wait/wake.
//!
//! # Platform notes
//!
//! - Windows requires Windows 8 / Server 2012 or newer. Finite nanosecond
//!   timeouts are rounded up to milliseconds and capped at `u32::MAX - 1`
//!   milliseconds (`u32::MAX` means infinite to `WaitOnAddress`).
//! - On Windows, a non-zero `pinfo` requires the `std` feature (it panics otherwise). Each atomic gets a named
//!   manual-reset event `Local\{pool:016x}futex{offset / 4}`, opened lazily and cached for the life of the process,
//!   so processes must be in the same session to see each other. Waits sleep in time slices of `ATOMICTRACK_WIN_SHARED_WAIT_TIME_SLICE_MS` milliseconds
//!   (default 20, rounded up to the system timer tick). The atomic is rechecked between each time slice, so the
//!   maximum latency from a missed wake is capped. Every `wake_all` with a non-zero `pinfo` uses two system calls, even with no waiters.
//! - macOS uses the private `__ulock_wait2` API. This can make an application
//!   unsuitable for App Store distribution. `__ulock_wait2` is available on
//!   macOS 10.15 and newer.
//! - Wasm requires the `atomics` target feature and nightly Rust's
//!   `stdarch_wasm_atomic_wait` feature.

use core::sync::atomic::{AtomicU32, Ordering};

/// Wait while `atomic` equals `expected`.
///
/// Returns immediately if the value is already different.
#[inline]
pub fn wait(atomic: &AtomicU32, expected: u32, pinfo: u128) {
    if atomic.load(Ordering::Relaxed) != expected {
        return;
    }
    let _ = imp::wait(atomic, expected, None, pinfo);
}

/// Wait while `atomic` equals `expected`, for at most `timeout_ns` nanoseconds.
///
/// Returns `false` only when the timeout has elapsed.
/// Returns `true` after a wake, a value mismatch, an interruption, or
/// another spurious return. Make sure to re-check the atomic value in any case.
///
/// A timeout of zero does not block. On Windows, positive values are rounded
/// up to the next whole millisecond, while other platforms retain nanosecond input.
#[inline]
#[must_use]
pub fn wait_timeout(atomic: &AtomicU32, expected: u32, timeout_ns: u64, pinfo: u128) -> bool {
    if atomic.load(Ordering::Relaxed) != expected {
        return true;
    }
    if timeout_ns == 0 {
        return false;
    }
    imp::wait(atomic, expected, Some(timeout_ns), pinfo)
}

/// Wake at most one waiter sleeping on `atomic`.
///
/// Returns `Some(number_woken)` on Linux and Wasm, and `None` elsewhere.
#[inline]
#[must_use]
pub fn wake_one(atomic: &AtomicU32) -> Option<u32> {
    imp::wake(atomic, false, 0)
}

/// Wake all waiters sleeping on `atomic`.
///
/// Returns `Some(number_woken)` on Linux and Wasm, and `None` elsewhere.
#[inline]
#[must_use]
pub fn wake_all(atomic: &AtomicU32, pinfo: u128) -> Option<u32> {
    imp::wake(atomic, true, pinfo)
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

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>, pinfo: u128) -> bool {
        if pinfo != 0 {
            return shared::wait(atomic, expected, timeout_ns, pinfo);
        }
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

    pub(super) fn wake(atomic: &AtomicU32, all: bool, pinfo: u128) -> Option<u32> {
        if pinfo != 0 {
            shared::wake(atomic, pinfo);
            return None;
        }
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

    #[cfg(not(feature = "std"))]
    mod shared {
        use super::AtomicU32;

        pub(super) fn wait(_atomic: &AtomicU32, _expected: u32, _timeout_ns: Option<u64>, _pinfo: u128) -> bool {
            panic!("cross-process waiting (waiting with non-zero pinfo) on Windows requires the `std` feature");
        }

        pub(super) fn wake(_atomic: &AtomicU32, _pinfo: u128) {
            panic!("cross-process waiting (waiting with non-zero pinfo) on Windows requires the `std` feature");
        }
    }

    #[cfg(feature = "std")]
    pub(super) mod shared {
        extern crate std;

        use super::AtomicU32;
        use core::{ffi::c_void, ptr, sync::atomic::Ordering};
        use std::{
            collections::{BTreeMap, btree_map::Entry},
            format,
            sync::{PoisonError, RwLock},
            thread,
            time::{Duration, Instant},
            vec::Vec,
        };

        const WAIT_SLICE_MS: u32 = crate::env_or_default!("ATOMICTRACK_WIN_SHARED_WAIT_TIME_SLICE_MS", "20", u32);
        const _: () = assert!(WAIT_SLICE_MS > 0 && WAIT_SLICE_MS < u32::MAX, "ATOMICTRACK_WIN_SHARED_WAIT_TIME_SLICE_MS should be within range 1..u32::MAX");

        const WAIT_OBJECT_0: u32 = 0;
        const WAIT_TIMEOUT: u32 = 0x102;

        #[cfg_attr(
            target_arch = "x86",
            link(name = "kernel32", kind = "raw-dylib", import_name_type = "undecorated")
        )]
        #[cfg_attr(not(target_arch = "x86"), link(name = "kernel32", kind = "raw-dylib"))]
        unsafe extern "system" {
            fn CreateEventW(attributes: *const c_void, manual_reset: i32, initial_state: i32, name: *const u16) -> *mut c_void;
            fn SetEvent(event: *mut c_void) -> i32;
            fn ResetEvent(event: *mut c_void) -> i32;
            fn WaitForSingleObject(handle: *mut c_void, timeout_ms: u32) -> u32;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }

        /// Manual reset events
        static EVENTS: RwLock<BTreeMap<(u64, u64), usize>> = RwLock::new(BTreeMap::new());

        pub(crate) fn key(atomic: &AtomicU32, pinfo: u128) -> (u64, u64) {
            let base = pinfo as u64;
            let address = (atomic as *const AtomicU32).addr() as u64;
            debug_assert!(address >= base, "the atomic must be *at* or *after* the base address specified in pinfo");
            ((pinfo >> 64) as u64, address.wrapping_sub(base) / size_of::<AtomicU32>() as u64)
        }

        pub(crate) fn name((pool, index): (u64, u64)) -> Vec<u16> {
            format!("Local\\{pool:016x}futex{index}").encode_utf16().chain([0]).collect()
        }

        fn event(key: (u64, u64)) -> Option<*mut c_void> {
            if let Some(&handle) = EVENTS.read().unwrap_or_else(PoisonError::into_inner).get(&key) {
                return Some(ptr::with_exposed_provenance_mut(handle));
            }
            let name = name(key);
            let created = unsafe { CreateEventW(ptr::null(), 1, 0, name.as_ptr()) };
            if created.is_null() {
                return None;
            }
            match EVENTS.write().unwrap_or_else(PoisonError::into_inner).entry(key) {
                Entry::Occupied(existing) => {
                    unsafe { CloseHandle(created) };
                    Some(ptr::with_exposed_provenance_mut(*existing.get()))
                },
                Entry::Vacant(slot) => {
                    slot.insert(created.expose_provenance());
                    Some(created)
                },
            }
        }

        pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>, pinfo: u128) -> bool {
            let event = event(key(atomic, pinfo));
            let deadline = timeout_ns.and_then(|ns| Instant::now().checked_add(Duration::from_nanos(ns)));
            loop {
                if atomic.load(Ordering::Acquire) != expected {
                    return true;
                }
                let time_slice_ms = match deadline {
                    None => WAIT_SLICE_MS,
                    Some(deadline) => {
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            return false;
                        }
                        remaining.as_nanos().div_ceil(1_000_000).min(WAIT_SLICE_MS as u128) as u32
                    },
                };
                match event.map(|event| unsafe { WaitForSingleObject(event, time_slice_ms) }) {
                    Some(WAIT_OBJECT_0) => return true,
                    Some(WAIT_TIMEOUT) => {},
                    _ => thread::sleep(Duration::from_millis(time_slice_ms as u64)),
                }
            }
        }

        pub(super) fn wake(atomic: &AtomicU32, pinfo: u128) {
            if let Some(event) = event(key(atomic, pinfo)) {
                unsafe {
                    SetEvent(event);
                    ResetEvent(event);
                }
            }
        }
    }
}

#[cfg(all(target_os = "linux", not(target_family = "wasm")))]
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

    const fn private_flag(pinfo: u128) -> i32 {
        if pinfo == 0 { FUTEX_PRIVATE_FLAG } else { 0 }
    }

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>, pinfo: u128) -> bool {
        let timeout = timeout_ns.map(relative_timespec);
        let timeout_ptr = timeout
            .as_ref()
            .map_or(ptr::null(), |value| value as *const Timespec);
        let result = unsafe {
            syscall(
                SYS_FUTEX,
                atomic as *const AtomicU32,
                FUTEX_WAIT | private_flag(pinfo),
                expected,
                timeout_ptr,
                ptr::null::<c_void>(),
                0_u32,
            )
        };
        result >= 0 || unsafe { *__errno_location() } != ETIMEDOUT
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool, pinfo: u128) -> Option<u32> {
        let count = if all { i32::MAX } else { 1 };
        let result = unsafe {
            syscall(
                SYS_FUTEX,
                atomic as *const AtomicU32,
                FUTEX_WAKE | private_flag(pinfo),
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
    const UL_COMPARE_AND_WAIT_SHARED: u32 = 3;
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

    const fn compare_and_wait(pinfo: u128) -> u32 {
        if pinfo == 0 { UL_COMPARE_AND_WAIT } else { UL_COMPARE_AND_WAIT_SHARED }
    }

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>, pinfo: u128) -> bool {
        let result = unsafe {
            __ulock_wait2(
                compare_and_wait(pinfo) | ULF_NO_ERRNO,
                atomic as *const AtomicU32 as *mut c_void,
                expected as u64,
                timeout_ns.unwrap_or(0),
                0,
            )
        };
        result != -ETIMEDOUT
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool, pinfo: u128) -> Option<u32> {
        let operation = compare_and_wait(pinfo) | ULF_NO_ERRNO | if all { ULF_WAKE_ALL } else { 0 };
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

    const UMTX_OP_WAKE: i32 = 3;
    const UMTX_OP_WAIT_UINT: i32 = 11;
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

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>, pinfo: u128) -> bool {
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
                if pinfo == 0 { UMTX_OP_WAIT_UINT_PRIVATE } else { UMTX_OP_WAIT_UINT },
                expected as usize,
                size,
                timeout_ptr,
            )
        };
        result >= 0 || unsafe { *__error() } != ETIMEDOUT
    }

    pub(super) fn wake(atomic: &AtomicU32, all: bool, pinfo: u128) -> Option<u32> {
        let count = if all { i32::MAX as usize } else { 1 };
        unsafe {
            let _ = _umtx_op(
                atomic as *const AtomicU32 as *mut c_void,
                if pinfo == 0 { UMTX_OP_WAKE_PRIVATE } else { UMTX_OP_WAKE },
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

    pub(super) fn wait(atomic: &AtomicU32, expected: u32, timeout_ns: Option<u64>, _pinfo: u128) -> bool {
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

    pub(super) fn wake(atomic: &AtomicU32, all: bool, _pinfo: u128) -> Option<u32> {
        let count = if all { u32::MAX } else { 1 };
        Some(unsafe { wasm::memory_atomic_notify(atomic as *const AtomicU32 as *mut i32, count) })
    }
}

#[cfg(not(any(
    target_os = "windows",
    all(target_os = "linux", not(target_family = "wasm")),
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
        wait(&atomic, 6, 0);
        assert!(wait_timeout(&atomic, 6, 1, 0));
    }

    #[test]
    fn zero_timeout_does_not_block() {
        let atomic = AtomicU32::new(7);
        assert!(!wait_timeout(&atomic, 7, 0, 0));
    }

    #[test]
    fn nonzero_timeout_elapses() {
        let atomic = AtomicU32::new(7);
        assert!((0..5).any(|_| !wait_timeout(&atomic, 7, 1_000_000, 0)));
    }

    fn wake_releases_or_races_with_waiter(pinfo: u128, wake: fn(&AtomicU32) -> Option<u32>) {
        let atomic = Arc::new(AtomicU32::new(0));
        let ready = Arc::new(AtomicU32::new(0));
        let worker_atomic = Arc::clone(&atomic);
        let worker_ready = Arc::clone(&ready);

        let worker = thread::spawn(move || {
            worker_ready.store(1, Ordering::Release);
            wait(&worker_atomic, 0, pinfo);
            assert_eq!(worker_atomic.load(Ordering::Acquire), 1);
        });

        while ready.load(Ordering::Acquire) == 0 {
            core::hint::spin_loop();
        }
        atomic.store(1, Ordering::Release);
        let _ = wake(&atomic);
        worker.join().unwrap();
    }

    #[test]
    fn wake_one_releases_or_races_with_waiter() {
        wake_releases_or_races_with_waiter(0, wake_one);
    }

    #[test]
    fn wake_all_releases_or_races_with_waiter() {
        wake_releases_or_races_with_waiter(0, |atomic| wake_all(atomic, 0));
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn shared_wake_all_releases_or_races_with_shared_waiter() {
        wake_releases_or_races_with_waiter(1, |atomic| wake_all(atomic, 1));
        wake_releases_or_races_with_waiter(u128::MAX, |atomic| wake_all(atomic, u128::MAX));
    }

    #[cfg(all(target_os = "windows", feature = "std"))]
    mod windows_shared {
        use super::*;
        use core::ffi::c_void;
        use std::{sync::mpsc, time::{Duration, Instant}};

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CreateFileMappingW(file: *mut c_void, attributes: *const c_void, protect: u32, size_high: u32, size_low: u32, name: *const u16) -> *mut c_void;
            fn MapViewOfFile(mapping: *mut c_void, access: u32, offset_high: u32, offset_low: u32, bytes: usize) -> *mut c_void;
            fn UnmapViewOfFile(base: *const c_void) -> i32;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }

        const PAGE_READWRITE: u32 = 0x04;
        const FILE_MAP_ALL_ACCESS: u32 = 0xF001F;
        const SIZE: usize = 4096;

        struct TwoViews {
            mapping: *mut c_void,
            a: usize,
            b: usize,
            pool: u64,
        }

        impl TwoViews {
            fn new(test: u64) -> Self {
                unsafe {
                    let mapping = CreateFileMappingW(ptr::without_provenance_mut(usize::MAX), ptr::null(), PAGE_READWRITE, 0, SIZE as u32, ptr::null());
                    assert!(!mapping.is_null());
                    let a = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, SIZE);
                    let b = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, SIZE);
                    assert!(!a.is_null() && !b.is_null() && a != b);
                    Self {
                        mapping,
                        a: a.expose_provenance(),
                        b: b.expose_provenance(),
                        pool: ((std::process::id() as u64) << 32) | test,
                    }
                }
            }

            fn pinfo_a(&self) -> u128 {
                ((self.pool as u128) << 64) | self.a as u128
            }

            fn pinfo_b(&self) -> u128 {
                ((self.pool as u128) << 64) | self.b as u128
            }

            fn atomic_a(&self, offset: usize) -> &'static AtomicU32 {
                unsafe { &*ptr::with_exposed_provenance::<AtomicU32>(self.a + offset) }
            }

            fn atomic_b(&self, offset: usize) -> &'static AtomicU32 {
                unsafe { &*ptr::with_exposed_provenance::<AtomicU32>(self.b + offset) }
            }
        }

        impl Drop for TwoViews {
            fn drop(&mut self) {
                unsafe {
                    UnmapViewOfFile(ptr::with_exposed_provenance(self.a));
                    UnmapViewOfFile(ptr::with_exposed_provenance(self.b));
                    CloseHandle(self.mapping);
                }
            }
        }

        use core::ptr;
        use super::super::imp::shared::{key, name};

        #[test]
        fn views_agree_on_the_event_and_pools_do_not() {
            let views = TwoViews::new(1);
            let a = views.atomic_a(8);
            let b = views.atomic_b(8);
            b.store(42, Ordering::Release);
            assert_eq!(a.load(Ordering::Acquire), 42);

            assert_eq!(key(a, views.pinfo_a()), (views.pool, 2));
            assert_eq!(key(a, views.pinfo_a()), key(b, views.pinfo_b()));
            assert_ne!(key(a, views.pinfo_a()), key(views.atomic_a(12), views.pinfo_a()));
            assert_ne!(key(a, views.pinfo_a()), key(b, views.pinfo_b() ^ (1 << 64)));

            let expected: std::vec::Vec<u16> = std::format!("Local\\{:016x}futex2", views.pool).encode_utf16().chain([0]).collect();
            assert_eq!(name(key(a, views.pinfo_a())), expected);
        }

        #[test]
        fn wake_through_another_view_releases_a_blocking_wait_without_a_value_change() {
            let views = TwoViews::new(2);
            let (a, pinfo_a) = (views.atomic_a(16), views.pinfo_a());
            let (b, pinfo_b) = (views.atomic_b(16), views.pinfo_b());
            let (sender, receiver) = mpsc::channel();
            let worker = thread::spawn(move || {
                wait(a, 0, pinfo_a);
                sender.send(()).unwrap();
            });

            thread::sleep(Duration::from_millis(50));
            let started = Instant::now();
            while receiver.try_recv().is_err() {
                assert!(started.elapsed() < Duration::from_secs(5), "shared wait was never released by a wake");
                let _ = wake_all(b, pinfo_b);
                thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(a.load(Ordering::Acquire), 0);
            worker.join().unwrap();
        }

        #[test]
        fn shared_wait_times_out_and_notices_unsignaled_changes() {
            let views = TwoViews::new(3);
            let (a, pinfo_a) = (views.atomic_a(0), views.pinfo_a());
            let b = views.atomic_b(0);

            let started = Instant::now();
            assert!(!wait_timeout(a, 0, 50_000_000, pinfo_a));
            assert!(started.elapsed() >= Duration::from_millis(50));
            assert!(wait_timeout(a, 1, 50_000_000, pinfo_a));

            let worker = thread::spawn(move || {
                let started = Instant::now();
                (wait_timeout(a, 0, 5_000_000_000, pinfo_a), started.elapsed())
            });
            thread::sleep(Duration::from_millis(30));
            b.store(1, Ordering::Release);
            let (woke, elapsed) = worker.join().unwrap();
            assert!(woke);
            assert!(elapsed < Duration::from_secs(1), "missed change took {elapsed:?}");
        }
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn shared_waits_time_out_and_skip_on_mismatch() {
        let atomic = AtomicU32::new(7);
        wait(&atomic, 6, 1);
        assert!(wait_timeout(&atomic, 6, 1, 1));
        assert!(!wait_timeout(&atomic, 7, 0, 1));
        assert!((0..5).any(|_| !wait_timeout(&atomic, 7, 1_000_000, 1)));
    }
}
