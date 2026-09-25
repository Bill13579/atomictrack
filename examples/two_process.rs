// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

//! Cross-process waiting between two processes on Windows, where
//! the parent creates a named file-mapping object holding a waiting pool
//! and an `AtomicTrack`, then spawns itself as the child,
//! which maps the same section at a *different* address.
//!
//! Each side wakes a wait in the other and reports the latency.
//!
//! `cargo run --release --example two_process` uses a non-zero `waiting_pool_join`, thus sharing a waiting pool, while
//! `cargo run --release --example two_process no_join` uses `WAITING_POOL_JOIN_NONE`, where wakes do not cross processes and thus hit their timeouts.

#[cfg(not(target_os = "windows"))]
fn main() {
    eprintln!("this example only runs on Windows");
}

#[cfg(target_os = "windows")]
fn main() {
    windows::main();
}

#[cfg(target_os = "windows")]
mod windows {
    use std::{
        env,
        ffi::c_void,
        process::Command,
        ptr::{self, NonNull},
        sync::atomic::{AtomicU32, AtomicU64, Ordering::SeqCst},
        thread,
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    use atomictrack::{
        AtomicTrack,
        waiting::{A_T_set_waiting_pool, AtomicTrackWaiting, WAITING_POOL_JOIN_NONE},
    };

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(file: *mut c_void, attributes: *const c_void, protect: u32, size_high: u32, size_low: u32, name: *const u16) -> *mut c_void;
        fn OpenFileMappingW(access: u32, inherit: i32, name: *const u16) -> *mut c_void;
        fn MapViewOfFile(mapping: *mut c_void, access: u32, offset_high: u32, offset_low: u32, bytes: usize) -> *mut c_void;
    }

    const PAGE_READWRITE: u32 = 0x04;
    const FILE_MAP_ALL_ACCESS: u32 = 0xF001F;

    const SIZE: usize = 64 * 1024;
    const POOL_OFFSET: usize = 0;
    const POOL_LEN: usize = 1024;
    const TRACK_OFFSET: usize = 4096;
    const CAPACITY: usize = 16;
    const STAMP_OFFSET: usize = 16 * 1024;

    const ID_A: u64 = 101;
    const ID_B: u64 = 202;
    const DELAY: Duration = Duration::from_millis(300);

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    fn now_ns() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    }

    fn map(mapping: *mut c_void) -> NonNull<u8> {
        NonNull::new(unsafe { MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, SIZE) }.cast::<u8>()).expect("MapViewOfFile failed")
    }

    struct Attached {
        handle: AtomicTrackWaiting,
        stamp: &'static AtomicU64,
    }

    fn attach(base: NonNull<u8>, join: u64, create: bool) -> Attached {
        unsafe {
            let pool = base.as_ptr().add(POOL_OFFSET).cast::<AtomicU32>();
            assert_eq!(A_T_set_waiting_pool(pool, POOL_LEN, join), 0, "A_T_set_waiting_pool failed");
            let track_ptr = NonNull::new_unchecked(base.as_ptr().add(TRACK_OFFSET));
            assert!(TRACK_OFFSET + AtomicTrack::layout(CAPACITY).unwrap().size() <= STAMP_OFFSET);
            let track: &'static AtomicTrack = if create {
                AtomicTrack::init_in_place_default(track_ptr, CAPACITY)
            } else {
                AtomicTrack::from_raw_default(track_ptr, CAPACITY)
            };
            Attached {
                handle: AtomicTrackWaiting::from_static(track),
                stamp: &*base.as_ptr().add(STAMP_OFFSET).cast::<AtomicU64>(),
            }
        }
    }

    fn latency_us(stamp: &AtomicU64) -> u64 {
        (now_ns().saturating_sub(stamp.load(SeqCst))) / 1_000
    }

    fn timeout_ns(join: u64) -> u64 {
        if join == WAITING_POOL_JOIN_NONE { 2_000_000_000 } else { 10_000_000_000 }
    }

    pub(super) fn main() {
        let args: Vec<String> = env::args().collect();
        match args.get(1).map(String::as_str) {
            Some("child") => child(&args[2], args[3].parse().unwrap()),
            mode => parent(mode == Some("no_join")),
        }
    }

    fn parent(no_join: bool) {
        let pid = std::process::id();
        let join = if no_join { WAITING_POOL_JOIN_NONE } else { 0x5EED_0000_0000_0000 | pid as u64 };
        let name = format!("Local\\atomictrack-two-process-{pid}");
        let mapping = unsafe { CreateFileMappingW(ptr::without_provenance_mut(usize::MAX), ptr::null(), PAGE_READWRITE, 0, SIZE as u32, wide(&name).as_ptr()) };
        assert!(!mapping.is_null(), "CreateFileMappingW failed");
        let base = map(mapping);
        let shared = attach(base, join, true);
        println!("[parent] pid {pid}, section mapped at {base:p}, waiting_pool_join {join:#x}{}", if no_join { " (no_join: NONE)" } else { "" });

        let mut child = Command::new(env::current_exe().unwrap())
            .args(["child", &name, &join.to_string()])
            .spawn()
            .unwrap();

        thread::sleep(DELAY);
        shared.stamp.store(now_ns(), SeqCst);
        let a = shared.handle.enter(ID_A).unwrap();

        thread::sleep(DELAY);
        shared.stamp.store(now_ns(), SeqCst);
        shared.handle.number(a).unwrap().raise_to(10).unwrap();

        let result = shared.handle.wait_gte_timeout(ID_B, 7, timeout_ns(join));
        println!("[parent] step 3: key wait_gte(ID_B, 7) woken by the child's raise_to -> {result:?}, wake latency {} us", latency_us(shared.stamp));

        let status = child.wait().unwrap();
        println!("[parent] child exited with {status}");
    }

    fn child(name: &str, join: u64) {
        let pid = std::process::id();
        let spacer = unsafe { CreateFileMappingW(ptr::without_provenance_mut(usize::MAX), ptr::null(), PAGE_READWRITE, 0, 4 << 20, ptr::null()) };
        let _spacer_view = unsafe { MapViewOfFile(spacer, FILE_MAP_ALL_ACCESS, 0, 0, 4 << 20) };

        let mapping = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, wide(name).as_ptr()) };
        assert!(!mapping.is_null(), "OpenFileMappingW failed");
        let base = map(mapping);
        let shared = attach(base, join, false);
        println!("[child]  pid {pid}, section mapped at {base:p}");

        let started = Instant::now();
        let a = shared.handle.wait_for_timeout(ID_A, timeout_ns(join));
        println!("[child]  step 1: wait_for(ID_A) woken by the parent's enter -> {a:?} after {:?}, wake latency {} us", started.elapsed(), latency_us(shared.stamp));
        let a = a.unwrap();

        let started = Instant::now();
        let reached = shared.handle.number(a).unwrap().wait_gte_timeout(10, timeout_ns(join));
        println!("[child]  step 2: number wait_gte(10) woken by the parent's raise_to -> {reached:?} after {:?}, wake latency {} us", started.elapsed(), latency_us(shared.stamp));

        let b = shared.handle.enter(ID_B).unwrap();
        thread::sleep(DELAY);
        shared.stamp.store(now_ns(), SeqCst);
        shared.handle.number(b).unwrap().raise_to(7).unwrap();
    }
}
