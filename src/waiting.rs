// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

//! A few things to note:
//! - All key-based (and not [`NumberId`] based) apis select and operate on ***only*** the first `Number` recovered from the ring with the provided id.
//! - `at_least` is not checked for the reserved MSB bit. Please don't use numbers that are that high.

//TODO: raise_to, add, and both leave methods notify even on errors or no-op updates. With wake_all and a shared futex pool, this can create wake storms. Change eventually if that becomes a problem.

extern crate std;
use std::time::Instant;

use core::{hash::{Hash, Hasher}, ptr::NonNull, sync::atomic::{AtomicU32, Ordering}};

use crate::{AtomicTrack, AtomicTrackInner, EMPTY_ID, EnterError, LeaveError, Number, NumberError, NumberId, futex, is_key_locked, is_suspended, math::{AtomicType, NumericType, gte_msb_masked}, without_suspended_bit};

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

const NUM_FUTEXES: usize = env_or_default!("ATOMICTRACK_FUTEX_POOL_SIZE", "1024", usize);
const _: () = {
    assert!(NUM_FUTEXES > 0, "FUTEX pool size must be something that isn't zero!");
};

const MAX_SPIN_LOOPS_BEFORE_SLEEP: usize = env_or_default!("ATOMICTRACK_MAX_SPIN_LOOPS_BEFORE_SLEEP", "10", usize);

static FUTEXES: [AtomicU32; NUM_FUTEXES] = [const { AtomicU32::new(0) }; NUM_FUTEXES];

#[cfg(target_pointer_width = "64")]
fn get_futex(a: &NonNull<AtomicTrackInner>, b: NumericType) -> &'static AtomicU32 {
    let mut hasher = hasher::RhmHasher::default();
    (a.as_ptr() as *const i64 as usize).hash(&mut hasher);
    b.hash(&mut hasher);
    &FUTEXES[hasher.finish() as usize % NUM_FUTEXES]
}

#[cfg(target_pointer_width = "32")]
fn get_futex(a: &NonNull<AtomicTrackInner>, b: NumericType) -> &'static AtomicU32 {
    let mut hasher = hasher::RhmHasher::default();
    (a.as_ptr() as *const i32 as usize).hash(&mut hasher);
    b.hash(&mut hasher);
    &FUTEXES[hasher.finish() as usize % NUM_FUTEXES]
}

pub struct AtomicTrackWaiting {
    inner: AtomicTrack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitError {
    InvalidId,
    NotFound,
}

pub struct NumberWaiting<'a, 'b> {
    inner: Number<'a>,
    atomic_track_waiting: &'b AtomicTrackWaiting,
}

impl Clone for AtomicTrackWaiting {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl AtomicTrackWaiting {
    pub fn new(capacity: usize) -> Self {
        Self { inner: AtomicTrack::new(capacity) }
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// [`find_min`](`AtomicTrackWaiting::find_min`) updates the current global min, this one just reads it.
    pub fn min(&self) -> NumericType {
        self.inner.min()
    }

    pub fn find_min(&self) -> NumericType {
        self.inner.find_min()
    }

    pub fn enter(&self, id: NumericType) -> Result<NumberId, EnterError> {
        self.enter_from(id, self.inner.min())
    }

    pub fn enter_from(&self, id: NumericType, at_least: NumericType) -> Result<NumberId, EnterError> {
        match self.inner.enter_from(id, at_least) {
            Ok(number_id) => {
                let f = get_futex(&self.inner.inner, number_id.id);
                f.fetch_add(1, Ordering::Release);
                let _ = futex::wake_all(f);
                Ok(number_id)
            },
            Err(e) => Err(e),
        }
    }

    /// Recover a [`NumberId`] from a key. This is usually fast but since it *can* scan the whole ring, keeping the [`NumberId`] directly is better.
    pub fn recover(&self, id: NumericType) -> Option<NumberId> {
        self.inner.recover(id)
    }

    /// Just like [`recover`](`AtomicTrackWaiting::recover`), this is usually fast but since it *can* scan the whole ring, using [`with_number`](`AtomicTrackWaiting::with_number`) is better if you can.
    pub fn with_id<R>(
        &self,
        id: NumericType,
        f: impl FnOnce(NumberWaiting<'_, '_>) -> R,
    ) -> Result<R, NumberError> {
        self.inner.with_id(id, |number| {
            f(NumberWaiting {
                inner: number,
                atomic_track_waiting: self,
            })
        })
    }

    pub fn with_number<R>(
        &self,
        number: NumberId,
        f: impl FnOnce(NumberWaiting<'_, '_>) -> R,
    ) -> Result<R, NumberError> {
        self.inner.with_number(number, |number| {
            f(NumberWaiting {
                inner: number,
                atomic_track_waiting: self,
            })
        })
    }

    pub fn number(&self, number: NumberId) -> Result<NumberWaiting<'_, '_>, NumberError> {
        self.inner.number(number).map(|number| NumberWaiting {
            inner: number,
            atomic_track_waiting: self,
        })
    }

    pub fn wait_for(&self, id: NumericType) -> Result<NumberId, WaitError> {
        self.__wait_for_enter_timeout(id, None)
    }

    pub fn wait_for_timeout(&self, id: NumericType, timeout_ns: u64) -> Result<NumberId, WaitError> {
        self.__wait_for_enter_timeout(id, Some((timeout_ns, Instant::now())))
    }

    pub fn wait_for_number(&self, id: NumericType) -> Result<NumberWaiting<'_, '_>, WaitError> {
        #[cfg(debug_assertions)]
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        match self.number(self.wait_for(id)?) {
            Ok(number_waiting) => Ok(number_waiting),
            // As long as the input ids have been checked for validity already somewhere before, number_ids from wait_for (`recover` underneath) are always valid (the id part is not zero or locked, the offset is within usize range and within bounds of the ring).
            Err(NumberError::InvalidId) | Err(NumberError::InvalidOffset) => unreachable!(),
            // get_slot_concurrent, and by extension `number` is just an indexing operation, they never return a NotFound error since NotFound checks additionally for whether the slot contains the actual id key.
            Err(NumberError::NotFound) => unreachable!(),
        }
    }

    pub fn wait_for_number_timeout(&self, id: NumericType, timeout_ns: u64) -> Result<NumberWaiting<'_, '_>, WaitError> {
        #[cfg(debug_assertions)]
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        match self.number(self.wait_for_timeout(id, timeout_ns)?) {
            Ok(number_waiting) => Ok(number_waiting),
            // As long as the input ids have been checked for validity already somewhere before, number_ids from wait_for (`recover` underneath) are always valid (the id part is not zero or locked, the offset is within usize range and within bounds of the ring).
            Err(NumberError::InvalidId) | Err(NumberError::InvalidOffset) => unreachable!(),
            // get_slot_concurrent, and by extension `number` is just an indexing operation, they never return a NotFound error since NotFound checks additionally for whether the slot contains the actual id key.
            Err(NumberError::NotFound) => unreachable!(),
        }
    }

    pub fn wait_gte(&self, id: NumericType, at_least: NumericType) -> Result<(NumericType, NumberId), WaitError> {
        match self.__wait_for_enter_and_at_least_timeout(id, at_least, None) {
            Ok((true, value, number_id)) => Ok((value, number_id)),
            Ok((false, _, _)) => unreachable!(),
            Err(e) => Err(e),
        }
    }

    pub fn wait_gte_timeout(&self, id: NumericType, at_least: NumericType, timeout_ns: u64) -> Result<(bool, NumericType, NumberId), WaitError> {
        self.__wait_for_enter_and_at_least_timeout(id, at_least, Some((timeout_ns, Instant::now())))
    }

    fn __wait_for_enter_timeout(&self, id: NumericType, timeout_ns: Option<(u64, Instant)>) -> Result<NumberId, WaitError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        let mut i = 0;
        let mut f = None;
        let mut futex_value_before = 0;
        let futex_getter = || get_futex(&self.inner.inner, id);
        loop {
            if i >= MAX_SPIN_LOOPS_BEFORE_SLEEP {
                futex_value_before = f.get_or_insert_with(&futex_getter).load(Ordering::Acquire); // Get the futex value before checking the number. Later on if the number is not gte at_least, we can load this value again, and if it has changed in between, we know that the number has changed as well (though spurious wakeups are possible).
            }

            if let Some(number_id) = self.inner.recover(id) {
                return Ok(number_id);
            }

            // If it's not, we do different things based on whether we've exhausted the number of spins we're willing to do.
            if i < MAX_SPIN_LOOPS_BEFORE_SLEEP {
                match &timeout_ns {
                    Some((timeout_ns, start)) => {
                        if start.elapsed().as_nanos() as u64 >= *timeout_ns {
                            return Err(WaitError::NotFound);
                        }
                    },
                    _ => {},
                }
                core::hint::spin_loop();
            } else {
                // Load the futex value again.
                let futex_value_after = f.get_or_insert_with(&futex_getter).load(Ordering::Acquire);
                if futex_value_before != futex_value_after {
                    // The futex value has changed, so the number has also possibly changed. We need to recheck.
                    continue;
                } else {
                    // Otherwise, we go to sleep.
                    match &timeout_ns {
                        Some((timeout_ns, start)) => {
                            let freeze = start.elapsed().as_nanos() as u64;
                            if freeze >= *timeout_ns {
                                return Err(WaitError::NotFound);
                            }
                            let _ = futex::wait_timeout(f.get_or_insert_with(&futex_getter), futex_value_after, timeout_ns - freeze);
                        },
                        _ => {
                            let _ = futex::wait(f.get_or_insert_with(&futex_getter), futex_value_after);
                        },
                    }
                }
            }

            i += 1;
        }
    }

    fn __wait_for_enter_and_at_least_timeout(&self, id: NumericType, at_least: NumericType, timeout_ns: Option<(u64, Instant)>) -> Result<(bool, NumericType, NumberId), WaitError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        let mut i = 0;
        let mut f = None;
        let mut futex_value_before = 0;
        let futex_getter = || get_futex(&self.inner.inner, id);
        let mut number = None;
        loop {
            let mut value = None;

            if i >= MAX_SPIN_LOOPS_BEFORE_SLEEP {
                futex_value_before = f.get_or_insert_with(&futex_getter).load(Ordering::Acquire); // Get the futex value before checking the number. Later on if the number is not gte at_least, we can load this value again, and if it has changed in between, we know that the number has changed as well (though spurious wakeups are possible).
            }

            if number.is_none() {
                if let Some(number_id) = self.inner.recover(id) {
                    //NOTE: This should never error since number_ids returned by recover should be valid (the input id itself has to be valid, which was checked earlier). "self.number" only indexes into the ring with the id and offset pair, it doesn't check whether that slot is actually occupied by the id specified in the number_id.
                    number = self.number(number_id).ok().map(|number_waiting| (number_waiting, number_id));
                }
            }

            if let Some((number, number_id)) = &number {
                let mut value_tmp;

                loop {
                    let key_before = number.inner.slot.id.load(Ordering::Acquire);
                    if key_before != number.inner.id {
                        return Err(WaitError::NotFound);
                    }

                    value_tmp = number.inner.slot.value.load(Ordering::Acquire);
                    let key_after = number.inner.slot.id.load(Ordering::Acquire);
                    if key_before != key_after {
                        core::hint::spin_loop();
                        continue;
                    }

                    break;
                }

                if is_suspended(value_tmp) {
                    return Err(WaitError::NotFound);
                }

                // Check if value is gte at_least.
                if gte_msb_masked(without_suspended_bit(value_tmp), at_least) {
                    return Ok((true, without_suspended_bit(value_tmp), number_id.clone()));
                }

                value = Some(value_tmp);
            }

            // If it's not, we do different things based on whether we've exhausted the number of spins we're willing to do.
            if i < MAX_SPIN_LOOPS_BEFORE_SLEEP {
                match &timeout_ns {
                    Some((timeout_ns, start)) => {
                        if start.elapsed().as_nanos() as u64 >= *timeout_ns {
                            match (value, &number) {
                                (Some(value), Some((_, number_id))) => {
                                    return Ok((false, without_suspended_bit(value), number_id.clone()));
                                },
                                _ => {},
                            }
                            return Err(WaitError::NotFound);
                        }
                    },
                    _ => {},
                }
                core::hint::spin_loop();
            } else {
                // Load the futex value again.
                let futex_value_after = f.get_or_insert_with(&futex_getter).load(Ordering::Acquire);
                if futex_value_before != futex_value_after {
                    // The futex value has changed, so the number has also possibly changed. We need to recheck.
                    continue;
                } else {
                    // Otherwise, we go to sleep.
                    match &timeout_ns {
                        Some((timeout_ns, start)) => {
                            let freeze = start.elapsed().as_nanos() as u64;
                            if freeze >= *timeout_ns {
                                match (value, &number) {
                                    (Some(value), Some((_, number_id))) => {
                                        return Ok((false, without_suspended_bit(value), number_id.clone()));
                                    },
                                    _ => {},
                                }
                                return Err(WaitError::NotFound);
                            }
                            let _ = futex::wait_timeout(f.get_or_insert_with(&futex_getter), futex_value_after, timeout_ns - freeze);
                        },
                        _ => {
                            let _ = futex::wait(f.get_or_insert_with(&futex_getter), futex_value_after);
                        },
                    }
                }
            }

            i += 1;
        }
    }

    pub fn leave(&self, number: NumberId) -> Result<(), LeaveError> {
        let result = self.inner.leave(number);
        let f = get_futex(&self.inner.inner, number.id);
        f.fetch_add(1, Ordering::Release);
        let _ = futex::wake_all(f);
        result
    }

    pub fn leave_concurrent(&self, number: NumberId) -> Result<(), LeaveError> {
        let result = self.inner.leave_concurrent(number);
        let f = get_futex(&self.inner.inner, number.id);
        f.fetch_add(1, Ordering::Release);
        let _ = futex::wake_all(f);
        result
    }
}

impl<'a, 'b> NumberWaiting<'a, 'b> {
    pub fn get(&self) -> Result<NumericType, NumberError> {
        self.inner.get()
    }

    pub fn raise_to(&self, at_least: NumericType) -> Result<NumericType, NumberError> {
        let result = self.inner.raise_to(at_least);
        self.signal_change();
        result
    }

    pub fn add(&self, delta: NumericType) -> Result<NumericType, NumberError> {
        let result = self.inner.add(delta);
        self.signal_change();
        result
    }

    pub fn signal_change(&self) {
        let f = get_futex(&self.atomic_track_waiting.inner.inner, self.inner.id);
        f.fetch_add(1, Ordering::Release);
        let _ = futex::wake_all(f);
    }

    pub fn wait_gte(&self, at_least: NumericType) -> Result<NumericType, WaitError> {
        match self.__wait_gte_timeout(at_least, None) {
            Ok((true, value)) => Ok(value),
            Ok((false, _)) => unreachable!(),
            Err(e) => Err(e),
        }
    }

    pub fn wait_gte_timeout(&self, at_least: NumericType, timeout_ns: u64) -> Result<(bool, NumericType), WaitError> {
        self.__wait_gte_timeout(at_least, Some((timeout_ns, Instant::now())))
    }

    fn __wait_gte_timeout(&self, at_least: NumericType, timeout_ns: Option<(u64, Instant)>) -> Result<(bool, NumericType), WaitError> {
        let mut i = 0;
        let mut f = None;
        let mut futex_value_before = 0;
        let futex_getter = || get_futex(&self.atomic_track_waiting.inner.inner, self.inner.id);
        loop {
            let mut value;

            if i >= MAX_SPIN_LOOPS_BEFORE_SLEEP {
                futex_value_before = f.get_or_insert_with(&futex_getter).load(Ordering::Acquire); // Get the futex value before checking the number. Later on if the number is not gte at_least, we can load this value again, and if it has changed in between, we know that the number has changed as well (though spurious wakeups are possible).
            }

            loop {
                let key_before = self.inner.slot.id.load(Ordering::Acquire);
                if key_before != self.inner.id {
                    return Err(WaitError::NotFound);
                }

                value = self.inner.slot.value.load(Ordering::Acquire);
                let key_after = self.inner.slot.id.load(Ordering::Acquire);
                if key_before != key_after {
                    core::hint::spin_loop();
                    continue;
                }

                break;
            }

            if is_suspended(value) {
                return Err(WaitError::NotFound);
            }

            // Check if value is gte at_least.
            if gte_msb_masked(without_suspended_bit(value), at_least) {
                return Ok((true, without_suspended_bit(value)));
            }

            // If it's not, we do different things based on whether we've exhausted the number of spins we're willing to do.
            if i < MAX_SPIN_LOOPS_BEFORE_SLEEP {
                match &timeout_ns {
                    Some((timeout_ns, start)) => {
                        if start.elapsed().as_nanos() as u64 >= *timeout_ns {
                            return Ok((false, without_suspended_bit(value)));
                        }
                    },
                    _ => {},
                }
                core::hint::spin_loop();
            } else {
                // Load the futex value again.
                let futex_value_after = f.get_or_insert_with(&futex_getter).load(Ordering::Acquire);
                if futex_value_before != futex_value_after {
                    // The futex value has changed, so the number has also possibly changed. We need to recheck.
                    continue;
                } else {
                    // Otherwise, we go to sleep.
                    match &timeout_ns {
                        Some((timeout_ns, start)) => {
                            let freeze = start.elapsed().as_nanos() as u64;
                            if freeze >= *timeout_ns {
                                return Ok((false, without_suspended_bit(value)));
                            }
                            let _ = futex::wait_timeout(f.get_or_insert_with(&futex_getter), futex_value_after, timeout_ns - freeze);
                        },
                        _ => {
                            let _ = futex::wait(f.get_or_insert_with(&futex_getter), futex_value_after);
                        },
                    }
                }
            }

            i += 1;
        }
    }

    /// # Safety
    /// Callers must preserve monotonicity and must not set the suspended bit.
    /// Callers must proactively call [`signal_change`](`NumberWaiting::signal_change`) after changing the number.
    pub unsafe fn atomic(&self) -> &AtomicType {
        unsafe { self.inner.atomic() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lock_key, math::MSB};
    use std::{
        sync::{Arc, Barrier},
        thread,
        time::Duration,
    };

    const TEST_TIMEOUT_NS: u64 = 2_000_000_000;

    #[test]
    fn recover_ignores_an_entry_that_is_still_locked() {
        let track = AtomicTrackWaiting::new(1);
        let inner = unsafe { track.inner.inner.as_ref() };
        let slot = &inner.slots[0];

        slot.id.store(lock_key(7), Ordering::Release);
        assert_eq!(track.recover(7), None);

        slot.value.store(0, Ordering::Release);
        slot.id.store(7, Ordering::Release);
        let number_id = NumberId { id: 7, offset: 0 };
        assert_eq!(track.recover(7), Some(number_id));
        track.leave(number_id).unwrap();
    }

    #[test]
    fn wait_for_validates_ids_times_out_and_wakes_on_enter() {
        let track = AtomicTrackWaiting::new(4);

        assert_eq!(track.wait_for_timeout(EMPTY_ID, 0), Err(WaitError::InvalidId));
        assert_eq!(track.wait_for_timeout(MSB, 0), Err(WaitError::InvalidId));
        assert_eq!(track.wait_for_timeout(7, 0), Err(WaitError::NotFound));

        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            waiter_barrier.wait();
            waiter_track.wait_for_timeout(7, TEST_TIMEOUT_NS)
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        let number_id = track.enter(7).unwrap();

        assert_eq!(waiter.join().unwrap(), Ok(number_id));
        assert_eq!(track.wait_for_timeout(7, 0), Ok(number_id));
    }

    #[test]
    fn key_wait_is_woken_when_the_number_advances() {
        let track = AtomicTrackWaiting::new(1);
        let number_id = track.enter(9).unwrap();
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            waiter_barrier.wait();
            waiter_track.wait_gte_timeout(9, 5, TEST_TIMEOUT_NS)
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        track.number(number_id).unwrap().add(5).unwrap();

        assert_eq!(waiter.join().unwrap(), Ok((true, 5, number_id)));
    }

    #[test]
    fn number_wait_is_woken_when_the_lane_leaves() {
        let track = AtomicTrackWaiting::new(1);
        let number_id = track.enter(11).unwrap();
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            let number = waiter_track.number(number_id).unwrap();
            waiter_barrier.wait();
            number.wait_gte_timeout(1, TEST_TIMEOUT_NS)
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        track.leave(number_id).unwrap();

        assert_eq!(waiter.join().unwrap(), Err(WaitError::NotFound));
    }

    #[test]
    fn timed_waits_return_the_last_observed_value() {
        let track = AtomicTrackWaiting::new(1);
        let number_id = track.enter_from(13, 3).unwrap();

        assert_eq!(
            track.wait_gte_timeout(13, 5, 0),
            Ok((false, 3, number_id))
        );
        assert_eq!(
            track.number(number_id).unwrap().wait_gte_timeout(5, 0),
            Ok((false, 3))
        );
    }

    #[test]
    fn key_wait_uses_only_the_first_recovered_placement() {
        let track = AtomicTrackWaiting::new(4);
        let hole = track.enter(1).unwrap();
        let later = track.enter(5).unwrap();
        track.number(later).unwrap().raise_to(22).unwrap();
        track.leave(hole).unwrap();

        let first = track.enter(5).unwrap();
        track.number(first).unwrap().raise_to(11).unwrap();

        assert_eq!(first.offset, 0);
        assert_eq!(later.offset, 1);
        assert_eq!(
            track.wait_gte_timeout(5, 20, 0),
            Ok((false, 11, first))
        );
    }

    #[test]
    fn threshold_waits_compare_values_across_wrap() {
        let track = AtomicTrackWaiting::new(1);
        let number_id = track.enter(17).unwrap();
        let number = track.number(number_id).unwrap();
        let quarter = NumericType::MAX / 4;

        for delta in [quarter, quarter, 1] {
            number.add(delta).unwrap();
        }

        let max_public = MSB - 1;
        assert_eq!(
            track.wait_gte_timeout(17, 1, 0),
            Ok((false, max_public, number_id))
        );
        assert_eq!(number.wait_gte_timeout(1, 0), Ok((false, max_public)));

        number.add(1).unwrap();
        assert_eq!(
            track.wait_gte_timeout(17, max_public, 0),
            Ok((true, 0, number_id))
        );
        assert_eq!(number.wait_gte_timeout(max_public, 0), Ok((true, 0)));
    }
}

mod hasher {
    // rapidhash V3 is Copyright (c) 2025 Nicolas De Carli and is used under the
    // MIT license. Its source is available at https://github.com/Nicoshev/rapidhash.

    //! Rustlang port of rapidhashMicro V3, Nicolas De Carli's hashing algorithm based upon wyhash by Wang Yi.

    extern crate alloc;

    #[allow(unused_imports)]
    use alloc::vec::Vec;

    use core::hash::Hasher;

    /// `write(b"hello"); write(b"world");` is not equivalent to `write(b"helloworld");`
    #[derive(Clone, Copy, Debug, Default)]
    pub(super) struct RhmHasher {
        state: u64,
    }

    #[allow(dead_code)]
    impl RhmHasher {
        pub const fn new() -> Self {
            Self { state: 0 }
        }

        pub const fn with_seed(seed: u64) -> Self {
            Self { state: seed }
        }
    }

    impl Hasher for RhmHasher {
        #[inline]
        fn finish(&self) -> u64 {
            self.state
        }

        #[inline]
        fn write(&mut self, bytes: &[u8]) {
            self.state = rapidhash_micro_with_seed(bytes, self.state);
        }
    }

    const RAPIDHASH_SECRET: [u64; 8] = [
        0x2d35_8dcc_aa6c_78a5,
        0x8bb8_4b93_962e_acc9,
        0x4b33_a62e_d433_d4a3,
        0x4d5a_2da5_1de1_aa47,
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x90ed_1765_281c_388c,
        0xaaaa_aaaa_aaaa_aaaa,
    ];

    #[inline(always)]
    fn rapid_multiply(a: u64, b: u64) -> (u64, u64) {
        let product = (a as u128) * (b as u128);
        (product as u64, (product >> 64) as u64)
    }

    #[inline(always)]
    fn rapid_mix(a: u64, b: u64) -> u64 {
        let (low, high) = rapid_multiply(a, b);
        low ^ high
    }

    #[inline(always)]
    fn rapid_read_64(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
            bytes[at + 4],
            bytes[at + 5],
            bytes[at + 6],
            bytes[at + 7],
        ])
    }

    #[inline(always)]
    fn rapid_read_32(bytes: &[u8], at: usize) -> u64 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as u64
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(super) fn rapidhash_micro(bytes: &[u8]) -> u64 {
        rapidhash_micro_with_seed(bytes, 0)
    }

    /// Hashes with rapidhashMicro V3.
    ///
    /// This is a small, non-cryptographic 64-bit hash. The output is identical on
    /// little- and big-endian targets.
    #[inline]
    pub(super) fn rapidhash_micro_with_seed(bytes: &[u8], mut seed: u64) -> u64 {
        let len = bytes.len();

        seed ^= rapid_mix(
            seed ^ RAPIDHASH_SECRET[2],
            RAPIDHASH_SECRET[1],
        );

        let mut a;
        let mut b;
        let mut remaining = len;
        let mut position = 0;

        if len <= 16 {
            if len >= 4 {
                seed ^= len as u64;
                if len >= 8 {
                    a = rapid_read_64(bytes, 0);
                    b = rapid_read_64(bytes, len - 8);
                } else {
                    a = rapid_read_32(bytes, 0);
                    b = rapid_read_32(bytes, len - 4);
                }
            } else if len != 0 {
                a = ((bytes[0] as u64) << 45) | bytes[len - 1] as u64;
                b = bytes[len >> 1] as u64;
            } else {
                a = 0;
                b = 0;
            }
        } else {
            if remaining > 80 {
                let mut see1 = seed;
                let mut see2 = seed;
                let mut see3 = seed;
                let mut see4 = seed;

                loop {
                    seed = rapid_mix(
                        rapid_read_64(bytes, position) ^ RAPIDHASH_SECRET[0],
                        rapid_read_64(bytes, position + 8) ^ seed,
                    );
                    see1 = rapid_mix(
                        rapid_read_64(bytes, position + 16) ^ RAPIDHASH_SECRET[1],
                        rapid_read_64(bytes, position + 24) ^ see1,
                    );
                    see2 = rapid_mix(
                        rapid_read_64(bytes, position + 32) ^ RAPIDHASH_SECRET[2],
                        rapid_read_64(bytes, position + 40) ^ see2,
                    );
                    see3 = rapid_mix(
                        rapid_read_64(bytes, position + 48) ^ RAPIDHASH_SECRET[3],
                        rapid_read_64(bytes, position + 56) ^ see3,
                    );
                    see4 = rapid_mix(
                        rapid_read_64(bytes, position + 64) ^ RAPIDHASH_SECRET[4],
                        rapid_read_64(bytes, position + 72) ^ see4,
                    );
                    position += 80;
                    remaining -= 80;

                    if remaining <= 80 {
                        break;
                    }
                }

                seed ^= see1;
                see2 ^= see3;
                seed ^= see4;
                seed ^= see2;
            }

            if remaining > 16 {
                seed = rapid_mix(
                    rapid_read_64(bytes, position) ^ RAPIDHASH_SECRET[2],
                    rapid_read_64(bytes, position + 8) ^ seed,
                );
                if remaining > 32 {
                    seed = rapid_mix(
                        rapid_read_64(bytes, position + 16) ^ RAPIDHASH_SECRET[2],
                        rapid_read_64(bytes, position + 24) ^ seed,
                    );
                    if remaining > 48 {
                        seed = rapid_mix(
                            rapid_read_64(bytes, position + 32) ^ RAPIDHASH_SECRET[1],
                            rapid_read_64(bytes, position + 40) ^ seed,
                        );
                        if remaining > 64 {
                            seed = rapid_mix(
                                rapid_read_64(bytes, position + 48) ^ RAPIDHASH_SECRET[1],
                                rapid_read_64(bytes, position + 56) ^ seed,
                            );
                        }
                    }
                }
            }

            a = rapid_read_64(bytes, position + remaining - 16) ^ remaining as u64;
            b = rapid_read_64(bytes, position + remaining - 8);
        }

        a ^= RAPIDHASH_SECRET[1];
        b ^= seed;
        (a, b) = rapid_multiply(a, b);
        rapid_mix(
            a ^ RAPIDHASH_SECRET[7],
            b ^ RAPIDHASH_SECRET[1] ^ remaining as u64,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rapidhash_micro_matches_v3_reference_vectors() {
            let vectors = [
                (0, 0x0338_dc4b_e2ce_cdae),
                (1, 0x1b8b_9978_58cd_243a),
                (2, 0x390c_f47a_e3cc_bef0),
                (3, 0x5e69_af77_64e5_410e),
                (4, 0x3950_0c66_56c1_5c24),
                (7, 0x13ab_fab8_cc7d_fa3a),
                (8, 0x4837_4b67_35e2_878e),
                (16, 0x8d62_e217_9a38_046f),
                (17, 0x405a_b354_d26a_9531),
                (32, 0xddca_d65e_2d0c_8b73),
                (33, 0xf347_f406_daa8_0e85),
                (48, 0x4fa6_904d_a48e_13a9),
                (49, 0xe1e9_ce6f_f120_7aeb),
                (64, 0xda1a_1bb5_fa78_999b),
                (65, 0x9150_92c3_0021_7090),
                (80, 0x7c3e_3bbf_cbaa_5bc6),
                (81, 0x32d6_9cab_c9c9_6203),
                (96, 0x58c3_682e_c38d_fc63),
                (160, 0xf584_8120_30fd_783b),
                (161, 0x670b_41a9_c3f5_06b0),
                (512, 0xf38c_d1c4_ca9e_1e72),
                (1024, 0x240a_e38e_ed77_dc84),
            ];

            for (len, expected) in vectors {
                let input = (0..len)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                    .collect::<Vec<_>>();
                assert_eq!(rapidhash_micro(&input), expected, "length {len}");
            }
        }
    }
}
