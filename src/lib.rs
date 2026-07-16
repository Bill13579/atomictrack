// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

//! Play with numbers.
//!
//! A few things to note:
//! - Once a number is entered, it pushes the floor up forever. Even after it is suspended, if that slot is ever reused again it will impose an arbitrary (based on who happened to use that slot before) floor on the new number that can be entered on that slot.
//! - For any [`AtomicTrack`], all numbers must remain within a contiguous circular range narrower than MAX/**4** (or 2^(BITS - 2)) (inclusive) *at all times.* This is so that there is a well-defined "ahead" and "behind" relationship between numbers even across wraparounds (as in transitive ordering).
//! - Greater than and less than comparisons use wrapping-aware versions internally, so you might be surprised at some of the behavior when seeing edge cases. For example, on a fresh track, `enter_from(id, MSB / 2)` will initialize the lane to 0, because the halfway point on the ring has no unambiguous ordering relative to zero. If you keep to the rule that all numbers remain within the circular range mentioned above though, you probably won't notice most of the time.
//! - Keys are not guaranteed to be unique; rather, it's loosely unique in the sense that it will tell you if it finds during the probe when inserting that there is a slot with that key already (*if* it finds it, it might not) or with [`AtomicTrack::recover`] and [`AtomicTrack::with_id`], returning the first matching lane with no promises that the result is unique. The [`NumberId`] should be considered the actual unique key.
//! - Unless there's a `_concurrent` version of a function present, otherwise all functions can be assumed thread-safe.
//! - [`AtomicTrack`] is `Send` and `Sync` and you can [`Clone`] it because it has an internal atomic reference count.
//! - ...That's basically it for now.

#![no_std]
#![cfg_attr(
    all(target_family = "wasm", target_feature = "atomics"),
    feature(stdarch_wasm_atomic_wait)
)]

extern crate alloc;
use alloc::{boxed::Box, vec::Vec};

use core::ptr::NonNull;

use core::sync::atomic::{AtomicUsize, Ordering};

// AtomicUsize for manual atomic reference counting and AtomicType for everything else.
pub mod math;
use math::{AtomicType, NumericType, MSB};

const MAX_PUBLIC: NumericType = MSB - 1;

pub mod cache_padded;
use cache_padded::CachePadded;

#[cfg(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    target_os = "freebsd",
    all(target_family = "wasm", target_feature = "atomics")
))]
pub mod futex;

use MSB as SUSPENDED_BIT;

use crate::math::{gte_msb_masked, max2_msb_masked, max3_msb_masked, min2_msb_masked};

const EMPTY_ID: NumericType = 0;
const KEY_LOCK_BIT: NumericType = SUSPENDED_BIT;

/// Get only the key bits from a number, meaning everything other than the MSB.
#[inline]
const fn key_bits(key: NumericType) -> NumericType {
    key & !KEY_LOCK_BIT
}

/// Is the key's lock bit (which is the MSB) set? If it's set (and it's not empty, since if it's empty then it can't logically be locked, though that's obvious but the compiler will strip this out) then it's locked.
#[inline]
const fn is_key_locked(key: NumericType) -> bool {
    (key & KEY_LOCK_BIT) != 0 && key != EMPTY_ID
}

/// Return the number after setting the lock bit (which is the MSB), regardless of whether it was set already or not.
#[inline]
const fn lock_key(key: NumericType) -> NumericType {
    key | KEY_LOCK_BIT
}

/// Return the number after setting the suspended bit (which is the MSB), regardless of whether it was set already or not.
#[inline]
const fn with_suspended_bit(value: NumericType) -> NumericType {
    value | MSB
}

/// Get only the value bits from a number, meaning everything other than the MSB.
#[inline]
const fn without_suspended_bit(value: NumericType) -> NumericType {
    value & !MSB
}

/// Is the value's suspended bit (which is the MSB) set? If it's set then it's suspended.
#[inline]
const fn is_suspended(value: NumericType) -> bool {
    (value & MSB) != 0
}

/// A value can't be larger than MAX_PUBLIC (because of the reserved MSB), so this checks for that.
#[inline]
const fn is_valid_public_value(value: NumericType) -> bool {
    value <= MAX_PUBLIC
}

pub struct AtomicTrack {
    inner: NonNull<AtomicTrackInner>,
}

pub struct AtomicTrackInner {
    handle_count: CachePadded<AtomicUsize>,
    min: CachePadded<AtomicType>,
    mask: usize,
    slots: Box<[CachePadded<Slot>]>,
}

struct Slot {
    id: AtomicType,
    value: AtomicType,
}

impl Slot {
    const fn new() -> Self {
        Self {
            id: AtomicType::new(EMPTY_ID),
            value: AtomicType::new(with_suspended_bit(0)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NumberId {
    pub id: NumericType,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnterError {
    InvalidId,
    InvalidValue,
    AlreadyPresent,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaveError {
    InvalidOffset,
    NotFound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberError {
    InvalidId,
    InvalidOffset,
    NotFound,
}

pub struct Number<'a> {
    slot: &'a CachePadded<Slot>,
    id: NumericType,
}

unsafe impl Send for AtomicTrack {}
unsafe impl Sync for AtomicTrack {}

impl Clone for AtomicTrack {
    fn clone(&self) -> Self {
        unsafe {
            self.inner
                .as_ref()
                .handle_count
                .fetch_add(1, Ordering::Relaxed);
        }
        Self { inner: self.inner }
    }
}

impl Drop for AtomicTrack {
    fn drop(&mut self) {
        unsafe {
            let inner = self.inner.as_ref();
            if inner.handle_count.fetch_sub(1, Ordering::Release) == 1 {
                core::sync::atomic::fence(Ordering::Acquire);
                drop(Box::from_raw(self.inner.as_ptr()));
            }
        }
    }
}

impl AtomicTrack {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two"
        );

        let slots = (0..capacity)
            .map(|_| CachePadded::new(Slot::new()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let inner = Box::new(AtomicTrackInner {
            handle_count: CachePadded::new(AtomicUsize::new(1)),
            min: CachePadded::new(AtomicType::new(0)),
            mask: capacity - 1,
            slots,
        });

        Self {
            inner: NonNull::from(Box::leak(inner)),
        }
    }

    pub fn capacity(&self) -> usize {
        unsafe { self.inner.as_ref().slots.len() }
    }

    /// [`find_min`](`AtomicTrack::find_min`) updates the current global min, this one just reads it.
    pub fn min(&self) -> NumericType {
        unsafe { self.inner.as_ref().min.load(Ordering::Acquire) }
    }

    pub fn find_min(&self) -> NumericType {
        unsafe { self.inner.as_ref().find_min() }
    }

    pub fn enter(&self, id: NumericType) -> Result<NumberId, EnterError> {
        self.enter_from(id, self.min())
    }

    pub fn enter_from(&self, id: NumericType, at_least: NumericType) -> Result<NumberId, EnterError> {
        unsafe { self.inner.as_ref().enter_from(id, at_least) }
    }

    /// Recover a [`NumberId`] from a key. This is usually fast but since it *can* scan the whole ring, keeping the [`NumberId`] directly is better.
    pub fn recover(&self, id: NumericType) -> Option<NumberId> {
        unsafe { self.inner.as_ref().recover(id) }
    }

    /// Just like [`recover`](`AtomicTrack::recover`), this is usually fast but since it *can* scan the whole ring, using [`with_number`](`AtomicTrack::with_number`) is better if you can.
    pub fn with_id<R>(&self, id: NumericType, f: impl FnOnce(Number<'_>) -> R) -> Result<R, NumberError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(NumberError::InvalidId);
        }
        let number = self.recover(id).ok_or(NumberError::NotFound)?;
        self.with_number(number, f)
    }

    pub fn with_number<R>(
        &self,
        number: NumberId,
        f: impl FnOnce(Number<'_>) -> R,
    ) -> Result<R, NumberError> {
        self.number(number).map(f)
    }

    pub fn number(&self, number: NumberId) -> Result<Number<'_>, NumberError> {
        let slot = unsafe { self.inner.as_ref().get_slot_concurrent(number)? };
        Ok(Number {
            slot,
            id: number.id,
        })
    }

    pub fn leave(&self, number: NumberId) -> Result<(), LeaveError> {
        unsafe { self.inner.as_ref().__leave(number, false) }
    }

    /// Use this if there's ever more than one thread that might be touching the same number concurrently. There aren't any performance penalties, but when this physical slot is reused later for a new entrant, they will have to enter at a value *at least* 1 greater than what was left behind (other conditions separate).
    pub fn leave_concurrent(&self, number: NumberId) -> Result<(), LeaveError> {
        unsafe { self.inner.as_ref().__leave(number, true) }
    }
}

impl AtomicTrackInner {
    fn __leave(&self, number: NumberId, __concurrent_write_leave: bool) -> Result<(), LeaveError> {
        let slot = match self.get_slot_concurrent(number) {
            Ok(slot) => slot,
            Err(NumberError::InvalidOffset) => return Err(LeaveError::InvalidOffset),
            Err(NumberError::InvalidId) | Err(NumberError::NotFound) => return Err(LeaveError::NotFound),
        };

        loop {
            let key_before = slot.id.load(Ordering::Acquire);
            if key_before != number.id {
                return Err(LeaveError::NotFound);
            }

            let current = slot.value.load(Ordering::Acquire);
            if is_suspended(current) {
                // Only initialization and leaving paths create suspended values, so if we see that the value is suspended here, another leave already did the job.
                return Ok(());
            }

            let key_after = slot.id.load(Ordering::Acquire);
            if key_before != key_after {
                core::hint::spin_loop();
                continue;
            }

            let desired_value = if __concurrent_write_leave {
                without_suspended_bit(current.wrapping_add(1))
            } else {
                without_suspended_bit(current)
            };

            let desired = with_suspended_bit(desired_value);

            match slot
                .value
                .compare_exchange(current, desired, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    slot.id.store(EMPTY_ID, Ordering::Release);
                    return Ok(());
                }
                Err(_) => {
                    core::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    fn enter_from(&self, id: NumericType, at_least: NumericType) -> Result<NumberId, EnterError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(EnterError::InvalidId);
        }
        if !is_valid_public_value(at_least) {
            return Err(EnterError::InvalidValue);
        }

        let start_idx = self.index_for(id);

        for probe_offset in 0..self.slots.len() {
            let idx = (start_idx + probe_offset) & self.mask;
            let slot = &self.slots[idx];
            let current_id = slot.id.load(Ordering::Acquire);

            if key_bits(current_id) == id {
                return Err(EnterError::AlreadyPresent);
            }

            if current_id == EMPTY_ID
                && slot
                    .id
                    .compare_exchange(EMPTY_ID, lock_key(id), Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                loop {
                    let current = slot.value.load(Ordering::Acquire);
                    debug_assert!(is_suspended(current));

                    // Resume from max(current_lane_floor, global_min, at_least)
                    let desired = max3_msb_masked(
                        without_suspended_bit(current),
                        self.min.load(Ordering::Acquire),
                        at_least,
                    );

                    // CAS to enter active state
                    match slot.value.compare_exchange(
                        current,
                        desired,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            slot.id.store(id, Ordering::Release);
                            return Ok(NumberId {
                                id,
                                offset: probe_offset as u64,
                            });
                        }
                        Err(_) => continue,
                    }
                }
            }
        }

        Err(EnterError::Full)
    }

    fn recover(&self, id: NumericType) -> Option<NumberId> {
        if let Some((number_id, false)) = self.__recover(id) {
            Some(number_id)
        } else {
            None
        }
    }

    fn __recover(&self, id: NumericType) -> Option<(NumberId, bool)> {
        if id == EMPTY_ID || is_key_locked(id) {
            return None;
        }

        let base_index = self.index_for(id);
        for probe_offset in 0..self.slots.len() {
            let index = (base_index + probe_offset) & self.mask;
            let slot = &self.slots[index];
            let current = slot.id.load(Ordering::Acquire);
            if key_bits(current) == id {
                return Some((
                    NumberId {
                        id,
                        offset: probe_offset as u64,
                    },
                    is_key_locked(current),
                ));
            }
        }

        None
    }

    fn find_min(&self) -> NumericType {
        let mut active_max: Option<NumericType> = None;

        // Goes through all slots to find the maximum active number.
        // This is only needed as a benchmark for the second loop to have an upper bound it can guarantee to not move min ahead of.
        // Thus, it only needs to ensure it looks at all active numbers when it runs.
        for slot in self.slots.iter() {
            loop {
                let id_before = slot.id.load(Ordering::Acquire);
                let value = slot.value.load(Ordering::Acquire);
                let id_after = slot.id.load(Ordering::Acquire);

                if id_before != id_after {
                    core::hint::spin_loop();
                    continue;
                }

                let number = without_suspended_bit(value);

                if !is_suspended(value) && key_bits(id_before) != EMPTY_ID {
                    active_max = Some(match active_max {
                        Some(current) => max2_msb_masked(current, number),
                        None => number,
                    });
                }

                break;
            }
        }

        let mut running_min;
        if let Some(active_max) = active_max {
            running_min = active_max;
        } else {
            // If there are no active numbers at all, considering that we have to be wrapping-aware, we're just gonna return the current global min and not update it at all.
            // `find_min` is lazy that way.
            return self.min.load(Ordering::Acquire);
        }

        'outer: for slot in self.slots.iter() {
            'retry: loop {
                let id_before = slot.id.load(Ordering::Acquire);
                let mut current = slot.value.load(Ordering::Acquire);

                'inner: while is_suspended(current) {
                    let current_value = without_suspended_bit(current);

                    if gte_msb_masked(current_value, running_min) {
                        let id_after = slot.id.load(Ordering::Acquire);
                        if id_before != id_after {
                            core::hint::spin_loop();
                            continue 'retry;
                        }

                        continue 'outer;
                    }

                    match slot.value.compare_exchange(
                        current,
                        with_suspended_bit(running_min),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => continue 'outer,
                        Err(actual) => {
                            current = actual;
                            continue 'inner;
                        }
                    }
                }

                let id_after = slot.id.load(Ordering::Acquire);
                if id_before != id_after {
                    core::hint::spin_loop();
                    continue 'retry;
                }

                if key_bits(id_before) != EMPTY_ID {
                    running_min = min2_msb_masked(running_min, without_suspended_bit(current));
                }

                continue 'outer;
            }
        }

        let mut current = self.min.load(Ordering::Relaxed);

        let previous = loop {
            let new_value = max2_msb_masked(current, running_min);

            match self.min.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel, // Successful read-modify-write
                Ordering::Acquire, // Failed comparison is only a load
            ) {
                Ok(previous) => break previous,
                Err(actual) => current = actual,
            }
        };

        max2_msb_masked(previous, running_min)
    }

    fn offset_as_usize(&self, offset: u64) -> Result<usize, NumberError> {
        let offset = usize::try_from(offset).map_err(|_| NumberError::InvalidOffset)?;
        if offset >= self.slots.len() {
            return Err(NumberError::InvalidOffset);
        }
        Ok(offset)
    }

    fn get_slot_concurrent(&self, number: NumberId) -> Result<&CachePadded<Slot>, NumberError> {
        if number.id == EMPTY_ID || is_key_locked(number.id) {
            return Err(NumberError::InvalidId);
        }
        let offset = self.offset_as_usize(number.offset)?;
        let index = self.index_for_offsetted(number.id, offset);

        Ok(&self.slots[index])
    }

    #[inline]
    fn index_for(&self, id: NumericType) -> usize {
        (id as usize) & self.mask
    }

    #[inline]
    fn index_for_offsetted(&self, id: NumericType, placement_offset: usize) -> usize {
        (id as usize + placement_offset) & self.mask
    }
}

impl<'a> Number<'a> {
    pub fn get(&self) -> Result<NumericType, NumberError> {
        loop {
            let key_before = self.slot.id.load(Ordering::Acquire);
            if key_before != self.id {
                return Err(NumberError::NotFound);
            }

            let value = self.slot.value.load(Ordering::Acquire);
            let key_after = self.slot.id.load(Ordering::Acquire);
            if key_before != key_after {
                core::hint::spin_loop();
                continue;
            }

            if is_suspended(value) {
                return Err(NumberError::NotFound);
            }

            return Ok(without_suspended_bit(value));
        }
    }

    pub fn raise_to(&self, at_least: NumericType) -> Result<NumericType, NumberError> {
        assert!(
            is_valid_public_value(at_least),
            "at_least must fit in the public number range"
        );

        loop {
            let key_before = self.slot.id.load(Ordering::Acquire);
            if key_before != self.id {
                return Err(NumberError::NotFound);
            }

            let current = self.slot.value.load(Ordering::Acquire);
            if is_suspended(current) {
                return Err(NumberError::NotFound);
            }

            let desired = max2_msb_masked(without_suspended_bit(current), at_least);

            let key_after = self.slot.id.load(Ordering::Acquire);
            if key_before != key_after {
                core::hint::spin_loop();
                continue;
            }

            match self.slot.value.compare_exchange(
                current,
                desired,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(desired),
                Err(_) => {
                    core::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    pub fn add(&self, delta: NumericType) -> Result<NumericType, NumberError> {
        loop {
            let key_before = self.slot.id.load(Ordering::Acquire);
            if key_before != self.id {
                return Err(NumberError::NotFound);
            }

            let current = self.slot.value.load(Ordering::Acquire);
            if is_suspended(current) {
                return Err(NumberError::NotFound);
            }

            let next = without_suspended_bit(current.wrapping_add(delta));

            let key_after = self.slot.id.load(Ordering::Acquire);
            if key_before != key_after {
                core::hint::spin_loop();
                continue;
            }

            match self.slot.value.compare_exchange(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(next),
                Err(_) => {
                    core::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    /// # Safety
    /// Callers must preserve monotonicity and must not set the suspended bit.
    pub unsafe fn atomic(&self) -> &AtomicType {
        &self.slot.value
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::{
        sync::{Arc, Barrier},
        thread,
        time::Duration,
    };

    use super::*;

    #[test]
    fn enter_recover_and_use_number() {
        let track = AtomicTrack::new(8);

        let number = track.enter(11).unwrap();
        assert_eq!(track.recover(11), Some(number));

        let current = track
            .with_number(number, |lane| {
                assert_eq!(lane.get().unwrap(), 0);
                lane.raise_to(5).unwrap()
            })
            .unwrap();

        assert_eq!(current, 5);
        assert_eq!(track.with_id(11, |lane| lane.get().unwrap()).unwrap(), 5);
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let track = AtomicTrack::new(4);
        assert!(track.enter(123).is_ok());
        assert_eq!(track.enter(123), Err(EnterError::AlreadyPresent));
    }

    #[test]
    fn same_id_can_identify_distinct_placements_after_a_hole() {
        let track = AtomicTrack::new(4);
        let hole = track.enter(1).unwrap();
        let first = track.enter(5).unwrap();

        track
            .with_number(first, |lane| lane.raise_to(11).unwrap())
            .unwrap();
        track.leave(hole).unwrap();

        let second = track.enter(5).unwrap();
        track
            .with_number(second, |lane| lane.raise_to(22).unwrap())
            .unwrap();

        assert_ne!(first, second);
        assert_eq!(first.offset, 1);
        assert_eq!(second.offset, 0);
        assert_eq!(track.recover(5), Some(second));
        assert_eq!(track.with_number(first, |lane| lane.get()).unwrap(), Ok(11));
        assert_eq!(
            track.with_number(second, |lane| lane.get()).unwrap(),
            Ok(22)
        );
    }

    #[test]
    fn ids_with_reserved_bit_are_rejected() {
        let track = AtomicTrack::new(4);
        assert_eq!(track.enter(SUSPENDED_BIT), Err(EnterError::InvalidId));
    }

    #[test]
    fn offset_is_probe_delta() {
        let track = AtomicTrack::new(4);
        let first = track.enter(1).unwrap();
        let second = track.enter(5).unwrap();

        assert_eq!(first.offset, 0);
        assert_eq!(second.offset, 1);
        assert_eq!(track.recover(1), Some(first));
        assert_eq!(track.recover(5), Some(second));
    }

    #[test]
    fn find_min_advances_global_min() {
        let track = AtomicTrack::new(8);
        let a = track.enter(1).unwrap();
        let b = track.enter(2).unwrap();

        track
            .with_number(a, |lane| lane.raise_to(10).unwrap())
            .unwrap();
        track
            .with_number(b, |lane| lane.raise_to(7).unwrap())
            .unwrap();

        assert_eq!(track.find_min(), 7);
        assert_eq!(track.min(), 7);

        track.with_number(b, |lane| lane.add(5).unwrap()).unwrap();
        assert_eq!(track.find_min(), 10);
        assert_eq!(track.min(), 10);
    }

    #[test]
    fn leave_and_reenter_reuse_lane_with_new_id() {
        let track = AtomicTrack::new(2);
        let a = track.enter(10).unwrap();
        track
            .with_number(a, |lane| lane.raise_to(9).unwrap())
            .unwrap();
        assert_eq!(track.find_min(), 9);

        track.leave(a).unwrap();
        let b = track.enter(20).unwrap();

        assert_eq!(
            track.with_number(a, |lane| lane.get()).unwrap(),
            Err(NumberError::NotFound)
        );
        assert_eq!(track.with_number(b, |lane| lane.get()).unwrap().unwrap(), 9);
    }

    #[test]
    fn suspended_lane_floor_survives_reentry_without_advancing_global_min() {
        let track = AtomicTrack::new(1);
        let a = track.enter(1).unwrap();

        track
            .with_number(a, |lane| lane.raise_to(12).unwrap())
            .unwrap();
        track.leave(a).unwrap();

        assert_eq!(track.find_min(), 0);
        assert_eq!(track.min(), 0);

        let b = track.enter(2).unwrap();
        assert_eq!(
            track.with_number(b, |lane| lane.get()).unwrap().unwrap(),
            12
        );
    }

    #[test]
    fn leave_concurrent_bumps_reentry_floor() {
        let track = AtomicTrack::new(1);
        let a = track.enter(1).unwrap();

        track
            .with_number(a, |lane| lane.raise_to(12).unwrap())
            .unwrap();
        track.leave_concurrent(a).unwrap();

        let b = track.enter(2).unwrap();
        assert_eq!(
            track.with_number(b, |lane| lane.get()).unwrap().unwrap(),
            13
        );
    }

    #[test]
    fn find_min_and_concurrent_leave_progress_across_wrap() {
        let track = AtomicTrack::new(1);
        let number = track.enter(1).unwrap();
        let quarter = NumericType::MAX / 4;

        for delta in [quarter, quarter, 1] {
            let value = track
                .with_number(number, |lane| lane.add(delta).unwrap())
                .unwrap();
            assert_eq!(track.find_min(), value);
            assert_eq!(track.min(), value);
        }

        assert_eq!(track.min(), MAX_PUBLIC);
        track.leave_concurrent(number).unwrap();
        assert_eq!(track.find_min(), MAX_PUBLIC);

        let replacement = track.enter(2).unwrap();
        assert_eq!(
            track.with_number(replacement, |lane| lane.get()).unwrap(),
            Ok(0)
        );
        assert_eq!(track.find_min(), 0);
        assert_eq!(track.min(), 0);
    }

    #[test]
    fn raise_to_progresses_across_wrap() {
        let track = AtomicTrack::new(1);
        let number = track.enter(1).unwrap();
        let quarter = NumericType::MAX / 4;

        for delta in [quarter, quarter, 1] {
            let value = track
                .with_number(number, |lane| lane.add(delta).unwrap())
                .unwrap();
            assert_eq!(track.find_min(), value);
        }

        assert_eq!(track.min(), MAX_PUBLIC);
        assert_eq!(
            track.with_number(number, |lane| lane.raise_to(1)).unwrap(),
            Ok(1)
        );
        assert_eq!(track.find_min(), 1);
        assert_eq!(track.min(), 1);
    }

    #[test]
    fn leave_can_beat_an_in_flight_update() {
        let track = Arc::new(AtomicTrack::new(4));
        let number = track.enter(77).unwrap();
        let barrier = Arc::new(Barrier::new(2));

        let worker_track = track.clone();
        let worker_barrier = barrier.clone();
        let handle = thread::spawn(move || {
            worker_track
                .with_number(number, |lane| {
                    worker_barrier.wait();
                    thread::sleep(Duration::from_millis(50));
                    lane.add(1)
                })
                .unwrap()
        });

        barrier.wait();
        track.leave_concurrent(number).unwrap();

        assert_eq!(handle.join().unwrap(), Err(NumberError::NotFound));
        assert_eq!(
            track.with_number(number, |lane| lane.get()).unwrap(),
            Err(NumberError::NotFound)
        );
    }
}
