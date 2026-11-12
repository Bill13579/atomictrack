// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

//! Play with numbers.
//!
//! A few things to note:
//! - Once a number is entered, it pushes the floor up forever. Even after it is suspended, if that slot is ever reused again it will impose an arbitrary (based on who happened to use that slot before) floor on the new number that can be entered on that slot.
//! - For any [`AtomicTrack`], all numbers must remain within a contiguous circular range narrower than MAX/**4** (or 2^(BITS - 2), or 2^(USER_BIT_LOW - 2) with custom reserved bits) (inclusive) *at all times.* This is so that there is a well-defined "ahead" and "behind" relationship between numbers even across wraparounds (as in transitive ordering).
//! - Greater than and less than comparisons use wrapping-aware versions internally, so you might be surprised at some of the behavior when seeing edge cases. For example, on a fresh track, `enter_from(id, MSB / 2)` will initialize the lane to 0, because the halfway point on the ring has no unambiguous ordering relative to zero. If you keep to the rule that all numbers remain within the circular range mentioned above though, you probably won't notice most of the time.
//! - Keys are not guaranteed to be unique; rather, it's loosely unique in the sense that it will tell you if it finds during the probe when inserting that there is a slot with that key already (*if* it finds it, it might not) or with [`AtomicTrack::recover`] and [`AtomicTrack::with_id`], returning the first matching lane with no promises that the result is unique. The [`NumberId`] should be considered the actual unique key.
//! - Unless there's a `_concurrent` version of a function present, otherwise all functions can be assumed thread-safe.
//! - [`AtomicTrack`] is `Send` and `Sync`.
//! - ...That's basically it for now.

#![no_std]
#![cfg_attr(
    all(target_family = "wasm", target_feature = "atomics"),
    feature(stdarch_wasm_atomic_wait)
)]
#![cfg_attr(
    all(target_arch = "wasm64", target_feature = "atomics"),
    feature(simd_wasm64)
)]

use core::{alloc::Layout, ops::Deref, ptr::{self, NonNull}};

use core::sync::atomic::Ordering;

pub mod math;
use math::{AtomicType, NumericType, MSB};

const MAX_PUBLIC: NumericType = MSB - 1;

pub mod utils;

pub mod cache_padded;
use cache_padded::CachePadded;

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_family = "wasm")),
    target_os = "macos",
    target_os = "freebsd",
    all(target_family = "wasm", target_feature = "atomics")
))]
pub mod futex;

use MSB as SUSPENDED_BIT;

use crate::math::{gte_masked, is_valid_user_bit_low, lt_masked, max2_masked, max2_special_masked, min2_masked, number_mask, user_mask};

const EMPTY_ID: NumericType = 0;
const KEY_LOCK_BIT: NumericType = SUSPENDED_BIT;

#[cfg(all(
    feature = "waiting",
    any(
        target_os = "windows",
        all(target_os = "linux", not(target_family = "wasm")),
        target_os = "macos",
        target_os = "freebsd",
        all(target_family = "wasm", target_feature = "atomics")
    )
))]
pub mod waiting;

#[cfg(all(
    feature = "waiting",
    not(any(
        target_os = "windows",
        all(target_os = "linux", not(target_family = "wasm")),
        target_os = "macos",
        target_os = "freebsd",
        all(target_family = "wasm", target_feature = "atomics")
    ))
))]
compile_error!(
    "the `waiting` module supports Windows, Linux, macOS, FreeBSD, and Wasm with the atomics target feature"
);

/// Get only the key bits from a number, meaning everything other than the MSB.
#[inline]
const fn key_bits(key: NumericType) -> NumericType {
    key & !KEY_LOCK_BIT
}

#[doc(hidden)]
#[inline]
pub const fn __key_bits(key: NumericType) -> NumericType {
    key_bits(key)
}

#[macro_export]
macro_rules! pair {
    ($static:expr, $runtime:expr $(,)?) => {
        $crate::__key_bits($crate::math::NumericType::wrapping_add(
            const { let id: $crate::math::NumericType = $static; id },
            ($runtime) as $crate::math::NumericType,
        ))
    };
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

#[repr(C)]
pub struct AtomicTrack<
    S: ?Sized = [CachePadded<Slot>],
    const USER_BIT_LOW: u32 = { NumericType::BITS },
    const USER_BITS_DEFAULT: NumericType = 0,
> {
    seed: u64,
    min: CachePadded<AtomicType>,
    slots: S,
}

pub type ArrayAtomicTrack<
    const N: usize,
    const USER_BIT_LOW: u32 = { NumericType::BITS },
    const USER_BITS_DEFAULT: NumericType = 0,
> = AtomicTrack<[CachePadded<Slot>; N], USER_BIT_LOW, USER_BITS_DEFAULT>;

pub struct Slot {
    id: AtomicType,
    value: AtomicType,
}

impl Slot {
    pub const fn new() -> Self {
        Self {
            id: AtomicType::new(EMPTY_ID),
            value: AtomicType::new(with_suspended_bit(0)),
        }
    }
}

impl Default for Slot {
    fn default() -> Self {
        Self::new()
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
    ExactFailed,
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
    /// Number already past the target ([`set_to`](`Number::set_to`)).
    PassedTarget(NumericType),
    /// [`add_within`](`Number::add_within`) did not add since the result would exceed the limit.
    Exceeded(NumericType),
    /// A `try_*` lost on the compare-exchange to a concurrent change.
    CasFailed,
    Suspended,
    NotSuspended,
    ExactFailed,
}

pub struct Number<'a, const USER_BIT_LOW: u32 = { NumericType::BITS }> {
    slot: &'a CachePadded<Slot>,
    id: NumericType,
}

const _: () = {
    const fn assert_send_sync<T: ?Sized + Send + Sync>() {}
    assert_send_sync::<AtomicTrack>();
};

impl<S: ?Sized, const L: u32, const I: NumericType> AtomicTrack<S, L, I> {
    const NUMBER_MASK: NumericType = number_mask(L);
    const VALID: () = {
        assert!(is_valid_user_bit_low(L), "USER_BIT_LOW should be within range 3..=NumericType::BITS");
        assert!(I & !user_mask(L) == 0, "USER_BITS_DEFAULT must only use the user bits");
    };

    /// The address the track was initialized at through [`init_in_place`](`AtomicTrack::init_in_place`),
    /// used as a seed to disperse futexes in different tracks (set once initially, and then stays the same value with subsequent processes that map it).
    /// Or for tracks made through [`ArrayAtomicTrack::new`], always the current address, since using that constructor implies private memory.
    #[cfg_attr(not(feature = "waiting"), allow(dead_code))]
    #[inline]
    pub(crate) fn seed(&self) -> u64 {
        if self.seed == 0 {
            (self as *const Self).addr() as u64
        } else {
            self.seed
        }
    }
}

impl<const N: usize, const L: u32, const I: NumericType> ArrayAtomicTrack<N, L, I> {
    pub const fn new() -> Self {
        const { assert!(N.is_power_of_two(), "capacity must be a power of two") };
        let () = Self::VALID;
        Self {
            seed: 0,
            min: CachePadded::new(AtomicType::new(0)),
            slots: [const { CachePadded::new(Slot::new()) }; N],
        }
    }
}

impl<const N: usize, const L: u32, const I: NumericType> Default for ArrayAtomicTrack<N, L, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const L: u32, const I: NumericType> Deref for ArrayAtomicTrack<N, L, I> {
    type Target = AtomicTrack<[CachePadded<Slot>], L, I>;

    fn deref(&self) -> &Self::Target {
        self
    }
}

impl AtomicTrack {
    pub fn layout(capacity: usize) -> Option<Layout> {
        if !capacity.is_power_of_two() {
            return None;
        }
        let slots = Layout::array::<CachePadded<Slot>>(capacity).ok()?;
        let (header, _) = Layout::new::<u64>().extend(Layout::new::<CachePadded<AtomicType>>()).ok()?;
        let (layout, _) = header.extend(slots).ok()?;
        Some(layout.pad_to_align())
    }

    /// # Safety
    /// Same as [`init_in_place`](`AtomicTrack::init_in_place`).
    pub unsafe fn init_in_place_default<'a>(ptr: NonNull<u8>, capacity: usize) -> &'a Self {
        unsafe { Self::init_in_place(ptr, capacity) }
    }

    /// # Safety
    /// Same as [`from_raw`](`AtomicTrack::from_raw`).
    pub unsafe fn from_raw_default<'a>(ptr: NonNull<u8>, capacity: usize) -> &'a Self {
        unsafe { Self::from_raw(ptr, capacity) }
    }
}

impl<const L: u32, const I: NumericType> AtomicTrack<[CachePadded<Slot>], L, I> {
    /// # Safety
    /// `ptr` must be valid for writes, and with layout [`layout(capacity)`](`AtomicTrack::layout`). It must be aligned. It must stay valid for lifetime `'a`.
    pub unsafe fn init_in_place<'a>(ptr: NonNull<u8>, capacity: usize) -> &'a Self {
        let () = Self::VALID;
        assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two"
        );
        assert!(
            AtomicTrack::layout(capacity).is_some(),
            "capacity is too large"
        );
        let track = ptr::slice_from_raw_parts_mut(ptr.as_ptr().cast::<CachePadded<Slot>>(), capacity) as *mut Self;
        unsafe {
            (&raw mut (*track).seed).write(ptr.as_ptr().addr() as u64);
            (&raw mut (*track).min).write(CachePadded::new(AtomicType::new(0)));
            let slots = (&raw mut (*track).slots).cast::<CachePadded<Slot>>();
            for i in 0..capacity {
                slots.add(i).write(CachePadded::new(Slot::new()));
            }
            &*track
        }
    }

    /// # Safety
    /// `ptr` must point to a track of exactly `capacity` number of slots that was initialized with [`init_in_place`](`AtomicTrack::init_in_place`) (or is otherwise laid out and initialized identically) and stays valid for `'a`.
    /// Also note that if `USER_BIT_LOW` and `USER_BITS_DEFAULT` is different between when initialization happened and usage here, existing user bits might potentially become interpreted as the number part instead of the user bits part or vice versa.
    pub unsafe fn from_raw<'a>(ptr: NonNull<u8>, capacity: usize) -> &'a Self {
        let () = Self::VALID;
        debug_assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two"
        );
        debug_assert!(
            AtomicTrack::layout(capacity).is_some(),
            "capacity is too large"
        );
        unsafe {
            &*(ptr::slice_from_raw_parts(ptr.as_ptr().cast::<CachePadded<Slot>>(), capacity) as *const Self)
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// [`find_min`](`AtomicTrack::find_min`) updates the current global min, this one just reads it.
    pub fn min(&self) -> NumericType {
        self.min.load(Ordering::Acquire)
    }

    pub fn enter(&self, id: NumericType) -> Result<NumberId, EnterError> {
        self.__enter_from(id, None, false)
    }

    /// Recover a [`NumberId`] from a key. This is usually fast but since it *can* scan the whole ring, keeping the [`NumberId`] directly is better.
    pub fn recover(&self, id: NumericType) -> Option<NumberId> {
        if let Some((number_id, false)) = self.__recover(id) {
            Some(number_id)
        } else {
            None
        }
    }

    /// Just like [`recover`](`AtomicTrack::recover`), this is usually fast but since it *can* scan the whole ring, using [`with_number`](`AtomicTrack::with_number`) is better if you can.
    pub fn with_id<R>(&self, id: NumericType, f: impl FnOnce(Number<'_, L>) -> R) -> Result<R, NumberError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(NumberError::InvalidId);
        }
        let number = self.recover(id).ok_or(NumberError::NotFound)?;
        self.with_number(number, f)
    }

    pub fn with_number<R>(
        &self,
        number: NumberId,
        f: impl FnOnce(Number<'_, L>) -> R,
    ) -> Result<R, NumberError> {
        self.number(number).map(f)
    }

    pub fn number(&self, number: NumberId) -> Result<Number<'_, L>, NumberError> {
        let slot = self.get_slot_concurrent(number)?;
        Ok(Number {
            slot,
            id: number.id,
        })
    }

    pub fn leave(&self, number: NumberId) -> Result<(), LeaveError> {
        self.__leave(number, false)
    }

    /// Use this if there's ever more than one thread that might be touching the same number concurrently. There aren't any performance penalties, but when this physical slot is reused later for a new entrant, they will have to enter at a value *at least* 1 greater than what was left behind (other conditions separate).
    pub fn leave_concurrent(&self, number: NumberId) -> Result<(), LeaveError> {
        self.__leave(number, true)
    }

    #[inline(always)]
    fn __leave(&self, number: NumberId, __concurrent_write_leave: bool) -> Result<(), LeaveError> {
        let slot = match self.get_slot_concurrent(number) {
            Ok(slot) => slot,
            Err(NumberError::InvalidOffset) => return Err(LeaveError::InvalidOffset),
            Err(NumberError::InvalidId) | Err(NumberError::NotFound) => return Err(LeaveError::NotFound),
            // Only value updates return these.
            Err(NumberError::PassedTarget(_)) | Err(NumberError::Exceeded(_)) | Err(NumberError::CasFailed)
            | Err(NumberError::Suspended) | Err(NumberError::NotSuspended) | Err(NumberError::ExactFailed) => unreachable!(),
        };

        loop {
            match slot.id.compare_exchange_weak(number.id, lock_key(number.id), Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(actual) if actual == lock_key(number.id) => core::hint::spin_loop(), // an enter, resume, or another leave is happening
                Err(actual) if actual == number.id => {}, // spurious failure
                Err(_) => return Err(LeaveError::NotFound),
            }
        }

        loop {
            let current = slot.value.load(Ordering::Acquire);
            let desired_value = if __concurrent_write_leave {
                current.wrapping_add(1) & Self::NUMBER_MASK
            } else {
                current & Self::NUMBER_MASK
            };
            let desired = with_suspended_bit(desired_value);
            if current == desired
                || slot
                    .value
                    .compare_exchange(current, desired, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                break;
            }
            core::hint::spin_loop();
        }

        slot.id.store(EMPTY_ID, Ordering::Release);

        Ok(())
    }

    #[inline(always)]
    fn __enter_from(&self, id: NumericType, at: Option<NumericType>, __exact: bool) -> Result<NumberId, EnterError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(EnterError::InvalidId);
        }
        if at.is_some_and(|at| !is_valid_public_value(at)) {
            return Err(EnterError::InvalidValue);
        }

        let start_idx = self.index_for(id);

        for probe_offset in 0..self.slots.len() {
            let idx = (start_idx + probe_offset) & self.mask();
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

                    // Resume from max(current_floor, at)
                    let two_way_max = match at {
                        None => Some(without_suspended_bit(current)),
                        Some(at) => max2_special_masked(
                            at,
                            without_suspended_bit(current),
                            __exact,
                            Self::NUMBER_MASK,
                        ),
                    };

                    let desired;
                    if let Some(two_way_max) = two_way_max {
                        desired = (two_way_max & Self::NUMBER_MASK) | I;
                    } else {
                        // Release the slot we locked since we're not using it
                        slot.id.store(EMPTY_ID, Ordering::Release);
                        return Err(EnterError::ExactFailed);
                    }

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
                        },
                        Err(_) => {
                            core::hint::spin_loop();
                            continue;
                        },
                    }
                }
            }
        }

        Err(EnterError::Full)
    }

    pub fn enter_from(&self, id: NumericType, at_least: NumericType) -> Result<NumberId, EnterError> {
        self.__enter_from(id, Some(at_least), false)
    }

    pub fn enter_at(&self, id: NumericType, at: NumericType) -> Result<NumberId, EnterError> {
        self.__enter_from(id, Some(at), true)
    }

    #[inline(always)]
    fn __recover(&self, id: NumericType) -> Option<(NumberId, bool)> {
        if id == EMPTY_ID || is_key_locked(id) {
            return None;
        }

        let base_index = self.index_for(id);
        for probe_offset in 0..self.slots.len() {
            let index = (base_index + probe_offset) & self.mask();
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

    pub fn find_min(&self) -> (NumericType, NumericType) {
        let mut active_max: Option<NumericType> = None;

        // Goes through all slots to find the maximum active number.
        // This is only needed as a benchmark for the second loop to have an upper bound it can guarantee to not move min ahead of.
        // Thus, it only needs to ensure it looks at all active numbers when it runs.
        for slot in self.slots.iter() {
            let value = slot.value.load(Ordering::Acquire);
            if !is_suspended(value) {
                let number = value & Self::NUMBER_MASK;
                active_max = Some(match active_max {
                    Some(current) => max2_masked(current, number, Self::NUMBER_MASK),
                    None => number,
                });
            }
        }

        if let Some(active_max) = active_max {
            self.find_min_bounded(active_max)
        } else {
            // If there are no active numbers at all, considering that we have to be wrapping-aware, we're just gonna return the current global min and not update it at all.
            // `find_min` is lazy that way.
            let min = self.min.load(Ordering::Acquire);
            return (min, min);
        }
    }

    pub fn find_min_bounded(&self, bound: NumericType) -> (NumericType, NumericType) {
        let mut running_min = bound & Self::NUMBER_MASK;

        // As long as the value is always not suspended when it is actively participating in determining the min (like in
        // the case of keeping data alive in a ring buffer), there's no need to check id at all here.
        'outer: for slot in self.slots.iter() {
            let mut current = slot.value.load(Ordering::Acquire);

            while is_suspended(current) {
                if gte_masked(current & Self::NUMBER_MASK, running_min, Self::NUMBER_MASK) {
                    continue 'outer;
                }

                match slot.value.compare_exchange(
                    current,
                    with_suspended_bit(running_min | (current & user_mask(L))),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => continue 'outer,
                    Err(actual) => {
                        current = actual;
                        core::hint::spin_loop();
                    },
                }
            }

            running_min = min2_masked(running_min, current & Self::NUMBER_MASK, Self::NUMBER_MASK);
        }

        let mut current = self.min.load(Ordering::Relaxed);

        let previous = loop {
            let new_value = max2_masked(current, running_min, Self::NUMBER_MASK);

            match self.min.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel, // Successful read-modify-write
                Ordering::Acquire, // Failed comparison is only a load
            ) {
                Ok(previous) => break previous,
                Err(actual) => {
                    current = actual;
                    core::hint::spin_loop();
                },
            }
        };

        (max2_masked(previous, running_min, Self::NUMBER_MASK), previous)
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
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    #[inline]
    fn index_for(&self, id: NumericType) -> usize {
        (id as usize) & self.mask()
    }

    #[inline]
    fn index_for_offsetted(&self, id: NumericType, placement_offset: usize) -> usize {
        (id as usize).wrapping_add(placement_offset) & self.mask()
    }
}

impl<'a, const L: u32> Number<'a, L> {
    const NUMBER_MASK: NumericType = number_mask(L);
    const USER_MASK: NumericType = user_mask(L);

    #[inline]
    fn get_value(&self) -> Result<NumericType, NumberError> {
        loop {
            let key_before = self.slot.id.load(Ordering::Acquire);
            if key_before != self.id {
                if key_before == lock_key(self.id) {
                    core::hint::spin_loop();
                    continue;
                }
                return Err(NumberError::NotFound);
            }

            let value = self.slot.value.load(Ordering::Acquire);
            let key_after = self.slot.id.load(Ordering::Acquire);
            if key_before != key_after {
                core::hint::spin_loop();
                continue;
            }

            return Ok(value);
        }
    }

    /// Lock this number by setting the lock bit on the its id (see [`AtomicTrack::__leave`]).
    #[inline]
    fn lock(&self) -> Result<(), NumberError> {
        loop {
            match self.slot.id.compare_exchange_weak(self.id, lock_key(self.id), Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Ok(()),
                Err(actual) if actual == lock_key(self.id) => core::hint::spin_loop(), // an enter, resume, or leave is happening
                Err(actual) if actual == self.id => {}, // spurious failure
                Err(_) => return Err(NumberError::NotFound),
            }
        }
    }

    pub fn get(&self) -> Result<NumericType, NumberError> {
        let value = self.get_value()?;
        if is_suspended(value) {
            return Err(NumberError::Suspended);
        }
        Ok(value)
    }

    pub fn raise_to(&self, at_least: NumericType) -> Result<NumericType, NumberError> {
        self.__raise_to(at_least, false, true, true)
    }

    /// Try to [`raise_to`](`Number::raise_to`) at least the provided target, returning [`Err(NumberError::CasFailed)`](`NumberError::CasFailed`) immediately if the compare-exchange failed.
    pub fn try_raise_to(&self, at_least: NumericType) -> Result<NumericType, NumberError> {
        self.__raise_to(at_least, false, false, true)
    }

    /// Attempts to set the number to exactly `target`, with user bits set to the ones passed in only if *this* call is the one that got the number there. Fails with [`NumberError::PassedTarget(current)`](`NumberError::PassedTarget`) if it goes past the target due to concurrent changes, and [`Ok(current)`](`Ok`) with the new value returned if the target is reached, though user bits might differ if it was set by a different, concurrent call.
    pub fn set_to(&self, target: NumericType) -> Result<NumericType, NumberError> {
        self.__raise_to(target, true, true, true)
    }

    /// [`set_to`](`Number::set_to`) with user bits ignored
    pub fn set_numeral(&self, target: NumericType) -> Result<NumericType, NumberError> {
        self.__raise_to(target, true, true, false)
    }

    /// Try to [`set_to`](`Number::set_to`) the provided target, returning [`Err(NumberError::CasFailed)`](`NumberError::CasFailed`) immediately if the compare-exchange failed.
    pub fn try_set_to(&self, target: NumericType) -> Result<NumericType, NumberError> {
        self.__raise_to(target, true, false, true)
    }

    /// [`try_set_to`](`Number::try_set_to`) with user bits ignored
    pub fn try_set_numeral(&self, target: NumericType) -> Result<NumericType, NumberError> {
        self.__raise_to(target, true, false, false)
    }

    #[inline(always)]
    fn __raise_to(&self, target: NumericType, __strict: bool, __retry: bool, __set_user_bits: bool) -> Result<NumericType, NumberError> {
        debug_assert!(
            is_valid_public_value(target),
            "target must fit in the public number range"
        );

        loop {
            let current = self.get_value()?;
            if is_suspended(current) {
                return Err(NumberError::Suspended);
            }

            let number = if __strict {
                if lt_masked(target, current, Self::NUMBER_MASK) {
                    return Err(NumberError::PassedTarget(current));
                }
                if current & Self::NUMBER_MASK == target & Self::NUMBER_MASK {
                    return Ok(current);
                }
                target & Self::NUMBER_MASK
            } else {
                max2_masked(current & Self::NUMBER_MASK, target & Self::NUMBER_MASK, Self::NUMBER_MASK)
            };

            let user_bits = if __set_user_bits { target } else { current }
                & Self::USER_MASK;

            let desired = number | user_bits;
            if desired == current {
                return Ok(current);
            }

            match self.slot.value.compare_exchange(
                current,
                desired,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(desired),
                Err(_) if __retry => {
                    core::hint::spin_loop();
                    continue;
                }
                Err(_) => return Err(NumberError::CasFailed),
            }
        }
    }

    /// Adds `delta` to the number.
    pub fn add(&self, delta: NumericType) -> Result<NumericType, NumberError> {
        self.__add(delta, None, false)
    }

    /// Adds `delta` to the number, setting user bits.
    pub fn adds(&self, delta: NumericType) -> Result<NumericType, NumberError> {
        self.__add(delta, None, true)
    }

    /// Adds `delta` if the number would be equal to or less than the `limit` after the addition, failing with
    /// [`NumberError::Exceeded(current)`](`NumberError::Exceeded`) otherwise.
    pub fn add_within(&self, delta: NumericType, limit: NumericType) -> Result<NumericType, NumberError> {
        self.__add(delta, Some(limit), false)
    }

    /// Adds `delta` if the number would be equal to or less than the `limit` after the addition, failing with
    /// [`NumberError::Exceeded(current)`](`NumberError::Exceeded`) otherwise, setting user bits.
    pub fn adds_within(&self, delta: NumericType, limit: NumericType) -> Result<NumericType, NumberError> {
        self.__add(delta, Some(limit), true)
    }

    #[inline(always)]
    fn __add(&self, delta: NumericType, limit: Option<NumericType>, __set_user_bits: bool) -> Result<NumericType, NumberError> {
        loop {
            let current = self.get_value()?;
            if is_suspended(current) {
                return Err(NumberError::Suspended);
            }

            if let Some(limit) = limit {
                let room = limit.wrapping_sub(current) & Self::NUMBER_MASK;
                if lt_masked(limit, current, Self::NUMBER_MASK) || delta & Self::NUMBER_MASK > room {
                    return Err(NumberError::Exceeded(current));
                }
            }

            let user_bits = if __set_user_bits { delta } else { current }
                & Self::USER_MASK;

            let next = (current.wrapping_add(delta) & Self::NUMBER_MASK) | user_bits;

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

    /// Suspend the number so that [`AtomicTrack::find_min`] begins pushing it forward rather as treating it as party to
    /// determining the minimum. After a number is put in suspended state, directly modifying the number without atomic compare
    /// exchange (through direct [`Number::atomic`], for instance) becomes riskier for that reason. The benefit is that it is able to
    /// maintain its current slot, so re-entering maintains the same atomic. User bits are also kept across suspends. Returns
    /// the last value before it was successfully put into suspend, or an error otherwise.
    pub fn suspend(&self) -> Result<NumericType, NumberError> {
        loop {
            let current = self.get_value()?;
            if is_suspended(current) {
                return Err(NumberError::Suspended);
            }
            match self.slot.value.compare_exchange(
                current,
                with_suspended_bit(current),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(current),
                Err(_) => {
                    core::hint::spin_loop();
                    continue;
                },
            }
        }
    }

    /// Resume at own floor.
    pub fn resume(&self) -> Result<NumericType, NumberError> {
        self.__resume(None, false)
    }

    /// Resume at `max(own floor, at_least)`.
    pub fn resume_from(&self, at_least: NumericType) -> Result<NumericType, NumberError> {
        self.__resume(Some(at_least), false)
    }

    /// Resume at exactly `at` or fail with [`NumberError::ExactFailed`] and stay suspended.
    pub fn resume_at(&self, at: NumericType) -> Result<NumericType, NumberError> {
        self.__resume(Some(at), true)
    }

    #[inline(always)]
    fn __resume(&self, at: Option<NumericType>, __exact: bool) -> Result<NumericType, NumberError> {
        if let Some(at) = at {
            debug_assert!(
                is_valid_public_value(at),
                "at must fit in the public number range"
            );
        }

        // Set the lock bit on the id before re-entering since a concurrent leave call shouldn't simultaneously clear a
        // slot whose value resume is making active, with an active value in an empty slot (tearing) as a result
        self.lock()?;
        let result = loop {
            let current = self.slot.value.load(Ordering::Acquire);
            if !is_suspended(current) {
                break Err(NumberError::NotSuspended);
            }

            let floor = current & Self::NUMBER_MASK;
            let target = match at {
                None => floor,
                Some(at) => match max2_special_masked(at, floor, __exact, Self::NUMBER_MASK) {
                    Some(target) => target,
                    None => break Err(NumberError::ExactFailed),
                },
            };
            let desired = (target & Self::NUMBER_MASK) | (current & Self::USER_MASK);
            match self.slot.value.compare_exchange(
                current,
                desired,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break Ok(desired),
                Err(_) => {
                    // This happens if find_min raises the value concurrently. Retry if that's the case.
                    core::hint::spin_loop();
                    continue;
                },
            }
        };
        self.slot.id.store(self.id, Ordering::Release);
        result
    }

    /// # Safety
    /// Callers must consider who else might be touching this number, in any way (including with provided functions like [`AtomicTrack::leave_concurrent`]).
    /// In addition, keep in mind what the value must follow. The monotonically growing
    /// part has to stay that way when changed manually, and the suspended bit has to be
    /// considered as well.
    pub unsafe fn atomic(&self) -> &'a AtomicType {
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

    static STATIC_TRACK: ArrayAtomicTrack<4> = ArrayAtomicTrack::new();

    const ONE_USER_BIT: u32 = NumericType::BITS - 1;
    const FLAG: NumericType = 1 << (NumericType::BITS - 2);
    const TWO_USER_BITS: u32 = NumericType::BITS - 2;
    const LOWER_FLAG_OF_TWO: NumericType = 1 << (NumericType::BITS - 3);

    #[test]
    fn user_bits_start_at_initial_and_reset_on_reentry() {
        let track = ArrayAtomicTrack::<1, ONE_USER_BIT, FLAG>::new();
        let a = track.enter(1).unwrap();
        let number = track.number(a).unwrap();
        assert_eq!(number.get(), Ok(FLAG));
        assert_eq!(number.adds(7), Ok(7));
        track.leave(a).unwrap();

        let b = track.enter(2).unwrap();
        assert_eq!(track.number(b).unwrap().get(), Ok(FLAG | 7));
    }

    #[test]
    fn raise_to_overwrites_user_bits_and_round_trips_get() {
        let track = ArrayAtomicTrack::<1, ONE_USER_BIT>::new();
        let number_id = track.enter(1).unwrap();
        let number = track.number(number_id).unwrap();

        assert_eq!(number.raise_to(FLAG | 5), Ok(FLAG | 5));
        assert_eq!(number.raise_to(3), Ok(5));
        assert_eq!(number.raise_to(FLAG), Ok(FLAG | 5));
        let value = number.get().unwrap();
        assert_eq!(number.raise_to(value), Ok(value));
    }

    #[test]
    fn adds_overwrites_user_bits_and_never_carries_the_counter_into_them() {
        let track = ArrayAtomicTrack::<1, TWO_USER_BITS, LOWER_FLAG_OF_TWO>::new();
        let counter = number_mask(TWO_USER_BITS);
        let high = LOWER_FLAG_OF_TWO << 1;
        let number_id = track.enter(1).unwrap();
        let number = track.number(number_id).unwrap();

        assert_eq!(number.adds(high | counter), Ok(high | counter));
        assert_eq!(number.adds(high | 1), Ok(high));
        assert_eq!(number.adds(LOWER_FLAG_OF_TWO | 5), Ok(LOWER_FLAG_OF_TWO | 5));
        assert_eq!(number.adds(high | LOWER_FLAG_OF_TWO), Ok(high | LOWER_FLAG_OF_TWO | 5));
        assert_eq!(number.adds(0), Ok(5));
    }

    #[test]
    fn add_leaves_user_bits() {
        let track = ArrayAtomicTrack::<1, TWO_USER_BITS, LOWER_FLAG_OF_TWO>::new();
        let counter = number_mask(TWO_USER_BITS);
        let high = LOWER_FLAG_OF_TWO << 1;
        let number_id = track.enter(1).unwrap();
        let number = track.number(number_id).unwrap();

        assert_eq!(number.add(counter), Ok(LOWER_FLAG_OF_TWO | counter));
        assert_eq!(number.add(1), Ok(LOWER_FLAG_OF_TWO)); // wraps within the counter
        assert_eq!(number.add(high | 5), Ok(LOWER_FLAG_OF_TWO | 5)); // delta's user bits are ignored
    }

    #[test]
    fn min_and_find_min_only_see_the_counter() {
        let track = ArrayAtomicTrack::<4, ONE_USER_BIT, FLAG>::new();
        let a = track.enter(1).unwrap();
        let b = track.enter(2).unwrap();
        track.number(a).unwrap().raise_to(FLAG | 10).unwrap();
        track.number(b).unwrap().raise_to(FLAG | 7).unwrap();

        assert_eq!(track.find_min(), (7, 0));
        assert_eq!(track.min(), 7);

        let c = track.enter(3).unwrap();
        assert_eq!(track.number(c).unwrap().get(), Ok(FLAG | 7));
    }

    #[test]
    fn custom_raw_constructors_apply_the_parameters() {
        let layout = AtomicTrack::layout(2).unwrap();
        unsafe {
            let ptr = NonNull::new(std::alloc::alloc(layout)).unwrap();
            let track: &AtomicTrack<_, ONE_USER_BIT, FLAG> = AtomicTrack::init_in_place(ptr, 2);
            let number_id = track.enter(1).unwrap();
            assert_eq!(track.number(number_id).unwrap().get(), Ok(FLAG));

            let cast: &AtomicTrack<_, ONE_USER_BIT, FLAG> = AtomicTrack::from_raw(ptr, 2);
            assert_eq!(cast.number(number_id).unwrap().adds(0), Ok(0));
            assert_eq!(track.number(number_id).unwrap().get(), Ok(0));

            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }

    #[test]
    fn static_track_is_usable() {
        let track: &AtomicTrack = &STATIC_TRACK;
        let number = track.enter(3).unwrap();
        assert_eq!(track.number(number).unwrap().raise_to(6), Ok(6));
        assert_eq!(track.find_min(), (6, 0));
        track.leave(number).unwrap();
    }

    #[test]
    fn layout_matches_fixed_track() {
        fn check<const N: usize>() {
            let fixed = ArrayAtomicTrack::<N>::new();
            let layout = AtomicTrack::layout(N).unwrap();
            let unsized_track: &AtomicTrack = &fixed;
            assert_eq!(size_of::<ArrayAtomicTrack<N>>(), layout.size());
            assert_eq!(align_of::<ArrayAtomicTrack<N>>(), layout.align());
            assert_eq!(core::mem::size_of_val(unsized_track), layout.size());
            assert_eq!(core::mem::align_of_val(unsized_track), layout.align());
            assert_eq!(unsized_track.capacity(), N);
        }
        check::<1>();
        check::<2>();
        check::<8>();
        check::<64>();

        assert_eq!(AtomicTrack::layout(0), None);
        assert_eq!(AtomicTrack::layout(3), None);
        assert_eq!(AtomicTrack::layout(usize::MAX / 2 + 1), None);
    }

    #[test]
    fn init_in_place_and_from_raw_share_state() {
        let layout = AtomicTrack::layout(4).unwrap();
        unsafe {
            let ptr = NonNull::new(std::alloc::alloc(layout)).unwrap();
            let track = AtomicTrack::init_in_place_default(ptr, 4);
            assert_eq!(track.capacity(), 4);
            assert_eq!(track.min(), 0);

            let number = track.enter(3).unwrap();
            track.number(number).unwrap().raise_to(9).unwrap();

            let cast = AtomicTrack::from_raw_default(ptr, 4);
            assert_eq!(cast.capacity(), 4);
            assert_eq!(cast.recover(3), Some(number));
            assert_eq!(cast.with_id(3, |lane| lane.get()).unwrap(), Ok(9));
            assert_eq!(cast.find_min(), (9, 0));
            assert_eq!(track.min(), 9);

            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }

    #[test]
    fn seed_is_the_birth_address_and_survives_being_seen_elsewhere() {
        let array = ArrayAtomicTrack::<2>::new();
        let array_track: &AtomicTrack = &array;
        assert_eq!(array_track.seed(), (array_track as *const AtomicTrack).addr() as u64);

        let layout = AtomicTrack::layout(2).unwrap();
        unsafe {
            let first = NonNull::new(std::alloc::alloc(layout)).unwrap();
            let second = NonNull::new(std::alloc::alloc(layout)).unwrap();
            let track = AtomicTrack::init_in_place_default(first, 2);
            assert_eq!(track.seed(), first.as_ptr().addr() as u64);
            assert_eq!(AtomicTrack::from_raw_default(first, 2).seed(), track.seed());

            ptr::copy_nonoverlapping(first.as_ptr(), second.as_ptr(), layout.size());
            let elsewhere = AtomicTrack::from_raw_default(second, 2);
            assert_eq!(elsewhere.seed(), first.as_ptr().addr() as u64);

            std::alloc::dealloc(first.as_ptr(), layout);
            std::alloc::dealloc(second.as_ptr(), layout);
        }
    }

    #[test]
    fn init_in_place_resets_dirty_memory() {
        let layout = AtomicTrack::layout(2).unwrap();
        unsafe {
            let ptr = NonNull::new(std::alloc::alloc(layout)).unwrap();
            ptr.as_ptr().write_bytes(0xAB, layout.size());
            let track = AtomicTrack::init_in_place_default(ptr, 2);
            assert_eq!(track.min(), 0);
            assert_eq!(track.recover(1), None);
            let number = track.enter(1).unwrap();
            assert_eq!(track.number(number).unwrap().get(), Ok(0));
            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of two")]
    fn init_in_place_rejects_non_power_of_two() {
        let mut buffer = ArrayAtomicTrack::<4>::new();
        unsafe {
            AtomicTrack::init_in_place_default(NonNull::from(&mut buffer).cast(), 3);
        }
    }

    #[test]
    fn enter_recover_and_use_number() {
        let track = ArrayAtomicTrack::<8>::new();

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
        let track = ArrayAtomicTrack::<4>::new();
        assert!(track.enter(123).is_ok());
        assert_eq!(track.enter(123), Err(EnterError::AlreadyPresent));
    }

    #[test]
    fn same_id_can_identify_distinct_placements_after_a_hole() {
        let track = ArrayAtomicTrack::<4>::new();
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
        let track = ArrayAtomicTrack::<4>::new();
        assert_eq!(track.enter(SUSPENDED_BIT), Err(EnterError::InvalidId));
    }

    #[test]
    fn offset_is_probe_delta() {
        let track = ArrayAtomicTrack::<4>::new();
        let first = track.enter(1).unwrap();
        let second = track.enter(5).unwrap();

        assert_eq!(first.offset, 0);
        assert_eq!(second.offset, 1);
        assert_eq!(track.recover(1), Some(first));
        assert_eq!(track.recover(5), Some(second));
    }

    #[test]
    fn find_min_advances_global_min() {
        let track = ArrayAtomicTrack::<8>::new();
        let a = track.enter(1).unwrap();
        let b = track.enter(2).unwrap();

        track
            .with_number(a, |lane| lane.raise_to(10).unwrap())
            .unwrap();
        track
            .with_number(b, |lane| lane.raise_to(7).unwrap())
            .unwrap();

        assert_eq!(track.find_min(), (7, 0));
        assert_eq!(track.min(), 7);

        track.with_number(b, |lane| lane.add(5).unwrap()).unwrap();
        assert_eq!(track.find_min(), (10, 7));
        assert_eq!(track.min(), 10);
    }

    #[test]
    fn leave_and_reenter_reuse_lane_with_new_id() {
        let track = ArrayAtomicTrack::<2>::new();
        let a = track.enter(10).unwrap();
        track
            .with_number(a, |lane| lane.raise_to(9).unwrap())
            .unwrap();
        assert_eq!(track.find_min(), (9, 0));

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
        let track = ArrayAtomicTrack::<1>::new();
        let a = track.enter(1).unwrap();

        track
            .with_number(a, |lane| lane.raise_to(12).unwrap())
            .unwrap();
        track.leave(a).unwrap();

        assert_eq!(track.find_min(), (0, 0));
        assert_eq!(track.min(), 0);

        let b = track.enter(2).unwrap();
        assert_eq!(
            track.with_number(b, |lane| lane.get()).unwrap().unwrap(),
            12
        );
    }

    #[test]
    fn leave_concurrent_bumps_reentry_floor() {
        let track = ArrayAtomicTrack::<1>::new();
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
        let track = ArrayAtomicTrack::<1>::new();
        let number = track.enter(1).unwrap();
        let quarter = NumericType::MAX / 4;
        let mut previous = 0;

        for delta in [quarter, quarter, 1] {
            let value = track
                .with_number(number, |lane| lane.add(delta).unwrap())
                .unwrap();
            assert_eq!(track.find_min(), (value, previous));
            assert_eq!(track.min(), value);
            previous = value;
        }

        assert_eq!(track.min(), MAX_PUBLIC);
        track.leave_concurrent(number).unwrap();
        assert_eq!(track.find_min(), (MAX_PUBLIC, MAX_PUBLIC));

        let replacement = track.enter(2).unwrap();
        assert_eq!(
            track.with_number(replacement, |lane| lane.get()).unwrap(),
            Ok(0)
        );
        assert_eq!(track.find_min(), (0, MAX_PUBLIC));
        assert_eq!(track.min(), 0);
    }

    #[test]
    fn raise_to_progresses_across_wrap() {
        let track = ArrayAtomicTrack::<1>::new();
        let number = track.enter(1).unwrap();
        let quarter = NumericType::MAX / 4;
        let mut previous = 0;

        for delta in [quarter, quarter, 1] {
            let value = track
                .with_number(number, |lane| lane.add(delta).unwrap())
                .unwrap();
            assert_eq!(track.find_min(), (value, previous));
            previous = value;
        }

        assert_eq!(track.min(), MAX_PUBLIC);
        assert_eq!(
            track.with_number(number, |lane| lane.raise_to(1)).unwrap(),
            Ok(1)
        );
        assert_eq!(track.find_min(), (1, MAX_PUBLIC));
        assert_eq!(track.min(), 1);
    }

    #[test]
    fn leave_can_beat_an_in_flight_update() {
        let track = Arc::new(ArrayAtomicTrack::<4>::new());
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

    #[test]
    fn find_min_bounded_advances_without_active_lanes() {
        let track = ArrayAtomicTrack::<4>::new();
        assert_eq!(track.find_min_bounded(8), (8, 0));
        assert_eq!(track.min(), 8);

        // Suspended floors were dragged to the bound, so new entrants start there.
        let a = track.enter(1).unwrap();
        assert_eq!(track.number(a).unwrap().get(), Ok(8));
    }

    #[test]
    fn find_min_bounded_ranges_dont_overlap() {
        let track = ArrayAtomicTrack::<4>::new();
        let a = track.enter(1).unwrap();
        track.number(a).unwrap().raise_to(3).unwrap();

        assert_eq!(track.find_min_bounded(8), (3, 0));
        assert_eq!(track.find_min_bounded(8), (3, 3)); // nothing new
        track.number(a).unwrap().raise_to(5).unwrap();
        assert_eq!(track.find_min_bounded(4), (4, 3)); // bound
        track.leave(a).unwrap();
        assert_eq!(track.find_min_bounded(9), (9, 4));
    }

    #[test]
    fn find_min_bounded_masks_the_bound() {
        let track = ArrayAtomicTrack::<2>::new();
        assert_eq!(track.find_min_bounded(SUSPENDED_BIT | 5), (5, 0));
        assert_eq!(track.min(), 5);
        let a = track.enter(1).unwrap(); // would be InvalidValue if min had picked up the MSB
        assert_eq!(track.number(a).unwrap().get(), Ok(5));
    }

    #[test]
    fn concurrent_find_min_bounded_ranges_tile_without_gaps_or_overlap() {
        const THREADS: usize = 4;
        let iterations = if cfg!(miri) { 50 } else { 20_000 };
        let track = Arc::new(ArrayAtomicTrack::<4>::new());
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: std::vec::Vec<_> = (0..THREADS)
            .map(|t| {
                let (track, barrier) = (track.clone(), barrier.clone());
                thread::spawn(move || {
                    barrier.wait();
                    let mut claimed = std::vec::Vec::new();
                    for i in 0..iterations {
                        let (new, previous) = track.find_min_bounded((i * THREADS + t) as NumericType);
                        if new != previous {
                            claimed.push((previous, new));
                        }
                    }
                    claimed
                })
            })
            .collect();
        let mut ranges: std::vec::Vec<_> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
        ranges.sort();
        let mut cursor = 0;
        for (previous, new) in ranges {
            assert_eq!(previous, cursor, "gap or overlap");
            cursor = new;
        }
        assert_eq!(cursor, track.min());
    }

    #[test]
    fn enter_at_is_exact() {
        let track = ArrayAtomicTrack::<2>::new();
        let a = track.enter_at(1, 6).unwrap();
        assert_eq!(track.number(a).unwrap().get(), Ok(6));
        assert_eq!(track.find_min_bounded(10), (6, 0)); // capped by a
        assert_eq!(track.enter_at(2, 4), Err(EnterError::ExactFailed)); // below min
        // Slot 0 was scanned before a's slot, while running_min was still 10, so its floor is 10:
        // floors are per slot, and an exact enter on an id is contingent on whichever empty slot enter found for the id and checked first.
        assert_eq!(track.enter_at(2, 7), Err(EnterError::ExactFailed));
        let b = track.enter_at(2, 10).unwrap();
        assert_eq!(track.number(b).unwrap().get(), Ok(10));
    }

    #[test]
    fn enter_at_releases_the_slot_it_rejects() {
        let track = ArrayAtomicTrack::<1>::new();
        let a = track.enter(1).unwrap();
        track.number(a).unwrap().raise_to(100).unwrap();
        track.leave(a).unwrap(); // floor 100, min still 0

        assert_eq!(track.enter_at(2, 50), Err(EnterError::ExactFailed));
        assert_eq!(track.recover(2), None);
        let b = track.enter_at(2, 100).unwrap(); // not AlreadyPresent / Full since the slot was released
        assert_eq!(track.number(b).unwrap().get(), Ok(100));
    }

    #[test]
    fn set_to_is_exact_and_rejects_a_passed_target() {
        let track = ArrayAtomicTrack::<2>::new();
        let a = track.enter(1).unwrap();
        let number = track.number(a).unwrap();

        assert_eq!(number.set_to(5), Ok(5));
        assert_eq!(number.set_to(5), Ok(5));
        assert_eq!(number.set_to(3), Err(NumberError::PassedTarget(5)));
        assert_eq!(number.get(), Ok(5));
        assert_eq!(number.try_set_to(9), Ok(9));
        assert_eq!(number.try_set_to(8), Err(NumberError::PassedTarget(9)));
        assert_eq!(number.try_raise_to(4), Ok(9)); // raising should never lower
        assert_eq!(number.try_raise_to(12), Ok(12));

        track.leave(a).unwrap();
        assert_eq!(number.set_to(20), Err(NumberError::NotFound));
        assert_eq!(number.try_raise_to(20), Err(NumberError::NotFound));
    }

    #[test]
    fn set_to_sets_the_user_bits_from_the_target_only_when_it_modifies() {
        let track = ArrayAtomicTrack::<1, ONE_USER_BIT>::new();
        let number = track.number(track.enter(1).unwrap()).unwrap();
        assert_eq!(number.set_to(FLAG | 5), Ok(FLAG | 5));
        assert_eq!(number.set_to(5), Ok(FLAG | 5)); // already at 5, so the user bits are left alone
        assert_eq!(number.set_to(7), Ok(7));
        assert_eq!(number.set_to(FLAG | 4), Err(NumberError::PassedTarget(7)));
    }

    #[test]
    fn set_numeral_leaves_the_user_bits() {
        let track = ArrayAtomicTrack::<1, ONE_USER_BIT>::new();
        let number = track.number(track.enter(1).unwrap()).unwrap();
        assert_eq!(number.set_to(FLAG | 5), Ok(FLAG | 5));
        assert_eq!(number.set_numeral(7), Ok(FLAG | 7));
        assert_eq!(number.set_numeral(7), Ok(FLAG | 7));
        assert_eq!(number.set_to(9), Ok(9));
        assert_eq!(number.set_numeral(FLAG | 12), Ok(12)); // the target's user bits are ignored too
        assert_eq!(number.set_numeral(11), Err(NumberError::PassedTarget(12)));

        assert_eq!(number.set_to(FLAG | 13), Ok(FLAG | 13));
        assert_eq!(number.try_set_numeral(15), Ok(FLAG | 15));
        assert_eq!(number.try_set_numeral(14), Err(NumberError::PassedTarget(FLAG | 15)));
    }

    #[test]
    fn try_updates_only_fail_on_contention_and_set_to_retries_through_it() {
        const ROUNDS: NumericType = if cfg!(miri) { 50 } else { 20_000 };
        let track = ArrayAtomicTrack::<2>::new();
        let a = track.enter(1).unwrap();
        let b = track.enter(2).unwrap();
        let track = &track;

        thread::scope(|s| {
            // `a`: one thread keeps adding while another makes single no-op attempts that can only lose the CAS.
            s.spawn(move || {
                let number = track.number(a).unwrap();
                for _ in 0..ROUNDS {
                    number.add(1).unwrap();
                }
            });
            s.spawn(move || {
                let number = track.number(a).unwrap();
                for _ in 0..ROUNDS {
                    match number.try_raise_to(0) {
                        Ok(_) | Err(NumberError::CasFailed) => {},
                        other => panic!("unexpected {other:?}"),
                    }
                }
            });
            // `b`: two threads race to set interleaved targets.
            for parity in 0..2 {
                s.spawn(move || {
                    let number = track.number(b).unwrap();
                    for target in (parity..ROUNDS).step_by(2) {
                        match number.set_to(target) {
                            Ok(value) => assert_eq!(value, target),
                            Err(NumberError::PassedTarget(_)) => {},
                            other => panic!("unexpected {other:?}"),
                        }
                    }
                });
            }
        });

        assert_eq!(track.number(a).unwrap().get(), Ok(ROUNDS));
        assert_eq!(track.number(b).unwrap().get(), Ok(ROUNDS - 1));
    }

    #[test]
    fn add_within_stops_at_the_limit() {
        let track = ArrayAtomicTrack::<1>::new();
        let number = track.number(track.enter(1).unwrap()).unwrap();

        assert_eq!(number.add_within(3, 5), Ok(3));
        assert_eq!(number.add_within(2, 5), Ok(5)); // exactly on the limit is fine
        assert_eq!(number.add_within(1, 5), Err(NumberError::Exceeded(5)));
        assert_eq!(number.add_within(0, 5), Ok(5));
        assert_eq!(number.add_within(0, 4), Err(NumberError::Exceeded(5)));
        assert_eq!(number.get(), Ok(5));
    }

    #[test]
    fn add_within_leaves_user_bits_and_adds_within_sets_them() {
        let track = ArrayAtomicTrack::<1, ONE_USER_BIT>::new();
        let number = track.number(track.enter(1).unwrap()).unwrap();
        assert_eq!(number.adds_within(FLAG | 4, 10), Ok(FLAG | 4));
        assert_eq!(number.add_within(1, 10), Ok(FLAG | 5));
        assert_eq!(number.adds_within(1, 10), Ok(6));
        assert_eq!(number.add_within(FLAG | 2, 10), Ok(8));
        assert_eq!(number.adds_within(FLAG | 3, 10), Err(NumberError::Exceeded(8)));
    }

    #[test]
    fn add_within_works_across_wrap() {
        let track = ArrayAtomicTrack::<1>::new();
        let number = track.number(track.enter(1).unwrap()).unwrap();
        // Do it in two steps since MAX_PUBLIC - 1 is more than half the range away from 0 (it counts as behind).
        let half = MAX_PUBLIC / 2;
        number.add(half).unwrap();
        assert_eq!(number.add(half), Ok(MAX_PUBLIC - 1));

        // MAX_PUBLIC - 1 + 4 wraps to 2.
        assert_eq!(number.add_within(5, 2), Err(NumberError::Exceeded(MAX_PUBLIC - 1)));
        assert_eq!(number.add_within(4, 2), Ok(2));
        assert_eq!(number.add_within(1, 2), Err(NumberError::Exceeded(2)));
    }

    #[test]
    fn concurrent_add_within_hands_out_exactly_up_to_the_limit() {
        const LIMIT: NumericType = if cfg!(miri) { 40 } else { 40_000 };
        const THREADS: usize = 4;
        let track = ArrayAtomicTrack::<1>::new();
        let a = track.enter(1).unwrap();
        let track = &track;

        let claimed: NumericType = thread::scope(|s| {
            let handles: std::vec::Vec<_> = (0..THREADS)
                .map(|_| {
                    s.spawn(move || {
                        let number = track.number(a).unwrap();
                        let mut claimed = 0;
                        loop {
                            match number.add_within(1, LIMIT) {
                                Ok(_) => claimed += 1,
                                Err(NumberError::Exceeded(found)) => {
                                    assert_eq!(found, LIMIT);
                                    return claimed;
                                },
                                other => panic!("unexpected {other:?}"),
                            }
                        }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });

        assert_eq!(claimed, LIMIT);
        assert_eq!(track.number(a).unwrap().get(), Ok(LIMIT));
    }
}
