// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Shiko Kudo
//
// Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
// or the MIT license (http://opensource.org), at your option.

//! A few things to note:
//! - All key-based (and not [`NumberId`] based) apis select and operate on ***only*** the first `Number` recovered from the ring with the provided id.
//! - `at_least` is not checked for the reserved MSB bit. Please don't use numbers that are that high.

//TODO: raise_to, add, and both leave methods notify even on errors or no-op updates. With wake_all and a shared futex pool, this can create wake storms. Change eventually if that becomes a problem.

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::time::Instant;

use core::{alloc::Layout, hash::Hasher, ptr::{self, NonNull}, sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering}};

#[cfg(feature = "std")]
use std::boxed::Box;

use crate::{AtomicTrack, EMPTY_ID, Slot, cache_padded::CachePadded, EnterError, LeaveError, Number, NumberError, NumberId, env_or_default, futex, is_key_locked, is_suspended, key_bits, math::{AtomicType, NumericType, number_mask, gte_masked, user_mask}, spin_for_step, utils::{MAX_LOOPS_BEFORE_SLEEP, MAX_SPINS, yield_now}, without_suspended_bit};

const DEFAULT_NUM_FUTEXES: usize = env_or_default!("ATOMICTRACK_FUTEX_POOL_SIZE", "1024", usize);
const _: () = {
    assert!(DEFAULT_NUM_FUTEXES > 0, "FUTEX pool size must be something that isn't zero!");
    assert!(DEFAULT_NUM_FUTEXES.is_power_of_two(), "FUTEX pool size must be power of two!");
};

pub const WAITING_POOL_JOIN_NONE: u64 = 0;

struct ListofAtomicsWrapper {
    s: &'static [AtomicU32],
    waiting_pool_join: u64,
}

impl ListofAtomicsWrapper {
    #[inline]
    fn pinfo(&self) -> u128 {
        if self.waiting_pool_join == WAITING_POOL_JOIN_NONE {
            0
        } else {
            ((self.waiting_pool_join as u128) << 64) | self.s.as_ptr().addr() as u128
        }
    }
}

static FUTEXES: AtomicPtr<ListofAtomicsWrapper> = AtomicPtr::new(
    &ListofAtomicsWrapper { s: &[], waiting_pool_join: WAITING_POOL_JOIN_NONE } as *const _ as *mut _
);

static FUTEX_POOL_FREE_INIT: AtomicU32 = AtomicU32::new(1);

fn prev_power_of_two(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        1 << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Sets the waiting pool (a simple array of atomic u32s).
///
/// Having `waiting_pool_join` be a non-zero value causes futex operations to assume cross-process. `WAITING_POOL_JOIN_NONE`
/// gives you process-private futexes as usual. Note that it is your responsibility to make sure that `ptr` is in a shared memory
/// location if you specify `waiting_pool_join`. Within each process, they *can* be mapped at different base addresses.
///
/// Returns 0 for success, 1 for when the pool was set by someone else instead (you can retry), -1 for invalid arguments.
///
/// Please call *before* you start waiting, or the default initializer when the length is zero will kick in.
///
/// **Important: `len` is rounded down to the previous power of two.**
///
/// # Safety
/// `ptr` must point to a valid length of accessible memory holding aligned 32-bit
/// values that remain valid until this library is unloaded, and that everyone also only ever
/// accesses through atomic operations.
/// **Also, if you have existing waits not set on a timeout, this might strand them forever.**
#[allow(non_snake_case)]
#[cfg(feature = "std")]
#[cfg_attr(feature = "capi", unsafe(no_mangle))]
pub unsafe extern "C" fn A_T_set_waiting_pool(ptr: *const AtomicU32, mut len: usize, waiting_pool_join: u64) -> i32 {
    len = prev_power_of_two(len);
    if len == 0 {
        return -1;
    }
    if ptr.is_null()
        || !ptr.is_aligned()
        || len > isize::MAX as usize / size_of::<AtomicU32>()
    {
        return -1;
    }
    let slice: &'static [AtomicU32] = unsafe { core::slice::from_raw_parts(ptr, len) };
    let wrapper = Box::new(ListofAtomicsWrapper { s: slice, waiting_pool_join });
    let base = FUTEX_POOL_FREE_INIT.load(Ordering::Acquire);
    if base % 2 == 1 {
        // settled state, can change.
        while FUTEX_POOL_FREE_INIT.load(Ordering::Acquire) == base {
            if let Ok(_) = FUTEX_POOL_FREE_INIT.compare_exchange_weak(base, base.wrapping_add(1), Ordering::Acquire, Ordering::Relaxed) {
                FUTEXES.store(Box::leak(wrapper), Ordering::Release);
                FUTEX_POOL_FREE_INIT.store(base.wrapping_add(2), Ordering::Release);
                return 0;
            }
        }
    }
    1
}

#[inline]
fn futexes() -> &'static ListofAtomicsWrapper {
    let futexes = unsafe { &*FUTEXES.load(Ordering::Acquire) };
    if !futexes.s.is_empty() {
        return futexes;
    }
    futexes_slow()
}

#[cold]
#[inline(never)]
fn futexes_slow() -> &'static ListofAtomicsWrapper {
    let s = unsafe {
        Box::<[AtomicU32]>::new_zeroed_slice(DEFAULT_NUM_FUTEXES).assume_init()
    };
    let mut wrapper = Box::new(
        ListofAtomicsWrapper { s: &[], waiting_pool_join: WAITING_POOL_JOIN_NONE },
    );
    let mut base = FUTEX_POOL_FREE_INIT.load(Ordering::Acquire);
    let mut futexes = unsafe { &*FUTEXES.load(Ordering::Acquire) };
    'outer: while futexes.s.is_empty() {
        if base % 2 == 0 {
            while FUTEX_POOL_FREE_INIT.load(Ordering::Acquire) % 2 == 0 {
                yield_now();
            }
        } else {
            // settled state, can change.
            while FUTEX_POOL_FREE_INIT.load(Ordering::Acquire) == base {
                if let Ok(_) = FUTEX_POOL_FREE_INIT.compare_exchange_weak(base, base.wrapping_add(1), Ordering::Acquire, Ordering::Relaxed) {
                    wrapper.s = Box::leak(s);
                    FUTEXES.store(Box::leak(wrapper), Ordering::Release);
                    FUTEX_POOL_FREE_INIT.store(base.wrapping_add(2), Ordering::Release);
                    futexes = unsafe { &*FUTEXES.load(Ordering::Acquire) };
                    break 'outer;
                }
            }
        }
        base = FUTEX_POOL_FREE_INIT.load(Ordering::Acquire);
        futexes = unsafe { &*FUTEXES.load(Ordering::Acquire) };
    }
    futexes
}

fn futex(i: usize) -> (&'static AtomicU32, u128) {
    let futexes = futexes();
    (&futexes.s[i & (futexes.s.len() - 1)], futexes.pinfo())
}

fn get_futex<const L: u32, const I: NumericType>(track: &AtomicTrack<[CachePadded<Slot>], L, I>, id: NumericType) -> (&'static AtomicU32, u128) {
    let mut hasher = hasher::RhmHasher::default();
    hasher.write(&track.seed().to_le_bytes());
    #[allow(clippy::unnecessary_cast)]
    hasher.write(&(id as u64).to_le_bytes());
    futex(hasher.finish() as usize)
}

/// For convenience, a stable hash that gives you an id from a string name.
pub const fn id_of(name: &str) -> NumericType {
    let bytes = name.as_bytes();
    let mut state = 0;
    loop {
        state = hasher::rapidhash_micro_with_seed(bytes, state);
        let id = key_bits(state as NumericType);
        if id != EMPTY_ID {
            return id;
        }
    }
}

#[repr(C)]
struct Shared<const L: u32, const I: NumericType, S: ?Sized = [CachePadded<Slot>]> {
    handle_count: AtomicUsize,
    track: AtomicTrack<S, L, I>,
}

fn shared_layout(capacity: usize) -> Layout {
    assert!(
        capacity.is_power_of_two(),
        "capacity must be a power of two"
    );
    let track = AtomicTrack::layout(capacity)
        .expect("capacity is too large");
    let (layout, _) = Layout::new::<AtomicUsize>()
        .extend(track)
        .expect("capacity is too large");
    layout.pad_to_align()
}

pub struct AtomicTrackWaiting<
    const USER_BIT_LOW: u32 = { NumericType::BITS },
    const USER_BITS_DEFAULT: NumericType = 0,
> {
    track: NonNull<AtomicTrack<[CachePadded<Slot>], USER_BIT_LOW, USER_BITS_DEFAULT>>,
    shared: Option<NonNull<Shared<USER_BIT_LOW, USER_BITS_DEFAULT>>>,
}

unsafe impl<const L: u32, const I: NumericType> Send for AtomicTrackWaiting<L, I> {}
unsafe impl<const L: u32, const I: NumericType> Sync for AtomicTrackWaiting<L, I> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitError {
    InvalidId,
    NotFound,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitGteStatus {
    Reached = 0,
    ReservedBitsModified = 1,
    TimedOut = 2,
}

pub struct NumberWaiting<
    'a,
    'b,
    const USER_BIT_LOW: u32 = { NumericType::BITS },
    const USER_BITS_DEFAULT: NumericType = 0,
> {
    inner: Number<'a, USER_BIT_LOW>,
    atomic_track_waiting: &'b AtomicTrackWaiting<USER_BIT_LOW, USER_BITS_DEFAULT>,
}

impl<const L: u32, const I: NumericType> Clone for AtomicTrackWaiting<L, I> {
    fn clone(&self) -> Self {
        if let Some(shared) = self.shared {
            unsafe { shared.as_ref() }.handle_count.fetch_add(1, Ordering::Relaxed);
        }
        Self { track: self.track, shared: self.shared }
    }
}

impl<const L: u32, const I: NumericType> Drop for AtomicTrackWaiting<L, I> {
    fn drop(&mut self) {
        let Some(shared) = self.shared else {
            return;
        };
        if unsafe { shared.as_ref() }.handle_count.fetch_sub(1, Ordering::Release) == 1 {
            core::sync::atomic::fence(Ordering::Acquire);
            let layout = shared_layout(self.track().capacity());
            unsafe { std::alloc::dealloc(shared.as_ptr().cast::<u8>(), layout) };
        }
    }
}

impl AtomicTrackWaiting {
    pub fn new_default(capacity: usize) -> Self {
        Self::new(capacity)
    }
}

impl<const L: u32, const I: NumericType> AtomicTrackWaiting<L, I> {
    const NUMBER_MASK: NumericType = number_mask(L);
    const USER_MASK: NumericType = user_mask(L);

    pub fn new(capacity: usize) -> Self {
        let layout = shared_layout(capacity);
        unsafe {
            let Some(ptr) = NonNull::new(std::alloc::alloc(layout)) else {
                std::alloc::handle_alloc_error(layout);
            };
            let shared = ptr::slice_from_raw_parts_mut(ptr.as_ptr().cast::<CachePadded<Slot>>(), capacity) as *mut Shared<L, I>;
            (&raw mut (*shared).handle_count).write(AtomicUsize::new(1));
            let track = &raw mut (*shared).track;
            AtomicTrack::<[CachePadded<Slot>], L, I>::init_in_place(NonNull::new_unchecked(track.cast::<u8>()), capacity);
            Self { track: NonNull::new_unchecked(track), shared: Some(NonNull::new_unchecked(shared)) }
        }
    }

    /// Wraps a track that is already initialized (for example, one placed in shared memory with [`AtomicTrack::init_in_place`] and reopened elsewhere with [`AtomicTrack::from_raw`]).
    /// The handle, and its clones, never free the track.
    ///
    /// # Safety
    /// `track` must point to an initialized track that stays valid (mapped, never freed, never moved) for as long as this handle,
    /// any clone of it, and any [`NumberWaiting`] or ongoing wait derived from them exist.
    pub unsafe fn from_raw(track: NonNull<AtomicTrack<[CachePadded<Slot>], L, I>>) -> Self {
        Self { track, shared: None }
    }

    /// Wraps a track that stays valid for the rest of the program, such as one in a shared memory mapping that is never unmapped. See [`from_raw`](`AtomicTrackWaiting::from_raw`).
    pub fn from_static(track: &'static AtomicTrack<[CachePadded<Slot>], L, I>) -> Self {
        unsafe { Self::from_raw(NonNull::from(track)) }
    }

    #[inline]
    fn track(&self) -> &AtomicTrack<[CachePadded<Slot>], L, I> {
        unsafe { self.track.as_ref() }
    }

    pub fn capacity(&self) -> usize {
        self.track().capacity()
    }

    /// [`find_min`](`AtomicTrackWaiting::find_min`) updates the current global min, this one just reads it.
    pub fn min(&self) -> NumericType {
        self.track().min()
    }

    pub fn find_min(&self) -> NumericType {
        self.track().find_min()
    }

    pub fn enter(&self, id: NumericType) -> Result<NumberId, EnterError> {
        self.enter_from(id, self.track().min())
    }

    pub fn enter_from(&self, id: NumericType, at_least: NumericType) -> Result<NumberId, EnterError> {
        match self.track().enter_from(id, at_least) {
            Ok(number_id) => {
                let (f, pinfo) = get_futex(self.track(), number_id.id);
                f.fetch_add(1, Ordering::Release);
                let _ = futex::wake_all(f, pinfo);
                Ok(number_id)
            },
            Err(e) => Err(e),
        }
    }

    /// Recover a [`NumberId`] from a key. This is usually fast but since it *can* scan the whole ring, keeping the [`NumberId`] directly is better.
    pub fn recover(&self, id: NumericType) -> Option<NumberId> {
        self.track().recover(id)
    }

    /// Just like [`recover`](`AtomicTrackWaiting::recover`), this is usually fast but since it *can* scan the whole ring, using [`with_number`](`AtomicTrackWaiting::with_number`) is better if you can.
    pub fn with_id<R>(
        &self,
        id: NumericType,
        f: impl FnOnce(NumberWaiting<'_, '_, L, I>) -> R,
    ) -> Result<R, NumberError> {
        self.track().with_id(id, |number| {
            f(NumberWaiting {
                inner: number,
                atomic_track_waiting: self,
            })
        })
    }

    pub fn with_number<R>(
        &self,
        number: NumberId,
        f: impl FnOnce(NumberWaiting<'_, '_, L, I>) -> R,
    ) -> Result<R, NumberError> {
        self.track().with_number(number, |number| {
            f(NumberWaiting {
                inner: number,
                atomic_track_waiting: self,
            })
        })
    }

    pub fn number(&self, number: NumberId) -> Result<NumberWaiting<'_, '_, L, I>, NumberError> {
        self.track().number(number).map(|number| NumberWaiting {
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

    pub fn wait_for_number(&self, id: NumericType) -> Result<NumberWaiting<'_, '_, L, I>, WaitError> {
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

    pub fn wait_for_number_timeout(&self, id: NumericType, timeout_ns: u64) -> Result<NumberWaiting<'_, '_, L, I>, WaitError> {
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
        match self.__wait_for_enter_and_at_least_timeout(id, at_least, None, None)? {
            (WaitGteStatus::Reached, value, number_id) => Ok((value, number_id)),
            _ => unreachable!(),
        }
    }

    pub fn wait_gte_timeout(&self, id: NumericType, at_least: NumericType, timeout_ns: u64) -> Result<(bool, NumericType, NumberId), WaitError> {
        let (status, value, number_id) = self.__wait_for_enter_and_at_least_timeout(id, at_least, None, Some((timeout_ns, Instant::now())))?;
        Ok((status == WaitGteStatus::Reached, value, number_id))
    }

    /// See [`NumberWaiting::wait_gte_sicrbm`].
    pub fn wait_gte_sicrbm(&self, id: NumericType, at_least: NumericType, expected_user_bits: NumericType) -> Result<(WaitGteStatus, NumericType, NumberId), WaitError> {
        self.__wait_for_enter_and_at_least_timeout(id, at_least, Some(expected_user_bits), None)
    }

    pub fn wait_gte_timeout_sicrbm(&self, id: NumericType, at_least: NumericType, expected_user_bits: NumericType, timeout_ns: u64) -> Result<(WaitGteStatus, NumericType, NumberId), WaitError> {
        self.__wait_for_enter_and_at_least_timeout(id, at_least, Some(expected_user_bits), Some((timeout_ns, Instant::now())))
    }

    /// Loads the current futex word for `id`. Check your own condition, then pass this loaded value to [`wait_spurious`](`AtomicTrackWaiting::wait_spurious`) if you still need to wait. Any signal on `id` after the load makes the wait return immediately, including spuriously.
    pub fn futex_word_value(&self, id: NumericType) -> Result<u32, WaitError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        Ok(get_futex(self.track(), id).0.load(Ordering::Acquire))
    }

    /// Sleeps while the futex word for `id` still equals `expected`. Can return spuriously, especially since the number of futexes is fixed and futexes are shared with other ids.
    pub fn wait_spurious(&self, id: NumericType, expected: u32) -> Result<(), WaitError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        let (f, pinfo) = get_futex(self.track(), id);
        futex::wait(f, expected, pinfo);
        Ok(())
    }

    /// Returns `Ok(false)` only when the operating system reports that the timeout elapsed.
    pub fn wait_spurious_timeout(&self, id: NumericType, expected: u32, timeout_ns: u64) -> Result<bool, WaitError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        let (f, pinfo) = get_futex(self.track(), id);
        Ok(futex::wait_timeout(f, expected, timeout_ns, pinfo))
    }

    fn __wait_for_enter_timeout(&self, id: NumericType, timeout_ns: Option<(u64, Instant)>) -> Result<NumberId, WaitError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        let mut i = 0;
        let mut f = None;
        let mut futex_value_before = 0;
        let futex_getter = || get_futex(self.track(), id);
        loop {
            if i >= MAX_LOOPS_BEFORE_SLEEP {
                futex_value_before = f.get_or_insert_with(&futex_getter).0.load(Ordering::Acquire); // Get the futex value before checking the number. Later on if the number is not gte at_least, we can load this value again, and if it has changed in between, we know that the number has changed as well (though spurious wakeups are possible).
            }

            //NOTE: `recover` returns None and stops probing if it finds a slot with the right key but that is still locked, but this is fine because if it's locked it should soon be unlocked, at which point the thread that finished adding in the id to the slot will wake this thread up again, and it will recheck, and recover will then find it this time, so given the contract of finding whatever was the first to be found in the ring with the provided id, this is good.
            if let Some(number_id) = self.track().recover(id) {
                return Ok(number_id);
            }

            i += 1;

            // If it's not, we do different things based on whether we've exhausted the number of spins we're willing to do.
            if i < MAX_SPINS {
                spin_for_step!(i);
            } else if i <= MAX_LOOPS_BEFORE_SLEEP { // <= instead of < since the futex_value_before load and the futex_value_after load sandwiches a single spin. This means that `i` will always be one ahead of what futex_value_before sees, so we need to not go into the sleep path for one more iteration since futex_value_before is not ready.
                yield_now();
            } else {
                // Load the futex value again.
                let futex_value_after = f.get_or_insert_with(&futex_getter).0.load(Ordering::Acquire);
                if futex_value_before != futex_value_after {
                    // The futex value has changed, so the number has also possibly changed. We need to recheck.
                    core::hint::spin_loop();
                } else {
                    // Otherwise, we go to sleep.
                    match &timeout_ns {
                        Some((timeout_ns, start)) => {
                            let freeze = start.elapsed().as_nanos() as u64;
                            if freeze >= *timeout_ns {
                                return Err(WaitError::NotFound);
                            }
                            let (word, pinfo) = *f.get_or_insert_with(&futex_getter);
                            let _ = futex::wait_timeout(word, futex_value_after, timeout_ns - freeze, pinfo);
                        },
                        _ => {
                            let (word, pinfo) = *f.get_or_insert_with(&futex_getter);
                            let _ = futex::wait(word, futex_value_after, pinfo);
                        },
                    }
                    continue;
                }
            }

            match &timeout_ns {
                Some((timeout_ns, start)) => {
                    if start.elapsed().as_nanos() as u64 >= *timeout_ns {
                        return Err(WaitError::NotFound);
                    }
                },
                _ => {},
            }
        }
    }

    fn __wait_for_enter_and_at_least_timeout(&self, id: NumericType, at_least: NumericType, expected_user_bits: Option<NumericType>, timeout_ns: Option<(u64, Instant)>) -> Result<(WaitGteStatus, NumericType, NumberId), WaitError> {
        if id == EMPTY_ID || is_key_locked(id) {
            return Err(WaitError::InvalidId);
        }
        let mut i = 0;
        let mut f = None;
        let mut futex_value_before = 0;
        let futex_getter = || get_futex(self.track(), id);
        let mut number = None;
        loop {
            let mut value = None;

            if i >= MAX_LOOPS_BEFORE_SLEEP {
                futex_value_before = f.get_or_insert_with(&futex_getter).0.load(Ordering::Acquire); // Get the futex value before checking the number. Later on if the number is not gte at_least, we can load this value again, and if it has changed in between, we know that the number has changed as well (though spurious wakeups are possible).
            }

            if number.is_none() {
                if let Some(number_id) = self.track().recover(id) {
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
                if gte_masked(value_tmp, at_least, Self::NUMBER_MASK) {
                    return Ok((WaitGteStatus::Reached, without_suspended_bit(value_tmp), *number_id));
                }

                if let Some(expected) = expected_user_bits
                    && (value_tmp ^ expected) & Self::USER_MASK != 0
                {
                    return Ok((WaitGteStatus::ReservedBitsModified, without_suspended_bit(value_tmp), *number_id));
                }

                value = Some(value_tmp);
            }

            i += 1;

            // If it's not, we do different things based on whether we've exhausted the number of spins we're willing to do.
            if i < MAX_SPINS {
                spin_for_step!(i);
            } else if i <= MAX_LOOPS_BEFORE_SLEEP { // <= instead of < since the futex_value_before load and the futex_value_after load sandwiches a single spin. This means that `i` will always be one ahead of what futex_value_before sees, so we need to not go into the sleep path for one more iteration since futex_value_before is not ready.
                yield_now();
            } else {
                // Load the futex value again.
                let futex_value_after = f.get_or_insert_with(&futex_getter).0.load(Ordering::Acquire);
                if futex_value_before != futex_value_after {
                    // The futex value has changed, so the number has also possibly changed. We need to recheck.
                    core::hint::spin_loop();
                } else {
                    // Otherwise, we go to sleep.
                    match &timeout_ns {
                        Some((timeout_ns, start)) => {
                            let freeze = start.elapsed().as_nanos() as u64;
                            if freeze >= *timeout_ns {
                                match (value, &number) {
                                    (Some(value), Some((_, number_id))) => {
                                        return Ok((WaitGteStatus::TimedOut, without_suspended_bit(value), *number_id));
                                    },
                                    _ => {},
                                }
                                return Err(WaitError::NotFound);
                            }
                            let (word, pinfo) = *f.get_or_insert_with(&futex_getter);
                            let _ = futex::wait_timeout(word, futex_value_after, timeout_ns - freeze, pinfo);
                        },
                        _ => {
                            let (word, pinfo) = *f.get_or_insert_with(&futex_getter);
                            let _ = futex::wait(word, futex_value_after, pinfo);
                        },
                    }
                    continue;
                }
            }

            match &timeout_ns {
                Some((timeout_ns, start)) => {
                    if start.elapsed().as_nanos() as u64 >= *timeout_ns {
                        match (value, &number) {
                            (Some(value), Some((_, number_id))) => {
                                return Ok((WaitGteStatus::TimedOut, without_suspended_bit(value), *number_id));
                            },
                            _ => {},
                        }
                        return Err(WaitError::NotFound);
                    }
                },
                _ => {},
            }
        }
    }

    pub fn leave(&self, number: NumberId) -> Result<(), LeaveError> {
        let result = self.track().leave(number);
        let (f, pinfo) = get_futex(self.track(), number.id);
        f.fetch_add(1, Ordering::Release);
        let _ = futex::wake_all(f, pinfo);
        result
    }

    pub fn leave_concurrent(&self, number: NumberId) -> Result<(), LeaveError> {
        let result = self.track().leave_concurrent(number);
        let (f, pinfo) = get_futex(self.track(), number.id);
        f.fetch_add(1, Ordering::Release);
        let _ = futex::wake_all(f, pinfo);
        result
    }
}

impl<'a, 'b, const L: u32, const I: NumericType> NumberWaiting<'a, 'b, L, I> {
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
        let (f, pinfo) = get_futex(self.atomic_track_waiting.track(), self.inner.id);
        f.fetch_add(1, Ordering::Release);
        let _ = futex::wake_all(f, pinfo);
    }

    pub fn wait_gte(&self, at_least: NumericType) -> Result<NumericType, WaitError> {
        match self.__wait_gte_timeout(at_least, None, None)? {
            (WaitGteStatus::Reached, value) => Ok(value),
            _ => unreachable!(),
        }
    }

    pub fn wait_gte_timeout(&self, at_least: NumericType, timeout_ns: u64) -> Result<(bool, NumericType), WaitError> {
        let (status, value) = self.__wait_gte_timeout(at_least, None, Some((timeout_ns, Instant::now())))?;
        Ok((status == WaitGteStatus::Reached, value))
    }

    /// SICRBM (Stop If Custom Reserved Bits Modified)
    pub fn wait_gte_sicrbm(&self, at_least: NumericType, expected_user_bits: NumericType) -> Result<(WaitGteStatus, NumericType), WaitError> {
        self.__wait_gte_timeout(at_least, Some(expected_user_bits), None)
    }

    pub fn wait_gte_timeout_sicrbm(&self, at_least: NumericType, expected_user_bits: NumericType, timeout_ns: u64) -> Result<(WaitGteStatus, NumericType), WaitError> {
        self.__wait_gte_timeout(at_least, Some(expected_user_bits), Some((timeout_ns, Instant::now())))
    }

    /// See [`AtomicTrackWaiting::futex_word_value`].
    pub fn futex_word_value(&self) -> u32 {
        get_futex(self.atomic_track_waiting.track(), self.inner.id).0.load(Ordering::Acquire)
    }

    /// See [`AtomicTrackWaiting::wait_spurious`].
    pub fn wait_spurious(&self, expected: u32) {
        let (f, pinfo) = get_futex(self.atomic_track_waiting.track(), self.inner.id);
        futex::wait(f, expected, pinfo);
    }

    /// See [`AtomicTrackWaiting::wait_spurious_timeout`].
    pub fn wait_spurious_timeout(&self, expected: u32, timeout_ns: u64) -> bool {
        let (f, pinfo) = get_futex(self.atomic_track_waiting.track(), self.inner.id);
        futex::wait_timeout(f, expected, timeout_ns, pinfo)
    }

    fn __wait_gte_timeout(&self, at_least: NumericType, expected_user_bits: Option<NumericType>, timeout_ns: Option<(u64, Instant)>) -> Result<(WaitGteStatus, NumericType), WaitError> {
        let mut i = 0;
        let mut f = None;
        let mut futex_value_before = 0;
        let futex_getter = || get_futex(self.atomic_track_waiting.track(), self.inner.id);
        loop {
            let mut value;

            if i >= MAX_LOOPS_BEFORE_SLEEP {
                futex_value_before = f.get_or_insert_with(&futex_getter).0.load(Ordering::Acquire); // Get the futex value before checking the number. Later on if the number is not gte at_least, we can load this value again, and if it has changed in between, we know that the number has changed as well (though spurious wakeups are possible).
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
            if gte_masked(value, at_least, AtomicTrackWaiting::<L, I>::NUMBER_MASK) {
                return Ok((WaitGteStatus::Reached, without_suspended_bit(value)));
            }

            if let Some(expected) = expected_user_bits
                && (value ^ expected) & AtomicTrackWaiting::<L, I>::USER_MASK != 0
            {
                return Ok((WaitGteStatus::ReservedBitsModified, without_suspended_bit(value)));
            }

            i += 1;

            // If it's not, we do different things based on whether we've exhausted the number of spins we're willing to do.
            if i < MAX_SPINS {
                spin_for_step!(i);
            } else if i <= MAX_LOOPS_BEFORE_SLEEP { // <= instead of < since the futex_value_before load and the futex_value_after load sandwiches a single spin. This means that `i` will always be one ahead of what futex_value_before sees, so we need to not go into the sleep path for one more iteration since futex_value_before is not ready.
                yield_now();
            } else {
                // Load the futex value again.
                let futex_value_after = f.get_or_insert_with(&futex_getter).0.load(Ordering::Acquire);
                if futex_value_before != futex_value_after {
                    // The futex value has changed, so the number has also possibly changed. We need to recheck.
                    core::hint::spin_loop();
                } else {
                    // Otherwise, we go to sleep.
                    match &timeout_ns {
                        Some((timeout_ns, start)) => {
                            let freeze = start.elapsed().as_nanos() as u64;
                            if freeze >= *timeout_ns {
                                return Ok((WaitGteStatus::TimedOut, without_suspended_bit(value)));
                            }
                            let (word, pinfo) = *f.get_or_insert_with(&futex_getter);
                            let _ = futex::wait_timeout(word, futex_value_after, timeout_ns - freeze, pinfo);
                        },
                        _ => {
                            let (word, pinfo) = *f.get_or_insert_with(&futex_getter);
                            let _ = futex::wait(word, futex_value_after, pinfo);
                        },
                    }
                    continue;
                }
            }

            match &timeout_ns {
                Some((timeout_ns, start)) => {
                    if start.elapsed().as_nanos() as u64 >= *timeout_ns {
                        return Ok((WaitGteStatus::TimedOut, without_suspended_bit(value)));
                    }
                },
                _ => {},
            }
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
        sync::{atomic::AtomicBool, mpsc, Arc, Barrier},
        thread,
        time::Duration,
        vec::Vec,
    };

    const TEST_TIMEOUT_NS: u64 = 2_000_000_000;
    const TEST_WATCHDOG: Duration = Duration::from_secs(2);

    fn run_while_futex_is_hot<R>(
        (futex, pinfo): (&'static AtomicU32, u128),
        operation: impl FnOnce() -> R,
    ) -> (R, Duration) {
        const HOT_THREADS: usize = 4;
        const WATCHDOG: Duration = Duration::from_secs(1);

        let stop = Arc::new(AtomicBool::new(false));
        let ready = Arc::new(Barrier::new(HOT_THREADS + 1));
        let mut workers = Vec::with_capacity(HOT_THREADS);

        for _ in 0..HOT_THREADS {
            let stop = Arc::clone(&stop);
            let ready = Arc::clone(&ready);
            workers.push(thread::spawn(move || {
                ready.wait();
                let started = Instant::now();
                while !stop.load(Ordering::Relaxed) && started.elapsed() < WATCHDOG {
                    futex.fetch_add(1, Ordering::Release);
                    let _ = futex::wake_all(futex, pinfo);
                }
            }));
        }

        ready.wait();
        let started = Instant::now();
        let result = operation();
        let elapsed = started.elapsed();

        stop.store(true, Ordering::Relaxed);
        for worker in workers {
            worker.join().unwrap();
        }

        (result, elapsed)
    }

    const ONE_USER_BIT: u32 = NumericType::BITS - 1;
    const FLAG: NumericType = 1 << (NumericType::BITS - 2);

    #[test]
    fn sicrbm_number_wait_stops_on_user_bit_change_but_not_on_counter_change() {
        let track = AtomicTrackWaiting::<ONE_USER_BIT>::new(4);
        let number_id = track.enter(3).unwrap();
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            let number = waiter_track.number(number_id).unwrap();
            waiter_barrier.wait();
            number.wait_gte_timeout_sicrbm(100, 0, TEST_TIMEOUT_NS)
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        let number = track.number(number_id).unwrap();
        number.add(5).unwrap();
        thread::sleep(Duration::from_millis(10));
        number.add(FLAG).unwrap();

        assert_eq!(waiter.join().unwrap(), Ok((WaitGteStatus::ReservedBitsModified, FLAG | 5)));
        assert_eq!(number.wait_gte_timeout_sicrbm(100, FLAG, 0), Ok((WaitGteStatus::TimedOut, FLAG | 5)));
        assert_eq!(number.wait_gte_sicrbm(5, 0), Ok((WaitGteStatus::Reached, FLAG | 5)));
        assert_eq!(number.wait_gte_timeout(100, 0), Ok((false, FLAG | 5)));
    }

    #[test]
    fn sicrbm_catches_a_change_made_before_the_wait_started() {
        let track = AtomicTrackWaiting::<ONE_USER_BIT>::new(4);
        let number_id = track.enter(3).unwrap();
        let number = track.number(number_id).unwrap();
        let seen = number.get().unwrap();
        number.add(FLAG).unwrap();

        assert_eq!(number.wait_gte_sicrbm(100, seen), Ok((WaitGteStatus::ReservedBitsModified, FLAG)));
        assert_eq!(track.wait_gte_sicrbm(3, 100, seen), Ok((WaitGteStatus::ReservedBitsModified, FLAG, number_id)));
        assert_eq!(number.wait_gte_timeout_sicrbm(100, seen | 7, 0), Ok((WaitGteStatus::ReservedBitsModified, FLAG)));
        assert_eq!(number.wait_gte_timeout_sicrbm(100, FLAG | 7, 0), Ok((WaitGteStatus::TimedOut, FLAG)));
    }

    #[test]
    fn sicrbm_key_wait_stops_on_user_bit_change() {
        let track = AtomicTrackWaiting::<ONE_USER_BIT, FLAG>::new(4);
        let number_id = track.enter(3).unwrap();
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            waiter_barrier.wait();
            waiter_track.wait_gte_timeout_sicrbm(3, 100, FLAG, TEST_TIMEOUT_NS)
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        track.number(number_id).unwrap().raise_to(2).unwrap();

        assert_eq!(waiter.join().unwrap(), Ok((WaitGteStatus::ReservedBitsModified, 2, number_id)));
        assert_eq!(track.wait_gte_timeout(3, 100, 0), Ok((false, 2, number_id)));
        assert_eq!(track.wait_gte_sicrbm(3, 2, FLAG), Ok((WaitGteStatus::Reached, 2, number_id)));
    }

    #[test]
    fn sicrbm_key_wait_compares_a_late_entrant_against_the_expected_bits() {
        let track = AtomicTrackWaiting::<ONE_USER_BIT, FLAG>::new(4);
        assert_eq!(track.wait_gte_timeout_sicrbm(3, 100, FLAG, 0), Err(WaitError::NotFound));

        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            waiter_barrier.wait();
            waiter_track.wait_gte_timeout_sicrbm(3, 100, 0, TEST_TIMEOUT_NS)
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        let number_id = track.enter(3).unwrap();

        assert_eq!(waiter.join().unwrap(), Ok((WaitGteStatus::ReservedBitsModified, FLAG, number_id)));
        assert_eq!(track.wait_gte_timeout_sicrbm(3, 100, FLAG, 0), Ok((WaitGteStatus::TimedOut, FLAG, number_id)));
    }

    #[test]
    fn wait_spurious_returns_immediately_after_a_signal_since_the_load() {
        let track = AtomicTrackWaiting::new_default(4);
        let number_id = track.enter(3).unwrap();
        let word = track.futex_word_value(3).unwrap();
        let number = track.number(number_id).unwrap();
        let number_word = number.futex_word_value();
        assert_eq!(word, number_word);
        number.add(1).unwrap();

        let started = Instant::now();
        assert_eq!(track.wait_spurious_timeout(3, word, TEST_TIMEOUT_NS), Ok(true));
        assert!(number.wait_spurious_timeout(number_word, TEST_TIMEOUT_NS));
        track.wait_spurious(3, word).unwrap();
        number.wait_spurious(number_word);
        assert!(started.elapsed() < Duration::from_millis(500));

        let started = Instant::now();
        while number.wait_spurious_timeout(number.futex_word_value(), 1_000_000) {
            assert!(started.elapsed() < TEST_WATCHDOG, "a fresh load never led to a timed out wait");
        }
    }

    #[test]
    fn wait_spurious_wakes_a_hand_rolled_wait_loop() {
        let track = AtomicTrackWaiting::new_default(4);
        let number_id = track.enter(3).unwrap();
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let (sender, receiver) = mpsc::sync_channel(1);
        let waiter = thread::spawn(move || {
            let number = waiter_track.number(number_id).unwrap();
            waiter_barrier.wait();
            loop {
                let word = number.futex_word_value();
                let value = number.get().unwrap();
                if value >= 5 {
                    sender.send(value).unwrap();
                    return;
                }
                number.wait_spurious(word);
            }
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        track.number(number_id).unwrap().raise_to(5).unwrap();

        assert_eq!(receiver.recv_timeout(TEST_WATCHDOG).expect("hand-rolled wait did not wake"), 5);
        waiter.join().unwrap();
    }

    #[test]
    fn pinfo_is_zero_without_a_join_and_packs_join_and_base_otherwise() {
        static POOL: [AtomicU32; 4] = [const { AtomicU32::new(0) }; 4];
        let private = ListofAtomicsWrapper { s: &POOL, waiting_pool_join: WAITING_POOL_JOIN_NONE };
        assert_eq!(private.pinfo(), 0);

        let shared = ListofAtomicsWrapper { s: &POOL, waiting_pool_join: 0xABCD };
        assert_eq!(shared.pinfo() >> 64, 0xABCD);
        assert_eq!(shared.pinfo() as u64 as usize, POOL.as_ptr().addr());

        let track = AtomicTrackWaiting::new_default(1);
        let (word, pinfo) = get_futex(track.track(), 5);
        assert_eq!(pinfo, futexes().pinfo());
        assert!(futexes().s.as_ptr_range().contains(&ptr::from_ref(word)));
    }

    #[test]
    fn get_futex_follows_the_seed_not_the_mapping() {
        let layout = AtomicTrack::layout(4).unwrap();
        unsafe {
            let first = NonNull::new(std::alloc::alloc(layout)).unwrap();
            let second = NonNull::new(std::alloc::alloc(layout)).unwrap();
            let track = AtomicTrack::init_in_place_default(first, 4);
            ptr::copy_nonoverlapping(first.as_ptr(), second.as_ptr(), layout.size());
            let elsewhere = AtomicTrack::from_raw_default(second, 4);

            for id in [1, 2, 3, 77, id_of("renderer")] {
                let (word, pinfo) = get_futex(track, id);
                let (word_elsewhere, pinfo_elsewhere) = get_futex(elsewhere, id);
                assert!(ptr::eq(word, word_elsewhere), "id {id}");
                assert_eq!(pinfo, pinfo_elsewhere);
            }

            let mut hasher = hasher::RhmHasher::default();
            hasher.write(&(first.as_ptr().addr() as u64).to_le_bytes());
            hasher.write(&77u64.to_le_bytes());
            assert!(ptr::eq(get_futex(elsewhere, 77).0, futex(hasher.finish() as usize).0));

            std::alloc::dealloc(first.as_ptr(), layout);
            std::alloc::dealloc(second.as_ptr(), layout);
        }
    }

    static STATIC_TRACK: crate::ArrayAtomicTrack<4> = crate::ArrayAtomicTrack::new();

    #[test]
    fn from_static_handles_share_the_track_and_wake_each_other() {
        let track: &'static AtomicTrack = &STATIC_TRACK;
        let handle = AtomicTrackWaiting::from_static(track);
        let number_id = handle.enter(21).unwrap();
        let waiter_handle = handle.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        let waiter = thread::spawn(move || {
            sender.send(waiter_handle.number(number_id).unwrap().wait_gte(4)).unwrap();
        });

        thread::sleep(Duration::from_millis(10));
        AtomicTrackWaiting::from_static(track).number(number_id).unwrap().raise_to(4).unwrap();
        assert_eq!(receiver.recv_timeout(TEST_WATCHDOG).expect("from_static wait did not wake"), Ok(4));
        waiter.join().unwrap();

        drop(handle);
        assert_eq!(track.number(number_id).unwrap().get(), Ok(4));
        track.leave(number_id).unwrap();
    }

    #[test]
    fn from_raw_handles_never_free_the_track() {
        let layout = AtomicTrack::layout(4).unwrap();
        unsafe {
            let memory = NonNull::new(std::alloc::alloc(layout)).unwrap();
            let track = NonNull::from(AtomicTrack::init_in_place_default(memory, 4));
            let handle = AtomicTrackWaiting::from_raw(track);
            let clones = [handle.clone(), handle.clone()];
            let number_id = handle.enter(9).unwrap();
            clones[0].number(number_id).unwrap().add(3).unwrap();
            drop(clones);
            drop(handle);

            assert_eq!(track.as_ref().number(number_id).unwrap().get(), Ok(3));
            std::alloc::dealloc(memory.as_ptr(), layout);
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn from_raw_handles_on_two_mappings_of_one_track_wake_each_other() {
        use core::ffi::c_void;

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn CreateFileMappingW(file: *mut c_void, attributes: *const c_void, protect: u32, size_high: u32, size_low: u32, name: *const u16) -> *mut c_void;
            fn MapViewOfFile(mapping: *mut c_void, access: u32, offset_high: u32, offset_low: u32, bytes: usize) -> *mut c_void;
            fn UnmapViewOfFile(base: *const c_void) -> i32;
            fn CloseHandle(handle: *mut c_void) -> i32;
        }

        let layout = AtomicTrack::layout(4).unwrap();
        unsafe {
            let mapping = CreateFileMappingW(ptr::without_provenance_mut(usize::MAX), ptr::null(), 0x04, 0, layout.size() as u32, ptr::null());
            assert!(!mapping.is_null());
            let view_a = NonNull::new(MapViewOfFile(mapping, 0xF001F, 0, 0, layout.size()).cast::<u8>()).unwrap();
            let view_b = NonNull::new(MapViewOfFile(mapping, 0xF001F, 0, 0, layout.size()).cast::<u8>()).unwrap();
            assert_ne!(view_a, view_b);

            let track_a = NonNull::from(AtomicTrack::init_in_place_default(view_a, 4));
            let track_b = NonNull::from(AtomicTrack::from_raw_default(view_b, 4));
            let handle_a = AtomicTrackWaiting::from_raw(track_a);
            let handle_b = AtomicTrackWaiting::from_raw(track_b);

            let number_id = handle_a.enter(33).unwrap();
            let (sender, receiver) = mpsc::sync_channel(1);
            let waiter = thread::spawn(move || {
                sender.send(handle_b.number(number_id).unwrap().wait_gte(6)).unwrap();
            });

            thread::sleep(Duration::from_millis(10));
            handle_a.number(number_id).unwrap().raise_to(6).unwrap();
            assert_eq!(receiver.recv_timeout(TEST_WATCHDOG).expect("wait through the other mapping did not wake"), Ok(6));
            waiter.join().unwrap();
            drop(handle_a);

            UnmapViewOfFile(view_a.as_ptr().cast());
            UnmapViewOfFile(view_b.as_ptr().cast());
            CloseHandle(mapping);
        }
    }

    #[test]
    fn wait_gte_status_discriminants_are_fixed() {
        assert_eq!(WaitGteStatus::Reached as u32, 0);
        assert_eq!(WaitGteStatus::ReservedBitsModified as u32, 1);
        assert_eq!(WaitGteStatus::TimedOut as u32, 2);
        assert_eq!(size_of::<WaitGteStatus>(), size_of::<core::ffi::c_int>());
    }

    #[test]
    fn futex_word_value_and_wait_spurious_validate_ids() {
        let track = AtomicTrackWaiting::new_default(4);
        let word = track.futex_word_value(3).unwrap();
        assert_eq!(track.futex_word_value(EMPTY_ID), Err(WaitError::InvalidId));
        assert_eq!(track.futex_word_value(MSB | 3), Err(WaitError::InvalidId));
        assert_eq!(track.wait_spurious(EMPTY_ID, word), Err(WaitError::InvalidId));
        assert_eq!(track.wait_spurious_timeout(MSB | 3, word, 0), Err(WaitError::InvalidId));
    }

    #[test]
    fn id_of_output_is_pinned() {
        #[cfg(not(feature = "smaller-atomics"))]
        let expected = [
            ("", 0x0338_dc4b_e2ce_cdae),
            ("renderer", 0x1df5_c40b_81fb_24e4),
            ("audio", 0x5969_afff_8526_d1e1),
        ];
        #[cfg(feature = "smaller-atomics")]
        let expected = [
            ("", 0x62ce_cdae),
            ("renderer", 0x01fb_24e4),
            ("audio", 0x0526_d1e1),
        ];
        for (name, id) in expected {
            assert_eq!(id_of(name), id, "{name:?}");
        }
    }

    #[test]
    fn id_of_is_const_and_yields_valid_distinct_ids() {
        const RENDERER: NumericType = id_of("renderer");
        assert_eq!(RENDERER, id_of("renderer"));

        let long = "x".repeat(200);
        let names = ["", "a", "b", "renderer", "audio", "sixteen byte key", "seventeen byte key", long.as_str()];
        let ids = names.map(id_of);
        for (name, id) in names.iter().zip(ids) {
            assert_ne!(id, EMPTY_ID, "{name:?}");
            assert!(!is_key_locked(id), "{name:?}");
        }
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "{:?} and {:?}", names[i], names[j]);
            }
        }
    }

    #[test]
    fn id_of_matches_writing_the_name_into_the_hasher() {
        for name in ["renderer", "audio", ""] {
            let mut hasher = hasher::RhmHasher::default();
            hasher.write(name.as_bytes());
            assert_eq!(id_of(name), key_bits(hasher.finish() as NumericType), "{name:?}");
        }
    }

    #[test]
    fn id_of_names_the_same_number_for_entering_and_waiting() {
        let track = AtomicTrackWaiting::new_default(8);
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            waiter_barrier.wait();
            waiter_track.wait_gte_timeout(id_of("renderer"), 3, TEST_TIMEOUT_NS)
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        let number_id = track.enter(id_of("renderer")).unwrap();
        track.number(number_id).unwrap().raise_to(3).unwrap();

        assert_eq!(waiter.join().unwrap(), Ok((true, 3, number_id)));
        assert_eq!(track.recover(id_of("renderer")), Some(number_id));
        assert_eq!(track.recover(id_of("audio")), None);
    }

    #[test]
    fn recover_ignores_an_entry_that_is_still_locked() {
        let track = AtomicTrackWaiting::new_default(1);
        let inner = track.track();
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
        let track = AtomicTrackWaiting::new_default(4);

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
    fn blocking_entry_waits_wake_on_enter() {
        let track = AtomicTrackWaiting::new_default(1);
        let barrier = Arc::new(Barrier::new(3));

        let id_waiter_track = track.clone();
        let id_waiter_barrier = Arc::clone(&barrier);
        let (id_sender, id_receiver) = mpsc::sync_channel(1);
        let id_waiter = thread::spawn(move || {
            id_waiter_barrier.wait();
            id_sender.send(id_waiter_track.wait_for(7)).unwrap();
        });

        let number_waiter_track = track.clone();
        let number_waiter_barrier = Arc::clone(&barrier);
        let (number_sender, number_receiver) = mpsc::sync_channel(1);
        let number_waiter = thread::spawn(move || {
            number_waiter_barrier.wait();
            let result = number_waiter_track
                .wait_for_number(7)
                .map(|number| number.get().unwrap());
            number_sender.send(result).unwrap();
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        let number_id = track.enter(7).unwrap();

        assert_eq!(
            id_receiver
                .recv_timeout(TEST_WATCHDOG)
                .expect("wait_for did not wake"),
            Ok(number_id)
        );
        assert_eq!(
            number_receiver
                .recv_timeout(TEST_WATCHDOG)
                .expect("wait_for_number did not wake"),
            Ok(0)
        );
        id_waiter.join().unwrap();
        number_waiter.join().unwrap();
    }

    #[test]
    fn key_wait_is_woken_when_the_number_advances() {
        let track = AtomicTrackWaiting::new_default(1);
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
    fn raise_to_wakes_blocking_key_and_number_waits() {
        let track = AtomicTrackWaiting::new_default(1);
        let number_id = track.enter(9).unwrap();
        let barrier = Arc::new(Barrier::new(3));

        let key_waiter_track = track.clone();
        let key_waiter_barrier = Arc::clone(&barrier);
        let (key_sender, key_receiver) = mpsc::sync_channel(1);
        let key_waiter = thread::spawn(move || {
            key_waiter_barrier.wait();
            key_sender.send(key_waiter_track.wait_gte(9, 5)).unwrap();
        });

        let number_waiter_track = track.clone();
        let number_waiter_barrier = Arc::clone(&barrier);
        let (number_sender, number_receiver) = mpsc::sync_channel(1);
        let number_waiter = thread::spawn(move || {
            let number = number_waiter_track.number(number_id).unwrap();
            number_waiter_barrier.wait();
            number_sender.send(number.wait_gte(5)).unwrap();
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        track.number(number_id).unwrap().raise_to(5).unwrap();

        assert_eq!(
            key_receiver
                .recv_timeout(TEST_WATCHDOG)
                .expect("key wait did not wake after raise_to"),
            Ok((5, number_id))
        );
        assert_eq!(
            number_receiver
                .recv_timeout(TEST_WATCHDOG)
                .expect("number wait did not wake after raise_to"),
            Ok(5)
        );
        key_waiter.join().unwrap();
        number_waiter.join().unwrap();
    }

    #[test]
    fn number_wait_is_woken_when_the_lane_leaves() {
        let track = AtomicTrackWaiting::new_default(1);
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
    fn concurrent_leave_wakes_a_blocking_number_wait() {
        let track = AtomicTrackWaiting::new_default(1);
        let number_id = track.enter(11).unwrap();
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let (sender, receiver) = mpsc::sync_channel(1);
        let waiter = thread::spawn(move || {
            let number = waiter_track.number(number_id).unwrap();
            waiter_barrier.wait();
            sender.send(number.wait_gte(1)).unwrap();
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        track.leave_concurrent(number_id).unwrap();

        assert_eq!(
            receiver
                .recv_timeout(TEST_WATCHDOG)
                .expect("number wait did not wake after leave_concurrent"),
            Err(WaitError::NotFound)
        );
        waiter.join().unwrap();
    }

    #[test]
    fn manual_atomic_update_wakes_after_signal_change() {
        let track = AtomicTrackWaiting::new_default(1);
        let number_id = track.enter(15).unwrap();
        let waiter_track = track.clone();
        let barrier = Arc::new(Barrier::new(2));
        let waiter_barrier = Arc::clone(&barrier);
        let (sender, receiver) = mpsc::sync_channel(1);
        let waiter = thread::spawn(move || {
            let number = waiter_track.number(number_id).unwrap();
            waiter_barrier.wait();
            sender.send(number.wait_gte(8)).unwrap();
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(10));
        let number = track.number(number_id).unwrap();
        unsafe {
            number.atomic().store(8, Ordering::Release);
        }
        number.signal_change();

        assert_eq!(
            receiver
                .recv_timeout(TEST_WATCHDOG)
                .expect("number wait did not wake after signal_change"),
            Ok(8)
        );
        waiter.join().unwrap();
    }

    #[test]
    fn timed_waits_return_the_last_observed_value() {
        let track = AtomicTrackWaiting::new_default(1);
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
    fn hot_shared_futex_bucket_does_not_starve_timed_waits() {
        const TIMEOUT_NS: u64 = 25_000_000;
        const MIN_ALLOWED: Duration = Duration::from_nanos(TIMEOUT_NS);
        const MAX_ALLOWED: Duration = Duration::from_millis(500);

        let track = AtomicTrackWaiting::new_default(1);
        let number_id = track.enter(1).unwrap();

        let (result, elapsed) = run_while_futex_is_hot(
            get_futex(track.track(), 2),
            || track.wait_for_timeout(2, TIMEOUT_NS),
        );
        assert_eq!(result, Err(WaitError::NotFound));
        assert!(elapsed >= MIN_ALLOWED, "entry wait returned early after {elapsed:?}");
        assert!(elapsed < MAX_ALLOWED, "entry wait took {elapsed:?}");

        let (result, elapsed) = run_while_futex_is_hot(
            get_futex(track.track(), number_id.id),
            || track.wait_gte_timeout(number_id.id, 1, TIMEOUT_NS),
        );
        assert_eq!(result, Ok((false, 0, number_id)));
        assert!(elapsed >= MIN_ALLOWED, "key wait returned early after {elapsed:?}");
        assert!(elapsed < MAX_ALLOWED, "key wait took {elapsed:?}");

        let number = track.number(number_id).unwrap();
        let (result, elapsed) = run_while_futex_is_hot(
            get_futex(track.track(), number_id.id),
            || number.wait_gte_timeout(1, TIMEOUT_NS),
        );
        assert_eq!(result, Ok((false, 0)));
        assert!(elapsed >= MIN_ALLOWED, "number wait returned early after {elapsed:?}");
        assert!(elapsed < MAX_ALLOWED, "number wait took {elapsed:?}");
    }

    #[test]
    fn key_wait_uses_only_the_first_recovered_placement() {
        let track = AtomicTrackWaiting::new_default(4);
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
        let track = AtomicTrackWaiting::new_default(1);
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
    const fn rapid_multiply(a: u64, b: u64) -> (u64, u64) {
        let product = (a as u128) * (b as u128);
        (product as u64, (product >> 64) as u64)
    }

    #[inline(always)]
    const fn rapid_mix(a: u64, b: u64) -> u64 {
        let (low, high) = rapid_multiply(a, b);
        low ^ high
    }

    #[inline(always)]
    const fn rapid_read_64(bytes: &[u8], at: usize) -> u64 {
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
    const fn rapid_read_32(bytes: &[u8], at: usize) -> u64 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as u64
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub(super) const fn rapidhash_micro(bytes: &[u8]) -> u64 {
        rapidhash_micro_with_seed(bytes, 0)
    }

    /// Hashes with rapidhashMicro V3.
    ///
    /// This is a small, non-cryptographic 64-bit hash. The output is identical on
    /// little- and big-endian targets.
    #[inline]
    pub(super) const fn rapidhash_micro_with_seed(bytes: &[u8], mut seed: u64) -> u64 {
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
        extern crate std;

        use super::*;
        use std::vec::Vec;

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
