use std::{
    sync::{
        Arc, Barrier,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use atomictrack::{ArrayAtomicTrack, EnterError, NumberError, math::NumericType};

#[test]
fn suspend_keeps_the_slot_and_stops_holding_min_back() {
    let track = ArrayAtomicTrack::<1>::new();
    let a = track.enter(1).unwrap();
    let n = track.number(a).unwrap();
    n.raise_to(5).unwrap();

    assert_eq!(n.suspend(), Ok(5));
    assert_eq!(n.suspend(), Err(NumberError::Suspended));
    assert_eq!(n.get(), Err(NumberError::Suspended));
    assert_eq!(track.enter(2), Err(EnterError::Full)); // still ours
    assert_eq!(track.recover(1), Some(a));

    // Suspended lanes don't bound the min, and get dragged up to it.
    assert_eq!(track.find_min_bounded(9), (9, 0));
    assert_eq!(n.resume(), Ok(9));
    assert_eq!(n.resume(), Err(NumberError::NotSuspended));
}

#[test]
fn resume_continues_from_own_position() {
    let track = ArrayAtomicTrack::<4>::new();
    let slow = track.enter(1).unwrap(); // pins min at 0
    let fast = track.enter(2).unwrap();
    let n = track.number(fast).unwrap();
    n.raise_to(7).unwrap();
    n.suspend().unwrap();
    assert_eq!(track.find_min_bounded(10).0, 0);
    assert_eq!(n.resume(), Ok(7)); // not 0
    let _ = slow;
}

#[test]
fn resume_at_is_exact_or_stays_suspended() {
    let track = ArrayAtomicTrack::<2>::new();
    let a = track.enter(1).unwrap();
    let n = track.number(a).unwrap();
    n.suspend().unwrap();
    assert_eq!(track.find_min_bounded(10), (10, 0));

    assert_eq!(n.resume_at(5), Err(NumberError::ExactFailed));
    assert_eq!(n.get(), Err(NumberError::Suspended)); // still suspended, still ours
    assert_eq!(n.resume_at(12), Ok(12));
    n.suspend().unwrap();
    assert_eq!(n.resume_from(3), Ok(12));
}

#[test]
fn leave_releases_a_suspended_lane() {
    let track = ArrayAtomicTrack::<1>::new();
    let a = track.enter(1).unwrap();
    track.number(a).unwrap().raise_to(4).unwrap();
    track.number(a).unwrap().suspend().unwrap();
    track.leave(a).unwrap();
    assert_eq!(track.recover(1), None);
    let b = track.enter(2).unwrap();
    assert_eq!(track.number(b).unwrap().get(), Ok(4));
    assert_eq!(track.number(a).unwrap().resume(), Err(NumberError::NotFound));
}

#[test]
fn user_bits_survive_suspension_and_forced_floors() {
    const ONE_USER_BIT: u32 = NumericType::BITS - 1;
    const FLAG: NumericType = 1 << (NumericType::BITS - 2);
    let track = ArrayAtomicTrack::<2, ONE_USER_BIT>::new();
    let a = track.enter(1).unwrap();
    let n = track.number(a).unwrap();
    n.raise_to(FLAG | 3).unwrap();
    assert_eq!(n.suspend(), Ok(FLAG | 3));
    track.find_min_bounded(8); // forces the floor to 8
    assert_eq!(n.resume(), Ok(FLAG | 8));
}

/// Tests for a race the id lock prevents: resume (owner) vs leave_concurrent (someone else). An active value in an
/// EMPTY slot trips enter's `debug_assert!(is_suspended(current))`, and a leaked slot shows up as `Full`.
#[test]
fn resume_vs_concurrent_leave() {
    let iterations = if cfg!(miri) { 30 } else { 5_000 };
    for round in 0..iterations {
        let track = Arc::new(ArrayAtomicTrack::<2>::new());
        let a = track.enter(1).unwrap();
        let go = Arc::new(Barrier::new(3));
        let stop = Arc::new(AtomicBool::new(false));

        let owner = {
            let (track, go) = (track.clone(), go.clone());
            thread::spawn(move || {
                go.wait();
                let n = track.number(a).unwrap();
                let mut g = 0;
                loop {
                    g += 1;
                    let r = match g % 3 {
                        0 => n.raise_to(g).map(|_| ()),
                        1 => n.suspend().map(|_| ()),
                        _ => n.resume_at(g).or_else(|e| if e == NumberError::ExactFailed { n.resume() } else { Err(e) }).map(|_| ()),
                    };
                    match r {
                        Ok(()) | Err(NumberError::Suspended) | Err(NumberError::NotSuspended) => {}
                        Err(NumberError::NotFound) => return,
                        Err(e) => panic!("{e:?}"),
                    }
                }
            })
        };
        // Throwing in a cleaner thread as well for good measure, since raising the floor makes the CASes in resume and
        // leave retry while the id lock is held.
        let cleaner = {
            let (track, go, stop) = (track.clone(), go.clone(), stop.clone());
            thread::spawn(move || {
                go.wait();
                let mut bound = 0;
                while !stop.load(Ordering::Acquire) {
                    bound += 1;
                    track.find_min_bounded(bound);
                }
            })
        };

        go.wait();
        for _ in 0..(round % 64) {
            std::hint::spin_loop();
        }
        track.leave_concurrent(a).unwrap();
        owner.join().unwrap();

        // Both slots must be empty with suspended values (enter debug_asserts the latter), i.e. no tearing.
        let x = track.enter(3).unwrap();
        let y = track.enter(4).unwrap();
        stop.store(true, Ordering::Release);
        cleaner.join().unwrap();
        assert!(track.number(x).unwrap().get().is_ok() && track.number(y).unwrap().get().is_ok());
        assert_eq!(track.recover(1), None);
    }
}
