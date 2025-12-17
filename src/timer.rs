mod failover;
mod gear;
mod slot;
mod wheel;

use std::{convert::Infallible, time::Instant};

pub use failover::*;
pub use wheel::*;

pub use crate::timer::gear::InsertError;

pub trait Timer {
    type Context;

    fn fire(&mut self, ctx: &mut Self::Context);
}

/// Trait for timer driver implementations.
///
/// Enables generic runtime code that works with any wheel variant.
pub trait TimerDriver<T: Timer>: Default {
    type Err;

    /// Insert a timer to fire at the given instant.
    ///
    /// Returns a handle for cancellation, or an error containing the timer
    /// if the wheel is at capacity.
    fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, Self::Err>;

    /// Cancel a pending timer.
    ///
    /// Returns the timer if still pending, `None` if already fired or invalid handle.
    fn cancel(&mut self, handle: TimerHandle) -> Option<T>;

    /// Poll the wheel, firing all timers due by `now`.
    ///
    /// Returns the number of timers fired.
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize;
}

impl<T, const G: usize, const R: u64, const S: usize, const P: usize> TimerDriver<T>
    for BitWheel<T, G, R, S, P>
where
    T: Timer,
{
    type Err = InsertError<T>;

    #[inline(always)]
    fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, Self::Err> {
        BitWheel::insert(self, when, timer)
    }

    #[inline(always)]
    fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        BitWheel::cancel(self, handle)
    }

    #[inline(always)]
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize {
        BitWheel::poll(self, now, ctx)
    }
}

impl<T, const G: usize, const R: u64, const S: usize, const P: usize, const F: u64> TimerDriver<T>
    for BitWheelWithFailover<T, G, R, S, P, F>
where
    T: Timer,
{
    type Err = Infallible;

    /// Infallible insert - always succeeds, overflows to BTreeMap.
    #[inline(always)]
    fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, Self::Err> {
        Ok(BitWheelWithFailover::insert(self, when, timer))
    }

    #[inline(always)]
    fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        BitWheelWithFailover::cancel(self, handle)
    }

    #[inline(always)]
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize {
        BitWheelWithFailover::poll(self, now, ctx)
    }
}

/// Handle for cancelling a pending timer.
///
/// # Ownership Semantics
///
/// `TimerHandle` is intentionally not `Clone` or `Copy`. This ensures:
///
/// - **Unique ownership**: Only one handle exists per scheduled timer.
/// - **Single cancellation**: Once `cancel()` consumes the handle, no
///   double-cancel is possible.
///
/// # Creation
///
/// Handles are only created by [`BitWheelInner::insert`]. The internal
/// fields (gear, slot, key, when_offset) are not publicly constructible,
/// guaranteeing that any handle passed to `cancel()` refers to a valid
/// timer entry (assuming it hasn't already fired).
///
/// # Validity
///
/// A handle becomes invalid once its `when_offset` tick has passed
/// (i.e., the timer has fired). Calling `cancel()` on an expired handle
/// safely returns `None`.
#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct TimerHandle {
    when_offset: u64,
    key: u32,
    gear: u8,
    slot: u8,
    overflow: bool,
}

// ============================================================
// STANDARD - 256 hotspot, 4ms resolution, ~16 second range
// Order timeouts, reject backoff, rate limiting
// ============================================================

/// Balanced. 256 hotspot, 4ms, ~16s. ~50KB.
pub type Wheel<T> = BitWheel<T, 2, 4, 32, 8>;

/// Small memory. 256 hotspot, 4ms, ~16s. ~25KB. More probing.
pub type SmallWheel<T> = BitWheel<T, 2, 4, 16, 16>;

/// Large memory. 256 hotspot, 4ms, ~16s. ~200KB. Tight tails.
pub type LargeWheel<T> = BitWheel<T, 2, 4, 128, 2>;

// ============================================================
// PRECISE - 256 hotspot, 1ms resolution, ~4 minute range
// Sub-5ms precision requirements
// ============================================================

/// Balanced. 256 hotspot, 1ms, ~4min. ~75KB.
pub type PreciseWheel<T> = BitWheel<T, 3, 1, 32, 8>;

/// Small memory. 256 hotspot, 1ms, ~4min. ~38KB. More probing.
pub type SmallPreciseWheel<T> = BitWheel<T, 3, 1, 16, 16>;

/// Large memory. 256 hotspot, 1ms, ~4min. ~300KB. Tight tails.
pub type LargePreciseWheel<T> = BitWheel<T, 3, 1, 128, 2>;

// ============================================================
// BURST - 512 hotspot, 4ms resolution, ~16 second range
// Reject storms, high-frequency order flow
// ============================================================

/// Balanced. 512 hotspot, 4ms, ~16s. ~100KB.
pub type BurstWheel<T> = BitWheel<T, 2, 4, 64, 8>;

/// Small memory. 512 hotspot, 4ms, ~16s. ~50KB. More probing.
pub type SmallBurstWheel<T> = BitWheel<T, 2, 4, 32, 16>;

/// Large memory. 512 hotspot, 4ms, ~16s. ~400KB. Tight tails.
pub type LargeBurstWheel<T> = BitWheel<T, 2, 4, 256, 2>;

// ============================================================
// WITH FAILOVER - BTreeMap overflow for guaranteed insertion
// ============================================================

pub type WheelWithFailover<T> = BitWheelWithFailover<T, 2, 4, 32, 8, 32>;
pub type SmallWheelWithFailover<T> = BitWheelWithFailover<T, 2, 4, 16, 16, 32>;
pub type LargeWheelWithFailover<T> = BitWheelWithFailover<T, 2, 4, 128, 2, 32>;

pub type PreciseWheelWithFailover<T> = BitWheelWithFailover<T, 3, 1, 32, 8, 32>;
pub type SmallPreciseWheelWithFailover<T> = BitWheelWithFailover<T, 3, 1, 16, 16, 32>;
pub type LargePreciseWheelWithFailover<T> = BitWheelWithFailover<T, 3, 1, 128, 2, 32>;

pub type BurstWheelWithFailover<T> = BitWheelWithFailover<T, 2, 4, 64, 8, 32>;
pub type SmallBurstWheelWithFailover<T> = BitWheelWithFailover<T, 2, 4, 32, 16, 32>;
pub type LargeBurstWheelWithFailover<T> = BitWheelWithFailover<T, 2, 4, 256, 2, 32>;

#[macro_export]
macro_rules! define_bitwheel {
    ($name:ident, $timer:ty, $num_gears:expr, $resolution_ms:expr, $slot_cap:expr, $max_probes:expr) => {
        pub type $name =
            $crate::BitWheel<$timer, $num_gears, $resolution_ms, $slot_cap, $max_probes>;
    };
}
