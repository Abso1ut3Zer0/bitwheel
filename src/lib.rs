mod failover;
mod gear;
mod slot;
mod wheel;

use std::time::Instant;

pub use failover::*;
pub use wheel::*;

use crate::gear::InsertError;

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("{0} timers failed to reschedule")]
pub struct PollError(usize);

pub trait Timer {
    type Context;
    fn fire(&mut self, now: Instant, ctx: &mut Self::Context) -> Option<Instant>;
}

/// Trait for timer driver implementations.
///
/// Enables generic runtime code that works with any wheel variant.
pub trait TimerDriver<T: Timer> {
    /// Insert a timer to fire at the given instant.
    ///
    /// Returns a handle for cancellation, or an error containing the timer
    /// if the wheel is at capacity.
    fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, InsertError<T>>;

    /// Cancel a pending timer.
    ///
    /// Returns the timer if still pending, `None` if already fired or invalid handle.
    fn cancel(&mut self, handle: TimerHandle) -> Option<T>;

    /// Poll the wheel, firing all timers due by `now`.
    ///
    /// Returns the number of timers fired, or an error if timers were lost
    /// during rescheduling.
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> Result<usize, PollError>;
}

impl<T, const G: usize, const R: u64, const S: usize, const P: usize> TimerDriver<T>
    for BitWheel<T, G, R, S, P>
where
    T: Timer,
{
    #[inline(always)]
    fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, InsertError<T>> {
        BitWheel::insert(self, when, timer)
    }

    #[inline(always)]
    fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        BitWheel::cancel(self, handle)
    }

    #[inline(always)]
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> Result<usize, PollError> {
        BitWheel::poll(self, now, ctx)
    }
}

impl<T, const G: usize, const R: u64, const S: usize, const P: usize, const F: u64> TimerDriver<T>
    for BitWheelWithFailover<T, G, R, S, P, F>
where
    T: Timer,
{
    /// Infallible insert - always succeeds, overflows to BTreeMap.
    #[inline(always)]
    fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, InsertError<T>> {
        Ok(BitWheelWithFailover::insert(self, when, timer))
    }

    #[inline(always)]
    fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        BitWheelWithFailover::cancel(self, handle)
    }

    /// Infallible poll - reschedule failures go to failover, never lost.
    #[inline(always)]
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> Result<usize, PollError> {
        Ok(BitWheelWithFailover::poll(self, now, ctx))
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
// BALANCED - General purpose, 5ms resolution, ~23 hour range
// ============================================================

/// Standard balanced. 5ms resolution, ~23 hours.
/// Hotspot: 128. Memory: ~200KB.
pub type BalancedWheel<T> = BitWheel<T, 4, 5, 16, 8>;

/// Low latency balanced. 5ms resolution, ~23 hours.
/// Hotspot: 128. Memory: ~800KB. Minimal probing.
pub type BalancedFastWheel<T> = BitWheel<T, 4, 5, 64, 2>;

/// Compact balanced. 5ms resolution, ~23 hours.
/// Hotspot: 128. Memory: ~100KB. More probing.
pub type BalancedLightWheel<T> = BitWheel<T, 4, 5, 8, 16>;

// ============================================================
// PRECISE - Fine timing, 1ms resolution, ~4.7 hour range
// ============================================================

/// Standard precise. 1ms resolution, ~4.7 hours.
/// Hotspot: 128. Memory: ~200KB.
pub type PreciseWheel<T> = BitWheel<T, 4, 1, 16, 8>;

/// Low latency precise. 1ms resolution, ~4.7 hours.
/// Hotspot: 128. Memory: ~800KB. Minimal probing.
pub type PreciseFastWheel<T> = BitWheel<T, 4, 1, 64, 2>;

/// Compact precise. 1ms resolution, ~4.7 hours.
/// Hotspot: 128. Memory: ~100KB. More probing.
pub type PreciseLightWheel<T> = BitWheel<T, 4, 1, 8, 16>;

// ============================================================
// BURST - High volume, 5ms resolution, ~23 hour range
// Double hotspot capacity (256) for traffic spikes
// ============================================================

/// Standard burst. 5ms resolution, ~23 hours.
/// Hotspot: 256. Memory: ~400KB.
pub type BurstWheel<T> = BitWheel<T, 4, 5, 32, 8>;

/// Low latency burst. 5ms resolution, ~23 hours.
/// Hotspot: 256. Memory: ~1.6MB. Minimal probing.
pub type BurstFastWheel<T> = BitWheel<T, 4, 5, 128, 2>;

/// Compact burst. 5ms resolution, ~23 hours.
/// Hotspot: 256. Memory: ~200KB. More probing.
pub type BurstLightWheel<T> = BitWheel<T, 4, 5, 16, 16>;

// ============================================================
// EXTENDED - Long duration, 16ms resolution, ~3 day range
// For sessions, reconnects, long keepalives
// ============================================================

/// Standard extended. 16ms resolution, ~3 days.
/// Hotspot: 128. Memory: ~200KB.
pub type ExtendedWheel<T> = BitWheel<T, 4, 16, 16, 8>;

/// Low latency extended. 16ms resolution, ~3 days.
/// Hotspot: 128. Memory: ~800KB. Minimal probing.
pub type ExtendedFastWheel<T> = BitWheel<T, 4, 16, 64, 2>;

/// Compact extended. 16ms resolution, ~3 days.
/// Hotspot: 128. Memory: ~100KB. More probing.
pub type ExtendedLightWheel<T> = BitWheel<T, 4, 16, 8, 16>;

// ============================================================
// BALANCED WITH FAILOVER - General purpose, 5ms resolution
// ============================================================

/// Standard balanced with failover. 5ms resolution, ~23 hours.
/// Hotspot: 128. Memory: ~200KB. Failover check: 115ms.
pub type BalancedWheelWithFailover<T> = BitWheelWithFailover<T, 4, 5, 16, 8, 23>;

/// Low latency balanced with failover. 5ms resolution, ~23 hours.
/// Hotspot: 128. Memory: ~800KB. Failover check: 115ms.
pub type BalancedFastWheelWithFailover<T> = BitWheelWithFailover<T, 4, 5, 64, 2, 23>;

/// Compact balanced with failover. 5ms resolution, ~23 hours.
/// Hotspot: 128. Memory: ~100KB. Failover check: 115ms.
pub type BalancedLightWheelWithFailover<T> = BitWheelWithFailover<T, 4, 5, 8, 16, 23>;

// ============================================================
// PRECISE WITH FAILOVER - Fine timing, 1ms resolution
// ============================================================

/// Standard precise with failover. 1ms resolution, ~4.7 hours.
/// Hotspot: 128. Memory: ~200KB. Failover check: 23ms.
pub type PreciseWheelWithFailover<T> = BitWheelWithFailover<T, 4, 1, 16, 8, 23>;

/// Low latency precise with failover. 1ms resolution, ~4.7 hours.
/// Hotspot: 128. Memory: ~800KB. Failover check: 23ms.
pub type PreciseFastWheelWithFailover<T> = BitWheelWithFailover<T, 4, 1, 64, 2, 23>;

/// Compact precise with failover. 1ms resolution, ~4.7 hours.
/// Hotspot: 128. Memory: ~100KB. Failover check: 23ms.
pub type PreciseLightWheelWithFailover<T> = BitWheelWithFailover<T, 4, 1, 8, 16, 23>;

// ============================================================
// BURST WITH FAILOVER - High volume, 5ms resolution
// ============================================================

/// Standard burst with failover. 5ms resolution, ~23 hours.
/// Hotspot: 256. Memory: ~400KB. Failover check: 115ms.
pub type BurstWheelWithFailover<T> = BitWheelWithFailover<T, 4, 5, 32, 8, 23>;

/// Low latency burst with failover. 5ms resolution, ~23 hours.
/// Hotspot: 256. Memory: ~1.6MB. Failover check: 115ms.
pub type BurstFastWheelWithFailover<T> = BitWheelWithFailover<T, 4, 5, 128, 2, 23>;

/// Compact burst with failover. 5ms resolution, ~23 hours.
/// Hotspot: 256. Memory: ~200KB. Failover check: 115ms.
pub type BurstLightWheelWithFailover<T> = BitWheelWithFailover<T, 4, 5, 16, 16, 23>;

// ============================================================
// EXTENDED WITH FAILOVER - Long duration, 16ms resolution
// ============================================================

/// Standard extended with failover. 16ms resolution, ~3 days.
/// Hotspot: 128. Memory: ~200KB. Failover check: 368ms.
pub type ExtendedWheelWithFailover<T> = BitWheelWithFailover<T, 4, 16, 16, 8, 23>;

/// Low latency extended with failover. 16ms resolution, ~3 days.
/// Hotspot: 128. Memory: ~800KB. Failover check: 368ms.
pub type ExtendedFastWheelWithFailover<T> = BitWheelWithFailover<T, 4, 16, 64, 2, 23>;

/// Compact extended with failover. 16ms resolution, ~3 days.
/// Hotspot: 128. Memory: ~100KB. Failover check: 368ms.
pub type ExtendedLightWheelWithFailover<T> = BitWheelWithFailover<T, 4, 16, 8, 16, 23>;
