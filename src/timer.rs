mod dual;
mod failover;
mod gear;
mod slot;
mod wheel;

use std::time::Instant;

pub use dual::*;
pub use failover::*;
pub use wheel::*;

pub use crate::timer::gear::InsertError;

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("{0} timers failed to reschedule")]
pub struct PollError(usize);

pub trait Timer {
    type Context;
    fn fire(&mut self, now: Instant, ctx: &mut Self::Context) -> Option<Instant>;

    /// Hint for dual-wheel routing. Override to return `true` for periodic timers.
    #[inline(always)]
    fn is_periodic(&self) -> bool {
        false
    }
}

/// Trait for timer driver implementations.
///
/// Enables generic runtime code that works with any wheel variant.
pub trait TimerDriver<T: Timer>: Default {
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

impl<
    T,
    const PG: usize,
    const PR: u64,
    const PS: usize,
    const OG: usize,
    const OR: u64,
    const OS: usize,
    const OP: usize,
> TimerDriver<T> for DualBitWheel<T, PG, PR, PS, OG, OR, OS, OP>
where
    T: Timer,
{
    #[inline(always)]
    fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, InsertError<T>> {
        if timer.is_periodic() {
            DualBitWheel::insert_periodic(self, when, timer)
        } else {
            DualBitWheel::insert_oneshot(self, when, timer)
        }
    }

    #[inline(always)]
    fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        DualBitWheel::cancel(self, handle)
    }

    #[inline(always)]
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> Result<usize, PollError> {
        DualBitWheel::poll(self, now, ctx)
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

// ============================================================
// PERIODIC WHEELS - Bounded population, full gear probing
// All use MAX_PROBES = 64 (full gear scan)
// ============================================================

// FAST PERIODIC - For high-frequency sampling (25-100ms periods)
// 25ms resolution, 2 gears → ~1.7 min range
pub type FastPeriodicWheel<T> = BitWheel<T, 2, 25, 8, 64>;

// STANDARD PERIODIC - For heartbeats (1-30s periods)
// 100ms resolution, 2 gears → ~7 min range
pub type StandardPeriodicWheel<T> = BitWheel<T, 2, 100, 8, 64>;

// RELAXED PERIODIC - For slow checks (1-5 min periods)
// 500ms resolution, 2 gears → ~34 min range
pub type RelaxedPeriodicWheel<T> = BitWheel<T, 2, 500, 8, 64>;

// EXTENDED PERIODIC - For very slow periodics (5+ min periods)
// 1s resolution, 2 gears → ~68 min range
pub type ExtendedPeriodicWheel<T> = BitWheel<T, 2, 1000, 8, 64>;

// ============================================================
// DUAL WHEELS - Periodic + Oneshot combinations
// Naming: {Periodic}{Oneshot}DualWheel
// ============================================================

// ------------------------------------------------------------
// FAST PERIODIC (25ms) + BALANCED ONESHOT (5ms, general purpose)
// ------------------------------------------------------------

/// Fast periodic + Balanced oneshot.
/// Periodic: 25ms res, ~1.7 min. Oneshot: 5ms res, ~23 hrs.
pub type FastBalancedDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 5, 16, 8>;

/// Fast periodic + Balanced Fast oneshot.
pub type FastBalancedFastDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 5, 64, 2>;

/// Fast periodic + Balanced Light oneshot.
pub type FastBalancedLightDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 5, 8, 16>;

// ------------------------------------------------------------
// FAST PERIODIC (25ms) + PRECISE ONESHOT (1ms, fine timing)
// ------------------------------------------------------------

/// Fast periodic + Precise oneshot.
/// Periodic: 25ms res, ~1.7 min. Oneshot: 1ms res, ~4.7 hrs.
pub type FastPreciseDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 1, 16, 8>;

/// Fast periodic + Precise Fast oneshot.
pub type FastPreciseFastDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 1, 64, 2>;

/// Fast periodic + Precise Light oneshot.
pub type FastPreciseLightDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 1, 8, 16>;

// ------------------------------------------------------------
// FAST PERIODIC (25ms) + BURST ONESHOT (5ms, high volume)
// ------------------------------------------------------------

/// Fast periodic + Burst oneshot.
/// Periodic: 25ms res, ~1.7 min. Oneshot: 5ms res, ~23 hrs, 2x hotspot.
pub type FastBurstDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 5, 32, 8>;

/// Fast periodic + Burst Fast oneshot.
pub type FastBurstFastDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 5, 128, 2>;

/// Fast periodic + Burst Light oneshot.
pub type FastBurstLightDualWheel<T> = DualBitWheel<T, 2, 25, 8, 4, 5, 16, 16>;

// ------------------------------------------------------------
// STANDARD PERIODIC (100ms) + BALANCED ONESHOT
// ------------------------------------------------------------

/// Standard periodic + Balanced oneshot.
/// Periodic: 100ms res, ~7 min. Oneshot: 5ms res, ~23 hrs.
pub type StandardBalancedDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 5, 16, 8>;

/// Standard periodic + Balanced Fast oneshot.
pub type StandardBalancedFastDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 5, 64, 2>;

/// Standard periodic + Balanced Light oneshot.
pub type StandardBalancedLightDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 5, 8, 16>;

// ------------------------------------------------------------
// STANDARD PERIODIC (100ms) + PRECISE ONESHOT
// ------------------------------------------------------------

/// Standard periodic + Precise oneshot.
/// Periodic: 100ms res, ~7 min. Oneshot: 1ms res, ~4.7 hrs.
pub type StandardPreciseDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 1, 16, 8>;

/// Standard periodic + Precise Fast oneshot.
pub type StandardPreciseFastDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 1, 64, 2>;

/// Standard periodic + Precise Light oneshot.
pub type StandardPreciseLightDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 1, 8, 16>;

// ------------------------------------------------------------
// STANDARD PERIODIC (100ms) + BURST ONESHOT
// ------------------------------------------------------------

/// Standard periodic + Burst oneshot.
/// Periodic: 100ms res, ~7 min. Oneshot: 5ms res, ~23 hrs, 2x hotspot.
pub type StandardBurstDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 5, 32, 8>;

/// Standard periodic + Burst Fast oneshot.
pub type StandardBurstFastDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 5, 128, 2>;

/// Standard periodic + Burst Light oneshot.
pub type StandardBurstLightDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 5, 16, 16>;

// ------------------------------------------------------------
// RELAXED PERIODIC (500ms) + BALANCED ONESHOT
// ------------------------------------------------------------

/// Relaxed periodic + Balanced oneshot.
/// Periodic: 500ms res, ~34 min. Oneshot: 5ms res, ~23 hrs.
pub type RelaxedBalancedDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 5, 16, 8>;

/// Relaxed periodic + Balanced Fast oneshot.
pub type RelaxedBalancedFastDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 5, 64, 2>;

/// Relaxed periodic + Balanced Light oneshot.
pub type RelaxedBalancedLightDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 5, 8, 16>;

// ------------------------------------------------------------
// RELAXED PERIODIC (500ms) + PRECISE ONESHOT
// ------------------------------------------------------------

/// Relaxed periodic + Precise oneshot.
/// Periodic: 500ms res, ~34 min. Oneshot: 1ms res, ~4.7 hrs.
pub type RelaxedPreciseDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 1, 16, 8>;

/// Relaxed periodic + Precise Fast oneshot.
pub type RelaxedPreciseFastDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 1, 64, 2>;

/// Relaxed periodic + Precise Light oneshot.
pub type RelaxedPreciseLightDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 1, 8, 16>;

// ------------------------------------------------------------
// RELAXED PERIODIC (500ms) + BURST ONESHOT
// ------------------------------------------------------------

/// Relaxed periodic + Burst oneshot.
/// Periodic: 500ms res, ~34 min. Oneshot: 5ms res, ~23 hrs, 2x hotspot.
pub type RelaxedBurstDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 5, 32, 8>;

/// Relaxed periodic + Burst Fast oneshot.
pub type RelaxedBurstFastDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 5, 128, 2>;

/// Relaxed periodic + Burst Light oneshot.
pub type RelaxedBurstLightDualWheel<T> = DualBitWheel<T, 2, 500, 8, 4, 5, 16, 16>;

// ------------------------------------------------------------
// EXTENDED PERIODIC (1s) + EXTENDED ONESHOT (16ms, long duration)
// ------------------------------------------------------------

/// Extended periodic + Extended oneshot.
/// Periodic: 1s res, ~68 min. Oneshot: 16ms res, ~3 days.
pub type ExtendedDualWheel<T> = DualBitWheel<T, 2, 1000, 8, 4, 16, 16, 8>;

/// Extended periodic + Extended Fast oneshot.
pub type ExtendedFastDualWheel<T> = DualBitWheel<T, 2, 1000, 8, 4, 16, 64, 2>;

/// Extended periodic + Extended Light oneshot.
pub type ExtendedLightDualWheel<T> = DualBitWheel<T, 2, 1000, 8, 4, 16, 8, 16>;

#[macro_export]
macro_rules! define_bitwheel {
    ($name:ident, $timer:ty, $num_gears:expr, $resolution_ms:expr, $slot_cap:expr, $max_probes:expr) => {
        pub type $name =
            $crate::BitWheel<$timer, $num_gears, $resolution_ms, $slot_cap, $max_probes>;
    };
}

/// Convenience macro for defining custom dual wheel configurations.
#[macro_export]
macro_rules! define_dual_bitwheel {
    (
        $name:ident,
        $timer:ty,
        periodic: { gears: $p_gears:expr, resolution_ms: $p_res:expr, slot_cap: $p_cap:expr },
        oneshot: { gears: $o_gears:expr, resolution_ms: $o_res:expr, slot_cap: $o_cap:expr, max_probes: $o_probes:expr }
    ) => {
        pub type $name = $crate::DualBitWheel<
            $timer,
            $p_gears,
            $p_res,
            $p_cap,
            $o_gears,
            $o_res,
            $o_cap,
            $o_probes,
        >;
    };
}
