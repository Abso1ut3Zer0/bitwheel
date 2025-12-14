use std::time::{Duration, Instant};

use crate::gear::{Gear, GearError};

mod gear;
mod slot;

pub const DEFAULT_SLOT_CAP: usize = 32;

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("{0} timers failed to reschedule")]
pub struct PollError(usize);

pub trait Timer {
    type Context;
    fn fire(&mut self, now: Instant, ctx: &mut Self::Context) -> Option<Instant>;
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
    key: u16,
    gear: u8,
    slot: u8,
}

pub struct BitWheel<
    T,
    const NUM_GEARS: usize,
    const RESOLUTION_MS: u64,
    const SLOT_CAP: usize,
    const MAX_PROBES: usize,
> {
    gears: [Gear<T, SLOT_CAP>; NUM_GEARS],
    epoch: Instant,
    current_tick: u64,
    next_fire_tick: Option<u64>,
}

impl<
    T,
    const NUM_GEARS: usize,
    const RESOLUTION_MS: u64,
    const SLOT_CAP: usize,
    const MAX_PROBES: usize,
> BitWheel<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES>
{
    pub fn with_epoch(epoch: Instant) -> Self {
        const {
            assert!(NUM_GEARS >= 1, "must have at least one gear");
            assert!(RESOLUTION_MS >= 1, "resolution must be at least 1ms");
            assert!(
                6 * NUM_GEARS + (64 - RESOLUTION_MS.leading_zeros() as usize) <= 64,
                "configuration would overflow u64 - reduce NUM_GEARS or RESOLUTION_MS"
            );
            assert!(MAX_PROBES < 64, "max probes must be less than 64");
        }

        Self {
            gears: std::array::from_fn(|_| Gear::new()),
            epoch,
            current_tick: 0,
            next_fire_tick: None,
        }
    }

    pub fn new() -> Self {
        Self::with_epoch(Instant::now())
    }

    /// Duration until next timer fires.
    #[inline(always)]
    pub fn duration_until_next(&self) -> Option<Duration> {
        self.next_fire_tick.map(|next| {
            let ticks_remaining = next.saturating_sub(self.current_tick);
            Duration::from_millis(ticks_remaining * RESOLUTION_MS)
        })
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.next_fire_tick.is_none()
    }

    pub fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, GearError> {
        let when_tick = self.instant_to_tick(when);
        let delay = when_tick.saturating_sub(self.current_tick).max(1);

        let gear_idx = self.gear_for_delay(delay);
        let target_slot = self.slot_for_tick(gear_idx, when_tick);

        let guard = self.gears[gear_idx].acquire_next_available(target_slot, MAX_PROBES)?;
        let actual_slot = guard.slot();
        let key = guard.insert(timer);

        self.next_fire_tick = Some(self.next_fire_tick.map_or(when_tick, |t| t.min(when_tick)));

        Ok(TimerHandle {
            when_offset: when_tick,
            key: key as u16,
            gear: gear_idx as u8,
            slot: actual_slot as u8,
        })
    }

    pub fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        // SAFETY: This remove is safe due to the following invariants:
        //
        // 1. TimerHandle is only created by insert(), which stores valid
        //    gear/slot/key values at the time of insertion.
        //
        // 2. TimerHandle has no Clone/Copy, so ownership is unique.
        //    Once cancel() takes ownership, no other cancel is possible.
        //
        // 3. The when_offset > current_tick check ensures the timer hasn't
        //    fired yet, so the entry must still exist in the wheel.
        //
        // Together these guarantee the key points to a valid, occupied entry.

        if handle.when_offset <= self.current_tick {
            return None;
        }

        let gear_idx = handle.gear as usize;
        let slot = handle.slot as usize;
        let key = handle.key as usize;

        if gear_idx >= NUM_GEARS {
            return None;
        }

        let guard = self.gears[gear_idx].acquire(slot);
        Some(guard.remove(key))
    }

    pub fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> Result<usize, PollError>
    where
        T: Timer,
    {
        let mut fired = 0usize;
        let mut lost = 0usize;

        let target_tick = self.instant_to_tick(now);

        if target_tick <= self.current_tick {
            return Ok(0);
        }

        for tick in (self.current_tick + 1)..=target_tick {
            self.poll_tick(tick, now, ctx, &mut fired, &mut lost);
        }

        self.current_tick = target_tick;
        self.recompute_next_fire();

        if lost > 0 {
            Err(PollError(lost))
        } else {
            Ok(fired)
        }
    }

    #[inline(always)]
    fn poll_tick(
        &mut self,
        tick: u64,
        now: Instant,
        ctx: &mut T::Context,
        fired: &mut usize,
        lost: &mut usize,
    ) where
        T: Timer,
    {
        for gear_idx in 0..NUM_GEARS {
            // Gear N only rotates when lower bits are zero
            if gear_idx > 0 {
                let mask = (1u64 << (6 * gear_idx)) - 1;
                if (tick & mask) != 0 {
                    continue;
                }
            }

            let slot = self.slot_for_tick(gear_idx, tick);
            self.drain_and_fire(gear_idx, slot, now, ctx, fired, lost);
        }
    }

    #[inline(always)]
    fn drain_and_fire(
        &mut self,
        gear_idx: usize,
        slot: usize,
        now: Instant,
        ctx: &mut T::Context,
        fired: &mut usize,
        lost: &mut usize,
    ) where
        T: Timer,
    {
        loop {
            let mut timer = {
                let guard = self.gears[gear_idx].acquire(slot);
                match guard.pop() {
                    Some(t) => t,
                    None => break,
                }
            };

            *fired += 1;

            if let Some(next_when) = timer.fire(now, ctx) {
                if self
                    .insert_excluding(next_when, timer, gear_idx, slot)
                    .is_err()
                {
                    *lost += 1;
                }
            }
        }
    }

    #[inline(always)]
    fn insert_excluding(
        &mut self,
        when: Instant,
        timer: T,
        excluded_gear: usize,
        excluded_slot: usize,
    ) -> Result<TimerHandle, GearError> {
        let when_tick = self.instant_to_tick(when);
        let delay = when_tick.saturating_sub(self.current_tick).max(1);

        let gear_idx = self.gear_for_delay(delay);
        let target_slot = self.slot_for_tick(gear_idx, when_tick);

        let guard = if gear_idx == excluded_gear {
            self.gears[gear_idx].acquire_next_available_excluding(
                excluded_slot,
                target_slot,
                MAX_PROBES,
            )?
        } else {
            self.gears[gear_idx].acquire_next_available(target_slot, MAX_PROBES)?
        };

        let actual_slot = guard.slot();
        let key = guard.insert(timer);

        self.next_fire_tick = Some(self.next_fire_tick.map_or(when_tick, |t| t.min(when_tick)));

        Ok(TimerHandle {
            when_offset: when_tick,
            key: key as u16,
            gear: gear_idx as u8,
            slot: actual_slot as u8,
        })
    }

    #[inline(always)]
    fn recompute_next_fire(&mut self) {
        self.next_fire_tick = None;

        for gear_idx in 0..NUM_GEARS {
            if let Some(tick) = self.next_fire_in_gear(gear_idx) {
                self.next_fire_tick = Some(self.next_fire_tick.map_or(tick, |t| t.min(tick)));
            }
        }
    }

    #[inline(always)]
    fn next_fire_in_gear(&self, gear_idx: usize) -> Option<u64> {
        let occupied = self.gears[gear_idx].occupied_bitmap();
        if occupied == 0 {
            return None;
        }

        let shift = gear_idx * 6;
        let current_slot = self.slot_for_tick(gear_idx, self.current_tick);

        // Rotate bitmap so current_slot + 1 is at bit 0
        // Then trailing_zeros gives distance to next occupied slot
        let rotation = (current_slot as u32 + 1) & 63;
        let rotated = occupied.rotate_right(rotation);

        let distance = rotated.trailing_zeros();
        if distance >= 64 {
            return None;
        }

        let next_slot = (current_slot + 1 + distance as usize) & 63;

        let ticks_to_slot = if next_slot > current_slot {
            next_slot - current_slot
        } else {
            64 - current_slot + next_slot
        };

        Some(self.current_tick + ((ticks_to_slot as u64) << shift))
    }

    #[inline(always)]
    fn instant_to_tick(&self, when: Instant) -> u64 {
        when.saturating_duration_since(self.epoch).as_millis() as u64 / RESOLUTION_MS
    }

    #[inline(always)]
    fn gear_for_delay(&self, delay: u64) -> usize {
        if delay == 0 {
            return 0;
        }
        let gear = (63 - delay.leading_zeros()) as usize / 6;
        gear.min(NUM_GEARS - 1)
    }

    #[inline(always)]
    fn slot_for_tick(&self, gear: usize, tick: u64) -> usize {
        let shift = gear * 6;
        ((tick >> shift) & 63) as usize
    }
}
