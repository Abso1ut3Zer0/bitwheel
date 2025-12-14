use std::time::{Duration, Instant};

use crate::gear::{Gear, GearError};

mod gear;
mod slot;

pub const DEFAULT_SLOT_CAP: usize = 32;

pub trait Timer {
    type Context;
    fn fire(&mut self, now: Instant, ctx: &mut Self::Context) -> Option<Instant>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct TimerHandle {
    when_offset: u64,
    key: u16,
    gear: u8,
    slot: u8,
}

pub struct BitWheel<
    T,
    const NUM_GEARS: usize = 5,
    const RESOLUTION_MS: u64 = 5,
    const SLOT_CAP: usize = DEFAULT_SLOT_CAP,
    const MAX_PROBES: usize = 3,
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

    #[inline]
    fn instant_to_tick(&self, when: Instant) -> u64 {
        when.saturating_duration_since(self.epoch).as_millis() as u64 / RESOLUTION_MS
    }

    #[inline]
    fn gear_for_delay(&self, delay: u64) -> usize {
        if delay == 0 {
            return 0;
        }
        let gear = (63 - delay.leading_zeros()) as usize / 6;
        gear.min(NUM_GEARS - 1)
    }

    #[inline]
    fn slot_for_tick(&self, gear: usize, tick: u64) -> usize {
        let shift = gear * 6;
        ((tick >> shift) & 63) as usize
    }
}
