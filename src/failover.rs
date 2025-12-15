use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use crate::{
    BitWheel, DEFAULT_GEARS, DEFAULT_MAX_PROBES, DEFAULT_RESOLUTION_MS, DEFAULT_SLOT_CAP,
    PollError, Timer, TimerHandle, gear::InsertError,
};

pub struct BitWheelWithFailover<
    T,
    const NUM_GEARS: usize = DEFAULT_GEARS,
    const RESOLUTION_MS: u64 = DEFAULT_RESOLUTION_MS,
    const SLOT_CAP: usize = DEFAULT_SLOT_CAP,
    const MAX_PROBES: usize = DEFAULT_MAX_PROBES,
> {
    wheel: BitWheel<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES>,
    failover: BTreeMap<(u64, u32), T>,
    sequence: u32,
}

impl<
    T,
    const NUM_GEARS: usize,
    const RESOLUTION_MS: u64,
    const SLOT_CAP: usize,
    const MAX_PROBES: usize,
> Default for BitWheelWithFailover<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    T,
    const NUM_GEARS: usize,
    const RESOLUTION_MS: u64,
    const SLOT_CAP: usize,
    const MAX_PROBES: usize,
> BitWheelWithFailover<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES>
{
    pub fn new() -> Self {
        Self {
            wheel: BitWheel::new(),
            failover: BTreeMap::new(),
            sequence: 0,
        }
    }

    pub fn with_epoch(epoch: Instant) -> Self {
        Self {
            wheel: BitWheel::with_epoch(epoch),
            failover: BTreeMap::new(),
            sequence: 0,
        }
    }

    pub fn boxed() -> Box<Self> {
        Box::new(Self::new())
    }

    pub fn boxed_with_epoch(epoch: Instant) -> Box<Self> {
        Box::new(Self::with_epoch(epoch))
    }

    /// Insert never fails — falls back to BTreeMap if wheel is full
    pub fn insert(&mut self, when: Instant, timer: T) -> TimerHandle {
        match self.wheel.insert(when, timer) {
            Ok(handle) => handle,
            Err(InsertError(t)) => {
                let when_tick = self.wheel.instant_to_tick(when);
                let seq = self.sequence;
                self.sequence = self.sequence.wrapping_add(1);
                self.failover.insert((when_tick, seq), t);

                TimerHandle {
                    when_offset: when_tick,
                    key: seq,
                    gear: u8::MAX,
                    slot: u8::MAX,
                    overflow: true,
                }
            }
        }
    }

    pub fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        if handle.overflow {
            self.failover.remove(&(handle.when_offset, handle.key))
        } else {
            self.wheel.cancel(handle)
        }
    }

    pub fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> Result<usize, PollError>
    where
        T: Timer,
    {
        let wheel_result = self.wheel.poll(now, ctx);
        let current_tick = self.wheel.current_tick();
        let mut failover_fired = 0;

        while let Some(entry) = self.failover.first_entry() {
            let &(when_tick, _) = entry.key();
            if when_tick > current_tick {
                break;
            }

            let mut timer = entry.remove();
            failover_fired += 1;

            if let Some(next_when) = timer.fire(now, ctx) {
                let _ = self.insert(next_when, timer);
            }
        }

        match wheel_result {
            Ok(wheel_fired) => Ok(wheel_fired + failover_fired),
            Err(e) => Err(e),
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.wheel.is_empty() && self.failover.is_empty()
    }

    #[inline(always)]
    pub fn duration_until_next(&self) -> Option<Duration> {
        let wheel_next = self.wheel.duration_until_next();
        let failover_next = self.failover.first_key_value().map(|((tick, _), _)| {
            let ticks = tick.saturating_sub(self.wheel.current_tick());
            Duration::from_millis(ticks * RESOLUTION_MS)
        });

        match (wheel_next, failover_next) {
            (Some(w), Some(f)) => Some(w.min(f)),
            (Some(w), None) => Some(w),
            (None, Some(f)) => Some(f),
            (None, None) => None,
        }
    }

    /// Timers currently in failover
    #[inline(always)]
    pub fn failover_len(&self) -> usize {
        self.failover.len()
    }
}
