use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use crate::{
    BitWheel, DEFAULT_GEARS, DEFAULT_MAX_PROBES, DEFAULT_RESOLUTION_MS, DEFAULT_SLOT_CAP, Timer,
    TimerHandle, gear::InsertError,
};

pub const DEFAULT_FAILOVER_INTERVAL: u64 = 32;

pub struct BitWheelWithFailover<
    T,
    const NUM_GEARS: usize = DEFAULT_GEARS,
    const RESOLUTION_MS: u64 = DEFAULT_RESOLUTION_MS,
    const SLOT_CAP: usize = DEFAULT_SLOT_CAP,
    const MAX_PROBES: usize = DEFAULT_MAX_PROBES,
    const FAILOVER_INTERVAL: u64 = DEFAULT_FAILOVER_INTERVAL,
> {
    wheel: BitWheel<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES>,
    failover: BTreeMap<(u64, u32), T>,
    sequence: u32,
    last_check: u64,
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
    const FAILOVER_INTERVAL: u64,
> BitWheelWithFailover<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES, FAILOVER_INTERVAL>
{
    pub fn new() -> Self {
        Self {
            wheel: BitWheel::new(),
            failover: BTreeMap::new(),
            last_check: 0,
            sequence: 0,
        }
    }

    pub fn with_epoch(epoch: Instant) -> Self {
        Self {
            wheel: BitWheel::with_epoch(epoch),
            failover: BTreeMap::new(),
            last_check: 0,
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

    pub fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize
    where
        T: Timer,
    {
        let fired_from_wheel = self.wheel.poll_with_failover(now, ctx, |when, timer| {
            let sequence = self.sequence;
            self.sequence = self.sequence.wrapping_add(1);
            self.failover.insert((when, sequence), timer);
        });

        let mut fired_from_failover = 0;

        // Check failover every FAILOVER_INTERVAL ticks
        let current_interval = self.wheel.current_tick() / FAILOVER_INTERVAL;

        if current_interval > self.last_check && !self.failover.is_empty() {
            self.last_check = current_interval;
            let current_tick = self.wheel.current_tick();

            while let Some(entry) = self.failover.first_entry() {
                let &(when_tick, _) = entry.key();
                if when_tick > current_tick {
                    break;
                }

                let mut timer = entry.remove();
                fired_from_failover += 1;

                if let Some(next_when) = timer.fire(now, ctx) {
                    let _ = self.insert(next_when, timer);
                }
            }
        }

        fired_from_wheel + fired_from_failover
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    // One lap = 64 ticks. At 1ms resolution, that's 64ms.
    const ONE_LAP_MS: u64 = 64;

    // ==================== Test Timer Implementations ====================

    struct OneShotTimer {
        id: usize,
        fired: Rc<Cell<bool>>,
    }

    impl OneShotTimer {
        fn new(id: usize) -> (Self, Rc<Cell<bool>>) {
            let fired = Rc::new(Cell::new(false));
            (
                Self {
                    id,
                    fired: Rc::clone(&fired),
                },
                fired,
            )
        }
    }

    impl Timer for OneShotTimer {
        type Context = Vec<usize>;

        fn fire(&mut self, _now: Instant, ctx: &mut Self::Context) -> Option<Instant> {
            self.fired.set(true);
            ctx.push(self.id);
            None
        }
    }

    struct PeriodicTimer {
        id: usize,
        period: Duration,
        max_fires: usize,
        fire_count: usize,
    }

    impl PeriodicTimer {
        fn new(id: usize, period: Duration, max_fires: usize) -> Self {
            Self {
                id,
                period,
                max_fires,
                fire_count: 0,
            }
        }
    }

    impl Timer for PeriodicTimer {
        type Context = Vec<(usize, usize)>;

        fn fire(&mut self, now: Instant, ctx: &mut Self::Context) -> Option<Instant> {
            self.fire_count += 1;
            ctx.push((self.id, self.fire_count));

            if self.fire_count < self.max_fires {
                Some(now + self.period)
            } else {
                None
            }
        }
    }

    struct CounterTimer;

    impl Timer for CounterTimer {
        type Context = usize;

        fn fire(&mut self, _now: Instant, ctx: &mut Self::Context) -> Option<Instant> {
            *ctx += 1;
            None
        }
    }

    // ==================== Construction Tests ====================

    #[test]
    fn test_new() {
        let wheel: Box<BitWheelWithFailover<OneShotTimer>> = BitWheelWithFailover::boxed();
        assert!(wheel.is_empty());
        assert!(wheel.duration_until_next().is_none());
        assert_eq!(wheel.failover_len(), 0);
    }

    #[test]
    fn test_with_epoch() {
        let epoch = Instant::now();
        let wheel: Box<BitWheelWithFailover<OneShotTimer>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);
        assert!(wheel.is_empty());
        assert_eq!(wheel.failover_len(), 0);
    }

    #[test]
    fn test_default() {
        let wheel: BitWheelWithFailover<OneShotTimer> = BitWheelWithFailover::default();
        assert!(wheel.is_empty());
    }

    // ==================== Insert Tests ====================

    #[test]
    fn test_insert_goes_to_wheel() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 32, 3>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let (timer, _fired) = OneShotTimer::new(1);
        let handle = wheel.insert(epoch + Duration::from_millis(100), timer);

        assert!(!wheel.is_empty());
        assert!(!handle.overflow);
        assert_eq!(wheel.failover_len(), 0);
    }

    #[test]
    fn test_insert_overflow_goes_to_failover() {
        let epoch = Instant::now();
        // Tiny wheel: 1 slot capacity, 1 probe
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // First insert goes to wheel
        let (t1, _) = OneShotTimer::new(1);
        let h1 = wheel.insert(when, t1);
        assert!(!h1.overflow);
        assert_eq!(wheel.failover_len(), 0);

        // Second insert overflows to failover
        let (t2, _) = OneShotTimer::new(2);
        let h2 = wheel.insert(when, t2);
        assert!(h2.overflow);
        assert_eq!(wheel.failover_len(), 1);
    }

    #[test]
    fn test_insert_never_fails() {
        let epoch = Instant::now();
        // Tiny wheel that will overflow
        let mut wheel: Box<BitWheelWithFailover<CounterTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Insert 100 timers - none should fail
        for _ in 0..100 {
            let _handle = wheel.insert(when, CounterTimer);
        }

        assert!(!wheel.is_empty());
        // Most should be in failover
        assert!(wheel.failover_len() > 90);
    }

    #[test]
    fn test_insert_failover_sequence_increments() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // These go to failover with incrementing sequences
        let (t2, _) = OneShotTimer::new(2);
        let h2 = wheel.insert(when, t2);

        let (t3, _) = OneShotTimer::new(3);
        let h3 = wheel.insert(when, t3);

        assert!(h2.overflow);
        assert!(h3.overflow);
        assert_eq!(h2.key, 0);
        assert_eq!(h3.key, 1);
    }

    // ==================== Cancel Tests ====================

    #[test]
    fn test_cancel_from_wheel() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 32, 3>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(42);
        let handle = wheel.insert(epoch + Duration::from_millis(100), timer);

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 42);
    }

    #[test]
    fn test_cancel_from_failover() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // This goes to failover
        let (t2, _) = OneShotTimer::new(42);
        let handle = wheel.insert(when, t2);
        assert!(handle.overflow);
        assert_eq!(wheel.failover_len(), 1);

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 42);
        assert_eq!(wheel.failover_len(), 0);
    }

    #[test]
    fn test_cancel_wrong_handle_returns_none() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // Insert to failover
        let (t2, _) = OneShotTimer::new(2);
        let handle = wheel.insert(when, t2);

        // Cancel it
        let when_offset = handle.when_offset;
        let key = handle.key;
        wheel.cancel(handle);

        // Try to cancel again with same handle info (manually constructed)
        let fake_handle = TimerHandle {
            when_offset,
            key,
            gear: u8::MAX,
            slot: u8::MAX,
            overflow: true,
        };
        let result = wheel.cancel(fake_handle);
        assert!(result.is_none());
    }

    // ==================== Poll Tests ====================

    #[test]
    fn test_poll_empty() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 32, 3>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let mut ctx = Vec::new();
        let fired = wheel.poll(epoch + Duration::from_millis(100), &mut ctx);

        assert_eq!(fired, 0);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_poll_fires_from_wheel() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 32, 3>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let (timer, fired_flag) = OneShotTimer::new(1);
        wheel.insert(epoch + Duration::from_millis(10), timer);

        let mut ctx = Vec::new();
        let fired = wheel.poll(epoch + Duration::from_millis(20), &mut ctx);

        assert_eq!(fired, 1);
        assert!(fired_flag.get());
        assert_eq!(ctx, vec![1]);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_poll_fires_from_failover_after_lap() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let (t1, f1) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // This goes to failover
        let (t2, f2) = OneShotTimer::new(2);
        let handle = wheel.insert(when, t2);
        assert!(handle.overflow);

        let mut ctx = Vec::new();

        // Poll at 20ms - wheel timer fires, but failover not drained yet (no lap boundary crossed)
        wheel.poll(epoch + Duration::from_millis(20), &mut ctx);
        assert!(f1.get());
        assert!(!f2.get()); // failover not drained yet
        assert_eq!(wheel.failover_len(), 1);

        // Poll past lap boundary (64ms) - failover drains
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS + 10), &mut ctx);
        assert!(f2.get());
        assert_eq!(wheel.failover_len(), 0);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_poll_failover_not_drained_before_interval() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // This goes to failover
        let (t2, f2) = OneShotTimer::new(2);
        wheel.insert(when, t2);

        let mut ctx = Vec::new();

        // Poll at 20ms - before interval boundary, failover should NOT drain
        wheel.poll(epoch + Duration::from_millis(20), &mut ctx);
        assert!(!f2.get());
        assert_eq!(wheel.failover_len(), 1);

        // Poll at 31ms - still before interval boundary
        wheel.poll(epoch + Duration::from_millis(31), &mut ctx);
        assert!(!f2.get());
        assert_eq!(wheel.failover_len(), 1);
    }

    #[test]
    fn test_poll_mixed_wheel_and_failover() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        // Insert at same time (one wheel, one failover)
        let (t1, f1) = OneShotTimer::new(1);
        wheel.insert(epoch + Duration::from_millis(10), t1); // wheel

        let (t2, f2) = OneShotTimer::new(2);
        wheel.insert(epoch + Duration::from_millis(10), t2); // failover

        let mut ctx = Vec::new();

        // Poll at 15ms - fires wheel timer only
        wheel.poll(epoch + Duration::from_millis(15), &mut ctx);
        assert!(f1.get());
        assert!(!f2.get());

        // Poll past lap boundary - fires failover timer
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS + 10), &mut ctx);
        assert!(f2.get());

        assert!(wheel.is_empty());
    }

    #[test]
    fn test_poll_before_deadline_no_fire() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(100);

        // One in wheel, one in failover
        let (t1, f1) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        let (t2, f2) = OneShotTimer::new(2);
        wheel.insert(when, t2);

        let mut ctx = Vec::new();
        let fired = wheel.poll(epoch + Duration::from_millis(50), &mut ctx);

        assert_eq!(fired, 0);
        assert!(!f1.get());
        assert!(!f2.get());
    }

    // ==================== Periodic Timer Tests ====================

    #[test]
    fn test_periodic_reschedules_to_wheel() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<PeriodicTimer, 4, 1, 32, 3>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let timer = PeriodicTimer::new(1, Duration::from_millis(10), 3);
        wheel.insert(epoch + Duration::from_millis(10), timer);

        let mut ctx = Vec::new();

        // Fire 1
        wheel.poll(epoch + Duration::from_millis(15), &mut ctx);
        assert_eq!(ctx, vec![(1, 1)]);
        assert!(!wheel.is_empty());
        assert_eq!(wheel.failover_len(), 0); // reschedule went to wheel

        // Fire 2
        wheel.poll(epoch + Duration::from_millis(30), &mut ctx);
        assert_eq!(ctx, vec![(1, 1), (1, 2)]);

        // Fire 3 (last)
        wheel.poll(epoch + Duration::from_millis(45), &mut ctx);
        assert_eq!(ctx, vec![(1, 1), (1, 2), (1, 3)]);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_periodic_in_failover_fires_on_lap_boundary() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<PeriodicTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let blocker = PeriodicTimer::new(99, Duration::from_millis(1000), 1);
        wheel.insert(when, blocker);

        // Periodic goes to failover
        let timer = PeriodicTimer::new(1, Duration::from_millis(100), 3);
        let handle = wheel.insert(when, timer);
        assert!(handle.overflow);

        let mut ctx = Vec::new();

        // Poll at 20ms - wheel timer fires, failover not yet
        wheel.poll(epoch + Duration::from_millis(20), &mut ctx);
        assert!(ctx.contains(&(99, 1)));
        assert!(!ctx.contains(&(1, 1)));

        // Poll past first lap - failover fires
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS + 10), &mut ctx);
        assert!(ctx.contains(&(1, 1)));

        // Poll past second lap - second fire
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS * 2 + 10), &mut ctx);
        assert!(ctx.contains(&(1, 2)));

        // Poll past third lap - third fire
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS * 3 + 10), &mut ctx);
        assert!(ctx.contains(&(1, 3)));
    }

    // ==================== Duration Until Next Tests ====================

    #[test]
    fn test_duration_until_next_empty() {
        let wheel: BitWheelWithFailover<OneShotTimer> = BitWheelWithFailover::new();
        assert!(wheel.duration_until_next().is_none());
    }

    #[test]
    fn test_duration_until_next_wheel_only() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 32, 3>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        wheel.insert(epoch + Duration::from_millis(50), timer);

        let duration = wheel.duration_until_next().unwrap();
        assert!(duration.as_millis() <= 50);
    }

    #[test]
    fn test_duration_until_next_with_failover() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(50);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // Same time goes to failover
        let (t2, _) = OneShotTimer::new(2);
        let handle = wheel.insert(when, t2);
        assert!(handle.overflow);

        let duration = wheel.duration_until_next().unwrap();
        assert!(duration.as_millis() <= 50);
    }

    #[test]
    fn test_duration_until_next_returns_min() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        // Wheel timer at 100ms
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(epoch + Duration::from_millis(100), t1);

        // Failover timer at 50ms (earlier)
        let (t2, _) = OneShotTimer::new(2);
        wheel.insert(epoch + Duration::from_millis(50), t2);

        let duration = wheel.duration_until_next().unwrap();
        assert!(duration.as_millis() <= 50);
    }

    // ==================== Failover Len Tests ====================

    #[test]
    fn test_failover_len_tracks_correctly() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        assert_eq!(wheel.failover_len(), 0);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);
        assert_eq!(wheel.failover_len(), 0);

        // Add to failover
        let (t2, _) = OneShotTimer::new(2);
        wheel.insert(when, t2);
        assert_eq!(wheel.failover_len(), 1);

        let (t3, _) = OneShotTimer::new(3);
        wheel.insert(when, t3);
        assert_eq!(wheel.failover_len(), 2);

        // Poll past lap boundary drains failover
        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS + 10), &mut ctx);
        assert_eq!(wheel.failover_len(), 0);
    }

    // ==================== Is Empty Tests ====================

    #[test]
    fn test_is_empty_wheel_only() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 32, 3>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        assert!(wheel.is_empty());

        let (timer, _) = OneShotTimer::new(1);
        wheel.insert(epoch + Duration::from_millis(10), timer);
        assert!(!wheel.is_empty());

        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(20), &mut ctx);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_is_empty_with_failover() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // Goes to failover
        let (t2, _) = OneShotTimer::new(2);
        wheel.insert(when, t2);

        assert!(!wheel.is_empty());

        // Poll fires wheel timer but not failover
        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(20), &mut ctx);
        assert!(!wheel.is_empty()); // failover still has timer

        // Poll past lap boundary fires failover
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS + 10), &mut ctx);
        assert!(wheel.is_empty());
    }

    // ==================== Stress Tests ====================

    #[test]
    fn test_many_timers_all_fire() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<CounterTimer, 4, 1, 32, 8>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        // Insert 1000 timers - many will overflow
        for i in 0..1000 {
            let delay = (i % 50) + 1; // delays 1-50ms
            wheel.insert(epoch + Duration::from_millis(delay as u64), CounterTimer);
        }

        let mut count = 0usize;

        // Poll past multiple laps to ensure all failover timers drain
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS * 3), &mut count);

        assert_eq!(count, 1000);
        assert!(wheel.is_empty());
        assert_eq!(wheel.failover_len(), 0);
    }

    #[test]
    fn test_sequence_wrapping() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<CounterTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        // Manually set sequence near max
        wheel.sequence = u32::MAX - 2;

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        wheel.insert(when, CounterTimer);

        // These go to failover and should wrap sequence
        let h1 = wheel.insert(when, CounterTimer);
        let h2 = wheel.insert(when, CounterTimer);
        let h3 = wheel.insert(when, CounterTimer);
        let h4 = wheel.insert(when, CounterTimer);

        assert_eq!(h1.key, u32::MAX - 2);
        assert_eq!(h2.key, u32::MAX - 1);
        assert_eq!(h3.key, u32::MAX);
        assert_eq!(h4.key, 0); // wrapped

        // All should still fire (poll past lap boundary)
        let mut count = 0usize;
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS + 10), &mut count);
        assert_eq!(count, 5);
    }

    // ==================== Lap Boundary Tests ====================

    #[test]
    fn test_failover_drains_at_interval_boundary() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill wheel
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1);

        // Goes to failover
        let (t2, f2) = OneShotTimer::new(2);
        wheel.insert(when, t2);

        let mut ctx = Vec::new();

        // Poll at 31ms - just before interval boundary
        wheel.poll(epoch + Duration::from_millis(31), &mut ctx);
        assert!(!f2.get());
        assert_eq!(wheel.failover_len(), 1);

        // Poll at 32ms - crosses interval boundary, failover drains
        wheel.poll(epoch + Duration::from_millis(32), &mut ctx);
        assert!(f2.get());
        assert_eq!(wheel.failover_len(), 0);
    }
    #[test]
    fn test_multiple_lap_boundaries() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheelWithFailover<OneShotTimer, 4, 1, 1, 1>> =
            BitWheelWithFailover::boxed_with_epoch(epoch);

        let mut ctx = Vec::new();

        // Insert and fire across multiple laps
        for lap in 0..5 {
            let when = epoch + Duration::from_millis(lap * ONE_LAP_MS + 10);

            // Fill wheel
            let (t1, _) = OneShotTimer::new(lap as usize * 2);
            wheel.insert(when, t1);

            // Goes to failover
            let (t2, _) = OneShotTimer::new(lap as usize * 2 + 1);
            wheel.insert(when, t2);
        }

        // Poll through all laps
        wheel.poll(epoch + Duration::from_millis(ONE_LAP_MS * 6), &mut ctx);

        assert_eq!(ctx.len(), 10); // 5 laps × 2 timers
        assert!(wheel.is_empty());
    }

    // ==================== Drop Behavior ====================

    #[test]
    fn test_drop_pending_timers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct DropCounter(Arc<AtomicUsize>);

        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        impl Timer for DropCounter {
            type Context = ();
            fn fire(&mut self, _now: Instant, _ctx: &mut ()) -> Option<Instant> {
                None
            }
        }

        let drop_count = Arc::new(AtomicUsize::new(0));
        let epoch = Instant::now();

        {
            let mut wheel: Box<BitWheelWithFailover<DropCounter, 4, 1, 1, 1>> =
                BitWheelWithFailover::boxed_with_epoch(epoch);

            let when = epoch + Duration::from_millis(100);

            // Insert 5 timers - some in wheel, some in failover
            for _ in 0..5 {
                wheel.insert(when, DropCounter(Arc::clone(&drop_count)));
            }

            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
            // wheel drops here
        }

        assert_eq!(drop_count.load(Ordering::SeqCst), 5);
    }
}
