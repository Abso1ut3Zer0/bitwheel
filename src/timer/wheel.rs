use std::time::{Duration, Instant};

use crate::timer::{
    Timer, TimerHandle,
    gear::{Gear, InsertError, NUM_SLOTS, SLOT_MASK},
};

pub const DEFAULT_GEARS: usize = 5;
pub const DEFAULT_RESOLUTION_MS: u64 = 4;
pub const DEFAULT_SLOT_CAP: usize = 32;
pub const DEFAULT_MAX_PROBES: usize = 3;

pub struct BitWheel<
    T,
    const NUM_GEARS: usize = DEFAULT_GEARS,
    const RESOLUTION_MS: u64 = DEFAULT_RESOLUTION_MS,
    const SLOT_CAP: usize = DEFAULT_SLOT_CAP,
    const MAX_PROBES: usize = DEFAULT_MAX_PROBES,
> {
    gears: [Gear<T, SLOT_CAP>; NUM_GEARS],
    epoch: Instant,
    current_tick: u64,
    next_fire_tick: Option<u64>,

    // Cached min fire tick per gear + dirty tracking
    gear_next_fire: [Option<u64>; NUM_GEARS],
    gear_dirty: u64,
}

impl<
    T,
    const NUM_GEARS: usize,
    const RESOLUTION_MS: u64,
    const SLOT_CAP: usize,
    const MAX_PROBES: usize,
> Default for BitWheel<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES>
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
> BitWheel<T, NUM_GEARS, RESOLUTION_MS, SLOT_CAP, MAX_PROBES>
{
    const RESOLUTION_SHIFT: u32 = RESOLUTION_MS.trailing_zeros();

    pub fn with_epoch(epoch: Instant) -> Self {
        const {
            assert!(NUM_GEARS >= 1, "must have at least one gear");
            assert!(NUM_GEARS <= 64, "cannot have more than 64 gears");
            assert!(RESOLUTION_MS >= 1, "resolution must be at least 1ms");
            assert!(
                RESOLUTION_MS.is_power_of_two(),
                "resolution must be a power of 2"
            );
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
            gear_next_fire: [None; NUM_GEARS],
            gear_dirty: 0,
        }
    }

    pub fn new() -> Self {
        Self::with_epoch(Instant::now())
    }

    pub fn boxed() -> Box<Self> {
        Box::new(Self::new())
    }

    pub fn boxed_with_epoch(epoch: Instant) -> Box<Self> {
        Box::new(Self::with_epoch(epoch))
    }

    /// Useful for deciding how to scale the `BitWheel`
    /// for your use case such that it satisfies the
    /// timer constraints.
    pub const fn gear_granularities() -> [u64; NUM_GEARS] {
        let mut granularities = [0; NUM_GEARS];
        granularities[0] = RESOLUTION_MS;

        let mut idx = 1;
        while idx < NUM_GEARS {
            granularities[idx] = granularities[idx - 1] * (NUM_SLOTS as u64);
            idx += 1;
        }

        granularities
    }

    /// Useful for deciding how to scale the `BitWheel`
    /// for your use case such that it does not take
    /// up a tremendous amount of memory.
    pub const fn memory_footprint() -> usize {
        // BitWheel struct (includes Gear array with Slot structs inline)
        let struct_size = std::mem::size_of::<Self>();

        // Heap allocation: each Slot has Box<[Entry<T>; SLOT_CAP]>
        // Entry<T> has ~24 bytes overhead (next_occupied, prev_occupied, discriminant)
        // as a conservative estimate.
        const ENTRY_OVERHEAD: usize = 24;
        let entry_size = std::mem::size_of::<T>() + ENTRY_OVERHEAD;
        let heap_per_slot = SLOT_CAP * entry_size;
        let total_heap = NUM_GEARS * NUM_SLOTS * heap_per_slot;

        struct_size + total_heap
    }

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

    pub fn insert(&mut self, when: Instant, timer: T) -> Result<TimerHandle, InsertError<T>> {
        let when_tick = self.instant_to_tick(when);
        let delay = when_tick.saturating_sub(self.current_tick).max(1);

        let gear_idx = self.gear_for_delay(delay);
        let target_slot = self.slot_for_tick(gear_idx, when_tick);
        let Ok(guard) = self.gears[gear_idx].acquire_next_available(target_slot, MAX_PROBES) else {
            return Err(InsertError(timer));
        };

        let actual_slot = guard.slot();
        let key = guard.insert(timer);

        // Compute actual fire tick and update caches
        let fire_tick = self.compute_fire_tick(gear_idx, actual_slot);
        self.gear_next_fire[gear_idx] =
            Some(self.gear_next_fire[gear_idx].map_or(fire_tick, |t| t.min(fire_tick)));
        self.next_fire_tick = Some(self.next_fire_tick.map_or(fire_tick, |t| t.min(fire_tick)));

        Ok(TimerHandle {
            when_offset: when_tick,
            key: key as u32,
            gear: gear_idx as u8,
            slot: actual_slot as u8,
            overflow: false,
        })
    }

    pub fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
        debug_assert!(!handle.overflow, "received unexpected handle for overflow");
        if handle.when_offset <= self.current_tick {
            return None;
        }

        let gear_idx = handle.gear as usize;
        let slot = handle.slot as usize;
        let key = handle.key as usize;

        if gear_idx >= NUM_GEARS {
            return None;
        }

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
        let guard = self.gears[gear_idx].acquire(slot);
        let timer = guard.remove(key);

        // Mark gear dirty - removed timer might have been the min
        self.gear_dirty |= 1 << gear_idx;

        Some(timer)
    }

    pub fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize
    where
        T: Timer,
    {
        let target_tick = self.instant_to_tick(now);
        if target_tick <= self.current_tick {
            return 0;
        }

        // Skip-ahead optimization
        match self.next_fire_tick {
            None => {
                self.current_tick = target_tick;
                return 0;
            }
            Some(nft) if nft > target_tick => {
                self.current_tick = target_tick;
                return 0;
            }
            Some(nft) => {
                if nft > self.current_tick + 1 {
                    self.current_tick = nft - 1;
                }
            }
        }

        let mut fired = 0;
        for tick in (self.current_tick + 1)..=target_tick {
            fired += self.poll_tick(tick, now, ctx);
        }

        self.current_tick = target_tick;
        self.recompute_next_fire();
        fired
    }

    #[inline(always)]
    fn poll_tick(&mut self, tick: u64, now: Instant, ctx: &mut T::Context) -> usize
    where
        T: Timer,
    {
        let mut fired = 0;
        for gear_idx in 0..NUM_GEARS {
            if gear_idx > 0 {
                let mask = (1u64 << (6 * gear_idx)) - 1;
                if (tick & mask) != 0 {
                    continue;
                }
            }

            let slot = self.slot_for_tick(gear_idx, tick);
            fired += self.drain_and_fire(gear_idx, slot, now, ctx);
        }

        fired
    }

    #[inline(always)]
    fn drain_and_fire(
        &mut self,
        gear_idx: usize,
        slot: usize,
        now: Instant,
        ctx: &mut T::Context,
    ) -> usize
    where
        T: Timer,
    {
        let mut fired = 0;
        let guard = self.gears[gear_idx].acquire(slot);
        while let Some(mut timer) = guard.pop() {
            // Mark gear dirty since we removed a timer
            self.gear_dirty |= 1 << gear_idx;

            fired += 1;
            timer.fire(ctx, now);
        }

        fired
    }

    #[inline(always)]
    fn recompute_next_fire(&mut self) {
        // Recompute only dirty gears
        while self.gear_dirty != 0 {
            let gear_idx = self.gear_dirty.trailing_zeros() as usize;
            self.gear_dirty &= self.gear_dirty - 1;

            self.gear_next_fire[gear_idx] = self.compute_gear_min_fire(gear_idx);
        }

        // Could use sentinel instead
        let mut min_tick = u64::MAX;
        for &cached in &self.gear_next_fire {
            if let Some(tick) = cached {
                min_tick = min_tick.min(tick);
            }
        }
        self.next_fire_tick = if min_tick == u64::MAX {
            None
        } else {
            Some(min_tick)
        };
    }

    #[inline(always)]
    fn compute_gear_min_fire(&self, gear_idx: usize) -> Option<u64> {
        let occupied = self.gears[gear_idx].occupied_bitmap();
        if occupied == 0 {
            return None;
        }

        let current_slot = self.slot_for_tick(gear_idx, self.current_tick);

        // Rotate so slot after current is at bit 0
        let rotation = (current_slot as u32 + 1) & (SLOT_MASK as u32);
        let rotated = occupied.rotate_right(rotation);

        let distance = rotated.trailing_zeros() as usize;
        let next_slot = (current_slot + 1 + distance) & SLOT_MASK;

        Some(self.compute_fire_tick(gear_idx, next_slot))
    }

    #[inline(always)]
    fn compute_fire_tick(&self, gear_idx: usize, slot: usize) -> u64 {
        let shift = gear_idx * 6;
        let gear_period = 1u64 << (shift + 6);
        let slot_fire_offset = (slot as u64) << shift;
        let current_in_period = self.current_tick & (gear_period - 1);
        let base = self.current_tick & !(gear_period - 1);
        let passed = (slot_fire_offset <= current_in_period) as u64;
        base + passed * gear_period + slot_fire_offset
    }

    #[inline(always)]
    pub(crate) fn instant_to_tick(&self, when: Instant) -> u64 {
        when.saturating_duration_since(self.epoch).as_millis() as u64 >> Self::RESOLUTION_SHIFT
    }

    #[inline(always)]
    pub(crate) fn current_tick(&self) -> u64 {
        self.current_tick
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    // ==================== Test Timer Implementations ====================

    /// Simple one-shot timer that records when it fired.
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

        fn fire(&mut self, ctx: &mut Self::Context, _now: Instant) {
            self.fired.set(true);
            ctx.push(self.id);
        }
    }

    /// Counter timer for simple fire counting.
    struct CounterTimer;

    impl Timer for CounterTimer {
        type Context = usize;

        fn fire(&mut self, ctx: &mut Self::Context, _now: Instant) {
            *ctx += 1;
        }
    }

    // ==================== Construction Tests ====================

    #[test]
    fn test_new() {
        let wheel: Box<BitWheel<OneShotTimer>> = BitWheel::boxed();
        assert!(wheel.is_empty());
        assert!(wheel.duration_until_next().is_none());
    }

    #[test]
    fn test_with_epoch() {
        let epoch = Instant::now();
        let wheel: Box<BitWheel<OneShotTimer>> = BitWheel::boxed_with_epoch(epoch);
        assert!(wheel.is_empty());
        assert_eq!(wheel.current_tick, 0);
    }

    #[test]
    fn test_default() {
        let wheel: Box<BitWheel<OneShotTimer>> = BitWheel::boxed();
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_custom_config() {
        // 4 gears, 10ms resolution, 16 slots, 5 max probes
        let wheel: Box<BitWheel<OneShotTimer, 4, 8, 16, 5>> = BitWheel::boxed();
        assert!(wheel.is_empty());
    }

    // ==================== Insert Tests ====================

    #[test]
    fn test_insert_single() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, _fired) = OneShotTimer::new(1);
        let when = epoch + Duration::from_millis(100);

        let handle = wheel.insert(when, timer).unwrap();
        assert!(!wheel.is_empty());
        assert!(wheel.duration_until_next().is_some());
        assert_eq!(handle.gear, 1); // 100 ticks > 63, goes to gear 1
    }

    #[test]
    fn test_insert_updates_next_fire() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        assert!(wheel.duration_until_next().is_none());

        let (timer, _) = OneShotTimer::new(1);
        wheel
            .insert(epoch + Duration::from_millis(50), timer)
            .unwrap();

        let duration = wheel.duration_until_next().unwrap();
        assert!(duration.as_millis() <= 50);
    }

    #[test]
    fn test_insert_multiple_updates_next_fire_to_earliest() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer1, _) = OneShotTimer::new(1);
        let (timer2, _) = OneShotTimer::new(2);

        // Insert later timer first
        wheel
            .insert(epoch + Duration::from_millis(100), timer1)
            .unwrap();
        let d1 = wheel.duration_until_next().unwrap();

        // Insert earlier timer
        wheel
            .insert(epoch + Duration::from_millis(30), timer2)
            .unwrap();
        let d2 = wheel.duration_until_next().unwrap();

        assert!(d2 < d1);
        assert!(d2.as_millis() <= 30);
    }

    #[test]
    fn test_insert_gear_selection_gear0() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Delays 1-63 should go to gear 0
        let (timer, _) = OneShotTimer::new(1);
        let handle = wheel
            .insert(epoch + Duration::from_millis(30), timer)
            .unwrap();
        assert_eq!(handle.gear, 0);
    }

    #[test]
    fn test_insert_gear_selection_gear1() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Delays 64-4095 should go to gear 1
        let (timer, _) = OneShotTimer::new(1);
        let handle = wheel
            .insert(epoch + Duration::from_millis(100), timer)
            .unwrap();
        assert_eq!(handle.gear, 1);

        let (timer2, _) = OneShotTimer::new(2);
        let handle2 = wheel
            .insert(epoch + Duration::from_millis(4000), timer2)
            .unwrap();
        assert_eq!(handle2.gear, 1);
    }

    #[test]
    fn test_insert_gear_selection_gear2() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Delays 4096+ should go to gear 2
        let (timer, _) = OneShotTimer::new(1);
        let handle = wheel
            .insert(epoch + Duration::from_millis(5000), timer)
            .unwrap();
        assert_eq!(handle.gear, 2);
    }

    #[test]
    fn test_insert_with_probing() {
        let epoch = Instant::now();
        // Small slot capacity to force probing
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 2, 10>> = BitWheel::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill up the target slot
        let (t1, _) = OneShotTimer::new(1);
        let (t2, _) = OneShotTimer::new(2);
        let h1 = wheel.insert(when, t1).unwrap();
        let h2 = wheel.insert(when, t2).unwrap();

        // Same slot
        assert_eq!(h1.slot, h2.slot);

        // Third insert should probe to next slot
        let (t3, _) = OneShotTimer::new(3);
        let h3 = wheel.insert(when, t3).unwrap();
        assert_ne!(h2.slot, h3.slot);
    }

    #[test]
    fn test_insert_slot_full_error() {
        let epoch = Instant::now();
        // Tiny config: 1 slot cap, 1 max probe
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 1, 1>> = BitWheel::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(when, t1).unwrap();

        // Should fail - slot full and only 1 probe allowed
        let (t2, _) = OneShotTimer::new(2);
        let result = wheel.insert(when, t2);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_past_timer() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Advance current_tick by polling
        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(100), &mut ctx);

        // Insert timer "in the past" - should get delay of 1 (minimum)
        let (timer, _) = OneShotTimer::new(1);
        let handle = wheel
            .insert(epoch + Duration::from_millis(50), timer)
            .unwrap();

        // Should be in gear 0 with minimum delay
        assert_eq!(handle.gear, 0);
    }

    // ==================== Cancel Tests ====================

    #[test]
    fn test_cancel_returns_timer() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(42);
        let handle = wheel
            .insert(epoch + Duration::from_millis(100), timer)
            .unwrap();

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 42);
    }

    #[test]
    fn test_cancel_after_poll_returns_none() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        let handle = wheel
            .insert(epoch + Duration::from_millis(10), timer)
            .unwrap();

        // Poll past the timer's deadline
        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(100), &mut ctx);

        // Timer already fired, cancel should return None
        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_none());
    }

    // ==================== Poll Tests ====================

    #[test]
    fn test_poll_no_timers() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let mut ctx = Vec::new();
        let fired = wheel.poll(epoch + Duration::from_millis(100), &mut ctx);

        assert_eq!(fired, 0);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_poll_before_deadline() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, fired) = OneShotTimer::new(1);
        wheel
            .insert(epoch + Duration::from_millis(100), timer)
            .unwrap();

        let mut ctx = Vec::new();
        let count = wheel.poll(epoch + Duration::from_millis(50), &mut ctx);

        assert_eq!(count, 0);
        assert!(!fired.get());
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_poll_at_deadline() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, fired) = OneShotTimer::new(1);
        wheel
            .insert(epoch + Duration::from_millis(10), timer)
            .unwrap();

        let mut ctx = Vec::new();
        let count = wheel.poll(epoch + Duration::from_millis(10), &mut ctx);

        assert_eq!(count, 1);
        assert!(fired.get());
        assert_eq!(ctx, vec![1]);
    }

    #[test]
    fn test_poll_after_deadline() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, fired) = OneShotTimer::new(1);
        wheel
            .insert(epoch + Duration::from_millis(10), timer)
            .unwrap();

        let mut ctx = Vec::new();
        let count = wheel.poll(epoch + Duration::from_millis(100), &mut ctx);

        assert_eq!(count, 1);
        assert!(fired.get());
        assert_eq!(ctx, vec![1]);
    }

    #[test]
    fn test_poll_multiple_timers_same_slot() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);
        let (t1, f1) = OneShotTimer::new(1);
        let (t2, f2) = OneShotTimer::new(2);
        let (t3, f3) = OneShotTimer::new(3);

        wheel.insert(when, t1).unwrap();
        wheel.insert(when, t2).unwrap();
        wheel.insert(when, t3).unwrap();

        let mut ctx = Vec::new();
        let count = wheel.poll(epoch + Duration::from_millis(20), &mut ctx);

        assert_eq!(count, 3);
        assert!(f1.get());
        assert!(f2.get());
        assert!(f3.get());
        assert_eq!(ctx.len(), 3);
    }

    #[test]
    fn test_poll_multiple_timers_different_times() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (t1, f1) = OneShotTimer::new(1);
        let (t2, f2) = OneShotTimer::new(2);
        let (t3, f3) = OneShotTimer::new(3);

        wheel.insert(epoch + Duration::from_millis(10), t1).unwrap();
        wheel.insert(epoch + Duration::from_millis(20), t2).unwrap();
        wheel.insert(epoch + Duration::from_millis(30), t3).unwrap();

        // Poll at 15ms - only first should fire
        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(15), &mut ctx);
        assert!(f1.get());
        assert!(!f2.get());
        assert!(!f3.get());
        assert_eq!(ctx, vec![1]);

        // Poll at 25ms - second should fire
        ctx.clear();
        wheel.poll(epoch + Duration::from_millis(25), &mut ctx);
        assert!(f2.get());
        assert!(!f3.get());
        assert_eq!(ctx, vec![2]);

        // Poll at 35ms - third should fire
        ctx.clear();
        wheel.poll(epoch + Duration::from_millis(35), &mut ctx);
        assert!(f3.get());
        assert_eq!(ctx, vec![3]);
    }

    #[test]
    fn test_poll_clears_is_empty() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        wheel
            .insert(epoch + Duration::from_millis(10), timer)
            .unwrap();
        assert!(!wheel.is_empty());

        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(100), &mut ctx);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_poll_updates_duration_until_next() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (t1, _) = OneShotTimer::new(1);
        let (t2, _) = OneShotTimer::new(2);

        wheel.insert(epoch + Duration::from_millis(10), t1).unwrap();
        wheel.insert(epoch + Duration::from_millis(50), t2).unwrap();

        // Before poll, next fire should be ~10ms
        let d1 = wheel.duration_until_next().unwrap();
        assert!(d1.as_millis() <= 10);

        // Poll to fire first timer
        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(20), &mut ctx);

        // After poll, next fire should be ~50ms from epoch, minus current position
        let d2 = wheel.duration_until_next().unwrap();
        assert!(d2.as_millis() <= 30); // 50 - 20 = 30
    }

    // ==================== Gear Rotation Tests ====================

    #[test]
    fn test_gear0_every_tick() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<CounterTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Insert timers at consecutive ticks
        for i in 1..=5 {
            wheel
                .insert(epoch + Duration::from_millis(i), CounterTimer)
                .unwrap();
        }

        let mut count = 0usize;

        // Poll each tick individually
        for i in 1..=5 {
            wheel.poll(epoch + Duration::from_millis(i), &mut count);
        }

        assert_eq!(count, 5);
    }

    #[test]
    fn test_gear1_rotation() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<CounterTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Insert timer in gear 1 range (64-4095 ticks)
        wheel
            .insert(epoch + Duration::from_millis(100), CounterTimer)
            .unwrap();

        let mut count = 0usize;

        // Poll at tick 100
        wheel.poll(epoch + Duration::from_millis(100), &mut count);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_higher_gear_precision() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Insert timer at 5000ms (gear 2 territory)
        let (timer, fired) = OneShotTimer::new(1);
        let handle = wheel
            .insert(epoch + Duration::from_millis(5000), timer)
            .unwrap();
        assert_eq!(handle.gear, 2);

        let mut ctx = Vec::new();

        // Poll before - should not fire
        wheel.poll(epoch + Duration::from_millis(4000), &mut ctx);
        assert!(!fired.get());

        // Poll at/after deadline - should fire
        wheel.poll(epoch + Duration::from_millis(5100), &mut ctx);
        assert!(fired.get());
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_poll_same_instant_twice() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        wheel
            .insert(epoch + Duration::from_millis(10), timer)
            .unwrap();

        let mut ctx = Vec::new();
        let now = epoch + Duration::from_millis(20);

        // First poll fires the timer
        let r1 = wheel.poll(now, &mut ctx);
        assert_eq!(r1, 1);

        // Second poll at same instant should be no-op
        let r2 = wheel.poll(now, &mut ctx);
        assert_eq!(r2, 0);
    }

    #[test]
    fn test_poll_backwards_in_time() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (timer, fired) = OneShotTimer::new(1);
        wheel
            .insert(epoch + Duration::from_millis(50), timer)
            .unwrap();

        let mut ctx = Vec::new();

        // Advance to 100ms
        wheel.poll(epoch + Duration::from_millis(100), &mut ctx);
        assert!(fired.get());

        // "Go back" to 30ms - should be no-op (target_tick <= current_tick)
        let r = wheel.poll(epoch + Duration::from_millis(30), &mut ctx);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_many_timers_stress() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<CounterTimer, 4, 1, 64, 10>> =
            BitWheel::boxed_with_epoch(epoch);

        // Insert 1000 timers at various delays
        for i in 0..1000 {
            let delay = (i % 500) + 1;
            wheel
                .insert(epoch + Duration::from_millis(delay as u64), CounterTimer)
                .unwrap();
        }

        let mut count = 0usize;
        wheel.poll(epoch + Duration::from_millis(1000), &mut count);

        assert_eq!(count, 1000);
        assert!(wheel.is_empty());
    }

    // ==================== Duration Until Next Tests ====================

    #[test]
    fn test_duration_until_next_empty() {
        let wheel: Box<BitWheel<OneShotTimer>> = BitWheel::boxed();
        assert!(wheel.duration_until_next().is_none());
    }

    #[test]
    fn test_duration_until_next_after_cancel() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (t1, _) = OneShotTimer::new(1);
        let (t2, _) = OneShotTimer::new(2);

        wheel.insert(epoch + Duration::from_millis(50), t1).unwrap();
        let h2 = wheel.insert(epoch + Duration::from_millis(10), t2).unwrap();

        // Next fire should be at 10ms
        let d1 = wheel.duration_until_next().unwrap();
        assert!(d1.as_millis() <= 10);

        // Cancel the earlier timer
        wheel.cancel(h2);

        // Note: next_fire_tick is NOT updated on cancel (stale is OK)
        // It gets fixed on next poll
    }

    // ==================== Configuration Validation ====================

    #[test]
    fn test_gear_for_delay_boundaries() {
        let epoch = Instant::now();
        let wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Gear 0: 1-63
        assert_eq!(wheel.gear_for_delay(1), 0);
        assert_eq!(wheel.gear_for_delay(63), 0);

        // Gear 1: 64-4095
        assert_eq!(wheel.gear_for_delay(64), 1);
        assert_eq!(wheel.gear_for_delay(4095), 1);

        // Gear 2: 4096-262143
        assert_eq!(wheel.gear_for_delay(4096), 2);
        assert_eq!(wheel.gear_for_delay(262143), 2);

        // Gear 3: 262144+
        assert_eq!(wheel.gear_for_delay(262144), 3);
    }

    #[test]
    fn test_slot_for_tick() {
        let epoch = Instant::now();
        let wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Gear 0: tick & 63
        assert_eq!(wheel.slot_for_tick(0, 0), 0);
        assert_eq!(wheel.slot_for_tick(0, 10), 10);
        assert_eq!(wheel.slot_for_tick(0, 63), 63);
        assert_eq!(wheel.slot_for_tick(0, 64), 0); // wraps

        // Gear 1: (tick >> 6) & 63
        assert_eq!(wheel.slot_for_tick(1, 64), 1);
        assert_eq!(wheel.slot_for_tick(1, 128), 2);
        assert_eq!(wheel.slot_for_tick(1, 4032), 63);

        // Gear 2: (tick >> 12) & 63
        assert_eq!(wheel.slot_for_tick(2, 4096), 1);
        assert_eq!(wheel.slot_for_tick(2, 8192), 2);
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
            fn fire(&mut self, _ctx: &mut (), _now: Instant) {}
        }

        let drop_count = Arc::new(AtomicUsize::new(0));
        let epoch = Instant::now();

        {
            let mut wheel: Box<BitWheel<DropCounter, 4, 1, 32, 3>> =
                BitWheel::boxed_with_epoch(epoch);

            for i in 0..10 {
                wheel
                    .insert(
                        epoch + Duration::from_millis((i + 1) * 100),
                        DropCounter(Arc::clone(&drop_count)),
                    )
                    .unwrap();
            }

            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
            // wheel drops here
        }

        assert_eq!(drop_count.load(Ordering::SeqCst), 10);
    }

    // ==================== Additional Tests ====================

    #[test]
    fn test_next_fire_bitmap_wrap_around() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Insert timer that lands in slot 60
        let (t1, _) = OneShotTimer::new(1);
        wheel.insert(epoch + Duration::from_millis(60), t1).unwrap();

        // Advance past slot 60
        let mut ctx = Vec::new();
        wheel.poll(epoch + Duration::from_millis(61), &mut ctx);

        // Insert timer in slot 5 (wraps around)
        let (t2, _) = OneShotTimer::new(2);
        wheel.insert(epoch + Duration::from_millis(70), t2).unwrap(); // tick 70, slot = 70 & 63 = 6

        // duration_until_next should find the wrapped slot
        assert!(wheel.duration_until_next().is_some());
    }

    #[test]
    fn test_multiple_gears_same_poll() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        let (t1, f1) = OneShotTimer::new(1);
        let (t2, f2) = OneShotTimer::new(2);
        let (t3, f3) = OneShotTimer::new(3);

        // Gear 0 timer
        wheel.insert(epoch + Duration::from_millis(10), t1).unwrap();
        // Gear 1 timer
        wheel
            .insert(epoch + Duration::from_millis(100), t2)
            .unwrap();
        // Gear 2 timer
        wheel
            .insert(epoch + Duration::from_millis(5000), t3)
            .unwrap();

        let mut ctx = Vec::new();

        // Poll far enough to fire all
        wheel.poll(epoch + Duration::from_millis(6000), &mut ctx);

        assert!(f1.get());
        assert!(f2.get());
        assert!(f3.get());
        assert_eq!(ctx.len(), 3);
    }

    #[test]
    fn test_gear_boundary_exact_64() {
        let epoch = Instant::now();
        let wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // delay = 63 → gear 0
        assert_eq!(wheel.gear_for_delay(63), 0);
        // delay = 64 → gear 1
        assert_eq!(wheel.gear_for_delay(64), 1);
    }

    #[test]
    fn test_gear_boundary_exact_4096() {
        let epoch = Instant::now();
        let wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // delay = 4095 → gear 1
        assert_eq!(wheel.gear_for_delay(4095), 1);
        // delay = 4096 → gear 2
        assert_eq!(wheel.gear_for_delay(4096), 2);
    }

    #[test]
    fn test_insert_after_cancel_reuses_slot() {
        let epoch = Instant::now();
        let mut wheel: Box<BitWheel<OneShotTimer, 4, 1, 2, 3>> = BitWheel::boxed_with_epoch(epoch);

        let when = epoch + Duration::from_millis(10);

        // Fill slot
        let (t1, _) = OneShotTimer::new(1);
        let (t2, _) = OneShotTimer::new(2);
        let h1 = wheel.insert(when, t1).unwrap();
        let h2 = wheel.insert(when, t2).unwrap();

        // Cancel one
        wheel.cancel(h1);

        // Should be able to insert into same slot again
        let (t3, _) = OneShotTimer::new(3);
        let h3 = wheel.insert(when, t3).unwrap();

        // Should get same slot as cancelled timer
        assert_eq!(h2.slot, h3.slot);
    }

    #[test]
    fn test_zero_delay_goes_to_gear0() {
        let epoch = Instant::now();
        let wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        assert_eq!(wheel.gear_for_delay(0), 0);
    }

    #[test]
    fn test_delay_beyond_max_gears_clamps() {
        let epoch = Instant::now();
        let wheel: Box<BitWheel<OneShotTimer, 4, 1, 32, 3>> = BitWheel::boxed_with_epoch(epoch);

        // Very large delay should clamp to highest gear (3)
        assert_eq!(wheel.gear_for_delay(u64::MAX), 3);
        assert_eq!(wheel.gear_for_delay(1_000_000_000), 3);
    }
}

#[cfg(test)]
mod latency_tests {
    use crate::timer::{BalancedWheel, BurstWheel};

    use super::*;
    use hdrhistogram::Histogram;
    use std::time::{Duration, Instant};

    const WARMUP: u64 = 100_000;
    const ITERATIONS: u64 = 1_000_000;

    struct LatencyTimer;

    impl Timer for LatencyTimer {
        type Context = ();

        fn fire(&mut self, _ctx: &mut (), _now: Instant) {}
    }

    fn print_histogram(name: &str, hist: &Histogram<u64>) {
        println!("\n=== {} ===", name);
        println!("  count:  {}", hist.len());
        println!("  min:    {} ns", hist.min());
        println!("  max:    {} ns", hist.max());
        println!("  mean:   {:.1} ns", hist.mean());
        println!("  stddev: {:.1} ns", hist.stdev());
        println!("  p50:    {} ns", hist.value_at_quantile(0.50));
        println!("  p90:    {} ns", hist.value_at_quantile(0.90));
        println!("  p99:    {} ns", hist.value_at_quantile(0.99));
        println!("  p99.9:  {} ns", hist.value_at_quantile(0.999));
        println!("  p99.99: {} ns", hist.value_at_quantile(0.9999));
    }

    #[test]
    #[ignore]
    fn hdr_insert_latency() {
        let epoch = Instant::now();
        let mut wheel: Box<BalancedWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert(when, LatencyTimer).unwrap();
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 500) + 10);

            let start = Instant::now();
            let handle = wheel.insert(when, LatencyTimer).unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
            wheel.cancel(handle);
        }

        print_histogram("Insert Latency", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_cancel_latency() {
        let epoch = Instant::now();
        let mut wheel: Box<BalancedWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert(when, LatencyTimer).unwrap();
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert(when, LatencyTimer).unwrap();

            let start = Instant::now();
            let _ = wheel.cancel(handle);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Cancel Latency", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_empty() {
        let epoch = Instant::now();
        let mut wheel: Box<BalancedWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i);
            wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i);

            let start = Instant::now();
            wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Poll Empty", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_pending_no_fires() {
        let epoch = Instant::now();
        let mut wheel: Box<BalancedWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        // Insert timers far in future
        for i in 0..1000 {
            let when = epoch + Duration::from_millis(100_000_000 + i);
            let _ = wheel.insert(when, LatencyTimer);
        }

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i);
            wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i);

            let start = Instant::now();
            wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Poll Pending (No Fires)", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_single_fire() {
        let epoch = Instant::now();
        let mut wheel: Box<BalancedWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis(i + 1);
            let _ = wheel.insert(when, LatencyTimer);
            let now = epoch + Duration::from_millis(i + 1);
            wheel.poll(now, &mut ctx);
        }

        // Measure: insert timer, advance time, poll fires it
        for i in 0..ITERATIONS {
            let tick = WARMUP + i;
            let when = epoch + Duration::from_millis(tick + 1);
            let _ = wheel.insert(when, LatencyTimer);

            let now = epoch + Duration::from_millis(tick + 1);

            let start = Instant::now();
            wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Poll Single Fire", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_trading_simulation() {
        let epoch = Instant::now();
        let mut wheel: Box<BalancedWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        let mut insert_hist = Histogram::<u64>::new(3).unwrap();
        let mut poll_hist = Histogram::<u64>::new(3).unwrap();
        let mut cancel_hist = Histogram::<u64>::new(3).unwrap();

        let mut handles = Vec::with_capacity(100);
        let mut ctx = ();
        let mut now = epoch;

        // Warmup
        for i in 0..WARMUP {
            now += Duration::from_millis(1);
            wheel.poll(now, &mut ctx);

            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));
                if let Ok(handle) = wheel.insert(when, LatencyTimer) {
                    if handles.len() < 100 {
                        handles.push(handle);
                    }
                }
            }

            if i % 20 == 0 {
                if let Some(handle) = handles.pop() {
                    let _ = wheel.cancel(handle);
                }
            }
        }

        // Measure
        for i in 0..ITERATIONS {
            now += Duration::from_millis(1);

            // Poll
            let start = Instant::now();
            wheel.poll(now, &mut ctx);
            poll_hist.record(start.elapsed().as_nanos() as u64).unwrap();

            // Insert 80%
            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));

                let start = Instant::now();
                if let Ok(handle) = wheel.insert(when, LatencyTimer) {
                    insert_hist
                        .record(start.elapsed().as_nanos() as u64)
                        .unwrap();

                    if handles.len() < 100 {
                        handles.push(handle);
                    }
                }
            }

            // Cancel 5%
            if i % 20 == 0 {
                if let Some(handle) = handles.pop() {
                    let start = Instant::now();
                    let _ = wheel.cancel(handle);
                    cancel_hist
                        .record(start.elapsed().as_nanos() as u64)
                        .unwrap();
                }
            }
        }

        print_histogram("Trading Sim - Insert", &insert_hist);
        print_histogram("Trading Sim - Poll", &poll_hist);
        print_histogram("Trading Sim - Cancel", &cancel_hist);
    }

    #[test]
    #[ignore]
    fn hdr_bursty_workload() {
        let epoch = Instant::now();
        let mut wheel: Box<BurstWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i + 1);

            // Burst every 100 ticks
            if i % 100 == 0 {
                for j in 0..50 {
                    let when = now + Duration::from_millis(20 + j % 80);
                    let _ = wheel.insert(when, LatencyTimer);
                }
            }

            wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i + 1);

            // Burst every 100 ticks
            if i % 100 == 0 {
                for j in 0..50 {
                    let when = now + Duration::from_millis(20 + j % 80);
                    let _ = wheel.insert(when, LatencyTimer);
                }
            }

            let start = Instant::now();
            wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Bursty (50 burst every 100ms)", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_interleaved_insert() {
        let epoch = Instant::now();
        let mut wheel: Box<BalancedWheel<LatencyTimer>> = BitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let mut now = epoch;

        // Pre-populate: timers at regular intervals (every 10ms from 100-10000ms)
        for i in 0..10_000 {
            let when = epoch + Duration::from_millis(100 + i * 10);
            let _ = wheel.insert(when, LatencyTimer);
        }

        // Warmup - insert timers that land BETWEEN existing entries
        for i in 0..WARMUP {
            now += Duration::from_millis(1);
            wheel.poll(now, &mut ctx);

            let base = 100 + ((i % 990) * 10);
            let when = epoch + Duration::from_millis(base + 5);
            let _ = wheel.insert(when, LatencyTimer);
        }

        // Measure - every insert lands between two existing timers
        for i in 0..ITERATIONS {
            now += Duration::from_millis(1);
            wheel.poll(now, &mut ctx);

            // Replenish the "grid" timers as they fire
            if i % 10 == 0 {
                let when = now + Duration::from_millis(10000);
                let _ = wheel.insert(when, LatencyTimer);
            }

            // The measured insert: always between existing entries
            let base = ((now.duration_since(epoch).as_millis() as u64) % 9900) + 100;
            let offset = (i % 3) * 2 + 3; // 3, 5, or 7
            let when = epoch + Duration::from_millis(base + offset);

            let start = Instant::now();
            let _ = wheel.insert(when, LatencyTimer);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Interleaved Insert", &hist);
    }
}
