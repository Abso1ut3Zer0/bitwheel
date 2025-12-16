use std::time::{Duration, Instant};

use crate::{
    BitWheel, PollError, Timer, TimerHandle,
    gear::{InsertError, NUM_SLOTS},
};

// Periodic defaults: small, coarse, probe everything
pub const DEFAULT_P_NUM_GEARS: usize = 2;
pub const DEFAULT_P_RESOLUTION_MS: u64 = 100;
pub const DEFAULT_P_SLOT_CAP: usize = 8;

// Oneshot defaults: larger, precise, limited probes
pub const DEFAULT_O_NUM_GEARS: usize = 5;
pub const DEFAULT_O_RESOLUTION_MS: u64 = 5;
pub const DEFAULT_O_SLOT_CAP: usize = 32;
pub const DEFAULT_O_MAX_PROBES: usize = 3;

/// High bit of gear field indicates periodic wheel
const PERIODIC_FLAG: u8 = 0x80;
const GEAR_MASK: u8 = 0x7F;

/// A dual-lane timer wheel with independent configurations for periodic and one-shot timers.
///
/// **Periodic wheel**: Optimized for bounded, long-duration timers (heartbeats, risk checks).
/// - Coarse resolution (100ms default) — precision doesn't matter for 30s heartbeats
/// - Fewer gears (2 default) — covers ~7 minutes, plenty for most periodic needs
/// - Full gear probing (64 slots) — always succeeds if under capacity
/// - Small slot capacity — bounded population
///
/// **Oneshot wheel**: Optimized for numerous, short-duration timers (order timeouts).
/// - Fine resolution (5ms default) — precision matters for tight timeouts
/// - More gears (5 default) — covers hours of delay range
/// - Limited probing — backpressure signal on overflow
/// - Larger slot capacity — handles burst load
pub struct DualBitWheel<
    T,
    // Periodic wheel configuration
    const P_NUM_GEARS: usize = DEFAULT_P_NUM_GEARS,
    const P_RESOLUTION_MS: u64 = DEFAULT_P_RESOLUTION_MS,
    const P_SLOT_CAP: usize = DEFAULT_P_SLOT_CAP,
    // Oneshot wheel configuration
    const O_NUM_GEARS: usize = DEFAULT_O_NUM_GEARS,
    const O_RESOLUTION_MS: u64 = DEFAULT_O_RESOLUTION_MS,
    const O_SLOT_CAP: usize = DEFAULT_O_SLOT_CAP,
    const O_MAX_PROBES: usize = DEFAULT_O_MAX_PROBES,
> {
    /// Periodic wheel with full gear probing (NUM_SLOTS = 64)
    periodic: BitWheel<T, P_NUM_GEARS, P_RESOLUTION_MS, P_SLOT_CAP, NUM_SLOTS>,
    /// Oneshot wheel with limited probing
    oneshot: BitWheel<T, O_NUM_GEARS, O_RESOLUTION_MS, O_SLOT_CAP, O_MAX_PROBES>,
}

impl<
    T,
    const P_NUM_GEARS: usize,
    const P_RESOLUTION_MS: u64,
    const P_SLOT_CAP: usize,
    const O_NUM_GEARS: usize,
    const O_RESOLUTION_MS: u64,
    const O_SLOT_CAP: usize,
    const O_MAX_PROBES: usize,
> Default
    for DualBitWheel<
        T,
        P_NUM_GEARS,
        P_RESOLUTION_MS,
        P_SLOT_CAP,
        O_NUM_GEARS,
        O_RESOLUTION_MS,
        O_SLOT_CAP,
        O_MAX_PROBES,
    >
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    T,
    const P_NUM_GEARS: usize,
    const P_RESOLUTION_MS: u64,
    const P_SLOT_CAP: usize,
    const O_NUM_GEARS: usize,
    const O_RESOLUTION_MS: u64,
    const O_SLOT_CAP: usize,
    const O_MAX_PROBES: usize,
>
    DualBitWheel<
        T,
        P_NUM_GEARS,
        P_RESOLUTION_MS,
        P_SLOT_CAP,
        O_NUM_GEARS,
        O_RESOLUTION_MS,
        O_SLOT_CAP,
        O_MAX_PROBES,
    >
{
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            periodic: BitWheel::with_epoch(now),
            oneshot: BitWheel::with_epoch(now),
        }
    }

    pub fn with_epoch(epoch: Instant) -> Self {
        Self {
            periodic: BitWheel::with_epoch(epoch),
            oneshot: BitWheel::with_epoch(epoch),
        }
    }

    pub fn with_wheels(
        periodic: BitWheel<T, P_NUM_GEARS, P_RESOLUTION_MS, P_SLOT_CAP, NUM_SLOTS>,
        oneshot: BitWheel<T, O_NUM_GEARS, O_RESOLUTION_MS, O_SLOT_CAP, O_MAX_PROBES>,
    ) -> Self {
        Self { periodic, oneshot }
    }

    pub fn boxed() -> Box<Self> {
        Box::new(Self::new())
    }

    pub fn boxed_with_epoch(epoch: Instant) -> Box<Self> {
        Box::new(Self::with_epoch(epoch))
    }

    pub fn boxed_with_wheels(
        periodic: BitWheel<T, P_NUM_GEARS, P_RESOLUTION_MS, P_SLOT_CAP, NUM_SLOTS>,
        oneshot: BitWheel<T, O_NUM_GEARS, O_RESOLUTION_MS, O_SLOT_CAP, O_MAX_PROBES>,
    ) -> Box<Self> {
        Box::new(Self::with_wheels(periodic, oneshot))
    }

    /// Capacity info for the periodic wheel.
    pub const fn periodic_capacity() -> PeriodicCapacityInfo {
        PeriodicCapacityInfo {
            num_gears: P_NUM_GEARS,
            resolution_ms: P_RESOLUTION_MS,
            slot_cap: P_SLOT_CAP,
            max_timers: P_NUM_GEARS * NUM_SLOTS * P_SLOT_CAP,
            max_duration_ms: P_RESOLUTION_MS * (1 << (6 * P_NUM_GEARS)),
        }
    }

    /// Capacity info for the oneshot wheel.
    pub const fn oneshot_capacity() -> OneshotCapacityInfo {
        OneshotCapacityInfo {
            num_gears: O_NUM_GEARS,
            resolution_ms: O_RESOLUTION_MS,
            slot_cap: O_SLOT_CAP,
            max_probes: O_MAX_PROBES,
            max_duration_ms: O_RESOLUTION_MS * (1 << (6 * O_NUM_GEARS)),
        }
    }

    /// Insert a periodic timer (heartbeat, recurring check).
    ///
    /// Periodic timers use full gear probing, so insert only fails if the
    /// entire wheel is at capacity. This should never happen with proper sizing.
    pub fn insert_periodic(
        &mut self,
        when: Instant,
        timer: T,
    ) -> Result<TimerHandle, InsertError<T>> {
        self.periodic.insert(when, timer).map(|mut h| {
            h.gear |= PERIODIC_FLAG;
            h
        })
    }

    /// Insert a one-shot timer (order timeout, event deadline).
    ///
    /// One-shot timers use limited probing. Insert failure indicates
    /// backpressure — the system should throttle new work.
    pub fn insert_oneshot(
        &mut self,
        when: Instant,
        timer: T,
    ) -> Result<TimerHandle, InsertError<T>> {
        self.oneshot.insert(when, timer)
    }

    /// Cancel a timer by handle.
    ///
    /// Returns the timer if it was still pending, None if already fired.
    pub fn cancel(&mut self, mut handle: TimerHandle) -> Option<T> {
        if handle.gear & PERIODIC_FLAG != 0 {
            handle.gear &= GEAR_MASK;
            self.periodic.cancel(handle)
        } else {
            self.oneshot.cancel(handle)
        }
    }

    /// Poll both wheels, firing due timers.
    ///
    /// Periodic reschedule failures panic in debug (configuration bug).
    /// One-shot reschedule failures are counted and returned as PollError.
    pub fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> Result<usize, PollError>
    where
        T: Timer,
    {
        // Periodic wheel: reschedule failure is a configuration bug
        let periodic_fired = self.periodic.poll_with_failover(now, ctx, |_when, _timer| {
            debug_assert!(
                false,
                "periodic timer reschedule failed - increase P_SLOT_CAP or P_NUM_GEARS"
            );
        });

        // One-shot wheel: reschedule failure is backpressure
        let mut oneshot_lost = 0usize;
        let oneshot_fired = self.oneshot.poll_with_failover(now, ctx, |_when, _timer| {
            oneshot_lost += 1;
        });

        let total_fired = periodic_fired + oneshot_fired;

        if oneshot_lost > 0 {
            Err(PollError(oneshot_lost))
        } else {
            Ok(total_fired)
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.periodic.is_empty() && self.oneshot.is_empty()
    }

    #[inline(always)]
    pub fn periodic_is_empty(&self) -> bool {
        self.periodic.is_empty()
    }

    #[inline(always)]
    pub fn oneshot_is_empty(&self) -> bool {
        self.oneshot.is_empty()
    }

    #[inline(always)]
    pub fn duration_until_next(&self) -> Option<Duration> {
        match (
            self.periodic.duration_until_next(),
            self.oneshot.duration_until_next(),
        ) {
            (Some(p), Some(o)) => Some(p.min(o)),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        }
    }

    /// Approximate memory footprint in bytes.
    pub const fn memory_footprint() -> usize {
        BitWheel::<T, P_NUM_GEARS, P_RESOLUTION_MS, P_SLOT_CAP, NUM_SLOTS>::memory_footprint()
            + BitWheel::<T, O_NUM_GEARS, O_RESOLUTION_MS, O_SLOT_CAP, O_MAX_PROBES>::memory_footprint()
    }
}

/// Capacity information for periodic wheel configuration.
#[derive(Debug, Clone, Copy)]
pub struct PeriodicCapacityInfo {
    pub num_gears: usize,
    pub resolution_ms: u64,
    pub slot_cap: usize,
    pub max_timers: usize,
    pub max_duration_ms: u64,
}

/// Capacity information for oneshot wheel configuration.
#[derive(Debug, Clone, Copy)]
pub struct OneshotCapacityInfo {
    pub num_gears: usize,
    pub resolution_ms: u64,
    pub slot_cap: usize,
    pub max_probes: usize,
    pub max_duration_ms: u64,
}

// ============================================================================
// Type Aliases
// ============================================================================

/// Standard dual wheel for typical trading workloads.
/// - Periodic: 2 gears @ 100ms, 8 slots/slot → ~1K timers, ~7 min range
/// - Oneshot: 5 gears @ 5ms, 32 slots/slot → ~10K timers, ~23 hr range
pub type TradingDualWheel<T> = DualBitWheel<T>;

/// High-frequency trading dual wheel with tighter oneshot precision.
/// - Periodic: 2 gears @ 100ms (same as standard)
/// - Oneshot: 4 gears @ 1ms, 64 slots/slot → ~16K timers, ~4.7 hr range
pub type HftDualWheel<T> = DualBitWheel<T, 2, 100, 8, 4, 1, 64, 5>;

/// Burst-capable dual wheel for high timer density.
/// - Periodic: 2 gears @ 100ms, 16 slots/slot
/// - Oneshot: 5 gears @ 5ms, 64 slots/slot, 8 probes
pub type BurstDualWheel<T> = DualBitWheel<T, 2, 100, 16, 5, 5, 64, 8>;

/// Light dual wheel for memory-constrained environments.
/// - Periodic: 2 gears @ 100ms, 4 slots/slot
/// - Oneshot: 4 gears @ 5ms, 16 slots/slot
pub type LightDualWheel<T> = DualBitWheel<T, 2, 100, 4, 4, 5, 16, 3>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

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
        let wheel: Box<DualBitWheel<OneShotTimer>> = DualBitWheel::boxed();
        assert!(wheel.is_empty());
        assert!(wheel.periodic_is_empty());
        assert!(wheel.oneshot_is_empty());
        assert!(wheel.duration_until_next().is_none());
    }

    #[test]
    fn test_with_epoch() {
        let epoch = Instant::now();
        let wheel: Box<DualBitWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_default() {
        let wheel: DualBitWheel<OneShotTimer> = DualBitWheel::default();
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_capacity_info() {
        let p_info = TradingDualWheel::<OneShotTimer>::periodic_capacity();
        assert_eq!(p_info.num_gears, 2);
        assert_eq!(p_info.resolution_ms, 100);
        assert_eq!(p_info.slot_cap, 8);
        // 2 gears * 64 slots * 8 cap = 1024
        assert_eq!(p_info.max_timers, 1024);

        let o_info = TradingDualWheel::<OneShotTimer>::oneshot_capacity();
        assert_eq!(o_info.num_gears, 5);
        assert_eq!(o_info.resolution_ms, 5);
        assert_eq!(o_info.slot_cap, 32);
        assert_eq!(o_info.max_probes, 3);
    }

    // ==================== Insert Tests ====================

    #[test]
    fn test_insert_periodic() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _fired) = OneShotTimer::new(1);
        // Periodic at 30s — well within 2-gear @ 100ms range
        let handle = wheel
            .insert_periodic(epoch + Duration::from_secs(30), timer)
            .unwrap();

        assert!(!wheel.is_empty());
        assert!(!wheel.periodic_is_empty());
        assert!(wheel.oneshot_is_empty());
        assert!(handle.gear & PERIODIC_FLAG != 0);
    }

    #[test]
    fn test_insert_oneshot() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _fired) = OneShotTimer::new(1);
        // Oneshot at 100ms
        let handle = wheel
            .insert_oneshot(epoch + Duration::from_millis(100), timer)
            .unwrap();

        assert!(!wheel.is_empty());
        assert!(wheel.periodic_is_empty());
        assert!(!wheel.oneshot_is_empty());
        assert!(handle.gear & PERIODIC_FLAG == 0);
    }

    #[test]
    fn test_insert_both_different_timescales() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (t1, _) = OneShotTimer::new(1);
        let (t2, _) = OneShotTimer::new(2);

        // Periodic: 30s heartbeat
        wheel
            .insert_periodic(epoch + Duration::from_secs(30), t1)
            .unwrap();
        // Oneshot: 50ms order timeout
        wheel
            .insert_oneshot(epoch + Duration::from_millis(50), t2)
            .unwrap();

        assert!(!wheel.periodic_is_empty());
        assert!(!wheel.oneshot_is_empty());
    }

    // ==================== Cancel Tests ====================

    #[test]
    fn test_cancel_periodic() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(42);
        let handle = wheel
            .insert_periodic(epoch + Duration::from_secs(30), timer)
            .unwrap();

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 42);
        assert!(wheel.periodic_is_empty());
    }

    #[test]
    fn test_cancel_oneshot() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(42);
        let handle = wheel
            .insert_oneshot(epoch + Duration::from_millis(100), timer)
            .unwrap();

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 42);
        assert!(wheel.oneshot_is_empty());
    }

    #[test]
    fn test_cancel_after_poll_returns_none() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        let handle = wheel
            .insert_oneshot(epoch + Duration::from_millis(10), timer)
            .unwrap();

        let mut ctx = Vec::new();
        wheel
            .poll(epoch + Duration::from_millis(100), &mut ctx)
            .unwrap();

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_none());
    }

    // ==================== Poll Tests ====================

    #[test]
    fn test_poll_empty() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let mut ctx = Vec::new();
        let result = wheel.poll(epoch + Duration::from_millis(100), &mut ctx);

        assert_eq!(result.unwrap(), 0);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_poll_fires_periodic() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, fired) = OneShotTimer::new(1);
        // 500ms — within periodic resolution
        wheel
            .insert_periodic(epoch + Duration::from_millis(500), timer)
            .unwrap();

        let mut ctx = Vec::new();
        let result = wheel.poll(epoch + Duration::from_millis(600), &mut ctx);

        assert_eq!(result.unwrap(), 1);
        assert!(fired.get());
        assert_eq!(ctx, vec![1]);
    }

    #[test]
    fn test_poll_fires_oneshot() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, fired) = OneShotTimer::new(1);
        wheel
            .insert_oneshot(epoch + Duration::from_millis(10), timer)
            .unwrap();

        let mut ctx = Vec::new();
        let result = wheel.poll(epoch + Duration::from_millis(20), &mut ctx);

        assert_eq!(result.unwrap(), 1);
        assert!(fired.get());
        assert_eq!(ctx, vec![1]);
    }

    #[test]
    fn test_poll_fires_both_different_timescales() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (t1, f1) = OneShotTimer::new(1);
        let (t2, f2) = OneShotTimer::new(2);

        // Periodic fires at ~500ms (100ms resolution)
        wheel
            .insert_periodic(epoch + Duration::from_millis(500), t1)
            .unwrap();
        // Oneshot fires at ~50ms (5ms resolution)
        wheel
            .insert_oneshot(epoch + Duration::from_millis(50), t2)
            .unwrap();

        let mut ctx = Vec::new();

        // Poll at 100ms — only oneshot should fire
        wheel
            .poll(epoch + Duration::from_millis(100), &mut ctx)
            .unwrap();
        assert!(!f1.get());
        assert!(f2.get());
        assert_eq!(ctx, vec![2]);

        // Poll at 600ms — periodic should fire
        wheel
            .poll(epoch + Duration::from_millis(600), &mut ctx)
            .unwrap();
        assert!(f1.get());
        assert_eq!(ctx, vec![2, 1]);
    }

    #[test]
    fn test_poll_before_deadline() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (t1, f1) = OneShotTimer::new(1);
        let (t2, f2) = OneShotTimer::new(2);

        wheel
            .insert_periodic(epoch + Duration::from_secs(30), t1)
            .unwrap();
        wheel
            .insert_oneshot(epoch + Duration::from_millis(100), t2)
            .unwrap();

        let mut ctx = Vec::new();
        let result = wheel.poll(epoch + Duration::from_millis(50), &mut ctx);

        assert_eq!(result.unwrap(), 0);
        assert!(!f1.get());
        assert!(!f2.get());
    }

    // ==================== Periodic Reschedule Tests ====================

    #[test]
    fn test_periodic_reschedules() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<PeriodicTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        // 500ms period, fires 3 times
        let timer = PeriodicTimer::new(1, Duration::from_millis(500), 3);
        wheel
            .insert_periodic(epoch + Duration::from_millis(500), timer)
            .unwrap();

        let mut ctx = Vec::new();

        // Fire 1 at ~500ms
        wheel
            .poll(epoch + Duration::from_millis(600), &mut ctx)
            .unwrap();
        assert_eq!(ctx, vec![(1, 1)]);
        assert!(!wheel.periodic_is_empty());

        // Fire 2 at ~1000ms
        wheel
            .poll(epoch + Duration::from_millis(1100), &mut ctx)
            .unwrap();
        assert_eq!(ctx, vec![(1, 1), (1, 2)]);

        // Fire 3 at ~1500ms (last)
        wheel
            .poll(epoch + Duration::from_millis(1600), &mut ctx)
            .unwrap();
        assert_eq!(ctx, vec![(1, 1), (1, 2), (1, 3)]);
        assert!(wheel.periodic_is_empty());
    }

    // ==================== Duration Until Next Tests ====================

    #[test]
    fn test_duration_until_next_empty() {
        let wheel: DualBitWheel<OneShotTimer> = DualBitWheel::new();
        assert!(wheel.duration_until_next().is_none());
    }

    #[test]
    fn test_duration_until_next_periodic_only() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        wheel
            .insert_periodic(epoch + Duration::from_secs(30), timer)
            .unwrap();

        let duration = wheel.duration_until_next().unwrap();
        // Should be approximately 30s (within resolution)
        assert!(duration.as_secs() <= 30);
        assert!(duration.as_secs() >= 29);
    }

    #[test]
    fn test_duration_until_next_oneshot_only() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        wheel
            .insert_oneshot(epoch + Duration::from_millis(50), timer)
            .unwrap();

        let duration = wheel.duration_until_next().unwrap();
        assert!(duration.as_millis() <= 50);
    }

    #[test]
    fn test_duration_until_next_returns_min() {
        let epoch = Instant::now();
        let mut wheel: Box<TradingDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (t1, _) = OneShotTimer::new(1);
        let (t2, _) = OneShotTimer::new(2);

        // Periodic at 30s
        wheel
            .insert_periodic(epoch + Duration::from_secs(30), t1)
            .unwrap();
        // Oneshot at 50ms (much earlier)
        wheel
            .insert_oneshot(epoch + Duration::from_millis(50), t2)
            .unwrap();

        let duration = wheel.duration_until_next().unwrap();
        // Should return oneshot's duration (the earlier one)
        assert!(duration.as_millis() <= 50);
    }

    // ==================== Full Gear Probe Test ====================

    #[test]
    fn test_periodic_full_gear_probe() {
        let epoch = Instant::now();
        // Small slot cap but full probing
        let mut wheel: Box<DualBitWheel<OneShotTimer, 2, 100, 2, 5, 5, 32, 3>> =
            DualBitWheel::boxed_with_epoch(epoch);

        // Insert many periodic timers at same time — should all succeed
        // because we probe entire gear (64 slots)
        let when = epoch + Duration::from_secs(30);
        for i in 0..100 {
            let (timer, _) = OneShotTimer::new(i);
            wheel.insert_periodic(when, timer).unwrap();
        }

        assert!(!wheel.periodic_is_empty());
    }

    // ==================== Stress Tests ====================

    #[test]
    fn test_many_timers() {
        let epoch = Instant::now();
        let mut wheel: Box<BurstDualWheel<CounterTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        // 50 periodic timers at various intervals
        for i in 0..50 {
            let when = epoch + Duration::from_millis(500 + (i % 10) * 100);
            wheel.insert_periodic(when, CounterTimer).unwrap();
        }

        // 500 oneshot timers
        for i in 0..500 {
            let when = epoch + Duration::from_millis((i % 100) + 10);
            wheel.insert_oneshot(when, CounterTimer).unwrap();
        }

        let mut count = 0usize;
        wheel
            .poll(epoch + Duration::from_secs(2), &mut count)
            .unwrap();

        assert_eq!(count, 550);
        assert!(wheel.is_empty());
    }

    // ==================== Type Alias Tests ====================

    #[test]
    fn test_hft_dual_wheel() {
        let epoch = Instant::now();
        let mut wheel: Box<HftDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        let (timer, fired) = OneShotTimer::new(1);
        // 1ms resolution oneshot
        wheel
            .insert_oneshot(epoch + Duration::from_millis(5), timer)
            .unwrap();

        let mut ctx = Vec::new();
        wheel
            .poll(epoch + Duration::from_millis(10), &mut ctx)
            .unwrap();

        assert!(fired.get());
    }

    #[test]
    fn test_light_dual_wheel() {
        let epoch = Instant::now();
        let wheel: Box<LightDualWheel<OneShotTimer>> = DualBitWheel::boxed_with_epoch(epoch);

        // Just verify it constructs
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
            let mut wheel: Box<TradingDualWheel<DropCounter>> =
                DualBitWheel::boxed_with_epoch(epoch);

            // 5 periodic
            for _ in 0..5 {
                wheel
                    .insert_periodic(
                        epoch + Duration::from_secs(30),
                        DropCounter(Arc::clone(&drop_count)),
                    )
                    .unwrap();
            }

            // 5 oneshot
            for _ in 0..5 {
                wheel
                    .insert_oneshot(
                        epoch + Duration::from_millis(100),
                        DropCounter(Arc::clone(&drop_count)),
                    )
                    .unwrap();
            }

            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        }

        assert_eq!(drop_count.load(Ordering::SeqCst), 10);
    }
}
