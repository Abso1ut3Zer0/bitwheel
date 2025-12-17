use std::time::{Duration, Instant};

use crate::timer::{
    BitWheel, PollError, Timer, TimerHandle,
    gear::{InsertError, NUM_SLOTS},
};

// Periodic defaults: small, coarse, probe everything
pub const DEFAULT_P_NUM_GEARS: usize = 2;
pub const DEFAULT_P_RESOLUTION_MS: u64 = 100;
pub const DEFAULT_P_SLOT_CAP: usize = 8;
pub const PERIODIC_PROBES: usize = 63;

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
    /// Periodic wheel with full gear probing (PERIODIC_PROBES)
    periodic: BitWheel<T, P_NUM_GEARS, P_RESOLUTION_MS, P_SLOT_CAP, PERIODIC_PROBES>,
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
        periodic: BitWheel<T, P_NUM_GEARS, P_RESOLUTION_MS, P_SLOT_CAP, PERIODIC_PROBES>,
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
        periodic: BitWheel<T, P_NUM_GEARS, P_RESOLUTION_MS, P_SLOT_CAP, PERIODIC_PROBES>,
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

#[cfg(test)]
mod tests {
    use crate::timer::{
        FastPreciseFastDualWheel, StandardBalancedDualWheel, StandardBalancedLightDualWheel,
        StandardBurstDualWheel,
    };

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
        let p_info = StandardBalancedDualWheel::<OneShotTimer>::periodic_capacity();
        assert_eq!(p_info.num_gears, 2);
        assert_eq!(p_info.resolution_ms, 100);
        assert_eq!(p_info.slot_cap, 8);
        // 2 gears * 64 slots * 8 cap = 1024
        assert_eq!(p_info.max_timers, 1024);

        let o_info = StandardBalancedDualWheel::<OneShotTimer>::oneshot_capacity();
        assert_eq!(o_info.num_gears, 4);
        assert_eq!(o_info.resolution_ms, 5);
        assert_eq!(o_info.slot_cap, 16);
        assert_eq!(o_info.max_probes, 8);
    }

    // ==================== Insert Tests ====================

    #[test]
    fn test_insert_periodic() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(42);
        let handle = wheel
            .insert_periodic(epoch + Duration::from_secs(30), timer)
            .unwrap();

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 42);
        // Note: is_empty() may return false until next poll() due to stale next_fire_tick
        // This is by design - stale cache only causes early poll, never missed timers
    }

    #[test]
    fn test_cancel_oneshot() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(42);
        let handle = wheel
            .insert_oneshot(epoch + Duration::from_millis(100), timer)
            .unwrap();

        let cancelled = wheel.cancel(handle);
        assert!(cancelled.is_some());
        assert_eq!(cancelled.unwrap().id, 42);
        // Note: is_empty() may return false until next poll() due to stale next_fire_tick
    }

    #[test]
    fn test_cancel_after_poll_returns_none() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut ctx = Vec::new();
        let result = wheel.poll(epoch + Duration::from_millis(100), &mut ctx);

        assert_eq!(result.unwrap(), 0);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_poll_fires_periodic() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<PeriodicTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let (timer, _) = OneShotTimer::new(1);
        wheel
            .insert_periodic(epoch + Duration::from_secs(30), timer)
            .unwrap();

        let duration = wheel.duration_until_next().unwrap();
        // With 100ms resolution, allow for quantization error
        // 30s = 300 ticks at 100ms, but slot assignment can shift timing
        assert!(duration.as_secs() <= 31);
        assert!(duration.as_secs() >= 23); // Allow ~7s variance from gear quantization
    }

    #[test]
    fn test_duration_until_next_oneshot_only() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<StandardBalancedDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
        let mut wheel: Box<DualBitWheel<OneShotTimer, 2, 100, 2, 4, 5, 32, 3>> =
            DualBitWheel::boxed_with_epoch(epoch);

        // Insert many periodic timers at same time — should all succeed
        // because we probe entire gear (63 slots)
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
        let mut wheel: Box<StandardBurstDualWheel<CounterTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
    fn test_fast_precise_dual_wheel() {
        let epoch = Instant::now();
        let mut wheel: Box<FastPreciseFastDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
    fn test_standard_balanced_light_dual_wheel() {
        let epoch = Instant::now();
        let wheel: Box<StandardBalancedLightDualWheel<OneShotTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

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
            let mut wheel: Box<StandardBalancedDualWheel<DropCounter>> =
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

#[cfg(test)]
mod latency_tests {
    use crate::timer::{DualBitWheel, StandardBalancedDualWheel, StandardBurstDualWheel, Timer};

    use hdrhistogram::Histogram;
    use std::time::{Duration, Instant};

    const WARMUP: u64 = 100_000;
    const ITERATIONS: u64 = 1_000_000;

    struct LatencyTimer;

    impl Timer for LatencyTimer {
        type Context = ();

        fn fire(&mut self, _now: Instant, _ctx: &mut ()) -> Option<Instant> {
            None
        }
    }

    struct PeriodicLatencyTimer {
        period: Duration,
        remaining: usize,
    }

    impl Timer for PeriodicLatencyTimer {
        type Context = ();

        fn fire(&mut self, now: Instant, _ctx: &mut ()) -> Option<Instant> {
            self.remaining = self.remaining.saturating_sub(1);
            if self.remaining > 0 {
                Some(now + self.period)
            } else {
                None
            }
        }
    }

    enum MixedLatencyTimer {
        OneShot,
        Periodic { period: Duration, remaining: usize },
    }

    impl Timer for MixedLatencyTimer {
        type Context = ();

        fn fire(&mut self, now: Instant, _ctx: &mut ()) -> Option<Instant> {
            match self {
                MixedLatencyTimer::OneShot => None,
                MixedLatencyTimer::Periodic { period, remaining } => {
                    *remaining = remaining.saturating_sub(1);
                    if *remaining > 0 {
                        Some(now + *period)
                    } else {
                        None
                    }
                }
            }
        }
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

    // ==================== Insert Latency ====================

    #[test]
    #[ignore]
    fn hdr_insert_oneshot_latency_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert_oneshot(when, LatencyTimer).unwrap();
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 500) + 10);

            let start = Instant::now();
            let handle = wheel.insert_oneshot(when, LatencyTimer).unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
            wheel.cancel(handle);
        }

        print_histogram("Dual - Insert Oneshot Latency", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_insert_periodic_latency_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 5000) + 500);
            let handle = wheel.insert_periodic(when, LatencyTimer).unwrap();
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 5000) + 500);

            let start = Instant::now();
            let handle = wheel.insert_periodic(when, LatencyTimer).unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
            wheel.cancel(handle);
        }

        print_histogram("Dual - Insert Periodic Latency", &hist);
    }

    // ==================== Cancel Latency ====================

    #[test]
    #[ignore]
    fn hdr_cancel_oneshot_latency_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert_oneshot(when, LatencyTimer).unwrap();
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert_oneshot(when, LatencyTimer).unwrap();

            let start = Instant::now();
            let _ = wheel.cancel(handle);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Cancel Oneshot Latency", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_cancel_periodic_latency_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 5000) + 500);
            let handle = wheel.insert_periodic(when, LatencyTimer).unwrap();
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 5000) + 500);
            let handle = wheel.insert_periodic(when, LatencyTimer).unwrap();

            let start = Instant::now();
            let _ = wheel.cancel(handle);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Cancel Periodic Latency", &hist);
    }

    // ==================== Poll Latency ====================

    #[test]
    #[ignore]
    fn hdr_poll_empty_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i);
            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i);

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Poll Empty", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_pending_no_fires_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        // Insert timers far in future - both wheels
        for i in 0..500 {
            let when = epoch + Duration::from_millis(100_000_000 + i);
            let _ = wheel.insert_oneshot(when, LatencyTimer);
        }
        for i in 0..50 {
            let when = epoch + Duration::from_millis(100_000_000 + i * 1000);
            let _ = wheel.insert_periodic(when, LatencyTimer);
        }

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i);
            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i);

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Poll Pending (No Fires)", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_single_oneshot_fire_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis(i + 1);
            let _ = wheel.insert_oneshot(when, LatencyTimer);
            let now = epoch + Duration::from_millis(i + 1);
            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in 0..ITERATIONS {
            let tick = WARMUP + i;
            let when = epoch + Duration::from_millis(tick + 1);
            let _ = wheel.insert_oneshot(when, LatencyTimer);

            let now = epoch + Duration::from_millis(tick + 1);

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Poll Single Oneshot Fire", &hist);
    }

    // ==================== Periodic Steady State ====================

    #[test]
    #[ignore]
    fn hdr_periodic_steady_state_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<PeriodicLatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // 10 periodic timers in periodic wheel, 100ms period (matches periodic resolution)
        for i in 0..10 {
            let when = epoch + Duration::from_millis(100 + i * 10);
            let timer = PeriodicLatencyTimer {
                period: Duration::from_millis(100),
                remaining: usize::MAX,
            };
            let _ = wheel.insert_periodic(when, timer);
        }

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i + 1);
            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i + 1);

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Periodic Steady State (10 timers @ 100ms)", &hist);
    }

    // ==================== Mixed Workload ====================

    #[test]
    #[ignore]
    fn hdr_mixed_periodic_oneshot_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<MixedLatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // 10 periodic heartbeats in periodic wheel, 500ms period
        for i in 0..10 {
            let when = epoch + Duration::from_millis(500 + i * 50);
            let timer = MixedLatencyTimer::Periodic {
                period: Duration::from_millis(500),
                remaining: usize::MAX,
            };
            let _ = wheel.insert_periodic(when, timer);
        }

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i + 1);

            // Insert 2 one-shot timers per tick (order timeouts) in oneshot wheel
            for j in 0..2 {
                let when = now + Duration::from_millis(50 + (i + j) % 50);
                let _ = wheel.insert_oneshot(when, MixedLatencyTimer::OneShot);
            }

            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i + 1);

            // Insert 2 one-shot timers per tick
            for j in 0..2 {
                let when = now + Duration::from_millis(50 + (i + j) % 50);
                let _ = wheel.insert_oneshot(when, MixedLatencyTimer::OneShot);
            }

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Mixed (10 periodic + 2 oneshot/tick)", &hist);
    }

    // ==================== Bursty Workload ====================

    #[test]
    #[ignore]
    fn hdr_bursty_workload_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBurstDualWheel<MixedLatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // 10 periodic heartbeats, 500ms period
        for i in 0..10 {
            let when = epoch + Duration::from_millis(500 + i * 50);
            let timer = MixedLatencyTimer::Periodic {
                period: Duration::from_millis(500),
                remaining: usize::MAX,
            };
            let _ = wheel.insert_periodic(when, timer);
        }

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i + 1);

            // Burst every 100 ticks
            if i % 100 == 0 {
                for j in 0..50 {
                    let when = now + Duration::from_millis(20 + j % 80);
                    let _ = wheel.insert_oneshot(when, MixedLatencyTimer::OneShot);
                }
            }

            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i + 1);

            // Burst every 100 ticks
            if i % 100 == 0 {
                for j in 0..50 {
                    let when = now + Duration::from_millis(20 + j % 80);
                    let _ = wheel.insert_oneshot(when, MixedLatencyTimer::OneShot);
                }
            }

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Bursty (10 periodic + 50 burst every 100ms)", &hist);
    }

    // ==================== Trading Simulation ====================

    #[test]
    #[ignore]
    fn hdr_trading_simulation_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut insert_hist = Histogram::<u64>::new(3).unwrap();
        let mut poll_hist = Histogram::<u64>::new(3).unwrap();
        let mut cancel_hist = Histogram::<u64>::new(3).unwrap();

        let mut handles = Vec::with_capacity(100);
        let mut ctx = ();
        let mut now = epoch;

        // Warmup
        for i in 0..WARMUP {
            now += Duration::from_millis(1);
            let _ = wheel.poll(now, &mut ctx);

            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));
                if let Ok(handle) = wheel.insert_oneshot(when, LatencyTimer) {
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
            let _ = wheel.poll(now, &mut ctx);
            poll_hist.record(start.elapsed().as_nanos() as u64).unwrap();

            // Insert 80%
            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));

                let start = Instant::now();
                if let Ok(handle) = wheel.insert_oneshot(when, LatencyTimer) {
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

        print_histogram("Dual Trading Sim - Insert", &insert_hist);
        print_histogram("Dual Trading Sim - Poll", &poll_hist);
        print_histogram("Dual Trading Sim - Cancel", &cancel_hist);
    }

    // ==================== Realistic Trading ====================

    #[test]
    #[ignore]
    fn hdr_realistic_trading_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<MixedLatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut insert_hist = Histogram::<u64>::new(3).unwrap();
        let mut poll_hist = Histogram::<u64>::new(3).unwrap();
        let mut cancel_hist = Histogram::<u64>::new(3).unwrap();

        let mut handles = Vec::with_capacity(100);
        let mut ctx = ();
        let mut now = epoch;

        // Background: 5 venue heartbeats @ 30s period in PERIODIC wheel
        for i in 0..5 {
            let when = epoch + Duration::from_secs(30) + Duration::from_millis(i * 100);
            let timer = MixedLatencyTimer::Periodic {
                period: Duration::from_secs(30),
                remaining: usize::MAX,
            };
            let _ = wheel.insert_periodic(when, timer);
        }

        // Warmup
        for i in 0..WARMUP {
            now += Duration::from_millis(1);
            let _ = wheel.poll(now, &mut ctx);

            // 80% of ticks: new order timeout (50-250ms) in ONESHOT wheel
            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));
                if let Ok(handle) = wheel.insert_oneshot(when, MixedLatencyTimer::OneShot) {
                    if handles.len() < 100 {
                        handles.push(handle);
                    }
                }
            }

            // 5% of ticks: order filled, cancel timeout
            if i % 20 == 0 {
                if let Some(handle) = handles.pop() {
                    let _ = wheel.cancel(handle);
                }
            }
        }

        // Measure
        for i in 0..ITERATIONS {
            now += Duration::from_millis(1);

            // Poll (always)
            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            poll_hist.record(start.elapsed().as_nanos() as u64).unwrap();

            // Insert order timeout 80%
            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));

                let start = Instant::now();
                if let Ok(handle) = wheel.insert_oneshot(when, MixedLatencyTimer::OneShot) {
                    insert_hist
                        .record(start.elapsed().as_nanos() as u64)
                        .unwrap();

                    if handles.len() < 100 {
                        handles.push(handle);
                    }
                }
            }

            // Cancel 5% (order filled)
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

        print_histogram(
            "Dual Realistic Trading - Insert (order timeout)",
            &insert_hist,
        );
        print_histogram("Dual Realistic Trading - Poll", &poll_hist);
        print_histogram("Dual Realistic Trading - Cancel (order fill)", &cancel_hist);
    }

    // ==================== Interleaved Insert ====================

    #[test]
    #[ignore]
    fn hdr_interleaved_insert_dual() {
        let epoch = Instant::now();
        let mut wheel: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let mut now = epoch;

        // Pre-populate oneshot wheel: timers at regular intervals (every 10ms from 100-10000ms)
        for i in 0..10_000 {
            let when = epoch + Duration::from_millis(100 + i * 10);
            let _ = wheel.insert_oneshot(when, LatencyTimer);
        }

        // Warmup - insert timers that land BETWEEN existing entries
        for i in 0..WARMUP {
            now += Duration::from_millis(1);
            let _ = wheel.poll(now, &mut ctx);

            let base = 100 + ((i % 990) * 10);
            let when = epoch + Duration::from_millis(base + 5);
            let _ = wheel.insert_oneshot(when, LatencyTimer);
        }

        // Measure - every insert lands between two existing timers
        for i in 0..ITERATIONS {
            now += Duration::from_millis(1);
            let _ = wheel.poll(now, &mut ctx);

            // Replenish the "grid" timers as they fire
            if i % 10 == 0 {
                let when = now + Duration::from_millis(10000);
                let _ = wheel.insert_oneshot(when, LatencyTimer);
            }

            // The measured insert: always between existing entries
            let base = ((now.duration_since(epoch).as_millis() as u64) % 9900) + 100;
            let offset = (i % 3) * 2 + 3; // 3, 5, or 7
            let when = epoch + Duration::from_millis(base + offset);

            let start = Instant::now();
            let _ = wheel.insert_oneshot(when, LatencyTimer);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Dual - Interleaved Insert", &hist);
    }

    // ==================== Comparison: Single vs Dual Poll ====================

    /// Direct comparison: poll overhead when both wheels have timers but none fire
    #[test]
    #[ignore]
    fn hdr_dual_vs_single_poll_overhead() {
        let epoch = Instant::now();

        // Setup dual wheel
        let mut dual: Box<StandardBalancedDualWheel<LatencyTimer>> =
            DualBitWheel::boxed_with_epoch(epoch);

        // Setup single wheel (using the oneshot config which is BalancedWheel)
        let mut single: Box<crate::timer::BalancedWheel<LatencyTimer>> =
            crate::timer::BitWheel::boxed_with_epoch(epoch);

        // Add background timers to both - far in future so they don't fire
        for i in 0..100 {
            let when = epoch + Duration::from_millis(100_000_000 + i);
            let _ = dual.insert_oneshot(when, LatencyTimer);
            let _ = single.insert(when, LatencyTimer);
        }
        for i in 0..10 {
            let when = epoch + Duration::from_millis(100_000_000 + i * 10000);
            let _ = dual.insert_periodic(when, LatencyTimer);
        }

        let mut dual_hist = Histogram::<u64>::new(3).unwrap();
        let mut single_hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup both
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i);
            let _ = dual.poll(now, &mut ctx);
            let _ = single.poll(now, &mut ctx);
        }

        // Measure dual
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i);

            let start = Instant::now();
            let _ = dual.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            dual_hist.record(elapsed).unwrap();
        }

        // Measure single
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i);

            let start = Instant::now();
            let _ = single.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            single_hist.record(elapsed).unwrap();
        }

        print_histogram("COMPARISON - Single BitWheel Poll", &single_hist);
        print_histogram("COMPARISON - Dual BitWheel Poll", &dual_hist);
    }

    /// Compare poll when timers ARE firing
    #[test]
    #[ignore]
    fn hdr_dual_vs_single_poll_with_fires() {
        let epoch = Instant::now();

        let mut dual_hist = Histogram::<u64>::new(3).unwrap();
        let mut single_hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Test dual wheel
        {
            let mut dual: Box<StandardBalancedDualWheel<LatencyTimer>> =
                DualBitWheel::boxed_with_epoch(epoch);

            // Warmup
            for i in 0..WARMUP {
                let when = epoch + Duration::from_millis(i + 1);
                let _ = dual.insert_oneshot(when, LatencyTimer);
                let now = epoch + Duration::from_millis(i + 1);
                let _ = dual.poll(now, &mut ctx);
            }

            // Measure
            for i in 0..ITERATIONS {
                let tick = WARMUP + i;
                let when = epoch + Duration::from_millis(tick + 1);
                let _ = dual.insert_oneshot(when, LatencyTimer);

                let now = epoch + Duration::from_millis(tick + 1);

                let start = Instant::now();
                let _ = dual.poll(now, &mut ctx);
                let elapsed = start.elapsed().as_nanos() as u64;

                dual_hist.record(elapsed).unwrap();
            }
        }

        // Test single wheel
        {
            let mut single: Box<crate::timer::BalancedWheel<LatencyTimer>> =
                crate::timer::BitWheel::boxed_with_epoch(epoch);

            // Warmup
            for i in 0..WARMUP {
                let when = epoch + Duration::from_millis(i + 1);
                let _ = single.insert(when, LatencyTimer);
                let now = epoch + Duration::from_millis(i + 1);
                let _ = single.poll(now, &mut ctx);
            }

            // Measure
            for i in 0..ITERATIONS {
                let tick = WARMUP + i;
                let when = epoch + Duration::from_millis(tick + 1);
                let _ = single.insert(when, LatencyTimer);

                let now = epoch + Duration::from_millis(tick + 1);

                let start = Instant::now();
                let _ = single.poll(now, &mut ctx);
                let elapsed = start.elapsed().as_nanos() as u64;

                single_hist.record(elapsed).unwrap();
            }
        }

        print_histogram(
            "COMPARISON - Single BitWheel Poll (with fires)",
            &single_hist,
        );
        print_histogram("COMPARISON - Dual BitWheel Poll (with fires)", &dual_hist);
    }
}
