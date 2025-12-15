//! Latency comparison benchmarks against other timer implementations.

#[cfg(test)]
mod btreemap {
    use hdrhistogram::Histogram;
    use std::collections::BTreeMap;
    use std::time::{Duration, Instant};

    const WARMUP: u64 = 100_000;
    const ITERATIONS: u64 = 1_000_000;

    // ============================================================
    // BTreeMap Timer Wheel Implementation
    // ============================================================

    trait Timer {
        type Context;
        fn fire(&mut self, now: Instant, ctx: &mut Self::Context) -> Option<Instant>;
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TimerHandle {
        when_tick: u64,
        sequence: u32,
    }

    struct BTreeTimerWheel<T> {
        tree: BTreeMap<(u64, u32), T>,
        epoch: Instant,
        sequence: u32,
    }

    impl<T> BTreeTimerWheel<T> {
        fn with_epoch(epoch: Instant) -> Self {
            Self {
                tree: BTreeMap::new(),
                epoch,
                sequence: 0,
            }
        }

        fn boxed_with_epoch(epoch: Instant) -> Box<Self> {
            Box::new(Self::with_epoch(epoch))
        }

        fn insert(&mut self, when: Instant, timer: T) -> TimerHandle {
            let when_tick = when.saturating_duration_since(self.epoch).as_millis() as u64;
            let sequence = self.sequence;
            self.sequence = self.sequence.wrapping_add(1);
            self.tree.insert((when_tick, sequence), timer);
            TimerHandle {
                when_tick,
                sequence,
            }
        }

        fn cancel(&mut self, handle: TimerHandle) -> Option<T> {
            self.tree.remove(&(handle.when_tick, handle.sequence))
        }

        fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize
        where
            T: Timer,
        {
            let current_tick = now.saturating_duration_since(self.epoch).as_millis() as u64;
            let mut fired = 0;

            loop {
                let should_fire = self
                    .tree
                    .first_key_value()
                    .map(|(&(tick, _), _)| tick <= current_tick)
                    .unwrap_or(false);

                if !should_fire {
                    break;
                }

                let entry = self.tree.first_entry().unwrap();
                let mut timer = entry.remove();
                fired += 1;

                if let Some(next_when) = timer.fire(now, ctx) {
                    self.insert(next_when, timer);
                }
            }

            fired
        }
    }

    // ============================================================
    // Test Timer Implementations
    // ============================================================

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

    // ============================================================
    // Latency Tests
    // ============================================================

    #[test]
    #[ignore]
    fn hdr_insert_latency_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<LatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert(when, LatencyTimer);
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 500) + 10);

            let start = Instant::now();
            let handle = wheel.insert(when, LatencyTimer);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
            wheel.cancel(handle);
        }

        print_histogram("Insert Latency", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_cancel_latency_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<LatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert(when, LatencyTimer);
            wheel.cancel(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let when = epoch + Duration::from_millis((i % 500) + 10);
            let handle = wheel.insert(when, LatencyTimer);

            let start = Instant::now();
            let _ = wheel.cancel(handle);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Cancel Latency", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_empty_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<LatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

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

        print_histogram("Poll Empty", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_pending_no_fires_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<LatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

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

        print_histogram("Poll Pending (No Fires)", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_poll_single_fire_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<LatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // Warmup
        for i in 0..WARMUP {
            let when = epoch + Duration::from_millis(i + 1);
            let _ = wheel.insert(when, LatencyTimer);
            let now = epoch + Duration::from_millis(i + 1);
            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in 0..ITERATIONS {
            let tick = WARMUP + i;
            let when = epoch + Duration::from_millis(tick + 1);
            let _ = wheel.insert(when, LatencyTimer);

            let now = epoch + Duration::from_millis(tick + 1);

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Poll Single Fire", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_periodic_steady_state_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<PeriodicLatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // 10 periodic timers, 1ms period
        for i in 0..10 {
            let when = epoch + Duration::from_millis(i + 1);
            let timer = PeriodicLatencyTimer {
                period: Duration::from_millis(1),
                remaining: usize::MAX,
            };
            let _ = wheel.insert(when, timer);
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

        print_histogram("Periodic Steady State (10 timers @ 1ms)", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_mixed_periodic_oneshot_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<MixedLatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // 10 periodic heartbeats, 10ms period
        for i in 0..10 {
            let when = epoch + Duration::from_millis(i + 10);
            let timer = MixedLatencyTimer::Periodic {
                period: Duration::from_millis(10),
                remaining: usize::MAX,
            };
            let _ = wheel.insert(when, timer);
        }

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i + 1);

            for j in 0..2 {
                let when = now + Duration::from_millis(50 + (i + j) % 50);
                let _ = wheel.insert(when, MixedLatencyTimer::OneShot);
            }

            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i + 1);

            for j in 0..2 {
                let when = now + Duration::from_millis(50 + (i + j) % 50);
                let _ = wheel.insert(when, MixedLatencyTimer::OneShot);
            }

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Mixed (10 periodic + 2 oneshot/tick)", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_bursty_workload_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<MixedLatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();

        // 10 periodic heartbeats, 10ms period
        for i in 0..10 {
            let when = epoch + Duration::from_millis(i + 10);
            let timer = MixedLatencyTimer::Periodic {
                period: Duration::from_millis(10),
                remaining: usize::MAX,
            };
            let _ = wheel.insert(when, timer);
        }

        // Warmup
        for i in 0..WARMUP {
            let now = epoch + Duration::from_millis(i + 1);

            if i % 100 == 0 {
                for j in 0..50 {
                    let when = now + Duration::from_millis(20 + j % 80);
                    let _ = wheel.insert(when, MixedLatencyTimer::OneShot);
                }
            }

            let _ = wheel.poll(now, &mut ctx);
        }

        // Measure
        for i in WARMUP..(WARMUP + ITERATIONS) {
            let now = epoch + Duration::from_millis(i + 1);

            if i % 100 == 0 {
                for j in 0..50 {
                    let when = now + Duration::from_millis(20 + j % 80);
                    let _ = wheel.insert(when, MixedLatencyTimer::OneShot);
                }
            }

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Bursty (10 periodic + 50 burst every 100ms)", &hist);
    }

    #[test]
    #[ignore]
    fn hdr_trading_simulation_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<LatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

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
                let handle = wheel.insert(when, LatencyTimer);
                if handles.len() < 100 {
                    handles.push(handle);
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

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            poll_hist.record(start.elapsed().as_nanos() as u64).unwrap();

            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));

                let start = Instant::now();
                let handle = wheel.insert(when, LatencyTimer);
                insert_hist
                    .record(start.elapsed().as_nanos() as u64)
                    .unwrap();

                if handles.len() < 100 {
                    handles.push(handle);
                }
            }

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
    fn hdr_realistic_trading_btreemap() {
        let epoch = Instant::now();
        let mut wheel: Box<BTreeTimerWheel<MixedLatencyTimer>> =
            BTreeTimerWheel::boxed_with_epoch(epoch);

        let mut insert_hist = Histogram::<u64>::new(3).unwrap();
        let mut poll_hist = Histogram::<u64>::new(3).unwrap();
        let mut cancel_hist = Histogram::<u64>::new(3).unwrap();

        let mut handles = Vec::with_capacity(100);
        let mut ctx = ();
        let mut now = epoch;

        // Background: 5 venue heartbeats @ 30s period
        for i in 0..5 {
            let when = epoch + Duration::from_secs(30) + Duration::from_millis(i * 100);
            let timer = MixedLatencyTimer::Periodic {
                period: Duration::from_secs(30),
                remaining: usize::MAX,
            };
            let _ = wheel.insert(when, timer);
        }

        // Warmup
        for i in 0..WARMUP {
            now += Duration::from_millis(1);
            let _ = wheel.poll(now, &mut ctx);

            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));
                let handle = wheel.insert(when, MixedLatencyTimer::OneShot);
                if handles.len() < 100 {
                    handles.push(handle);
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

            let start = Instant::now();
            let _ = wheel.poll(now, &mut ctx);
            poll_hist.record(start.elapsed().as_nanos() as u64).unwrap();

            if i % 5 != 0 {
                let when = now + Duration::from_millis(50 + (i % 200));

                let start = Instant::now();
                let handle = wheel.insert(when, MixedLatencyTimer::OneShot);
                insert_hist
                    .record(start.elapsed().as_nanos() as u64)
                    .unwrap();

                if handles.len() < 100 {
                    handles.push(handle);
                }
            }

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

        print_histogram("Realistic Trading - Insert (order timeout)", &insert_hist);
        print_histogram("Realistic Trading - Poll", &poll_hist);
        print_histogram("Realistic Trading - Cancel (order fill)", &cancel_hist);
    }
}
