use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::{Duration, Instant};

use bitwheel::{BitWheel, Timer};

// ==================== Benchmark Timer Types ====================

struct BenchOneShotTimer;

impl Timer for BenchOneShotTimer {
    type Context = ();

    fn fire(&mut self, _now: Instant, _ctx: &mut ()) -> Option<Instant> {
        None
    }
}

struct BenchPeriodicTimer {
    period: Duration,
    remaining: usize,
}

impl BenchPeriodicTimer {
    fn new(period: Duration, fires: usize) -> Self {
        Self {
            period,
            remaining: fires,
        }
    }
}

impl Timer for BenchPeriodicTimer {
    type Context = usize;

    fn fire(&mut self, now: Instant, ctx: &mut usize) -> Option<Instant> {
        *ctx += 1;
        self.remaining -= 1;
        if self.remaining > 0 {
            Some(now + self.period)
        } else {
            None
        }
    }
}

// ==================== Helpers ====================

type TestWheel = BitWheel<BenchOneShotTimer, 4, 1, 64, 8>;
type PeriodicWheel = BitWheel<BenchPeriodicTimer, 4, 1, 64, 8>;

fn bench_wheel() -> Box<TestWheel> {
    TestWheel::boxed()
}

fn bench_wheel_periodic() -> Box<PeriodicWheel> {
    PeriodicWheel::boxed()
}

// ==================== Insert Benchmarks ====================

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    group.bench_function("single", |b| {
        let epoch = Instant::now();
        let mut wheel = bench_wheel();
        let when = epoch + Duration::from_millis(100);

        b.iter(|| {
            let handle = wheel.insert(when, BenchOneShotTimer).unwrap();
            wheel.cancel(handle);
            black_box(())
        });
    });

    group.bench_function("short_delay_burst", |b| {
        let epoch = Instant::now();

        b.iter_custom(|iters| {
            let mut wheel = bench_wheel();
            let start = Instant::now();

            for i in 0..iters {
                let delay = 100 + (i % 400);
                let when = epoch + Duration::from_millis(delay);
                let _ = wheel.insert(when, BenchOneShotTimer); // ignore error
            }

            start.elapsed()
        });
    });

    for pct_short in [50, 70, 90] {
        group.bench_with_input(
            BenchmarkId::new("mixed_delays", format!("{}pct_short", pct_short)),
            &pct_short,
            |b, &pct_short| {
                let epoch = Instant::now();

                b.iter_custom(|iters| {
                    let mut wheel = bench_wheel();
                    let start = Instant::now();

                    for i in 0..iters {
                        let delay = if (i % 100) < pct_short as u64 {
                            10 + (i % 90) // short: 10-100ms
                        } else if (i % 100) < 95 {
                            100 + (i % 900) // medium: 100ms-1s
                        } else {
                            1000 + (i % 9000) // long: 1-10s
                        };
                        let when = epoch + Duration::from_millis(delay);
                        let _ = black_box(wheel.insert(when, BenchOneShotTimer));
                    }

                    start.elapsed()
                });
            },
        );
    }

    group.finish();
}

// ==================== Cancel Benchmarks ====================

fn bench_cancel(c: &mut Criterion) {
    c.bench_function("cancel", |b| {
        let epoch = Instant::now();
        // Larger slot cap to avoid collisions
        let mut wheel: Box<BitWheel<BenchOneShotTimer, 4, 1, 256, 16>> = BitWheel::boxed();

        // Pre-fill with spread-out timers
        let mut handles: Vec<_> = (0..5000)
            .filter_map(|i| {
                let when = epoch + Duration::from_millis(i * 100 + 10000);
                wheel.insert(when, BenchOneShotTimer).ok()
            })
            .collect();

        b.iter(|| {
            if let Some(handle) = handles.pop() {
                black_box(wheel.cancel(handle))
            } else {
                // Refill when exhausted
                handles = (0..5000)
                    .filter_map(|i| {
                        let when = epoch + Duration::from_millis(i * 100 + 10000);
                        wheel.insert(when, BenchOneShotTimer).ok()
                    })
                    .collect();
                None
            }
        });
    });
}

// ==================== Poll Benchmarks ====================

fn bench_poll(c: &mut Criterion) {
    let mut group = c.benchmark_group("poll");

    group.bench_function("empty", |b| {
        let epoch = Instant::now();
        let mut wheel = bench_wheel();
        let mut ctx = ();
        let mut tick = 0u64;

        b.iter(|| {
            tick += 1;
            let now = epoch + Duration::from_millis(tick);
            black_box(wheel.poll(now, &mut ctx))
        });
    });

    group.bench_function("pending_no_fires", |b| {
        let epoch = Instant::now();
        let mut wheel = bench_wheel();

        // Insert timers far in future
        for i in 0..1000 {
            let when = epoch + Duration::from_millis(100_000 + i);
            let _ = wheel.insert(when, BenchOneShotTimer);
        }

        let mut ctx = ();
        let mut tick = 0u64;

        b.iter(|| {
            tick += 1;
            let now = epoch + Duration::from_millis(tick % 50_000);
            black_box(wheel.poll(now, &mut ctx))
        });
    });

    for num_timers in [100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("fire_all", num_timers),
            &num_timers,
            |b, &num_timers| {
                let epoch = Instant::now();

                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;

                    for _ in 0..iters {
                        let mut wheel = bench_wheel();
                        let mut ctx = ();

                        for i in 0..num_timers {
                            let when = epoch + Duration::from_millis(i as u64);
                            let _ = wheel.insert(when, BenchOneShotTimer);
                        }

                        let start = Instant::now();
                        let _ =
                            wheel.poll(epoch + Duration::from_millis(num_timers as u64), &mut ctx);
                        total += start.elapsed();
                    }

                    total
                });
            },
        );
    }

    group.finish();
}

// ==================== Mixed Workload Benchmarks ====================

fn bench_trading_workload(c: &mut Criterion) {
    c.bench_function("trading_workload", |b| {
        let epoch = Instant::now();

        b.iter_custom(|iters| {
            let mut wheel = bench_wheel();
            let mut handles = Vec::with_capacity(100);
            let mut ctx = ();
            let mut now = epoch;

            let start = Instant::now();

            for i in 0..iters {
                now += Duration::from_millis(1);

                // Poll every iteration
                let _ = wheel.poll(now, &mut ctx);

                // Insert 80% of iterations
                if i % 5 != 0 {
                    let delay = 50 + (i % 200);
                    let when = now + Duration::from_millis(delay);
                    if let Ok(handle) = wheel.insert(when, BenchOneShotTimer) {
                        if handles.len() < 100 {
                            handles.push(handle);
                        }
                    }
                }

                // Cancel 5% of iterations
                if i % 20 == 0 {
                    if let Some(handle) = handles.pop() {
                        let _ = wheel.cancel(handle);
                    }
                }
            }

            start.elapsed()
        });
    });
}

// ==================== Duration Until Next Benchmark ====================

fn bench_duration_until_next(c: &mut Criterion) {
    c.bench_function("duration_until_next", |b| {
        let epoch = Instant::now();
        let mut wheel = bench_wheel();

        for i in 0..1000 {
            let when = epoch + Duration::from_millis(100 + i);
            let _ = wheel.insert(when, BenchOneShotTimer);
        }

        b.iter(|| black_box(wheel.duration_until_next()));
    });
}

// Add to benches/bitwheel.rs

// ==================== Periodic Timer Benchmarks ====================

fn bench_periodic(c: &mut Criterion) {
    let mut group = c.benchmark_group("periodic");

    // Pure periodic steady state
    for num_timers in [1, 10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("steady_state", num_timers),
            &num_timers,
            |b, &num_timers| {
                let epoch = Instant::now();
                let mut wheel = bench_wheel_periodic();
                let mut ctx = 0usize;

                for i in 0..num_timers {
                    let offset = Duration::from_millis(i as u64);
                    let timer = BenchPeriodicTimer::new(Duration::from_millis(1), usize::MAX);
                    let _ = wheel.insert(epoch + offset + Duration::from_millis(1), timer);
                }

                let mut tick = 0u64;

                b.iter(|| {
                    tick += 1;
                    let now = epoch + Duration::from_millis(tick);
                    black_box(wheel.poll(now, &mut ctx))
                });
            },
        );
    }

    // Mixed: periodic background + one-shot arrivals
    // Simulates heartbeats + order timeouts
    for (num_periodic, oneshot_rate) in [(10, 2), (10, 10), (50, 5), (50, 20)] {
        group.bench_with_input(
            BenchmarkId::new(
                "mixed",
                format!("{}periodic_{}oneshot_per_tick", num_periodic, oneshot_rate),
            ),
            &(num_periodic, oneshot_rate),
            |b, &(num_periodic, oneshot_rate)| {
                let epoch = Instant::now();
                let mut wheel: Box<BitWheel<MixedTimer, 4, 1, 64, 8>> = BitWheel::boxed();
                let mut ctx = 0usize;

                // Background periodic timers (heartbeats every 10ms)
                for i in 0..num_periodic {
                    let offset = Duration::from_millis(i as u64);
                    let timer = MixedTimer::Periodic {
                        period: Duration::from_millis(10),
                        remaining: usize::MAX,
                    };
                    let _ = wheel.insert(epoch + offset + Duration::from_millis(10), timer);
                }

                let mut tick = 0u64;

                b.iter(|| {
                    tick += 1;
                    let now = epoch + Duration::from_millis(tick);

                    // Insert new one-shot timers (order timeouts, 50-100ms delay)
                    for j in 0..oneshot_rate {
                        let delay = 50 + (tick + j) % 50;
                        let when = now + Duration::from_millis(delay);
                        let _ = wheel.insert(when, MixedTimer::OneShot);
                    }

                    black_box(wheel.poll(now, &mut ctx))
                });
            },
        );
    }

    // Bursty: periodic background + occasional burst of one-shots
    // Simulates heartbeats + order fill burst
    for (num_periodic, burst_size) in [(10, 50), (10, 100), (50, 100)] {
        group.bench_with_input(
            BenchmarkId::new(
                "bursty",
                format!("{}periodic_{}burst_every_100", num_periodic, burst_size),
            ),
            &(num_periodic, burst_size),
            |b, &(num_periodic, burst_size)| {
                let epoch = Instant::now();
                let mut wheel: Box<BitWheel<MixedTimer, 4, 1, 128, 16>> = BitWheel::boxed();
                let mut ctx = 0usize;

                // Background periodic timers
                for i in 0..num_periodic {
                    let offset = Duration::from_millis(i as u64);
                    let timer = MixedTimer::Periodic {
                        period: Duration::from_millis(10),
                        remaining: usize::MAX,
                    };
                    let _ = wheel.insert(epoch + offset + Duration::from_millis(10), timer);
                }

                let mut tick = 0u64;

                b.iter(|| {
                    tick += 1;
                    let now = epoch + Duration::from_millis(tick);

                    // Burst of one-shots every 100 ticks
                    if tick % 100 == 0 {
                        for j in 0..burst_size {
                            let delay = 20 + j % 80;
                            let when = now + Duration::from_millis(delay);
                            let _ = wheel.insert(when, MixedTimer::OneShot);
                        }
                    }

                    black_box(wheel.poll(now, &mut ctx))
                });
            },
        );
    }

    group.finish();
}

// Mixed timer type for mixed workload benchmark
enum MixedTimer {
    OneShot,
    Periodic { period: Duration, remaining: usize },
}

impl Timer for MixedTimer {
    type Context = usize;

    fn fire(&mut self, now: Instant, ctx: &mut usize) -> Option<Instant> {
        *ctx += 1;
        match self {
            MixedTimer::OneShot => None,
            MixedTimer::Periodic { period, remaining } => {
                *remaining -= 1;
                if *remaining > 0 {
                    Some(now + *period)
                } else {
                    None
                }
            }
        }
    }
}

criterion_group!(
    benches,
    bench_insert,
    bench_cancel,
    bench_poll,
    bench_periodic,
    bench_trading_workload,
    bench_duration_until_next,
);

criterion_main!(benches);
