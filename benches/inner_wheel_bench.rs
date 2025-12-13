use bitwheel::wheel::{InnerWheel, InsertResult};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Benchmark single insert operation (no probing)
fn bench_single_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_ops");

    // Measure single insert into empty slot
    group.bench_function("insert_empty_slot", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;

        b.iter(|| {
            let r = unsafe { wheel.insert(slot, 42) };
            // Remove immediately to keep slot empty
            if let Ok(r) = r {
                unsafe { wheel.remove(r.slot, r.key) };
            }
            slot = (slot + 1) & 255;
            black_box(r)
        });
    });

    // Measure single remove
    group.bench_function("remove", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut results: Vec<InsertResult> = Vec::with_capacity(1024);

        // Pre-fill
        for i in 0..1024 {
            let r = unsafe { wheel.insert(i % 256, i as u64) }.unwrap();
            results.push(r);
        }

        let mut idx = 0usize;

        b.iter(|| {
            let r = &results[idx];
            // Use try_remove since we might have already removed
            let v = unsafe { wheel.try_remove(r.slot, r.key) };
            idx = (idx + 1) % 1024;

            // Re-insert to keep wheel populated
            if v.is_some() {
                let new_r = unsafe { wheel.insert(r.slot, 42) }.unwrap();
                results[idx] = new_r;
            }

            black_box(v)
        });
    });

    // Measure single try_remove (hit)
    group.bench_function("try_remove_hit", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;

        b.iter(|| {
            // Insert then immediately try_remove
            let r = unsafe { wheel.insert(slot, 42) }.unwrap();
            let v = unsafe { wheel.try_remove(r.slot, r.key) };
            slot = (slot + 1) & 255;
            black_box(v)
        });
    });

    // Measure single try_remove (miss)
    group.bench_function("try_remove_miss", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;
        let mut key = 0usize;

        b.iter(|| {
            // Empty wheel, all try_removes miss
            let v = unsafe { wheel.try_remove(slot, key) };
            slot = (slot + 1) & 255;
            key = (key + 1) & 3;
            black_box(v)
        });
    });

    // Measure single pop_slot (with entry)
    group.bench_function("pop_slot_hit", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;

        b.iter(|| {
            // Insert then pop
            unsafe { wheel.insert(slot, 42).unwrap() };
            let v = unsafe { wheel.pop_slot(slot) };
            slot = (slot + 1) & 255;
            black_box(v)
        });
    });

    // Measure single pop_slot (empty)
    group.bench_function("pop_slot_miss", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;

        b.iter(|| {
            let v = unsafe { wheel.pop_slot(slot) };
            slot = (slot + 1) & 255;
            black_box(v)
        });
    });

    group.finish();
}

/// Benchmark realistic timer patterns
fn bench_timer_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("timer_patterns");

    // Pattern 1: Insert timer, let it fire (pop), repeat
    // This is the happy path - no cancellation
    group.bench_function("insert_fire_cycle", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;

        b.iter(|| {
            let r = unsafe { wheel.insert(slot, 42) }.unwrap();
            let v = unsafe { wheel.pop_slot(r.slot) };
            slot = (slot + 1) & 255;
            black_box(v)
        });
    });

    // Pattern 2: Insert timer, cancel it, repeat
    // Common for request timeouts that complete normally
    group.bench_function("insert_cancel_cycle", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;

        b.iter(|| {
            let r = unsafe { wheel.insert(slot, 42) }.unwrap();
            let v = unsafe { wheel.remove(r.slot, r.key) };
            slot = (slot + 1) & 255;
            black_box(v)
        });
    });

    // Pattern 3: Periodic ticker - pop then reinsert
    group.bench_function("ticker_reinsert", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);

        // Pre-insert one timer per slot
        for slot in 0..256 {
            unsafe { wheel.insert(slot, slot as u64).unwrap() };
        }

        let mut slot = 0usize;

        b.iter(|| {
            // Pop current slot's timer
            if let Some(v) = unsafe { wheel.pop_slot(slot) } {
                // Reinsert (simulating periodic timer)
                unsafe { wheel.insert(slot, v).unwrap() };
            }
            slot = (slot + 1) & 255;
        });
    });

    // Pattern 4: Steady state with some churn
    // 50% fill, alternating insert/cancel
    group.bench_function("steady_state_churn", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut pending: Vec<Option<InsertResult>> = vec![None; 512];

        // Pre-fill 50%
        for i in 0..512 {
            let r = unsafe { wheel.insert(i % 256, i as u64) }.unwrap();
            pending[i] = Some(r);
        }

        let mut idx = 0usize;

        b.iter(|| {
            // Cancel old timer if exists
            if let Some(r) = pending[idx].take() {
                unsafe { wheel.try_remove(r.slot, r.key) };
            }

            // Insert new timer
            let r = unsafe { wheel.insert(idx % 256, idx as u64) }.unwrap();
            pending[idx] = Some(r);

            idx = (idx + 1) % 512;
        });
    });

    group.finish();
}

/// Benchmark probing overhead (for documentation purposes)
fn bench_probing_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("probing_overhead");

    // Best case: empty slot
    group.bench_function("probe_0", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 4);
        let mut slot = 0usize;

        b.iter(|| {
            let r = unsafe { wheel.insert(slot, 42) }.unwrap();
            unsafe { wheel.remove(r.slot, r.key) };
            slot = (slot + 1) & 255;
            black_box(r)
        });
    });

    // Probe 1 slot (first slot full)
    group.bench_function("probe_1", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 1);

        // Fill even slots
        for slot in (0..256).step_by(2) {
            unsafe { wheel.insert(slot, 0).unwrap() };
        }

        let mut slot = 0usize;

        b.iter(|| {
            // Insert at even slot, probes to odd
            let r = unsafe { wheel.insert(slot, 42) }.unwrap();
            unsafe { wheel.remove(r.slot, r.key) };
            slot = (slot + 2) & 255;
            black_box(r)
        });
    });

    // Probe 4 slots
    group.bench_function("probe_4", |b| {
        let mut wheel: InnerWheel<u64> = InnerWheel::new(256, 1);

        // Fill slots 0-3, 8-11, 16-19, etc
        for base in (0..256).step_by(8) {
            for offset in 0..4 {
                if base + offset < 256 {
                    unsafe { wheel.insert(base + offset, 0).unwrap() };
                }
            }
        }

        let mut slot = 0usize;

        b.iter(|| {
            // Insert at base, probes 4 slots
            let r = unsafe { wheel.insert(slot, 42) }.unwrap();
            unsafe { wheel.remove(r.slot, r.key) };
            slot = (slot + 8) & 255;
            black_box(r)
        });
    });

    group.finish();
}

/// Benchmark slot capacity impact
fn bench_slot_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("slot_capacity");

    for cap in [2, 4, 8, 16] {
        group.bench_with_input(BenchmarkId::new("insert_remove", cap), &cap, |b, &cap| {
            let mut wheel: InnerWheel<u64> = InnerWheel::new(256, cap);
            let mut slot = 0usize;

            b.iter(|| {
                let r = unsafe { wheel.insert(slot, 42) }.unwrap();
                let v = unsafe { wheel.remove(r.slot, r.key) };
                slot = (slot + 1) & 255;
                black_box(v)
            });
        });

        group.bench_with_input(BenchmarkId::new("pop_full_slot", cap), &cap, |b, &cap| {
            let mut wheel: InnerWheel<u64> = InnerWheel::new(256, cap);
            let mut slot = 0usize;

            b.iter(|| {
                // Fill slot
                for i in 0..cap {
                    unsafe { wheel.insert(slot, i as u64).unwrap() };
                }
                // Pop all
                for _ in 0..cap {
                    black_box(unsafe { wheel.pop_slot(slot) });
                }
                slot = (slot + 1) & 255;
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_insert,
    bench_timer_patterns,
    bench_probing_overhead,
    bench_slot_capacity,
);

criterion_main!(benches);
