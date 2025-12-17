use std::{
    fmt::Debug,
    time::{Duration, Instant},
};

pub use heap::{DEFAULT_TICKER_CAP, MIN_PERIOD, TickerHeap};

pub type TickerHeap16<T> = TickerHeap<T, 16>;
pub type TickerHeap32<T> = TickerHeap<T, 32>;
pub type TickerHeap64<T> = TickerHeap<T, 64>;
pub type TickerHeap128<T> = TickerHeap<T, 128>;
pub type TickerHeap256<T> = TickerHeap<T, 256>;
pub type TickerHeap512<T> = TickerHeap<T, 512>;

#[derive(Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("could not insert into ticker heap")]
pub struct InsertError<T>(pub T);

impl<T> Debug for InsertError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("could not insert into ticker heap")
    }
}

pub trait Ticker {
    type Context;
    fn fire(&mut self, ctx: &mut Self::Context);
    fn period(&self) -> Duration;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TickerHandle(pub(crate) u16);

pub trait TickerDriver<T: Ticker> {
    fn insert(&mut self, ticker: T, now: Instant) -> Result<TickerHandle, InsertError<T>>;
    fn remove(&mut self, handle: TickerHandle) -> Option<T>;
    fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize;
    fn peek_next_fire(&self) -> Option<Instant>;
    fn is_active(&self, handle: TickerHandle) -> bool;
    fn len(&self) -> usize;

    fn cancel(&mut self, handle: TickerHandle) -> bool {
        self.remove(handle).is_some()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

mod heap {
    use super::*;
    use std::mem::MaybeUninit;

    pub const DEFAULT_TICKER_CAP: usize = 32;
    pub const MIN_PERIOD: Duration = Duration::from_millis(1);
    const NONE: u16 = u16::MAX;

    struct Entry<T> {
        ticker: T,
        fire_at: Instant,
    }

    /// Fixed-capacity min-heap for periodic tickers.
    ///
    /// Optimized for small n (16-128 tickers) with:
    /// - u16 indices for cache-friendly heap operations
    /// - Stable handles across reschedules
    /// - O(1) peek, O(log n) insert/remove/poll
    pub struct TickerHeap<T, const CAP: usize = DEFAULT_TICKER_CAP> {
        entries: [MaybeUninit<Entry<T>>; CAP],
        free_stack: [u16; CAP],
        free_len: u16,
        heap: [u16; CAP],
        heap_len: u16,
        heap_pos: [u16; CAP],
    }

    impl<T, const CAP: usize> Default for TickerHeap<T, CAP> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T, const CAP: usize> TickerHeap<T, CAP> {
        pub fn new() -> Self {
            const {
                assert!(CAP > 0, "capacity must be > 0");
                assert!(CAP <= u16::MAX as usize - 1, "capacity must fit in u16");
            }

            let mut free_stack = [0u16; CAP];
            for i in 0..CAP {
                free_stack[i] = i as u16;
            }

            Self {
                entries: unsafe { MaybeUninit::uninit().assume_init() },
                free_stack,
                free_len: CAP as u16,
                heap: [NONE; CAP],
                heap_len: 0,
                heap_pos: [NONE; CAP],
            }
        }

        #[inline]
        fn swim(&mut self, mut pos: u16) {
            while pos > 0 {
                let parent = (pos - 1) / 2;
                if self.compare(pos, parent).is_lt() {
                    self.swap(pos, parent);
                    pos = parent;
                } else {
                    break;
                }
            }
        }

        #[inline]
        fn sink(&mut self, mut pos: u16) {
            loop {
                let left = 2 * pos + 1;
                let right = 2 * pos + 2;
                let mut smallest = pos;

                if left < self.heap_len && self.compare(left, smallest).is_lt() {
                    smallest = left;
                }
                if right < self.heap_len && self.compare(right, smallest).is_lt() {
                    smallest = right;
                }

                if smallest == pos {
                    break;
                }

                self.swap(pos, smallest);
                pos = smallest;
            }
        }

        #[inline]
        fn swap(&mut self, a: u16, b: u16) {
            let a_idx = a as usize;
            let b_idx = b as usize;
            self.heap.swap(a_idx, b_idx);
            self.heap_pos[self.heap[a_idx] as usize] = a;
            self.heap_pos[self.heap[b_idx] as usize] = b;
        }

        #[inline]
        fn compare(&self, a: u16, b: u16) -> std::cmp::Ordering {
            let entry_a = self.heap[a as usize] as usize;
            let entry_b = self.heap[b as usize] as usize;
            unsafe {
                let fire_a = (*self.entries[entry_a].as_ptr()).fire_at;
                let fire_b = (*self.entries[entry_b].as_ptr()).fire_at;
                fire_a.cmp(&fire_b)
            }
        }
    }

    impl<T: Ticker, const CAP: usize> TickerDriver<T> for TickerHeap<T, CAP> {
        fn insert(&mut self, ticker: T, now: Instant) -> Result<TickerHandle, InsertError<T>> {
            if self.free_len == 0 {
                return Err(InsertError(ticker));
            }

            self.free_len -= 1;
            let entry_idx = self.free_stack[self.free_len as usize];

            let period = ticker.period().max(MIN_PERIOD);
            let entry = Entry {
                ticker,
                fire_at: now + period,
            };
            self.entries[entry_idx as usize].write(entry);

            let heap_pos = self.heap_len;
            self.heap[heap_pos as usize] = entry_idx;
            self.heap_pos[entry_idx as usize] = heap_pos;
            self.heap_len += 1;

            self.swim(heap_pos);

            Ok(TickerHandle(entry_idx))
        }

        fn remove(&mut self, handle: TickerHandle) -> Option<T> {
            let entry_idx = handle.0;

            if entry_idx as usize >= CAP || self.heap_pos[entry_idx as usize] == NONE {
                return None;
            }

            let heap_pos = self.heap_pos[entry_idx as usize];

            self.heap_len -= 1;
            if heap_pos < self.heap_len {
                let last_idx = self.heap_len;
                self.heap[heap_pos as usize] = self.heap[last_idx as usize];
                self.heap_pos[self.heap[heap_pos as usize] as usize] = heap_pos;

                self.sink(heap_pos);
                self.swim(heap_pos);
            }

            self.heap_pos[entry_idx as usize] = NONE;

            let entry = unsafe { self.entries[entry_idx as usize].assume_init_read() };
            self.free_stack[self.free_len as usize] = entry_idx;
            self.free_len += 1;

            Some(entry.ticker)
        }

        fn poll(&mut self, now: Instant, ctx: &mut T::Context) -> usize {
            let mut fired = 0;

            while self.heap_len > 0 {
                let entry_idx = self.heap[0] as usize;

                let fire_at = unsafe { (*self.entries[entry_idx].as_ptr()).fire_at };

                if fire_at > now {
                    break;
                }

                let entry = unsafe { self.entries[entry_idx].assume_init_mut() };
                entry.ticker.fire(ctx);
                fired += 1;

                let period = entry.ticker.period().max(MIN_PERIOD);
                entry.fire_at = now + period;

                self.sink(0);
            }

            fired
        }

        #[inline]
        fn peek_next_fire(&self) -> Option<Instant> {
            if self.heap_len == 0 {
                return None;
            }

            let entry_idx = self.heap[0] as usize;
            Some(unsafe { (*self.entries[entry_idx].as_ptr()).fire_at })
        }

        #[inline]
        fn is_active(&self, handle: TickerHandle) -> bool {
            let entry_idx = handle.0 as usize;
            entry_idx < CAP && self.heap_pos[entry_idx] != NONE
        }

        #[inline]
        fn len(&self) -> usize {
            self.heap_len as usize
        }
    }

    impl<T, const CAP: usize> Drop for TickerHeap<T, CAP> {
        fn drop(&mut self) {
            for i in 0..self.heap_len {
                let entry_idx = self.heap[i as usize] as usize;
                unsafe { self.entries[entry_idx].assume_init_drop() };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    // ==================== Test Ticker Implementations ====================

    struct SimpleTicker {
        id: usize,
        period: Duration,
        fire_count: Rc<Cell<usize>>,
    }

    impl SimpleTicker {
        fn new(id: usize, period_ms: u64) -> (Self, Rc<Cell<usize>>) {
            let fire_count = Rc::new(Cell::new(0));
            (
                Self {
                    id,
                    period: Duration::from_millis(period_ms),
                    fire_count: Rc::clone(&fire_count),
                },
                fire_count,
            )
        }
    }

    impl Ticker for SimpleTicker {
        type Context = Vec<usize>;

        fn fire(&mut self, ctx: &mut Self::Context) {
            self.fire_count.set(self.fire_count.get() + 1);
            ctx.push(self.id);
        }

        fn period(&self) -> Duration {
            self.period
        }
    }

    struct CounterTicker {
        period: Duration,
    }

    impl CounterTicker {
        fn new(period_ms: u64) -> Self {
            Self {
                period: Duration::from_millis(period_ms),
            }
        }
    }

    impl Ticker for CounterTicker {
        type Context = usize;

        fn fire(&mut self, ctx: &mut Self::Context) {
            *ctx += 1;
        }

        fn period(&self) -> Duration {
            self.period
        }
    }

    // ==================== Construction ====================

    #[test]
    fn test_new_empty() {
        let heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert!(heap.peek_next_fire().is_none());
    }

    #[test]
    fn test_default() {
        let heap: TickerHeap<CounterTicker, 8> = TickerHeap::default();
        assert!(heap.is_empty());
    }

    // ==================== Insert ====================

    #[test]
    fn test_insert_single() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let handle = heap.insert(CounterTicker::new(100), now).unwrap();

        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 1);
        assert!(heap.is_active(handle));
        assert!(heap.peek_next_fire().is_some());
    }

    #[test]
    fn test_insert_multiple() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let h1 = heap.insert(CounterTicker::new(100), now).unwrap();
        let h2 = heap.insert(CounterTicker::new(200), now).unwrap();
        let h3 = heap.insert(CounterTicker::new(50), now).unwrap();

        assert_eq!(heap.len(), 3);
        assert!(heap.is_active(h1));
        assert!(heap.is_active(h2));
        assert!(heap.is_active(h3));
    }

    #[test]
    fn test_insert_at_capacity() {
        let mut heap: TickerHeap<CounterTicker, 2> = TickerHeap::new();
        let now = Instant::now();

        let h1 = heap.insert(CounterTicker::new(100), now);
        let h2 = heap.insert(CounterTicker::new(100), now);
        let h3 = heap.insert(CounterTicker::new(100), now);

        assert!(h1.is_ok());
        assert!(h2.is_ok());
        assert!(h3.is_err());

        // Verify we get the ticker back on error
        let err = h3.unwrap_err();
        assert_eq!(err.0.period(), Duration::from_millis(100));
    }

    #[test]
    fn test_insert_after_remove_reuses_slot() {
        let mut heap: TickerHeap<CounterTicker, 2> = TickerHeap::new();
        let now = Instant::now();

        let h1 = heap.insert(CounterTicker::new(100), now).unwrap();
        let h2 = heap.insert(CounterTicker::new(100), now).unwrap();

        // Full
        assert!(heap.insert(CounterTicker::new(100), now).is_err());

        // Remove one
        heap.remove(h1);

        // Can insert again
        let h3 = heap.insert(CounterTicker::new(100), now);
        assert!(h3.is_ok());

        // h2 still active
        assert!(heap.is_active(h2));
    }

    // ==================== Heap Ordering ====================

    #[test]
    fn test_peek_returns_earliest() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        // Insert in non-sorted order
        heap.insert(CounterTicker::new(300), now).unwrap();
        heap.insert(CounterTicker::new(100), now).unwrap();
        heap.insert(CounterTicker::new(200), now).unwrap();

        let next = heap.peek_next_fire().unwrap();
        let expected = now + Duration::from_millis(100);

        assert_eq!(next, expected);
    }

    #[test]
    fn test_poll_fires_in_order() {
        let mut heap: TickerHeap<SimpleTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let (t1, _) = SimpleTicker::new(1, 100);
        let (t2, _) = SimpleTicker::new(2, 50);
        let (t3, _) = SimpleTicker::new(3, 150);

        heap.insert(t1, now).unwrap();
        heap.insert(t2, now).unwrap();
        heap.insert(t3, now).unwrap();

        let mut ctx = Vec::new();

        // Poll at 60ms - only ticker 2 (50ms) should fire
        let fired = heap.poll(now + Duration::from_millis(60), &mut ctx);
        assert_eq!(fired, 1);
        assert_eq!(ctx, vec![2]);

        // Poll at 110ms - ticker 2 fires again (rescheduled to 110), ticker 1 fires (100)
        ctx.clear();
        let fired = heap.poll(now + Duration::from_millis(110), &mut ctx);
        assert_eq!(fired, 2);
        assert!(ctx.contains(&1));
        assert!(ctx.contains(&2));
    }

    // ==================== Remove / Cancel ====================

    #[test]
    fn test_remove_returns_ticker() {
        let mut heap: TickerHeap<SimpleTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let (ticker, _) = SimpleTicker::new(42, 100);
        let handle = heap.insert(ticker, now).unwrap();

        let removed = heap.remove(handle);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, 42);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_remove_invalid_handle() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();

        let fake_handle = TickerHandle(99);
        assert!(heap.remove(fake_handle).is_none());
    }

    #[test]
    fn test_double_remove() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let handle = heap.insert(CounterTicker::new(100), now).unwrap();

        assert!(heap.remove(handle).is_some());
        assert!(heap.remove(handle).is_none());
    }

    #[test]
    fn test_cancel_returns_bool() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let handle = heap.insert(CounterTicker::new(100), now).unwrap();

        assert!(heap.cancel(handle));
        assert!(!heap.cancel(handle));
    }

    #[test]
    fn test_remove_middle_maintains_heap() {
        let mut heap: TickerHeap<SimpleTicker, 8> = TickerHeap::new();
        let now = Instant::now();

        let (t1, _) = SimpleTicker::new(1, 100);
        let (t2, _) = SimpleTicker::new(2, 50);
        let (t3, _) = SimpleTicker::new(3, 150);
        let (t4, _) = SimpleTicker::new(4, 75);

        heap.insert(t1, now).unwrap();
        let h2 = heap.insert(t2, now).unwrap();
        heap.insert(t3, now).unwrap();
        heap.insert(t4, now).unwrap();

        // Remove ticker 2 (the earliest)
        heap.remove(h2);

        // Next fire should now be ticker 4 (75ms)
        let next = heap.peek_next_fire().unwrap();
        assert_eq!(next, now + Duration::from_millis(75));
    }

    // ==================== Poll ====================

    #[test]
    fn test_poll_empty() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();
        let mut ctx = 0usize;

        let fired = heap.poll(now, &mut ctx);
        assert_eq!(fired, 0);
        assert_eq!(ctx, 0);
    }

    #[test]
    fn test_poll_before_fire_time() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        heap.insert(CounterTicker::new(100), now).unwrap();

        let mut ctx = 0usize;
        let fired = heap.poll(now + Duration::from_millis(50), &mut ctx);

        assert_eq!(fired, 0);
        assert_eq!(ctx, 0);
    }

    #[test]
    fn test_poll_at_fire_time() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        heap.insert(CounterTicker::new(100), now).unwrap();

        let mut ctx = 0usize;
        let fired = heap.poll(now + Duration::from_millis(100), &mut ctx);

        assert_eq!(fired, 1);
        assert_eq!(ctx, 1);
    }

    #[test]
    fn test_poll_reschedules() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        heap.insert(CounterTicker::new(100), now).unwrap();

        let mut ctx = 0usize;

        // First fire at 100ms
        heap.poll(now + Duration::from_millis(100), &mut ctx);
        assert_eq!(ctx, 1);
        assert_eq!(heap.len(), 1); // Still in heap

        // Next fire should be at 200ms (100 + 100)
        let next = heap.peek_next_fire().unwrap();
        assert_eq!(next, now + Duration::from_millis(200));

        // Fire again at 200ms
        heap.poll(now + Duration::from_millis(200), &mut ctx);
        assert_eq!(ctx, 2);
    }

    #[test]
    fn test_poll_no_catchup_semantics() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        heap.insert(CounterTicker::new(50), now).unwrap();

        let mut ctx = 0usize;

        // Poll at 250ms - fires once (no catch-up), reschedules to 300ms
        let fired = heap.poll(now + Duration::from_millis(250), &mut ctx);
        assert_eq!(fired, 1);
        assert_eq!(ctx, 1);

        // Next fire at 300ms (250 + 50), not catching up
        let next = heap.peek_next_fire().unwrap();
        assert_eq!(next, now + Duration::from_millis(300));
    }

    #[test]
    fn test_poll_incremental() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        heap.insert(CounterTicker::new(50), now).unwrap();

        let mut ctx = 0usize;

        // Poll incrementally - this is how it would be used in practice
        heap.poll(now + Duration::from_millis(50), &mut ctx);
        assert_eq!(ctx, 1);

        heap.poll(now + Duration::from_millis(100), &mut ctx);
        assert_eq!(ctx, 2);

        heap.poll(now + Duration::from_millis(150), &mut ctx);
        assert_eq!(ctx, 3);

        heap.poll(now + Duration::from_millis(200), &mut ctx);
        assert_eq!(ctx, 4);

        heap.poll(now + Duration::from_millis(250), &mut ctx);
        assert_eq!(ctx, 5);
    }

    // ==================== Handle Stability ====================

    #[test]
    fn test_handle_stable_across_fires() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let handle = heap.insert(CounterTicker::new(100), now).unwrap();

        let mut ctx = 0usize;

        // Fire multiple times
        heap.poll(now + Duration::from_millis(100), &mut ctx);
        heap.poll(now + Duration::from_millis(200), &mut ctx);
        heap.poll(now + Duration::from_millis(300), &mut ctx);

        // Handle still valid
        assert!(heap.is_active(handle));

        // Can still cancel
        assert!(heap.cancel(handle));
        assert!(!heap.is_active(handle));
    }

    // ==================== is_active ====================

    #[test]
    fn test_is_active_after_insert() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let handle = heap.insert(CounterTicker::new(100), now).unwrap();
        assert!(heap.is_active(handle));
    }

    #[test]
    fn test_is_active_after_remove() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let handle = heap.insert(CounterTicker::new(100), now).unwrap();
        heap.remove(handle);
        assert!(!heap.is_active(handle));
    }

    #[test]
    fn test_is_active_invalid_handle() {
        let heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();

        assert!(!heap.is_active(TickerHandle(0)));
        assert!(!heap.is_active(TickerHandle(99)));
    }

    // ==================== Drop Behavior ====================

    #[test]
    fn test_drop_cleans_up_entries() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropTicker {
            drop_count: Rc<Cell<usize>>,
        }

        impl Ticker for DropTicker {
            type Context = ();
            fn fire(&mut self, _ctx: &mut Self::Context) {}
            fn period(&self) -> Duration {
                Duration::from_millis(100)
            }
        }

        impl Drop for DropTicker {
            fn drop(&mut self) {
                self.drop_count.set(self.drop_count.get() + 1);
            }
        }

        {
            let mut heap: TickerHeap<DropTicker, 4> = TickerHeap::new();
            let now = Instant::now();

            heap.insert(
                DropTicker {
                    drop_count: Rc::clone(&drop_count),
                },
                now,
            )
            .unwrap();
            heap.insert(
                DropTicker {
                    drop_count: Rc::clone(&drop_count),
                },
                now,
            )
            .unwrap();
            heap.insert(
                DropTicker {
                    drop_count: Rc::clone(&drop_count),
                },
                now,
            )
            .unwrap();

            assert_eq!(drop_count.get(), 0);
        }

        assert_eq!(drop_count.get(), 3);
    }

    #[test]
    fn test_drop_after_partial_remove() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropTicker {
            drop_count: Rc<Cell<usize>>,
        }

        impl Ticker for DropTicker {
            type Context = ();
            fn fire(&mut self, _ctx: &mut Self::Context) {}
            fn period(&self) -> Duration {
                Duration::from_millis(100)
            }
        }

        impl Drop for DropTicker {
            fn drop(&mut self) {
                self.drop_count.set(self.drop_count.get() + 1);
            }
        }

        {
            let mut heap: TickerHeap<DropTicker, 4> = TickerHeap::new();
            let now = Instant::now();

            let h1 = heap
                .insert(
                    DropTicker {
                        drop_count: Rc::clone(&drop_count),
                    },
                    now,
                )
                .unwrap();
            heap.insert(
                DropTicker {
                    drop_count: Rc::clone(&drop_count),
                },
                now,
            )
            .unwrap();

            heap.remove(h1);
            assert_eq!(drop_count.get(), 1);
        }

        assert_eq!(drop_count.get(), 2);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_capacity_one() {
        let mut heap: TickerHeap<CounterTicker, 1> = TickerHeap::new();
        let now = Instant::now();

        let h = heap.insert(CounterTicker::new(100), now).unwrap();
        assert!(heap.insert(CounterTicker::new(100), now).is_err());

        heap.remove(h);
        assert!(heap.insert(CounterTicker::new(100), now).is_ok());
    }

    #[test]
    fn test_same_fire_time() {
        let mut heap: TickerHeap<SimpleTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        let (t1, _) = SimpleTicker::new(1, 100);
        let (t2, _) = SimpleTicker::new(2, 100);
        let (t3, _) = SimpleTicker::new(3, 100);

        heap.insert(t1, now).unwrap();
        heap.insert(t2, now).unwrap();
        heap.insert(t3, now).unwrap();

        let mut ctx = Vec::new();
        let fired = heap.poll(now + Duration::from_millis(100), &mut ctx);

        assert_eq!(fired, 3);
        assert_eq!(ctx.len(), 3);
        // All three fired (order may vary due to heap structure)
        assert!(ctx.contains(&1));
        assert!(ctx.contains(&2));
        assert!(ctx.contains(&3));
    }

    #[test]
    fn test_zero_period_clamped_to_1ms() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        // Zero period gets clamped to 1ms
        heap.insert(CounterTicker::new(0), now).unwrap();

        let mut ctx = 0usize;

        // At now, nothing fires (fire_at = now + 1ms)
        let fired = heap.poll(now, &mut ctx);
        assert_eq!(fired, 0);

        // At now + 1ms, it fires
        let fired = heap.poll(now + Duration::from_millis(1), &mut ctx);
        assert_eq!(fired, 1);

        // Reschedules to now + 2ms
        let next = heap.peek_next_fire().unwrap();
        assert_eq!(next, now + Duration::from_millis(2));
    }

    // ==================== Stress Tests ====================

    #[test]
    fn test_fill_and_drain() {
        let mut heap: TickerHeap<CounterTicker, 8> = TickerHeap::new();
        let now = Instant::now();

        let mut handles = Vec::new();
        for i in 0..8 {
            let h = heap
                .insert(CounterTicker::new((i + 1) as u64 * 10), now)
                .unwrap();
            handles.push(h);
        }

        assert_eq!(heap.len(), 8);

        for h in handles {
            heap.remove(h);
        }

        assert!(heap.is_empty());
    }

    #[test]
    fn test_interleaved_insert_remove() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        // Insert and immediately remove - never exceeds capacity
        for _ in 0..100 {
            let h = heap.insert(CounterTicker::new(100), now).unwrap();
            heap.remove(h);
        }

        assert!(heap.is_empty());
    }

    #[test]
    fn test_long_running_incremental_poll() {
        let mut heap: TickerHeap<CounterTicker, 4> = TickerHeap::new();
        let now = Instant::now();

        heap.insert(CounterTicker::new(10), now).unwrap();
        heap.insert(CounterTicker::new(25), now).unwrap();
        heap.insert(CounterTicker::new(50), now).unwrap();

        let mut ctx = 0usize;

        // Poll incrementally every 10ms for 100ms
        for i in 1..=10 {
            heap.poll(now + Duration::from_millis(i * 10), &mut ctx);
        }

        // 10ms ticker: 10 fires (at 10,20,30,40,50,60,70,80,90,100)
        // 25ms ticker: 3 fires (at 25→30, 55→60, 85→90)
        // 50ms ticker: 2 fires (at 50, 100)
        assert_eq!(ctx, 10 + 3 + 2);
    }
}

#[cfg(test)]
mod latency_tests {
    use super::*;
    use hdrhistogram::Histogram;
    use std::time::{Duration, Instant};

    const WARMUP: u64 = 100_000;
    const ITERATIONS: u64 = 1_000_000;

    // ==================== Test Ticker for Benchmarks ====================

    struct BenchTicker {
        period: Duration,
    }

    impl BenchTicker {
        fn new(period_ms: u64) -> Self {
            Self {
                period: Duration::from_millis(period_ms),
            }
        }
    }

    impl Ticker for BenchTicker {
        type Context = ();

        fn fire(&mut self, _ctx: &mut Self::Context) {}

        fn period(&self) -> Duration {
            self.period
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
    fn hdr_insert_latency() {
        let mut heap: TickerHeap<BenchTicker, 64> = TickerHeap::new();
        let mut hist = Histogram::<u64>::new(3).unwrap();
        let now = Instant::now();

        // Warmup
        for i in 0..WARMUP {
            let period = (i % 500) + 10;
            let handle = heap.insert(BenchTicker::new(period), now).unwrap();
            heap.remove(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let period = (i % 500) + 10;

            let start = Instant::now();
            let handle = heap.insert(BenchTicker::new(period), now).unwrap();
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
            heap.remove(handle);
        }

        print_histogram("Insert Latency", &hist);
    }

    // ==================== Remove Latency ====================

    #[test]
    #[ignore]
    fn hdr_remove_latency() {
        let mut heap: TickerHeap<BenchTicker, 64> = TickerHeap::new();
        let mut hist = Histogram::<u64>::new(3).unwrap();
        let now = Instant::now();

        // Warmup
        for i in 0..WARMUP {
            let period = (i % 500) + 10;
            let handle = heap.insert(BenchTicker::new(period), now).unwrap();
            heap.remove(handle);
        }

        // Measure
        for i in 0..ITERATIONS {
            let period = (i % 500) + 10;
            let handle = heap.insert(BenchTicker::new(period), now).unwrap();

            let start = Instant::now();
            heap.remove(handle);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Remove Latency", &hist);
    }

    // ==================== Poll Empty ====================

    #[test]
    #[ignore]
    fn hdr_poll_empty() {
        let mut heap: TickerHeap<BenchTicker, 64> = TickerHeap::new();
        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let now = Instant::now();

        // Warmup
        for i in 0..WARMUP {
            let poll_time = now + Duration::from_millis(i);
            heap.poll(poll_time, &mut ctx);
        }

        // Measure
        for i in 0..ITERATIONS {
            let poll_time = now + Duration::from_millis(WARMUP + i);

            let start = Instant::now();
            heap.poll(poll_time, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Poll Empty", &hist);
    }

    // ==================== Poll Pending (No Fires) ====================

    #[test]
    #[ignore]
    fn hdr_poll_pending_no_fires() {
        let mut heap: TickerHeap<BenchTicker, 64> = TickerHeap::new();
        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let now = Instant::now();

        // Insert tickers far in future
        for i in 0..16 {
            let period = 100_000_000 + i;
            heap.insert(BenchTicker::new(period), now).unwrap();
        }

        // Warmup
        for i in 0..WARMUP {
            let poll_time = now + Duration::from_millis(i);
            heap.poll(poll_time, &mut ctx);
        }

        // Measure
        for i in 0..ITERATIONS {
            let poll_time = now + Duration::from_millis(WARMUP + i);

            let start = Instant::now();
            heap.poll(poll_time, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Poll Pending (No Fires)", &hist);
    }

    // ==================== Poll Single Fire ====================

    #[test]
    #[ignore]
    fn hdr_poll_single_fire() {
        let mut heap: TickerHeap<BenchTicker, 64> = TickerHeap::new();
        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let now = Instant::now();

        // One ticker that fires every ms
        heap.insert(BenchTicker::new(1), now).unwrap();

        // Warmup
        for i in 0..WARMUP {
            let poll_time = now + Duration::from_millis(i + 1);
            heap.poll(poll_time, &mut ctx);
        }

        // Measure
        for i in 0..ITERATIONS {
            let poll_time = now + Duration::from_millis(WARMUP + i + 1);

            let start = Instant::now();
            heap.poll(poll_time, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Poll Single Fire", &hist);
    }

    // ==================== Periodic Steady State ====================

    #[test]
    #[ignore]
    fn hdr_periodic_steady_state() {
        let mut heap: TickerHeap<BenchTicker, 64> = TickerHeap::new();
        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let now = Instant::now();

        // 10 tickers with 1ms period (all fire every tick)
        for i in 0..10 {
            heap.insert(BenchTicker::new(1), now + Duration::from_micros(i * 100))
                .unwrap();
        }

        // Warmup
        for i in 0..WARMUP {
            let poll_time = now + Duration::from_millis(i + 1);
            heap.poll(poll_time, &mut ctx);
        }

        // Measure
        for i in 0..ITERATIONS {
            let poll_time = now + Duration::from_millis(WARMUP + i + 1);

            let start = Instant::now();
            heap.poll(poll_time, &mut ctx);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Periodic Steady State (10 tickers @ 1ms)", &hist);
    }

    // ==================== Realistic Heartbeat Scenario ====================

    #[test]
    #[ignore]
    fn hdr_realistic_heartbeats() {
        let mut heap: TickerHeap<BenchTicker, 32> = TickerHeap::new();
        let mut insert_hist = Histogram::<u64>::new(3).unwrap();
        let mut poll_hist = Histogram::<u64>::new(3).unwrap();
        let mut remove_hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let now = Instant::now();

        // Simulate: 5 venue heartbeats at different periods
        let mut handles = Vec::new();
        for i in 0..5 {
            let period = 10 + i * 5; // 10, 15, 20, 25, 30ms
            let h = heap.insert(BenchTicker::new(period), now).unwrap();
            handles.push(h);
        }

        // Warmup
        for i in 0..WARMUP {
            let poll_time = now + Duration::from_millis(i + 1);
            heap.poll(poll_time, &mut ctx);

            // Occasionally add/remove a ticker (simulating venue connect/disconnect)
            // Keep adds and removes balanced
            if i % 1000 == 0 && handles.len() < 20 {
                let h = heap.insert(BenchTicker::new(50), poll_time).unwrap();
                handles.push(h);
            }
            if i % 1000 == 500 && handles.len() > 5 {
                if let Some(h) = handles.pop() {
                    heap.remove(h);
                }
            }
        }

        // Measure
        for i in 0..ITERATIONS {
            let poll_time = now + Duration::from_millis(WARMUP + i + 1);

            // Poll
            let start = Instant::now();
            heap.poll(poll_time, &mut ctx);
            poll_hist.record(start.elapsed().as_nanos() as u64).unwrap();

            // Occasionally add ticker (venue reconnect)
            if i % 1000 == 0 && handles.len() < 20 {
                let start = Instant::now();
                let h = heap.insert(BenchTicker::new(50), poll_time).unwrap();
                insert_hist
                    .record(start.elapsed().as_nanos() as u64)
                    .unwrap();
                handles.push(h);
            }

            // Occasionally remove ticker (venue disconnect)
            if i % 1000 == 500 && handles.len() > 5 {
                if let Some(h) = handles.pop() {
                    let start = Instant::now();
                    heap.remove(h);
                    remove_hist
                        .record(start.elapsed().as_nanos() as u64)
                        .unwrap();
                }
            }
        }

        print_histogram("Heartbeat Scenario - Insert", &insert_hist);
        print_histogram("Heartbeat Scenario - Poll", &poll_hist);
        print_histogram("Heartbeat Scenario - Remove", &remove_hist);
    }

    // ==================== Stable Handle Cancel After Fires ====================

    #[test]
    #[ignore]
    fn hdr_cancel_after_fires() {
        let mut heap: TickerHeap<BenchTicker, 64> = TickerHeap::new();
        let mut hist = Histogram::<u64>::new(3).unwrap();
        let mut ctx = ();
        let now = Instant::now();

        // Warmup
        for i in 0..WARMUP {
            let poll_time = now + Duration::from_millis(i);
            let h = heap.insert(BenchTicker::new(1), poll_time).unwrap();
            // Fire it multiple times
            heap.poll(poll_time + Duration::from_millis(5), &mut ctx);
            heap.remove(h);
        }

        // Measure: insert, let it fire several times, then cancel
        for i in 0..ITERATIONS {
            let base_time = now + Duration::from_millis(WARMUP + i * 10);
            let h = heap.insert(BenchTicker::new(1), base_time).unwrap();

            // Fire it 3 times
            heap.poll(base_time + Duration::from_millis(1), &mut ctx);
            heap.poll(base_time + Duration::from_millis(2), &mut ctx);
            heap.poll(base_time + Duration::from_millis(3), &mut ctx);

            // Now cancel with original handle
            let start = Instant::now();
            heap.remove(h);
            let elapsed = start.elapsed().as_nanos() as u64;

            hist.record(elapsed).unwrap();
        }

        print_histogram("Cancel After Multiple Fires", &hist);
    }
}
