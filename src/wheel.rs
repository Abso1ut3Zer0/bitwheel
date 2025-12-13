use slab::Slab;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TimerWheelError {
    #[error("timer wheel full: all {0} slots at capacity")]
    WheelFull(usize),
}

/// Slot backed by a fixed-capacity slab.
struct Slot<T> {
    slab: Slab<T>,
    capacity: usize,
}

impl<T> Slot<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            slab: Slab::with_capacity(capacity),
            capacity,
        }
    }

    #[inline]
    fn insert(&mut self, timer: T) -> Result<usize, T> {
        if self.slab.len() >= self.capacity {
            return Err(timer);
        }
        Ok(self.slab.insert(timer))
    }

    #[inline]
    fn remove(&mut self, key: usize) {
        self.slab.remove(key);
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        let key = self.slab.iter().next().map(|(k, _)| k)?;
        self.slab.try_remove(key)
    }

    #[inline]
    fn is_full(&self) -> bool {
        self.slab.len() >= self.capacity
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.slab.is_empty()
    }

    #[inline]
    fn len(&self) -> usize {
        self.slab.len()
    }
}

/// Single-level timer wheel.
pub struct InnerWheel<T> {
    slots: Box<[Slot<T>]>,
    mask: usize,
    slot_capacity: usize,
}

/// Result of a successful insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertResult {
    pub slot: usize,
    pub key: usize,
}

impl<T> InnerWheel<T> {
    /// Create a new wheel.
    ///
    /// `num_slots` and `slot_capacity` are rounded up to the next power of 2.
    pub fn new(num_slots: usize, slot_capacity: usize) -> Self {
        let num_slots = num_slots.next_power_of_two();
        let slot_capacity = slot_capacity.next_power_of_two();

        Self {
            slots: (0..num_slots)
                .map(|_| Slot::with_capacity(slot_capacity))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            mask: num_slots - 1,
            slot_capacity,
        }
    }

    /// Insert at slot, probing forward if full.
    #[inline]
    pub fn insert(&mut self, slot: usize, timer: T) -> Result<InsertResult, TimerWheelError> {
        let num_slots = self.slots.len();
        let mut t = timer;

        for probe in 0..num_slots {
            let idx = (slot + probe) & self.mask;
            let s = unsafe { self.slots.get_unchecked_mut(idx) };

            match s.insert(t) {
                Ok(key) => {
                    return Ok(InsertResult { slot: idx, key });
                }
                Err(rejected) => {
                    t = rejected;
                }
            }
        }

        Err(TimerWheelError::WheelFull(num_slots))
    }

    /// Remove by slot + key. O(1).
    #[inline]
    pub fn remove(&mut self, slot: usize, key: usize) {
        debug_assert!(slot <= self.mask);
        unsafe { self.slots.get_unchecked_mut(slot) }.remove(key)
    }

    /// Pop any entry from slot.
    #[inline]
    pub fn pop_slot(&mut self, slot: usize) -> Option<T> {
        debug_assert!(slot <= self.mask);
        unsafe { self.slots.get_unchecked_mut(slot) }.pop()
    }

    #[inline]
    pub fn slot_is_empty(&self, slot: usize) -> bool {
        debug_assert!(slot <= self.mask);
        unsafe { self.slots.get_unchecked(slot) }.is_empty()
    }

    #[inline]
    pub fn slot_is_full(&self, slot: usize) -> bool {
        debug_assert!(slot <= self.mask);
        unsafe { self.slots.get_unchecked(slot) }.is_full()
    }

    pub fn len(&self) -> usize {
        self.slots.iter().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_empty())
    }

    #[inline]
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn slot_capacity(&self) -> usize {
        self.slot_capacity
    }

    #[inline]
    pub fn mask(&self) -> usize {
        self.mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_of_two_rounding() {
        let wheel: InnerWheel<u32> = InnerWheel::new(100, 3);

        assert_eq!(wheel.num_slots(), 128); // 100 → 128
        assert_eq!(wheel.slot_capacity(), 4); // 3 → 4
        assert_eq!(wheel.mask(), 127);
    }

    #[test]
    fn test_insert_remove() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r = wheel.insert(10, 42).unwrap();
        assert_eq!(r.slot, 10);

        let removed = wheel.remove(r.slot, r.key);
        // assert_eq!(removed, Some(42));
    }

    #[test]
    fn test_probing() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 2);

        wheel.insert(10, 1).unwrap();
        wheel.insert(10, 2).unwrap();

        // Slot 10 full, probes to 11
        let r = wheel.insert(10, 3).unwrap();
        assert_eq!(r.slot, 11);
    }

    #[test]
    fn test_wheel_full() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(4, 1);

        wheel.insert(0, 0).unwrap();
        wheel.insert(1, 1).unwrap();
        wheel.insert(2, 2).unwrap();
        wheel.insert(3, 3).unwrap();

        let result = wheel.insert(0, 99);
        assert!(matches!(result, Err(TimerWheelError::WheelFull(4))));
    }

    #[test]
    fn test_reinsert_after_cancel() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 2);

        let r1 = wheel.insert(10, 1).unwrap();
        let _r2 = wheel.insert(10, 2).unwrap();

        assert!(wheel.slot_is_full(10));

        wheel.remove(r1.slot, r1.key);

        // Can insert again
        let r3 = wheel.insert(10, 3).unwrap();
        assert_eq!(r3.slot, 10);
    }

    #[test]
    fn test_pop_slot() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        wheel.insert(10, 1).unwrap();
        wheel.insert(10, 2).unwrap();

        assert!(wheel.pop_slot(10).is_some());
        assert!(wheel.pop_slot(10).is_some());
        assert!(wheel.pop_slot(10).is_none());
    }
}
