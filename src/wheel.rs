use crate::slot::Slot;
use thiserror::Error;

/// Default maximum probes before giving up on insert.
pub const DEFAULT_MAX_PROBES: usize = 3;

#[derive(Debug, Error)]
pub enum WheelError {
    #[error("wheel full: exhausted {probes} probes starting at slot {slot}")]
    WheelFull { slot: usize, probes: usize },
}

/// Result of a successful insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertResult {
    pub slot: usize,
    pub key: usize,
}

/// Single-level timer wheel backed by fixed-size slots.
///
/// This is a low-level primitive with unsafe API. Caller is responsible for:
/// - Ensuring `slot < num_slots` for all slot-based operations
/// - Ensuring `key < slot_capacity` for all key-based operations
/// - Ensuring entry is occupied when calling `remove`
///
/// Debug assertions help catch violations during development.
pub struct InnerWheel<T> {
    slots: Box<[Slot<T>]>,
    num_slots: usize,
    slot_capacity: usize,
    mask: usize,
    max_probes: usize,
}

impl<T> InnerWheel<T> {
    /// Create a new wheel.
    ///
    /// `num_slots` is rounded up to the next power of 2.
    /// Uses `DEFAULT_MAX_PROBES` (3) for probe limit.
    pub fn new(num_slots: usize, slot_capacity: usize) -> Self {
        Self::with_max_probes(num_slots, slot_capacity, DEFAULT_MAX_PROBES)
    }

    /// Create a new wheel with custom max probes.
    ///
    /// `num_slots` is rounded up to the next power of 2.
    /// `max_probes` is clamped to `min(max_probes, num_slots)`.
    pub fn with_max_probes(num_slots: usize, slot_capacity: usize, max_probes: usize) -> Self {
        let num_slots = num_slots.next_power_of_two();
        let max_probes = max_probes.min(num_slots);

        Self {
            slots: (0..num_slots)
                .map(|_| Slot::with_capacity(slot_capacity))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            num_slots,
            slot_capacity,
            mask: num_slots - 1,
            max_probes,
        }
    }

    /// Insert timer at slot, probing forward if full.
    ///
    /// Returns the actual slot and key on success.
    /// Fails after `max_probes` attempts.
    ///
    /// # Safety
    /// Caller must ensure `slot < num_slots`.
    #[inline]
    pub unsafe fn insert(&mut self, slot: usize, timer: T) -> Result<InsertResult, WheelError> {
        debug_assert!(
            slot < self.num_slots,
            "slot {slot} out of bounds (num_slots: {})",
            self.num_slots
        );

        for probe in 0..self.max_probes {
            let idx = (slot + probe) & self.mask;

            // SAFETY: idx is always < num_slots due to mask
            let s = unsafe { self.slots.get_unchecked_mut(idx) };

            if s.is_full() {
                continue;
            }

            // SAFETY: we just checked slot is not full
            let key = unsafe { s.insert(timer) };
            return Ok(InsertResult { slot: idx, key });
        }

        Err(WheelError::WheelFull {
            slot,
            probes: self.max_probes,
        })
    }

    /// Remove by slot + key.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `slot < num_slots`
    /// - `key < slot_capacity`
    /// - Entry at (slot, key) is occupied
    #[inline]
    pub unsafe fn remove(&mut self, slot: usize, key: usize) -> T {
        debug_assert!(
            slot < self.num_slots,
            "slot {slot} out of bounds (num_slots: {})",
            self.num_slots
        );
        debug_assert!(
            key < self.slot_capacity,
            "key {key} out of bounds (slot_capacity: {})",
            self.slot_capacity
        );

        // SAFETY: caller guarantees bounds and occupancy
        let s = unsafe { self.slots.get_unchecked_mut(slot) };
        unsafe { s.remove(key) }
    }

    /// Try to remove by slot + key. Returns None if not occupied.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `slot < num_slots`
    /// - `key < slot_capacity`
    #[inline]
    pub unsafe fn try_remove(&mut self, slot: usize, key: usize) -> Option<T> {
        debug_assert!(
            slot < self.num_slots,
            "slot {slot} out of bounds (num_slots: {})",
            self.num_slots
        );
        debug_assert!(
            key < self.slot_capacity,
            "key {key} out of bounds (slot_capacity: {})",
            self.slot_capacity
        );

        // SAFETY: caller guarantees bounds
        let s = unsafe { self.slots.get_unchecked_mut(slot) };
        unsafe { s.try_remove(key) }
    }

    /// Pop any entry from slot. Returns None if slot is empty.
    ///
    /// # Safety
    /// Caller must ensure `slot < num_slots`.
    #[inline]
    pub unsafe fn pop_slot(&mut self, slot: usize) -> Option<T> {
        debug_assert!(
            slot < self.num_slots,
            "slot {slot} out of bounds (num_slots: {})",
            self.num_slots
        );

        // SAFETY: caller guarantees slot < num_slots
        let s = unsafe { self.slots.get_unchecked_mut(slot) };
        s.try_pop().map(|(_, value)| value)
    }

    /// Check if a slot is empty.
    ///
    /// # Safety
    /// Caller must ensure `slot < num_slots`.
    #[inline]
    pub unsafe fn slot_is_empty(&self, slot: usize) -> bool {
        debug_assert!(
            slot < self.num_slots,
            "slot {slot} out of bounds (num_slots: {})",
            self.num_slots
        );

        // SAFETY: caller guarantees slot < num_slots
        unsafe { self.slots.get_unchecked(slot) }.is_empty()
    }

    /// Check if a slot is full.
    ///
    /// # Safety
    /// Caller must ensure `slot < num_slots`.
    #[inline]
    pub unsafe fn slot_is_full(&self, slot: usize) -> bool {
        debug_assert!(
            slot < self.num_slots,
            "slot {slot} out of bounds (num_slots: {})",
            self.num_slots
        );

        // SAFETY: caller guarantees slot < num_slots
        unsafe { self.slots.get_unchecked(slot) }.is_full()
    }

    /// Total number of timers across all slots.
    pub fn len(&self) -> usize {
        self.slots.iter().map(|s| s.len()).sum()
    }

    /// Check if wheel has no timers.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_empty())
    }

    /// Check if wheel is completely full.
    pub fn is_full(&self) -> bool {
        self.slots.iter().all(|s| s.is_full())
    }

    #[inline]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    #[inline]
    pub fn slot_capacity(&self) -> usize {
        self.slot_capacity
    }

    #[inline]
    pub fn max_probes(&self) -> usize {
        self.max_probes
    }

    #[inline]
    pub fn total_capacity(&self) -> usize {
        self.num_slots * self.slot_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    // ==================== Construction ====================

    #[test]
    fn test_new_empty() {
        let wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        assert!(wheel.is_empty());
        assert!(!wheel.is_full());
        assert_eq!(wheel.len(), 0);
        assert_eq!(wheel.num_slots(), 256);
        assert_eq!(wheel.slot_capacity(), 4);
        assert_eq!(wheel.total_capacity(), 1024);
        assert_eq!(wheel.max_probes(), DEFAULT_MAX_PROBES);
    }

    #[test]
    fn test_with_max_probes() {
        let wheel: InnerWheel<u32> = InnerWheel::with_max_probes(256, 4, 16);

        assert_eq!(wheel.max_probes(), 16);
    }

    #[test]
    fn test_max_probes_clamped() {
        // max_probes > num_slots should be clamped
        let wheel: InnerWheel<u32> = InnerWheel::with_max_probes(8, 4, 100);

        assert_eq!(wheel.max_probes(), 8);
    }

    #[test]
    fn test_power_of_two_rounding() {
        let wheel: InnerWheel<u32> = InnerWheel::new(100, 4);

        assert_eq!(wheel.num_slots(), 128); // 100 → 128
    }

    #[test]
    fn test_already_power_of_two() {
        let wheel: InnerWheel<u32> = InnerWheel::new(64, 4);

        assert_eq!(wheel.num_slots(), 64);
    }

    #[test]
    fn test_small_wheel() {
        let wheel: InnerWheel<u32> = InnerWheel::new(4, 2);

        assert_eq!(wheel.num_slots(), 4);
        assert_eq!(wheel.slot_capacity(), 2);
        assert_eq!(wheel.total_capacity(), 8);
    }

    // ==================== Basic Insert ====================

    #[test]
    fn test_insert_single() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let result = unsafe { wheel.insert(10, 42) }.unwrap();

        assert_eq!(result.slot, 10);
        assert_eq!(result.key, 0);
        assert_eq!(wheel.len(), 1);
        assert!(!wheel.is_empty());
    }

    #[test]
    fn test_insert_multiple_same_slot() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(10, 100) }.unwrap();
        let r1 = unsafe { wheel.insert(10, 200) }.unwrap();
        let r2 = unsafe { wheel.insert(10, 300) }.unwrap();
        let r3 = unsafe { wheel.insert(10, 400) }.unwrap();

        // All in same slot, sequential keys
        assert_eq!(r0, InsertResult { slot: 10, key: 0 });
        assert_eq!(r1, InsertResult { slot: 10, key: 1 });
        assert_eq!(r2, InsertResult { slot: 10, key: 2 });
        assert_eq!(r3, InsertResult { slot: 10, key: 3 });
        assert_eq!(wheel.len(), 4);
    }

    #[test]
    fn test_insert_different_slots() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(0, 100) }.unwrap();
        let r1 = unsafe { wheel.insert(50, 200) }.unwrap();
        let r2 = unsafe { wheel.insert(100, 300) }.unwrap();
        let r3 = unsafe { wheel.insert(200, 400) }.unwrap();

        assert_eq!(r0.slot, 0);
        assert_eq!(r1.slot, 50);
        assert_eq!(r2.slot, 100);
        assert_eq!(r3.slot, 200);
        assert_eq!(wheel.len(), 4);
    }

    // ==================== Probing Behavior ====================

    #[test]
    fn test_probe_to_next_slot() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 2);

        // Fill slot 10
        unsafe {
            wheel.insert(10, 1).unwrap();
            wheel.insert(10, 2).unwrap();
        }

        // Next insert should probe to slot 11
        let result = unsafe { wheel.insert(10, 3) }.unwrap();

        assert_eq!(result.slot, 11);
        assert_eq!(result.key, 0);
    }

    #[test]
    fn test_probe_multiple_slots() {
        let mut wheel: InnerWheel<u32> = InnerWheel::with_max_probes(256, 1, 16);

        // Fill slots 10, 11, 12
        unsafe {
            wheel.insert(10, 1).unwrap();
            wheel.insert(11, 2).unwrap();
            wheel.insert(12, 3).unwrap();
        }

        // Insert at 10 should probe to 13
        let result = unsafe { wheel.insert(10, 4) }.unwrap();

        assert_eq!(result.slot, 13);
    }

    #[test]
    fn test_probe_wrap_around() {
        let mut wheel: InnerWheel<u32> = InnerWheel::with_max_probes(4, 1, 4);

        // Fill slots 2, 3
        unsafe {
            wheel.insert(2, 1).unwrap();
            wheel.insert(3, 2).unwrap();
        }

        // Insert at 2 should wrap to slot 0
        let result = unsafe { wheel.insert(2, 3) }.unwrap();

        assert_eq!(result.slot, 0);
    }

    // ==================== Max Probes Exceeded ====================

    #[test]
    fn test_max_probes_exceeded() {
        let mut wheel: InnerWheel<u32> = InnerWheel::with_max_probes(256, 1, 4);

        // Fill slots 10, 11, 12, 13
        unsafe {
            wheel.insert(10, 1).unwrap();
            wheel.insert(11, 2).unwrap();
            wheel.insert(12, 3).unwrap();
            wheel.insert(13, 4).unwrap();
        }

        // Insert at 10 should fail after 4 probes
        let result = unsafe { wheel.insert(10, 5) };

        assert!(matches!(
            result,
            Err(WheelError::WheelFull {
                slot: 10,
                probes: 4
            })
        ));
    }

    #[test]
    fn test_max_probes_with_gaps() {
        let mut wheel: InnerWheel<u32> = InnerWheel::with_max_probes(256, 1, 4);

        // Fill slots 10, 11, 12, 13 but leave 14 empty
        unsafe {
            wheel.insert(10, 1).unwrap();
            wheel.insert(11, 2).unwrap();
            wheel.insert(12, 3).unwrap();
            wheel.insert(13, 4).unwrap();
        }

        // Insert at 10 should fail - can't reach slot 14
        let result = unsafe { wheel.insert(10, 5) };
        assert!(result.is_err());

        // But insert at 11 can reach 14
        let result = unsafe { wheel.insert(11, 5) }.unwrap();
        assert_eq!(result.slot, 14);
    }

    #[test]
    fn test_wheel_still_usable_after_probe_failure() {
        let mut wheel: InnerWheel<u32> = InnerWheel::with_max_probes(256, 1, 2);

        // Fill slots 10, 11
        unsafe {
            wheel.insert(10, 1).unwrap();
            wheel.insert(11, 2).unwrap();
        }

        // Fail at slot 10
        let result = unsafe { wheel.insert(10, 3) };
        assert!(result.is_err());

        // But can still insert at slot 12
        let result = unsafe { wheel.insert(12, 3) }.unwrap();
        assert_eq!(result.slot, 12);
    }

    // ==================== Remove ====================

    #[test]
    fn test_remove_single() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r = unsafe { wheel.insert(10, 42) }.unwrap();
        let value = unsafe { wheel.remove(r.slot, r.key) };

        assert_eq!(value, 42);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_remove_multiple_same_slot() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(10, 100) }.unwrap();
        let r1 = unsafe { wheel.insert(10, 200) }.unwrap();
        let r2 = unsafe { wheel.insert(10, 300) }.unwrap();

        // Remove in different order
        assert_eq!(unsafe { wheel.remove(r1.slot, r1.key) }, 200);
        assert_eq!(unsafe { wheel.remove(r0.slot, r0.key) }, 100);
        assert_eq!(unsafe { wheel.remove(r2.slot, r2.key) }, 300);

        assert!(wheel.is_empty());
    }

    #[test]
    fn test_remove_different_slots() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(10, 100) }.unwrap();
        let r1 = unsafe { wheel.insert(50, 200) }.unwrap();
        let r2 = unsafe { wheel.insert(100, 300) }.unwrap();

        assert_eq!(unsafe { wheel.remove(r1.slot, r1.key) }, 200);
        assert_eq!(unsafe { wheel.remove(r2.slot, r2.key) }, 300);
        assert_eq!(unsafe { wheel.remove(r0.slot, r0.key) }, 100);

        assert!(wheel.is_empty());
    }

    #[test]
    fn test_remove_probed_entry() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 1);

        // Fill slot 10, next goes to 11
        unsafe { wheel.insert(10, 100).unwrap() };
        let r = unsafe { wheel.insert(10, 200) }.unwrap();

        assert_eq!(r.slot, 11);

        // Remove the probed entry
        let value = unsafe { wheel.remove(r.slot, r.key) };
        assert_eq!(value, 200);
    }

    // ==================== Try Remove ====================

    #[test]
    fn test_try_remove_occupied() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r = unsafe { wheel.insert(10, 42) }.unwrap();
        let result = unsafe { wheel.try_remove(r.slot, r.key) };

        assert_eq!(result, Some(42));
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_try_remove_vacant() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r = unsafe { wheel.insert(10, 42) }.unwrap();
        unsafe { wheel.remove(r.slot, r.key) };

        // Second try_remove should return None
        let result = unsafe { wheel.try_remove(r.slot, r.key) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_try_remove_never_occupied() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        // Insert at slot 10
        unsafe { wheel.insert(10, 42).unwrap() };

        // Try remove from different slot
        let result = unsafe { wheel.try_remove(20, 0) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_try_remove_double() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r = unsafe { wheel.insert(10, 42) }.unwrap();

        let first = unsafe { wheel.try_remove(r.slot, r.key) };
        let second = unsafe { wheel.try_remove(r.slot, r.key) };

        assert_eq!(first, Some(42));
        assert_eq!(second, None);
    }

    // ==================== Pop Slot ====================

    #[test]
    fn test_pop_slot_empty() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let result = unsafe { wheel.pop_slot(10) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_pop_slot_single() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        unsafe { wheel.insert(10, 42).unwrap() };

        let result = unsafe { wheel.pop_slot(10) };
        assert_eq!(result, Some(42));

        let result = unsafe { wheel.pop_slot(10) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_pop_slot_multiple() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        unsafe {
            wheel.insert(10, 100).unwrap();
            wheel.insert(10, 200).unwrap();
            wheel.insert(10, 300).unwrap();
        }

        let mut values = vec![];
        while let Some(v) = unsafe { wheel.pop_slot(10) } {
            values.push(v);
        }

        assert_eq!(values.len(), 3);
        assert!(values.contains(&100));
        assert!(values.contains(&200));
        assert!(values.contains(&300));
    }

    #[test]
    fn test_pop_slot_with_gaps() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(10, 100) }.unwrap();
        let _r1 = unsafe { wheel.insert(10, 200) }.unwrap();
        let r2 = unsafe { wheel.insert(10, 300) }.unwrap();

        // Create gaps
        unsafe {
            wheel.remove(r0.slot, r0.key);
            wheel.remove(r2.slot, r2.key);
        }

        // Only 200 should remain
        let result = unsafe { wheel.pop_slot(10) };
        assert_eq!(result, Some(200));

        let result = unsafe { wheel.pop_slot(10) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_pop_slot_doesnt_affect_other_slots() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        unsafe {
            wheel.insert(10, 100).unwrap();
            wheel.insert(20, 200).unwrap();
        }

        // Pop from slot 10
        assert_eq!(unsafe { wheel.pop_slot(10) }, Some(100));
        assert_eq!(unsafe { wheel.pop_slot(10) }, None);

        // Slot 20 still has its value
        assert_eq!(unsafe { wheel.pop_slot(20) }, Some(200));
    }

    // ==================== Slot State ====================

    #[test]
    fn test_slot_is_empty() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        assert!(unsafe { wheel.slot_is_empty(10) });

        unsafe { wheel.insert(10, 42).unwrap() };

        assert!(!unsafe { wheel.slot_is_empty(10) });
        assert!(unsafe { wheel.slot_is_empty(11) });
    }

    #[test]
    fn test_slot_is_full() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 2);

        assert!(!unsafe { wheel.slot_is_full(10) });

        unsafe {
            wheel.insert(10, 1).unwrap();
        }
        assert!(!unsafe { wheel.slot_is_full(10) });

        unsafe {
            wheel.insert(10, 2).unwrap();
        }
        assert!(unsafe { wheel.slot_is_full(10) });
    }

    #[test]
    fn test_slot_state_after_remove() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 2);

        let r0 = unsafe { wheel.insert(10, 1) }.unwrap();
        let r1 = unsafe { wheel.insert(10, 2) }.unwrap();

        assert!(unsafe { wheel.slot_is_full(10) });

        unsafe { wheel.remove(r0.slot, r0.key) };

        assert!(!unsafe { wheel.slot_is_full(10) });
        assert!(!unsafe { wheel.slot_is_empty(10) });

        unsafe { wheel.remove(r1.slot, r1.key) };

        assert!(unsafe { wheel.slot_is_empty(10) });
    }

    // ==================== Reinsert After Remove ====================

    #[test]
    fn test_reinsert_same_slot_after_remove() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 2);

        let r0 = unsafe { wheel.insert(10, 100) }.unwrap();
        let r1 = unsafe { wheel.insert(10, 200) }.unwrap();

        assert!(unsafe { wheel.slot_is_full(10) });

        // Remove one
        unsafe { wheel.remove(r0.slot, r0.key) };

        // Reinsert should go to same slot
        let r2 = unsafe { wheel.insert(10, 300) }.unwrap();
        assert_eq!(r2.slot, 10);

        // Values correct
        assert_eq!(unsafe { wheel.remove(r1.slot, r1.key) }, 200);
        assert_eq!(unsafe { wheel.remove(r2.slot, r2.key) }, 300);
    }

    #[test]
    fn test_reinsert_reuses_key() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(10, 100) }.unwrap();
        let r1 = unsafe { wheel.insert(10, 200) }.unwrap();

        // Remove first
        unsafe { wheel.remove(r0.slot, r0.key) };

        // Reinsert should reuse key (LIFO free list)
        let r2 = unsafe { wheel.insert(10, 300) }.unwrap();
        assert_eq!(r2.slot, r0.slot);
        assert_eq!(r2.key, r0.key);

        // Remove second
        unsafe { wheel.remove(r1.slot, r1.key) };

        // Next reinsert reuses r1.key
        let r3 = unsafe { wheel.insert(10, 400) }.unwrap();
        assert_eq!(r3.key, r1.key);
    }

    // ==================== Complex Types ====================

    #[test]
    fn test_string_values() {
        let mut wheel: InnerWheel<String> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(10, "hello".to_string()) }.unwrap();
        let r1 = unsafe { wheel.insert(10, "world".to_string()) }.unwrap();

        assert_eq!(unsafe { wheel.remove(r0.slot, r0.key) }, "hello");
        assert_eq!(unsafe { wheel.remove(r1.slot, r1.key) }, "world");
    }

    #[test]
    fn test_vec_values() {
        let mut wheel: InnerWheel<Vec<i32>> = InnerWheel::new(256, 4);

        let r0 = unsafe { wheel.insert(10, vec![1, 2, 3]) }.unwrap();
        let r1 = unsafe { wheel.insert(20, vec![4, 5, 6]) }.unwrap();

        assert_eq!(unsafe { wheel.remove(r1.slot, r1.key) }, vec![4, 5, 6]);
        assert_eq!(unsafe { wheel.remove(r0.slot, r0.key) }, vec![1, 2, 3]);
    }

    // ==================== Drop Behavior ====================

    #[test]
    fn test_drop_with_occupied() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<usize>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        {
            let mut wheel: InnerWheel<DropCounter> = InnerWheel::new(256, 4);

            unsafe {
                wheel
                    .insert(10, DropCounter(Rc::clone(&drop_count)))
                    .unwrap();
                wheel
                    .insert(10, DropCounter(Rc::clone(&drop_count)))
                    .unwrap();
                wheel
                    .insert(20, DropCounter(Rc::clone(&drop_count)))
                    .unwrap();
            }

            assert_eq!(drop_count.get(), 0);
        }

        // All 3 should be dropped
        assert_eq!(drop_count.get(), 3);
    }

    #[test]
    fn test_drop_with_mixed_slots() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<usize>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        {
            let mut wheel: InnerWheel<DropCounter> = InnerWheel::new(256, 4);

            let r0 = unsafe { wheel.insert(10, DropCounter(Rc::clone(&drop_count))) }.unwrap();
            unsafe {
                wheel
                    .insert(20, DropCounter(Rc::clone(&drop_count)))
                    .unwrap()
            };
            let r2 = unsafe { wheel.insert(30, DropCounter(Rc::clone(&drop_count))) }.unwrap();

            // Remove two
            unsafe {
                wheel.remove(r0.slot, r0.key);
                wheel.remove(r2.slot, r2.key);
            }

            // 2 dropped from remove
            assert_eq!(drop_count.get(), 2);
        }

        // 1 more dropped when wheel drops
        assert_eq!(drop_count.get(), 3);
    }

    // ==================== Stress Tests ====================

    #[test]
    fn test_fill_and_drain() {
        let mut wheel: InnerWheel<u32> = InnerWheel::with_max_probes(256, 4, 256);

        // Fill completely
        let mut results = vec![];
        for i in 0..1024 {
            let r = unsafe { wheel.insert(i % 256, i as u32) }.unwrap();
            results.push(r);
        }

        assert!(wheel.is_full());
        assert_eq!(wheel.len(), 1024);

        // Drain via remove
        for (i, r) in results.iter().enumerate() {
            let value = unsafe { wheel.remove(r.slot, r.key) };
            assert_eq!(value, i as u32);
        }

        assert!(wheel.is_empty());
    }

    #[test]
    fn test_fill_and_pop() {
        let mut wheel: InnerWheel<u32> = InnerWheel::with_max_probes(64, 4, 64);

        // Fill
        for i in 0..256 {
            unsafe { wheel.insert(i % 64, i as u32).unwrap() };
        }

        assert!(wheel.is_full());

        // Pop all slots
        let mut count = 0;
        for slot in 0..64 {
            while let Some(_) = unsafe { wheel.pop_slot(slot) } {
                count += 1;
            }
        }

        assert_eq!(count, 256);
        assert!(wheel.is_empty());
    }

    #[test]
    fn test_alternating_insert_remove() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(256, 4);

        for i in 0..1000u32 {
            let r = unsafe { wheel.insert(i as usize % 256, i) }.unwrap();
            let value = unsafe { wheel.remove(r.slot, r.key) };
            assert_eq!(value, i);
        }

        assert!(wheel.is_empty());
    }

    // ==================== Len and Capacity ====================

    #[test]
    fn test_len_tracking() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(64, 4);

        assert_eq!(wheel.len(), 0);

        let mut results = vec![];
        for i in 0..50 {
            let r = unsafe { wheel.insert(i % 64, i as u32) }.unwrap();
            results.push(r);
            assert_eq!(wheel.len(), i + 1);
        }

        for (i, r) in results.iter().enumerate() {
            unsafe { wheel.remove(r.slot, r.key) };
            assert_eq!(wheel.len(), 49 - i);
        }
    }

    #[test]
    fn test_len_with_try_remove() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(64, 4);

        let r = unsafe { wheel.insert(10, 42) }.unwrap();
        assert_eq!(wheel.len(), 1);

        // First try_remove decrements
        unsafe { wheel.try_remove(r.slot, r.key) };
        assert_eq!(wheel.len(), 0);

        // Second try_remove doesn't decrement
        unsafe { wheel.try_remove(r.slot, r.key) };
        assert_eq!(wheel.len(), 0);
    }

    #[test]
    fn test_len_with_pop() {
        let mut wheel: InnerWheel<u32> = InnerWheel::new(64, 4);

        unsafe {
            wheel.insert(10, 1).unwrap();
            wheel.insert(10, 2).unwrap();
            wheel.insert(10, 3).unwrap();
        }

        assert_eq!(wheel.len(), 3);

        unsafe { wheel.pop_slot(10) };
        assert_eq!(wheel.len(), 2);

        unsafe { wheel.pop_slot(10) };
        assert_eq!(wheel.len(), 1);

        unsafe { wheel.pop_slot(10) };
        assert_eq!(wheel.len(), 0);

        // Pop on empty doesn't affect len
        unsafe { wheel.pop_slot(10) };
        assert_eq!(wheel.len(), 0);
    }
}
