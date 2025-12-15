use crate::{DEFAULT_SLOT_CAP, slot::Slot};
use std::{
    cell::{Cell, UnsafeCell},
    fmt::Debug,
};

pub const NUM_SLOTS: usize = 64;
pub const SLOT_MASK: usize = 63;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("no available slot")]
pub struct NoAvailableSlot;

#[derive(Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("wheel capacity exceeded")]
pub struct InsertError<T>(pub T);

impl<T> Debug for InsertError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("wheel capacity exceeded")
    }
}

/// A single gear (level) in the timer wheel.
///
/// Fixed at 64 slots. Uses interior mutability with debug-mode
/// lock tracking to ensure safe concurrent slot access.
pub struct Gear<T, const SLOT_CAP: usize = DEFAULT_SLOT_CAP> {
    slots: [UnsafeCell<Slot<T, SLOT_CAP>>; NUM_SLOTS],
    occupied: Cell<u64>,
    #[cfg(debug_assertions)]
    locked: Cell<u64>,
}

#[allow(unused)]
impl<T, const SLOT_CAP: usize> Gear<T, SLOT_CAP> {
    /// Create a new gear with the given slot capacity.
    pub fn new() -> Self {
        // Create array of UnsafeCell<Slot<T>>
        let slots: [UnsafeCell<Slot<T, SLOT_CAP>>; NUM_SLOTS] =
            std::array::from_fn(|_| UnsafeCell::new(Slot::new()));

        Self {
            slots,
            occupied: Cell::new(0),
            #[cfg(debug_assertions)]
            locked: Cell::new(0),
        }
    }

    /// Acquire exclusive access to a slot.
    ///
    /// # Panics
    /// Debug builds panic if slot is already acquired.
    #[inline(always)]
    pub fn acquire(&self, slot: usize) -> SlotGuard<'_, T, SLOT_CAP> {
        debug_assert!(slot < NUM_SLOTS, "slot {slot} out of bounds");

        #[cfg(debug_assertions)]
        {
            let bit = 1u64 << slot;
            let locked = self.locked.get();
            assert!(locked & bit == 0, "slot {slot} already acquired");
            self.locked.set(locked | bit);
        }

        SlotGuard { gear: self, slot }
    }

    /// Find and acquire the next available (non-full) slot.
    ///
    /// Probes forward from `target`.
    /// Returns error if no slot found within `max_probes`.
    ///
    /// # Panics
    /// Debug builds panic if a probed slot is already acquired.
    pub fn acquire_next_available(
        &self,
        target: usize,
        max_probes: usize,
    ) -> Result<SlotGuard<'_, T, SLOT_CAP>, NoAvailableSlot> {
        debug_assert!(target < NUM_SLOTS, "target {target} out of bounds");

        for probe in 0..max_probes {
            let slot = (target + probe) & SLOT_MASK;

            // SAFETY: We're only reading is_full, no mutation
            let is_full = unsafe { (*self.slots[slot].get()).is_full() };

            if !is_full {
                return Ok(self.acquire(slot));
            }
        }

        Err(NoAvailableSlot)
    }

    /// Find and acquire the next available (non-full) slot,
    /// excluding a specific slot on re-entrancy.
    ///
    /// Probes forward from `target`, skipping `excluded` slot.
    /// Returns error if no slot found within `max_probes`.
    ///
    /// # Panics
    /// Debug builds panic if a probed slot is already acquired.
    pub fn acquire_next_available_excluding(
        &self,
        excluded: usize,
        target: usize,
        max_probes: usize,
    ) -> Result<SlotGuard<'_, T, SLOT_CAP>, NoAvailableSlot> {
        debug_assert!(target < NUM_SLOTS, "target {target} out of bounds");
        debug_assert!(excluded < NUM_SLOTS, "excluded {excluded} out of bounds");

        for probe in 0..max_probes {
            let slot = (target + probe) & SLOT_MASK;

            if slot == excluded {
                continue;
            }

            // Check if slot has room (peek without acquiring)
            // SAFETY: We're only reading is_full, no mutation
            let is_full = unsafe { (*self.slots[slot].get()).is_full() };

            if !is_full {
                return Ok(self.acquire(slot));
            }
        }

        Err(NoAvailableSlot)
    }

    /// Check if a slot has entries (via bitmap).
    #[inline(always)]
    pub fn is_slot_occupied(&self, slot: usize) -> bool {
        debug_assert!(slot < NUM_SLOTS, "slot {slot} out of bounds");
        (self.occupied.get() & (1u64 << slot)) != 0
    }

    /// Check if entire gear is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.occupied.get() == 0
    }

    /// Get the occupied bitmap (for iteration).
    #[inline(always)]
    pub fn occupied_bitmap(&self) -> u64 {
        self.occupied.get()
    }

    /// Number of slots (always 64).
    #[inline(always)]
    pub const fn num_slots(&self) -> usize {
        NUM_SLOTS
    }

    /// Slot mask (always 63).
    #[inline(always)]
    pub const fn slot_mask(&self) -> usize {
        SLOT_MASK
    }

    #[inline(always)]
    fn set_occupied(&self, slot: usize) {
        let bit = 1u64 << slot;
        self.occupied.set(self.occupied.get() | bit);
    }

    #[inline(always)]
    fn clear_occupied_if_empty(&self, slot: usize, is_empty: bool) {
        if is_empty {
            let bit = 1u64 << slot;
            self.occupied.set(self.occupied.get() & !bit);
        }
    }
}

/// Exclusive access to a slot within a gear.
///
/// Provides safe methods to manipulate the slot.
/// Automatically releases the lock on drop (debug builds).
pub struct SlotGuard<'a, T, const SLOT_CAP: usize> {
    gear: &'a Gear<T, SLOT_CAP>,
    slot: usize,
}

#[allow(unused)]
impl<'a, T, const SLOT_CAP: usize> SlotGuard<'a, T, SLOT_CAP> {
    /// Which slot this guard holds.
    #[inline(always)]
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Insert a value into the slot. Returns the key.
    ///
    /// # Panics
    /// Panics if slot is full.
    #[inline(always)]
    pub fn insert(&self, value: T) -> usize {
        let slot = self.slot_mut();
        assert!(!slot.is_full(), "slot {} is full", self.slot);

        // SAFETY: Slot::insert requires not full, which we just checked
        let key = unsafe { slot.insert(value) };
        self.gear.set_occupied(self.slot);
        key
    }

    /// Try to insert a value. Returns Err if slot is full.
    #[inline(always)]
    pub fn try_insert(&self, value: T) -> Result<usize, InsertError<T>> {
        let slot = self.slot_mut();
        if slot.is_full() {
            return Err(InsertError(value));
        }

        // SAFETY: Slot::insert requires not full, which we just checked
        let key = unsafe { slot.insert(value) };
        self.gear.set_occupied(self.slot);
        Ok(key)
    }

    /// Pop any entry from the slot.
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let slot = self.slot_mut();
        let result = slot.try_pop().map(|(_, v)| v);
        self.gear
            .clear_occupied_if_empty(self.slot, slot.is_empty());
        result
    }

    /// Try to remove a specific entry by key.
    #[inline(always)]
    pub fn try_remove(&self, key: usize) -> Option<T> {
        let slot = self.slot_mut();
        // SAFETY: try_remove requires key < capacity,
        // but handles invalid key gracefully
        let result = unsafe { slot.try_remove(key) };
        self.gear
            .clear_occupied_if_empty(self.slot, slot.is_empty());
        result
    }

    /// Try to remove a specific entry by key.
    #[inline(always)]
    pub fn remove(&self, key: usize) -> T {
        let slot = self.slot_mut();
        // SAFETY: remove requires key < capacity,
        // otherwise it will panic
        let result = unsafe { slot.remove(key) };
        self.gear
            .clear_occupied_if_empty(self.slot, slot.is_empty());
        result
    }

    /// Check if slot is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.slot_ref().is_empty()
    }

    /// Check if slot is full.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.slot_ref().is_full()
    }

    /// Number of entries in slot.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slot_ref().len()
    }

    #[inline(always)]
    fn slot_ref(&self) -> &Slot<T, SLOT_CAP> {
        // SAFETY: We have exclusive access via the guard
        unsafe { &*self.gear.slots[self.slot].get() }
    }

    #[inline(always)]
    fn slot_mut(&self) -> &mut Slot<T, SLOT_CAP> {
        // SAFETY: We have exclusive access via the guard
        unsafe { &mut *self.gear.slots[self.slot].get() }
    }
}

impl<T, const SLOT_CAP: usize> Drop for SlotGuard<'_, T, SLOT_CAP> {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        {
            let bit = 1u64 << self.slot;
            self.gear.locked.set(self.gear.locked.get() & !bit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    // ==================== Construction ====================

    #[test]
    fn test_new_gear() {
        let gear: Gear<u32, 4> = Gear::new();
        assert!(gear.is_empty());
        assert_eq!(gear.occupied_bitmap(), 0);
        assert_eq!(gear.num_slots(), 64);
        assert_eq!(gear.slot_mask(), 63);
    }

    // ==================== Basic Acquire/Insert/Pop ====================

    #[test]
    fn test_acquire_insert_pop() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(10);
        assert_eq!(guard.slot(), 10);
        assert!(guard.is_empty());
        assert_eq!(guard.len(), 0);

        let key = guard.insert(42);
        assert_eq!(key, 0);
        assert!(!guard.is_empty());
        assert_eq!(guard.len(), 1);
        assert!(gear.is_slot_occupied(10));

        let value = guard.pop();
        assert_eq!(value, Some(42));
        assert!(guard.is_empty());
        assert_eq!(guard.len(), 0);
        assert!(!gear.is_slot_occupied(10));
    }

    #[test]
    fn test_insert_multiple_same_slot() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(5);

        let _k0 = guard.insert(100);
        let _k1 = guard.insert(200);
        let _k2 = guard.insert(300);
        let _k3 = guard.insert(400);

        assert_eq!(guard.len(), 4);
        assert!(guard.is_full());
        assert!(gear.is_slot_occupied(5));

        // Pop returns in LIFO order (from occupied list head)
        assert_eq!(guard.pop(), Some(400));
        assert_eq!(guard.pop(), Some(300));
        assert_eq!(guard.pop(), Some(200));
        assert_eq!(guard.pop(), Some(100));
        assert_eq!(guard.pop(), None);

        assert!(guard.is_empty());
        assert!(!gear.is_slot_occupied(5));
    }

    #[test]
    fn test_multiple_slots_independent() {
        let gear: Gear<u32, 4> = Gear::new();

        // Insert into different slots
        {
            let guard = gear.acquire(5);
            guard.insert(100);
            guard.insert(101);
        }

        {
            let guard = gear.acquire(20);
            guard.insert(200);
        }

        {
            let guard = gear.acquire(63);
            guard.insert(300);
            guard.insert(301);
            guard.insert(302);
        }

        assert!(gear.is_slot_occupied(5));
        assert!(gear.is_slot_occupied(20));
        assert!(gear.is_slot_occupied(63));
        assert!(!gear.is_slot_occupied(0));
        assert!(!gear.is_slot_occupied(10));

        // Verify each slot has correct data
        {
            let guard = gear.acquire(5);
            assert_eq!(guard.len(), 2);
        }
        {
            let guard = gear.acquire(20);
            assert_eq!(guard.len(), 1);
        }
        {
            let guard = gear.acquire(63);
            assert_eq!(guard.len(), 3);
        }
    }

    // ==================== try_insert ====================

    #[test]
    fn test_try_insert_success() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(10);
        let result = guard.try_insert(42);
        assert!(result.is_ok());
        assert_eq!(guard.len(), 1);
    }

    #[test]
    fn test_try_insert_full() {
        let gear: Gear<u32, 2> = Gear::new();

        let guard = gear.acquire(10);
        guard.insert(1);
        guard.insert(2);
        assert!(guard.is_full());

        let result = guard.try_insert(3);
        assert!(matches!(result, Err(InsertError(3))));
    }

    // ==================== try_remove ====================

    #[test]
    fn test_try_remove_success() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(5);
        let key = guard.insert(42);

        let value = guard.try_remove(key);
        assert_eq!(value, Some(42));
        assert!(guard.is_empty());
    }

    #[test]
    fn test_try_remove_wrong_key() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(5);
        let key = guard.insert(42);

        // Remove with different key
        let other_key = if key == 0 { 1 } else { 0 };
        let value = guard.try_remove(other_key);
        assert_eq!(value, None);

        // Original still there
        assert_eq!(guard.len(), 1);
    }

    #[test]
    fn test_try_remove_double() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(5);
        let key = guard.insert(42);

        let first = guard.try_remove(key);
        let second = guard.try_remove(key);

        assert_eq!(first, Some(42));
        assert_eq!(second, None);
    }

    #[test]
    fn test_try_remove_specific_from_multiple() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(5);
        let k0 = guard.insert(100);
        let k1 = guard.insert(200);
        let k2 = guard.insert(300);

        // Remove middle one
        assert_eq!(guard.try_remove(k1), Some(200));
        assert_eq!(guard.len(), 2);

        // Remove first
        assert_eq!(guard.try_remove(k0), Some(100));
        assert_eq!(guard.len(), 1);

        // Remove last
        assert_eq!(guard.try_remove(k2), Some(300));
        assert!(guard.is_empty());
    }

    // ==================== Bitmap Tracking ====================

    #[test]
    fn test_bitmap_set_on_insert() {
        let gear: Gear<u32, 4> = Gear::new();

        assert_eq!(gear.occupied_bitmap(), 0);

        {
            let guard = gear.acquire(0);
            guard.insert(1);
        }
        assert_eq!(gear.occupied_bitmap(), 0b1);

        {
            let guard = gear.acquire(5);
            guard.insert(1);
        }
        assert_eq!(gear.occupied_bitmap(), 0b100001);

        {
            let guard = gear.acquire(63);
            guard.insert(1);
        }
        assert_eq!(gear.occupied_bitmap(), 0b1 << 63 | 0b100001);
    }

    #[test]
    fn test_bitmap_cleared_on_empty() {
        let gear: Gear<u32, 4> = Gear::new();

        {
            let guard = gear.acquire(10);
            guard.insert(1);
            guard.insert(2);
        }
        assert!(gear.is_slot_occupied(10));

        {
            let guard = gear.acquire(10);
            guard.pop();
            // Still has one entry
            assert!(gear.is_slot_occupied(10));

            guard.pop();
            // Now empty
            assert!(!gear.is_slot_occupied(10));
        }
    }

    #[test]
    fn test_bitmap_cleared_on_try_remove() {
        let gear: Gear<u32, 4> = Gear::new();

        let key;
        {
            let guard = gear.acquire(10);
            key = guard.insert(42);
        }
        assert!(gear.is_slot_occupied(10));

        {
            let guard = gear.acquire(10);
            guard.try_remove(key);
        }
        assert!(!gear.is_slot_occupied(10));
    }

    // ==================== Guard Lifecycle ====================

    #[test]
    fn test_guard_drops_releases_lock() {
        let gear: Gear<u32, 4> = Gear::new();

        {
            let guard = gear.acquire(10);
            guard.insert(42);
        }

        // Should be able to acquire again after drop
        let guard = gear.acquire(10);
        assert_eq!(guard.pop(), Some(42));
    }

    #[test]
    fn test_multiple_guards_different_slots() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard1 = gear.acquire(10);
        let guard2 = gear.acquire(20);
        let guard3 = gear.acquire(30);

        guard1.insert(100);
        guard2.insert(200);
        guard3.insert(300);

        assert_eq!(guard1.len(), 1);
        assert_eq!(guard2.len(), 1);
        assert_eq!(guard3.len(), 1);
    }

    #[test]
    fn test_reacquire_after_operations() {
        let gear: Gear<u32, 4> = Gear::new();

        for round in 0..10 {
            let guard = gear.acquire(5);
            let key = guard.insert(round);
            assert_eq!(guard.try_remove(key), Some(round));
        }

        assert!(!gear.is_slot_occupied(5));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "already acquired")]
    fn test_double_acquire_panics() {
        let gear: Gear<u32, 4> = Gear::new();

        let _guard1 = gear.acquire(10);
        let _guard2 = gear.acquire(10); // panic!
    }

    // ==================== acquire_next_available ====================

    #[test]
    fn test_acquire_next_available_empty_gear() {
        let gear: Gear<u32, 4> = Gear::new();

        // excluded=0, target=10 - should get slot 10
        let guard = gear.acquire_next_available_excluding(0, 10, 3).unwrap();
        assert_eq!(guard.slot(), 10);
    }

    #[test]
    fn test_acquire_next_available_probes_forward() {
        let gear: Gear<u32, 1> = Gear::new(); // capacity 1

        // Fill slots 10, 11
        {
            let guard = gear.acquire(10);
            guard.insert(1);
        }
        {
            let guard = gear.acquire(11);
            guard.insert(1);
        }

        // excluded=0, target=10, should probe to 12
        let guard = gear.acquire_next_available_excluding(0, 10, 5).unwrap();
        assert_eq!(guard.slot(), 12);
    }

    #[test]
    fn test_acquire_next_available_wraps_around() {
        let gear: Gear<u32, 1> = Gear::new();

        // Fill slot 63
        {
            let guard = gear.acquire(63);
            guard.insert(1);
        }

        // excluded=1, target=63, should wrap to 0
        let guard = gear.acquire_next_available_excluding(1, 63, 3).unwrap();
        assert_eq!(guard.slot(), 0);
    }

    #[test]
    fn test_acquire_next_available_finds_last_probe() {
        let gear: Gear<u32, 1> = Gear::new();

        // Fill slots 10, 11, but not 12
        gear.acquire(10).insert(1);
        gear.acquire(11).insert(1);

        // excluded=0, should find 12 on third probe
        let guard = gear.acquire_next_available_excluding(0, 10, 3).unwrap();
        assert_eq!(guard.slot(), 12);
    }

    #[test]
    fn test_acquire_next_available_max_probes_limits() {
        let gear: Gear<u32, 1> = Gear::new();

        // Fill slot 10
        gear.acquire(10).insert(1);

        // excluded=0, max_probes=1 only checks slot 10, fails
        let result = gear.acquire_next_available_excluding(0, 10, 1);
        assert!(matches!(result, Err(NoAvailableSlot)));

        // max_probes=2 checks 10, 11, succeeds
        let guard = gear.acquire_next_available_excluding(0, 10, 2).unwrap();
        assert_eq!(guard.slot(), 11);
    }

    #[test]
    fn test_acquire_next_available_skips_excluded() {
        let gear: Gear<u32, 4> = Gear::new();

        // Target is excluded, should go to next
        let guard = gear.acquire_next_available_excluding(10, 10, 3).unwrap();
        assert_eq!(guard.slot(), 11);
    }

    #[test]
    fn test_acquire_next_available_wraps_past_excluded() {
        let gear: Gear<u32, 1> = Gear::new();

        // Fill slots 62, 63
        {
            gear.acquire(62).insert(1);
            gear.acquire(63).insert(1);
        }

        // Target 62, excluded 0, should wrap to 1
        let guard = gear.acquire_next_available_excluding(0, 62, 5).unwrap();
        assert_eq!(guard.slot(), 1);
    }

    #[test]
    fn test_acquire_next_available_fails_all_full() {
        let gear: Gear<u32, 1> = Gear::new();

        // Fill slots 10, 11, 12
        for slot in 10..=12 {
            gear.acquire(slot).insert(1);
        }

        // excluded=9, target=10, max_probes=3 -> checks 10,11,12 all full
        let result = gear.acquire_next_available_excluding(9, 10, 3);
        assert!(matches!(result, Err(NoAvailableSlot)));
    }

    #[test]
    fn test_acquire_next_available_fails_excluded_only_option() {
        let gear: Gear<u32, 1> = Gear::new();

        // Fill slots 10, 12
        gear.acquire(10).insert(1);
        gear.acquire(12).insert(1);

        // excluded=11, target=10, max_probes=3 -> checks 10(full), 11(excluded), 12(full)
        let result = gear.acquire_next_available_excluding(11, 10, 3);
        assert!(matches!(result, Err(NoAvailableSlot)));
    }

    // ==================== Edge Slot Indices ====================

    #[test]
    fn test_slot_zero() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(0);
        let key = guard.insert(42);
        assert!(gear.is_slot_occupied(0));

        assert_eq!(guard.try_remove(key), Some(42));
        assert!(!gear.is_slot_occupied(0));
    }

    #[test]
    fn test_slot_63() {
        let gear: Gear<u32, 4> = Gear::new();

        let guard = gear.acquire(63);
        let key = guard.insert(42);
        assert!(gear.is_slot_occupied(63));
        assert_eq!(gear.occupied_bitmap(), 1u64 << 63);

        assert_eq!(guard.try_remove(key), Some(42));
        assert!(!gear.is_slot_occupied(63));
    }

    #[test]
    fn test_all_slots() {
        let gear: Gear<u32, 2> = Gear::new();

        // Insert into all 64 slots
        for slot in 0..64 {
            let guard = gear.acquire(slot);
            guard.insert(slot as u32);
        }

        assert_eq!(gear.occupied_bitmap(), u64::MAX);

        // Pop from all
        for slot in 0..64 {
            let guard = gear.acquire(slot);
            assert_eq!(guard.pop(), Some(slot as u32));
        }

        assert!(gear.is_empty());
        assert_eq!(gear.occupied_bitmap(), 0);
    }

    // ==================== Complex Types ====================

    #[test]
    fn test_string_values() {
        let gear: Gear<String, 4> = Gear::new();

        let guard = gear.acquire(10);
        guard.insert("hello".to_string());
        guard.insert("world".to_string());

        assert_eq!(guard.pop(), Some("world".to_string()));
        assert_eq!(guard.pop(), Some("hello".to_string()));
    }

    #[test]
    fn test_vec_values() {
        let gear: Gear<Vec<i32>, 4> = Gear::new();

        let guard = gear.acquire(10);
        guard.insert(vec![1, 2, 3]);
        guard.insert(vec![4, 5, 6]);

        let v1 = guard.pop().unwrap();
        let v2 = guard.pop().unwrap();

        assert!(v1 == vec![4, 5, 6] || v1 == vec![1, 2, 3]);
        assert!(v2 == vec![4, 5, 6] || v2 == vec![1, 2, 3]);
        assert_ne!(v1, v2);
    }

    // ==================== Drop Behavior ====================

    #[test]
    fn test_drop_on_gear_drop() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<usize>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        {
            let gear: Gear<DropCounter, 4> = Gear::new();

            {
                let guard = gear.acquire(10);
                guard.insert(DropCounter(Rc::clone(&drop_count)));
                guard.insert(DropCounter(Rc::clone(&drop_count)));
            }

            {
                let guard = gear.acquire(20);
                guard.insert(DropCounter(Rc::clone(&drop_count)));
            }

            assert_eq!(drop_count.get(), 0);
        }

        // All 3 dropped when gear dropped
        assert_eq!(drop_count.get(), 3);
    }

    #[test]
    fn test_drop_on_pop() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<usize>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let gear: Gear<DropCounter, 4> = Gear::new();

        {
            let guard = gear.acquire(10);
            guard.insert(DropCounter(Rc::clone(&drop_count)));
            guard.insert(DropCounter(Rc::clone(&drop_count)));

            let _ = guard.pop();
            assert_eq!(drop_count.get(), 1);

            let _ = guard.pop();
            assert_eq!(drop_count.get(), 2);
        }
    }

    #[test]
    fn test_drop_on_try_remove() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<usize>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let gear: Gear<DropCounter, 4> = Gear::new();

        let guard = gear.acquire(10);
        let key = guard.insert(DropCounter(Rc::clone(&drop_count)));
        assert_eq!(drop_count.get(), 0);

        let _ = guard.try_remove(key);
        assert_eq!(drop_count.get(), 1);
    }

    // ==================== Stress Tests ====================

    #[test]
    fn test_fill_and_drain_repeatedly() {
        let gear: Gear<u32, 8> = Gear::new();

        for round in 0..100 {
            let guard = gear.acquire(round % 64);

            for i in 0..8 {
                guard.insert(round as u32 * 100 + i);
            }
            assert!(guard.is_full());

            for _ in 0..8 {
                guard.pop();
            }
            assert!(guard.is_empty());
        }

        assert!(gear.is_empty());
    }

    #[test]
    fn test_interleaved_slots() {
        let gear: Gear<u32, 4> = Gear::new();

        // Insert into even slots
        for slot in (0..64).step_by(2) {
            gear.acquire(slot).insert(slot as u32);
        }

        assert_eq!(gear.occupied_bitmap(), 0x5555555555555555);

        // Insert into odd slots
        for slot in (1..64).step_by(2) {
            gear.acquire(slot).insert(slot as u32);
        }

        assert_eq!(gear.occupied_bitmap(), u64::MAX);

        // Pop all
        for slot in 0..64 {
            let guard = gear.acquire(slot);
            assert_eq!(guard.pop(), Some(slot as u32));
        }

        assert!(gear.is_empty());
    }

    #[test]
    fn test_random_insert_remove_pattern() {
        let gear: Gear<u32, 4> = Gear::new();

        // Insert into various slots
        let mut keys = Vec::new();
        for slot in [5, 10, 15, 20, 25] {
            let guard = gear.acquire(slot);
            let key = guard.insert(slot as u32 * 10);
            keys.push((slot, key));
        }

        // Remove some
        {
            let guard = gear.acquire(10);
            guard.try_remove(keys[1].1);
        }
        {
            let guard = gear.acquire(20);
            guard.try_remove(keys[3].1);
        }

        // Check remaining
        assert!(gear.is_slot_occupied(5));
        assert!(!gear.is_slot_occupied(10));
        assert!(gear.is_slot_occupied(15));
        assert!(!gear.is_slot_occupied(20));
        assert!(gear.is_slot_occupied(25));
    }

    #[test]
    fn test_acquire_during_drain_simulation() {
        let gear: Gear<u32, 4> = Gear::new();

        // Simulate: draining slot 10 while inserting into slot 15
        // This is the re-entrancy pattern we need

        {
            let guard = gear.acquire(10);
            guard.insert(100);
            guard.insert(101);
            guard.insert(102);
        }

        // Simulate poll loop
        let drain_guard = gear.acquire(10);

        while let Some(value) = drain_guard.pop() {
            // "Fire" the timer, which wants to reschedule
            let new_value = value + 1000;

            // Acquire different slot for insert (simulating reschedule)
            let insert_guard = gear.acquire_next_available_excluding(10, 15, 3).unwrap();
            insert_guard.insert(new_value);
        }

        // Verify: slot 10 empty, slot 15 has reschedules
        drop(drain_guard);

        assert!(!gear.is_slot_occupied(10));
        assert!(gear.is_slot_occupied(15));

        let guard = gear.acquire(15);
        assert_eq!(guard.len(), 3);

        let mut values: Vec<u32> = Vec::new();
        while let Some(v) = guard.pop() {
            values.push(v);
        }
        values.sort();
        assert_eq!(values, vec![1100, 1101, 1102]);
    }
}
