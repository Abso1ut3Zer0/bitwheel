use crate::slot::Slot;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WheelError {
    #[error("wheel full: all {0} slots at capacity")]
    WheelFull(usize),
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
}

impl<T> InnerWheel<T> {
    /// Create a new wheel.
    ///
    /// `num_slots` is rounded up to the next power of 2.
    pub fn new(num_slots: usize, slot_capacity: usize) -> Self {
        let num_slots = num_slots.next_power_of_two();

        Self {
            slots: (0..num_slots)
                .map(|_| Slot::with_capacity(slot_capacity))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            num_slots,
            slot_capacity,
            mask: num_slots - 1,
        }
    }

    /// Insert timer at slot, probing forward if full.
    ///
    /// Returns the actual slot and key on success.
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

        for probe in 0..self.num_slots {
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

        Err(WheelError::WheelFull(self.num_slots))
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
        unsafe {
            let s = self.slots.get_unchecked_mut(slot);
            s.remove(key)
        }
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
        unsafe {
            let s = self.slots.get_unchecked_mut(slot);
            s.try_remove(key)
        }
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
        unsafe {
            let s = self.slots.get_unchecked_mut(slot);
            s.try_pop()
        }
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
        unsafe { self.slots.get_unchecked(slot).is_empty() }
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
        unsafe { self.slots.get_unchecked(slot).is_full() }
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
    pub fn total_capacity(&self) -> usize {
        self.num_slots * self.slot_capacity
    }
}
