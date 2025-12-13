const NONE: usize = usize::MAX;

enum Entry<T> {
    Vacant { next: usize },
    Occupied(T),
}

/// Fixed-size slab with unsafe API.
pub struct Slot<T> {
    entries: Box<[Entry<T>]>,
    free_head: usize,
    len: usize,
    capacity: usize,
}

impl<T> Slot<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        let entries = (0..capacity)
            .map(|i| Entry::Vacant {
                next: if i + 1 < capacity { i + 1 } else { NONE },
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            entries,
            free_head: if capacity > 0 { 0 } else { NONE },
            len: 0,
            capacity,
        }
    }

    /// Insert a value. Returns the key.
    ///
    /// # Safety
    /// Caller must ensure slot is not full.
    #[inline(always)]
    pub unsafe fn insert(&mut self, value: T) -> usize {
        let key = self.free_head;
        let entry = unsafe { self.entries.get_unchecked_mut(key) };

        let Entry::Vacant { next } = entry else {
            unsafe { core::hint::unreachable_unchecked() }
        };

        self.free_head = *next;
        *entry = Entry::Occupied(value);
        self.len += 1;
        key
    }

    /// Remove by key.
    ///
    /// # Safety
    /// Caller must ensure key < capacity and entry is occupied.
    #[inline(always)]
    pub unsafe fn remove(&mut self, key: usize) -> T {
        let entry = unsafe { self.entries.get_unchecked_mut(key) };

        let Entry::Occupied(_) = entry else {
            unsafe { core::hint::unreachable_unchecked() }
        };

        let old = std::mem::replace(
            entry,
            Entry::Vacant {
                next: self.free_head,
            },
        );
        self.free_head = key;
        self.len -= 1;

        match old {
            Entry::Occupied(value) => value,
            Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    /// Try to remove by key. Returns None if not occupied.
    ///
    /// # Safety
    /// Caller must ensure key < capacity.
    #[inline(always)]
    pub unsafe fn try_remove(&mut self, key: usize) -> Option<T> {
        let entry = unsafe { self.entries.get_unchecked_mut(key) };

        if let Entry::Occupied(_) = entry {
            let old = std::mem::replace(
                entry,
                Entry::Vacant {
                    next: self.free_head,
                },
            );
            self.free_head = key;
            self.len -= 1;

            match old {
                Entry::Occupied(value) => Some(value),
                Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
            }
        } else {
            None
        }
    }

    /// Check if key is occupied.
    ///
    /// # Safety
    /// Caller must ensure key < capacity.
    #[inline(always)]
    pub unsafe fn is_occupied(&self, key: usize) -> bool {
        matches!(
            unsafe { self.entries.get_unchecked(key) },
            Entry::Occupied(_)
        )
    }

    /// Try to pop any occupied entry.
    #[inline(always)]
    pub fn try_pop(&mut self) -> Option<T> {
        for i in 0..self.capacity {
            // SAFETY: i < capacity
            let entry = unsafe { self.entries.get_unchecked_mut(i) };

            if let Entry::Occupied(_) = entry {
                let old = std::mem::replace(
                    entry,
                    Entry::Vacant {
                        next: self.free_head,
                    },
                );
                self.free_head = i;
                self.len -= 1;

                return match old {
                    Entry::Occupied(value) => Some(value),
                    // SAFETY: we just checked it's occupied
                    Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
                };
            }
        }
        None
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    // ==================== Basic Operations ====================

    #[test]
    fn test_new_empty() {
        let slot: Slot<u32> = Slot::with_capacity(4);

        assert!(slot.is_empty());
        assert!(!slot.is_full());
        assert_eq!(slot.len(), 0);
        assert_eq!(slot.capacity(), 4);
    }

    #[test]
    fn test_insert_single() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let key = unsafe { slot.insert(42) };

        assert_eq!(key, 0);
        assert_eq!(slot.len(), 1);
        assert!(!slot.is_empty());
        assert!(!slot.is_full());
    }

    #[test]
    fn test_insert_multiple() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };
        let k3 = unsafe { slot.insert(40) };

        // Keys should be sequential on fresh slot
        assert_eq!(k0, 0);
        assert_eq!(k1, 1);
        assert_eq!(k2, 2);
        assert_eq!(k3, 3);
        assert_eq!(slot.len(), 4);
        assert!(slot.is_full());
    }

    #[test]
    fn test_remove_single() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let key = unsafe { slot.insert(42) };
        let value = unsafe { slot.remove(key) };

        assert_eq!(value, 42);
        assert!(slot.is_empty());
        assert_eq!(slot.len(), 0);
    }

    #[test]
    fn test_remove_multiple_fifo() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        // Remove in insertion order
        assert_eq!(unsafe { slot.remove(k0) }, 10);
        assert_eq!(unsafe { slot.remove(k1) }, 20);
        assert_eq!(unsafe { slot.remove(k2) }, 30);
        assert!(slot.is_empty());
    }

    #[test]
    fn test_remove_multiple_lifo() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        // Remove in reverse order
        assert_eq!(unsafe { slot.remove(k2) }, 30);
        assert_eq!(unsafe { slot.remove(k1) }, 20);
        assert_eq!(unsafe { slot.remove(k0) }, 10);
        assert!(slot.is_empty());
    }

    #[test]
    fn test_remove_middle() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        // Remove from middle
        assert_eq!(unsafe { slot.remove(k1) }, 20);
        assert_eq!(slot.len(), 2);

        // Others still accessible
        assert_eq!(unsafe { slot.remove(k0) }, 10);
        assert_eq!(unsafe { slot.remove(k2) }, 30);
    }

    // ==================== Free List Behavior ====================

    #[test]
    fn test_free_list_reuse_single() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(10) };
        unsafe { slot.remove(k0) };

        // Should reuse key 0
        let k1 = unsafe { slot.insert(20) };
        assert_eq!(k1, k0);
    }

    #[test]
    fn test_free_list_lifo_order() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        // Remove in order: 0, 1, 2
        unsafe {
            slot.remove(k0);
            slot.remove(k1);
            slot.remove(k2);
        }

        // Free list is LIFO, so reinsert should give: 2, 1, 0
        let new_k0 = unsafe { slot.insert(100) };
        let new_k1 = unsafe { slot.insert(200) };
        let new_k2 = unsafe { slot.insert(300) };

        assert_eq!(new_k0, k2); // 2 was last freed
        assert_eq!(new_k1, k1);
        assert_eq!(new_k2, k0);
    }

    #[test]
    fn test_free_list_interleaved() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        // Insert 3
        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        // Remove middle
        unsafe { slot.remove(k1) };

        // Insert should reuse k1
        let k3 = unsafe { slot.insert(40) };
        assert_eq!(k3, k1);

        // Remove first
        unsafe { slot.remove(k0) };

        // Insert should reuse k0
        let k4 = unsafe { slot.insert(50) };
        assert_eq!(k4, k0);

        // Verify values
        assert_eq!(unsafe { slot.remove(k0) }, 50);
        assert_eq!(unsafe { slot.remove(k1) }, 40);
        assert_eq!(unsafe { slot.remove(k2) }, 30);
    }

    #[test]
    fn test_fill_empty_refill() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        // Fill completely
        for i in 0..4 {
            unsafe { slot.insert(i as u32) };
        }
        assert!(slot.is_full());

        // Empty completely
        for i in 0..4 {
            unsafe { slot.try_remove(i) };
        }
        assert!(slot.is_empty());

        // Refill completely
        for i in 0..4 {
            unsafe { slot.insert((i + 100) as u32) };
        }
        assert!(slot.is_full());
        assert_eq!(slot.len(), 4);
    }

    // ==================== try_remove ====================

    #[test]
    fn test_try_remove_occupied() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let key = unsafe { slot.insert(42) };
        let result = unsafe { slot.try_remove(key) };

        assert_eq!(result, Some(42));
        assert!(slot.is_empty());
    }

    #[test]
    fn test_try_remove_vacant() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let key = unsafe { slot.insert(42) };
        unsafe { slot.remove(key) };

        // Second try_remove should return None
        let result = unsafe { slot.try_remove(key) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_try_remove_never_occupied() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        // Key 2 was never occupied
        unsafe { slot.insert(10) }; // key 0
        unsafe { slot.insert(20) }; // key 1

        let result = unsafe { slot.try_remove(2) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_try_remove_double() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let key = unsafe { slot.insert(42) };

        let first = unsafe { slot.try_remove(key) };
        let second = unsafe { slot.try_remove(key) };

        assert_eq!(first, Some(42));
        assert_eq!(second, None);
    }

    // ==================== is_occupied ====================

    #[test]
    fn test_is_occupied_fresh() {
        let slot: Slot<u32> = Slot::with_capacity(4);

        // All vacant initially
        for i in 0..4 {
            assert!(!unsafe { slot.is_occupied(i) });
        }
    }

    #[test]
    fn test_is_occupied_after_insert() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let key = unsafe { slot.insert(42) };

        assert!(unsafe { slot.is_occupied(key) });
        assert!(!unsafe { slot.is_occupied(1) });
        assert!(!unsafe { slot.is_occupied(2) });
    }

    #[test]
    fn test_is_occupied_after_remove() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let key = unsafe { slot.insert(42) };
        assert!(unsafe { slot.is_occupied(key) });

        unsafe { slot.remove(key) };
        assert!(!unsafe { slot.is_occupied(key) });
    }

    // ==================== try_pop ====================

    #[test]
    fn test_try_pop_empty() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        assert_eq!(slot.try_pop(), None);
    }

    #[test]
    fn test_try_pop_single() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        unsafe { slot.insert(42) };

        assert_eq!(slot.try_pop(), Some(42));
        assert!(slot.is_empty());
    }

    #[test]
    fn test_try_pop_multiple() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        unsafe {
            slot.insert(10);
            slot.insert(20);
            slot.insert(30);
        }

        let mut values = vec![];
        while let Some(v) = slot.try_pop() {
            values.push(v);
        }

        assert_eq!(values.len(), 3);
        assert!(values.contains(&10));
        assert!(values.contains(&20));
        assert!(values.contains(&30));
        assert!(slot.is_empty());
    }

    #[test]
    fn test_try_pop_with_gaps() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(10) };
        let _k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        // Create gaps
        unsafe {
            slot.remove(k0);
            slot.remove(k2);
        }

        // Only 20 should remain
        assert_eq!(slot.try_pop(), Some(20));
        assert_eq!(slot.try_pop(), None);
    }

    #[test]
    fn test_try_pop_then_insert() {
        let mut slot: Slot<u32> = Slot::with_capacity(2);

        unsafe {
            slot.insert(10);
            slot.insert(20);
        }

        slot.try_pop();
        assert_eq!(slot.len(), 1);

        // Should be able to insert again
        unsafe { slot.insert(30) };
        assert_eq!(slot.len(), 2);
    }

    // ==================== Capacity Edge Cases ====================

    #[test]
    fn test_capacity_zero() {
        let slot: Slot<u32> = Slot::with_capacity(0);

        assert!(slot.is_empty());
        assert!(slot.is_full()); // len >= capacity (0 >= 0)
        assert_eq!(slot.capacity(), 0);
        assert_eq!(slot.len(), 0);
    }

    #[test]
    fn test_capacity_one() {
        let mut slot: Slot<u32> = Slot::with_capacity(1);

        assert!(!slot.is_full());

        let key = unsafe { slot.insert(42) };
        assert!(slot.is_full());
        assert_eq!(key, 0);

        let value = unsafe { slot.remove(key) };
        assert_eq!(value, 42);
        assert!(!slot.is_full());
    }

    #[test]
    fn test_capacity_large() {
        let mut slot: Slot<u32> = Slot::with_capacity(1000);

        for i in 0..1000 {
            unsafe { slot.insert(i) };
        }

        assert!(slot.is_full());
        assert_eq!(slot.len(), 1000);

        for i in 0..1000 {
            unsafe { slot.try_remove(i) };
        }

        assert!(slot.is_empty());
    }

    // ==================== Complex Types ====================

    #[test]
    fn test_string_values() {
        let mut slot: Slot<String> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert("hello".to_string()) };
        let k1 = unsafe { slot.insert("world".to_string()) };

        assert_eq!(unsafe { slot.remove(k0) }, "hello");
        assert_eq!(unsafe { slot.remove(k1) }, "world");
    }

    #[test]
    fn test_vec_values() {
        let mut slot: Slot<Vec<i32>> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert(vec![1, 2, 3]) };
        let k1 = unsafe { slot.insert(vec![4, 5, 6]) };

        assert_eq!(unsafe { slot.remove(k1) }, vec![4, 5, 6]);
        assert_eq!(unsafe { slot.remove(k0) }, vec![1, 2, 3]);
    }

    #[test]
    fn test_tuple_values() {
        let mut slot: Slot<(u32, &str)> = Slot::with_capacity(4);

        let k0 = unsafe { slot.insert((1, "one")) };
        let k1 = unsafe { slot.insert((2, "two")) };

        assert_eq!(unsafe { slot.remove(k0) }, (1, "one"));
        assert_eq!(unsafe { slot.remove(k1) }, (2, "two"));
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
            let mut slot: Slot<DropCounter> = Slot::with_capacity(4);

            unsafe {
                slot.insert(DropCounter(Rc::clone(&drop_count)));
                slot.insert(DropCounter(Rc::clone(&drop_count)));
                slot.insert(DropCounter(Rc::clone(&drop_count)));
            }

            assert_eq!(drop_count.get(), 0);
        }

        // All 3 should be dropped
        assert_eq!(drop_count.get(), 3);
    }

    #[test]
    fn test_drop_with_mixed() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<usize>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        {
            let mut slot: Slot<DropCounter> = Slot::with_capacity(4);

            let k0 = unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };
            unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };
            let k2 = unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };

            // Remove two, leaving one occupied
            unsafe {
                slot.remove(k0);
                slot.remove(k2);
            }

            // 2 dropped from remove
            assert_eq!(drop_count.get(), 2);
        }

        // 1 more dropped when slot drops
        assert_eq!(drop_count.get(), 3);
    }

    #[test]
    fn test_drop_empty() {
        let drop_count = Rc::new(Cell::new(0));

        struct DropCounter(Rc<Cell<usize>>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        {
            let mut slot: Slot<DropCounter> = Slot::with_capacity(4);

            let k0 = unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };
            unsafe { slot.remove(k0) };

            assert_eq!(drop_count.get(), 1);
        }

        // No additional drops - slot was empty
        assert_eq!(drop_count.get(), 1);
    }

    // ==================== Stress / Fuzz-like Tests ====================

    #[test]
    fn test_repeated_fill_drain() {
        let mut slot: Slot<u32> = Slot::with_capacity(8);

        for round in 0..100 {
            // Fill
            for i in 0..8 {
                unsafe { slot.insert(round * 8 + i) };
            }
            assert!(slot.is_full());

            // Drain
            for _ in 0..8 {
                slot.try_pop();
            }
            assert!(slot.is_empty());
        }
    }

    #[test]
    fn test_alternating_insert_remove() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        for i in 0..100u32 {
            let key = unsafe { slot.insert(i) };
            let value = unsafe { slot.remove(key) };
            assert_eq!(value, i);
        }

        assert!(slot.is_empty());
    }

    #[test]
    fn test_random_pattern() {
        let mut slot: Slot<u32> = Slot::with_capacity(8);
        let mut keys = Vec::new();

        // Insert 5
        for i in 0..5 {
            keys.push(unsafe { slot.insert(i) });
        }

        // Remove 2
        unsafe {
            slot.remove(keys[1]);
            slot.remove(keys[3]);
        }

        // Insert 3 more
        for i in 10..13 {
            keys.push(unsafe { slot.insert(i) });
        }

        assert_eq!(slot.len(), 6);

        // Drain all
        let mut values = vec![];
        while let Some(v) = slot.try_pop() {
            values.push(v);
        }

        assert_eq!(values.len(), 6);
        assert!(values.contains(&0));
        assert!(!values.contains(&1)); // was removed
        assert!(values.contains(&2));
        assert!(!values.contains(&3)); // was removed
        assert!(values.contains(&4));
        assert!(values.contains(&10));
        assert!(values.contains(&11));
        assert!(values.contains(&12));
    }

    #[test]
    fn test_key_stability() {
        let mut slot: Slot<u32> = Slot::with_capacity(4);

        // Insert values
        let k0 = unsafe { slot.insert(100) };
        let k1 = unsafe { slot.insert(200) };
        let k2 = unsafe { slot.insert(300) };

        // Keys should remain valid until removed
        assert!(unsafe { slot.is_occupied(k0) });
        assert!(unsafe { slot.is_occupied(k1) });
        assert!(unsafe { slot.is_occupied(k2) });

        // Remove middle
        unsafe { slot.remove(k1) };

        // k0 and k2 still valid
        assert!(unsafe { slot.is_occupied(k0) });
        assert!(!unsafe { slot.is_occupied(k1) });
        assert!(unsafe { slot.is_occupied(k2) });

        // Can still remove by original keys
        assert_eq!(unsafe { slot.remove(k0) }, 100);
        assert_eq!(unsafe { slot.remove(k2) }, 300);
    }
}
