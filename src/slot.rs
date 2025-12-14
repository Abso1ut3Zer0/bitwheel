use crate::DEFAULT_SLOT_CAP;

const NONE: usize = usize::MAX;

enum Entry<T> {
    Vacant {
        next_free: usize,
    },
    Occupied {
        value: T,
        next_occupied: usize,
        prev_occupied: usize,
    },
}

/// Fixed-size slab with unsafe API.
///
/// Single allocation with intrusive free list and occupied list.
/// Occupied list enables O(1) pop.
///
/// # Safety
///
/// This is a low-level primitive. Callers must ensure:
/// - `insert` is not called when full
/// - `remove` is only called with valid, occupied keys
/// - `key < capacity` for all key-based operations
pub struct Slot<T, const CAP: usize = DEFAULT_SLOT_CAP> {
    entries: Box<[Entry<T>; CAP]>,
    free_head: usize,
    occupied_head: usize,
    len: usize,
}

#[allow(unused)]
impl<T, const CAP: usize> Slot<T, CAP> {
    pub fn new() -> Self {
        const {
            assert!(CAP > 0, "capacity must be > 0");
        }

        let entries = Box::new(std::array::from_fn(|i| Entry::Vacant {
            next_free: if i + 1 < CAP { i + 1 } else { NONE },
        }));

        Self {
            entries,
            free_head: if CAP > 0 { 0 } else { NONE },
            occupied_head: NONE,
            len: 0,
        }
    }

    /// Insert a value. Returns the key.
    ///
    /// # Safety
    /// Caller must ensure slot is not full.
    #[inline]
    pub unsafe fn insert(&mut self, value: T) -> usize {
        let key = self.free_head;

        // SAFETY: caller guarantees not full, so free_head is valid
        let next_free = {
            let entry = unsafe { self.entries.get_unchecked(key) };
            match entry {
                Entry::Vacant { next_free } => *next_free,
                Entry::Occupied { .. } => unsafe { core::hint::unreachable_unchecked() },
            }
        };

        self.free_head = next_free;

        // Update old head's prev pointer if it exists
        let old_head = self.occupied_head;
        if old_head != NONE {
            // SAFETY: occupied_head is valid when != NONE
            let old_head_entry = unsafe { self.entries.get_unchecked_mut(old_head) };
            match old_head_entry {
                Entry::Occupied { prev_occupied, .. } => *prev_occupied = key,
                Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        // SAFETY: key is valid (was free_head)
        let entry = unsafe { self.entries.get_unchecked_mut(key) };
        *entry = Entry::Occupied {
            value,
            next_occupied: old_head,
            prev_occupied: NONE,
        };

        self.occupied_head = key;
        self.len += 1;

        key
    }

    /// Remove by key.
    ///
    /// # Safety
    /// Caller must ensure key < capacity and entry is occupied.
    #[inline]
    pub unsafe fn remove(&mut self, key: usize) -> T {
        // First, read the prev/next indices
        // SAFETY: caller guarantees key < capacity and occupied
        let (next_occupied, prev_occupied) = {
            let entry = unsafe { self.entries.get_unchecked(key) };
            match entry {
                Entry::Occupied {
                    next_occupied,
                    prev_occupied,
                    ..
                } => (*next_occupied, *prev_occupied),
                Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
            }
        };

        // Unlink from occupied list - update prev's next pointer
        if prev_occupied != NONE {
            // SAFETY: prev_occupied is valid when != NONE
            let prev_entry = unsafe { self.entries.get_unchecked_mut(prev_occupied) };
            match prev_entry {
                Entry::Occupied {
                    next_occupied: prev_next,
                    ..
                } => *prev_next = next_occupied,
                Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
            }
        } else {
            // Was head
            self.occupied_head = next_occupied;
        }

        // Unlink from occupied list - update next's prev pointer
        if next_occupied != NONE {
            // SAFETY: next_occupied is valid when != NONE
            let next_entry = unsafe { self.entries.get_unchecked_mut(next_occupied) };
            match next_entry {
                Entry::Occupied {
                    prev_occupied: next_prev,
                    ..
                } => *next_prev = prev_occupied,
                Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        // Take value and link into free list
        // SAFETY: key is valid
        let entry = unsafe { self.entries.get_unchecked_mut(key) };
        let old = std::mem::replace(
            entry,
            Entry::Vacant {
                next_free: self.free_head,
            },
        );
        self.free_head = key;
        self.len -= 1;

        match old {
            Entry::Occupied { value, .. } => value,
            Entry::Vacant { .. } => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    /// Try to remove by key. Returns None if not occupied.
    ///
    /// # Safety
    /// Caller must ensure key < capacity.
    #[inline]
    pub unsafe fn try_remove(&mut self, key: usize) -> Option<T> {
        // SAFETY: caller guarantees key < capacity
        let is_occupied = {
            let entry = unsafe { self.entries.get_unchecked(key) };
            matches!(entry, Entry::Occupied { .. })
        };

        if is_occupied {
            // SAFETY: we just verified it's occupied
            Some(unsafe { self.remove(key) })
        } else {
            None
        }
    }

    /// Check if key is occupied.
    ///
    /// # Safety
    /// Caller must ensure key < capacity.
    #[inline]
    pub unsafe fn is_occupied(&self, key: usize) -> bool {
        // SAFETY: caller guarantees key < capacity
        let entry = unsafe { self.entries.get_unchecked(key) };
        matches!(entry, Entry::Occupied { .. })
    }

    /// Pop any occupied entry. O(1).
    /// Returns (key, value) so caller knows which key was popped.
    #[inline]
    pub fn try_pop(&mut self) -> Option<(usize, T)> {
        if self.occupied_head == NONE {
            return None;
        }

        let key = self.occupied_head;
        // SAFETY: occupied_head != NONE means it's valid and occupied
        let value = unsafe { self.remove(key) };
        Some((key, value))
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub const fn is_full(&self) -> bool {
        self.len >= CAP
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        CAP
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
        let slot: Slot<u32, 4> = Slot::new();

        assert!(slot.is_empty());
        assert!(!slot.is_full());
        assert_eq!(slot.len(), 0);
        assert_eq!(slot.capacity(), 4);
    }

    #[test]
    fn test_insert_single() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };

        assert_eq!(key, 0);
        assert_eq!(slot.len(), 1);
        assert!(!slot.is_empty());
        assert!(!slot.is_full());
    }

    #[test]
    fn test_insert_multiple() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };
        let k3 = unsafe { slot.insert(40) };

        assert_eq!(k0, 0);
        assert_eq!(k1, 1);
        assert_eq!(k2, 2);
        assert_eq!(k3, 3);
        assert_eq!(slot.len(), 4);
        assert!(slot.is_full());
    }

    #[test]
    fn test_remove_single() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };
        let value = unsafe { slot.remove(key) };

        assert_eq!(value, 42);
        assert!(slot.is_empty());
        assert_eq!(slot.len(), 0);
    }

    #[test]
    fn test_remove_multiple_fifo() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        assert_eq!(unsafe { slot.remove(k0) }, 10);
        assert_eq!(unsafe { slot.remove(k1) }, 20);
        assert_eq!(unsafe { slot.remove(k2) }, 30);
        assert!(slot.is_empty());
    }

    #[test]
    fn test_remove_multiple_lifo() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        assert_eq!(unsafe { slot.remove(k2) }, 30);
        assert_eq!(unsafe { slot.remove(k1) }, 20);
        assert_eq!(unsafe { slot.remove(k0) }, 10);
        assert!(slot.is_empty());
    }

    #[test]
    fn test_remove_middle() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        assert_eq!(unsafe { slot.remove(k1) }, 20);
        assert_eq!(slot.len(), 2);

        assert_eq!(unsafe { slot.remove(k0) }, 10);
        assert_eq!(unsafe { slot.remove(k2) }, 30);
    }

    // ==================== Free List Behavior ====================

    #[test]
    fn test_free_list_reuse_single() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        unsafe { slot.remove(k0) };

        let k1 = unsafe { slot.insert(20) };
        assert_eq!(k1, k0);
    }

    #[test]
    fn test_free_list_lifo_order() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        unsafe {
            slot.remove(k0);
            slot.remove(k1);
            slot.remove(k2);
        }

        // Free list is LIFO: k2 -> k1 -> k0
        let new_k0 = unsafe { slot.insert(100) };
        let new_k1 = unsafe { slot.insert(200) };
        let new_k2 = unsafe { slot.insert(300) };

        assert_eq!(new_k0, k2);
        assert_eq!(new_k1, k1);
        assert_eq!(new_k2, k0);
    }

    #[test]
    fn test_free_list_interleaved() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        unsafe { slot.remove(k1) };

        let k3 = unsafe { slot.insert(40) };
        assert_eq!(k3, k1);

        unsafe { slot.remove(k0) };

        let k4 = unsafe { slot.insert(50) };
        assert_eq!(k4, k0);

        assert_eq!(unsafe { slot.remove(k0) }, 50);
        assert_eq!(unsafe { slot.remove(k1) }, 40);
        assert_eq!(unsafe { slot.remove(k2) }, 30);
    }

    #[test]
    fn test_fill_empty_refill() {
        let mut slot: Slot<u32, 4> = Slot::new();

        for i in 0..4 {
            unsafe { slot.insert(i as u32) };
        }
        assert!(slot.is_full());

        for i in 0..4 {
            unsafe { slot.try_remove(i) };
        }
        assert!(slot.is_empty());

        for i in 0..4 {
            unsafe { slot.insert((i + 100) as u32) };
        }
        assert!(slot.is_full());
        assert_eq!(slot.len(), 4);
    }

    // ==================== try_remove ====================

    #[test]
    fn test_try_remove_occupied() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };
        let result = unsafe { slot.try_remove(key) };

        assert_eq!(result, Some(42));
        assert!(slot.is_empty());
    }

    #[test]
    fn test_try_remove_vacant() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };
        unsafe { slot.remove(key) };

        let result = unsafe { slot.try_remove(key) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_try_remove_never_occupied() {
        let mut slot: Slot<u32, 4> = Slot::new();

        unsafe { slot.insert(10) };
        unsafe { slot.insert(20) };

        let result = unsafe { slot.try_remove(2) };
        assert_eq!(result, None);
    }

    #[test]
    fn test_try_remove_double() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };

        let first = unsafe { slot.try_remove(key) };
        let second = unsafe { slot.try_remove(key) };

        assert_eq!(first, Some(42));
        assert_eq!(second, None);
    }

    // ==================== is_occupied ====================

    #[test]
    fn test_is_occupied_fresh() {
        let slot: Slot<u32, 4> = Slot::new();

        for i in 0..4 {
            assert!(!unsafe { slot.is_occupied(i) });
        }
    }

    #[test]
    fn test_is_occupied_after_insert() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };

        assert!(unsafe { slot.is_occupied(key) });
        assert!(!unsafe { slot.is_occupied(1) });
        assert!(!unsafe { slot.is_occupied(2) });
    }

    #[test]
    fn test_is_occupied_after_remove() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };
        assert!(unsafe { slot.is_occupied(key) });

        unsafe { slot.remove(key) };
        assert!(!unsafe { slot.is_occupied(key) });
    }

    // ==================== try_pop ====================

    #[test]
    fn test_try_pop_empty() {
        let mut slot: Slot<u32, 4> = Slot::new();

        assert_eq!(slot.try_pop(), None);
    }

    #[test]
    fn test_try_pop_single() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let key = unsafe { slot.insert(42) };

        let result = slot.try_pop();
        assert_eq!(result, Some((key, 42)));
        assert!(slot.is_empty());
    }

    #[test]
    fn test_try_pop_multiple() {
        let mut slot: Slot<u32, 4> = Slot::new();

        unsafe {
            slot.insert(10);
            slot.insert(20);
            slot.insert(30);
        }

        let mut values = vec![];
        while let Some((_, v)) = slot.try_pop() {
            values.push(v);
        }

        assert_eq!(values.len(), 3);
        assert!(values.contains(&10));
        assert!(values.contains(&20));
        assert!(values.contains(&30));
        assert!(slot.is_empty());
    }

    #[test]
    fn test_try_pop_returns_lifo() {
        let mut slot: Slot<u32, 4> = Slot::new();

        unsafe {
            slot.insert(10);
            slot.insert(20);
            slot.insert(30);
        }

        // Occupied list is LIFO, so pop should return in reverse order
        assert_eq!(slot.try_pop(), Some((2, 30)));
        assert_eq!(slot.try_pop(), Some((1, 20)));
        assert_eq!(slot.try_pop(), Some((0, 10)));
        assert_eq!(slot.try_pop(), None);
    }

    #[test]
    fn test_try_pop_with_gaps() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) };
        let _k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };

        unsafe {
            slot.remove(k0);
            slot.remove(k2);
        }

        let result = slot.try_pop();
        assert_eq!(result, Some((1, 20)));
        assert_eq!(slot.try_pop(), None);
    }

    #[test]
    fn test_try_pop_then_insert() {
        let mut slot: Slot<u32, 2> = Slot::new();

        unsafe {
            slot.insert(10);
            slot.insert(20);
        }

        slot.try_pop();
        assert_eq!(slot.len(), 1);

        unsafe { slot.insert(30) };
        assert_eq!(slot.len(), 2);
    }

    // ==================== Occupied List Integrity ====================

    #[test]
    fn test_occupied_list_single_element() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k = unsafe { slot.insert(42) };

        let v = unsafe { slot.remove(k) };
        assert_eq!(v, 42);

        assert_eq!(slot.try_pop(), None);
    }

    #[test]
    fn test_occupied_list_remove_head() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let _k0 = unsafe { slot.insert(10) };
        let _k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) }; // This is the head

        unsafe { slot.remove(k2) };

        let mut values = vec![];
        while let Some((_, v)) = slot.try_pop() {
            values.push(v);
        }
        assert_eq!(values.len(), 2);
        assert!(values.contains(&10));
        assert!(values.contains(&20));
    }

    #[test]
    fn test_occupied_list_remove_tail() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(10) }; // This is the tail
        let _k1 = unsafe { slot.insert(20) };
        let _k2 = unsafe { slot.insert(30) };

        unsafe { slot.remove(k0) };

        let mut values = vec![];
        while let Some((_, v)) = slot.try_pop() {
            values.push(v);
        }
        assert_eq!(values.len(), 2);
        assert!(values.contains(&20));
        assert!(values.contains(&30));
    }

    #[test]
    fn test_occupied_list_remove_middle() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let _k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) }; // Middle
        let _k2 = unsafe { slot.insert(30) };

        unsafe { slot.remove(k1) };

        let mut values = vec![];
        while let Some((_, v)) = slot.try_pop() {
            values.push(v);
        }
        assert_eq!(values.len(), 2);
        assert!(values.contains(&10));
        assert!(values.contains(&30));
    }

    #[test]
    fn test_occupied_list_complex_operations() {
        let mut slot: Slot<u32, 8> = Slot::new();

        let _k0 = unsafe { slot.insert(10) };
        let k1 = unsafe { slot.insert(20) };
        let k2 = unsafe { slot.insert(30) };
        let k3 = unsafe { slot.insert(40) };
        let _k4 = unsafe { slot.insert(50) };

        unsafe {
            slot.remove(k1);
            slot.remove(k3);
        }

        let _k5 = unsafe { slot.insert(60) };
        let _k6 = unsafe { slot.insert(70) };

        unsafe { slot.remove(k2) };

        let mut remaining: Vec<u32> = vec![];
        while let Some((_, v)) = slot.try_pop() {
            remaining.push(v);
        }

        assert_eq!(remaining.len(), 4);
        assert!(remaining.contains(&10)); // k0
        assert!(remaining.contains(&50)); // k4
        assert!(remaining.contains(&60)); // k5
        assert!(remaining.contains(&70)); // k6
    }

    // ==================== Capacity Edge Cases ====================

    #[test]
    fn test_capacity_one() {
        let mut slot: Slot<u32, 1> = Slot::new();

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
        let mut slot: Slot<u32, 1000> = Slot::new();

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
        let mut slot: Slot<String, 4> = Slot::new();

        let k0 = unsafe { slot.insert("hello".to_string()) };
        let k1 = unsafe { slot.insert("world".to_string()) };

        assert_eq!(unsafe { slot.remove(k0) }, "hello");
        assert_eq!(unsafe { slot.remove(k1) }, "world");
    }

    #[test]
    fn test_vec_values() {
        let mut slot: Slot<Vec<i32>, 4> = Slot::new();

        let k0 = unsafe { slot.insert(vec![1, 2, 3]) };
        let k1 = unsafe { slot.insert(vec![4, 5, 6]) };

        assert_eq!(unsafe { slot.remove(k1) }, vec![4, 5, 6]);
        assert_eq!(unsafe { slot.remove(k0) }, vec![1, 2, 3]);
    }

    #[test]
    fn test_tuple_values() {
        let mut slot: Slot<(u32, &str), 4> = Slot::new();

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
            let mut slot: Slot<DropCounter, 4> = Slot::new();

            unsafe {
                slot.insert(DropCounter(Rc::clone(&drop_count)));
                slot.insert(DropCounter(Rc::clone(&drop_count)));
                slot.insert(DropCounter(Rc::clone(&drop_count)));
            }

            assert_eq!(drop_count.get(), 0);
        }

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
            let mut slot: Slot<DropCounter, 4> = Slot::new();

            let k0 = unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };
            unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };
            let k2 = unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };

            unsafe {
                slot.remove(k0);
                slot.remove(k2);
            }

            assert_eq!(drop_count.get(), 2);
        }

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
            let mut slot: Slot<DropCounter, 4> = Slot::new();

            let k0 = unsafe { slot.insert(DropCounter(Rc::clone(&drop_count))) };
            unsafe { slot.remove(k0) };

            assert_eq!(drop_count.get(), 1);
        }

        assert_eq!(drop_count.get(), 1);
    }

    // ==================== Stress / Fuzz-like Tests ====================

    #[test]
    fn test_repeated_fill_drain() {
        let mut slot: Slot<u32, 8> = Slot::new();

        for round in 0..100 {
            for i in 0..8 {
                unsafe { slot.insert(round * 8 + i) };
            }
            assert!(slot.is_full());

            while slot.try_pop().is_some() {}
            assert!(slot.is_empty());
        }
    }

    #[test]
    fn test_alternating_insert_remove() {
        let mut slot: Slot<u32, 4> = Slot::new();

        for i in 0..100u32 {
            let key = unsafe { slot.insert(i) };
            let value = unsafe { slot.remove(key) };
            assert_eq!(value, i);
        }

        assert!(slot.is_empty());
    }

    #[test]
    fn test_random_pattern() {
        let mut slot: Slot<u32, 8> = Slot::new();
        let mut keys = Vec::new();

        for i in 0..5 {
            keys.push(unsafe { slot.insert(i) });
        }

        unsafe {
            slot.remove(keys[1]);
            slot.remove(keys[3]);
        }

        for i in 10..13 {
            keys.push(unsafe { slot.insert(i) });
        }

        assert_eq!(slot.len(), 6);

        let mut values = vec![];
        while let Some((_, v)) = slot.try_pop() {
            values.push(v);
        }

        assert_eq!(values.len(), 6);
        assert!(values.contains(&0));
        assert!(!values.contains(&1));
        assert!(values.contains(&2));
        assert!(!values.contains(&3));
        assert!(values.contains(&4));
        assert!(values.contains(&10));
        assert!(values.contains(&11));
        assert!(values.contains(&12));
    }

    #[test]
    fn test_key_stability() {
        let mut slot: Slot<u32, 4> = Slot::new();

        let k0 = unsafe { slot.insert(100) };
        let k1 = unsafe { slot.insert(200) };
        let k2 = unsafe { slot.insert(300) };

        assert!(unsafe { slot.is_occupied(k0) });
        assert!(unsafe { slot.is_occupied(k1) });
        assert!(unsafe { slot.is_occupied(k2) });

        unsafe { slot.remove(k1) };

        assert!(unsafe { slot.is_occupied(k0) });
        assert!(!unsafe { slot.is_occupied(k1) });
        assert!(unsafe { slot.is_occupied(k2) });

        assert_eq!(unsafe { slot.remove(k0) }, 100);
        assert_eq!(unsafe { slot.remove(k2) }, 300);
    }

    // ==================== Large Capacity Tests ====================

    #[test]
    fn test_large_capacity_pop_performance() {
        let mut slot: Slot<u32, 1024> = Slot::new();

        for i in 0..10 {
            unsafe { slot.insert(i) };
        }

        for _ in 0..10 {
            assert!(slot.try_pop().is_some());
        }
        assert!(slot.try_pop().is_none());
    }

    #[test]
    fn test_large_capacity_sparse() {
        let mut slot: Slot<u32, 1024> = Slot::new();

        let mut keys = Vec::new();
        for i in 0..100 {
            keys.push(unsafe { slot.insert(i) });
        }

        for k in keys.iter().take(90) {
            unsafe { slot.remove(*k) };
        }

        assert_eq!(slot.len(), 10);

        let mut count = 0;
        while slot.try_pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 10);
    }
}
