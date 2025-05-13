//! This module contains utilities to generate and validate rank assignments.

#![allow(dead_code)] // until fully used publically

use std::cmp::min;
use std::collections::HashMap;
use std::mem::replace;
use std::rc::Rc;

use bitmaps::Bitmap;

const CHUNK_SIZE: usize = 1024;

/// Ranks is a bidirectional map between a dense set of ranks
/// and data of type `T`. Ranks also keeps track of the set of
/// free ranks, and can allocate new ranks for incoming values.
pub(crate) struct Ranks<T> {
    forward: Vec<Option<Rc<T>>>,
    reverse: HashMap<Rc<T>, usize>,
    // A more scalable implementation of this would keep a tree
    // of bitmaps (could be high fanout) indicating which chunks
    // had free elements available.
    allocated: Vec<Bitmap<CHUNK_SIZE>>,
}

// SAFETY: This is needed because Rc is not Send. However, because we never
// expose the Rc outside of the implementation, we wil always be sending the
// *full collection of Rcs*, and so this is actually safe to do. (And unfortunately
// this cannot be expressed directly in the Rust type system.)
unsafe impl<T> Send for Ranks<T> {}
// SAFETY: Sending `&Ranks` is safe because there is no interior mutability, so
// sharing a const reference across threads does not require synchronized access.
unsafe impl<T> Sync for Ranks<T> {}

impl<T> Ranks<T> {
    /// Create a new rank map with the given size.
    pub(crate) fn new(size: usize) -> Self {
        assert!(size > 0, "cannot allocate zero ranks");
        Self {
            forward: vec![None; size],
            reverse: HashMap::new(),
            allocated: vec![Bitmap::new(); size.div_ceil(CHUNK_SIZE)],
        }
    }
}

impl<T> Ranks<T>
where
    T: std::cmp::Eq + std::hash::Hash,
{
    /// Assign a rank to the given value. Returns a defined rank
    /// when assignment was successful, otherwise the value is
    /// returned back.
    pub(crate) fn assign(&mut self, value: T) -> Result<usize, T> {
        let rank = match self.allocate() {
            Some(rank) => rank,
            None => return Err(value),
        };
        let value = Rc::new(value);
        self.forward[rank] = Some(Rc::clone(&value));
        if let Some(prev_rank) = self.reverse.insert(value, rank) {
            // The value was assigned to another rank. Clean it up.
            self.forward[prev_rank].take().unwrap();
            self.set(prev_rank, false);
        }
        Ok(rank)
    }

    /// Remove the rank mapping of the provided value; return its
    /// previously assigned rank, if any.
    pub(crate) fn unassign(&mut self, value: T) -> Option<usize> {
        let value = Rc::new(value);
        let rank = self.reverse.remove(&value)?;
        self.forward[rank].take().unwrap();
        self.set(rank, false);
        Some(rank)
    }

    /// Insert the given item at the given rank. Returns the item
    /// that previously occupied the rank, if any.
    pub(crate) fn insert(&mut self, rank: usize, value: T) -> Option<T> {
        let value = Rc::new(value);
        let prev = replace(&mut self.forward[rank], Some(Rc::clone(&value))).map(|prev| {
            // There was a previous value at this rank.
            // Clean up the reverse, and take unwrap the
            // value.
            self.reverse.remove(&prev).unwrap();
            Rc::into_inner(prev).unwrap()
        });

        if let Some(prev_rank) = self.reverse.insert(value, rank) {
            // The value was assigned to another rank, which must
            // now be unassigned.
            self.forward[prev_rank].take().unwrap(); // must exist in forward            
            self.set(prev_rank, false);
        }
        self.set(rank, true);
        prev
    }

    /// Remove the value at the given rank, returning it it if it exists.
    pub(crate) fn remove(&mut self, rank: usize) -> Option<T> {
        let value = self.forward[rank].take()?;
        assert!(self.reverse.remove(&value).is_some());
        self.set(rank, false);
        Some(Rc::into_inner(value).unwrap())
    }

    /// Get the value at the given rank, if it exists.
    pub(crate) fn get(&self, rank: usize) -> Option<&T> {
        self.forward.get(rank)?.as_ref().map(Rc::as_ref)
    }

    /// Get a mutable reference to the value at the given rank, if it exists.
    pub(crate) fn get_mut(&mut self, rank: usize) -> Option<&mut T> {
        self.forward.get_mut(rank)?.as_mut().and_then(Rc::get_mut)
    }

    /// Returns whether the provided rank is defined in the rank mapping.
    pub(crate) fn contains(&self, rank: usize) -> bool {
        self.forward[rank].is_some()
    }

    /// Return the rank of the given value, if it is assigned.
    pub(crate) fn rank(&self, value: &T) -> Option<&usize> {
        self.reverse.get(value)
    }

    /// Tells whether this set of ranks are fully assigned
    pub(crate) fn is_full(&self) -> bool {
        self.allocated
            .iter()
            .enumerate()
            .all(|(idx, chunk)| match chunk.first_false_index() {
                Some(last) => {
                    last == min(CHUNK_SIZE * (idx + 1), self.forward.len()) - CHUNK_SIZE * idx
                }
                None => true,
            })
    }

    fn set(&mut self, index: usize, occupied: bool) {
        let chunk = index / CHUNK_SIZE;
        let off = index % CHUNK_SIZE;
        if let Some(chunk) = self.allocated.get_mut(chunk) {
            chunk.set(off, occupied);
        }
    }

    fn allocate(&mut self) -> Option<usize> {
        for (idx, chunk) in self.allocated.iter_mut().enumerate() {
            if let Some(free) = chunk.first_false_index() {
                chunk.set(free, true);
                let rank = idx * CHUNK_SIZE + free;
                if rank < self.forward.len() {
                    return Some(rank);
                } else {
                    return None;
                }
            }
        }
        None
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Option<&T>> {
        self.forward.iter().map(|rc| rc.as_ref().map(Rc::as_ref))
    }
}

pub struct IntoIter<T> {
    iter: std::vec::IntoIter<Option<Rc<T>>>,
}

// SAFETY: This is needed because Rc is not Send. However, the full vector
// is fully owned by the IntoIter, and no Rcs are held outside of this vector,
// and so this is safe to do.
unsafe impl<T> Send for IntoIter<T> {}

impl<T> IntoIterator for Ranks<T> {
    type Item = Option<T>;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.forward.into_iter(),
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = Option<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;

        Some(item.map(|rc| Rc::into_inner(rc).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let ranks: Ranks<i32> = Ranks::new(10);
        assert_eq!(ranks.forward.len(), 10);
        assert_eq!(ranks.reverse.len(), 0);
        assert_eq!(ranks.allocated.len(), 1);
    }

    #[test]
    fn test_assign() {
        let mut ranks = Ranks::new(1);
        let rank = ranks.assign(42).unwrap();
        assert_eq!(ranks.get(rank), Some(&42));
        assert_eq!(ranks.rank(&42), Some(&rank));
        // Out of capacity now.
        assert_eq!(ranks.assign(123).unwrap_err(), 123);
    }

    #[test]
    fn test_unassign() {
        let mut ranks = Ranks::new(10);
        let rank = ranks.assign(42).unwrap();
        assert_eq!(ranks.unassign(42), Some(rank));
        assert_eq!(ranks.get(rank), None);
        assert_eq!(ranks.rank(&42), None);
        assert_eq!(ranks.unassign(43), None);
    }

    #[test]
    fn test_insert() {
        let mut ranks = Ranks::new(10);
        let old_value = ranks.insert(0, 42);
        assert_eq!(old_value, None);
        assert_eq!(ranks.get(0), Some(&42));
        assert_eq!(ranks.insert(0, 123), Some(42));
    }

    #[test]
    fn test_remove() {
        let mut ranks = Ranks::new(10);
        ranks.insert(0, 42);
        let removed_value = ranks.remove(0);
        assert_eq!(removed_value, Some(42));
        assert_eq!(ranks.get(0), None);
        assert_eq!(ranks.assign(123).unwrap(), 0);
    }

    #[test]
    fn test_get() {
        let mut ranks = Ranks::new(10);
        ranks.insert(0, 42);
        assert_eq!(ranks.get(0), Some(&42));
        assert_eq!(ranks.get(1), None);
    }

    #[test]
    fn test_rank() {
        let mut ranks = Ranks::new(10);
        let rank = ranks.assign(42).unwrap();
        assert_eq!(ranks.rank(&42), Some(&rank));
        assert_eq!(ranks.rank(&43), None);
    }

    #[test]
    fn test_is_full() {
        let sizes = [
            1,
            2,
            100,
            CHUNK_SIZE - 1,
            CHUNK_SIZE,
            CHUNK_SIZE + 1,
            2 * CHUNK_SIZE - 1,
            2 * CHUNK_SIZE,
            2 * CHUNK_SIZE + 1,
        ];
        for size in sizes {
            let mut ranks = Ranks::new(size);
            for n in 0..size {
                assert!(!ranks.is_full());
                assert_eq!(ranks.assign(n).unwrap(), n);
            }
            assert!(ranks.is_full());
        }
    }

    #[test]
    fn test_iter() {
        let mut ranks = Ranks::new(3);
        ranks.assign(10).unwrap();
        ranks.assign(20).unwrap();
        ranks.assign(30).unwrap();

        let mut iter = ranks.iter();
        assert_eq!(iter.next(), Some(Some(&10)));
        assert_eq!(iter.next(), Some(Some(&20)));
        assert_eq!(iter.next(), Some(Some(&30)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let mut ranks = Ranks::new(3);
        ranks.assign(10).unwrap();
        ranks.assign(20).unwrap();
        ranks.assign(30).unwrap();

        let mut into_iter = ranks.into_iter();
        assert_eq!(into_iter.next(), Some(Some(10)));
        assert_eq!(into_iter.next(), Some(Some(20)));
        assert_eq!(into_iter.next(), Some(Some(30)));
        assert_eq!(into_iter.next(), None);
    }
}
