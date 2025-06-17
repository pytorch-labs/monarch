/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ptr::NonNull;
use std::sync::Arc;

use atomic_refcell::AtomicRef;
use atomic_refcell::AtomicRefCell;
use atomic_refcell::AtomicRefMut;
use atomic_refcell::BorrowError;
use atomic_refcell::BorrowMutError;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;

/// A container that dynamically checks borrow rules in an aliasing-aware fashion.
///
/// `AliasTrackingRefCell`s can alias one another, and a mutable borrow of one
/// `AliasTrackingRefCell` will mutably borrow all its aliases as well. That means that
/// trying to borrow an alias at that point will panic.
///
/// The API for `AliasTrackingRefCell` is very similar to [`RefCell`](std::cell::RefCell),
/// with some modifications to account for our explicit management of aliasing.
///
/// # Example
/// ```
/// // type TensorCell = AliasTrackingRefCell<Tensor>;
/// use torch_sys::TensorCell;
/// use torch_sys::test_make_alias;
/// use torch_sys::test_make_tensor;
/// let my_tensor = test_make_tensor();
/// let my_tensor_alias = unsafe { test_make_alias(&my_tensor) };
/// let cell = TensorCell::new(my_tensor);
/// let aliased_cell = TensorCell::new_with_alias(my_tensor_alias, &cell);
///
/// // Can immutably borrow as many times as you want.
/// // You can use the output like a `&Tensor`.
/// let tensor_ref = cell.borrow();
/// let second_borrow = cell.borrow();
/// let tensor_alias_ref = aliased_cell.borrow();
///
/// // But this would panic! Since we already have immutable borrows active.
/// // let oops = cell.borrow_mut()
/// ```
///
/// # Implementation notes
///
/// The idea is to hold onto a shared-ownership value (`alias_tracker`) that
/// correctly models the aliasing relationships of the underlying tensor
/// storage. So, two `AliasTrackingRefCell`s that contain aliased data
/// will share a reference to the same `alias_tracker`.
///
/// We then use an [`AtomicRefCell`] over that value to dynamically enforce
/// borrow rules for the aliases set collectively (and thread-safely).
///
/// The rest of the implementation closely copies the standard library's
/// implementation of `RefCell`.
pub struct AliasTrackingRefCell<T: ?Sized> {
    alias_tracker: Arc<AtomicRefCell<()>>,
    value: UnsafeCell<T>,
}

// SAFETY: `AliasTrackingRefCell<T> is a cell of T and acts like a reference.
unsafe impl<T: ?Sized + Send> Send for AliasTrackingRefCell<T> {}
// SAFETY: `AliasTrackingRefCell<T> is a cell of T and acts like a reference.
unsafe impl<T: ?Sized + Send + Sync> Sync for AliasTrackingRefCell<T> {}

impl<T> AliasTrackingRefCell<T> {
    /// Creates a new `AliasTrackingRefCell` that owns the given `T`. This
    /// `AliasTrackingRefCell` will not alias anything.
    #[inline]
    pub fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            alias_tracker: Arc::new(AtomicRefCell::new(())),
        }
    }

    /// Creates a new `AliasTrackingRefCell` that aliases `alias`.
    #[inline]
    pub fn new_with_alias(value: T, alias: &Self) -> Self {
        Self {
            value: UnsafeCell::new(value),
            alias_tracker: alias.alias_tracker.clone(),
        }
    }
}

impl<T: ?Sized> AliasTrackingRefCell<T> {
    /// Immutably borrows the given `T` and all its aliases. The borrow lasts until
    /// `AliasTrackingRef` is dropped.
    ///
    /// # Panics
    /// Will panic if the given `T` or any of its aliases are mutably borrowed.
    /// For a non-panicking version, see [`try_borrow`](AliasTrackingRefCell::try_borrow).
    pub fn borrow(&self) -> AliasTrackingRef<T> {
        match self.try_borrow() {
            Ok(borrow) => borrow,
            Err(e) => panic!("borrow failed: {:?}", e),
        }
    }

    /// Immutably borrows the given `T` and all its aliases, returning an error if
    /// it or any of its aliases are currently mutably borrowed. The borrow
    /// lasts until the `AliasTrackingRef` is dropped.
    ///
    /// This is a non-panicking version of [`borrow`](AliasTrackingRefCell::borrow).
    pub fn try_borrow(&self) -> Result<AliasTrackingRef<T>, BorrowError> {
        Ok(AliasTrackingRef {
            borrow: self.alias_tracker.try_borrow()?,
            // SAFETY: The alias_tracker borrow guarantees that there is only
            // immutable access to the given `T`.
            value: unsafe { NonNull::new_unchecked(self.value.get()) },
        })
    }

    /// Mutably borrows the tensor and all its aliases. The borrow lasts until
    /// `AliasTrackingRefMut` is dropped.
    ///
    /// # Panics
    /// Will panic if the `Tensor` or any of its aliases are borrowed. For a
    /// non-panicking version, see
    /// [`try_borrow_mut`](TensorCell::try_borrow_mut).
    pub fn borrow_mut(&self) -> AliasTrackingRefMut<T> {
        match self.try_borrow_mut() {
            Ok(borrow) => borrow,
            Err(e) => panic!("borrow_mut failed: {:?}", e),
        }
    }

    /// Mutably borrows the tensor and all its aliases, returning an error if
    /// the tensor is currently borrowed. The borrow lasts until `TensorRefMut`
    /// is dropped.
    ///
    /// This is a non-panicking version of [`borrow_mut`](TensorCell::borrow_mut).
    pub fn try_borrow_mut(&self) -> Result<AliasTrackingRefMut<T>, BorrowMutError> {
        Ok(AliasTrackingRefMut {
            borrow: self.alias_tracker.try_borrow_mut()?,
            // SAFETY: The alias_tracker mutable borrow guarantees unique access.
            value: unsafe { NonNull::new_unchecked(self.value.get()) },
        })
    }

    /// Returns true if this `TensorCell` aliases `other`.
    pub fn aliases(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.alias_tracker, &other.alias_tracker)
    }

    /// Returns a pointer to the alias tracker. Useful for de-duping borrows.
    /// Visibility limited to crate as it exposes TensorCell internal
    /// representation.
    pub(crate) fn alias_ptr(&self) -> *const AtomicRefCell<()> {
        Arc::as_ptr(&self.alias_tracker)
    }

    /// Returns a reference to the tensor, without checking for borrows.
    ///
    /// SAFETY: The caller must ensure that it holds a borrow on this tensor.
    pub unsafe fn get_unchecked(&self) -> &T {
        // SAFETY: see above
        unsafe { self.value.get().as_ref().unwrap() }
    }
}

impl<T: ?Sized + PartialEq> PartialEq for AliasTrackingRefCell<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self.borrow() == *other.borrow()
    }
}

impl<T: ?Sized + Eq> Eq for AliasTrackingRefCell<T> {}

impl<T: ?Sized + Debug> Debug for AliasTrackingRefCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.try_borrow() {
            Ok(borrow) => f
                .debug_struct("AliasTrackingRefCell")
                .field("value", &borrow)
                .finish(),
            Err(_) => f
                .debug_struct("AliasTrackingRefCell")
                .field("value", &"<mutably borrowed elsewhere>")
                .finish(),
        }
    }
}

impl<T: Serialize> Serialize for AliasTrackingRefCell<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let borrow = self.try_borrow().map_err(serde::ser::Error::custom)?;
        borrow.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for AliasTrackingRefCell<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = T::deserialize(deserializer)?;
        Ok(Self::new(value))
    }
}

pub struct AliasTrackingRef<'a, T: ?Sized + 'a> {
    // NB: we use a pointer instead of `&'a T` to avoid `noalias` violations,
    // because a `Ref` argument doesn't hold immutability for the entire 'a
    // lifetime, only until it drops.
    value: NonNull<T>,
    // This is not used, but holding the borrow is what guards the tensor
    // value.
    #[allow(dead_code)]
    borrow: AtomicRef<'a, ()>,
}

// SAFETY: `AliasTrackingRef<'_, T> acts as a reference.
unsafe impl<'a, T: ?Sized> Sync for AliasTrackingRef<'a, T> where for<'b> &'b T: Sync {}
// SAFETY: `AliasTrackingRef<'_, T> acts as a reference.
unsafe impl<'a, T: ?Sized> Send for AliasTrackingRef<'a, T> where for<'b> &'b T: Send {}

impl<'a, T: ?Sized + Debug + 'a> Debug for AliasTrackingRef<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        <T as Debug>::fmt(self, f)
    }
}

impl<'a, T: ?Sized> Deref for AliasTrackingRef<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We hold shared borrow of the value.
        unsafe { self.value.as_ref() }
    }
}

pub struct AliasTrackingRefMut<'a, T: ?Sized + 'a> {
    // NB: we use a pointer instead of `&'a T` to avoid `noalias` violations,
    // because a `Ref` argument doesn't hold immutability for the entire 'a
    // lifetime, only until it drops.
    value: NonNull<T>,
    // This is not used, but holding the borrow is what guards the tensor
    // value.
    #[allow(dead_code)]
    borrow: AtomicRefMut<'a, ()>,
}

// SAFETY: `AliasTrackingRefMut<'_, T> acts as a mutable reference.
unsafe impl<'a, T: ?Sized> Sync for AliasTrackingRefMut<'a, T> where for<'b> &'b T: Sync {}
// SAFETY: `AliasTrackingRefMut<'_, T> acts as a mutable reference.
unsafe impl<'a, T: ?Sized> Send for AliasTrackingRefMut<'a, T> where for<'b> &'b T: Send {}

impl<'a, T: ?Sized + Debug + 'a> Debug for AliasTrackingRefMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        <T as Debug>::fmt(self, f)
    }
}

impl<'a, T: ?Sized> Deref for AliasTrackingRefMut<'a, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We hold an exclusive borrow of the value.
        unsafe { self.value.as_ref() }
    }
}

impl<'b, T: ?Sized> DerefMut for AliasTrackingRefMut<'b, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We hold an exclusive borrow of the value.
        unsafe { self.value.as_mut() }
    }
}

/// `CloneUnsafe` is a trait that allows us to have the `AliasTrackingRefCell`
/// implement `Clone` for that type. The `clone_unsafe` method is unsafe because
/// it does not create an independent copy of the underlying type but instead
/// the returned value will be tracked like any other alias, and borrow-checking
/// rules will be enforced across both cells.
pub trait CloneUnsafe {
    unsafe fn clone_unsafe(&self) -> Self;
}

impl<T: CloneUnsafe> Clone for AliasTrackingRefCell<T> {
    fn clone(&self) -> Self {
        Self {
            alias_tracker: self.alias_tracker.clone(),
            // SAFETY: We use the alias tracker to ensure that we are handling the underlying
            // value safely.
            value: UnsafeCell::new(unsafe { self.value.get().as_ref().unwrap().clone_unsafe() }),
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::Tensor;
    use crate::bridge::ffi::test_make_alias;
    use crate::bridge::ffi::test_make_tensor;

    #[should_panic]
    #[test]
    fn clone_then_mut_borrow() {
        let t = test_make_tensor();
        let cell = AliasTrackingRefCell::new(t);
        let _borrow = cell.borrow();

        let clone = cell.clone();
        // uh oh!
        clone.borrow_mut();
    }

    #[should_panic]
    #[test]
    fn alias_mut_borrow() {
        let t = test_make_tensor();
        #[allow(clippy::undocumented_unsafe_blocks)]
        let t_alias = unsafe { test_make_alias(&t) };
        let cell = AliasTrackingRefCell::new(t);
        let cell_alias = AliasTrackingRefCell::new_with_alias(t_alias, &cell);
        let _borrow = cell.borrow();

        // uh oh!
        cell_alias.borrow_mut();
    }

    #[test]
    fn alias_mut_borrow_scoped() {
        let t = test_make_tensor();
        #[allow(clippy::undocumented_unsafe_blocks)]
        let t_alias = unsafe { test_make_alias(&t) };
        let cell = AliasTrackingRefCell::new(t);
        let cell_alias = AliasTrackingRefCell::new_with_alias(t_alias, &cell);
        {
            let _borrow = cell.borrow();
        }

        // This is fine, the previous borrow went away.
        {
            cell_alias.borrow_mut();
        }
        // Same.
        {
            cell_alias.borrow();
        }
    }

    #[test]
    fn try_borrow() {
        let t = test_make_tensor();
        let cell = AliasTrackingRefCell::new(t);
        {
            let b1 = cell.try_borrow();
            assert!(b1.is_ok());
            let b2 = cell.try_borrow();
            assert!(b2.is_ok());
            let b3 = cell.try_borrow();
            assert!(b3.is_ok());

            let borrow_mut = cell.try_borrow_mut();
            assert!(borrow_mut.is_err());
        }
        let borrow_mut = cell.try_borrow_mut();
        assert!(borrow_mut.is_ok());
        let borrow = cell.try_borrow();
        assert!(borrow.is_err());
    }

    #[test]
    fn serialize() -> Result<()> {
        let c1 = AliasTrackingRefCell::new(test_make_tensor());
        let buf = bincode::serialize(&c1)?;
        let c2: AliasTrackingRefCell<Tensor> = bincode::deserialize(&buf)?;
        assert_eq!(*c1.borrow(), *c2.borrow());
        Ok(())
    }
}
