use std::any::Any;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use atomic_refcell::AtomicRefCell;
use derive_more::Display;
use thiserror::Error;

use crate::RValue;
use crate::cell::AliasTrackingRef;
use crate::cell::AliasTrackingRefMut;

/// Errors that can occur while calling an operator.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum BorrowError {
    #[error("cannot borrow with type: {0}")]
    InvalidBorrow(BorrowType),
}

/// Abstracts over the different types of borrows we can have.
#[derive(Debug)]
pub enum Borrow<'a> {
    // Dead code because we never access these, just hold onto them as guards.
    #[allow(dead_code)]
    Shared(AliasTrackingRef<'a, dyn Any>),
    #[allow(dead_code)]
    Mutable(AliasTrackingRefMut<'a, dyn Any>),
}

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq)]
pub enum BorrowType {
    Shared,
    Mutable,
}

/// A helper that batches multiple borrows for a single borrower, deduping them
/// so we don't accidentally borrow the same alias twice.
#[derive(Debug)]
pub struct MultiBorrow<'a> {
    cells: Vec<(&'a crate::cell::AliasTrackingRefCell<dyn Any>, BorrowType)>,
}

impl<'a> MultiBorrow<'a> {
    pub fn new() -> Self {
        Self { cells: Vec::new() }
    }

    pub fn borrow(&self) -> Result<Vec<Borrow>, BorrowError> {
        // Dedupe borrows so that we don't accidentally borrow the same alias twice.
        let mut alias_ptrs: HashMap<
            *const AtomicRefCell<()>,
            (&crate::cell::AliasTrackingRefCell<dyn Any>, BorrowType),
        > = HashMap::new();
        for (cell, borrow_type) in &self.cells {
            let alias_ptr = cell.alias_ptr();

            match alias_ptrs.entry(alias_ptr) {
                Entry::Vacant(entry) => {
                    entry.insert((cell, *borrow_type));
                }
                Entry::Occupied(mut entry) => match (entry.get(), borrow_type) {
                    // Upgrade a shared borrow to a mutable borrow.
                    ((_, BorrowType::Shared), BorrowType::Mutable) => {
                        entry.insert((cell, BorrowType::Mutable));
                    }
                    // Otherwise just leave the existing entry as it is.
                    _ => (),
                },
            }
        }

        let mut ret = Vec::new();
        for (_, (cell, borrow_type)) in alias_ptrs {
            match borrow_type {
                BorrowType::Mutable => ret.push(Borrow::Mutable(
                    cell.try_borrow_mut()
                        .map_err(|_| BorrowError::InvalidBorrow(borrow_type))?,
                )),
                BorrowType::Shared => ret.push(Borrow::Shared(
                    cell.try_borrow()
                        .map_err(|_| BorrowError::InvalidBorrow(borrow_type))?,
                )),
            }
        }
        Ok(ret)
    }

    pub fn add(&mut self, arg: &'a RValue, borrow_type: BorrowType) {
        match arg {
            RValue::Tensor(cell) => {
                self.cells.push((cell, borrow_type));
            }
            RValue::TensorList(cells) => {
                for cell in cells {
                    // If this is a write to a tensor list, just borrow every
                    // tensor mutably.
                    self.cells.push((cell, borrow_type));
                }
            }
            RValue::Opaque(val) => self.cells.push((val, borrow_type)),
            _ => (),
        };
    }
}
