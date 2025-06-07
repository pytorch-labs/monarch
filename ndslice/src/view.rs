/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! View planning and design
//!
//! This module implements `Slice::view(...)` with semantics analogous
//! to `torch.Tensor.view(...)`. The goal is to reinterpret the memory
//! layout of an existing `Slice` without copying, assuming the new
//! shape is element-count compatible and layout-compatible with the
//! base slice.
//!
//! # Objective
//!
//! Provide an API like:
//!
//! ```ignore
//! let v: View<'_> = slice.view(&[2, 3, 4])?;
//! let reshaped: Slice = v.into_slice()?;
//! ```
//!
//! ## Requirements
//!
//! - The new shape must have the same number of elements as the base.
//!   ✅
//! - The new shape must be layout-compatible — i.e. its logical
//!   traversal must match the base slice's physical memory order. ⏳
//!   (partially enforced)
//! - No memory copying or reallocation is performed. ✅
//! - The returned `View` supports further transformations (e.g.
//!   `transpose`, etc.) before being finalized as a `Slice`. ⏳
//!
//! ## Stride Compatibility (Contiguity-like Condition)
//!
//! To match PyTorch semantics, the layout of the proposed view must
//! be compatible with the base slice's strides. This requires that
//! the dimensions of the view either:
//!
//! - Correspond directly to dimensions of the base, or
//! - Span across multiple base dimensions whose strides satisfy the
//!   contiguity-like condition:
//!
//! ```text
//! ∀ i = d .. d+k−1:
//!     stride[i] == stride[i+1] * size[i+1]
//! ```
//!
//! This condition ensures the new view can be projected onto the base
//! memory without ambiguity or aliasing. If this fails, `view()` must
//! return an error. We currently do not support automatic copying to
//! make incompatible views possible.
//!
//! # Design
//!
//! We introduce a `View<'a>` type that holds:
//!
//! ```ignore
//! pub struct View<'a> {
//!     base: &'a dyn LayoutMap,
//!     offset: usize,
//!     sizes: Vec<usize>,
//!     strides: Vec<usize>,
//! }
//! ```
//!
//! The `View` acts as a deferred layout reinterpretation over a base
//! `LayoutMap`. It allows chaining and validation without eagerly
//! materializing a new `Slice`.
//!
//! ## Responsibilities
//!
//! - ✅ `View::new(base, sizes)`:
//!     - Computes offset from base
//!     - Computes row-major strides for sizes
//!     - Validates that total element count matches base
//!     - Constructs a `View` (without validating layout yet)
//!
//! - ⏳ `View::validate_layout()`:
//!     - Iterates over all coordinates in the view
//!     - Maps each coordinate to a linear offset via the view
//!     - Uses `base.coord_of(offset)` to check round-trip validity
//!     - Ensures all addresses produced by the view are reachable in
//!       the base
//!
//! - ⏳ `View::into_slice()`:
//!     - Not yet implemented
//!     - Will run `validate_layout()`
//!     - Will return a new `Slice { offset, sizes, strides }`
//!
//! ## Slice API
//!
//! ✅
//! ```ignore
//! impl Slice {
//!     pub fn view(&self, new_shape: &[usize]) -> Result<View<'_>, SliceError> {
//!         View::new(self, new_shape.to_vec())
//!     }
//! }
//! ```
//!
//! ## Error Handling
//!
//! ✅ View construction and finalization may fail if the shape or
//! layout is incompatible with the base slice. To report these
//! failures, we added:
//!
//! ```ignore
//! #[derive(Error, Debug)]
//! pub enum SliceError {
//!     ...
//!     #[error("incompatible view: {reason}")]
//!     IncompatibleView { reason: String },
//! }
//! ```
//!
//! Used to signal:
//! - Mismatched element count ✅
//! - Unreachable origin offset ✅
//! - Layout incompatibility ⏳
//!
//! # Summary
//!
//! This design mirrors PyTorch’s `Tensor.view()` behavior while
//! embracing Rust’s type system and layout abstraction. The `View`
//! type is a pure, cheap, composable transformation that defers
//! validation and finalization until explicitly requested.
//!
//! ## Row-Major to Column-Major Conversion
//!
//! As a proof of concept for the generality of `View`, we implement a
//! transformation that reinterprets a row-major `Slice` as
//! column-major — and vice versa — via `View::transpose(...)`, by
//! modifying strides while preserving sizes and offset.
//!
//! For example:
//!
//! ```ignore
//! // Original row-major Slice:
//! sizes:   [3, 4]
//! strides: [4, 1]
//!
//! // View as column-major (via transpose):
//! sizes:   [4, 3]
//! strides: [1, 4]
//! ```
use crate::layout::LayoutMap;
use crate::slice::SliceError;

pub struct View<'a> {
    pub base: &'a dyn LayoutMap,
    pub offset: usize,
    pub sizes: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<'a> View<'a> {
    /// Constructs a new `View` over an existing layout with the given
    /// shape.
    ///
    /// This function creates a logical reinterpretation of the `base`
    /// layout using a new shape and standard row-major strides. The
    /// result is a lightweight, composable transformation that does
    /// not copy or reallocate memory.
    ///
    /// # Invariants established by this constructor:
    ///
    /// - The new shape's element count matches that of the base:
    ///   `∏(sizes) == ∏(base.shape())`
    ///
    /// - The new view starts at coordinate `[0, 0, ..., 0]`, which is
    ///   guaranteed to map to a valid flat offset in the base layout
    ///   (`offset_of(origin)`).
    ///
    /// - The `strides` field defines a valid **row-major layout**
    ///   over `sizes`, such that `offset + dot(strides, coord)`
    ///   computes the flat offset of any coordinate in the view.
    ///
    /// # What is NOT yet validated:
    ///
    /// - This function does **not** check that the entire layout
    ///   defined by the view's shape and strides is compatible with
    ///   the base layout.
    ///
    /// - In particular, it does **not** verify that all coordinates
    ///   in the view map to addresses that are valid (reachable and
    ///   non-aliased) in the base.
    ///
    /// - It also does not validate that the stride pattern is
    ///   layout-compatible with the base's physical memory ordering
    ///   (e.g., contiguity conditions).
    ///
    /// # Why validation is deferred:
    ///
    /// This design mirrors PyTorch’s `Tensor.view()` behavior:
    /// - `View::new(...)` is cheap and composable
    /// - Full validation is performed only at finalization (e.g., in
    ///   `into_slice()`), after all transformations (e.g.,
    ///   `.view().transpose().reshape()`) are complete.
    ///
    /// This enables flexible and efficient layout manipulation
    /// without prematurely committing to a particular representation.
    ///
    /// # Errors
    ///
    /// Returns `SliceError::IncompatibleView` if:
    /// - The total number of elements in `sizes` does not match the
    ///   base
    /// - The origin offset `[0, 0, ..., 0]` is not reachable in the
    ///   base
    pub fn new(base: &'a dyn LayoutMap, sizes: Vec<usize>) -> Result<Self, SliceError> {
        // Compute standard row-major strides.
        let mut strides = vec![1; sizes.len()];
        for i in (0..sizes.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * sizes[i + 1];
        }

        let view_elem_count = sizes.iter().product::<usize>();
        let base_elem_count = base.sizes().iter().product::<usize>();
        if view_elem_count != base_elem_count {
            return Err(SliceError::IncompatibleView {
                reason: format!(
                    "element count mismatch: base has {}, view wants {}",
                    base_elem_count, view_elem_count
                ),
            });
        }

        let origin = vec![0; sizes.len()];
        let offset = base
            .offset_of(&origin)
            .map_err(|_e| SliceError::IncompatibleView {
                reason: "could not compute origin offset in base layout".into(),
            })?;

        Ok(Self {
            base,
            offset,
            sizes,
            strides,
        })
    }
}
