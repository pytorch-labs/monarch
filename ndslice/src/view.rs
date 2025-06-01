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
//! - The new shape must be layout-compatible — i.e. its logical
//!   traversal must match the base slice's physical memory order.
//! - No memory copying or reallocation is performed.
//! - The returned `View` supports further transformations (e.g.
//!   `reshape`, `transpose`, etc.) before being finalized as a `Slice`.
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
//!     base: &'a dyn AffineMap, // The original layout we are viewing
//!     offset: usize,           // The linear offset at the logical origin of the view
//!     sizes: Vec<usize>,       // New shape
//!     strides: Vec<usize>,     // Strides defining the view layout
//! }
//! ```
//!
//! The `View` acts as a deferred layout reinterpretation over a base
//! `AffineMap`. It allows chaining and validation without eagerly
//! materializing a new `Slice`.
//!
//! ## Responsibilities
//!
//! - `View::new(base, sizes)`:
//!     - Computes offset from base
//!     - Computes row-major strides for sizes
//!     - Validates that total element count matches base
//!     - Constructs a `View` (without validating layout yet)
//!
//! - `View::validate_layout()`:
//!     - Iterates over all coordinates in the view
//!     - Maps each coordinate to a linear offset via the view
//!     - Uses `base.coord_of(offset)` to check round-trip validity
//!     - Ensures all addresses produced by the view are reachable in
//!      the base
//!
//! - `View::into_slice()`:
//!     - Runs `validate_layout()`
//!     - Returns a new `Slice { offset, sizes, strides }`
//!
//! - `AffineMap` is implemented for `View`, allowing it to
//!   participate in layout-aware operations.
//!
//! ## Slice API
//!
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
//! View construction and finalization may fail if the shape or layout
//! is incompatible with the base slice. To report these failures, we
//! extend the `SliceError` enum with a new variant:
//!
//! ```ignore
//! #[derive(Error, Debug)]
//! pub enum SliceError {
//!     // existing variants...
//!     #[error("invalid dims: expected {expected}, got {got}")]
//!     InvalidDims { expected: usize, got: usize },
//!
//!     #[error("nonrectangular shape")]
//!     NonrectangularShape,
//!
//!     #[error("nonunique strides")]
//!     NonuniqueStrides,
//!
//!     #[error("stride {stride} must be larger than size of previous space {space}")]
//!     StrideTooSmall { stride: usize, space: usize },
//!
//!     #[error("index {index} out of range {total}")]
//!     IndexOutOfRange { index: usize, total: usize },
//!
//!     #[error("value {value} not in slice")]
//!     ValueNotInSlice { value: usize },
//!
//!     // new:
//!     #[error("incompatible view: {reason}")]
//!     IncompatibleView { reason: String },
//! }
//! ```
//!
//! This variant is used to signal structural errors such as:
//! - Mismatched element count
//! - Layout incompatibility
//! - Invalid or unreachable offset projections
//!
//! ## Testing Plan
//!
//! - `test_view_success()`:
//!     - Create a row-major slice
//!     - Call `.view(...)` to a valid shape
//!     - Finalize with `.into_slice()`
//!     - Assert offsets and shapes are correct
//!
//! - `test_view_count_mismatch()`:
//!     - Shape product mismatch → error
//!
//! - `test_view_layout_mismatch()`:
//!     - Construct a noncontiguous slice, try to view as row-major
//!     - Layout validation fails
//!
//! - `test_view_chainable()`:
//!     - Call `.view().into_slice()` and validate result
//!
//! ## Future Extensions
//!
//! - `View::transpose(dim0, dim1)`
//! - `View::reshape(...)` (non-row-major)
//! - `View::squeeze()`
//! - `impl AffineMapInverse for View`
//! - `View::to_owned()`
//!
//! # Summary
//!
//! This design mirrors PyTorch’s `Tensor.view()` behavior while
//! embracing Rust’s type system and layout abstraction. The `View`
//! type is a pure, cheap, composable transformation that defers
//! validation and finalization until explicitly requested.
//! ## Row-Major to Column-Major Conversion
//!
//! As a proof of concept for the generality of `View`, we implement a
//! transformation that reinterprets a row-major `Slice` as
//! column-major — and vice versa — by modifying strides while
//! preserving sizes and offset.
//!
//! For example:
//!
//! ```ignore
//! // Original row-major Slice:
//! sizes:   [3, 4]
//! strides: [4, 1]
//!
//! // View as column-major:
//! sizes:   [3, 4]
//! strides: [1, 3]
//! ```
//!
//! This demonstrates that `View` can express not just reshaping, but
//! full layout reordering under affine constraints, without copying
//! or loss of generality. These transformations are expressed by
//! modifying the `strides` field in `View`, while keeping the base
//! memory unchanged.
//!
//! This reinforces the decision to model layouts abstractly using
//! `AffineMap`, and to use `View` as a composable affine
//! transformation builder.
