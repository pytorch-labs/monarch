/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::slice::Slice;
use crate::slice::SliceError;
use crate::view::View;

mod sealed {
    // Private trait — only types in this module can implement it
    pub trait Sealed {}
}

/// A trait for memory layouts that map multidimensional coordinates
/// (in `ℕⁿ`) to linear memory offsets (`ℕ¹`) via an affine
/// transformation.
///
/// This abstraction describes how an `n`-dimensional shape is laid
/// out in memory using a strided affine map:
///
/// ```text
/// offset_of(x) = offset + dot(strides, x)
/// ```
///
/// This corresponds to an affine function `ℕⁿ → ℕ¹`, where `x` is a
/// coordinate in logical space, `strides` encodes layout, and
/// `offset` is the base address.
///
/// Implementors define how coordinates in `n`-dimensional space are
/// translated to flat memory locations, enabling support for
/// row-major, column-major, and custom layouts.
pub trait LayoutMap: sealed::Sealed {
    /// The number of dimensions in the domain of the map.
    fn rank(&self) -> usize;

    /// The shape of the domain (number of elements per dimension).
    fn sizes(&self) -> &[usize];

    /// Maps a multidimensional coordinate to a linear memory offset.
    fn offset_of(&self, coord: &[usize]) -> Result<usize, SliceError>;
}

/// A trait for memory layouts that support inverse mapping from
/// linear offsets (`ℕ¹`) back to multidimensional coordinates (in
/// `ℕⁿ`).
///
/// This defines the inverse of the affine layout transformation given
/// by [`LayoutMap::offset_of`], where an offset is mapped back to a
/// coordinate in logical space—if possible.
///
/// Not all layouts are invertible: aliasing, gaps, or padding may
/// prevent a one-to-one correspondence between coordinates and
/// offsets. However, standard layouts like contiguous row-major or
/// column-major typically do support inversion.
///
/// Implementors define how to reconstruct the coordinate `x ∈ ℕⁿ` for
/// a given offset `o ∈ ℕ`, or return `None` if no such coordinate
/// exists.
pub trait LayoutMapInverse: sealed::Sealed {
    /// Computes the multidimensional coordinate for a given linear
    /// offset, or returns `None` if the offset is out of bounds.
    fn coord_of(&self, offset: usize) -> Option<Vec<usize>>;
}

/// Extension trait for applying shape transformations to layout-aware types.
///
/// This trait enables ergonomic, composable construction of [`View`]s
/// over types that implement [`LayoutMap`] — including [`Slice`] and
/// other layout-aware data structures. It supports zero-copy
/// reinterpretation of memory layout, subject to shape and stride
/// compatibility.
///
/// # Purpose
///
/// - Enables `.view(&[...]) -> Result<View>` syntax
/// - Defers layout validation until explicitly finalized
/// - Facilitates transformation chaining (e.g., `view → transpose`)
///   using composable `View` operations
///
/// # Requirements
///
/// Only [`LayoutMap`] is required, which allows forward coordinate-to-offset
/// mapping. Inversion (via [`LayoutMapInverse`]) is **not** required to
/// construct a `View`, only to finalize it into a [`Slice`] (e.g., via
/// `View::into_slice()`).
///
/// # Behavior
///
/// Calling `.view(&sizes)`:
/// - Computes row-major strides over `sizes`
/// - Computes the offset of `[0, 0, ..., 0]` in the base layout
/// - Constructs a [`View`] with new shape and strides
/// - **Does not** validate full layout compatibility — that is
///   deferred
///
/// # Example
///
/// ```rust
/// use ndslice::Slice
/// use ndslice::layout::LayoutTransformExt;
///
/// let base = Slice::new_row_major([2, 3]); // 2×3 row-major layout
/// let view = base.view(&[3, 2])?; // Valid reshape: 6 elements
/// ```
///
/// # Notes
///
/// - If `sizes` do not multiply to the same number of elements as the
///   base, an error is returned.
/// - If the new origin offset is not reachable in the base layout,
///   the view is rejected.
///
/// # See Also
///
/// - [`View`] — Lazy layout reinterpretation
/// - [`View::new`] — Raw constructor
/// - [`LayoutMap`] — Forward affine mapping trait
pub trait LayoutTransformExt {
    fn view(&self, sizes: &[usize]) -> Result<View<'_>, SliceError>;
}

/// Blanket implementation of [`LayoutTransformExt`] for all types
/// that implement [`LayoutMap`].
///
/// This enables ergonomic access to `.view(...)` on any
/// layout-compatible type, such as [`Slice`], without modifying the
/// type itself.
///
/// The returned [`View`] is a lightweight logical reinterpretation of
/// the layout using row-major semantics. It does not yet validate
/// layout compatibility; that logic will be implemented as part of
/// `View::into_slice()` in a future revision.
///
/// # Example
///
/// ```rust
/// use ndslice::Slice
/// use ndslice::layout::LayoutTransformExt;
///
/// let base = Slice::new_row_major([2, 3]);
/// let view = base.view(&[3, 2])?;
/// ```
impl<T> LayoutTransformExt for T
where
    T: LayoutMap,
{
    fn view(&self, sizes: &[usize]) -> Result<View<'_>, SliceError> {
        View::new(self, sizes.to_vec())
    }
}

impl sealed::Sealed for Slice {}

impl LayoutMap for Slice {
    fn rank(&self) -> usize {
        self.sizes().len()
    }

    fn sizes(&self) -> &[usize] {
        self.sizes()
    }

    fn offset_of(&self, coord: &[usize]) -> Result<usize, SliceError> {
        if coord.len() != self.rank() {
            return Err(SliceError::InvalidDims {
                expected: self.rank(),
                got: coord.len(),
            });
        }

        // Dot product ∑ᵢ (strideᵢ × coordᵢ)
        let linear_offset = self
            .strides()
            .iter()
            .zip(coord)
            .map(|(s, i)| s * i)
            .sum::<usize>();

        Ok(self.offset() + linear_offset)
    }
}

impl LayoutMapInverse for Slice {
    fn coord_of(&self, value: usize) -> Option<Vec<usize>> {
        let mut pos = value.checked_sub(self.offset())?;
        let mut result = vec![0; self.rank()];

        let mut dims: Vec<_> = self
            .strides()
            .iter()
            .zip(self.sizes().iter().enumerate())
            .collect();

        dims.sort_by_key(|&(stride, _)| *stride);

        // Invert: offset = base + ∑ᵢ (strideᵢ × coordᵢ)
        // Solve for coordᵢ by peeling off largest strides first:
        //   coordᵢ = ⌊pos / strideᵢ⌋
        //   pos   -= coordᵢ × strideᵢ
        // If any coordᵢ ≥ sizeᵢ or pos ≠ 0 at the end, the offset is
        // invalid.
        for &(stride, (i, &size)) in dims.iter().rev() {
            let index = if size > 1 { pos / stride } else { 0 };
            if index >= size {
                return None;
            }
            result[i] = index;
            pos -= index * stride;
        }

        (pos == 0).then_some(result)
    }
}

impl<'a> sealed::Sealed for View<'a> {}

impl<'a> LayoutMap for View<'a> {
    fn rank(&self) -> usize {
        self.sizes.len()
    }

    fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    fn offset_of(&self, coord: &[usize]) -> Result<usize, SliceError> {
        if coord.len() != self.sizes.len() {
            return Err(SliceError::InvalidDims {
                expected: self.sizes.len(),
                got: coord.len(),
            });
        }

        // Compute offset = base_offset + dot(strides, coord)
        let offset = self
            .strides
            .iter()
            .zip(coord.iter())
            .map(|(s, i)| s * i)
            .sum::<usize>();

        Ok(self.offset + offset)
    }
}
