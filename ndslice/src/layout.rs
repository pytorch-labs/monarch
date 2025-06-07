/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::slice::Slice;
use crate::slice::SliceError;

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
