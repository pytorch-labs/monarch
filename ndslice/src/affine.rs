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

/// A trait for affine maps from integer coordinates to linear memory
/// offsets.
///
/// This abstraction captures multidimensional layouts that can be
/// interpreted as an affine transformation: `f(x) = offset +
/// dot(strides, x)`.
///
/// Implementors of this trait define how multidimensional indices map
/// to linear locations in memory.
pub trait AffineMap: sealed::Sealed {
    /// The number of dimensions in the domain of the map.
    fn rank(&self) -> usize;

    /// The shape of the domain (number of elements per dimension).
    fn sizes(&self) -> &[usize];

    /// Maps a multidimensional coordinate to a linear memory offset.
    fn offset_of(&self, coord: &[usize]) -> Result<usize, SliceError>;
}

/// A trait for affine maps that support inverse lookup from linear
/// offsets back to multidimensional coordinates.
///
/// This captures the inverse of the layout transformation defined by
/// [`AffineMap::offset_of`]. Not all affine maps are invertible, but
/// common layouts (e.g., row-major slices) are.
pub trait AffineMapInverse: sealed::Sealed {
    /// Computes the multidimensional coordinate for a given linear
    /// offset, or returns `None` if the offset is out of bounds.
    fn coord_of(&self, offset: usize) -> Option<Vec<usize>>;
}

impl sealed::Sealed for Slice {}

impl AffineMap for Slice {
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

impl AffineMapInverse for Slice {
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
