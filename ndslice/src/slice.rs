/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde::Deserialize;
use serde::Serialize;

/// The type of error for slice operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SliceError {
    #[error("invalid dims: expected {expected}, got {got}")]
    InvalidDims { expected: usize, got: usize },

    #[error("nonrectangular shape")]
    NonrectangularShape,

    #[error("nonunique strides")]
    NonuniqueStrides,

    #[error("stride {stride} must be larger than size of previous space {space}")]
    StrideTooSmall { stride: usize, space: usize },

    #[error("index {index} out of range {total}")]
    IndexOutOfRange { index: usize, total: usize },

    #[error("value {value} not in slice")]
    ValueNotInSlice { value: usize },

    #[error("incompatible view: {reason}")]
    IncompatibleView { reason: String },
}

/// Slice is a compact representation of indices into the flat
/// representation of an n-dimensional array. Given an offset, sizes of
/// each dimension, and strides for each dimension, Slice can compute
/// indices into the flat array.
///
/// For example, the following describes a dense 4x4x4 array in row-major
/// order:
/// ```
/// # use ndslice::Slice;
/// let s = Slice::new(0, vec![4, 4, 4], vec![16, 4, 1]).unwrap();
/// assert!(s.iter().eq(0..(4 * 4 * 4)));
/// ```
///
/// Slices allow easy slicing by subsetting and striding. For example,
/// we can fix the index of the second dimension by dropping it and
/// adding that index (multiplied by the previous size) to the offset.
///
/// ```
/// # use ndslice::Slice;
/// let s = Slice::new(0, vec![2, 4, 2], vec![8, 2, 1]).unwrap();
/// let selected_index = 3;
/// let sub = Slice::new(2 * selected_index, vec![2, 2], vec![8, 1]).unwrap();
/// let coords = [[0, 0], [0, 1], [1, 0], [1, 1]];
/// for coord @ [x, y] in coords {
///     assert_eq!(
///         sub.location(&coord).unwrap(),
///         s.location(&[x, 3, y]).unwrap()
///     );
/// }
/// ```
// TODO: Consider representing this by arrays parameterized by the slice
// dimensionality.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Slice {
    offset: usize,
    sizes: Vec<usize>,
    strides: Vec<usize>,
}

impl Slice {
    /// Create a new Slice with the provided offset, sizes, and
    /// strides. New performs validation to ensure that sizes and strides
    /// are compatible:
    ///   - They have to be the same length (i.e., same number of dimensions)
    ///   - They have to be rectangular (i.e., stride n+1 has to evenly divide into stride n)
    ///   - Strides must be nonoverlapping (each stride has to be larger than the previous space)
    pub fn new(offset: usize, sizes: Vec<usize>, strides: Vec<usize>) -> Result<Self, SliceError> {
        if sizes.len() != strides.len() {
            return Err(SliceError::InvalidDims {
                expected: sizes.len(),
                got: strides.len(),
            });
        }
        let mut combined: Vec<(usize, usize)> =
            strides.iter().cloned().zip(sizes.iter().cloned()).collect();
        combined.sort();

        let mut prev_stride: Option<usize> = None;
        let mut prev_size: Option<usize> = None;
        let mut total: usize = 1;
        for (stride, size) in combined {
            if let Some(prev_stride) = prev_stride {
                if stride % prev_stride != 0 {
                    return Err(SliceError::NonrectangularShape);
                }
                // Strides for single element dimensions can repeat, because they are unused
                if stride == prev_stride && size != 1 && prev_size.unwrap_or(1) != 1 {
                    return Err(SliceError::NonuniqueStrides);
                }
            }
            if total > stride {
                return Err(SliceError::StrideTooSmall {
                    stride,
                    space: total,
                });
            }
            total = stride * size;
            prev_stride = Some(stride);
            prev_size = Some(size);
        }

        Ok(Slice {
            offset,
            sizes,
            strides,
        })
    }

    /// Create a new slice of the given sizes in row-major order.
    pub fn new_row_major(sizes: impl Into<Vec<usize>>) -> Self {
        let sizes = sizes.into();
        // "flip it and reverse it" --Missy Elliott
        let mut strides: Vec<usize> = sizes.clone();
        let _ = strides.iter_mut().rev().fold(1, |acc, n| {
            let next = *n * acc;
            *n = acc;
            next
        });
        Self {
            offset: 0,
            sizes,
            strides,
        }
    }

    /// Create one celled slice.
    pub fn new_single_multi_dim_cell(dims: usize) -> Self {
        Self {
            offset: 0,
            sizes: vec![1; dims],
            strides: vec![1; dims],
        }
    }

    /// The number of dimensions in this slice.
    pub fn num_dim(&self) -> usize {
        self.sizes.len()
    }

    /// This is the offset from which the first value in the Slice begins.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The shape of the slice; that is, the size of each dimension.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// The strides of the slice; that is, the distance between each
    /// element at a given index in the underlying array.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Return the location of the provided coordinates.
    pub fn location(&self, coord: &[usize]) -> Result<usize, SliceError> {
        if coord.len() != self.sizes.len() {
            return Err(SliceError::InvalidDims {
                expected: self.sizes.len(),
                got: coord.len(),
            });
        }
        Ok(self.offset
            + coord
                .iter()
                .zip(&self.strides)
                .map(|(pos, stride)| pos * stride)
                .sum::<usize>())
    }

    /// Return the coordinates of the provided value in the n-d space of this
    /// Slice.
    pub fn coordinates(&self, value: usize) -> Result<Vec<usize>, SliceError> {
        let mut pos = value
            .checked_sub(self.offset)
            .ok_or(SliceError::ValueNotInSlice { value })?;
        let mut result = vec![0; self.sizes.len()];
        let mut sorted_info: Vec<_> = self
            .strides
            .iter()
            .zip(self.sizes.iter().enumerate())
            .collect();
        sorted_info.sort_by_key(|&(stride, _)| *stride);
        for &(stride, (i, &size)) in sorted_info.iter().rev() {
            let (index, new_pos) = if size > 1 {
                (pos / stride, pos % stride)
            } else {
                (0, pos)
            };
            if index >= size {
                return Err(SliceError::ValueNotInSlice { value });
            }
            result[i] = index;
            pos = new_pos;
        }
        if pos != 0 {
            return Err(SliceError::ValueNotInSlice { value });
        }
        Ok(result)
    }

    /// Given a logical index (in row-major order), return the
    /// physical memory offset of that element according to this
    /// slice’s layout.
    ///
    /// The index is interpreted as a position in row-major traversal
    /// — that is, iterating across columns within rows. This method
    /// decodes the index into a multidimensional coordinate, and then
    /// applies the slice’s `strides` to compute the memory offset of
    /// that coordinate.
    ///
    /// For example, with shape `[3, 4]` (3 rows, 4 columns) and
    /// column-major layout:
    ///
    /// ```text
    /// sizes  = [3, 4]         // rows, cols
    /// strides = [1, 3]        // column-major: down, then right
    ///
    /// Logical matrix:
    ///   A  B  C  D
    ///   E  F  G  H
    ///   I  J  K  L
    ///
    /// Memory layout:
    /// offset 0  → [0, 0] = A
    /// offset 1  → [1, 0] = E
    /// offset 2  → [2, 0] = I
    /// offset 3  → [0, 1] = B
    /// offset 4  → [1, 1] = F
    /// offset 5  → [2, 1] = J
    /// offset 6  → [0, 2] = C
    /// offset 7  → [1, 2] = G
    /// offset 8  → [2, 2] = K
    /// offset 9  → [0, 3] = D
    /// offset 10 → [1, 3] = H
    /// offset 11 → [2, 3] = L
    ///
    /// Then:
    ///   index = 1  → coordinate [0, 1]  → offset = 0*1 + 1*3 = 3
    /// ```
    ///
    /// Returns an error if `index >= product(sizes)`.
    pub fn get(&self, index: usize) -> Result<usize, SliceError> {
        let mut val = self.offset;
        let mut rest = index;
        let mut total = 1;
        for (size, stride) in self.sizes.iter().zip(self.strides.iter()).rev() {
            total *= size;
            val += (rest % size) * stride;
            rest /= size;
        }
        if index < total {
            Ok(val)
        } else {
            Err(SliceError::IndexOutOfRange { index, total })
        }
    }

    /// The total length of the slice's indices.
    pub fn len(&self) -> usize {
        self.sizes.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterator over the slice's indices.
    pub fn iter(&self) -> SliceIterator {
        SliceIterator {
            slice: self,
            pos: CartesianIterator::new(&self.sizes),
        }
    }

    /// Iterator over sub-dimensions of the slice.
    pub fn dim_iter(&self, dims: usize) -> DimSliceIterator {
        DimSliceIterator {
            pos: CartesianIterator::new(&self.sizes[0..dims]),
        }
    }

    /// Returns the index into the flattened representation of `self` where
    /// `self[index] == value`.
    pub fn index(&self, value: usize) -> Result<usize, SliceError> {
        let coords = self.coordinates(value)?;
        let mut stride = 1;
        let mut result = 0;

        for (idx, size) in coords.iter().rev().zip(self.sizes.iter().rev()) {
            result += *idx * stride;
            stride *= size;
        }

        Ok(result)
    }

    /// The returned [`MapSlice`] is a view of this slice, with its elements
    /// mapped using the provided mapping function.
    pub fn map<T, F>(&self, mapper: F) -> MapSlice<'_, T, F>
    where
        F: Fn(usize) -> T,
    {
        MapSlice {
            slice: self,
            mapper,
        }
    }

    /// Returns a new [`Slice`] with the given shape by reinterpreting
    /// the layout of this slice.
    ///
    /// Constructs a new shape with standard row-major strides, using
    /// the same base offset. Returns an error if the reshaped view
    /// would access coordinates not valid in the original slice.
    ///
    /// # Requirements
    ///
    /// - This slice must be contiguous and have offset == 0.
    /// - The number of elements must match:
    ///   `self.sizes().iter().product() == new_sizes.iter().product()`
    /// - Each flat offset in the proposed view must be valid in `self`.
    ///
    /// # Errors
    ///
    /// Returns [`SliceError::IncompatibleView`] if:
    /// - The element count differs
    /// - The base offset is nonzero
    /// - Any offset in the view is not reachable in the original slice
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndslice::Slice;
    /// let base = Slice::new_row_major(&[2, 3, 4]);
    /// let reshaped = base.view(&[6, 4]).unwrap();
    /// ```
    pub fn view(&self, new_sizes: &[usize]) -> Result<Slice, SliceError> {
        let view_elems: usize = new_sizes.iter().product();
        let base_elems: usize = self.sizes().iter().product();

        // TODO: This version of `view` requires that `self` be
        // "dense":
        //
        //   - `self.offset == 0`
        //   - `self.strides` match the row-major layout for
        //     `self.sizes`
        //   - `self.len() == self.sizes.iter().product::<usize>()`
        //
        // Future iterations of this function will aim to relax or
        // remove the "dense" requirement where possible.

        if view_elems != base_elems {
            return Err(SliceError::IncompatibleView {
                reason: format!(
                    "element count mismatch: base has {}, view wants {}",
                    base_elems, view_elems
                ),
            });
        }
        if self.offset != 0 {
            return Err(SliceError::IncompatibleView {
                reason: format!("view requires base offset = 0, but found {}", self.offset),
            });
        }
        // Compute row-major strides.
        let mut new_strides = vec![1; new_sizes.len()];
        for i in (0..new_sizes.len().saturating_sub(1)).rev() {
            new_strides[i] = new_strides[i + 1] * new_sizes[i + 1];
        }

        // Validate that every address in the new view maps to a valid
        // coordinate in base.
        for coord in CartesianIterator::new(new_sizes) {
            #[allow(clippy::identity_op)]
            let offset_in_view = 0 + coord
                .iter()
                .zip(&new_strides)
                .map(|(i, s)| i * s)
                .sum::<usize>();

            if self.coordinates(offset_in_view).is_err() {
                return Err(SliceError::IncompatibleView {
                    reason: format!("offset {} not reachable in base", offset_in_view),
                });
            }
        }

        Ok(Slice {
            offset: 0,
            sizes: new_sizes.to_vec(),
            strides: new_strides,
        })
    }

    /// Returns a sub-slice of `self` starting at `starts`, of size `lens`.
    pub fn subview(&self, starts: &[usize], lens: &[usize]) -> Result<Slice, SliceError> {
        if starts.len() != self.num_dim() || lens.len() != self.num_dim() {
            return Err(SliceError::InvalidDims {
                expected: self.num_dim(),
                got: starts.len().max(lens.len()),
            });
        }

        for (d, (&start, &len)) in starts.iter().zip(lens).enumerate() {
            if start + len > self.sizes[d] {
                return Err(SliceError::IndexOutOfRange {
                    index: start + len,
                    total: self.sizes[d],
                });
            }
        }

        let offset = self.location(starts)?;
        Ok(Slice {
            offset,
            sizes: lens.to_vec(),
            strides: self.strides.clone(),
        })
    }
}

impl std::fmt::Display for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<'a> IntoIterator for &'a Slice {
    type Item = usize;
    type IntoIter = SliceIterator<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct SliceIterator<'a> {
    slice: &'a Slice,
    pos: CartesianIterator<'a>,
}

impl<'a> Iterator for SliceIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.pos.next() {
            None => None,
            Some(pos) => Some(self.slice.location(&pos).unwrap()),
        }
    }
}

/// Iterates over the Cartesian product of a list of dimension sizes.
///
/// Given a list of dimension sizes `[d₀, d₁, ..., dₖ₋₁]`, this yields
/// all coordinate tuples `[i₀, i₁, ..., iₖ₋₁]` where each `iⱼ ∈
/// 0..dⱼ`.
///
/// Coordinates are yielded in row-major order (last dimension varies
/// fastest).
pub struct DimSliceIterator<'a> {
    pos: CartesianIterator<'a>,
}

impl<'a> Iterator for DimSliceIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.pos.next()
    }
}

/// Iterates over all coordinate tuples in an N-dimensional space.
///
/// Yields each point in row-major order for the shape defined by
/// `dims`, where each coordinate lies in `[0..dims[i])`.
/// # Example
/// ```ignore
/// let iter = CartesianIterator::new(&[2, 3]);
/// let coords: Vec<_> = iter.collect();
/// assert_eq!(coords, vec![
///     vec![0, 0], vec![0, 1], vec![0, 2],
///     vec![1, 0], vec![1, 1], vec![1, 2],
/// ]);
/// ```
struct CartesianIterator<'a> {
    dims: &'a [usize],
    index: usize,
}

impl<'a> CartesianIterator<'a> {
    fn new(dims: &'a [usize]) -> Self {
        CartesianIterator { dims, index: 0 }
    }
}

impl<'a> Iterator for CartesianIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dims.iter().product::<usize>() {
            return None;
        }

        let mut result: Vec<usize> = vec![0; self.dims.len()];
        let mut rest = self.index;
        for (i, dim) in self.dims.iter().enumerate().rev() {
            result[i] = rest % dim;
            rest /= dim;
        }
        self.index += 1;
        Some(result)
    }
}

/// MapSlice is a view of the underlying Slice that maps each rank
/// into a different type.
pub struct MapSlice<'a, T, F>
where
    F: Fn(usize) -> T,
{
    slice: &'a Slice,
    mapper: F,
}

impl<'a, T, F> MapSlice<'a, T, F>
where
    F: Fn(usize) -> T,
{
    /// The underlying slice sizes.
    pub fn sizes(&self) -> &[usize] {
        &self.slice.sizes
    }

    /// The underlying slice strides.
    pub fn strides(&self) -> &[usize] {
        &self.slice.strides
    }

    /// The mapped value at the provided coordinates. See [`Slice::location`].
    pub fn location(&self, coord: &[usize]) -> Result<T, SliceError> {
        self.slice.location(coord).map(&self.mapper)
    }

    /// The mapped value at the provided index. See [`Slice::get`].
    pub fn get(&self, index: usize) -> Result<T, SliceError> {
        self.slice.get(index).map(&self.mapper)
    }

    /// The underlying slice length.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;
    use std::vec;

    use super::*;

    #[test]
    fn test_cartesian_iterator() {
        let dims = vec![2, 2, 2];
        let iter = CartesianIterator::new(&dims);
        let products: Vec<Vec<usize>> = iter.collect();
        assert_eq!(
            products,
            vec![
                vec![0, 0, 0],
                vec![0, 0, 1],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![1, 0, 0],
                vec![1, 0, 1],
                vec![1, 1, 0],
                vec![1, 1, 1],
            ]
        );
    }

    #[test]
    #[allow(clippy::explicit_counter_loop)]
    fn test_slice() {
        let s = Slice::new(0, vec![2, 3], vec![3, 1]).unwrap();
        for i in 0..4 {
            assert_eq!(s.get(i).unwrap(), i);
        }

        {
            // Test IntoIter
            let mut current = 0;
            for index in &s {
                assert_eq!(index, current);
                current += 1;
            }
        }

        let s = Slice::new(0, vec![3, 4, 5], vec![20, 5, 1]).unwrap();
        assert_eq!(s.get(3 * 4 + 1).unwrap(), 13);

        let s = Slice::new(0, vec![2, 2, 2], vec![4, 32, 1]).unwrap();
        assert_eq!(s.get(0).unwrap(), 0);
        assert_eq!(s.get(1).unwrap(), 1);
        assert_eq!(s.get(2).unwrap(), 32);
        assert_eq!(s.get(3).unwrap(), 33);
        assert_eq!(s.get(4).unwrap(), 4);
        assert_eq!(s.get(5).unwrap(), 5);
        assert_eq!(s.get(6).unwrap(), 36);
        assert_eq!(s.get(7).unwrap(), 37);

        let s = Slice::new(0, vec![2, 2, 2], vec![32, 4, 1]).unwrap();
        assert_eq!(s.get(0).unwrap(), 0);
        assert_eq!(s.get(1).unwrap(), 1);
        assert_eq!(s.get(2).unwrap(), 4);
        assert_eq!(s.get(4).unwrap(), 32);
    }

    #[test]
    fn test_slice_iter() {
        let s = Slice::new(0, vec![2, 3], vec![3, 1]).unwrap();
        assert!(s.iter().eq(0..6));

        let s = Slice::new(10, vec![10, 2], vec![10, 5]).unwrap();
        assert!(s.iter().eq((10..=105).step_by(5)));

        // Implementaion corresponds with Slice::get.
        assert!(s.iter().eq((0..s.len()).map(|i| s.get(i).unwrap())));
    }

    #[test]
    fn test_dim_slice_iter() {
        let s = Slice::new(0, vec![2, 3], vec![3, 1]).unwrap();
        let sub_dims: Vec<_> = s.dim_iter(1).collect();
        assert_eq!(sub_dims, vec![vec![0], vec![1]]);
    }

    #[test]
    fn test_slice_coordinates() {
        let s = Slice::new(0, vec![2, 3], vec![3, 1]).unwrap();
        assert_eq!(s.coordinates(0).unwrap(), vec![0, 0]);
        assert_eq!(s.coordinates(3).unwrap(), vec![1, 0]);
        assert_matches!(
            s.coordinates(6),
            Err(SliceError::ValueNotInSlice { value: 6 })
        );

        let s = Slice::new(10, vec![2, 3], vec![3, 1]).unwrap();
        assert_matches!(
            s.coordinates(6),
            Err(SliceError::ValueNotInSlice { value: 6 })
        );
        assert_eq!(s.coordinates(10).unwrap(), vec![0, 0]);
        assert_eq!(s.coordinates(13).unwrap(), vec![1, 0]);

        let s = Slice::new(0, vec![2, 1, 1], vec![1, 1, 1]).unwrap();
        assert_eq!(s.coordinates(1).unwrap(), vec![1, 0, 0]);
    }

    #[test]
    fn test_slice_index() {
        let s = Slice::new(0, vec![2, 3], vec![3, 1]).unwrap();
        assert_eq!(s.index(3).unwrap(), 3);
        assert!(s.index(14).is_err());

        let s = Slice::new(0, vec![2, 2], vec![4, 2]).unwrap();
        assert_eq!(s.index(2).unwrap(), 1);
    }

    #[test]
    fn test_slice_map() {
        let s = Slice::new(0, vec![2, 3], vec![3, 1]).unwrap();
        let m = s.map(|i| i * 2);
        assert_eq!(m.get(0).unwrap(), 0);
        assert_eq!(m.get(3).unwrap(), 6);
        assert_eq!(m.get(5).unwrap(), 10);
    }

    #[test]
    fn test_slice_size_one() {
        let s = Slice::new(0, vec![1, 1], vec![1, 1]).unwrap();
        assert_eq!(s.get(0).unwrap(), 0);
    }

    #[test]
    fn test_row_major() {
        let s = Slice::new_row_major(vec![4, 4, 4]);
        assert_eq!(s.offset(), 0);
        assert_eq!(s.sizes(), &[4, 4, 4]);
        assert_eq!(s.strides(), &[16, 4, 1]);
    }

    #[test]
    fn test_slice_view_smoke() {
        use crate::Slice;

        let base = Slice::new_row_major([2, 3, 4]);

        // Reshape: compatible shape and layout
        let view = base.view(&[6, 4]).unwrap();
        assert_eq!(view.sizes(), &[6, 4]);
        assert_eq!(view.offset(), 0);
        assert_eq!(view.strides(), &[4, 1]);
        assert_eq!(
            view.location(&[5, 3]).unwrap(),
            base.location(&[1, 2, 3]).unwrap()
        );

        // Reshape: identity (should succeed)
        let view = base.view(&[2, 3, 4]).unwrap();
        assert_eq!(view.sizes(), base.sizes());
        assert_eq!(view.strides(), base.strides());

        // Reshape: incompatible shape (wrong element count)
        let err = base.view(&[5, 4]);
        assert!(err.is_err());

        // Reshape: incompatible layout (simulate select)
        let selected = Slice::new(1, vec![2, 3], vec![6, 1]).unwrap(); // not offset=0
        let err = selected.view(&[3, 2]);
        assert!(err.is_err());

        // Reshape: flat 1D view
        let flat = base.view(&[24]).unwrap();
        assert_eq!(flat.sizes(), &[24]);
        assert_eq!(flat.strides(), &[1]);
        assert_eq!(
            flat.location(&[23]).unwrap(),
            base.location(&[1, 2, 3]).unwrap()
        );
    }

    #[test]
    fn test_view_of_view_when_dense() {
        // Start with a dense base: 2 × 3 × 4 = 24 elements.
        let base = Slice::new_row_major([2, 3, 4]);

        // First view: flatten to 1D.
        let flat = base.view(&[24]).unwrap();
        assert_eq!(flat.sizes(), &[24]);
        assert_eq!(flat.strides(), &[1]);
        assert_eq!(flat.offset(), 0); // Still dense.

        // Second view: reshape 1D to 6 × 4.
        let reshaped = flat.view(&[6, 4]).unwrap();
        assert_eq!(reshaped.sizes(), &[6, 4]);
        assert_eq!(reshaped.strides(), &[4, 1]);
        assert_eq!(reshaped.offset(), 0);

        // Location agreement check
        assert_eq!(
            reshaped.location(&[5, 3]).unwrap(),
            base.location(&[1, 2, 3]).unwrap()
        );
    }
}
