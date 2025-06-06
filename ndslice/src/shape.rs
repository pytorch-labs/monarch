/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::fmt;

use itertools::izip;
use serde::Deserialize;
use serde::Serialize;

use crate::DimSliceIterator;
use crate::Slice;
use crate::SliceError;
use crate::selection::Selection;

// We always retain dimensions here even if they are selected out.

#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    #[error("label slice dimension mismatch: {labels_dim} != {slice_dim}")]
    DimSliceMismatch { labels_dim: usize, slice_dim: usize },

    #[error("invalid labels `{labels:?}`")]
    InvalidLabels { labels: Vec<String> },

    #[error("empty range {range}")]
    EmptyRange { range: Range },

    #[error("out of range {range} for dimension {dim} of size {size}")]
    OutOfRange {
        range: Range,
        dim: String,
        size: usize,
    },

    #[error("selection `{expr}` exceeds dimensionality {num_dim}")]
    SelectionTooDeep { expr: Selection, num_dim: usize },

    #[error("dynamic selection `{expr}`")]
    SelectionDynamic { expr: Selection },

    #[error("{index} out of range for dimension {dim} of size {size}")]
    IndexOutOfRange {
        index: usize,
        dim: String,
        size: usize,
    },

    #[error(transparent)]
    SliceError(#[from] SliceError),
}

/// A shape is a [`Slice`] with labeled dimensions and a selection API.
#[derive(Clone, Deserialize, Serialize, PartialEq, Hash, Debug)]
pub struct Shape {
    /// The labels for each dimension in slice.
    labels: Vec<String>,
    /// The slice itself, which describes the topology of the shape.
    slice: Slice,
}

impl Shape {
    /// Creates a new shape with the provided labels, which describe the
    /// provided Slice.
    ///
    /// Shapes can also be constructed by way of the [`shape`] macro, which
    /// creates a by-construction correct slice in row-major order given a set of
    /// sized dimensions.
    pub fn new(labels: Vec<String>, slice: Slice) -> Result<Self, ShapeError> {
        if labels.len() != slice.num_dim() {
            return Err(ShapeError::DimSliceMismatch {
                labels_dim: labels.len(),
                slice_dim: slice.num_dim(),
            });
        }
        Ok(Self { labels, slice })
    }

    /// Restrict this shape along a named dimension using a [`Range`]. The
    /// provided range must be nonempty.
    //
    /// A shape defines a "strided view" where a strided view is a
    /// triple (`offset, `sizes`, `strides`). Each coordinate maps to
    /// a flat memory index using the formula:
    /// ``` text
    ///     index = offset + ∑ i_k * strides[k]
    /// ```
    /// where `i_k` is the coordinate in dimension `k`.
    ///
    /// The `select(dim, range)` operation restricts the view to a
    /// subrange along a single dimension. It refines the shape by
    /// updating the `offset`, `sizes[dim]`, and `strides[dim]` to
    /// describe a logically reindexed subregion:
    ///
    /// ```text
    ///     offset       += begin x strides[dim]
    ///     sizes[dim]    = (end - begin) / step
    ///     strides[dim] *= step
    /// ```
    ///
    /// This transformation preserves the strided layout and avoids
    /// copying data. After `select`, the view behaves as if indexing
    /// starts at zero in the selected dimension, with a new length
    /// and stride. From the user's perspective, nothing changes —
    /// indexing remains zero-based, and the resulting shape can be
    /// used like any other. The transformation is internal: the
    /// view's offset and stride absorb the selection logic.
    ///
    /// `select` is composable — it can be applied repeatedly, even on
    /// the same dimension, to refine the view incrementally.
    pub fn select<R: Into<Range>>(&self, label: &str, range: R) -> Result<Self, ShapeError> {
        let dim = self.dim(label)?;
        let range: Range = range.into();
        if range.is_empty() {
            return Err(ShapeError::EmptyRange { range });
        }

        let mut offset = self.slice.offset();
        let mut sizes = self.slice.sizes().to_vec();
        let mut strides = self.slice.strides().to_vec();

        let (begin, end, stride) = range.resolve(sizes[dim]);
        if begin >= sizes[dim] {
            return Err(ShapeError::OutOfRange {
                range,
                dim: label.to_string(),
                size: sizes[dim],
            });
        }

        offset += begin * strides[dim];
        sizes[dim] = (end - begin) / stride;
        strides[dim] *= stride;

        Ok(Self {
            labels: self.labels.clone(),
            slice: Slice::new(offset, sizes, strides).expect("cannot create invalid slice"),
        })
    }

    /// Produces an iterator over subshapes by fixing the first `dims`
    /// dimensions.
    ///
    /// For a shape of rank `n`, this yields `∏ sizes[0..dims]`
    /// subshapes, each with the first `dims` dimensions restricted to
    /// size 1. The remaining dimensions are left unconstrained.
    ///
    /// This is useful for structured traversal of slices within a
    /// multidimensional shape. See [`SelectIterator`] for details and
    /// examples.
    ///
    /// # Errors
    /// Returns an error if `dims == 0` or `dims >= self.rank()`.
    pub fn select_iter(&self, dims: usize) -> Result<SelectIterator, ShapeError> {
        let num_dims = self.slice().num_dim();
        if dims == 0 || dims >= num_dims {
            return Err(ShapeError::SliceError(SliceError::IndexOutOfRange {
                index: dims,
                total: num_dims,
            }));
        }

        Ok(SelectIterator {
            shape: self,
            iter: self.slice().dim_iter(dims),
        })
    }

    /// Sub-set this shape by select a particular row of the given indices
    /// The resulting shape will no longer have dimensions for the given indices
    /// Example shape.index(vec![("gpu", 3), ("host", 0)])
    pub fn index(&self, indices: Vec<(String, usize)>) -> Result<Shape, ShapeError> {
        let mut offset = self.slice.offset();
        let mut names = Vec::new();
        let mut sizes = Vec::new();
        let mut strides = Vec::new();
        let mut used_indices_count = 0;
        let slice = self.slice();
        for (dim, size, stride) in izip!(self.labels.iter(), slice.sizes(), slice.strides()) {
            if let Some(index) = indices
                .iter()
                .find_map(|(name, index)| if *name == *dim { Some(index) } else { None })
            {
                if *index >= *size {
                    return Err(ShapeError::IndexOutOfRange {
                        index: *index,
                        dim: dim.clone(),
                        size: *size,
                    });
                }
                offset += index * stride;
                used_indices_count += 1;
            } else {
                names.push(dim.clone());
                sizes.push(*size);
                strides.push(*stride);
            }
        }
        if used_indices_count != indices.len() {
            let unused_indices = indices
                .iter()
                .filter(|(key, _)| !self.labels.contains(key))
                .map(|(key, _)| key.clone())
                .collect();
            return Err(ShapeError::InvalidLabels {
                labels: unused_indices,
            });
        }
        let slice = Slice::new(offset, sizes, strides)?;
        Shape::new(names, slice)
    }

    /// The per-dimension labels of this shape.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// The slice describing the shape.
    pub fn slice(&self) -> &Slice {
        &self.slice
    }

    /// Return a set of labeled coordinates for the given rank.
    pub fn coordinates(&self, rank: usize) -> Result<Vec<(String, usize)>, ShapeError> {
        let coords = self.slice.coordinates(rank)?;
        Ok(coords
            .iter()
            .zip(self.labels.iter())
            .map(|(i, l)| (l.to_string(), *i))
            .collect())
    }

    pub fn dim(&self, label: &str) -> Result<usize, ShapeError> {
        self.labels
            .iter()
            .position(|l| l == label)
            .ok_or_else(|| ShapeError::InvalidLabels {
                labels: vec![label.to_string()],
            })
    }

    /// Return the 0-dimensional single element shape
    pub fn unity() -> Shape {
        Shape::new(vec![], Slice::new(0, vec![], vec![]).expect("unity")).expect("unity")
    }
}

/// Iterator over subshapes obtained by fixing a prefix of dimensions.
///
/// This iterator is produced by [`Shape::select_iter(dims)`], and
/// yields one `Shape` per coordinate prefix in the first `dims`
/// dimensions.
///
/// For a shape of `n` dimensions, each yielded shape has:
/// - The first `dims` dimensions restricted to size 1 (i.e., fixed
///   via `select`)
/// - The remaining `n - dims` dimensions left unconstrained
///
/// This allows structured iteration over "slices" of the original
/// shape: for example with `n` = 3, `select_iter(1)` walks through 2D
/// planes, while `select_iter(2)` yields 1D subshapes.
///
/// # Example
/// ```ignore
/// let s = shape!(zone = 2, host = 2, gpu = 8);
/// let views: Vec<_> = s.select_iter(2).unwrap().collect();
/// assert_eq!(views.len(), 4);
/// assert_eq!(views[0].slice().sizes(), &[1, 1, 8]);
/// ```
/// The above example can be interpreted as: for each `(zone, host)`
/// pair, `select_iter(2)` yields a `Shape` describing the associated
/// row of GPUs — a view into the `[1, 1, 8]` subregion of the full
/// `[2, 2, 8]` shape.
pub struct SelectIterator<'a> {
    shape: &'a Shape,
    iter: DimSliceIterator<'a>,
}

impl<'a> Iterator for SelectIterator<'a> {
    type Item = Shape;

    fn next(&mut self) -> Option<Self::Item> {
        let pos = self.iter.next()?;
        let mut shape = self.shape.clone();
        for (dim, index) in pos.iter().enumerate() {
            shape = shape.select(&self.shape.labels()[dim], *index).unwrap();
        }
        Some(shape)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Just display the sizes of each dimension, for now.
        // Once we have a selection algebra, we can provide a
        // better Display implementation.
        write!(f, "{{")?;
        for dim in 0..self.labels.len() {
            write!(f, "{}={}", self.labels[dim], self.slice.sizes()[dim])?;
            if dim < self.labels.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "}}")
    }
}

/// Construct a new shape with the given set of dimension-size pairs in row-major
/// order.
///
/// ```
/// let s = ndslice::shape!(host = 2, gpu = 8);
/// assert_eq!(s.labels(), &["host".to_string(), "gpu".to_string()]);
/// assert_eq!(s.slice().sizes(), &[2, 8]);
/// assert_eq!(s.slice().strides(), &[8, 1]);
/// ```
#[macro_export]
macro_rules! shape {
    ( $( $label:ident = $size:expr_2021 ),* $(,)? ) => {
        {
            let mut labels = Vec::new();
            let mut sizes = Vec::new();

            $(
                labels.push(stringify!($label).to_string());
                sizes.push($size);
            )*

            $crate::shape::Shape::new(labels, $crate::Slice::new_row_major(sizes)).unwrap()
        }
    };
}

/// Perform a sub-selection on the provided [`Shape`] object.
///
/// This macro chains `.select()` calls to apply multiple labeled
/// dimension restrictions in a fluent way.
///
/// ```
/// let s = ndslice::shape!(host = 2, gpu = 8);
/// let s = ndslice::select!(s, host = 1, gpu = 4..).unwrap();
/// assert_eq!(s.labels(), &["host".to_string(), "gpu".to_string()]);
/// assert_eq!(s.slice().sizes(), &[1, 4]);
/// ```
#[macro_export]
macro_rules! select {
    ($shape:ident, $label:ident = $range:expr_2021) => {
        $shape.select(stringify!($label), $range)
    };

    ($shape:ident, $label:ident = $range:expr_2021, $($labels:ident = $ranges:expr_2021),+) => {
        $shape.select(stringify!($label), $range).and_then(|shape| $crate::select!(shape, $($labels = $ranges),+))
    };
}

/// A range of indices, with a stride. Ranges are convertible from
/// native Rust ranges.
///
/// Deriving `Eq`, `Ord` and `Hash` is sound because all fields are
/// `Ord` and comparison is purely structural over `(start, end,
/// step)`.
#[derive(
    Debug,
    Clone,
    Eq,
    Hash,
    PartialEq,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord
)]
pub struct Range(pub usize, pub Option<usize>, pub usize);

impl Range {
    pub(crate) fn resolve(&self, size: usize) -> (usize, usize, usize) {
        match self {
            Range(begin, Some(end), stride) => (*begin, std::cmp::min(size, *end), *stride),
            Range(begin, None, stride) => (*begin, size, *stride),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Range(begin, Some(end), _) if end <= begin)
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Range(begin, None, stride) => write!(f, "{}::{}", begin, stride),
            Range(begin, Some(end), stride) => write!(f, "{}:{}:{}", begin, end, stride),
        }
    }
}

impl From<std::ops::Range<usize>> for Range {
    fn from(r: std::ops::Range<usize>) -> Self {
        Self(r.start, Some(r.end), 1)
    }
}

impl From<std::ops::RangeInclusive<usize>> for Range {
    fn from(r: std::ops::RangeInclusive<usize>) -> Self {
        Self(*r.start(), Some(*r.end() + 1), 1)
    }
}

impl From<std::ops::RangeFrom<usize>> for Range {
    fn from(r: std::ops::RangeFrom<usize>) -> Self {
        Self(r.start, None, 1)
    }
}

impl From<usize> for Range {
    fn from(idx: usize) -> Self {
        Self(idx, Some(idx + 1), 1)
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use super::*;

    #[test]
    fn test_basic() {
        let s = shape!(host = 2, gpu = 8);
        assert_eq!(&s.labels, &["host".to_string(), "gpu".to_string()]);
        assert_eq!(s.slice.offset(), 0);
        assert_eq!(s.slice.sizes(), &[2, 8]);
        assert_eq!(s.slice.strides(), &[8, 1]);

        assert_eq!(s.to_string(), "{host=2,gpu=8}");
    }

    #[test]
    fn test_select() {
        let s = shape!(host = 2, gpu = 8);

        assert_eq!(
            s.slice().iter().collect::<Vec<_>>(),
            &[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8 + 1,
                8 + 2,
                8 + 3,
                8 + 4,
                8 + 5,
                8 + 6,
                8 + 7
            ]
        );

        assert_eq!(
            select!(s, host = 1)
                .unwrap()
                .slice()
                .iter()
                .collect::<Vec<_>>(),
            &[8, 8 + 1, 8 + 2, 8 + 3, 8 + 4, 8 + 5, 8 + 6, 8 + 7]
        );

        assert_eq!(
            select!(s, gpu = 2..)
                .unwrap()
                .slice()
                .iter()
                .collect::<Vec<_>>(),
            &[2, 3, 4, 5, 6, 7, 8 + 2, 8 + 3, 8 + 4, 8 + 5, 8 + 6, 8 + 7]
        );

        assert_eq!(
            select!(s, gpu = 3..5)
                .unwrap()
                .slice()
                .iter()
                .collect::<Vec<_>>(),
            &[3, 4, 8 + 3, 8 + 4]
        );

        assert_eq!(
            select!(s, gpu = 3..5, host = 1)
                .unwrap()
                .slice()
                .iter()
                .collect::<Vec<_>>(),
            &[8 + 3, 8 + 4]
        );
    }

    #[test]
    fn test_select_iter() {
        let s = shape!(replica = 2, host = 2, gpu = 8);
        let selections: Vec<_> = s.select_iter(2).unwrap().collect();
        assert_eq!(selections[0].slice().sizes(), &[1, 1, 8]);
        assert_eq!(selections[1].slice().sizes(), &[1, 1, 8]);
        assert_eq!(selections[2].slice().sizes(), &[1, 1, 8]);
        assert_eq!(selections[3].slice().sizes(), &[1, 1, 8]);
        assert_eq!(
            selections,
            &[
                select!(s, replica = 0, host = 0).unwrap(),
                select!(s, replica = 0, host = 1).unwrap(),
                select!(s, replica = 1, host = 0).unwrap(),
                select!(s, replica = 1, host = 1).unwrap()
            ]
        );
    }

    #[test]
    fn test_coordinates() {
        let s = shape!(host = 2, gpu = 8);
        assert_eq!(
            s.coordinates(0).unwrap(),
            vec![("host".to_string(), 0), ("gpu".to_string(), 0)]
        );
        assert_eq!(
            s.coordinates(1).unwrap(),
            vec![("host".to_string(), 0), ("gpu".to_string(), 1)]
        );
        assert_eq!(
            s.coordinates(8).unwrap(),
            vec![("host".to_string(), 1), ("gpu".to_string(), 0)]
        );
        assert_eq!(
            s.coordinates(9).unwrap(),
            vec![("host".to_string(), 1), ("gpu".to_string(), 1)]
        );

        assert_matches!(
            s.coordinates(16).unwrap_err(),
            ShapeError::SliceError(SliceError::ValueNotInSlice { value: 16 })
        );
    }

    #[test]
    fn test_select_bad() {
        let s = shape!(host = 2, gpu = 8);

        assert_matches!(
            select!(s, gpu = 1..1).unwrap_err(),
            ShapeError::EmptyRange {
                range: Range(1, Some(1), 1)
            },
        );

        assert_matches!(
            select!(s, gpu = 8).unwrap_err(),
            ShapeError::OutOfRange {
                range: Range(8, Some(9), 1),
                dim,
                size: 8,
            } if dim == "gpu",
        );
    }

    #[test]
    fn test_shape_index() {
        let n_hosts = 5;
        let n_gpus = 7;

        // Index first dim
        let s = shape!(host = n_hosts, gpu = n_gpus);
        assert_eq!(
            s.index(vec![("host".to_string(), 0)]).unwrap(),
            Shape::new(
                vec!["gpu".to_string()],
                Slice::new(0, vec![n_gpus], vec![1]).unwrap()
            )
            .unwrap()
        );

        // Index last dims
        let offset = 1;
        assert_eq!(
            s.index(vec![("gpu".to_string(), offset)]).unwrap(),
            Shape::new(
                vec!["host".to_string()],
                Slice::new(offset, vec![n_hosts], vec![n_gpus]).unwrap()
            )
            .unwrap()
        );

        // Index middle dim
        let n_zone = 2;
        let s = shape!(zone = n_zone, host = n_hosts, gpu = n_gpus);
        let offset = 3;
        assert_eq!(
            s.index(vec![("host".to_string(), offset)]).unwrap(),
            Shape::new(
                vec!["zone".to_string(), "gpu".to_string()],
                Slice::new(
                    offset * n_gpus,
                    vec![n_zone, n_gpus],
                    vec![n_hosts * n_gpus, 1]
                )
                .unwrap()
            )
            .unwrap()
        );

        // Out of range
        assert!(
            shape!(gpu = n_gpus)
                .index(vec![("gpu".to_string(), n_gpus)])
                .is_err()
        );
        // Invalid dim
        assert!(
            shape!(gpu = n_gpus)
                .index(vec![("non-exist-dim".to_string(), 0)])
                .is_err()
        );
    }
}
