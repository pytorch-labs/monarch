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
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Hash)]
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

    /// Sub-set this shape by selecting a [`Range`] from a named dimension.
    /// The provided range must be nonempty.
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

    /// Sub-set this shape by iterating over selection of dims sub-dimensions as Shape
    /// iterator.
    /// dims must be in range [0, num_dim - 1].
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

    fn dim(&self, label: &str) -> Result<usize, ShapeError> {
        self.labels
            .iter()
            .position(|l| l == label)
            .ok_or_else(|| ShapeError::InvalidLabels {
                labels: vec![label.to_string()],
            })
    }
}

/// Iterator over sub-shapes of a given shape.
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
    ( $( $label:ident = $size:expr ),* $(,)? ) => {
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

/// Perform a sub-selection on the provided shaped object (a `[Shape]`, or a [`crate::Mesh`].
///
/// ```
/// let s = ndslice::shape!(host = 2, gpu = 8);
/// let s = ndslice::select!(s, host = 1, gpu = 4..).unwrap();
/// assert_eq!(s.labels(), &["host".to_string(), "gpu".to_string()]);
/// assert_eq!(s.slice().sizes(), &[1, 4]);
/// ```
#[macro_export]
macro_rules! select {
    ($shape:ident, $label:ident = $range:expr) => {
        $shape.select(stringify!($label), $range)
    };

    ($shape:ident, $label:ident = $range:expr, $($labels:ident = $ranges:expr),+) => {
        $shape.select(stringify!($label), $range).and_then(|shape| $crate::select!(shape, $($labels = $ranges),+))
    };
}

/// A range of indices, with a stride. Ranges are convertible from
/// native Rust ranges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
