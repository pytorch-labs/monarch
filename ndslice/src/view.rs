/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

/// Errors that can occur when constructing or validating an `Extent`.
#[derive(Debug, thiserror::Error)]
pub enum ExtentError {
    /// The number of labels does not match the number of sizes.
    ///
    /// This occurs when constructing an `Extent` from parallel
    /// `Vec<String>` and `Vec<usize>` inputs that are not the same
    /// length.
    #[error("label/sizes dimension mismatch: {num_labels} != {num_sizes}")]
    DimMismatch {
        /// Number of dimension labels provided.
        num_labels: usize,
        /// Number of dimension sizes provided.
        num_sizes: usize,
    },
}

/// `Extent` defines the logical shape of a multidimensional space by
/// assigning a size to each named dimension. It abstracts away memory
/// layout and focuses solely on structure — what dimensions exist and
/// how many elements each contains.
///
/// Conceptually, it corresponds to a coordinate space in the
/// mathematical sense.
///
/// Internally, `Extent` is represented as:
/// - `labels`: dimension names like `"zone"`, `"host"`, `"gpu"`
/// - `sizes`: number of elements in each dimension, independent of
///            stride or storage layout
#[derive(Clone, Deserialize, Serialize, PartialEq, Hash, Debug)]
pub struct Extent {
    labels: Vec<String>,
    sizes: Vec<usize>,
}

impl Extent {
    /// Creates a new `Extent` from the given labels and sizes.
    ///
    /// Returns an error if the number of labels and sizes do not
    /// match.
    pub fn new(labels: Vec<String>, sizes: Vec<usize>) -> Result<Self, ExtentError> {
        if labels.len() != sizes.len() {
            return Err(ExtentError::DimMismatch {
                num_labels: labels.len(),
                num_sizes: sizes.len(),
            });
        }

        Ok(Self { labels, sizes })
    }

    /// Returns the ordered list of dimension labels in this extent.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns the dimension sizes, ordered to match the labels.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Returns the size of the dimension with the given label, if it
    /// exists.
    pub fn get(&self, label: &str) -> Option<usize> {
        self.labels
            .iter()
            .position(|l| l == label)
            .map(|i| self.sizes[i])
    }

    /// Returns the number of dimensions in this extent.
    pub fn num_dim(&self) -> usize {
        self.labels.len()
    }

    /// Creates a `Point` in this extent with the given coordinates.
    ///
    /// Returns an error if the coordinate dimensionality does not
    /// match.
    pub fn point(&self, coords: Vec<usize>) -> Result<Point, PointError> {
        if coords.len() != self.num_dim() {
            return Err(PointError::DimMismatch {
                expected: self.num_dim(),
                actual: coords.len(),
            });
        }

        Ok(Point {
            coords,
            extent: self.clone(),
        })
    }
}

/// Errors that can occur when constructing or evaluating a `Point`.
#[derive(Debug, Error)]
pub enum PointError {
    /// The number of coordinates does not match the number of
    /// dimensions defined by the associated extent.
    ///
    /// This occurs when creating a `Point` with a coordinate vector
    /// of incorrect length relative to the dimensionality of the
    /// extent.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch {
        /// Number of dimensions expected from the extent.
        expected: usize,
        /// Number of coordinates actually provided.
        actual: usize,
    },
}

/// `Point` represents a specific coordinate within the
/// multi-dimensional space defined by an `Extent`.
#[derive(Clone, Deserialize, Serialize, PartialEq, Hash, Debug)]
pub struct Point {
    coords: Vec<usize>,
    extent: Extent,
}

/// Extension trait for creating a `Point` from a coordinate vector
/// and an `Extent`.
///
/// This trait provides the `.in_(&extent)` method, which constructs a
/// `Point` using the caller as the coordinate vector and the given
/// extent as the shape context.
///
/// # Example
/// ```
/// use ndslice::view::Extent;
/// use ndslice::view::InExtent;
///
/// let extent = Extent::new(vec!["x".into(), "y".into()], vec![3, 4]).unwrap();
/// let point = vec![1, 2].in_(&extent).unwrap();
/// assert_eq!(point.rank(), 1 * 4 + 2);
/// ```
pub trait InExtent {
    fn in_(self, extent: &Extent) -> Result<Point, PointError>;
}

impl InExtent for Vec<usize> {
    /// Creates a `Point` with this coordinate vector in the given
    /// extent.
    ///
    /// Delegates to `Extent::point`.
    fn in_(self, extent: &Extent) -> Result<Point, PointError> {
        extent.point(self)
    }
}

impl Point {
    /// Creates a new `Point`. Most users should prefer
    /// `Extent::point`.
    #[allow(dead_code)]
    fn new(coords: Vec<usize>, extent: Extent) -> Result<Self, PointError> {
        if coords.len() != extent.num_dim() {
            return Err(PointError::DimMismatch {
                expected: extent.num_dim(),
                actual: coords.len(),
            });
        }

        Ok(Point { coords, extent })
    }

    /// Returns a reference to the coordinate vector for this point.
    pub fn coords(&self) -> &Vec<usize> {
        &self.coords
    }

    /// Returns a reference to the extent associated with this point.
    pub fn extent(&self) -> &Extent {
        &self.extent
    }

    /// Computes the row-major logical rank of this point within its
    /// extent.
    ///
    /// ```text
    /// Σ (coord[i] × ∏(sizes[j] for j > i))
    /// ```
    ///
    /// where `coord` is the point's coordinate and `sizes` is the
    /// extent's dimension sizes.
    pub fn rank(&self) -> usize {
        let mut stride = 1;
        let mut result = 0;
        for (c, size) in self
            .coords
            .iter()
            .rev()
            .zip(self.extent().sizes().iter().rev())
        {
            result += *c * stride;
            stride *= size;
        }

        result
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_point_creation() {
        let extent = Extent::new(vec!["x".into(), "y".into(), "z".into()], vec![4, 5, 6]).unwrap();
        let _p1 = extent.point(vec![1, 2, 3]).unwrap();
        let _p2 = vec![1, 2, 3].in_(&extent).unwrap();
    }
}
