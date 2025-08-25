/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::ops::Index;
use std::sync::Arc;

use serde::Deserialize;
use serde::Serialize;
use thiserror::Error;

use crate::Range;
use crate::Slice;
use crate::SliceIterator;
use crate::slice::CartesianIterator;

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
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq, Hash, Debug)]
pub struct Extent {
    inner: Arc<ExtentData>,
}

fn _assert_extent_traits()
where
    Extent: Send + Sync + 'static,
{
}

// `ExtentData` is represented as:
// - `labels`: dimension names like `"zone"`, `"host"`, `"gpu"`
// - `sizes`: number of elements in each dimension, independent of
//   stride or storage layout
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq, Hash, Debug)]
struct ExtentData {
    labels: Vec<String>,
    sizes: Vec<usize>,
}

impl Extent {
    /// Creates a new `Extent` from the given labels and sizes.
    pub fn new(labels: Vec<String>, sizes: Vec<usize>) -> Result<Self, ExtentError> {
        if labels.len() != sizes.len() {
            return Err(ExtentError::DimMismatch {
                num_labels: labels.len(),
                num_sizes: sizes.len(),
            });
        }

        Ok(Self {
            inner: Arc::new(ExtentData { labels, sizes }),
        })
    }

    pub fn unity() -> Extent {
        Extent::new(vec![], vec![]).unwrap()
    }

    /// Returns the ordered list of dimension labels in this extent.
    pub fn labels(&self) -> &[String] {
        &self.inner.labels
    }

    /// Returns the dimension sizes, ordered to match the labels.
    pub fn sizes(&self) -> &[usize] {
        &self.inner.sizes
    }

    /// Returns the size of the dimension with the given label, if it
    /// exists.
    pub fn size(&self, label: &str) -> Option<usize> {
        self.position(label).map(|pos| self.sizes()[pos])
    }

    /// Returns the position of the dimension with the given label, if
    /// it exists exists.
    pub fn position(&self, label: &str) -> Option<usize> {
        self.labels().iter().position(|l| l == label)
    }

    /// Creates a `Point` in this extent with the given coordinates.
    ///
    /// Returns an error if the coordinate dimensionality does not
    /// match.
    pub fn point(&self, coords: Vec<usize>) -> Result<Point, PointError> {
        if coords.len() != self.len() {
            return Err(PointError::DimMismatch {
                expected: self.len(),
                actual: coords.len(),
            });
        }

        Ok(Point {
            coords,
            extent: Extent {
                inner: Arc::clone(&self.inner),
            },
        })
    }

    /// Returns the point corresponding to the provided rank in this extent.
    pub fn point_of_rank(&self, mut rank: usize) -> Result<Point, PointError> {
        if rank >= self.num_ranks() {
            return Err(PointError::OutOfRange {
                size: self.len(),
                rank,
            });
        }

        let mut stride: usize = self.sizes().iter().product();
        let mut coords = vec![0; self.len()];
        for (i, size) in self.sizes().iter().enumerate() {
            stride /= size;
            coords[i] = rank / stride;
            rank %= stride;
        }

        Ok(Point {
            coords,
            extent: self.clone(),
        })
    }

    /// The number of dimensions in the extent.
    pub fn len(&self) -> usize {
        self.sizes().len()
    }

    /// Whether the extent has zero dimensionbs.
    pub fn is_empty(&self) -> bool {
        self.sizes().is_empty()
    }

    /// The number of ranks in the extent.
    pub fn num_ranks(&self) -> usize {
        self.sizes().iter().product()
    }

    /// Convert this extent into its labels and sizes.
    pub fn into_inner(self) -> (Vec<String>, Vec<usize>) {
        match Arc::try_unwrap(self.inner) {
            Ok(data) => (data.labels, data.sizes),
            Err(shared) => (shared.labels.clone(), shared.sizes.clone()),
        }
    }

    /// Creates a slice representing the full extent.
    pub fn to_slice(&self) -> Slice {
        Slice::new_row_major(self.sizes())
    }

    /// Iterate over this extens labels and sizes.
    pub fn iter(&self) -> impl Iterator<Item = (String, usize)> + use<'_> {
        self.labels()
            .iter()
            .zip(self.sizes().iter())
            .map(|(l, s)| (l.clone(), *s))
    }

    /// Iterate points in this extent.
    pub fn points(&self) -> ExtentPointsIterator {
        ExtentPointsIterator {
            extent: self,
            pos: CartesianIterator::new(self.sizes().to_vec()),
        }
    }
}

impl std::fmt::Display for Extent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.sizes().len();
        write!(f, "{{")?;
        for i in 0..n {
            write!(f, "'{}': {}", self.labels()[i], self.sizes()[i])?;
            if i != n - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")?;
        Ok(())
    }
}

/// An iterator for points in an extent.
pub struct ExtentPointsIterator<'a> {
    extent: &'a Extent,
    pos: CartesianIterator,
}

impl<'a> Iterator for ExtentPointsIterator<'a> {
    type Item = Point;

    fn next(&mut self) -> Option<Self::Item> {
        Some(Point {
            coords: self.pos.next()?,
            extent: self.extent.clone(),
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

    /// The point is out of range for the extent.
    #[error("out of range: size of extent is {size}; does not contain rank {rank}")]
    OutOfRange { size: usize, rank: usize },
}

/// `Point` represents a specific coordinate within the
/// multi-dimensional space defined by an `Extent`.
///
/// Coordinate values can be accessed by indexing:
///
/// ```
/// use ndslice::extent;
///
/// let ext = extent!(zone = 2, host = 4, gpu = 8);
/// let point = ext.point(vec![1, 2, 3]).unwrap();
/// assert_eq!(point[0], 1);
/// assert_eq!(point[1], 2);
/// assert_eq!(point[2], 3);
/// ```
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq, Hash, Debug)]
pub struct Point {
    coords: Vec<usize>,
    extent: Extent,
}

impl Index<usize> for Point {
    type Output = usize;

    /// Returns the coordinate value for the given dimension index.
    /// This allows using `point[0]` syntax instead of
    /// `point.coords()[0]`.
    fn index(&self, dim: usize) -> &Self::Output {
        &self.coords[dim]
    }
}

impl<'a> IntoIterator for &'a Point {
    type Item = usize;
    type IntoIter = std::iter::Cloned<std::slice::Iter<'a, usize>>;

    /// Iterates over the coordinate values of this point.
    ///
    /// This allows using `for coord in &point { ... }` syntax to
    /// iterate through each dimension's coordinate value.
    fn into_iter(self) -> Self::IntoIter {
        self.coords.iter().cloned()
    }
}

fn _assert_point_traits()
where
    Point: Send + Sync + 'static,
{
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
/// use ndslice::Extent;
/// use ndslice::view::InExtent;
/// let extent = Extent::new(vec!["x".into(), "y".into()], vec![3, 4]).unwrap();
/// let point = vec![1, 2].in_(&extent).unwrap();
/// assert_eq!(point.rank(), 1 * 4 + 2);
/// ```
pub trait InExtent {
    fn in_(self, extent: &Extent) -> Result<Point, PointError>;
}

impl InExtent for Vec<usize> {
    /// Creates a `Point` with the provided coordinates in the given
    /// extent.
    ///
    /// Delegates to `Extent::point`.
    fn in_(self, extent: &Extent) -> Result<Point, PointError> {
        extent.point(self)
    }
}

impl Point {
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

    /// The dimensionality of this point.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Is this the 0d constant `[]`?
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.coords.len();
        for i in 0..n {
            write!(
                f,
                "{}={}/{}",
                self.extent.labels()[i],
                self.coords[i],
                self.extent.sizes()[i]
            )?;
            if i != n - 1 {
                write!(f, ",")?;
            }
        }
        Ok(())
    }
}

/// Errors that occur while operating on views.
#[derive(Debug, Error)]
pub enum ViewError {
    /// The provided dimension does not exist in the relevant extent.
    #[error("no such dimension: {0}")]
    InvalidDim(String),

    /// A view was attempted to be constructed from an empty (resolved) range.
    #[error("empty range: {range} for dimension {dim} of size {size}")]
    EmptyRange {
        range: Range,
        dim: String,
        size: usize,
    },
}

/// A view is a collection of ranks, organized into an extent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct View {
    labels: Vec<String>,
    slice: Slice,
}

impl View {
    /// The extent of this view. Every point in this space is defined.
    pub fn extent(&self) -> Extent {
        Extent::new(self.labels.clone(), self.slice.sizes().to_vec()).unwrap()
    }

    /// Iterate over the ranks in this view. The iterator returns both each rank,
    /// as well as the corresponding point in the extent of this view.
    pub fn iter(&self) -> ViewIterator {
        ViewIterator {
            extent: self.extent(),
            pos: self.slice.iter(),
        }
    }
}

impl std::fmt::Display for View {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.labels.len();
        for i in 0..n {
            write!(f, "{}={}", self.labels[i], self.slice.sizes()[i])?;
            if i != n - 1 {
                write!(f, ",")?;
            }
        }
        Ok(())
    }
}

/// The iterator over views.
pub struct ViewIterator {
    extent: Extent,     // Note that `extent` and...
    pos: SliceIterator, // ... `pos` share the same `Slice`.
}

impl Iterator for ViewIterator {
    type Item = (Point, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // This is a rank in the base space.
        let rank = self.pos.next()?;
        // Here, we convert to view space.
        let coords = self.pos.slice.coordinates(rank).unwrap();
        let point = coords.in_(&self.extent).unwrap();
        Some((point, rank))
    }
}

/// Viewable is a common trait implemented for data structures from which views
/// may be created. This allows us to provide a consistent API for constructing
/// and composing views.
pub trait Viewable {
    /// The labels of the dimensions in this view.
    fn labels(&self) -> Vec<String>;

    /// The slice representing this view.
    /// Note: this representation may change.
    fn slice(&self) -> Slice;
}

impl Viewable for View {
    fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }

    fn slice(&self) -> Slice {
        self.slice.clone()
    }
}

impl Viewable for Extent {
    fn labels(&self) -> Vec<String> {
        self.labels().to_vec()
    }

    fn slice(&self) -> Slice {
        self.to_slice()
    }
}

// We would make this impl<T: Viewable> From<T> for View,
// except this conflicts with the blanket impl for From<&T> for View.
impl From<Extent> for View {
    fn from(extent: Extent) -> Self {
        View {
            labels: extent.labels().to_vec(),
            slice: extent.slice(),
        }
    }
}

/// Extension methods for view construction.
pub trait ViewExt: Viewable {
    /// Construct a view comprising the range of points along the provided dimension.
    ///
    /// ## Examples
    ///
    /// ```
    /// use ndslice::Range;
    /// use ndslice::ViewExt;
    /// use ndslice::extent;
    ///
    /// let ext = extent!(zone = 4, host = 2, gpu = 8);
    ///
    /// // Subselect zone index 0.
    /// assert_eq!(ext.range("zone", 0).unwrap().iter().count(), 16);
    ///
    /// // Even GPUs within zone 0
    /// assert_eq!(
    ///     ext.range("zone", 0)
    ///         .unwrap()
    ///         .range("gpu", Range(0, None, 2))
    ///         .unwrap()
    ///         .iter()
    ///         .count(),
    ///     8
    /// );
    /// ```
    fn range<R: Into<Range>>(&self, dim: &str, range: R) -> Result<View, ViewError>;

    /// Group by view on `dim`. The returned iterator enumerates all groups
    /// as views in the extent of `dim` to the last dimension of the view.
    ///
    /// ## Examples
    ///
    /// ```
    /// use ndslice::ViewExt;
    /// use ndslice::extent;
    ///
    /// let ext = extent!(zone = 4, host = 2, gpu = 8);
    ///
    /// // We generate one view for each zone.
    /// assert_eq!(ext.group_by("host").unwrap().count(), 4);
    ///
    /// let mut parts = ext.group_by("host").unwrap();
    ///
    /// let zone0 = parts.next().unwrap();
    /// let mut zone0_points = zone0.iter();
    /// assert_eq!(zone0.extent(), extent!(host = 2, gpu = 8));
    /// assert_eq!(
    ///     zone0_points.next().unwrap(),
    ///     (extent!(host = 2, gpu = 8).point(vec![0, 0]).unwrap(), 0)
    /// );
    /// assert_eq!(
    ///     zone0_points.next().unwrap(),
    ///     (extent!(host = 2, gpu = 8).point(vec![0, 1]).unwrap(), 1)
    /// );
    ///
    /// let zone1 = parts.next().unwrap();
    /// assert_eq!(zone1.extent(), extent!(host = 2, gpu = 8));
    /// assert_eq!(
    ///     zone1.iter().next().unwrap(),
    ///     (extent!(host = 2, gpu = 8).point(vec![0, 0]).unwrap(), 16)
    /// );
    /// ```
    fn group_by(&self, dim: &str) -> Result<impl Iterator<Item = View>, ViewError>;
}

impl<T: Viewable> ViewExt for T {
    fn range<R: Into<Range>>(&self, dim: &str, range: R) -> Result<View, ViewError> {
        let range = range.into();
        let dim = self
            .labels()
            .iter()
            .position(|l| dim == l)
            .ok_or_else(|| ViewError::InvalidDim(dim.to_string()))?;
        let (mut offset, mut sizes, mut strides) = self.slice().into_inner();
        let (begin, end, step) = range.resolve(sizes[dim]);
        if end <= begin {
            return Err(ViewError::EmptyRange {
                range,
                dim: dim.to_string(),
                size: sizes[dim],
            });
        }

        offset += strides[dim] * begin;
        sizes[dim] = (end - begin).div_ceil(step);
        strides[dim] *= step;
        let slice = Slice::new(offset, sizes, strides).unwrap();

        Ok(View {
            labels: self.labels().clone(),
            slice,
        })
    }

    fn group_by(&self, dim: &str) -> Result<impl Iterator<Item = View>, ViewError> {
        let dim = self
            .labels()
            .iter()
            .position(|l| dim == l)
            .ok_or_else(|| ViewError::InvalidDim(dim.to_string()))?;

        let (offset, sizes, strides) = self.slice().into_inner();
        let mut ranks = Slice::new(offset, sizes[..dim].to_vec(), strides[..dim].to_vec())
            .unwrap()
            .iter();

        let labels = self.labels()[dim..].to_vec();
        let sizes = sizes[dim..].to_vec();
        let strides = strides[dim..].to_vec();

        Ok(std::iter::from_fn(move || {
            let rank = ranks.next()?;
            let slice = Slice::new(rank, sizes.clone(), strides.clone()).unwrap();
            Some(View {
                labels: labels.clone(),
                slice,
            })
        }))
    }
}

/// Construct a new extent with the given set of dimension-size pairs.
///
/// ```
/// let s = ndslice::extent!(host = 2, gpu = 8);
/// assert_eq!(s.labels(), &["host".to_string(), "gpu".to_string()]);
/// assert_eq!(s.sizes(), &[2, 8]);
/// ```
#[macro_export]
macro_rules! extent {
    ( $( $label:ident = $size:expr_2021 ),* $(,)? ) => {
        {
            let mut labels = Vec::new();
            let mut sizes = Vec::new();

            $(
                labels.push(stringify!($label).to_string());
                sizes.push($size);
            )*

            $crate::view::Extent::new(labels, sizes).unwrap()
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Shape;
    use crate::shape;

    #[test]
    fn test_points_basic() {
        let extent = extent!(x = 4, y = 5, z = 6);
        let _p1 = extent.point(vec![1, 2, 3]).unwrap();
        let _p2 = vec![1, 2, 3].in_(&extent).unwrap();

        assert_eq!(extent.num_ranks(), 4 * 5 * 6);

        let p3 = extent.point_of_rank(0).unwrap();
        assert_eq!(p3.coords(), &[0, 0, 0]);
        assert_eq!(p3.rank(), 0);

        let p4 = extent.point_of_rank(1).unwrap();
        assert_eq!(p4.coords(), &[0, 0, 1]);
        assert_eq!(p4.rank(), 1);

        let p5 = extent.point_of_rank(2).unwrap();
        assert_eq!(p5.coords(), &[0, 0, 2]);
        assert_eq!(p5.rank(), 2);

        let p6 = extent.point_of_rank(6 * 5 + 1).unwrap();
        assert_eq!(p6.coords(), &[1, 0, 1]);
        assert_eq!(p6.rank(), 6 * 5 + 1);
        assert_eq!(p6[0], 1);
        assert_eq!(p6[1], 0);
        assert_eq!(p6[2], 1);

        assert_eq!(extent.points().collect::<Vec<_>>().len(), 4 * 5 * 6);
        for (rank, point) in extent.points().enumerate() {
            let &[x, y, z] = &**point.coords() else {
                panic!("invalid coords");
            };
            assert_eq!(z + y * 6 + x * 6 * 5, rank);
            assert_eq!(point.rank(), rank);
        }
    }

    macro_rules! assert_view {
        ($view:expr, $extent:expr,  $( $($coord:expr),+ => $rank:expr );* $(;)?) => {
            let view = $view;
            assert_eq!(view.extent(), $extent);
            let expected: Vec<_> = vec![$(($extent.point(vec![$($coord),+]).unwrap(), $rank)),*];
            let actual: Vec<_> = view.iter().collect();
            assert_eq!(actual, expected);
        };
    }

    #[test]
    fn test_view_basic() {
        let extent = extent!(x = 4, y = 4);
        assert_view!(
            extent.range("x", 0..2).unwrap(),
            extent!(x = 2, y = 4),
            0, 0 => 0;
            0, 1 => 1;
            0, 2 => 2;
            0, 3 => 3;
            1, 0 => 4;
            1, 1 => 5;
            1, 2 => 6;
            1, 3 => 7;
        );
        assert_view!(
            extent.range("x", 1).unwrap().range("y", 2..).unwrap(),
            extent!(x = 1, y = 2),
            0, 0 => 6;
            0, 1 => 7;
        );
        assert_view!(
            extent.range("y", Range(0, None, 2)).unwrap(),
            extent!(x = 4, y = 2),
            0, 0 => 0;
            0, 1 => 2;
            1, 0 => 4;
            1, 1 => 6;
            2, 0 => 8;
            2, 1 => 10;
            3, 0 => 12;
            3, 1 => 14;
        );
        assert_view!(
            extent.range("y", Range(0, None, 2)).unwrap().range("x", 2..).unwrap(),
            extent!(x = 2, y = 2),
            0, 0 => 8;
            0, 1 => 10;
            1, 0 => 12;
            1, 1 => 14;
        );

        let extent = extent!(x = 10, y = 2);
        assert_view!(
            extent.range("x", Range(0, None, 2)).unwrap(),
            extent!(x = 5, y = 2),
            0, 0 => 0;
            0, 1 => 1;
            1, 0 => 4;
            1, 1 => 5;
            2, 0 => 8;
            2, 1 => 9;
            3, 0 => 12;
            3, 1 => 13;
            4, 0 => 16;
            4, 1 => 17;
        );
        assert_view!(
            extent.range("x", Range(0, None, 2)).unwrap().range("x", 2..).unwrap().range("y", 1).unwrap(),
            extent!(x = 3, y = 1),
            0, 0 => 9;
            1, 0 => 13;
            2, 0 => 17;
        );

        let extent = extent!(zone = 4, host = 2, gpu = 8);
        assert_view!(
            extent.range("zone", 0).unwrap().range("gpu", Range(0, None, 2)).unwrap(),
            extent!(zone = 1, host = 2, gpu = 4),
            0, 0, 0 => 0;
            0, 0, 1 => 2;
            0, 0, 2 => 4;
            0, 0, 3 => 6;
            0, 1, 0 => 8;
            0, 1, 1 => 10;
            0, 1, 2 => 12;
            0, 1, 3 => 14;
        );

        let extent = extent!(x = 3);
        assert_view!(
            extent.range("x", Range(0, None, 2)).unwrap(),
            extent!(x = 2),
            0 => 0;
            1 => 2;
        );
    }

    #[test]
    fn test_point_indexing() {
        let extent = Extent::new(vec!["x".into(), "y".into(), "z".into()], vec![4, 5, 6]).unwrap();
        let point = extent.point(vec![1, 2, 3]).unwrap();

        assert_eq!(point[0], 1);
        assert_eq!(point[1], 2);
        assert_eq!(point[2], 3);
    }

    #[test]
    #[should_panic]
    fn test_point_indexing_out_of_bounds() {
        let extent = Extent::new(vec!["x".into(), "y".into()], vec![4, 5]).unwrap();
        let point = extent.point(vec![1, 2]).unwrap();

        let _ = point[5]; // Should panic
    }

    #[test]
    fn test_point_into_iter() {
        let extent = Extent::new(vec!["x".into(), "y".into(), "z".into()], vec![4, 5, 6]).unwrap();
        let point = extent.point(vec![1, 2, 3]).unwrap();

        let coords: Vec<usize> = (&point).into_iter().collect();
        assert_eq!(coords, vec![1, 2, 3]);

        let mut sum = 0;
        for coord in &point {
            sum += coord;
        }
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_extent_basic() {
        let extent = extent!(x = 10, y = 5, z = 1);
        assert_eq!(
            extent.iter().collect::<Vec<_>>(),
            vec![
                ("x".to_string(), 10),
                ("y".to_string(), 5),
                ("z".to_string(), 1)
            ]
        );
    }

    #[test]
    fn test_extent_display() {
        let extent = Extent::new(vec!["x".into(), "y".into(), "z".into()], vec![4, 5, 6]).unwrap();
        assert_eq!(format!("{}", extent), "{'x': 4, 'y': 5, 'z': 6}");

        let empty_extent = Extent::new(vec![], vec![]).unwrap();
        assert_eq!(format!("{}", empty_extent), "{}");
    }

    #[test]
    fn test_extent_0d() {
        let e = Extent::new(vec![], vec![]).unwrap();
        assert_eq!(e.num_ranks(), 1);
        let points: Vec<_> = e.points().collect();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].coords(), &[]);
        assert_eq!(points[0].rank(), 0);
    }

    #[test]
    fn test_point_display() {
        let extent = Extent::new(vec!["x".into(), "y".into(), "z".into()], vec![4, 5, 6]).unwrap();
        let point = extent.point(vec![1, 2, 3]).unwrap();
        assert_eq!(format!("{}", point), "x=1/4,y=2/5,z=3/6");

        assert!(extent.point(vec![]).is_err());

        let empty_extent = Extent::new(vec![], vec![]).unwrap();
        let empty_point = empty_extent.point(vec![]).unwrap();
        assert_eq!(format!("{}", empty_point), "");
    }

    #[test]
    fn test_relative_point() {
        // Given a rank in the root shape, return the corresponding point in the
        // provided shape, which is a view of the root shape.
        pub fn relative_point(rank_on_root_mesh: usize, shape: &Shape) -> anyhow::Result<Point> {
            let coords = shape.slice().coordinates(rank_on_root_mesh)?;
            let extent = Extent::new(shape.labels().to_vec(), shape.slice().sizes().to_vec())?;
            Ok(extent.point(coords)?)
        }

        let root_shape = shape! { replicas = 4, hosts = 4, gpus = 4 };
        // rows are `hosts`, cols are gpus
        // replicas = 0
        //     0,    1,  2,    3,
        //     (4),  5,  (6),  7,
        //     8,    9,  10,   11,
        //     (12), 13, (14), 15,
        // replicas = 3, which is [replicas=0] + 48
        //     48,   49, 50,   51,
        //     (52), 53, (54), 55,
        //     56,   57, 58,   59,
        //     (60), 61, (62), 63,
        let sliced_shape = root_shape
            .select("replicas", crate::Range(0, Some(4), 3))
            .unwrap()
            .select("hosts", crate::Range(1, Some(4), 2))
            .unwrap()
            .select("gpus", crate::Range(0, Some(4), 2))
            .unwrap();
        let ranks_on_root_mesh = &[4, 6, 12, 14, 52, 54, 60, 62];
        assert_eq!(
            sliced_shape.slice().iter().collect::<Vec<_>>(),
            ranks_on_root_mesh,
        );

        let ranks_on_sliced_mesh = ranks_on_root_mesh
            .iter()
            .map(|&r| relative_point(r, &sliced_shape).unwrap().rank());
        assert_eq!(
            ranks_on_sliced_mesh.collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn test_iter_subviews() {
        let extent = extent!(zone = 4, host = 4, gpu = 8);

        assert_eq!(extent.group_by("gpu").unwrap().count(), 16);
        assert_eq!(extent.group_by("zone").unwrap().count(), 1);

        let mut parts = extent.group_by("gpu").unwrap();
        assert_view!(
            parts.next().unwrap(),
            extent!(gpu = 8),
            0 => 0;
            1 => 1;
            2 => 2;
            3 => 3;
            4 => 4;
            5 => 5;
            6 => 6;
            7 => 7;
        );
        assert_view!(
            parts.next().unwrap(),
            extent!(gpu = 8),
            0 => 8;
            1 => 9;
            2 => 10;
            3 => 11;
            4 => 12;
            5 => 13;
            6 => 14;
            7 => 15;
        );
    }
}
