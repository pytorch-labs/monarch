/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Dimensional reshaping of slices and shapes.
//!
//! This module defines utilities for transforming a [`Slice`] or
//! [`Shape`] by factoring large extents into smaller ones under a
//! given limit. The result is a reshaped view with increased
//! dimensionality and preserved memory layout.
//!
//! This is useful for hierarchical routing, structured fanout, and
//! other multidimensional layout transformations.
//!
//! For [`Shape`]s, reshaping also expands dimension labels using a
//! `label/N` naming convention, preserving the semantics of the
//! original shape in the reshaped reshape_with_limit.
//!
//! See [`reshape_with_limit`] and [`reshape_shape`] for entry points.
use std::fmt;

use crate::shape::Shape;
use crate::slice::Slice;

/// Coordinate vector used throughout reshape logic. Semantically
/// represents a point in multidimensional space.
pub type Coord = Vec<usize>;

/// A reshaped version of a `Shape`, with factored dimensions and
/// updated labels.
///
/// This type preserves coordinate bijections with the original shape
/// and provides access to the transformed layout and label mappings.
pub struct ReshapedShape {
    /// The reshaped shape, with new labels and underlying factored
    /// slice.
    pub shape: Shape,

    /// For each original dimension label, the list of sizes it was
    /// split into.
    pub factors: Vec<(String, Vec<usize>)>,
}

#[allow(dead_code)]
const _: () = {
    fn assert<T: Send + Sync + 'static>() {}
    let _ = assert::<ReshapedShape>;
};

impl std::fmt::Debug for ReshapedShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReshapedShape")
            .field("labels", &self.shape.labels())
            .field("sizes", &self.shape.slice().sizes())
            .field("strides", &self.shape.slice().strides())
            .field("offset", &self.shape.slice().offset())
            .field("factors", &self.factors)
            .finish()
    }
}

impl std::fmt::Display for ReshapedShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReshapedShape {{ [off={} sz={:?} st={:?} lab={:?} fac={:?}] }}",
            self.shape.slice().offset(),
            self.shape.slice().sizes(),
            self.shape.slice().strides(),
            self.shape.labels(),
            self.factors
        )
    }
}

/// Returns, for each size, a list of factors that respect the given
/// limit. If a size is ≤ limit, it is returned as a singleton.
/// Otherwise, it is factored greedily using divisors ≤ limit, from
/// largest to smallest.
///
/// For best results, dimensions should be chosen to allow factoring
/// into small values under the selected limit (e.g., ≤ 32).
/// Large prime numbers cannot be broken down and will remain as-is,
/// limiting reshaping potential.
///
/// Prefer powers of 2 or other highly composite numbers
/// (e.g., 8, 16, 32, 60, 120) over large primes (e.g., 17, 37, 113)
/// when designing shapes intended for reshaping.
pub(crate) fn factor_dims(sizes: &[usize], limit: Limit) -> Vec<Vec<usize>> {
    let limit = limit.get();
    sizes
        .iter()
        .map(|&size| {
            if size <= limit {
                return vec![size];
            }
            let mut rem = size;
            let mut factors = Vec::new();
            for d in (2..=limit).rev() {
                while rem % d == 0 {
                    factors.push(d);
                    rem /= d;
                }
            }
            if rem > 1 {
                factors.push(rem);
            }
            factors
        })
        .collect()
}

/// Constructs a function that maps coordinates from the original
/// slice to equivalent coordinates in the reshaped slice, preserving
/// their flat (linear) position.
pub fn to_reshaped_coord<'a>(
    original: &'a Slice,
    reshaped: &'a Slice,
) -> impl Fn(&[usize]) -> Vec<usize> + 'a {
    let original = original.clone();
    let reshaped = reshaped.clone();
    move |coord: &[usize]| -> Coord {
        let flat = original.location(coord).unwrap();
        reshaped.coordinates(flat).unwrap()
    }
}

/// Constructs a function that maps coordinates from the reshaped
/// slice back to equivalent coordinates in the original slice,
/// preserving their flat (linear) position.
pub fn to_original_coord<'a>(
    reshaped: &'a Slice,
    original: &'a Slice,
) -> impl Fn(&[usize]) -> Vec<usize> + 'a {
    let reshaped = reshaped.clone();
    let original = original.clone();
    move |coord: &[usize]| -> Coord {
        let flat = reshaped.location(coord).unwrap();
        original.coordinates(flat).unwrap()
    }
}

/// A shaping constraint that bounds the maximum extent allowed in any
/// reshaped dimension.
///
/// This limit controls how a given dimension is factored during
/// reshaping. Values larger than `limit` are recursively decomposed
/// into smaller factors (e.g., `reshape_with_limit([1024],
/// Limit::new(32))` → `[32, 32]`).
///
/// The default limit is `32`, which balances fanout depth and layout
/// regularity.
///
/// # Example
/// ```
/// use ndslice::reshape::Limit;
/// let limit = Limit::new(64);
/// assert_eq!(limit.get(), 64);
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Limit(usize);

impl Limit {
    /// Creates a new `Limit`. Panics if less than 1.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "Limit must be at least 1");
        Self(n)
    }

    /// Returns the inner value.
    pub fn get(self) -> usize {
        self.0
    }
}

impl Default for Limit {
    fn default() -> Self {
        Self(32)
    }
}

impl From<usize> for Limit {
    fn from(n: usize) -> Self {
        Self::new(n)
    }
}

/// A trait for types that can be reshaped into a higher-dimensional
/// view by factoring large extents into smaller ones.
///
/// This is implemented for [`Slice`], enabling ergonomic access to
/// [`reshape_with_limit`] as a method.
///
/// # Example
/// ```
/// use ndslice::Slice;
/// use ndslice::reshape::Limit;
/// use ndslice::reshape::ReshapeSliceExt;
///
/// let slice = Slice::new_row_major(vec![1024]);
/// let reshaped = slice.reshape_with_limit(Limit::new(32));
/// assert_eq!(reshaped.sizes(), &[32, 32]);
/// ```
/// # Returns
/// A reshaped [`Slice`] with increased dimensionality and preserved
/// layout.
pub trait ReshapeSliceExt {
    /// Returns a reshaped version of this structure by factoring each
    /// dimension into smaller extents no greater than `limit`,
    /// preserving memory layout and flat index semantics. See
    /// [`reshape_with_limit`] for full behavior and rationale.
    ///
    /// # Arguments
    /// - `limit`: maximum size allowed in any reshaped dimension
    ///
    /// # Returns
    /// A reshaped [`Slice`] with increased dimensionality and a
    /// bijective mapping to the original.
    fn reshape_with_limit(&self, limit: Limit) -> Slice;
}

impl ReshapeSliceExt for Slice {
    fn reshape_with_limit(&self, limit: Limit) -> Slice {
        reshape_with_limit(self, limit)
    }
}

/// Extension trait for reshaping `Shape`s by factoring large dimensions.
pub trait ReshapeShapeExt {
    /// Produces a reshaped version of the shape with expanded
    /// dimensions under the given size limit.
    fn reshape(&self, limit: Limit) -> ReshapedShape;
}

impl ReshapeShapeExt for Shape {
    fn reshape(&self, limit: Limit) -> ReshapedShape {
        reshape_shape(self, limit)
    }
}

/// For convenient `slice.reshape_with_limit()`, `shape.reshape()`
/// syntax, `use reshape::prelude::*`.
pub mod prelude {
    pub use super::ReshapeShapeExt;
    pub use super::ReshapeSliceExt;
}

/// Reshapes a slice by factoring each dimension into smaller extents
/// under the given limit.
///
/// This transformation increases dimensionality by breaking large
/// sizes into products of smaller factors (e.g., `[1024]` with limit
/// 32 becomes `[32, 32]`). The result is a new [`Slice`] that
/// preserves memory layout and flat index semantics.
///
/// Factoring is greedy, starting from the largest divisors ≤ `limit`.
/// Dimensions that cannot be factored under the limit are left
/// unchanged.
///
/// # Arguments
/// - `slice`: the original multidimensional slice
/// - `limit`: maximum extent allowed in any factored subdimension
///
/// # Returns
/// A reshaped [`Slice`] with updated sizes and strides.
///
/// # Example
/// ```
/// use ndslice::Slice;
/// use ndslice::reshape::Limit;
/// use ndslice::reshape::reshape_with_limit;
///
/// let slice = Slice::new_row_major(vec![1024]);
/// let reshaped = reshape_with_limit(&slice, Limit::new(32));
/// assert_eq!(reshaped.sizes(), &[32, 32]);
/// ```
pub fn reshape_with_limit(slice: &Slice, limit: Limit) -> Slice {
    let orig_sizes = slice.sizes();
    let orig_strides = slice.strides();

    // Step 1: Factor each size into subdimensions ≤ limit.
    let factored_sizes = factor_dims(orig_sizes, limit);

    // Step 2: Compute reshaped sizes and strides (row-major only).
    let reshaped_sizes: Vec<usize> = factored_sizes.iter().flatten().cloned().collect();
    let mut reshaped_strides = Vec::with_capacity(reshaped_sizes.len());

    for (&orig_stride, factors) in orig_strides.iter().zip(&factored_sizes) {
        let mut sub_strides = Vec::with_capacity(factors.len());
        let mut stride = orig_stride;
        for &f in factors.iter().rev() {
            sub_strides.push(stride);
            stride *= f;
        }
        sub_strides.reverse();
        reshaped_strides.extend(sub_strides);
    }

    Slice::new(slice.offset(), reshaped_sizes, reshaped_strides).unwrap()
}

/// Reshapes a labeled [`Shape`] by factoring large extents into
/// smaller ones, producing a new shape with expanded dimensionality
/// and updated labels.
///
/// This uses [`reshape_with_limit`] on the underlying slice and [`expand_labels`]
/// to generate labels for each factored dimension.
///
/// # Arguments
/// - `shape`: the labeled shape to reshape
/// - `limit`: maximum extent allowed per factored dimension
///
/// # Returns
/// A new [`ReshapedShape`] with an updated [`Shape`] and dimension
/// factoring metadata.
///
/// # Panics
/// Panics if constructing the new `Shape` fails. This should not
/// occur unless the reshaped slice and labels are inconsistent (a
/// programming logic error).
pub fn reshape_shape(shape: &Shape, limit: Limit) -> ReshapedShape {
    let reshaped_slice = shape.slice().reshape_with_limit(limit);
    let original_labels = shape.labels();
    let original_sizes = shape.slice().sizes();

    let factors = factor_dims(original_sizes, limit);
    let factored_dims: Vec<(String, Vec<usize>)> =
        original_labels.iter().cloned().zip(factors).collect();

    let labels = expand_labels(&factored_dims);
    let shape = Shape::new(labels, reshaped_slice).expect("invalid reshaped shape");

    ReshapedShape {
        shape,
        factors: factored_dims,
    }
}

/// Expands factored dimension labels into one label per subdimension.
///
/// Each input pair `(label, factors)` represents an original
/// dimension and the extents it was factored into. If a dimension was
/// not factored, it will have a single-element vector.
///
/// For example:
/// - `[("zone", vec![2]), ("gpu", vec![2, 2, 2])]`
///   becomes `["zone", "gpu/0", "gpu/1", "gpu/2"]`
///
/// This is used to generate new labels for reshaped shapes, where the
/// dimensionality increases due to factoring.
///
/// # Arguments
/// - `factors`: a list of factored dimension extents, paired with
///   their labels
///
/// # Returns
/// - A `Vec<String>` of expanded labels, one for each reshaped
///   dimension.
pub fn expand_labels(factors: &[(String, Vec<usize>)]) -> Vec<String> {
    let mut labels = Vec::new();
    for (label, dims) in factors {
        if dims.len() == 1 {
            labels.push(label.clone());
        } else {
            for (i, _) in dims.iter().enumerate() {
                labels.push(format!("{}/{}", label, i));
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Slice;
    use crate::shape;

    #[test]
    fn test_factor_dims_basic() {
        assert_eq!(
            factor_dims(&[6, 8], Limit::from(4)),
            vec![vec![3, 2], vec![4, 2]]
        );
        assert_eq!(factor_dims(&[5], Limit::from(3)), vec![vec![5]]);
        assert_eq!(factor_dims(&[30], Limit::from(5)), vec![vec![5, 3, 2]]);
    }

    // Verify that reshaping preserves memory layout by checking:
    // 1. Coordinate round-tripping: original → reshaped → original
    // 2. Flat index equality: original and reshaped coordinates map
    //    to the same linear index
    // 3. Index inversion: reshaped flat index maps back to the same
    //    reshaped coordinate
    //
    // Together, these checks ensure that the reshaped view is
    // layout-preserving and provides a bijective mapping between
    // coordinate systems.
    #[macro_export]
    macro_rules! assert_layout_preserved {
        ($original:expr, $reshaped:expr) => {{
            // Iterate over all coordinates in the original slice.
            for coord in $original.dim_iter($original.num_dim()) {
                let forward = to_reshaped_coord($original, &$reshaped);
                let inverse = to_original_coord(&$reshaped, $original);
                // Apply the forward coordinate mapping from original
                // to reshaped space.
                let reshaped_coord = forward(&coord);
                // Inverse mapping: reshaped coord → original coord.
                let roundtrip = inverse(&reshaped_coord);
                assert_eq!(
                    roundtrip, coord,
                    "Inverse mismatch: reshaped {:?} → original {:?}, expected {:?}",
                    reshaped_coord, roundtrip, coord
                );
                // Compute flat index in the original slice.
                let flat_orig = $original.location(&coord).unwrap();
                // Compute flat index in the reshaped slice.
                let flat_reshaped = $reshaped.location(&reshaped_coord).unwrap();
                // Check that the flat index is preserved by the
                // reshaping.
                assert_eq!(
                    flat_orig, flat_reshaped,
                    "Flat index mismatch: original {:?} → reshaped {:?}",
                    coord, reshaped_coord
                );
                // Invert the reshaped flat index back to coordinates.
                let recovered = $reshaped.coordinates(flat_reshaped).unwrap();
                // Ensure coordinate inversion is correct (round
                // trip).
                assert_eq!(
                    reshaped_coord, recovered,
                    "Coordinate mismatch: flat index {} → expected {:?}, got {:?}",
                    flat_reshaped, reshaped_coord, recovered
                );
            }
        }};
    }

    #[test]
    fn test_reshape_split_1d_row_major() {
        let s = Slice::new_row_major(vec![1024]);
        let reshaped = s.reshape_with_limit(Limit::from(8));

        assert_eq!(reshaped.offset(), 0);
        assert_eq!(reshaped.sizes(), &vec![8, 8, 8, 2]);
        assert_eq!(reshaped.strides(), &vec![128, 16, 2, 1]);
        assert_eq!(
            factor_dims(s.sizes(), Limit::from(8)),
            vec![vec![8, 8, 8, 2]]
        );

        assert_layout_preserved!(&s, &reshaped);
    }

    #[test]
    fn test_reshape_6_with_limit_2() {
        let s = Slice::new_row_major(vec![6]);
        let reshaped = reshape_with_limit(&s, Limit::from(2));
        assert_eq!(factor_dims(s.sizes(), Limit::from(2)), vec![vec![2, 3]]);
        assert_layout_preserved!(&s, &reshaped);
    }

    #[test]
    fn test_reshape_identity_noop_2d() {
        // All dimensions ≤ limit.
        let original = Slice::new_row_major(vec![4, 8]);
        let reshaped = original.reshape_with_limit(Limit::from(8));

        assert_eq!(reshaped.sizes(), original.sizes());
        assert_eq!(reshaped.strides(), original.strides());
        assert_eq!(reshaped.offset(), original.offset());
        assert_eq!(
            vec![vec![4], vec![8]],
            original
                .sizes()
                .iter()
                .map(|&n| vec![n])
                .collect::<Vec<_>>()
        );
        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_empty_slice() {
        // 0-dimensional slice.
        let original = Slice::new_row_major(vec![]);
        let reshaped = reshape_with_limit(&original, Limit::from(8));

        assert_eq!(reshaped.sizes(), original.sizes());
        assert_eq!(reshaped.strides(), original.strides());
        assert_eq!(reshaped.offset(), original.offset());

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_mixed_dims_3d() {
        // 3D slice with one dimension exceeding the limit.
        let original = Slice::new_row_major(vec![6, 8, 10]);
        let reshaped = original.reshape_with_limit(Limit::from(4));

        assert_eq!(
            factor_dims(original.sizes(), Limit::from(4)),
            vec![vec![3, 2], vec![4, 2], vec![2, 5]]
        );
        assert_eq!(reshaped.sizes(), &[3, 2, 4, 2, 2, 5]);

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_all_large_dims() {
        // 3D slice with all dimensions exceeding the limit.
        let original = Slice::new_row_major(vec![12, 18, 20]);
        let reshaped = original.reshape_with_limit(Limit::from(4));

        assert_eq!(
            factor_dims(original.sizes(), Limit::from(4)),
            vec![vec![4, 3], vec![3, 3, 2], vec![4, 5]]
        );
        assert_eq!(reshaped.sizes(), &[4, 3, 3, 3, 2, 4, 5]);

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_split_1d_factors_3_3_2_2() {
        // 36 = 3 × 3 × 2 × 2.
        let original = Slice::new_row_major(vec![36]);
        let reshaped = reshape_with_limit(&original, Limit::from(3));

        assert_eq!(
            factor_dims(original.sizes(), Limit::from(3)),
            vec![vec![3, 3, 2, 2]]
        );
        assert_eq!(reshaped.sizes(), &[3, 3, 2, 2]);
        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_large_prime_dimension() {
        // Prime larger than limit, cannot be factored.
        let original = Slice::new_row_major(vec![7]);
        let reshaped = reshape_with_limit(&original, Limit::from(4));

        // Should remain as-is since 7 is prime > 4
        assert_eq!(factor_dims(original.sizes(), Limit::from(4)), vec![vec![7]]);
        assert_eq!(reshaped.sizes(), &[7]);

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_split_1d_factors_5_3_2() {
        // 30 = 5 × 3 × 2, all ≤ limit.
        let original = Slice::new_row_major(vec![30]);
        let reshaped = reshape_with_limit(&original, Limit::from(5));

        assert_eq!(
            factor_dims(original.sizes(), Limit::from(5)),
            vec![vec![5, 3, 2]]
        );
        assert_eq!(reshaped.sizes(), &[5, 3, 2]);
        assert_eq!(reshaped.strides(), &[6, 2, 1]);

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_factors_2_6_2_8_8() {
        // 12 = 6 × 2, 64 = 8 × 8 — all ≤ 8
        let original = Slice::new_row_major(vec![2, 12, 64]);
        let reshaped = original.reshape_with_limit(Limit::from(8));

        assert_eq!(
            factor_dims(original.sizes(), Limit::from(8)),
            vec![vec![2], vec![6, 2], vec![8, 8]]
        );
        assert_eq!(reshaped.sizes(), &[2, 6, 2, 8, 8]);
        assert_eq!(reshaped.strides(), &[768, 128, 64, 8, 1]);

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_all_dims_within_limit() {
        // Original shape: [2, 3, 4] — all ≤ limit (4).
        let original = Slice::new_row_major(vec![2, 3, 4]);
        let reshaped = original.reshape_with_limit(Limit::from(4));

        assert_eq!(
            factor_dims(original.sizes(), Limit::from(4)),
            vec![vec![2], vec![3], vec![4]]
        );
        assert_eq!(reshaped.sizes(), &[2, 3, 4]);
        assert_eq!(reshaped.strides(), original.strides());
        assert_eq!(reshaped.offset(), original.offset());

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_reshape_degenerate_dimension() {
        // Degenerate dimension should remain unchanged.
        let original = Slice::new_row_major(vec![1, 12]);
        let reshaped = original.reshape_with_limit(Limit::from(4));

        assert_eq!(
            factor_dims(original.sizes(), Limit::from(4)),
            vec![vec![1], vec![4, 3]]
        );
        assert_eq!(reshaped.sizes(), &[1, 4, 3]);

        assert_layout_preserved!(&original, &reshaped);
    }

    #[test]
    fn test_select_then_reshape() {
        // Original shape: 2 zones, 3 hosts, 4 gpus
        let original = shape!(zone = 2, host = 3, gpu = 4);

        // Select the zone=1 plane: shape becomes [1, 3, 4]
        let selected = original.select("zone", 1).unwrap();
        assert_eq!(selected.slice().offset(), 12); // Nonzero offset.
        assert_eq!(selected.slice().sizes(), &[1, 3, 4]);

        // Reshape the selected slice using limit=2 in row-major
        // layout.
        let reshaped = selected.slice().reshape_with_limit(Limit::from(2));

        assert_eq!(
            factor_dims(selected.slice().sizes(), Limit::from(2)),
            vec![vec![1], vec![3], vec![2, 2]]
        );
        assert_eq!(reshaped.sizes(), &[1, 3, 2, 2]);
        assert_eq!(reshaped.strides(), &[12, 4, 2, 1]);
        assert_eq!(reshaped.offset(), 12); // Offset verified preserved.

        assert_layout_preserved!(selected.slice(), &reshaped);
    }

    #[test]
    fn test_select_host_plane_then_reshape() {
        // Original shape: 2 zones, 3 hosts, 4 gpus.
        let original = shape!(zone = 2, host = 3, gpu = 4);
        // Select the host=2 plane: shape becomes [2, 1, 4].
        let selected = original.select("host", 2).unwrap();
        // Reshape the selected slice using limit=2 in row-major
        // layout.
        let reshaped = selected.slice().reshape_with_limit(Limit::from(2));

        assert_layout_preserved!(selected.slice(), &reshaped);
    }

    #[test]
    fn test_reshape_after_select_no_factoring_due_to_primes() {
        // Original shape: 3 zones, 4 hosts, 5 gpus
        let original = shape!(zone = 3, host = 4, gpu = 5);
        // First select: fix zone = 1 → shape: [1, 4, 5].
        let selected_zone = original.select("zone", 1).unwrap();
        assert_eq!(selected_zone.slice().sizes(), &[1, 4, 5]);
        // Second select: fix host = 2 → shape: [1, 1, 5].
        let selected_host = selected_zone.select("host", 2).unwrap();
        assert_eq!(selected_host.slice().sizes(), &[1, 1, 5]);
        // Reshape with limit = 2.
        let reshaped = selected_host.slice().reshape_with_limit(Limit::from(2));

        assert_eq!(
            factor_dims(selected_host.slice().sizes(), Limit::from(2)),
            vec![vec![1], vec![1], vec![5]]
        );
        assert_eq!(reshaped.sizes(), &[1, 1, 5]);

        assert_layout_preserved!(selected_host.slice(), &reshaped);
    }

    #[test]
    fn test_reshape_after_multiple_selects_triggers_factoring() {
        // Original shape: 2 zones, 4 hosts, 8 gpus
        let original = shape!(zone = 2, host = 4, gpu = 8);
        // Select zone=1 → shape: [1, 4, 8]
        let selected_zone = original.select("zone", 1).unwrap();
        assert_eq!(selected_zone.slice().sizes(), &[1, 4, 8]);

        // Select host=2 → shape: [1, 1, 8]
        let selected_host = selected_zone.select("host", 2).unwrap();
        assert_eq!(selected_host.slice().sizes(), &[1, 1, 8]);

        // Reshape with limit = 2 → gpu=8 should factor
        let reshaped = selected_host.slice().reshape_with_limit(Limit::from(2));

        assert_eq!(
            factor_dims(selected_host.slice().sizes(), Limit::from(2)),
            vec![vec![1], vec![1], vec![2, 2, 2]]
        );
        assert_eq!(reshaped.sizes(), &[1, 1, 2, 2, 2]);

        assert_layout_preserved!(selected_host.slice(), &reshaped);
    }

    #[test]
    fn test_expand_labels_singleton_dims() {
        let factors = vec![("x".into(), vec![2]), ("y".into(), vec![4])];
        let expected = vec!["x", "y"];
        assert_eq!(expand_labels(&factors), expected);
    }

    #[test]
    fn test_expand_labels_factored_dims() {
        let factors = vec![("gpu".into(), vec![2, 2, 2])];
        let expected = vec!["gpu/0", "gpu/1", "gpu/2"];
        assert_eq!(expand_labels(&factors), expected);
    }

    #[test]
    fn test_expand_labels_mixed_dims() {
        let factors = vec![("zone".into(), vec![2]), ("gpu".into(), vec![2, 2])];
        let expected = vec!["zone", "gpu/0", "gpu/1"];
        assert_eq!(expand_labels(&factors), expected);
    }

    #[test]
    fn test_expand_labels_empty() {
        let factors: Vec<(String, Vec<usize>)> = vec![];
        let expected: Vec<String> = vec![];
        assert_eq!(expand_labels(&factors), expected);
    }

    #[test]
    fn test_reshape_shape_noop() {
        let shape = shape!(x = 4, y = 8);
        let reshaped = reshape_shape(&shape, Limit::from(8));
        assert_eq!(reshaped.shape.labels(), &["x", "y"]);
        assert_eq!(reshaped.shape.slice(), shape.slice());
    }

    #[test]
    fn test_reshape_shape_factored() {
        let shape = shape!(gpu = 8);
        let reshaped = reshape_shape(&shape, Limit::from(2));
        assert_eq!(reshaped.shape.labels(), &["gpu/0", "gpu/1", "gpu/2"]);
        assert_eq!(reshaped.shape.slice().sizes(), &[2, 2, 2]);

        let expected = shape.slice().reshape_with_limit(Limit::from(2));
        assert_eq!(reshaped.shape.slice(), &expected);
    }

    #[test]
    fn test_reshape_shape_singleton() {
        let shape = shape!(x = 3);
        let reshaped = reshape_shape(&shape, Limit::from(8));
        assert_eq!(reshaped.shape.labels(), &["x"]);
        assert_eq!(reshaped.shape.slice(), shape.slice());
    }

    #[test]
    fn test_reshape_shape_prime_exceeds_limit() {
        let shape = shape!(x = 11);
        let reshaped = reshape_shape(&shape, Limit::from(5));
        assert_eq!(reshaped.shape.labels(), &["x"]);
        assert_eq!(reshaped.shape.slice(), shape.slice());
    }

    #[test]
    fn test_reshape_shape_mixed_dims() {
        let shape = shape!(zone = 2, gpu = 8);
        let reshaped = reshape_shape(&shape, Limit::from(2));
        assert_eq!(
            reshaped.shape.labels(),
            &["zone", "gpu/0", "gpu/1", "gpu/2"]
        );
        assert_eq!(reshaped.shape.slice().sizes(), &[2, 2, 2, 2]);

        let expected = shape.slice().reshape_with_limit(Limit::from(2));
        assert_eq!(reshaped.shape.slice(), &expected);
    }

    #[test]
    fn test_reshape_shape_after_selects() {
        // Original shape: 2 zones, 4 hosts, 8 gpus
        let original = shape!(zone = 2, host = 4, gpu = 8);

        // Select zone=1 → shape: [1, 4, 8]
        let selected_zone = original.select("zone", 1).unwrap();
        assert_eq!(selected_zone.slice().sizes(), &[1, 4, 8]);

        // Select host=2 → shape: [1, 1, 8]
        let selected_host = selected_zone.select("host", 2).unwrap();
        assert_eq!(selected_host.slice().sizes(), &[1, 1, 8]);

        // Reshape shape through high-level API
        let reshaped = reshape_shape(&selected_host, Limit::from(2));

        // Labels should be: zone, host, gpu/0, gpu/1, gpu/2
        assert_eq!(
            reshaped.shape.labels(),
            &["zone", "host", "gpu/0", "gpu/1", "gpu/2"]
        );

        // Sizes should reflect factored GPU dimension
        assert_eq!(reshaped.shape.slice().sizes(), &[1, 1, 2, 2, 2]);

        // Check against low-level equivalent reshaped slice
        let expected = selected_host.slice().reshape_with_limit(Limit::from(2));
        assert_eq!(reshaped.shape.slice(), &expected);
    }
}
