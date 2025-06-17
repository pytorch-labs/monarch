/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod stencil {
    /// Generates the von Neumann stencil for grids of arbitrary
    /// dimensionality.
    ///
    /// The von Neumann neighborhood consists of all neighbors offset by
    /// ±1 along a single axis, with all other coordinates unchanged. In
    /// `n` dimensions, this yields `2 * n` neighbors.
    ///
    /// For example, in 3D, this returns the offsets:
    /// ```text
    /// [-1, 0, 0], [1, 0, 0],
    /// [0, -1, 0], [0, 1, 0],
    /// [0, 0, -1], [0, 0, 1]
    /// ```
    pub fn von_neumann_neighbors<const N: usize>() -> Vec<[isize; N]> {
        let mut offsets = Vec::with_capacity(2 * N);

        for axis in 0..N {
            let mut offset_pos = [0; N];
            offset_pos[axis] = 1;
            offsets.push(offset_pos);

            let mut offset_neg = [0; N];
            offset_neg[axis] = -1;
            offsets.push(offset_neg);
        }

        offsets
    }

    /// Generates the Moore stencil for grids of arbitrary
    /// dimensionality.
    ///
    /// The Moore neighborhood consists of all neighbors where each
    /// coordinate offset is in {-1, 0, 1}. In `N` dimensions, this
    /// yields `3^N - 1` neighbors (excluding the center point).
    ///
    /// For example, in 3D, this returns offsets like:
    /// ```text
    /// [-1, -1, -1], [-1, -1, 0], ..., [1, 1, 1] (excluding [0, 0, 0])
    /// ```
    pub fn moore_neighbors<const N: usize>() -> Vec<[isize; N]> {
        let mut offsets = Vec::new();
        let mut prefix = [0isize; N];

        fn build<const N: usize>(
            index: usize,
            prefix: &mut [isize; N],
            offsets: &mut Vec<[isize; N]>,
        ) {
            if index == N {
                if prefix.iter().any(|&x| x != 0) {
                    offsets.push(*prefix);
                }
                return;
            }

            for delta in -1..=1 {
                prefix[index] = delta;
                build::<N>(index + 1, prefix, offsets);
            }
        }

        build::<N>(0, &mut prefix, &mut offsets);
        offsets
    }
}

/// Applies a stencil pattern to coordinates, returning valid
/// resulting coordinates.
///
/// Given base coordinates and a set of offset vectors (the stencil),
/// computes the coordinates that result from applying each offset.
/// Only returns coordinates that fall within the specified bounds.
///
/// # Arguments
///
/// * `coords` - Base coordinates in N-dimensional space
/// * `sizes` - Size bounds for each dimension (coordinates must be <
///   size)
/// * `offsets` - Collection of offset vectors to apply to the base
///   coordinates
///
/// # Returns
///
/// An iterator yielding valid coordinates (as `Vec<usize>`) for each
/// offset that produces in-bounds results. Out-of-bounds results are
/// filtered out.
///
/// # Panics
///
/// Panics if `coords` and `sizes` have different lengths, or if any
/// offset vector has a different length than `coords`.
///
/// # Examples
///
/// ```rust
/// let coords = &[1, 1];
/// let sizes = &[3, 3];
/// let offsets: [[isize; 2]; 4] = [[-1, 0], [1, 0], [0, -1], [0, 1]];
///
/// let results: Vec<_> = ndslice::utils::apply_stencil(coords, sizes, &offsets).collect();
/// // Results in: [[0, 1], [2, 1], [1, 0], [1, 2]]
/// ```
pub fn apply_stencil<'a, const N: usize>(
    coords: &'a [usize; N],
    sizes: &'a [usize; N],
    offsets: &'a [[isize; N]],
) -> impl Iterator<Item = [usize; N]> + 'a {
    offsets.iter().filter_map(move |offset| {
        let mut p = [0usize; N];
        for dim in 0..N {
            let c = coords[dim] as isize;
            let size = sizes[dim] as isize;
            let val = c + offset[dim];
            if val < 0 || val >= size {
                return None;
            }
            p[dim] = val as usize;
        }
        Some(p)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_von_neumann_neighbors_2d() {
        let offsets = stencil::von_neumann_neighbors::<2>();
        assert_eq!(offsets.len(), 4);

        let expected: [[isize; 2]; 4] = [[1, 0], [-1, 0], [0, 1], [0, -1]];

        for e in expected.iter() {
            assert!(offsets.contains(e), "missing offset {:?}", e);
        }
    }

    #[test]
    fn test_von_neumann_neighbors_3d() {
        let offsets = stencil::von_neumann_neighbors::<3>();
        assert_eq!(offsets.len(), 6);

        #[rustfmt::skip]
        let expected: [[isize; 3]; 6] = [
            [1, 0, 0],  [-1, 0, 0],
            [0, 1, 0],  [0, -1, 0],
            [0, 0, 1],  [0, 0, -1],
        ];

        for e in expected.iter() {
            assert!(offsets.contains(e), "missing offset {:?}", e);
        }
    }

    #[test]
    fn test_moore_neighbors_2d() {
        let offsets = stencil::moore_neighbors::<2>();
        assert_eq!(offsets.len(), 8); // 3^2 - 1

        #[rustfmt::skip]
        let expected: [[isize; 2]; 8] = [
            [-1, -1], [-1, 0], [-1, 1],
            [ 0, -1],          [ 0, 1],
            [ 1, -1], [ 1, 0], [ 1, 1],
        ];

        for e in expected.iter() {
            assert!(offsets.contains(e), "missing offset {:?}", e);
        }
    }

    #[test]
    fn test_moore_neighbors_3d() {
        let offsets = stencil::moore_neighbors::<3>();
        assert_eq!(offsets.len(), 26); // 3^3 - 1

        // Spot-check just a few offsets, no need to write out all 26
        #[rustfmt::skip]
        let expected: [[isize; 3]; 5] = [
            [-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  1,  1],
            [-1, -1, -1],
        ];

        for e in expected.iter() {
            assert!(offsets.contains(e), "missing offset {:?}", e);
        }
    }

    #[test]
    fn test_apply_stencil_2d() {
        let coords: [usize; 2] = [1, 1];
        let sizes: [usize; 2] = [3, 3];

        #[rustfmt::skip]
        let offsets: [[isize; 2]; 4] = [
            [-1, 0], [1, 0],
            [0, -1], [0, 1],
        ];

        let results: Vec<_> = apply_stencil(&coords, &sizes, &offsets).collect();
        let expected: [[usize; 2]; 4] = [[0, 1], [2, 1], [1, 0], [1, 2]];

        for e in expected.iter() {
            assert!(results.contains(e), "missing result {:?}", e);
        }
        assert_eq!(results.len(), expected.len());
    }
}
