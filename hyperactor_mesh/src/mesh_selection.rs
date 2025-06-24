/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// until used publically
#![allow(dead_code)]

use ndslice::Selection;
use ndslice::shape::Range;
use ndslice::shape::Shape;

/// Specifies how to handle dimensions that exist in one mesh but not the other
#[derive(Clone, Copy)]
pub enum MappingMode {
    /// Broadcast to any dimensions that exist only in the target mesh
    /// Dimensions only in origin mesh are handled by rank iteration
    BroadcastMissing,
    /// Error if any dimensions exist in only one of the meshes
    ExactMatch,
}

/// Describes how two meshes should be aligned
#[derive(Clone, Copy)]
pub enum AlignPolicy {
    /// Creates a mapping from each origin rank to the entire target mesh
    Broadcast,
    /// Maps matching dimensions 1:1 and handles missing dimensions according to MappingMode
    Mapped {
        /// Specifies how to handle dimensions that exist in one mesh but not the other
        mode: MappingMode,
    },
}

/// Verifies that any matching dimensions between origin and target shapes have equal sizes
fn verify_dimension_sizes(origin: &Shape, target: &Shape) -> Result<(), anyhow::Error> {
    for label in origin.labels() {
        if let Some(target_pos) = target.labels().iter().position(|l| l == label) {
            let origin_pos = origin.labels().iter().position(|l| l == label).unwrap();
            let origin_size = origin.slice().sizes()[origin_pos];
            let target_size = target.slice().sizes()[target_pos];

            if origin_size != target_size {
                return Err(anyhow::Error::msg(format!(
                    "dimension {} has mismatched sizes: {} vs {}",
                    label, origin_size, target_size
                )));
            }
        }
    }
    Ok(())
}

/// Given a set of coordinates, create a set of single-value ranges for each dimension..
fn exact_mapping<'a>(
    target_selection: &Selection,
    coords: &'a [(String, usize)],
    origin_rank: &usize,
) -> Result<Vec<(&'a str, Range)>, anyhow::Error> {
    let coord_dim = coords.iter().map(|(_, d)| *d).collect::<Vec<_>>();
    if target_selection.contains(&coord_dim) {
        Ok(coords
            .iter()
            .map(|(label, index)| (label.as_str(), Range::from(*index)))
            .collect::<Vec<(&'a str, Range)>>())
    } else {
        Err(anyhow::Error::msg(format!(
            "origin rank {} is not selected in target",
            origin_rank
        )))
    }
}
