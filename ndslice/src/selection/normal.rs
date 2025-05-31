/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeSet;

use crate::selection::LabelKey;
use crate::shape;

/// A normalized form of `Selection`, used during canonicalization.
///
/// This structure uses `BTreeSet` for `Union` and `Intersection` to
/// enable flattening, deduplication, and deterministic ordering.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NormalizedSelection {
    False,
    True,
    All(Box<NormalizedSelection>),
    First(Box<NormalizedSelection>),
    Range(shape::Range, Box<NormalizedSelection>),
    Label(Vec<LabelKey>, Box<NormalizedSelection>),
    Any(Box<NormalizedSelection>),
    Union(BTreeSet<NormalizedSelection>),
    Intersection(BTreeSet<NormalizedSelection>),
}
