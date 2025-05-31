/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeSet;

use crate::selection::LabelKey;
use crate::selection::SelectionSYM;
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

impl SelectionSYM for NormalizedSelection {
    fn true_() -> Self {
        Self::True
    }

    fn false_() -> Self {
        Self::False
    }

    fn all(inner: Self) -> Self {
        Self::All(Box::new(inner))
    }

    fn first(inner: Self) -> Self {
        Self::First(Box::new(inner))
    }

    fn range<R: Into<shape::Range>>(range: R, inner: Self) -> Self {
        Self::Range(range.into(), Box::new(inner))
    }

    fn label<L: Into<LabelKey>>(labels: Vec<L>, inner: Self) -> Self {
        Self::Label(
            labels.into_iter().map(Into::into).collect(),
            Box::new(inner),
        )
    }

    fn any(inner: Self) -> Self {
        Self::Any(Box::new(inner))
    }

    fn intersection(lhs: Self, rhs: Self) -> Self {
        let mut set = BTreeSet::new();
        set.insert(lhs);
        set.insert(rhs);
        Self::Intersection(set)
    }

    fn union(lhs: Self, rhs: Self) -> Self {
        let mut set = BTreeSet::new();
        set.insert(lhs);
        set.insert(rhs);
        Self::Union(set)
    }
}
