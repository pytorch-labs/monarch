/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeSet;

use crate::Selection;
use crate::selection::LabelKey;
use crate::selection::SelectionSYM;
use crate::selection::dsl;
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

impl From<NormalizedSelection> for Selection {
    /// Converts the normalized form back into a standard `Selection`.
    ///
    /// Logical semantics are preserved, but normalized shape (e.g.,
    /// set-based unions and intersections) is reconstructed as
    /// left-associated binary trees.
    fn from(norm: NormalizedSelection) -> Self {
        use NormalizedSelection::*;
        use dsl::*;

        match norm {
            True => true_(),
            False => false_(),
            All(inner) => all((*inner).into()),
            First(inner) => first((*inner).into()),
            Any(inner) => any((*inner).into()),
            Union(set) => set
                .into_iter()
                .map(Into::into)
                .reduce(Selection::union)
                .unwrap_or_else(false_),
            Intersection(set) => set
                .into_iter()
                .map(Into::into)
                .reduce(Selection::intersection)
                .unwrap_or_else(true_),
            Range(r, inner) => Selection::range(r, (*inner).into()),
            Label(labels, inner) => Selection::label(labels, (*inner).into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_structurally_eq;
    use crate::selection;
    use crate::selection::parse::parse;

    /// Verifies that:
    /// - Duplicate subtrees are structurally deduplicated by
    ///   normalization
    /// - The normalized form reifies to the expected `Selection` in
    ///   this case
    #[test]
    fn normalization_deduplicates_and_reifies() {
        let sel = parse("(* & *) | (* & *)").unwrap();
        let norm = sel.fold::<NormalizedSelection>();

        // Expected: Union { Intersection { All(True) } }
        use NormalizedSelection::*;
        let mut inner = BTreeSet::new();
        inner.insert(All(Box::new(True)));

        let mut outer = BTreeSet::new();
        outer.insert(Intersection(inner));

        assert_eq!(norm, Union(outer));

        use selection::dsl::*;
        let reified = norm.into();
        let expected = all(true_());

        assert_structurally_eq!(&reified, &expected);
    }
}
