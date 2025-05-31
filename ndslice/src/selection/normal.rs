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

impl NormalizedSelection {
    /// Applies a transformation to each child node of the selection.
    ///
    /// This performs a single-layer traversal, applying `f` to each
    /// immediate child and reconstructing the outer node with the
    /// transformed children.
    pub fn trav<F>(self, mut f: F) -> Self
    where
        F: FnMut(Self) -> Self,
    {
        use NormalizedSelection::*;

        match self {
            All(inner) => All(Box::new(f(*inner))),
            First(inner) => First(Box::new(f(*inner))),
            Any(inner) => Any(Box::new(f(*inner))),
            Range(r, inner) => Range(r, Box::new(f(*inner))),
            Label(labels, inner) => Label(labels, Box::new(f(*inner))),
            Union(set) => Union(set.into_iter().map(f).collect()),
            Intersection(set) => Intersection(set.into_iter().map(f).collect()),
            leaf @ (True | False) => leaf,
        }
    }
}

/// A trait representing a single bottom-up rewrite rule on normalized
/// selections.
///
/// Implementors define a single transformation step, applied after
/// children have already been normalized. These rewrite rules are
/// composed during normalization to simplify or canonicalize
/// selection expressions.
///
/// This trait serves as the building block for constructing
/// normalization passes. Future variants may include top-down rules
/// or contextual rewrites.
///
/// See [`normalize_with`] for integration.
pub trait RewriteRule {
    /// Applies a rewrite step to a node whose children have already
    /// been recursively rewritten.
    fn rewrite(&self, node: NormalizedSelection) -> NormalizedSelection;
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

/// A normalization rule that applies simple algebraic identities.
#[derive(Default)]
pub struct IdentityRules;

impl RewriteRule for IdentityRules {
    fn rewrite(&self, node: NormalizedSelection) -> NormalizedSelection {
        use NormalizedSelection::*;

        match node {
            // All(All(x)) → All(x)
            All(inner) => match *inner {
                All(grandchild) => All(grandchild),
                True => True,
                False => False,
                _ => All(inner),
            },

            // Intersection(True, x) → x
            Intersection(set) => {
                if set.contains(&True) && set.len() > 1 {
                    let mut s = set;
                    s.remove(&True);
                    s.into_iter().next().unwrap()
                } else {
                    Intersection(set)
                }
            }

            // Union(x) → x
            Union(set) if set.len() == 1 => set.into_iter().next().unwrap(),

            _ => node,
        }
    }
}

impl NormalizedSelection {
    pub fn rewrite_bottom_up(self, rule: &impl RewriteRule) -> Self {
        let mapped = self.trav(|child| child.rewrite_bottom_up(rule));
        rule.rewrite(mapped)
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

    #[test]
    fn normalize_smoke_test() {
        use crate::assert_structurally_eq;
        use crate::selection::dsl::*;
        use crate::selection::normalize;
        use crate::selection::parse::parse;

        // The expression (*,*) | (*,*) parses as
        // Union(All(All(True)), All(All(True))) and normalizes all
        // the way down to True.
        let sel = parse("(*,*) | (*,*)").unwrap();
        let normed = normalize(&sel);
        let expected = true_();

        assert_structurally_eq!(&normed, &expected);
    }
}
