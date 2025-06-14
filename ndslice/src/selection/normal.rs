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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
/// Implementors define a transformation step applied after children
/// have been rewritten. These rules are composed into normalization
/// passes (see [`normalize`]) to simplify or canonicalize selection
/// expressions.
///
/// This trait forms the basis for extensible normalization. Future
/// systems may support top-down or contextual rewrites as well.
pub trait RewriteRule {
    /// Applies a rewrite step to a node whose children have already
    /// been recursively rewritten.
    fn rewrite(&self, node: NormalizedSelection) -> NormalizedSelection;
}

impl<R1: RewriteRule, R2: RewriteRule> RewriteRule for (R1, R2) {
    fn rewrite(&self, node: NormalizedSelection) -> NormalizedSelection {
        self.1.rewrite(self.0.rewrite(node))
    }
}

/// Extension trait for composing rewrite rules in a fluent style.
///
/// This trait provides a `then` method that allows chaining rewrite
/// rules together, creating a pipeline where rules are applied
/// left-to-right.
pub trait RewriteRuleExt: RewriteRule + Sized {
    /// Chains this rule with another rule, creating a composite rule
    /// that applies `self` first, then `other`.
    fn then<R: RewriteRule>(self, other: R) -> (Self, R) {
        (self, other)
    }
}

impl<T: RewriteRule> RewriteRuleExt for T {}

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
    // Identity rewrites:
    //
    // - All(All(x))           → All(x)    // idempotence
    // - All(True)             → True      // identity
    // - All(False)            → False     // passthrough
    // - Intersection(True, x) → x         // identity
    // - Intersection({x})     → x         // trivial
    // - Intersection({})      → True      // identity
    // - Union(False, x)       → x         // identity
    // - Union({x})            → x         // trivial
    // - Union({})             → False     // trivial
    //
    // Absorbtion rules like `Union(True, x) → x` are handled in a
    // different rewrite.
    fn rewrite(&self, node: NormalizedSelection) -> NormalizedSelection {
        use NormalizedSelection::*;

        match node {
            All(inner) => match *inner {
                All(grandchild) => All(grandchild), // All(All(x)) → All(x)
                True => True,                       // All(True) → True
                False => False,                     // All(False) → False
                _ => All(inner),
            },

            Intersection(mut set) => {
                set.remove(&True); // Intersection(True, ...)  → ...
                match set.len() {
                    0 => True,
                    1 => set.into_iter().next().unwrap(), // Intersection(x) → x
                    _ => Intersection(set),
                }
            }

            Union(mut set) => {
                set.remove(&False); // Union(False, ...) → ...
                match set.len() {
                    0 => False,
                    1 => set.into_iter().next().unwrap(), // Union(x) → x
                    _ => Union(set),
                }
            }

            _ => node,
        }
    }
}

/// A normalization rule that flattens nested unions and
/// intersections.
#[derive(Default)]
pub struct FlatteningRules;

impl RewriteRule for FlatteningRules {
    // Flattening rewrites:
    //
    // - Union(a, Union(b, c))               → Union(a, b, c)           // flatten nested unions
    // - Intersection(a, Intersection(b, c)) → Intersection(a, b, c)    // flatten nested intersections
    fn rewrite(&self, node: NormalizedSelection) -> NormalizedSelection {
        use NormalizedSelection::*;

        match node {
            Union(set) => {
                let mut flattened = BTreeSet::new();
                for item in set {
                    match item {
                        Union(inner_set) => {
                            flattened.extend(inner_set);
                        }
                        other => {
                            flattened.insert(other);
                        }
                    }
                }
                Union(flattened)
            }
            Intersection(set) => {
                let mut flattened = BTreeSet::new();
                for item in set {
                    match item {
                        Intersection(inner_set) => {
                            flattened.extend(inner_set);
                        }
                        other => {
                            flattened.insert(other);
                        }
                    }
                }
                Intersection(flattened)
            }
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

        assert_structurally_eq!(&normed.into(), &expected);
    }

    #[test]
    fn test_union_flattening() {
        use NormalizedSelection::*;

        // Create Union(a, Union(b, c)) manually
        let inner_union = {
            let mut set = BTreeSet::new();
            set.insert(All(Box::new(True))); // represents 'b'
            set.insert(Any(Box::new(True))); // represents 'c'
            Union(set)
        };

        let outer_union = {
            let mut set = BTreeSet::new();
            set.insert(First(Box::new(True))); // represents 'a'
            set.insert(inner_union);
            Union(set)
        };

        let rule = FlatteningRules;
        let result = rule.rewrite(outer_union);

        // Should be flattened to Union(a, b, c)
        if let Union(set) = result {
            assert_eq!(set.len(), 3);
            assert!(set.contains(&First(Box::new(True))));
            assert!(set.contains(&All(Box::new(True))));
            assert!(set.contains(&Any(Box::new(True))));
        } else {
            panic!("Expected Union, got {:?}", result);
        }
    }

    #[test]
    fn test_intersection_flattening() {
        use NormalizedSelection::*;

        // Create Intersection(a, Intersection(b, c)) manually
        let inner_intersection = {
            let mut set = BTreeSet::new();
            set.insert(All(Box::new(True))); // represents 'b'
            set.insert(Any(Box::new(True))); // represents 'c'
            Intersection(set)
        };

        let outer_intersection = {
            let mut set = BTreeSet::new();
            set.insert(First(Box::new(True))); // represents 'a'
            set.insert(inner_intersection);
            Intersection(set)
        };

        let rule = FlatteningRules;
        let result = rule.rewrite(outer_intersection);

        // Should be flattened to Intersection(a, b, c)
        if let Intersection(set) = result {
            assert_eq!(set.len(), 3);
            assert!(set.contains(&First(Box::new(True))));
            assert!(set.contains(&All(Box::new(True))));
            assert!(set.contains(&Any(Box::new(True))));
        } else {
            panic!("Expected Intersection, got {:?}", result);
        }
    }
}
