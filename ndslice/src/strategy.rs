/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Property-based generators for [`Selection`] and related types.
//!
//! These strategies are used in `proptest`-based tests to construct
//! randomized selection expressions for testing evaluation, routing,
//! and normalization logic.
//!
//! The main entry point is [`gen_selection(depth)`], which generates
//! a structurally diverse [`Selection`] of bounded depth, supporting
//! the `True`, `Range`, `All`, `Union`, and `Intersection`
//! constructors.
//!
//! Example usage:
//!
//! ```
//! use proptest::prelude::*;
//!
//! use crate::selection::strategy::gen_selection;
//!
//! proptest! {
//!     #[test]
//!     fn test_selection(s in gen_selection(3)) {
//!         // Use `s` as input to evaluation or routing tests
//!     }
//! }
//! ```
//!
//! This module is only included in test builds (`#[cfg(test)]`).

use proptest::prelude::*;

use crate::Slice;
use crate::selection::EvalOpts;
use crate::selection::Selection;
use crate::selection::dsl;
use crate::shape::Range;

/// Generates a random [`Slice`] with up to `max_dims` dimensions,
/// where each dimension has a size between 1 and `max_len`
/// (inclusive).
///
/// The slice is constructed using standard row-major layout with no
/// offset, making it suitable for use in evaluation, routing, and
/// normalization tests.
///
/// This generator is used in property-based tests to provide diverse
/// input shapes for selection and routing logic.
///
/// # Parameters
///
/// - `max_dims`: Maximum number of dimensions in the generated slice.
/// - `max_len`: Maximum size per dimension.
///
/// # Example
///
/// ```
/// use proptest::prelude::*;
///
/// use crate::selection::strategy::gen_slice;
///
/// proptest! {
///     #[test]
///     fn test_slice_generation(slice in gen_slice(4, 8)) {
///         assert!(!slice.sizes().is_empty());
///     }
/// }
/// ```
pub fn gen_slice(max_dims: usize, max_len: usize) -> impl Strategy<Value = Slice> {
    prop::collection::vec(1..=max_len, 1..=max_dims).prop_map(Slice::new_row_major)
}

/// Generates a pair `(base, subview)` where:
/// - `base` is a randomly shaped row-major `Slice`,
/// - `subview` is a valid rectangular region within `base`.
pub fn gen_slice_and_subview(
    max_dims: usize,
    max_len: usize,
) -> impl Strategy<Value = (Slice, Slice)> {
    assert!(max_dims <= 8, "Supports up to 4 dimensions explicitly");

    gen_slice(max_dims, max_len).prop_flat_map(|base| {
        let sizes = base.sizes().to_vec();

        // Strategy per dimension
        let dim_strat = |extent| {
            if extent == 0 {
                Just((0, 0)).boxed()
            } else {
                (0..extent)
                    .prop_flat_map(move |start| {
                        (1..=extent - start).prop_map(move |len| (start, len))
                    })
                    .boxed()
            }
        };

        // Explicit match based on dims
        match sizes.len() {
            1 => (dim_strat(sizes[0]),).prop_map(|(a,)| vec![a]).boxed(),
            2 => (dim_strat(sizes[0]), dim_strat(sizes[1]))
                .prop_map(|(a, b)| vec![a, b])
                .boxed(),
            3 => (
                dim_strat(sizes[0]),
                dim_strat(sizes[1]),
                dim_strat(sizes[2]),
            )
                .prop_map(|(a, b, c)| vec![a, b, c])
                .boxed(),
            4 => (
                dim_strat(sizes[0]),
                dim_strat(sizes[1]),
                dim_strat(sizes[2]),
                dim_strat(sizes[3]),
            )
                .prop_map(|(a, b, c, d)| vec![a, b, c, d])
                .boxed(),
            5 => (
                dim_strat(sizes[0]),
                dim_strat(sizes[1]),
                dim_strat(sizes[2]),
                dim_strat(sizes[3]),
                dim_strat(sizes[4]),
            )
                .prop_map(|(a, b, c, d, e)| vec![a, b, c, d, e])
                .boxed(),
            6 => (
                dim_strat(sizes[0]),
                dim_strat(sizes[1]),
                dim_strat(sizes[2]),
                dim_strat(sizes[3]),
                dim_strat(sizes[4]),
                dim_strat(sizes[5]),
            )
                .prop_map(|(a, b, c, d, e, f)| vec![a, b, c, d, e, f])
                .boxed(),
            7 => (
                dim_strat(sizes[0]),
                dim_strat(sizes[1]),
                dim_strat(sizes[2]),
                dim_strat(sizes[3]),
                dim_strat(sizes[4]),
                dim_strat(sizes[5]),
                dim_strat(sizes[6]),
            )
                .prop_map(|(a, b, c, d, e, f, g)| vec![a, b, c, d, e, f, g])
                .boxed(),
            8 => (
                dim_strat(sizes[0]),
                dim_strat(sizes[1]),
                dim_strat(sizes[2]),
                dim_strat(sizes[3]),
                dim_strat(sizes[4]),
                dim_strat(sizes[5]),
                dim_strat(sizes[6]),
                dim_strat(sizes[7]),
            )
                .prop_map(|(a, b, c, d, e, f, g, h)| vec![a, b, c, d, e, f, g, h])
                .boxed(),
            _ => unreachable!("max_dims constrained to 8"),
        }
        .prop_map(move |ranges| {
            let (starts, lens): (Vec<_>, Vec<_>) = ranges.into_iter().unzip();
            let subview = base.subview(&starts, &lens).expect("valid subview");
            (base.clone(), subview)
        })
    })
}

/// Recursively generates a random `Selection` expression of bounded
/// depth, aligned with the given slice `shape`.
///
/// Each recursive call corresponds to one dimension of the shape,
/// starting from `dim`, and constructs a selection operator (`range`,
/// `all`, `intersection`, etc.) that applies at that level.
///
/// The recursion proceeds until either:
/// - `depth == 0`, which limits structural complexity, or
/// - `dim >= shape.len()`, which prevents exceeding the
///   dimensionality.
///
/// In both cases, the recursion terminates with a `true_()` leaf
/// node, effectively selecting all remaining elements.
///
/// The resulting selections are guaranteed to be valid under a strict
/// validation regime: they contain no empty ranges, no out-of-bounds
/// accesses, and no dynamic constructs like `any` or `first`.
pub fn gen_selection(depth: u32, shape: Vec<usize>, dim: usize) -> BoxedStrategy<Selection> {
    let leaf = Just(dsl::true_()).boxed();

    if depth == 0 || dim >= shape.len() {
        return leaf;
    }

    let recur = {
        let shape = shape.clone();
        move || gen_selection(depth - 1, shape.clone(), dim + 1)
    };

    let range_strategy = {
        let dim_size = shape[dim];
        (0..dim_size)
            .prop_flat_map(move |start| {
                let max_len = dim_size - start;
                (1..=max_len).prop_flat_map(move |len| {
                    (1..=len).prop_map(move |step| {
                        let r = Range(start, Some(start + len), step);
                        dsl::range(r, dsl::true_())
                    })
                })
            })
            .boxed()
    };

    let all = recur().prop_map(dsl::all).boxed();

    let union = (recur(), recur())
        .prop_map(|(a, b)| dsl::union(a, b))
        .boxed();

    let inter = (recur(), recur())
        .prop_map(|(a, b)| dsl::intersection(a, b))
        .boxed();

    prop_oneof![
        2 => leaf,
        3 => range_strategy,
        3 => all,
        2 => union,
        2 => inter,
    ]
    .prop_filter("valid selection", move |s| {
        let slice = Slice::new_row_major(shape.clone());
        let eval_opts = EvalOpts {
            disallow_dynamic_selections: true,
            ..EvalOpts::strict()
        };
        s.validate(&eval_opts, &slice).is_ok()
    })
    .boxed()
}

mod tests {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use proptest::strategy::ValueTree;
    use proptest::test_runner::Config;
    use proptest::test_runner::TestRunner;

    use super::*;
    use crate::selection::EvalOpts;
    use crate::selection::routing::RoutingFrame;
    use crate::selection::test_utils::collect_commactor_routing_tree;
    use crate::selection::test_utils::collect_routed_paths;

    #[test]
    fn print_some_slices() {
        let mut runner = TestRunner::new(Config::default());

        for _ in 0..256 {
            let strat = gen_slice(4, 8); // up to 4 dimensions, max size per dim = 8
            let value = strat.new_tree(&mut runner).unwrap().current();
            println!("{:?}", value);
        }
    }

    proptest! {
        #[test]
        fn test_slice_properties(slice in gen_slice(4, 8)) {
            let total_size: usize = slice.sizes().iter().product();
            prop_assert!(total_size > 0);
        }
    }

    #[test]
    fn print_some_selections() {
        let mut runner = TestRunner::new(Config::default());

        for _ in 0..256 {
            let strat = gen_selection(3, vec![2, 4, 8], 0);
            let value = strat.new_tree(&mut runner).unwrap().current();
            println!("{:?}", value);
        }
    }

    // Test `trace_route` exhibits path determinism.
    //
    // This test instantiates a general theorem about the selection
    // algebra and its routing semantics:
    //
    //   ∀ `S`, `T`, and `n`,
    //     `n ∈ eval(S, slice)` ∧ `n ∈ eval(T, slice)` ⇒
    //     `route(n, S) == route(n, T)`.
    //
    // This property enables us to enforce in-order delivery using
    // only per-sender, per-peer sequence numbers. Since every message
    // to a given destination follows the same deterministic path
    // through the mesh, intermediate nodes can forward messages in
    // order, and receivers can detect missing or out-of-order
    // messages using only local state.
    //
    // This test uses `trace_route` to observe the path to each
    // overlapping destination node under `S` and `T`, asserting that
    // the results agree.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 8, ..ProptestConfig::default()
        })]
        #[test]
        fn trace_route_path_determinism(
            slice in gen_slice(4, 8)
        ) {
            let shape = slice.sizes().to_vec();

            let mut runner = TestRunner::default();
            let s = gen_selection(4, shape.clone(), 0).new_tree(&mut runner).unwrap().current();
            let t = gen_selection(4, shape.clone(), 0).new_tree(&mut runner).unwrap().current();

            let eval_opts = EvalOpts::strict();
            let sel_s: HashSet<_> = s.eval(&eval_opts, &slice).unwrap().collect();
            let sel_t: HashSet<_> = t.eval(&eval_opts, &slice).unwrap().collect();
            let ranks: Vec<_> = sel_s.intersection(&sel_t).cloned().collect();

            if ranks.is_empty() {
                println!("skipping empty intersection");
            } else {
                println!("testing {} nodes", ranks.len());
                for rank in ranks {
                    let coords = slice.coordinates(rank).unwrap();
                    let start_s = RoutingFrame::root(s.clone(), slice.clone());
                    let start_t = RoutingFrame::root(t.clone(), slice.clone());
                    let path_s = start_s.trace_route(&coords).unwrap();
                    let path_t = start_t.trace_route(&coords).unwrap();
                    prop_assert_eq!(
                        path_s.clone(),
                        path_t.clone(),
                        "path to {:?} differs under S and T\nS path: {:?}\nT path: {:?}",
                        rank, path_s, path_t
                    );
                }
            }
        }
    }

    // Test `collect_routed_paths` exhibits path determinism.
    //
    // This test instantiates the same fundamental property as the
    // `trace_route` test, but does so using `collect_routed_paths`,
    // which performs a breadth-first traversal of the routing tree.
    //
    //   ∀ `S`, `T`, and `n`,
    //     `n ∈ eval(S, slice)` ∧ `n ∈ eval(T, slice)` ⇒
    //     `route(n, S) == route(n, T)`.
    //
    // The property guarantees that every destination node reachable
    // by both `S` and `T` is routed to via the same deterministic
    // path.
    //
    // This test avoids calls to `eval` by intersecting the routed
    // destinations directly. For each rank routed to by both
    // selections, it compares the path returned by
    // `collect_routed_paths`, ensuring the selection algebra routes
    // consistently regardless of expression structure or traversal
    // order.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128, ..ProptestConfig::default()
        })]
        #[test]
        fn collect_routed_path_determinism(
            slice in gen_slice(4, 8)
        ) {
            let shape = slice.sizes().to_vec();

            let mut runner = TestRunner::default();
            let s = gen_selection(4, shape.clone(), 0).new_tree(&mut runner).unwrap().current();
            let t = gen_selection(4, shape.clone(), 0).new_tree(&mut runner).unwrap().current();

            let paths_s = collect_routed_paths(&s, &slice);
            let paths_t = collect_routed_paths(&t, &slice);
            let ranks: Vec<_> = paths_s.delivered.keys()
                .filter(|r| paths_t.delivered.contains_key(*r))
                .cloned()
                .collect();

            if ranks.is_empty() {
                println!("skipping empty intersection");
            } else {
                println!("testing {} nodes", ranks.len());
                for rank in ranks {
                    let path_s = paths_s.delivered.get(&rank).unwrap();
                    let path_t = paths_t.delivered.get(&rank).unwrap();
                    prop_assert_eq!(
                        path_s.clone(),
                        path_t.clone(),
                        "path to {:?} differs under S and T\nS path: {:?}\nT path: {:?}",
                        rank, path_s, path_t
                    );
                }
            }
        }
    }

    // Test `collect_commactor_routing_tree` exhibits path
    // determinism.
    //
    // This test instantiates the same routing path determinism
    // property as in `collect_routed_path_determinism`, but uses the
    // full CommActor-style routing simulation instead.
    //
    //   ∀ `S`, `T`, and `n`,
    //     `n ∈ eval(S, slice)` ∧ `n ∈ eval(T, slice)` ⇒
    //     `route(n, S) == route(n, T)`.
    //
    // This ensures that every destination rank reachable by both `S`
    // and `T` receives its message along the same logical path, even
    // when selection expressions differ structurally.
    //
    // The test avoids explicit calls to eval by intersecting the
    // delivered ranks from both traversals. For each rank delivered
    // to by both selections, it compares the delivery path recorded
    // in `CommActorRoutingTree::delivered`. This validates that
    // CommActor message routing is structurally deterministic.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 128, ..ProptestConfig::default()
        })]
        #[test]
        fn collect_commactor_routed_path_determinism(
            slice in gen_slice(4, 8)
        ) {
            let extents = slice.sizes().to_vec();

            let mut runner = TestRunner::default();
            let s = gen_selection(4, extents.clone(), 0).new_tree(&mut runner).unwrap().current();
            let t = gen_selection(4, extents.clone(), 0).new_tree(&mut runner).unwrap().current();

            let tree_s = collect_commactor_routing_tree(&s, &slice);
            let tree_t = collect_commactor_routing_tree(&t, &slice);

            let ranks: Vec<_> = tree_s
                .delivered
                .keys()
                .filter(|r| tree_t.delivered.contains_key(*r))
                .cloned()
                .collect();

            if ranks.is_empty() {
                println!("skipping empty intersection");
            } else {
                println!("testing {} nodes", ranks.len());
                for rank in ranks {
                    let path_s = &tree_s.delivered[&rank];
                    let path_t = &tree_t.delivered[&rank];
                    prop_assert_eq!(
                        path_s.clone(),
                        path_t.clone(),
                        "path to {:?} differs under S and T\nS path: {:?}\nT path: {:?}",
                        rank, path_s, path_t
                    );
                }
            }
        }
    }

    // Property test: Unique Predecessor Theorem
    //
    // This test verifies a structural invariant of the routing graph
    // produced by `collect_routed_paths`, which performs a
    // breadth-first traversal of a selection over a multidimensional
    // mesh.
    //
    // ───────────────────────────────────────────────────────────────
    // Unique Predecessor Theorem
    //
    // In a full routing traversal, each coordinate `x` is the target
    // of at most one `RoutingStep::Forward` from a distinct
    // coordinate `y ≠ x`.
    //
    // Any additional frames that reach `x` arise only from:
    //   - self-forwarding (i.e., `x → x`)
    //   - structural duplication from the same parent node (e.g., via
    //     unions)
    //
    // This ensures that routing paths form a tree-like structure
    // rooted at the origin, with no multiple distinct predecessors
    // except in the degenerate (self-loop) or duplicated-parent
    // cases.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, ..ProptestConfig::default()
        })]
        #[test]
        fn collect_routed_paths_unique_predecessor(
            slice in gen_slice(4, 8)
        ) {
            let shape = slice.sizes().to_vec();

            let mut runner = TestRunner::default();
            let s = gen_selection(4, shape.clone(), 0).new_tree(&mut runner).unwrap().current();

            let tree = collect_routed_paths(&s, &slice);

            for (node, preds) in tree.predecessors {
                let non_self_preds: Vec<_> = preds.clone().into_iter()
                    .filter(|&p| p != node)
                    .collect();

                prop_assert!(
                    non_self_preds.len() <= 1,
                    "Node {} had multiple non-self predecessors: {:?} (selection: {})",
                    node,
                    non_self_preds,
                    s,
                );
            }
        }
    }

    // Property test: Unique Predecessor Theorem (CommActor Routing)
    //
    // This test verifies structural invariants of the routing graph
    // produced by `collect_commactor_routing_tree`, which simulates
    // CommActor-style peer-to-peer multicast forwarding.
    //
    // ───────────────────────────────────────────────────────────────
    // Unique Predecessor Theorem
    //
    // In a full routing traversal, each coordinate `x` is the target
    // of at most one `RoutingStep::Forward` from a distinct
    // coordinate `y ≠ x`.
    //
    // Any additional frames that reach `x` arise only from:
    //   - structural duplication from the same parent node (e.g., via
    //     unions)
    //
    // Unlike the general `collect_routed_paths`, CommActor routing
    // never performs self-forwarding (`x → x`). This test confirms
    // that as well.
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256, ..ProptestConfig::default()
        })]
        #[test]
        fn commactor_routed_paths_unique_predecessor(
            slice in gen_slice(4, 8)
        ) {
            let shape = slice.sizes().to_vec();

            let mut runner = TestRunner::default();
            let s = gen_selection(4, shape.clone(), 0).new_tree(&mut runner).unwrap().current();

            let tree = collect_commactor_routing_tree(&s, &slice);

            let mut preds: HashMap<usize, HashSet<usize>> = HashMap::new();

            for (from, frames) in &tree.forwards {
                for frame in frames {
                    let to = slice.location(&frame.here).unwrap();

                    // We assert that a CommActor never forwards to
                    // itself.
                    prop_assert_ne!(
                        *from, to,
                        "CommActor forwarded to itself: {} → {} (selection: {})",
                        from, to, s
                    );

                    preds.entry(to).or_default().insert(*from);
                }
            }

            for (node, parents) in preds {
                let non_self_preds: Vec<_> = parents.into_iter()
                    .filter(|&p| p != node)
                    .collect();

                prop_assert!(
                    non_self_preds.len() <= 1,
                    "Node {} had multiple non-self predecessors: {:?} (selection: {})",
                    node,
                    non_self_preds,
                    s,
                );
            }
        }
    }

    // Theorem (Subview-Coordinate Inclusion):
    //
    // For any rectangular subview `V` of a base slice `B`, each
    // coordinate `v` in `V` maps to a coordinate `b = view_offset +
    // v` that must be in `B`.
    //
    // This test verifies that all coordinates of a generated subview
    // are valid when translated back into the coordinate system of
    // the base slice.
    proptest! {
        #[test]
        fn test_gen_slice_and_subview((base, subview) in gen_slice_and_subview(4, 8)) {
            for idx in subview.iter() {
                let v = subview.coordinates(idx).unwrap();
                let view_offset_in_base = base.coordinates(subview.offset()).unwrap();

                // b = view_offset + v
                let b: Vec<_> = v.iter()
                    .zip(&view_offset_in_base)
                    .map(|(sub_c, offset)| sub_c + offset)
                    .collect();

                assert!(base.location(&b).is_ok());
            }
        }
    }
}
