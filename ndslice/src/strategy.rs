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
    use std::collections::HashSet;

    use proptest::strategy::ValueTree;
    use proptest::test_runner::Config;
    use proptest::test_runner::TestRunner;

    use super::*;
    use crate::selection::EvalOpts;
    use crate::selection::routing::RoutingFrame;

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

    // Ensures `trace_route` exhibits path determinism.
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
            cases: 32, ..ProptestConfig::default()
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
}
