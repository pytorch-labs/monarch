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

use crate::selection::Selection;
use crate::selection::dsl;
use crate::shape::Range;

pub fn gen_selection(depth: u32) -> BoxedStrategy<Selection> {
    let leaf = Just(dsl::true_()).boxed();

    if depth == 0 {
        return leaf;
    }

    let recur = move || gen_selection(depth - 1);

    let range = (0usize..3, 1usize..3)
        .prop_flat_map(move |(start, len)| {
            recur().prop_map(move |inner| {
                dsl::range(Range(start, Some(start + len), 1), inner.clone())
            })
        })
        .boxed();

    let all = recur().prop_map(dsl::all).boxed();

    let union = (recur(), recur())
        .prop_map(|(a, b)| dsl::union(a, b))
        .boxed();

    let inter = (recur(), recur())
        .prop_map(|(a, b)| dsl::intersection(a, b))
        .boxed();

    prop_oneof![
        2 => leaf,
        3 => range,
        3 => all,
        2 => union,
        2 => inter,
    ]
    .boxed()
}

mod tests {
    use proptest::test_runner::Config;
    use proptest::test_runner::TestRunner;

    use super::*;

    #[test]
    fn sample_many() {
        let mut runner = TestRunner::new(Config::default());

        for _ in 0..5 {
            let strat = gen_selection(3);
            let value = strat.new_tree(&mut runner).unwrap().current();
            println!("{:?}", value);
        }
    }
}
