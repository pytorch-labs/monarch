/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! This module defines a parser of a compact syntax used to
//! describe hierarchical selections over multidimensional meshes.
//! ```text
//! expression       ::= union
//! union            ::= intersection ( "|" intersection )*
//! intersection     ::= chain ( "&" chain )*
//! chain            ::= group ( "," group )*
//! group            ::= range
//!                    | index
//!                    | wildcard
//!                    | any
//!                    | "(" expression ")"
//! range            ::= number? ":" number? ( ":" number )?
//! index            ::= number
//! wildcard         ::= "*"
//! any              ::= "?"
//! number           ::= [0-9]+
//! ```
//!
//! Notes:
//! - `,` separates **nested dimensions** (i.e., descent into next
//!   dimension).
//! - `|` is union, `&` is intersection. `&` binds tighter than `|`.
//! - `*` selects all values at the current dimension and descends.
//! - `?` selects a random value at the current dimension and descends.
//! - A range like `2:5:1` has the form `start:end:step`. Missing
//!   parts default to:
//!     - `start = 0`
//!     - `end = full extent`
//!     - `step = 1`
//! - An index like `3` is shorthand for the range `3:4`.
//! - Parentheses `()` allow grouping for precedence control and
//!   nesting of chains.
//! - Whitespace is not allowed (although the `parse` function will
//!   admit and strip it.)
//! - Expressions like `(*,*,1:4|5:6)` are valid and parsed as:
//!     - A union of two expressions:
//!         1. The chain `*,*,1:4`
//!         2. The standalone `5:6`
//!     - The union applies at the top level, not just within a dimension.
//!     - To apply the union **only to the third dimension**, parentheses must be used:
//!       e.g., `*,*,(1:4|5:6)`

use nom::IResult;
use nom::Parser as _;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::char;
use nom::character::complete::digit1;
use nom::combinator::map;
use nom::combinator::map_res;
use nom::combinator::opt;
use nom::multi::separated_list1;
use nom::sequence::delimited;
use nom::sequence::preceded;

use crate::selection::Selection;
use crate::selection::dsl;
use crate::shape;

fn number(input: &str) -> IResult<&str, usize> {
    map_res(digit1, str::parse).parse(input)
}

fn index(input: &str) -> IResult<&str, Selection> {
    map(number, |n| dsl::range(n, dsl::true_())).parse(input)
}

fn range(input: &str) -> IResult<&str, Selection> {
    let (input, (start, end, step)) = (
        opt(number),
        preceded(char(':'), opt(number)),
        opt(preceded(char(':'), number)),
    )
        .parse(input)?;

    Ok((
        input,
        dsl::range(
            shape::Range(start.unwrap_or(0), end, step.unwrap_or(1)),
            dsl::true_(),
        ),
    ))
}

fn wildcard(input: &str) -> IResult<&str, Selection> {
    map(tag("*"), |_| dsl::all(dsl::true_())).parse(input)
}

fn any(input: &str) -> IResult<&str, Selection> {
    map(tag("?"), |_| dsl::any(dsl::true_())).parse(input)
}

fn group(input: &str) -> IResult<&str, Selection> {
    alt((
        delimited(char('('), expression, char(')')),
        range,
        index,
        wildcard,
        any,
    ))
    .parse(input)
}

fn chain(input: &str) -> IResult<&str, Selection> {
    map(separated_list1(char(','), group), |dims| {
        dims.into_iter()
            .rev()
            .fold(dsl::true_(), |acc, dim| nest(dim, acc))
    })
    .parse(input)
}

// Recursively nests `dim` into `tail`, modeling `,` (dimension
// descent).
//
// For example, the chain `*,1:4,?` is parsed as:
//   [All(True), Range(1..4, True), Any(True)]
//
// Nesting proceeds right to left:
//   Any(True) → Range(1..4, Any(True)) → All(Range(1..4, Any(True)))
//
// Unions and intersections nest both branches.
fn nest(dim: Selection, tail: Selection) -> Selection {
    match dim {
        Selection::All(inner) => dsl::all(nest(*inner, tail)),
        Selection::Any(inner) => dsl::any(nest(*inner, tail)),
        Selection::Range(r, inner) => dsl::range(r, nest(*inner, tail)),
        Selection::Union(a, b) => dsl::union(nest(*a, tail.clone()), nest(*b, tail)),
        Selection::Intersection(a, b) => dsl::intersection(nest(*a, tail.clone()), nest(*b, tail)),
        Selection::True => tail,
        Selection::False => dsl::false_(),
        other => panic!("unexpected selection variant in chain: {:?}", other),
    }
}

fn intersection(input: &str) -> IResult<&str, Selection> {
    map(separated_list1(char('&'), chain), |items| {
        items.into_iter().reduce(dsl::intersection).unwrap()
    })
    .parse(input)
}

pub fn expression(input: &str) -> IResult<&str, Selection> {
    map(separated_list1(char('|'), intersection), |items| {
        items.into_iter().reduce(dsl::union).unwrap()
    })
    .parse(input)
}

/// Parses a selection expression from a string, ignoring all
/// whitespace.
///
/// # Arguments
///
/// * `input` - A string slice containing the selection expression to
///   parse.
///
/// # Returns
///
/// * `Ok(Selection)` if parsing succeeds
/// * `Err(anyhow::Error)` with a detailed error message if parsing
///   fails
pub fn parse(input: &str) -> anyhow::Result<Selection> {
    use nom::combinator::all_consuming;

    let input: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    let (_, selection) = all_consuming(expression)
        .parse(&input)
        .map_err(|err| anyhow::anyhow!("Failed to parse selection: {err:?} (input: {input:?})"))?;
    Ok(selection)
}

#[cfg(test)]
mod tests {
    use crate::selection::Selection;
    use crate::shape;

    // Parse an input string to a selection.
    fn parse(input: &str) -> Selection {
        super::parse(input).unwrap()
    }

    #[macro_export]
    macro_rules! assert_parses_to {
        ($input:expr_2021, $expected:expr_2021) => {{
            let actual = $crate::selection::parse::tests::parse($input);
            $crate::assert_structurally_eq!($expected, actual);
        }};
    }

    #[test]
    fn test_selection_11() {
        use crate::selection::dsl::*;

        assert_parses_to!("*", all(true_()));
        assert_parses_to!("*,*", all(all(true_())));
        assert_parses_to!("*,*,*", all(all(all(true_()))));

        assert_parses_to!("4:8", range(shape::Range(4, Some(8), 1), true_()));
        assert_parses_to!("4:", range(shape::Range(4, None, 1), true_()));
        assert_parses_to!("4", range(shape::Range(4, Some(5), 1), true_()));
        assert_parses_to!(":", range(shape::Range(0, None, 1), true_()));

        assert_parses_to!("0,0,0", range(0, range(0, range(0, true_()))));
        assert_parses_to!("1,1,1", range(1, range(1, range(1, true_()))));
        assert_parses_to!("*,0", all(range(0, true_())));
        assert_parses_to!("*,0,*", all(range(0, all(true_()))));
        assert_parses_to!("*,0,*", all(range(0, all(true_()))));
        assert_parses_to!("*,*,4:", all(all(range(4.., true_()))));
        assert_parses_to!(
            "*,*,1::2",
            all(all(range(shape::Range(1, None, 2), true_())))
        );

        assert_parses_to!(
            "*,*,:4|*,*,4:",
            union(
                all(all(range(0..4, true_()))),
                all(all(range(shape::Range(4, None, 1), true_())))
            )
        );
        assert_parses_to!(
            "*,0,:4|*,1,4:8",
            union(
                all(range(0, range(0..4, true_()))),
                all(range(1, range(4..8, true_()))),
            )
        );
        assert_parses_to!(
            "*,*,(:2|6:)",
            all(all(union(
                range(0..2, true_()),
                range(shape::Range(6, None, 1), true_()),
            )))
        );
        assert_parses_to!(
            "*,*,(1:4:2|5::2)",
            all(all(union(
                range(shape::Range(1, Some(4), 2), true_()),
                range(shape::Range(5, None, 2), true_()),
            )))
        );
        assert_parses_to!("*&*", intersection(all(true_()), all(true_())));
        assert_parses_to!(
            "*&*,*,4:8",
            intersection(all(true_()), all(all(range(4..8, true_()))))
        );
        assert_parses_to!(
            "*,*,0:5&*,*,4:8",
            intersection(
                all(all(range(0..5, true_()))),
                all(all(range(4..8, true_()))),
            )
        );

        assert_parses_to!("?,?,?", any(any(any(true_()))));
        assert_parses_to!("0,?,:4", range(0, any(range(0..4, true_()))));
        assert_parses_to!("0,?", range(0, any(true_())));
        assert_parses_to!("?", any(true_()));
        assert_parses_to!(
            "0,0,?|0,0,?",
            union(
                range(0, range(0, any(true_()))),
                range(0, range(0, any(true_()))),
            )
        );
        assert_parses_to!(
            "*,*,1:4|5:6",
            union(all(all(range(1..4, true_()))), range(5..6, true_()))
        );
        assert_parses_to!(
            "*,*,(1:4|5:6)",
            all(all(union(range(1..4, true_()), range(5..6, true_()))))
        );
        assert_parses_to!(
            "*,(1:4|5:6),*",
            all(union(
                range(shape::Range(1, Some(4), 1), all(true_())),
                range(shape::Range(5, Some(6), 1), all(true_())),
            ))
        );
        assert_parses_to!(
            "(*,*,*)&(*,*,*)",
            intersection(all(all(all(true_()))), all(all(all(true_()))))
        );
        assert_parses_to!(
            "(0,*,*)&(0,(1|3),*)",
            intersection(
                range(0, all(all(true_()))),
                range(0, union(range(1, all(true_())), range(3, all(true_())))),
            )
        );
        assert_parses_to!(
            "(*,*,(:2|6:))&(*,*,4:)",
            intersection(
                all(all(union(
                    range(0..2, true_()),
                    range(shape::Range(6, None, 1), true_()),
                ))),
                all(all(range(shape::Range(4, None, 1), true_()))),
            )
        );
        assert_parses_to!("((1:4),2)", range(1..4, range(2, true_())));
    }

    #[test]
    fn test_12() {
        use crate::dsl::all;
        use crate::dsl::true_;

        assert_parses_to!("*,*,*,*,*,*", all(all(all(all(all(all(true_())))))));
    }
}
