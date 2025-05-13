//! Pretty-printing utilities for selection expressions.
//!
//! This module defines `SelectionSYM` implementations that render
//! selection expressions in human-readable or structured forms.
//!
//! The `Display` implementation for [`Selection`] delegates to this
//! module and uses the `SelectionPretty` representation.
use crate::Selection;
use crate::selection::LabelKey;
use crate::selection::SelectionSYM;
use crate::shape;

/// A structured pretty-printer that renders [`Selection`] expressions
/// in DSL constructor form.
///
/// This type implements [`SelectionSYM`] and emits expressions like
/// `all(range(0..4, true_()))`, which mirror the Rust-based DSL used
/// to construct `Selection` values programmatically.
///
/// Internally used by [`Selection::fmt`] to support human-readable
/// display of selection expressions in their canonical constructor
/// form.
///
/// Use [`Selection::fold`] or the [`pretty`] helper to produce a
/// `SelectionPretty`:
///
/// ```rust
/// use ndslice::selection::dsl::*;
/// use ndslice::selection::pretty::pretty;
/// let expr = all(range(0..4, true_()));
/// println!("{}", pretty(&expr)); // prints: all(range(0..4, true_()))
/// ```
pub struct SelectionPretty(String);

impl std::fmt::Display for SelectionPretty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl SelectionSYM for SelectionPretty {
    fn false_() -> Self {
        SelectionPretty("false_()".into())
    }
    fn true_() -> Self {
        SelectionPretty("true_()".into())
    }
    fn all(s: Self) -> Self {
        SelectionPretty(format!("all({})", s.0))
    }
    fn first(s: Self) -> Self {
        SelectionPretty(format!("first({})", s.0))
    }
    fn range<R: Into<shape::Range>>(range: R, s: Self) -> Self {
        let r = range.into();
        SelectionPretty(format!("range({}, {})", r, s.0))
    }
    fn label<L: Into<LabelKey>>(labels: Vec<L>, s: Self) -> Self {
        let labels_str = labels
            .into_iter()
            .map(|l| l.into().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        SelectionPretty(format!("label([{}], {})", labels_str, s.0))
    }
    fn any(s: Self) -> Self {
        SelectionPretty(format!("any({})", s.0))
    }
    fn intersection(a: Self, b: Self) -> Self {
        SelectionPretty(format!("intersection({}, {})", a.0, b.0))
    }
    fn union(a: Self, b: Self) -> Self {
        SelectionPretty(format!("union({}, {})", a.0, b.0))
    }
}

/// Renders a [`Selection`] as a structured DSL expression.
///
/// This function folds the input selection using the [`SelectionSYM`]
/// interface, producing a [`SelectionPretty`] — a human-readable
/// representation in canonical constructor form (e.g.,
/// `all(range(0..4, true_()))`).
///
/// Useful for debugging, diagnostics, and implementing `Display`.
///
/// # Example
///
/// ```rust
/// use ndslice::selection::dsl::*;
/// use ndslice::selection::pretty::pretty;
/// let expr = all(range(0..4, true_()));
/// println!("{}", pretty(&expr)); // prints: all(range(0..4, true_()))
/// ```
pub fn pretty(selection: &Selection) -> SelectionPretty {
    selection.fold()
}

/// A structured formatter that renders [`Selection`] expressions in
/// compact surface syntax.
///
/// This type implements [`SelectionSYM`] and emits selection
/// expressions using the same textual format accepted by both:
///
/// - The [`sel!`] macro (defined in the `hypermesh_macros` crate)
/// - The string-based [`parse`](crate::selection::parse::parse)
///   function
///
/// Examples of this syntax include:
///
/// - `*`
/// - `0, 1..4, *`
/// - `["A100"]?`
/// - `(0, (0 | 2), *) & (0, *, *)`
///   — intersection of two 3D expressions; simplifies to just `0, (0
///     | 2), *` since the second operand is a superset
///
/// Used internally by the [`compact`] helper and [`Selection::fmt`]
/// to produce concise, user-facing representations of selection
/// expressions.
///
/// # Example
///
/// ```rust
/// use ndslice::selection::dsl::*;
/// use ndslice::selection::pretty::compact;
/// let expr = all(range(0..4, true_()));
/// println!("{}", compact(&expr)); // prints: (*, 0..4)
/// ```
pub struct SelectionCompact(String);

impl std::fmt::Display for SelectionCompact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl SelectionSYM for SelectionCompact {
    fn true_() -> Self {
        SelectionCompact("".into())
    }

    fn false_() -> Self {
        panic!("SelectionCompact: `false` has no compact surface syntax")
    }

    fn first(_: Self) -> Self {
        panic!("SelectionCompact: `first` has no compact surface syntax")
    }

    fn all(s: Self) -> Self {
        if s.0.is_empty() {
            SelectionCompact("*".into())
        } else {
            SelectionCompact(format!("*,{}", s.0))
        }
    }

    fn range<R: Into<shape::Range>>(range: R, s: Self) -> Self {
        let r = range.into();
        let range_str = match (r.0, r.1, r.2) {
            (start, Some(end), 1) => format!("{}:{}", start, end),
            (start, Some(end), step) => format!("{}:{}:{}", start, end, step),
            (start, None, step) => format!("{}::{}", start, step),
        };
        if s.0.is_empty() {
            SelectionCompact(range_str)
        } else {
            SelectionCompact(format!("{},{}", range_str, s.0))
        }
    }

    fn label<L: Into<LabelKey>>(labels: Vec<L>, s: Self) -> Self {
        let label_str = labels
            .into_iter()
            .map(|l| l.into().to_string())
            .collect::<Vec<_>>()
            .join(",");
        if s.0.is_empty() {
            panic!("SelectionCompact: label requires a combinator like '*', '?', or a range");
        }
        SelectionCompact(format!("[{}]{}", label_str, s.0))
    }

    fn any(s: Self) -> Self {
        if s.0.is_empty() {
            SelectionCompact("?".into())
        } else {
            SelectionCompact(format!("?,{}", s.0))
        }
    }

    fn intersection(a: Self, b: Self) -> Self {
        SelectionCompact(format!("({}&{})", a.0, b.0))
    }

    fn union(a: Self, b: Self) -> Self {
        SelectionCompact(format!("({}|{})", a.0, b.0))
    }
}

/// Returns a [`SelectionCompact`] rendering of the given
/// [`Selection`] expression.
///
/// This produces a string in the surface syntax used by the `sel!`
/// macro and the [`parse`] function, such as:
///
/// ```
/// use ndslice::selection::Selection;
/// use ndslice::selection::dsl::*;
/// use ndslice::selection::pretty::compact;
///
/// let sel = all(range(0..4, true_()));
/// assert_eq!(compact(&sel).to_string(), "*,0:4");
/// ```
/// [`parse`]: crate::selection::parse::parse
pub fn compact(selection: &Selection) -> SelectionCompact {
    selection.fold()
}

#[cfg(test)]
mod tests {
    use crate::assert_round_trip;
    use crate::selection::Selection;
    use crate::shape;

    #[test]
    fn test_selection_to_compact_and_back() {
        use crate::selection::dsl::*;

        assert_round_trip!(all(true_()));
        assert_round_trip!(all(all(true_())));
        assert_round_trip!(all(all(all(true_()))));

        assert_round_trip!(range(shape::Range(4, Some(8), 1), true_()));
        assert_round_trip!(range(shape::Range(4, None, 1), true_()));
        assert_round_trip!(range(shape::Range(4, Some(5), 1), true_()));
        assert_round_trip!(range(shape::Range(0, None, 1), true_()));

        assert_round_trip!(range(0, range(0, range(0, true_()))));
        assert_round_trip!(range(1, range(1, range(1, true_()))));
        assert_round_trip!(all(range(0, true_())));
        assert_round_trip!(all(range(0, all(true_()))));

        assert_round_trip!(all(all(range(4.., true_()))));
        assert_round_trip!(all(all(range(shape::Range(1, None, 2), true_()))));

        assert_round_trip!(union(
            all(all(range(0..4, true_()))),
            all(all(range(shape::Range(4, None, 1), true_()))),
        ));
        assert_round_trip!(union(
            all(range(0, range(0..4, true_()))),
            all(range(1, range(4..8, true_()))),
        ));
        assert_round_trip!(union(
            all(all(range(0..2, true_()))),
            all(all(range(shape::Range(6, None, 1), true_()))),
        ));
        assert_round_trip!(union(
            all(all(range(shape::Range(1, Some(4), 2), true_()))),
            all(all(range(shape::Range(5, None, 2), true_()))),
        ));
        assert_round_trip!(intersection(all(true_()), all(true_())));
        assert_round_trip!(intersection(all(true_()), all(all(range(4..8, true_())))));
        assert_round_trip!(intersection(
            all(all(range(0..5, true_()))),
            all(all(range(4..8, true_()))),
        ));

        assert_round_trip!(any(any(any(true_()))));
        assert_round_trip!(range(0, any(range(0..4, true_()))));
        assert_round_trip!(range(0, any(true_())));
        assert_round_trip!(any(true_()));
        assert_round_trip!(union(
            range(0, range(0, any(true_()))),
            range(0, range(0, any(true_()))),
        ));
        assert_round_trip!(union(all(all(range(1..4, true_()))), range(5..6, true_())));
        assert_round_trip!(all(all(union(range(1..4, true_()), range(5..6, true_())))));
        assert_round_trip!(all(union(
            range(shape::Range(1, Some(4), 1), all(true_())),
            range(shape::Range(5, Some(6), 1), all(true_())),
        )));
        assert_round_trip!(intersection(all(all(all(true_()))), all(all(all(true_()))),));
        assert_round_trip!(intersection(
            range(0, all(all(true_()))),
            range(0, union(range(1, all(true_())), range(3, all(true_())))),
        ));
        assert_round_trip!(intersection(
            all(all(union(
                range(0..2, true_()),
                range(shape::Range(6, None, 1), true_()),
            ))),
            all(all(range(shape::Range(4, None, 1), true_()))),
        ));
        assert_round_trip!(range(1..4, range(2, true_())));

        // TODO(SF, 2025-05-05): There isn't parse support for `label`
        // yet.
        // assert_round_trip!(label(vec!["A100"], all(true_())));
    }
}
