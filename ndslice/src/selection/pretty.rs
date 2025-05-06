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

pub(crate) struct SelectionPretty(String);

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

pub(crate) fn pretty(selection: &Selection) -> SelectionPretty {
    match selection {
        Selection::False => SelectionPretty::false_(),
        Selection::True => SelectionPretty::true_(),
        Selection::All(inner) => SelectionPretty::all(pretty(inner)),
        Selection::First(inner) => SelectionPretty::first(pretty(inner)),
        Selection::Range(r, inner) => SelectionPretty::range(r.clone(), pretty(inner)),
        Selection::Label(labels, inner) => SelectionPretty::label(labels.clone(), pretty(inner)),
        Selection::Any(inner) => SelectionPretty::any(pretty(inner)),
        Selection::Intersection(a, b) => SelectionPretty::intersection(pretty(a), pretty(b)),
        Selection::Union(a, b) => SelectionPretty::union(pretty(a), pretty(b)),
    }
}

/// Converts a [`Selection`] into its compact surface syntax
/// representation.
///
/// For example, the selection: `Union(All(All(Range(1..4, True))),
/// Range(5..6, True))` is formatted as `"*,*,1:4|5:6"`.
pub fn to_compact_syntax(selection: &Selection) -> String {
    match selection {
        Selection::False => "false".to_string(),
        Selection::True => "".to_string(),
        Selection::All(inner) => {
            let inner_str = to_compact_syntax(inner);
            if inner_str.is_empty() {
                "*".to_string()
            } else {
                format!("*,{}", inner_str)
            }
        }
        Selection::First(inner) => {
            let inner_str = to_compact_syntax(inner);
            if inner_str.is_empty() {
                "0".to_string()
            } else {
                format!("0,{}", inner_str)
            }
        }
        Selection::Range(r, inner) => {
            let range_str = match (r.0, r.1, r.2) {
                (start, Some(end), 1) => format!("{}:{}", start, end),
                (start, Some(end), step) => format!("{}:{}:{}", start, end, step),
                (start, None, step) => format!("{}::{}", start, step),
            };
            let inner_str = to_compact_syntax(inner);
            if inner_str.is_empty() {
                range_str
            } else {
                format!("{},{}", range_str, inner_str)
            }
        }
        Selection::Label(labels, inner) => {
            let inner_str = to_compact_syntax(inner);
            if inner_str.is_empty() {
                format!(
                    "[{}]",
                    labels
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
            } else {
                format!(
                    "[{}],{}",
                    labels
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    inner_str
                )
            }
        }
        Selection::Any(inner) => {
            let inner_str = to_compact_syntax(inner);
            if inner_str.is_empty() {
                "?".to_string()
            } else {
                format!("?,{}", inner_str)
            }
        }
        Selection::Intersection(a, b) => {
            format!("({}&{})", to_compact_syntax(a), to_compact_syntax(b))
        }
        Selection::Union(a, b) => format!("({}|{})", to_compact_syntax(a), to_compact_syntax(b)),
    }
}

#[cfg(test)]
mod tests {
    use crate::selection::Selection;
    use crate::shape;

    // Parse an input string to a selection.
    fn parse(input: &str) -> Selection {
        use nom::combinator::all_consuming;

        use crate::selection::parse::expression;

        let (_, selection) = all_consuming(expression)(input).unwrap();
        selection
    }

    macro_rules! assert_round_trip {
        ($selection:expr) => {{
            let compact = to_compact_syntax($selection);
            let parsed = parse(&compact);
            assert!(
                structurally_equal($selection, &parsed),
                "input: {} \n compact: {}\n parsed: {}",
                $selection,
                compact,
                parsed
            );
        }};
    }

    #[test]
    fn test_selection_to_compact_and_back() {
        use super::to_compact_syntax;
        use crate::selection::dsl::*;
        use crate::selection::structurally_equal;

        assert_round_trip!(&all(true_()));
        assert_round_trip!(&all(all(true_())));
        assert_round_trip!(&all(all(all(true_()))));

        assert_round_trip!(&range(shape::Range(4, Some(8), 1), true_()));
        assert_round_trip!(&range(shape::Range(4, None, 1), true_()));
        assert_round_trip!(&range(shape::Range(4, Some(5), 1), true_()));
        assert_round_trip!(&range(shape::Range(0, None, 1), true_()));

        assert_round_trip!(&range(0, range(0, range(0, true_()))));
        assert_round_trip!(&range(1, range(1, range(1, true_()))));
        assert_round_trip!(&all(range(0, true_())));
        assert_round_trip!(&all(range(0, all(true_()))));
        assert_round_trip!(&all(all(range(4.., true_()))));
        assert_round_trip!(&all(all(range(shape::Range(1, None, 2), true_()))));

        assert_round_trip!(&union(
            all(all(range(0..4, true_()))),
            all(all(range(shape::Range(4, None, 1), true_()))),
        ));
        assert_round_trip!(&union(
            all(range(0, range(0..4, true_()))),
            all(range(1, range(4..8, true_()))),
        ));
        assert_round_trip!(&union(
            all(all(range(0..2, true_()))),
            all(all(range(shape::Range(6, None, 1), true_()))),
        ));
        assert_round_trip!(&union(
            all(all(range(shape::Range(1, Some(4), 2), true_()))),
            all(all(range(shape::Range(5, None, 2), true_()))),
        ));
        assert_round_trip!(&intersection(all(true_()), all(true_())));
        assert_round_trip!(&intersection(all(true_()), all(all(range(4..8, true_())))));
        assert_round_trip!(&intersection(
            all(all(range(0..5, true_()))),
            all(all(range(4..8, true_()))),
        ));

        assert_round_trip!(&any(any(any(true_()))));
        assert_round_trip!(&range(0, any(range(0..4, true_()))));
        assert_round_trip!(&range(0, any(true_())));
        assert_round_trip!(&any(true_()));
        assert_round_trip!(&union(
            range(0, range(0, any(true_()))),
            range(0, range(0, any(true_()))),
        ));
        assert_round_trip!(&union(all(all(range(1..4, true_()))), range(5..6, true_())));
        assert_round_trip!(&all(all(union(range(1..4, true_()), range(5..6, true_())))));
        assert_round_trip!(&all(union(
            range(shape::Range(1, Some(4), 1), all(true_())),
            range(shape::Range(5, Some(6), 1), all(true_())),
        )));
        assert_round_trip!(&intersection(
            all(all(all(true_()))),
            all(all(all(true_()))),
        ));
        assert_round_trip!(&intersection(
            range(0, all(all(true_()))),
            range(0, union(range(1, all(true_())), range(3, all(true_())))),
        ));
        assert_round_trip!(&intersection(
            all(all(union(
                range(0..2, true_()),
                range(shape::Range(6, None, 1), true_()),
            ))),
            all(all(range(shape::Range(4, None, 1), true_()))),
        ));
        assert_round_trip!(&range(1..4, range(2, true_())));
    }
}
