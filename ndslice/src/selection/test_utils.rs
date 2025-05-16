use crate::selection::Selection;

/// Parse an input string to a selection.
pub fn parse(input: &str) -> Selection {
    use nom::combinator::all_consuming;

    use crate::selection::parse::expression;

    let (_, selection) = all_consuming(expression)(input).unwrap();
    selection
}

#[macro_export]
macro_rules! assert_structurally_eq {
    ($expected:expr, $actual:expr) => {{
        let expected = &$expected;
        let actual = &$actual;
        assert!(
            $crate::selection::structurally_equal(expected, actual),
            "Selections do not match.\nExpected: {:#?}\nActual:   {:#?}",
            expected,
            actual,
        );
    }};
}

#[macro_export]
macro_rules! assert_round_trip {
    ($selection:expr) => {{
        let selection: Selection = $selection; // take ownership
        // Convert `Selection` to representation as compact
        // syntax.
        let compact = $crate::selection::pretty::compact(&selection).to_string();
        // Parse a `Selection` from the compact syntax
        // representation.
        let parsed = $crate::selection::test_utils::parse(&compact);
        // Check that the input and parsed `Selection`s are
        // structurally equivalent.
        assert!(
            $crate::selection::structurally_equal(&selection, &parsed),
            "input: {} \n compact: {}\n parsed: {}",
            selection,
            compact,
            parsed
        );
    }};
}
