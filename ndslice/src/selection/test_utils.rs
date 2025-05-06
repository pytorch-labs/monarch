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
