//! This crate provides hyperactor's mesh abstractions.

#![feature(assert_matches)]

pub mod actor_mesh;
pub mod alloc;
mod assign;
pub mod bootstrap;
pub mod comm;
pub mod mesh;
pub mod proc_mesh;
pub mod reference;
pub mod shortuuid;
mod test_utils;

pub use actor_mesh::ActorMesh;
pub use actor_mesh::SlicedActorMesh;
pub use bootstrap::bootstrap;
pub use bootstrap::bootstrap_or_die;
pub use comm::CommActor;
pub use hyperactor_mesh_macros::sel;
pub use mesh::Mesh;
pub use ndslice::selection;
pub use ndslice::shape;
pub use proc_mesh::ProcMesh;
pub use proc_mesh::SlicedProcMesh;

#[cfg(test)]
mod tests {

    #[test]
    fn basic() {
        use ndslice::selection::dsl;
        use ndslice::selection::structurally_equal;

        let actual = sel!(*, 0:4, *);
        let expected = dsl::all(dsl::range(
            ndslice::shape::Range(0, Some(4), 1),
            dsl::all(dsl::true_()),
        ));
        assert!(structurally_equal(&actual, &expected));
    }

    #[cfg(FALSE)]
    #[test]
    fn shouldnt_compile() {
        let _ = sel!(foobar);
    }
    // error: sel! parse failed: unexpected token: Ident { sym: foobar, span: #0 bytes(605..611) }
    //   --> fbcode/monarch/hyperactor_mesh_macros/tests/basic.rs:19:13
    //    |
    // 19 |     let _ = sel!(foobar);
    //    |             ^^^^^^^^^^^^ in this macro invocation
    //   --> fbcode/monarch/hyperactor_mesh_macros/src/lib.rs:12:1
    //    |
    //    = note: in this expansion of `sel!`

    use hyperactor_mesh_macros::sel;
    use ndslice::selection::Selection;
    use ndslice::selection::structurally_equal;

    fn parse(input: &str) -> Selection {
        use ndslice::selection::parse::expression;
        use nom::combinator::all_consuming;

        let (_, selection) = all_consuming(expression)(input).unwrap();
        selection
    }

    // A copy from `hyperactor_mesh::test_utils`. Replicate for now.
    #[macro_export]
    macro_rules! assert_structurally_eq {
        ($expected:expr, $actual:expr) => {{
            let expected = &$expected;
            let actual = &$actual;
            assert!(
                structurally_equal(expected, actual),
                "Selections do not match.\nExpected: {:#?}\nActual:   {:#?}",
                expected,
                actual,
            );
        }};
    }

    #[test]
    fn token_parser() {
        use ndslice::selection::dsl::*;
        use ndslice::shape;

        assert_structurally_eq!(all(true_()), sel!(*));
        assert_structurally_eq!(range(3, true_()), sel!(3));
        assert_structurally_eq!(range(1..4, true_()), sel!(1:4));
        assert_structurally_eq!(all(range(1..4, true_())), sel!(*, 1:4));
        assert_structurally_eq!(range(shape::Range(0, None, 1), true_()), sel!(:));
        assert_structurally_eq!(any(true_()), sel!(?));
        assert_structurally_eq!(any(range(1..4, all(true_()))), sel!(?, 1:4, *));
        assert_structurally_eq!(union(range(0, true_()), range(1, true_())), sel!(0 | 1));
        assert_structurally_eq!(
            intersection(range(0..4, true_()), range(2..6, true_())),
            sel!(0:4 & 2:6)
        );
        assert_structurally_eq!(range(shape::Range(0, None, 1), true_()), sel!(:));
        assert_structurally_eq!(all(true_()), sel!(*));
        assert_structurally_eq!(any(true_()), sel!(?));
        assert_structurally_eq!(all(all(all(true_()))), sel!(*, *, *));
        assert_structurally_eq!(intersection(all(true_()), all(true_())), sel!(* & *));
        assert_structurally_eq!(
            all(all(union(
                range(0..2, true_()),
                range(shape::Range(6, None, 1), true_())
            ))),
            sel!(*, *, (:2|6:))
        );
        assert_structurally_eq!(
            all(all(range(shape::Range(1, None, 2), true_()))),
            sel!(*, *, 1::2)
        );
        assert_structurally_eq!(parse("0,?,:4"), sel!(0, ?, :4));
        assert_structurally_eq!(range(shape::Range(1, Some(4), 2), true_()), sel!(1:4:2));
        assert_structurally_eq!(range(shape::Range(0, None, 2), true_()), sel!(::2));
        assert_structurally_eq!(
            union(range(0..4, true_()), range(4..8, true_())),
            sel!(0:4 | 4:8)
        );
        assert_structurally_eq!(
            intersection(range(0..4, true_()), range(2..6, true_())),
            sel!(0:4 & 2:6)
        );
        assert_structurally_eq!(
            all(union(range(1..4, all(true_())), range(5..6, all(true_())))),
            sel!(*, (1:4 | 5:6), *)
        );
        assert_structurally_eq!(
            range(
                0,
                intersection(
                    range(1..4, range(7, true_())),
                    range(2..5, range(7, true_()))
                )
            ),
            sel!(0, (1:4 & 2:5), 7)
        );
        assert_structurally_eq!(
            all(all(union(
                union(range(0..2, true_()), range(4..6, true_())),
                range(shape::Range(6, None, 1), true_())
            ))),
            sel!(*, *, (:2 | 4:6 | 6:))
        );
        assert_structurally_eq!(intersection(all(true_()), all(true_())), sel!(* & *));
        assert_structurally_eq!(union(all(true_()), all(true_())), sel!(* | *));
        assert_structurally_eq!(
            intersection(
                range(0..2, true_()),
                union(range(1, true_()), range(2, true_()))
            ),
            sel!(0:2 & (1 | 2))
        );
        assert_structurally_eq!(
            all(all(intersection(
                range(1..2, true_()),
                range(2..3, true_())
            ))),
            sel!(*,*,(1:2&2:3))
        );
        assert_structurally_eq!(
            intersection(all(all(all(true_()))), all(all(all(true_())))),
            sel!((*,*,*) & (*,*,*))
        );
        assert_structurally_eq!(
            intersection(
                range(0, all(all(true_()))),
                range(0, union(range(1, all(true_())), range(3, all(true_()))))
            ),
            sel!((0, *, *) & (0, (1 | 3), *))
        );
        assert_structurally_eq!(
            intersection(
                range(0, all(all(true_()))),
                range(
                    0,
                    union(
                        range(1, range(2..5, true_())),
                        range(3, range(2..5, true_()))
                    )
                )
            ),
            sel!((0, *, *) & (0, (1 | 3), 2:5))
        );
        assert_structurally_eq!(all(true_()), sel!((*)));
        assert_structurally_eq!(range(1..4, range(2, true_())), sel!(((1:4), 2)));
        assert_structurally_eq!(sel!(1:4 & 5:6 | 7:8), sel!((1:4 & 5:6) | 7:8));
        assert_structurally_eq!(
            union(
                intersection(all(all(true_())), all(all(true_()))),
                all(all(true_()))
            ),
            sel!((*,*) & (*,*) | (*,*))
        );
        assert_structurally_eq!(all(true_()), sel!(*));
        assert_structurally_eq!(sel!(((1:4))), sel!(1:4));
        assert_structurally_eq!(sel!(*, (*)), sel!(*, *));
        assert_structurally_eq!(
            intersection(
                range(0, range(1..4, true_())),
                range(0, union(range(2, all(true_())), range(3, all(true_()))))
            ),
            sel!((0,1:4)&(0,(2|3),*))
        );

        //assert_structurally_eq!(true_(), sel!(foo)); // sel! macro: parse error: Parsing Error: Error { input: "foo", code: Tag }

        assert_structurally_eq!(
            sel!(0 & (0, (1|3), *)),
            intersection(
                range(0, true_()),
                range(0, union(range(1, all(true_())), range(3, all(true_()))))
            )
        );
        assert_structurally_eq!(
            sel!(0 & (0, (3|1), *)),
            intersection(
                range(0, true_()),
                range(0, union(range(3, all(true_())), range(1, all(true_()))))
            )
        );
        assert_structurally_eq!(
            sel!((*, *, *) & (*, *, (2 | 4))),
            intersection(
                all(all(all(true_()))),
                all(all(union(range(2, true_()), range(4, true_()))))
            )
        );
        assert_structurally_eq!(
            sel!((*, *, *) & (*, *, (4 | 2))),
            intersection(
                all(all(all(true_()))),
                all(all(union(range(4, true_()), range(2, true_()))))
            )
        );
        assert_structurally_eq!(
            sel!((*, (1|2)) & (*, (2|1))),
            intersection(
                all(union(range(1, true_()), range(2, true_()))),
                all(union(range(2, true_()), range(1, true_())))
            )
        );
        assert_structurally_eq!(
            sel!((*, *, *) & *),
            intersection(all(all(all(true_()))), all(true_()))
        );
        assert_structurally_eq!(
            sel!(* & (*, *, *)),
            intersection(all(true_()), all(all(all(true_()))))
        );

        assert_structurally_eq!(
            sel!( (*, *, *) & ((*, *, *) & (*, *, *)) ),
            intersection(
                all(all(all(true_()))),
                intersection(all(all(all(true_()))), all(all(all(true_()))))
            )
        );
        assert_structurally_eq!(
            sel!((1, *, *) | (0 & (0, 3, *))),
            union(
                range(1, all(all(true_()))),
                intersection(range(0, true_()), range(0, range(3, all(true_()))))
            )
        );
        assert_structurally_eq!(
            sel!(((0, *)| (1, *)) & ((1, *) | (0, *))),
            intersection(
                union(range(0, all(true_())), range(1, all(true_()))),
                union(range(1, all(true_())), range(0, all(true_())))
            )
        );
        assert_structurally_eq!(sel!(*, 8:8), all(range(8..8, true_())));
        assert_structurally_eq!(
            sel!((*, 1) & (*, 8 : 8)),
            intersection(all(range(1..2, true_())), all(range(8..8, true_())))
        );
        assert_structurally_eq!(
            sel!((*, 8 : 8) | (*, 1)),
            union(all(range(8..8, true_())), all(range(1..2, true_())))
        );
        assert_structurally_eq!(
            sel!((*, 1) | (*, 2:8)),
            union(all(range(1..2, true_())), all(range(2..8, true_())))
        );
        assert_structurally_eq!(
            sel!((*, *, *) & (*, *, 2:8)),
            intersection(all(all(all(true_()))), all(all(range(2..8, true_()))))
        );
    }
}
