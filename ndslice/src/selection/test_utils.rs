/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::ControlFlow;

use nom::Parser as _;

use crate::Slice;
use crate::selection::Selection;
use crate::selection::routing::RoutingAction;
use crate::selection::routing::RoutingFrame;
use crate::selection::routing::RoutingFrameKey;
use crate::selection::routing::RoutingStep;
use crate::selection::routing::resolve_routing;

/// Parse an input string to a selection.
pub fn parse(input: &str) -> Selection {
    use nom::combinator::all_consuming;

    use crate::selection::parse::expression;

    let (_, selection) = all_consuming(expression).parse(input).unwrap();
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

/// Determines whether routing frame deduplication is enabled.
///
/// By default, deduplication is enabled to reduce redundant routing
/// steps and improve performance. However, correctness must not
/// depend on deduplication.
///
/// This behavior can be disabled for debugging or testing purposes by
/// setting the environment variable:
/// ```ignore
/// HYPERACTOR_SELECTION_DISABLE_ROUTING_FRAME_DEDUPLICATION = 1
/// ```
/// When disabled, all routing steps—including structurally redundant
/// ones—will be visited, potentially causing re-entry into previously
/// seen coordinates. This switch helps validate that correctness
/// derives from the routing algebra itself—not from memoization or
/// key-based filtering.
fn allow_frame_dedup() -> bool {
    // Default: true (deduplication via memoization and normalization
    // is enabled unless explicitly disabled).
    std::env::var("HYPERACTOR_SELECTION_DISABLE_ROUTING_FRAME_DEDUPLICATION")
        .map_or(true, |val| val != "1")
}

// == Testing (`collect_routed_paths` mesh simulation) ===

/// Message type used in the `collect_routed_paths` mesh routing
/// simulation.
///
/// Each message tracks the current routing state (`frame`) and
/// the full path (`path`) taken from the origin to the current
/// node, represented as a list of flat indices.
///
/// As the message is forwarded, `path` is extended. This allows
/// complete routing paths to be observed at the point of
/// delivery.
pub struct RoutedMessage<T> {
    pub path: Vec<usize>,
    pub frame: RoutingFrame,
    pub _payload: std::marker::PhantomData<T>,
}

impl<T> RoutedMessage<T> {
    pub fn new(path: Vec<usize>, frame: RoutingFrame) -> Self {
        Self {
            path,
            frame,
            _payload: std::marker::PhantomData,
        }
    }
}

#[derive(Default)]
pub struct RoutedPathTree {
    // Map from rank → delivery path (flat indices).
    pub delivered: HashMap<usize, Vec<usize>>,

    // Map from rank → set of direct predecessor ranks (flat
    // indices).
    pub predecessors: HashMap<usize, HashSet<usize>>,
}

/// Simulates routing from the origin through a slice using a
/// `Selection`, collecting all delivery destinations **along with
/// their routing paths**.
//
/// Each returned entry is a tuple `(dst, path)`, where `dst` is the
/// flat index of a delivery node, and `path` is the list of flat
/// indices representing the route taken from the origin to that node.
//
/// Routing begins at `[0, 0, ..., 0]` and proceeds
/// dimension-by-dimension. At each hop, `next_steps` determines the
/// next set of forwarding frames.
//
/// A node is considered a delivery target if:
/// - its `selection` is `Selection::True`, and
/// - it is at the final dimension.
//
///   Useful in tests for verifying full routing paths and ensuring
///   correctness.
pub fn collect_routed_paths(selection: &Selection, slice: &Slice) -> RoutedPathTree {
    use std::collections::VecDeque;

    let mut pending = VecDeque::new();
    let mut delivered = HashMap::new();
    let mut seen = HashSet::new();
    let mut predecessors: HashMap<usize, HashSet<usize>> = HashMap::new();

    let root_frame = RoutingFrame::root(selection.clone(), slice.clone());
    let origin = slice.location(&root_frame.here).unwrap();
    pending.push_back(RoutedMessage::<()>::new(vec![origin], root_frame));

    while let Some(RoutedMessage { path, frame, .. }) = pending.pop_front() {
        let mut visitor = |step: RoutingStep| {
            if let RoutingStep::Forward(next_frame) = step {
                let key = RoutingFrameKey::new(&next_frame);
                let should_insert = if allow_frame_dedup() {
                    seen.insert(key) // true → not seen before
                } else {
                    true // unconditionally insert
                };
                if should_insert {
                    let next_rank = slice.location(&next_frame.here).unwrap();
                    let parent_rank = *path.last().unwrap();
                    predecessors
                        .entry(next_rank)
                        .or_default()
                        .insert(parent_rank);

                    let mut next_path = path.clone();
                    next_path.push(next_rank);

                    match next_frame.action() {
                        RoutingAction::Deliver => {
                            if let Some(previous) = delivered.insert(next_rank, next_path.clone()) {
                                panic!(
                                    "over-delivery detected: node {} delivered twice\nfirst: {:?}\nsecond: {:?}",
                                    next_rank, previous, next_path
                                );
                            }
                        }
                        RoutingAction::Forward => {
                            pending.push_back(RoutedMessage::new(next_path, next_frame));
                        }
                    }
                }
            }
            ControlFlow::Continue(())
        };

        let _ = frame.next_steps(
            &mut |_| panic!("Choice encountered in collect_routed_nodes"),
            &mut visitor,
        );
    }

    RoutedPathTree {
        delivered,
        predecessors,
    }
}

/// Simulates routing from the origin and returns the set of
/// destination nodes (as flat indices) selected by the
/// `Selection`.
///
/// This function discards routing paths and retains only the
/// final delivery targets. It is useful in tests to compare
/// routing results against selection evaluation.
pub fn collect_routed_nodes(selection: &Selection, slice: &Slice) -> Vec<usize> {
    collect_routed_paths(selection, slice)
        .delivered
        .keys()
        .cloned()
        .collect()
}

// == Testing (`collect_commactor_routing_tree` mesh simulation) ===

/// Captures the logical structure of a CommActor multicast operation.
///
/// This type models how a message is delivered and forwarded through
/// a mesh under CommActor routing semantics. It is used in tests to
/// verify path determinism and understand message propagation
/// behavior.
///
/// - `delivered`: ranks where the message was delivered (`post`
///   called)
/// - `visited`: all ranks that participated, including forwarding
///   only
/// - `forwards`: maps each rank to the routing frames it forwarded
#[derive(Default)]
pub struct CommActorRoutingTree {
    // Ranks that were delivered the message (i.e. called `post`). Map
    // from rank → delivery path (flat rank indices) from root to that
    // rank.
    pub delivered: HashMap<usize, Vec<usize>>,

    // Ranks that participated in the multicast - either by delivering
    // the message or forwarding it to peers.
    pub visited: HashSet<usize>,

    /// Map from rank → routing frames this rank forwarded to other
    /// ranks.
    pub forwards: HashMap<usize, Vec<RoutingFrame>>,
}

/// Represents a routing step in the `collect_commactor_routing_tree`
/// simulation.
///
/// Each instance models a message being forwarded from one rank to
/// another, including the routing frames being propagated and the
/// multicast path taken so far.
///
/// - `from`: the sender rank
/// - `to`: the receiver rank
/// - `frames`: routing frames to evaluate at the receiver
/// - `path`: the multicast path from root to this step
#[derive(Debug)]
pub struct ForwardMessage {
    /// The rank that is forwarding the message.
    #[allow(dead_code)] // Never read.
    pub from: usize,

    /// The rank receiving the message.
    pub to: usize,

    /// The routing frames being forwarded.
    pub frames: Vec<RoutingFrame>,

    /// The multicast path taken so far.
    pub path: Vec<usize>,
}

/// `collect_commactor_routing_tree` simulates how messages propagate
/// through a mesh of `CommActor`s during multicast, reconstructing
/// the full logical routing tree.
///
/// This function mirrors the behavior of `CommActor::handle_message`
/// and `CommActor::forward`, using the shared `resolve_routing` logic
/// to determine delivery and forwarding at each step. Starting from
/// the root frame, it simulates how the message would be forwarded
/// peer-to-peer through the system.
///
/// The returned `CommActorRoutingTree` includes:
/// - `delivered`: ranks where the message would be delivered (i.e.,
///   `post` called)
/// - `visited`: all ranks that received or forwarded the message
/// - `forwards`: frames forwarded from each rank to peers
///
/// This model is used in tests to validate routing behavior,
/// especially invariants like path determinism and delivery coverage.
pub fn collect_commactor_routing_tree(
    selection: &Selection,
    slice: &Slice,
) -> CommActorRoutingTree {
    use std::collections::VecDeque;

    let mut pending = VecDeque::new();
    let mut tree = CommActorRoutingTree::default();

    let root_frame = RoutingFrame::root(selection.clone(), slice.clone());
    let origin = slice.location(&root_frame.here).unwrap();
    pending.push_back(ForwardMessage {
        from: origin,
        to: origin,
        frames: vec![root_frame],
        path: vec![origin],
    });

    while let Some(ForwardMessage {
        from: _,
        to: rank,
        frames: dests,
        path,
    }) = pending.pop_front()
    {
        // This loop models the core of `CommActor::handle(...,
        // fwd_message: ForwardMessage)` +
        // `CommActor::handle_message(... next_steps...)`.
        // - `resolve_routing` corresponds to the call in `handle`
        // - delivery and forwarding match the logic in `handle_message`
        // - each forward step simulates `CommActor::forward`

        tree.visited.insert(rank);

        let (deliver_here, forwards) =
            resolve_routing(rank, dests, &mut |_| panic!("choice unexpected")).unwrap();

        if deliver_here {
            tree.delivered.insert(rank, path.clone());
        }

        let messages: Vec<_> = forwards
            .into_iter()
            .map(|(peer, peer_frames)| {
                tree.forwards
                    .entry(rank)
                    .or_default()
                    .extend(peer_frames.clone());

                let mut peer_path = path.clone();
                peer_path.push(peer);

                ForwardMessage {
                    from: rank,
                    to: peer,
                    frames: peer_frames,
                    path: peer_path,
                }
            })
            .collect();

        for message in messages {
            pending.push_back(message);
        }
    }

    tree
}
