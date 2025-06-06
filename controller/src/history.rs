/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;

use hyperactor::clock::Clock;
use hyperactor::data::Serialized;
use monarch_messages::client::Exception;
use monarch_messages::controller::Seq;
use monarch_messages::controller::WorkerError;
use monarch_messages::worker::Ref;

/// An invocation tracks a discrete node in the graph of operations executed by
/// the worker based on instructions from the client.
/// It is useful for tracking the dependencies of an operation and propagating
/// failures. In the future this will be used with more data dependency tracking
/// to support better failure handling.
// Allowing dead code until we do something smarter with defs, uses etc.
#[allow(dead_code)]
#[derive(Debug)]
struct Invocation {
    /// The sequence number of the invocation. This should be unique and increasing across all
    /// invocations.
    seq: Seq,
    /// The references that this invocation defines or redefines. Effectively the
    /// output of the invocation.
    defs: Vec<Ref>,
    /// The references that this invocation uses. Effectively the input of the invocation.
    uses: Vec<Ref>,
    /// The result of the invocation. This is set when the invocation is completed or
    /// when a failure is inferred. A successful result will always supersede any failure.
    result: Option<Result<Serialized, Exception>>,
    /// The seqs for the invocations that depend on this invocation. Useful for propagating failures.
    users: HashSet<Seq>,
    /// If we have reported the result eagerly, we want to make sure to not double report. This also
    /// lets us know when we can stop traversing when finding unreported results
    reported: bool,
}

impl Invocation {
    fn new(seq: Seq, uses: Vec<Ref>, defs: Vec<Ref>) -> Self {
        Self {
            seq,
            uses,
            defs,
            result: None,
            users: HashSet::new(),
            reported: false,
        }
    }

    fn add_user(&mut self, user: Seq) {
        self.users.insert(user);
    }

    /// Invocation results can only go from valid to failed, or be
    /// set if the invocation result is empty.
    fn set_result(&mut self, result: Result<Serialized, Exception>) {
        if self.result.is_none() || matches!((&self.result, &result), (Some(Ok(_)), Err(_))) {
            self.result = Some(result);
        }
    }

    fn set_exception(&mut self, exception: Exception) {
        match exception {
            Exception::Error(_, caused_by, error) => {
                self.set_result(Err(Exception::Error(self.seq, caused_by, error)));
            }
            Exception::Failure(_) => {
                tracing::error!(
                    "system failures {:?} can never be assigned for an invocation",
                    exception
                );
            }
        }
    }

    fn exception(&self) -> Option<&Exception> {
        self.result
            .as_ref()
            .map(Result::as_ref)
            .and_then(Result::err)
    }

    #[allow(dead_code)]
    fn value(&self) -> Option<&Serialized> {
        self.result
            .as_ref()
            .map(Result::as_ref)
            .and_then(Result::ok)
    }
}

#[derive(Debug, PartialEq)]
enum RefStatus {
    // The invocation for this ref is still in progress.
    Invoked(Seq),
    // The invocation for this ref has errored.
    Errored(Exception),
}

/// The history of invocations sent by the client to be executed on the workers.
/// This is used to track dependencies between invocations and to propagate exceptions.
/// It purges history for completed invocations to avoid memory bloat.
/// TODO: Revisit this setup around purging refs automatically once we start doing
/// more complex data dependency tracking. We will want to be more aware of things like
/// borrows, drops etc. directly.
#[derive(Debug)]
#[allow(dead_code)]
pub struct History {
    /// The first incomplete Seq for each rank. This is used to determine which
    /// Seqs are no longer relevant and can be purged from the history.
    first_incomplete_seqs: MinVector<Seq>,
    /// The minimum incomplete Seq across all ranks.
    min_incomplete_seq: Seq,
    /// A map of seq to the invocation that it represents.
    invocations: HashMap<Seq, Invocation>,
    /// A map of reference to the seq for the invocation that defines it. This is used to
    /// compute dependencies between invocations.
    invocation_for_ref: HashMap<Ref, RefStatus>,
    // Refs to be deleted in mark_worker_complete_and_propagate_failures
    marked_for_deletion: HashSet<Ref>,
    // Last seq to be invoked
    max_seq: OptionSeq,
    // The first incompleted Seq for each rank derived from both client and controller request_status messages
    // This is needed because the client may retain invocations past the time completed such as in call_fetch_shard().result()
    first_incomplete_seqs_controller: MinVector<Seq>,
    // Memoized minimum incompleted Seq across all ranks of first_incomplete_seqs_controller
    min_incompleted_seq_controller: Seq,
    // The deadline for the next expected completed seq. This is updated only when the previous deadline
    // has been met.
    //
    // Tuple fields are:
    // - the seq we expect to be completed
    // - the deadline
    // - if it has already been reported to the client
    deadline: Option<(Seq, tokio::time::Instant, bool)>,
}

/// A vector that keeps track of the minimum value.
#[derive(Debug)]
struct MinVector<T> {
    data: Vec<T>,
    value_counts: BTreeMap<T, usize>,
}

impl<T> MinVector<T>
where
    T: Ord + Copy,
{
    fn new(data: Vec<T>) -> Self {
        let mut value_counts = BTreeMap::new();
        for &value in &data {
            *value_counts.entry(value).or_insert(0) += 1;
        }
        MinVector { data, value_counts }
    }

    fn set(&mut self, index: usize, value: T) {
        // Decrease the count of the old value
        let old_value = self.data[index];
        if let Some(count) = self.value_counts.get_mut(&old_value) {
            *count -= 1;
            if *count == 0 {
                self.value_counts.remove(&old_value);
            }
        }
        // Update the value in the vector
        self.data[index] = value;

        // Increase the count of the new value
        *self.value_counts.entry(value).or_insert(0) += 1;
    }

    fn get(&self, index: usize) -> T {
        self.data[index]
    }

    fn min(&self) -> T {
        *self.value_counts.keys().next().unwrap()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn vec(&self) -> &Vec<T> {
        &self.data
    }
}

impl History {
    pub fn new(world_size: usize) -> Self {
        Self {
            first_incomplete_seqs: MinVector::new(vec![Seq::default(); world_size]),
            min_incomplete_seq: Seq::default(),
            invocation_for_ref: HashMap::new(),
            invocations: HashMap::new(),
            marked_for_deletion: HashSet::new(),
            max_seq: OptionSeq::from(None),
            first_incomplete_seqs_controller: MinVector::new(vec![Seq::default(); world_size]),
            min_incompleted_seq_controller: Seq::default(),
            deadline: None,
        }
    }

    #[cfg(test)]
    pub fn first_incomplete_seqs(&self) -> &[Seq] {
        self.first_incomplete_seqs.vec()
    }

    pub fn first_incomplete_seqs_controller(&self) -> &[Seq] {
        self.first_incomplete_seqs_controller.vec()
    }

    pub fn min_incomplete_seq_reported(&self) -> Seq {
        self.min_incompleted_seq_controller
    }

    pub fn world_size(&self) -> usize {
        self.first_incomplete_seqs.len()
    }

    pub fn delete_invocations_for_refs(&mut self, refs: Vec<Ref>) {
        self.marked_for_deletion.extend(refs);

        self.marked_for_deletion
            .retain(|ref_| match self.invocation_for_ref.get(ref_) {
                Some(RefStatus::Invoked(seq)) => {
                    if seq < &self.min_incomplete_seq {
                        self.invocation_for_ref.remove(ref_);
                        false
                    } else {
                        true
                    }
                }
                Some(RefStatus::Errored(_)) => {
                    self.invocation_for_ref.remove(ref_);
                    false
                }
                None => true,
            });
    }

    /// Add an invocation to the history.
    pub fn add_invocation(
        &mut self,
        seq: Seq,
        uses: Vec<Ref>,
        defs: Vec<Ref>,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        let mut results = Vec::new();
        let input_seq = OptionSeq::from(seq);
        assert!(
            input_seq >= self.max_seq,
            "nonmonotonic seq: {:?}; current max: {:?}",
            seq,
            self.max_seq,
        );
        self.max_seq = input_seq;
        let mut invocation = Invocation::new(seq, uses.clone(), defs.clone());

        for use_ in uses {
            // The invocation for every use_ should add this seq as a user.
            match self.invocation_for_ref.get(&use_) {
                Some(RefStatus::Errored(exception)) => {
                    // We know that this invocation hasn't been completed yet, so we can
                    // directly call set_exception on it.
                    invocation.set_exception(exception.clone());
                    results.push((seq, Some(Err(exception.clone()))));
                    invocation.reported = true;
                }
                Some(RefStatus::Invoked(invoked_seq)) => {
                    if let Some(invocation) = self.invocations.get_mut(invoked_seq) {
                        invocation.add_user(seq)
                    }
                }
                None => tracing::debug!(
                    "ignoring dependency on potentially complete invocation for ref: {:?}",
                    use_
                ),
            }
        }
        for def in defs {
            self.invocation_for_ref.insert(
                def,
                match invocation.exception() {
                    Some(err) => RefStatus::Errored(err.clone()),
                    None => RefStatus::Invoked(seq.clone()),
                },
            );
        }

        self.invocations.insert(seq, invocation);

        results
    }

    /// Propagate worker error to the invocation with the given Seq. This will also propagate
    /// to all seqs that depend on this seq directly or indirectly.
    pub fn propagate_exception(&mut self, seq: Seq, exception: Exception) {
        let mut queue = vec![seq];
        let mut visited = HashSet::new();

        while let Some(seq) = queue.pop() {
            if !visited.insert(seq) {
                continue;
            }

            let Some(invocation) = self.invocations.get_mut(&seq) else {
                continue;
            };

            // Overwrite the error, so we are using the last error for this invocation to send
            // to the client.
            for def in invocation.defs.iter() {
                match self.invocation_for_ref.get(def) {
                    Some(RefStatus::Invoked(invoked_seq)) if *invoked_seq == seq => self
                        .invocation_for_ref
                        .insert(*def, RefStatus::Errored(exception.clone())),
                    _ => None,
                };
            }
            invocation.set_exception(exception.clone());
            queue.extend(invocation.users.iter());
        }
    }

    fn find_unreported_dependent_exceptions(
        &mut self,
        seq: Seq,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        let mut queue = vec![seq];
        let mut visited = HashSet::new();
        let mut results = Vec::new();

        while let Some(seq) = queue.pop() {
            if !visited.insert(seq) {
                continue;
            }

            let Some(invocation) = self.invocations.get_mut(&seq) else {
                continue;
            };

            if !matches!(invocation.result, Some(Err(_))) || invocation.reported {
                continue;
            }

            invocation.reported = true;

            results.push((seq, invocation.result.clone()));

            queue.extend(invocation.users.iter());
        }
        results
    }

    pub fn report_deadline_missed(&mut self) {
        if let Some((seq, time, _)) = self.deadline {
            self.deadline = Some((seq, time, true));
        }
    }

    pub fn deadline(
        &mut self,
        expected_progress: u64,
        timeout: tokio::time::Duration,
        clock: &impl Clock,
    ) -> Option<(Seq, tokio::time::Instant, bool)> {
        let previous_deadline_completed = match self.deadline {
            Some((expected_seq, ..)) => self.min_incompleted_seq_controller > expected_seq,
            None => self.max_seq.inner().is_some(),
        };

        if previous_deadline_completed {
            let next_expected_completed_seq = std::cmp::min(
                OptionSeq::from(u64::from(self.min_incompleted_seq_controller) + expected_progress),
                self.max_seq.clone(),
            );

            self.deadline =
                next_expected_completed_seq
                    .into_inner()
                    .map(|next_expected_completed_seq| {
                        (next_expected_completed_seq, clock.now() + timeout, false)
                    });
        }
        self.deadline
    }

    pub fn update_deadline_tracking(&mut self, rank: usize, seq: Seq) {
        // rank_completed also calls this so that we stay up to date with client request_status messages.
        // However, controller request_status messages may be ahead of the client as the client may retain invocations
        // past the time completed so we should take the max
        self.first_incomplete_seqs_controller.set(
            rank,
            std::cmp::max(seq, self.first_incomplete_seqs.get(rank)),
        );

        self.min_incompleted_seq_controller = self.first_incomplete_seqs_controller.min();
    }

    /// Mark the given rank as completed up to but excluding the given Seq. This will also purge history for
    /// any Seqs that are no longer relevant (completed on all ranks).
    pub fn rank_completed(
        &mut self,
        rank: usize,
        seq: Seq,
    ) -> Vec<(Seq, Option<Result<Serialized, Exception>>)> {
        self.first_incomplete_seqs.set(rank, seq);
        let prev = self.min_incomplete_seq;
        self.min_incomplete_seq = self.first_incomplete_seqs.min();
        self.update_deadline_tracking(rank, seq);

        let mut results: Vec<(Seq, Option<Result<Serialized, Exception>>)> = Vec::new();
        for i in Seq::iter_between(prev, self.min_incomplete_seq) {
            if let Some(invocation) = self.invocations.remove(&i) {
                let retain = if let Some(result) = invocation.result {
                    let is_err = result.is_err();
                    if !invocation.reported {
                        results.push((i, Some(result)));
                    }
                    is_err
                } else {
                    // Do not retain successful invocations.
                    results.push((i, None));
                    false
                };

                if retain {
                    // Retain the def history because we may need it to propagate
                    // errors in the future. We rely here on the fact that the invocation
                    // above has been marked as failed by way of failure propagation.
                    for def in &invocation.defs {
                        match self.invocation_for_ref.get(def) {
                            Some(RefStatus::Invoked(seq)) if *seq == i => {
                                self.invocation_for_ref.remove(def)
                            }
                            _ => None,
                        };
                    }
                }
            }
        }

        // Propagate results to the client even if it is behind the completion frontier
        // if we can determine for sure that it is completed
        results.extend(self.find_unreported_dependent_exceptions(seq));

        results
    }

    #[cfg(test)]
    fn get_invocation(&self, seq: Seq) -> Option<&Invocation> {
        self.invocations.get(&seq)
    }

    pub fn set_result(&mut self, seq: Seq, result: Result<Serialized, WorkerError>) {
        if let Some(invocation) = self.invocations.get_mut(&seq) {
            invocation.set_result(result.map_err(|e| Exception::Error(seq, seq, e)));
        }
    }
}

/// Struct representing an optional `Seq`, where `None` is always considered the
/// smallest. This type is to make it easier to compare `Option<Seq>` with `Seq`
/// or `Option<Seq>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OptionSeq(Option<Seq>);

impl OptionSeq {
    /// Return inner ref.
    pub fn inner(&self) -> &Option<Seq> {
        &self.0
    }

    /// Return inner.
    pub fn into_inner(self) -> Option<Seq> {
        self.0
    }
}

impl PartialOrd for OptionSeq {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OptionSeq {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.0, other.0) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        }
    }
}

impl From<u64> for OptionSeq {
    fn from(value: u64) -> Self {
        OptionSeq(Some(Seq::from(value)))
    }
}

impl From<Seq> for OptionSeq {
    fn from(seq: Seq) -> Self {
        OptionSeq(Some(seq))
    }
}

impl From<Option<Seq>> for OptionSeq {
    fn from(value: Option<Seq>) -> Self {
        OptionSeq(value)
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use hyperactor::id;

    use super::*;

    struct InvocationUsersIterator<'a> {
        history: &'a History,
        stack: Vec<&'a Invocation>,
        visited: HashSet<Seq>,
    }

    impl<'a> Iterator for InvocationUsersIterator<'a> {
        type Item = &'a Invocation;

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(invocation) = self.stack.pop() {
                if !self.visited.insert(invocation.seq) {
                    continue;
                }
                self.stack.extend(
                    invocation
                        .users
                        .iter()
                        .filter_map(|seq| self.history.invocations.get(seq)),
                );
                return Some(invocation);
            }
            None
        }
    }

    impl History {
        /// Get an iterator of Seqs that are users of (or dependent on the completion of) the given Seq.
        /// This is useful for propagating failures. This will return an empty iterator if the given Seq is
        /// not in the history. So this should be called before the invocation is marked as completed for the
        /// given rank.
        /// The Seq passed to this function will also be included in the iterator.
        pub(crate) fn iter_users_transitive(&self, seq: Seq) -> impl Iterator<Item = Seq> + '_ {
            let invocations = self
                .invocations
                .get(&seq)
                .map_or(Vec::default(), |invocation| vec![invocation]);

            InvocationUsersIterator {
                history: self,
                stack: invocations,
                visited: HashSet::new(),
            }
            .map(|invocation| invocation.seq)
        }
    }

    #[test]
    fn simple_history() {
        let mut history = History::new(2);
        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.add_invocation(2.into(), vec![Ref { id: 3 }], vec![Ref { id: 4 }]);
        history.add_invocation(3.into(), vec![Ref { id: 3 }], vec![Ref { id: 5 }]);
        history.add_invocation(4.into(), vec![Ref { id: 3 }], vec![Ref { id: 6 }]);
        history.add_invocation(5.into(), vec![Ref { id: 4 }], vec![Ref { id: 7 }]);
        history.add_invocation(6.into(), vec![Ref { id: 4 }], vec![Ref { id: 8 }]);

        let mut res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        res.sort();
        assert_eq!(
            res,
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );

        history.rank_completed(0, 2.into());
        let mut res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        res.sort();
        assert_eq!(
            res,
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );

        history.rank_completed(1, 2.into());
        let res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        assert_eq!(res, vec![]);

        // Test that we can still add invocations after all ranks have completed that seq.
        history.add_invocation(7.into(), vec![Ref { id: 1 }], vec![]);
    }

    #[test]
    fn delete_errored_invocations() {
        let mut history = History::new(1);
        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.propagate_exception(
            0.into(),
            Exception::Error(
                0.into(),
                0.into(),
                WorkerError {
                    backtrace: "worker error happened".to_string(),
                    worker_actor_id: id!(test[234].testactor[6]),
                },
            ),
        );
        history.delete_invocations_for_refs(vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.rank_completed(0, 1.into());
        assert_eq!(history.invocation_for_ref.len(), 1);
        history.delete_invocations_for_refs(vec![Ref { id: 3 }]);
        history.rank_completed(0, 2.into());
        assert!(history.invocation_for_ref.is_empty());
    }

    #[test]
    fn redefinitions() {
        let mut history = History::new(2);
        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.add_invocation(2.into(), vec![Ref { id: 3 }], vec![Ref { id: 4 }]);

        let mut res = history
            .iter_users_transitive(1.into())
            .collect::<Vec<Seq>>();
        res.sort();
        assert_eq!(res, vec![1.into(), 2.into()]);

        history.add_invocation(3.into(), vec![Ref { id: 3 }], vec![Ref { id: 3 }]);
        history.add_invocation(4.into(), vec![Ref { id: 3 }], vec![Ref { id: 6 }]);
        history.add_invocation(5.into(), vec![Ref { id: 4 }], vec![Ref { id: 7 }]);
        history.add_invocation(6.into(), vec![Ref { id: 4 }], vec![Ref { id: 8 }]);

        history.rank_completed(0, 2.into());
        history.rank_completed(1, 2.into());

        let res = history
            .iter_users_transitive(3.into())
            .collect::<Vec<Seq>>();
        assert_eq!(res, vec![3.into(), 4.into()]);
    }

    #[test]
    fn min_vector() {
        // Test initialization
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut min_vector = MinVector::new(data.clone());

        // Test length
        assert_eq!(min_vector.len(), data.len());

        // Test initial vector
        assert_eq!(min_vector.vec(), &data);

        // Test initial minimum
        assert_eq!(min_vector.min(), 1);

        // Test get method
        for (i, &value) in data.iter().enumerate() {
            assert_eq!(min_vector.get(i), value);
        }

        // Test set method and min update
        min_vector.set(0, 0); // Change first element to 0
        assert_eq!(min_vector.get(0), 0);
        assert_eq!(min_vector.min(), 0);
        min_vector.set(1, 7); // Change second element to 7
        assert_eq!(min_vector.get(1), 7);
        assert_eq!(min_vector.min(), 0);
        min_vector.set(0, 8); // Change first element to 8
        assert_eq!(min_vector.get(0), 8);
        assert_eq!(min_vector.min(), 1); // Minimum should now be 1

        // Test setting a value that already exists
        min_vector.set(2, 5); // Change third element to 5
        assert_eq!(min_vector.get(2), 5);
        assert_eq!(min_vector.min(), 1);

        // Test setting a value to the current minimum
        min_vector.set(3, 0); // Change fourth element to 0
        assert_eq!(min_vector.get(3), 0);
        assert_eq!(min_vector.min(), 0);

        // Test setting all elements to the same value
        for i in 0..min_vector.len() {
            min_vector.set(i, 10);
        }
        assert_eq!(min_vector.min(), 10);
        assert_eq!(min_vector.vec(), &vec![10; min_vector.len()]);
    }

    #[test]
    fn failure_propagation() {
        let mut history = History::new(2);

        history.add_invocation(0.into(), vec![], vec![Ref { id: 1 }, Ref { id: 2 }]);
        history.add_invocation(1.into(), vec![Ref { id: 1 }], vec![Ref { id: 3 }]);
        history.add_invocation(
            2.into(),
            vec![Ref { id: 3 }],
            vec![Ref { id: 4 }, Ref { id: 5 }],
        );
        history.add_invocation(3.into(), vec![Ref { id: 2 }], vec![Ref { id: 6 }]);
        history.add_invocation(4.into(), vec![Ref { id: 5 }], vec![Ref { id: 6 }]);

        // No error before propagation
        for i in 1..=3 {
            assert!(
                history
                    .get_invocation(i.into())
                    .unwrap()
                    .exception()
                    .is_none()
            );
        }

        // Failure happened to invocation 1, invocations 2, 4 should be marked as failed because they
        // depend on 1 directly or indirectly.
        history.propagate_exception(
            1.into(),
            Exception::Error(
                1.into(),
                1.into(),
                WorkerError {
                    backtrace: "worker error happened".to_string(),
                    worker_actor_id: "test[234].testactor[6]".parse().unwrap(),
                },
            ),
        );

        // Error should be set for all invocations that depend on the failed invocation
        for i in [1, 2, 4] {
            assert!(
                history
                    .get_invocation(i.into())
                    .unwrap()
                    .exception()
                    .is_some()
            );
        }

        // Error should not be set for invocations that do not depend on the failed invocation
        for i in [0, 3] {
            assert!(
                history
                    .get_invocation(i.into())
                    .unwrap()
                    .exception()
                    .is_none()
            );
        }

        // A failed but completed invocation should still lead to all its
        // invocations being marked as failed even if they appear in the future.

        // Delete until 2.
        history.rank_completed(0, 2.into());
        history.rank_completed(1, 2.into());

        for i in [3, 4, 5, 6] {
            assert_matches!(
                history.invocation_for_ref.get(&i.into()),
                Some(RefStatus::Errored(_)),
            );
            // Invocation should start from 5, so i+2
            history.add_invocation((i + 2).into(), vec![Ref { id: i }], vec![Ref { id: 7 }]);
            assert!(
                history
                    .get_invocation((i + 2).into())
                    .unwrap()
                    .exception()
                    .is_some()
            );
        }

        // Test if you can fill a valid result on an errored invocation 2.
        history.set_result(
            2.into(),
            Ok(Serialized::serialize(&"2".to_string()).unwrap()),
        );
        // check that seq 2 is still errored
        assert!(
            history
                .get_invocation((2).into())
                .unwrap()
                .exception()
                .is_some()
        );
        assert!(
            history
                .get_invocation((2).into())
                .unwrap()
                .value()
                .is_none()
        );
    }

    #[test]
    fn test_option_seq_comparision() {
        assert_eq!(OptionSeq::from(None), OptionSeq::from(None));
        assert_eq!(OptionSeq::from(1), OptionSeq::from(Seq::from(1)));
        assert_eq!(OptionSeq::from(1), OptionSeq::from(Some(Seq::from(1))));

        assert!(OptionSeq::from(None) < OptionSeq::from(0));
        assert!(OptionSeq::from(0) < OptionSeq::from(1));

        assert!(OptionSeq::from(0) > OptionSeq::from(None));
        assert!(OptionSeq::from(1) > OptionSeq::from(0));
    }
}
