//! Defines the accumulator trait and some common accumulators.

use std::marker::PhantomData;

/// An accumulator is a object that accumulates updates into a state.
pub trait Accumulator {
    /// The type of the accumulated state.
    type State;
    /// The type of the updates sent to the accumulator. Updates will be
    /// accumulated into type [Self::State].
    type Update;

    /// Accumulate an update into the current state.
    fn accumulate(&self, state: &mut Self::State, update: &Self::Update);
}

/// Accumulate the sum of received updates. The inner function performs the
/// summation between an update and the current state.
struct SumAccumulator<T>(PhantomData<T>);

impl<T: std::ops::Add<Output = T> + Copy> Accumulator for SumAccumulator<T> {
    type State = T;
    type Update = T;

    fn accumulate(&self, state: &mut T, update: &T) {
        *state = *state + *update;
    }
}

/// Accumulate the sum of received updates.
pub fn sum<T: std::ops::Add<Output = T> + Copy>() -> impl Accumulator<State = T, Update = T> {
    SumAccumulator(PhantomData)
}

/// Accumulate the order of received updates. The inner function performs the
/// comparison between an update and the current state. For example, if the
/// operation is `min`, the accumulator will return the minimum of all the
/// received updates.
struct OrdAccumulator<T>(fn(T, T) -> T);

impl<T: Ord + Copy> Accumulator for OrdAccumulator<T> {
    type State = T;
    type Update = T;

    fn accumulate(&self, state: &mut T, update: &T) {
        *state = (self.0)(*state, *update);
    }
}

/// Accumulate the min of received updates (i.e. the smallest value of all
/// received updates).
pub fn min<T: Ord + Copy>() -> impl Accumulator<State = T, Update = T> {
    OrdAccumulator(std::cmp::min)
}

/// Accumulate the max of received updates (i.e. the largest value of all
/// received updates).
pub fn max<T: Ord + Copy>() -> impl Accumulator<State = T, Update = T> {
    OrdAccumulator(std::cmp::max)
}
