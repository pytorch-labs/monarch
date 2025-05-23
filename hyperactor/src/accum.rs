/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Defines the accumulator trait and some common accumulators.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::OnceLock;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::Named;
use crate::data::Serialized;
use crate::intern_typename;

/// An accumulator is a object that accumulates updates into a state.
pub trait Accumulator {
    /// The type of the accumulated state.
    type State;
    /// The type of the updates sent to the accumulator. Updates will be
    /// accumulated into type [Self::State].
    type Update;
    /// The type of the comm reducer used by this accumulator.
    type Reducer: CommReducer<Update = Self::Update> + Named;

    /// Accumulate an update into the current state.
    fn accumulate(&self, state: &mut Self::State, update: &Self::Update);

    /// The typehash of the underlying [Self::Reducer] type.
    fn reducer_typehash(&self) -> u64 {
        <Self::Reducer as Named>::typehash()
    }
}

/// Commutative reducer for an accumulator. This is used to coallesce updates.
/// For example, if the accumulator is a sum, its reducer calculates and returns
/// the sum of 2 updates. This is helpful in split ports, where a large number
/// of updates can be reduced into a smaller number of updates before being sent
/// to the parent port.
pub trait CommReducer {
    /// The type of updates to be reduced.
    type Update;

    /// Reduce 2 updates into a single update.
    fn reduce(&self, left: Self::Update, right: Self::Update) -> Self::Update;
}

/// Type erased version of [CommReducer].
pub(crate) trait ErasedCommReducer {
    /// Reduce 2 updates into a single update.
    fn reduce_erased(&self, left: &Serialized, right: &Serialized) -> anyhow::Result<Serialized>;

    /// Reducer an non-empty vector of updates. Return Error if the vector is
    /// empty.
    fn reduce_updates(
        &self,
        updates: Vec<Serialized>,
    ) -> Result<Serialized, (anyhow::Error, Vec<Serialized>)> {
        if updates.is_empty() {
            return Err((anyhow::anyhow!("empty updates"), updates));
        }
        if updates.len() == 1 {
            return Ok(updates.into_iter().next().expect("checked above"));
        }

        let mut iter = updates.iter();
        let first = iter.next().unwrap();
        let second = iter.next().unwrap();
        let init = match self.reduce_erased(first, second) {
            Ok(v) => v,
            Err(e) => return Err((e, updates)),
        };
        let reduced = match iter.try_fold(init, |acc, e| self.reduce_erased(&acc, e)) {
            Ok(v) => v,
            Err(e) => return Err((e, updates)),
        };
        Ok(reduced)
    }

    /// Typehash of the underlying [`CommReducer`] type.
    fn typehash(&self) -> u64;
}

impl<R, T> ErasedCommReducer for R
where
    R: CommReducer<Update = T> + Named,
    T: Serialize + DeserializeOwned + Named,
{
    fn reduce_erased(&self, left: &Serialized, right: &Serialized) -> anyhow::Result<Serialized> {
        let left = left.deserialized::<T>()?;
        let right = right.deserialized::<T>()?;
        let result = self.reduce(left, right);
        Ok(Serialized::serialize(&result)?)
    }

    fn typehash(&self) -> u64 {
        R::typehash()
    }
}

// Register factory instead of ErasedCommReducer trait object because the
// object could have internal state, and cannot be shared.
struct ReducerFactory(fn() -> Box<dyn ErasedCommReducer + Sync + Send + 'static>);

inventory::collect!(ReducerFactory);

inventory::submit! {
    ReducerFactory(|| Box::new(SumReducer::<i64>(PhantomData)))
}
inventory::submit! {
    ReducerFactory(|| Box::new(SumReducer::<u64>(PhantomData)))
}
inventory::submit! {
    ReducerFactory(|| Box::new(MaxReducer::<i64>(PhantomData)))
}
inventory::submit! {
    ReducerFactory(|| Box::new(MaxReducer::<u64>(PhantomData)))
}
inventory::submit! {
    ReducerFactory(|| Box::new(MinReducer::<i64>(PhantomData)))
}
inventory::submit! {
    ReducerFactory(|| Box::new(MinReducer::<u64>(PhantomData)))
}

/// Build a reducer object with the given typehash's [CommReducer] type, and
/// return the type-erased version of it.
pub(crate) fn resolve_reducer(
    typehash: u64,
) -> Option<Box<dyn ErasedCommReducer + Sync + Send + 'static>> {
    static FACTORY_MAP: OnceLock<HashMap<u64, &'static ReducerFactory>> = OnceLock::new();
    let factories = FACTORY_MAP.get_or_init(|| {
        let mut map = HashMap::new();
        for factory in inventory::iter::<ReducerFactory> {
            map.insert(factory.0().typehash(), factory);
        }
        map
    });

    factories.get(&typehash).map(|f| f.0())
}

struct SumReducer<T>(PhantomData<T>);

impl<T: std::ops::Add<Output = T> + Copy + 'static> CommReducer for SumReducer<T> {
    type Update = T;

    fn reduce(&self, left: T, right: T) -> T {
        left + right
    }
}

impl<T: Named> Named for SumReducer<T> {
    fn typename() -> &'static str {
        intern_typename!(Self, "hyperactor::accum::SumReducer<{}>", T)
    }
}

/// Accumulate the sum of received updates. The inner function performs the
/// summation between an update and the current state.
struct SumAccumulator<T>(PhantomData<T>);

impl<T: std::ops::Add<Output = T> + Copy + Named + 'static> Accumulator for SumAccumulator<T> {
    type State = T;
    type Update = T;
    type Reducer = SumReducer<T>;

    fn accumulate(&self, state: &mut T, update: &T) {
        *state = *state + *update;
    }
}

/// Accumulate the sum of received updates.
pub fn sum<T: std::ops::Add<Output = T> + Copy + Named + 'static>()
-> impl Accumulator<State = T, Update = T> {
    SumAccumulator(PhantomData)
}

struct MaxReducer<T>(PhantomData<T>);

impl<T: Ord> CommReducer for MaxReducer<T> {
    type Update = T;

    fn reduce(&self, left: T, right: T) -> T {
        std::cmp::max(left, right)
    }
}

impl<T: Named> Named for MaxReducer<T> {
    fn typename() -> &'static str {
        intern_typename!(Self, "hyperactor::accum::MaxReducer<{}>", T)
    }
}

/// Accumulate the max of received updates.
struct MaxAccumulator<T>(PhantomData<T>);

impl<T: Ord + Copy + Named + 'static> Accumulator for MaxAccumulator<T> {
    type State = T;
    type Update = T;
    type Reducer = MaxReducer<T>;

    fn accumulate(&self, state: &mut T, update: &T) {
        *state = std::cmp::max(*state, *update);
    }
}

/// Accumulate the max of received updates (i.e. the largest value of all
/// received updates).
pub fn max<T: Ord + Copy + Named + 'static>() -> impl Accumulator<State = T, Update = T> {
    MaxAccumulator(PhantomData::<T>)
}

struct MinReducer<T>(PhantomData<T>);

impl<T: Ord> CommReducer for MinReducer<T> {
    type Update = T;

    fn reduce(&self, left: T, right: T) -> T {
        std::cmp::min(left, right)
    }
}

impl<T: Named> Named for MinReducer<T> {
    fn typename() -> &'static str {
        intern_typename!(Self, "hyperactor::accum::MinReducer<{}>", T)
    }
}

/// Accumulate the min of received updates.
struct MinAccumulator<T>(PhantomData<T>);

impl<T: Ord + Copy + Named + 'static> Accumulator for MinAccumulator<T> {
    type State = T;
    type Update = T;
    type Reducer = MinReducer<T>;

    fn accumulate(&self, state: &mut T, update: &T) {
        *state = std::cmp::min(*state, *update);
    }
}

/// Accumulate the min of received updates (i.e. the smallest value of all
/// received updates).
pub fn min<T: Ord + Copy + Named + 'static>() -> impl Accumulator<State = T, Update = T> {
    MinAccumulator(PhantomData)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Named;

    #[test]
    fn test_comm_reducer() {
        fn serialize<T: Serialize + Named>(values: Vec<T>) -> Vec<Serialized> {
            values
                .into_iter()
                .map(|n| Serialized::serialize(&n).unwrap())
                .collect()
        }

        let u64_numbers: Vec<_> = serialize(vec![1u64, 3u64, 1100u64]);
        let i64_numbers: Vec<_> = serialize(vec![-123i64, 33i64, 110i64]);
        {
            let typehash = <MaxReducer<u64> as Named>::typehash();
            assert_eq!(
                resolve_reducer(typehash)
                    .unwrap()
                    .reduce_updates(u64_numbers.clone())
                    .unwrap()
                    .deserialized::<u64>()
                    .unwrap(),
                1100u64,
            );

            let typehash = <MinReducer<u64> as Named>::typehash();
            assert_eq!(
                resolve_reducer(typehash)
                    .unwrap()
                    .reduce_updates(u64_numbers.clone())
                    .unwrap()
                    .deserialized::<u64>()
                    .unwrap(),
                1u64,
            );

            let typehash = <SumReducer<u64> as Named>::typehash();
            assert_eq!(
                resolve_reducer(typehash)
                    .unwrap()
                    .reduce_updates(u64_numbers)
                    .unwrap()
                    .deserialized::<u64>()
                    .unwrap(),
                1104u64,
            );
        }

        {
            let typehash = <MaxReducer<i64> as Named>::typehash();
            assert_eq!(
                resolve_reducer(typehash)
                    .unwrap()
                    .reduce_updates(i64_numbers.clone())
                    .unwrap()
                    .deserialized::<i64>()
                    .unwrap(),
                110i64,
            );

            let typehash = <MinReducer<i64> as Named>::typehash();
            assert_eq!(
                resolve_reducer(typehash)
                    .unwrap()
                    .reduce_updates(i64_numbers.clone())
                    .unwrap()
                    .deserialized::<i64>()
                    .unwrap(),
                -123i64,
            );

            let typehash = <SumReducer<i64> as Named>::typehash();
            assert_eq!(
                resolve_reducer(typehash)
                    .unwrap()
                    .reduce_updates(i64_numbers)
                    .unwrap()
                    .deserialized::<i64>()
                    .unwrap(),
                20i64,
            );
        }
    }

    #[test]
    fn test_accum_reducer() {
        assert_eq!(
            sum::<u64>().reducer_typehash(),
            <SumReducer::<u64> as Named>::typehash(),
        );
        assert_eq!(
            sum::<i64>().reducer_typehash(),
            <SumReducer::<i64> as Named>::typehash(),
        );

        assert_eq!(
            min::<u64>().reducer_typehash(),
            <MinReducer::<u64> as Named>::typehash(),
        );
        assert_eq!(
            min::<i64>().reducer_typehash(),
            <MinReducer::<i64> as Named>::typehash(),
        );

        assert_eq!(
            max::<u64>().reducer_typehash(),
            <MaxReducer::<u64> as Named>::typehash(),
        );
        assert_eq!(
            max::<i64>().reducer_typehash(),
            <MaxReducer::<i64> as Named>::typehash(),
        );
    }
}
