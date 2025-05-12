//! Core mesh components for the hyperactor framework.
//!
//! Provides [`Slice`], a compact representation of a subset of a
//! multidimensional array. See [`Slice`] for more details.
//!
//! This crate defines the foundational abstractions used in
//! hyperactor's mesh layer, including multidimensional shapes and
//! selection algebra. The crate avoids dependencies on procedural
//! macros and other higher-level constructs, enabling reuse in both
//! runtime and macro contexts.

#![feature(assert_matches)]
#![recursion_limit = "512"]

mod slice;
pub use slice::DimSliceIterator;
pub use slice::Slice;
pub use slice::SliceError;
pub use slice::SliceIterator;

/// Selection algebra for describing multidimensional mesh regions.
pub mod selection;

/// Core types for representing multidimensional shapes and strides.
pub mod shape;

/// Reshaping transformations for multidimensional slices and shapes.
pub mod reshape;

/// The selection expression type used to define routing constraints.
pub use selection::Selection;
/// DSL-style constructors for building `Selection` expressions.
pub use selection::dsl;
/// Represents an interval with an optional end and step, used to
/// define extents in `Shape` and coordinate filters in `Selection`.
pub use shape::Range;
/// Describes the size and layout of a multidimensional mesh.
pub use shape::Shape;
/// Errors that can occur during shape construction or validation.
pub use shape::ShapeError;

/// Property-based generators for randomized test input.
#[cfg(test)]
pub mod strategy;
