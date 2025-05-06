//! Rust bindings for libtorch C++ APIs.
//!
//! These bindings were designed with the following goals:
//! - Fundamental types should look and perform close to the C++ version. In
//!   particular, we bind [`Tensor`] and [`IValue`] by hand, so that they can be
//!   passed around by value rather than requiring a heap allocation.
//! - We want to minimize the amount of application logic that needs to be
//!   written in C++, and avoid complex invariants that need to be maintained
//!   across languages.
//! - Types exposed by the bindings should behave like regular Rust types. In
//!   particular, they should be safe; from safe Rust code we should never
//!   be able to violate Rust's invariants.
//!
//! At the moment, these bindings implement the minimal functionality needed to
//! work with the PyTorch object model and perform dispatch on PyTorch ops.
//!
//! # Example
//! ```
//! # use std::collections::HashMap;
//! # use std::error::Error;
//! # use torch_sys::RValue;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! let sizes = RValue::from(vec![2, 3]);
//!
//! let mut outputs =
//!     torch_sys::call_op::call_op("aten::ones", "", &[sizes.clone()], &HashMap::new(), true)?;
//! let t1 = outputs.pop().unwrap();
//!
//! // Can do kwargs as well
//! let kwargs = HashMap::from([("size".into(), sizes)]);
//! let mut outputs = torch_sys::call_op::call_op("aten::ones", "", &[], &kwargs, true)?;
//! let t2 = outputs.pop().unwrap();
//!
//! let mut outputs =
//!     torch_sys::call_op::call_op("aten::allclose", "", &[t1, t2], &HashMap::new(), true)?;
//! let result = outputs.pop().unwrap();
//!
//! assert!(result.try_into()?);
//! # Ok(())
//! # }
//! ```
//!
//! # Safety
//! These are considerations that apply to bindings that deal with `Tensor` (and
//! by extension, `IValue`s since they can contain `Tensor`s). If a binding
//! violates these rules, they must be marked `unsafe` and the additional
//! constraints should be documented.
//!
//! ## Mutability
//!
//! **Rule**: If a binding can potentially mutate a `Tensor`, the safe Rust
//! function signature *must* take take it by either value or `&mut Tensor`.
//!
//! You must manually audit the C++ implementation to determine whether it not
//! it can mutate its arguments.
//!
//! Notably, this is true even if the C++ signature receives a `const Tensor&`.
//! You can still mutate a `Tensor` obtained that way! LibTorch's C++ API
//! doesn't have a concept of an immutable `Tensor` object, so we must rely on
//! manual auditing to ensure that a Rust `&Tensor` is immutable.
//!
//! ## Aliasing
//!
//! **Rule**: A safe binding *must not* produce a new alias of an existing
//! `Tensor`.
//!
//! You must manually audit the C++ implementation to determine whether or not
//! it can produce a new alias. This may involve inserting dynamic aliasing
//! checks if aliasing relationships are not known statically (e.g.
//! `aten::contiguous`).
//!
//! We want the Rust compiler to be correctly tracking ownership and borrowing
//! of `Tensor` and enforcing the invariant that only one mutable reference to a
//! `Tensor` can exist at a time.
//!
//! If a C++ object returned a new alias of an existing `Tensor`, the Rust
//! compiler would treat them as independent `Tensor` objects, and would not be
//! able to prevent a data race if we tried to mutate them on two different
//! threads.
//!
//! In Rust, shared ownership + mutability is handled by having a smart pointer
//! own a value that is synchronized (e.g. the `Arc<Mutex<T>>` pattern). We
//! cannot synchronize access to the C++ underlying `TensorImpl` without
//! changing the implementation of `at::Tensor`, so we must disallow shared
//! ownership in Rust code.

#![feature(assert_matches)]
#![feature(once_cell_try)]

mod bindings;
mod borrow;
mod bridge;
pub mod call_op;
mod cell;
mod device;
mod ivalue;
mod layout;
mod memory_format;
mod pyobject;
mod rvalue;
mod scalar_type;
mod tensor;

pub mod backend;
pub mod cuda;
pub mod nccl;

/// Binding for `c10::Layout`.
pub use bindings::root::c10::Layout;
/// Binding for `c10::MemoryFormat`.
pub use bindings::root::c10::MemoryFormat;
/// Binding for `c10::ScalarType`.
pub use bindings::root::c10::ScalarType;
pub use borrow::Borrow;
pub use borrow::BorrowError;
pub use borrow::BorrowType;
pub use borrow::MultiBorrow;
pub use cell::CloneUnsafe;
pub use device::CudaDevice;
pub use device::Device;
pub use device::DeviceIndex;
pub use device::DeviceParseError;
pub use device::DeviceType;
pub use ivalue::IValue;
pub use ivalue::IValueKind;
pub use ivalue::OpaqueIValue;
pub use rvalue::RValue;
pub use rvalue::rvalue_to_ivalue;
pub use tensor::Tensor;
pub use tensor::TensorCell;

pub use crate::bridge::ffi::deep_clone;
pub use crate::bridge::ffi::factory_float_tensor;
/// Remote serde implementation.
pub use crate::layout::LayoutDef;
/// Remote serde implementation.
pub use crate::memory_format::MemoryFormatDef;
/// Remote serde implementation.
pub use crate::scalar_type::ScalarTypeDef;
pub mod testing {
    /// Compares two tensors with `torch.allclose`.
    pub use crate::bridge::ffi::allclose;
}
pub use crate::bridge::ffi::factory_empty;
pub use crate::bridge::ffi::factory_zeros;
// Only here to make them available to doctests!
#[doc(hidden)]
pub use crate::bridge::ffi::test_make_alias;
#[doc(hidden)]
pub use crate::bridge::ffi::test_make_tensor;
