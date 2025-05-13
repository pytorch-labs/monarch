use std::collections::HashMap;

use cxx::ExternType;
use cxx::type_id;
use thiserror::Error;

use crate::IValue;
use crate::IValueKind;
use crate::RValue;
use crate::TensorCell;
use crate::borrow::BorrowError;
use crate::borrow::BorrowType;
use crate::borrow::MultiBorrow;
use crate::bridge::ffi::AliasInfo;
use crate::bridge::ffi::AliasKind;
use crate::bridge::ffi::Kwarg;
use crate::bridge::ffi::Tensor;
pub use crate::bridge::ffi::get_schema_args_info;
use crate::ivalue::OpaqueIValue;
use crate::ivalue::OpaqueIValueCell;
use crate::rvalue::rvalue_to_ivalue;

/// Errors that can occur while calling an operator.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CallOpError {
    #[error("torch operator error {0}")]
    TorchOperatorError(#[from] cxx::Exception),

    #[error("error borrowing: {0}")]
    BorrowError(#[from] BorrowError),

    #[error("invalid kwarg '{kwarg}' for op: '{operator}.{overload}'")]
    InvalidKwargs {
        kwarg: String,
        operator: String,
        overload: String,
    },
}

/// An opaque type that represents the type of an argument to a torch operator.
/// This is essentially used to interface with C++ code and should not be instantiated
/// or owned by Rust code.
#[repr(C)]
pub struct TypePtr {
    _private: [u8; 0],
}

// SAFETY: Register our custom bindings with cxx. This is just treating
// at::TypePtr as an opaque type and we would only have refs to it.
unsafe impl ExternType for TypePtr {
    type Id = type_id!("c10::TypePtr");
    type Kind = cxx::kind::Opaque;
}

impl TypePtr {
    #[allow(dead_code)]
    #[inline]
    pub fn is_tensor(&self) -> bool {
        crate::bridge::ffi::type_ptr_is_tensor(self)
    }

    #[allow(dead_code)]
    #[inline]
    pub fn is_tensor_list(&self) -> bool {
        crate::bridge::ffi::type_ptr_is_tensor_list(self)
    }

    #[allow(dead_code)]
    #[inline]
    pub fn is_optional_tensor(&self) -> bool {
        crate::bridge::ffi::type_ptr_is_optional_tensor(self)
    }

    #[allow(dead_code)]
    #[inline]
    pub fn is_optional_tensor_list(&self) -> bool {
        crate::bridge::ffi::type_ptr_is_optional_tensor_list(self)
    }
}

impl std::fmt::Debug for TypePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypePtr").field("type", &"<...>").finish()
    }
}

fn get_aliased_rvalue<'a>(
    alias_info: &'a AliasInfo,
    args: &'a [RValue],
    kwargs: &'a HashMap<String, RValue>,
) -> &'a RValue {
    match args.get(alias_info.arg_idx) {
        // The alias references an arg.
        Some(rvalue) => rvalue,
        None => {
            // This alias references a kwarg.
            let (_name, rvalue) = kwargs
                .iter()
                .find(|(key, _)| *key == &alias_info.arg_name)
                // The aliased value must have been passed in through
                // either args or kwargs, panic if not.
                .unwrap();
            rvalue
        }
    }
}

/// Create a TensorCell out of a tensor, with the correct aliasing information.
fn make_tensor_cell(
    tensor: Tensor,
    alias_info: &AliasInfo,
    args: &[RValue],
    kwargs: &HashMap<String, RValue>,
) -> TensorCell {
    match alias_info.kind {
        AliasKind::NewValue => TensorCell::new(tensor),
        AliasKind::Alias => match get_aliased_rvalue(alias_info, args, kwargs) {
            RValue::Tensor(cell) => TensorCell::new_with_alias(tensor, cell),
            // TODO: call_op should broken down into helpers and directly used in stream.rs
            // and there if wirevalue was IValue we just create a new TensorCell even if it is being
            // aliased as it will not be tracked on the rust worker yet.
            RValue::Opaque(_) => TensorCell::new(tensor),
            _ => panic!("must be a tensor to create an aliases tensorcell"),
        },
        _ => panic!("unsupported alias kind"),
    }
}

fn make_opaque_ivalue_cell(
    ivalue: OpaqueIValue,
    alias_info: &AliasInfo,
    args: &[RValue],
    kwargs: &HashMap<String, RValue>,
) -> OpaqueIValueCell {
    match alias_info.kind {
        AliasKind::NewValue => OpaqueIValueCell::new(ivalue),
        AliasKind::Alias => match get_aliased_rvalue(alias_info, args, kwargs) {
            RValue::Opaque(cell) => OpaqueIValueCell::new_with_alias(ivalue, cell),
            _ => panic!("must be an opaque ivalue to create an aliases opaque ivalue cell"),
        },
        _ => panic!("unsupported alias kind"),
    }
}

/// Call a PyTorch-dispatched operator by name.
///
/// `op_name` is the fully qualified name of the operator, like `"aten::add"`.
///
/// `overload` is the name of the overload, like `"Scalar"`. Due to a
/// quirk of libtorch, the `default` overload must be called by passing
/// an empty string.
///
/// `flatten_results` is a flag that indicates whether the results of the
/// operator should be flattened into a single list. Extracting out values
/// from lists, tuples and dicts recursively.
///
/// # Errors
/// If the called operator throws an exception, a [`cxx::Exception`]
/// will be returned which contains the C++ exception.
pub fn call_op(
    op_name: impl AsRef<str>,
    overload: impl AsRef<str>,
    args: &[RValue],
    kwargs: &HashMap<String, RValue>,
    flatten_results: bool,
) -> Result<Vec<RValue>, CallOpError> {
    // SAFETY: We will be making an unchecked clone of each tensor to pass to to
    // C++, so we need to hold a borrow of each input tensor for the duration of
    // this function.
    let mut multiborrow = MultiBorrow::new();

    let mutates = get_schema_args_info(op_name.as_ref(), overload.as_ref())?;

    // Queue up borrows for the args.
    for (arg, arg_mutability) in args.iter().zip(&mutates) {
        let borrow_type = if arg_mutability.is_mutable {
            BorrowType::Mutable
        } else {
            BorrowType::Shared
        };
        multiborrow.add(arg, borrow_type);
    }

    // Queue up borrows for the kwargs.
    for (key, arg) in kwargs.iter() {
        let arg_mutability = mutates.iter().find(|arg| &arg.name == key).ok_or_else(|| {
            CallOpError::InvalidKwargs {
                kwarg: key.to_string(),
                operator: op_name.as_ref().to_string(),
                overload: overload.as_ref().to_string(),
            }
        })?;
        let borrow_type = if arg_mutability.is_mutable {
            BorrowType::Mutable
        } else {
            BorrowType::Shared
        };
        multiborrow.add(arg, borrow_type);
    }

    // Actually execute the borrows.
    let _borrows = multiborrow.borrow()?;

    let mut ivalue_args: Vec<IValue> = args
        .iter()
        // SAFETY: The borrows above guard the unchecked clones done by
        // `rvalue_to_ivalue`. This may result in multiple mutable references to
        // tensor data, but the C++ side is responsible for making sure that is safe
        // within the context of a single operator invocation.
        .map(|rvalue| unsafe { rvalue_to_ivalue(rvalue) })
        .collect();
    let mut ivalue_kwargs: Vec<Kwarg> = kwargs
        .iter()
        .map(|(key, value)| Kwarg {
            name: key.clone(),
            // SAFETY: see above
            arg: unsafe { rvalue_to_ivalue(value) },
        })
        .collect();

    // SAFETY: we will be unifying the ownership of potential aliases in the
    // returned TensorCells so this is okay to call.
    let call_op_result = unsafe {
        crate::bridge::ffi::call_op_raw(
            op_name.as_ref(),
            overload.as_ref(),
            &mut ivalue_args,
            &mut ivalue_kwargs,
            flatten_results,
        )?
    };
    Ok(call_op_result
        .outputs
        .into_iter()
        .zip(call_op_result.alias_infos)
        .map(|(ivalue, alias_info)| match ivalue.kind() {
            IValueKind::Tensor => RValue::Tensor(make_tensor_cell(
                ivalue.to_tensor().unwrap(),
                &alias_info,
                args,
                kwargs,
            )),
            IValueKind::Bool => RValue::Bool(ivalue.to_bool().unwrap()),
            IValueKind::Int => RValue::Int(ivalue.to_int().unwrap()),
            IValueKind::IntList => RValue::IntList(ivalue.to_int_list().unwrap()),
            IValueKind::Double => RValue::Double(ivalue.to_double().unwrap()),
            IValueKind::String => RValue::String(ivalue.to_string().unwrap()),
            IValueKind::TensorList => {
                let mut tensors = Vec::new();
                let tensor_list = ivalue.to_tensor_list().unwrap();
                for tensor in tensor_list {
                    tensors.push(make_tensor_cell(tensor, &alias_info, args, kwargs));
                }
                RValue::TensorList(tensors)
            }
            IValueKind::Device => RValue::Device(ivalue.to_device().unwrap()),
            IValueKind::None => RValue::None,
            IValueKind::Other => RValue::Opaque(make_opaque_ivalue_cell(
                ivalue.to_opaque().unwrap(),
                &alias_info,
                args,
                kwargs,
            )),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use core::panic;

    use super::*;
    use crate::CloneUnsafe;
    use crate::bridge::ffi::AliasKind;
    use crate::bridge::ffi::call_op_raw;
    use crate::bridge::ffi::is_alias;

    #[test]
    fn call_op_raw_basic() {
        let iv = IValue::from(vec![2, 3].as_slice());
        #[allow(clippy::undocumented_unsafe_blocks)]
        let mut results =
            unsafe { call_op_raw("aten::ones", "", &mut [iv], &mut [], false) }.unwrap();

        assert_eq!(results.outputs.len(), 1, "Expected 1 output");
        assert_eq!(results.alias_infos.len(), 1, "Expected 1 output");

        let t1 = results.outputs.pop().unwrap();
        assert!(
            matches!(results.alias_infos[0].kind, AliasKind::NewValue),
            "output should be a new value"
        );

        let iv = IValue::from(vec![2, 3].as_slice());
        #[allow(clippy::undocumented_unsafe_blocks)]
        let mut results =
            unsafe { call_op_raw("aten::ones", "", &mut [iv], &mut [], false) }.unwrap();

        assert_eq!(results.outputs.len(), 1, "Expected 1 output");
        assert_eq!(results.alias_infos.len(), 1, "Expected 1 output");

        let t2 = results.outputs.pop().unwrap();
        assert!(
            matches!(results.alias_infos[0].kind, AliasKind::NewValue),
            "output should be a new value"
        );

        #[allow(clippy::undocumented_unsafe_blocks)]
        let results =
            unsafe { call_op_raw("aten::allclose", "", &mut [t1, t2], &mut [], false) }.unwrap();
        assert_eq!(results.outputs.len(), 1, "Expected 1 output");
        assert_eq!(results.alias_infos.len(), 1, "Expected 1 output");
        assert!(
            matches!(results.alias_infos[0].kind, AliasKind::NewValue),
            "output should be a new value"
        );

        assert!(
            results.outputs[0]
                .to_bool()
                .expect("expected boolean return"),
            "expected allclose to be true",
        );
    }

    #[test]
    fn call_op_raw_with_aliasing() {
        let size = IValue::from(vec![2, 3].as_slice());
        #[allow(clippy::undocumented_unsafe_blocks)]
        let mut results =
            unsafe { call_op_raw("aten::ones", "", &mut [size], &mut [], false) }.unwrap();
        assert_eq!(results.outputs.len(), 1, "Expected 1 output");
        assert_eq!(results.alias_infos.len(), 1, "Expected 1 output");

        let t1 = results.outputs.pop().unwrap();
        assert!(
            matches!(results.alias_infos[0].kind, AliasKind::NewValue),
            "output should be a new value"
        );

        let size = IValue::from(vec![2, 3].as_slice());
        let mut args = vec![t1, size];
        #[allow(clippy::undocumented_unsafe_blocks)]
        let mut results =
            unsafe { call_op_raw("aten::view", "", args.as_mut_slice(), &mut [], false) }.unwrap();
        assert_eq!(results.outputs.len(), 1, "Expected 1 output");
        assert_eq!(results.alias_infos.len(), 1, "Expected 1 output");

        assert!(
            matches!(results.alias_infos[0].kind, AliasKind::Alias),
            "output should be an alias"
        );
        assert!(
            matches!(results.alias_infos[0].arg_idx, 0),
            "alias should point to the first input"
        );
        let t2 = results.outputs.pop().unwrap();

        assert!(
            is_alias(
                #[allow(clippy::undocumented_unsafe_blocks)]
                unsafe {
                    &args[0].clone_unsafe().to_tensor().unwrap()
                },
                &t2.to_tensor().unwrap(),
            ),
            "c++ tensors should alias"
        );
    }

    #[test]
    fn call_op_raw_with_chunk_aliasing() {
        let size = IValue::from(vec![2, 3].as_slice());
        #[allow(clippy::undocumented_unsafe_blocks)]
        let mut results =
            unsafe { call_op_raw("aten::ones", "", &mut [size], &mut [], false) }.unwrap();
        assert_eq!(results.outputs.len(), 1, "Expected 1 output");
        assert_eq!(results.alias_infos.len(), 1, "Expected 1 output");

        let t1 = results.outputs.pop().unwrap();
        assert!(
            matches!(results.alias_infos[0].kind, AliasKind::NewValue),
            "output should be a new value"
        );

        #[allow(clippy::undocumented_unsafe_blocks)]
        let mut results = unsafe {
            call_op_raw(
                "aten::chunk",
                "",
                &mut [t1.clone_unsafe(), IValue::from(2)],
                &mut [],
                false,
            )
        }
        .unwrap();

        assert_eq!(results.outputs.len(), 1, "Expected 1 output");
        assert_eq!(results.alias_infos.len(), 1, "Expected 1 output");

        let chunked_list = results.outputs.pop().unwrap();
        assert!(
            matches!(results.alias_infos[0].kind, AliasKind::Alias),
            "chunk output should be an alias"
        );
        assert_eq!(
            results.alias_infos[0].arg_idx, 0,
            "chunk output should alias the first input"
        );

        let chunked_list = chunked_list
            .to_tensor_list()
            .expect("return of chunk should be a tensor list");

        let tensor = t1.to_tensor().unwrap();
        for chunk in &chunked_list {
            assert!(is_alias(&tensor, chunk,), "c++ tensors should alias");
        }
    }

    /// Convenience function to avoid lots of unwrapping in test code.
    ///
    /// # Panics
    /// Panics if the arg has more than one result, or if an error occurred.
    fn call_op_one(
        op_name: impl AsRef<str>,
        overload: impl AsRef<str>,
        args: &[RValue],
        kwargs: &HashMap<String, RValue>,
    ) -> RValue {
        let mut results = call_op(op_name, overload, args, kwargs, false).unwrap();
        assert_eq!(results.len(), 1);
        results.pop().unwrap()
    }

    #[test]
    fn call_op_basic() {
        let rv = RValue::from(vec![2, 3]);
        let t1 = call_op_one("aten::ones", "", &[rv.clone()], &HashMap::new());
        let t2 = call_op_one("aten::ones", "", &[rv.clone()], &HashMap::new());

        match (&t1, &t2) {
            (RValue::Tensor(t1), RValue::Tensor(t2)) => {
                assert!(!t1.aliases(t2));
            }
            _ => panic!("expected tensor"),
        }

        let result = call_op("aten::allclose", "", &[t1, t2], &HashMap::new(), false)
            .unwrap()
            .pop()
            .unwrap();

        assert!(
            matches!(result, RValue::Bool(true)),
            "Expected true for allclose output"
        );
    }

    #[test]
    fn call_op_multi_alias() {
        let rv = RValue::from(vec![2, 3]);
        let t1 = call_op_one("aten::ones", "", &[rv.clone()], &HashMap::new());
        let t1_view = call_op_one("aten::view", "", &[t1.clone(), rv.clone()], &HashMap::new());

        let t1_cell: TensorCell = t1.clone().try_into().unwrap();
        let t1_view_cell: TensorCell = t1_view.try_into().unwrap();
        assert!(t1_cell.aliases(&t1_view_cell));

        // Two threads can call non-mutating ops on the same alias with no problem.
        let handle1 = std::thread::spawn(move || {
            for _ in 0..1000 {
                call_op(
                    "aten::add",
                    "Tensor",
                    &[t1.clone(), t1.clone()],
                    &HashMap::new(),
                    false,
                )
                .unwrap();
            }
        });

        let handle2 = std::thread::spawn(move || {
            let t1_view: RValue = t1_view_cell.clone().into();
            for _ in 0..1000 {
                call_op(
                    "aten::add",
                    "Tensor",
                    &[t1_view.clone(), t1_view.clone()],
                    &HashMap::new(),
                    false,
                )
                .unwrap();
            }
        });
        handle1.join().unwrap();
        handle2.join().unwrap();
    }

    /// Trying to call an op with a mutable and immutable borrow of the same alias should work.
    #[test]
    fn call_op_multi_alias_mutable() {
        let rv = RValue::from(vec![2, 3]);
        let t1 = call_op_one("aten::ones", "", &[rv.clone()], &HashMap::new());
        let t1_view = call_op_one("aten::view", "", &[t1.clone(), rv.clone()], &HashMap::new());

        call_op_one(
            "aten::add_",
            "Tensor",
            &[t1_view.clone(), t1_view.clone()],
            &HashMap::new(),
        );
    }

    /// Test that we implicitly convert scalar args to tensors for the appropriate
    /// operations.
    #[test]
    fn call_op_implicit_scalar_to_tensor() {
        let tensor = call_op_one(
            "aten::ones",
            "",
            &[RValue::from(vec![2, 3])],
            &HashMap::new(),
        );
        call_op_one(
            "aten::add_",
            "Tensor",
            &[tensor, RValue::Int(1)],
            &HashMap::new(),
        );
    }

    #[should_panic]
    #[test]
    fn call_op_mutating_while_borrowed() {
        let rv = RValue::from(vec![2, 3]);
        let t1 = call_op_one("aten::ones", "", &[rv.clone()], &HashMap::new());
        let t1_view = call_op_one("aten::view", "", &[t1.clone(), rv.clone()], &HashMap::new());

        let t1_cell: TensorCell = t1.clone().try_into().unwrap();
        let t1_view_cell: TensorCell = t1_view.try_into().unwrap();
        assert!(t1_cell.aliases(&t1_view_cell));

        // Two threads can call non-mutating ops on the same alias with no problem.
        let handle1 = std::thread::spawn(move || {
            for _ in 0..1000 {
                call_op_one(
                    "aten::add",
                    "Tensor",
                    &[t1.clone(), t1.clone()],
                    &HashMap::new(),
                );
            }
        });

        let handle2 = std::thread::spawn(move || {
            let t1_view: RValue = t1_view_cell.clone().into();
            // Trying to mutate this tensor while it is borrowed by the first
            // thread should panic!
            for _ in 0..1000 {
                call_op_one(
                    "aten::add_",
                    "Tensor",
                    &[t1_view.clone(), t1_view.clone()],
                    &HashMap::new(),
                );
            }
        });
        handle1.join().unwrap();
        handle2.join().unwrap();
    }

    #[test]
    fn kwargs() {
        let rv = RValue::from(vec![2, 3]);
        let kwargs = HashMap::from([("size".into(), rv)]);
        let t1 = call_op_one("aten::ones", "", &[], &kwargs.clone());
        let t2 = call_op_one("aten::ones", "", &[], &kwargs);

        match (&t1, &t2) {
            (RValue::Tensor(t1), RValue::Tensor(t2)) => {
                assert!(!t1.aliases(t2));
            }
            _ => panic!("expected tensor"),
        }

        let result = call_op("aten::allclose", "", &[t1, t2], &HashMap::new(), true)
            .unwrap()
            .pop()
            .unwrap();

        assert!(
            matches!(result, RValue::Bool(true)),
            "Expected true for allclose output"
        );
    }

    #[test]
    fn kwargs_alias() {
        let rv = RValue::from(vec![2, 3]);
        let kwargs = HashMap::from([("size".into(), rv.clone())]);
        let t1 = call_op_one("aten::ones", "", &[], &kwargs);

        let kwargs = HashMap::from([("size".into(), rv.clone()), ("self".into(), t1.clone())]);
        let t1_view = call_op_one("aten::view", "", &[], &kwargs);

        let t1_cell: TensorCell = t1.clone().try_into().unwrap();
        let t1_view_cell: TensorCell = t1_view.try_into().unwrap();
        assert!(t1_cell.aliases(&t1_view_cell));
    }

    #[test]
    fn kwargs_chunk_alias() {
        let size = RValue::from(vec![2, 3]);
        let kwargs = HashMap::from([("size".into(), size)]);
        let t1 = call_op_one("aten::ones", "", &[], &kwargs);

        let kwargs = HashMap::from([("self".into(), t1.clone()), ("chunks".into(), 2.into())]);
        let chunked = call_op_one("aten::chunk", "", &[], &kwargs);

        let cells: Vec<TensorCell> = chunked
            .try_into()
            .expect("return of chunk should be a tensor list");

        let original_cell: TensorCell = t1.try_into().expect("return of ones should be a tensor");
        for cell in cells {
            assert!(original_cell.aliases(&cell));
        }
    }

    #[should_panic]
    #[test]
    fn kwargs_mutate_double_borrow() {
        let size = RValue::from(vec![2, 3]);
        let kwargs = HashMap::from([("size".into(), size)]);
        let t1 = call_op_one("aten::ones", "", &[], &kwargs.clone());
        let t1_view = call_op_one("aten::view", "", &[t1.clone()], &kwargs);
        let t1_view_cell: TensorCell = t1_view.try_into().unwrap();

        let handle1 = std::thread::spawn(move || {
            for _ in 0..1000 {
                let kwargs = HashMap::from([("size".into(), t1.clone())]);
                call_op_one("aten::add", "Tensor", &[t1.clone()], &kwargs);
            }
        });

        let handle2 = std::thread::spawn(move || {
            let t1_view: RValue = t1_view_cell.into();
            // Trying to mutate this tensor while it is borrowed by the first
            // thread should panic!
            for _ in 0..1000 {
                let kwargs = HashMap::from([("size".into(), t1_view.clone())]);
                call_op_one("aten::add_", "Tensor", &[t1_view.clone()], &kwargs);
            }
        });
        handle1.join().unwrap();
        handle2.join().unwrap();
    }

    #[test]
    fn test_flatten_results() {
        let size = RValue::from(vec![5, 2]);
        let kwargs = HashMap::from([("size".into(), size)]);
        let t = call_op_one("aten::ones", "", &[], &kwargs.clone());
        let res = call_op(
            "aten::split_with_sizes",
            "",
            &[t, vec![1, 4].into()],
            &HashMap::new(),
            true,
        )
        .unwrap();
        assert_eq!(res.len(), 2);
        match (&res[0], &res[1]) {
            (RValue::Tensor(t1), RValue::Tensor(t2)) => {
                assert_eq!(t1.borrow().numel(), 2);
                assert_eq!(t2.borrow().numel(), 8);
            }
            _ => panic!("unexpected results: {:?}", res),
        }
    }

    #[test]
    fn test_call_op_mutating_self_with_no_return() {
        let t: TensorCell = call_op_one("aten::ones", "", &[vec![5, 1].into()], &HashMap::new())
            .try_into()
            .unwrap();
        let res = call_op(
            "aten::_foreach_add_",
            "Scalar",
            &[vec![t.clone()].into(), 1.into()],
            &HashMap::new(),
            true,
        )
        .unwrap();
        assert_eq!(res.len(), 0);
        let expected: TensorCell = call_op_one(
            "aten::full",
            "",
            &[vec![5, 1].into(), 2.into()],
            &HashMap::new(),
        )
        .try_into()
        .unwrap();
        assert!(t.borrow().equal(&expected.borrow()));
    }

    #[test]
    fn test_call_op_arg_types() {
        let args_info =
            crate::bridge::ffi::get_schema_args_info("aten::_foreach_add_", "Scalar").unwrap();
        assert_eq!(args_info.len(), 2);
        assert_eq!(args_info[0].name, "self");
        assert!(args_info[0].is_mutable);
        assert!(args_info[0].type_.is_tensor_list());
        assert!(!args_info[0].type_.is_tensor());
        assert_eq!(args_info[1].name, "scalar");
        assert!(!args_info[1].is_mutable);
        assert!(!args_info[1].type_.is_tensor());
        assert!(!args_info[1].type_.is_tensor_list());

        let args_info =
            crate::bridge::ffi::get_schema_args_info("aten::_foreach_add_", "Tensor").unwrap();
        assert_eq!(args_info[1].name, "other");
        assert!(!args_info[1].is_mutable);
        assert!(args_info[1].type_.is_tensor());
        assert!(!args_info[1].type_.is_tensor_list());
    }
}
