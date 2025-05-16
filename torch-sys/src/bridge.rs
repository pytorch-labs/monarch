use std::fmt::Debug;
use std::fmt::Error;
use std::fmt::Formatter;

use crate::IValue;
use crate::backend::BoxedBackend;
use crate::backend::BoxedWork;

#[allow(dead_code)]
#[cxx::bridge(namespace = "monarch")]
pub(crate) mod ffi {
    // These are here to instruct the CXX codegen to generate concrete
    // specializations of `rust::Vec` for these types in C++.
    // Otherwise, you'll get a linker error about undefined symbols.
    impl Vec<IValue> {}
    impl Vec<Tensor> {}
    impl Box<BoxedBackend> {}
    impl Box<BoxedWork> {}

    #[derive(Debug)]
    enum AliasKind {
        NewValue,
        Alias,
    }

    #[derive(Debug)]
    struct AliasInfo {
        kind: AliasKind,
        arg_idx: usize,
        arg_name: String,
    }

    #[derive(Debug)]
    struct OpCallResult {
        outputs: Vec<IValue>,
        alias_infos: Vec<AliasInfo>,
    }

    #[derive(Debug)]
    struct SchemaArgInfo<'a> {
        name: String,
        is_mutable: bool,
        type_: &'a TypePtr,
        num_elements: i32,
        // This is an operator level flag but we have it in schema arg
        // at the moment. This should be moved to an operator level
        // struct in future especially when we have more fields.
        allows_number_as_tensor: bool,
    }

    #[derive(Debug)]
    struct Kwarg {
        name: String,
        arg: IValue,
    }

    #[derive(Debug)]
    enum ReduceOp {
        Sum = 0,
        Avg = 1,
        Max = 2,
        Min = 3,
    }

    #[derive(Debug)]
    struct AllreduceOptions {
        reduce_op: ReduceOp,
    }

    #[derive(Debug)]
    struct BarrierOptions {
        timeout: usize, // milliseconds
    }

    #[derive(Debug)]
    struct ReduceOptions {
        reduce_op: ReduceOp,
        root_rank: i32,
    }

    #[derive(Debug)]
    struct ReduceScatterOptions {
        reduce_op: ReduceOp,
    }

    #[derive(Debug)]
    struct GatherOptions {
        root_rank: i32,
        timeout: usize, // milliseconds
    }

    #[derive(Debug)]
    struct ScatterOptions {
        root_rank: i32,
        timeout: usize, // milliseconds
    }

    #[derive(Debug)]
    struct BroadcastOptions {
        root_rank: i32,
    }

    #[derive(Debug)]
    struct AllToAllOptions {
        timeout: usize, // milliseconds
    }

    extern "Rust" {
        type BoxedWork;
        fn wait(&self) -> Result<()>;
        fn is_completed(&self) -> Result<bool>;
    }

    extern "Rust" {
        // We use `BoxedBackend` as the bridge to the our C++ backend impl.
        // TODO(agallagher): Fill this out.
        type BoxedBackend;
        fn allreduce(
            &self,
            tensors: &CxxVector<Tensor>,
            opts: AllreduceOptions,
        ) -> Result<Box<BoxedWork>>;
        fn allgather(&self, output: &CxxVector<Tensor>, input: &Tensor) -> Result<Box<BoxedWork>>;
        fn _allgather_base(&self, output: &Tensor, input: &Tensor) -> Result<Box<BoxedWork>>;
        fn barrier(&self, opts: BarrierOptions) -> Result<Box<BoxedWork>>;
        fn reduce(&self, input: &Tensor, opts: ReduceOptions) -> Result<Box<BoxedWork>>;
        fn _reduce_scatter_base(
            &self,
            output: &Tensor,
            input: &Tensor,
            opts: ReduceScatterOptions,
        ) -> Result<Box<BoxedWork>>;
        fn send(
            &self,
            tensors: &CxxVector<Tensor>,
            dst_rank: i32,
            tag: i32,
        ) -> Result<Box<BoxedWork>>;
        fn recv(
            &self,
            tensors: &CxxVector<Tensor>,
            src_rank: i32,
            tag: i32,
        ) -> Result<Box<BoxedWork>>;
        fn gather(
            &self,
            outputs: &CxxVector<Tensor>,
            input: &Tensor,
            opts: GatherOptions,
        ) -> Result<Box<BoxedWork>>;
        fn scatter(
            &self,
            output: &Tensor,
            inputs: &CxxVector<Tensor>,
            opts: ScatterOptions,
        ) -> Result<Box<BoxedWork>>;
        fn broadcast(
            &self,
            tensors: &CxxVector<Tensor>,
            opts: BroadcastOptions,
        ) -> Result<Box<BoxedWork>>;
        fn alltoall_base(
            &self,
            output_buffer: &Tensor,
            input_buffer: &Tensor,
            opts: AllToAllOptions,
        ) -> Result<Box<BoxedWork>>;
        fn alltoall(
            &self,
            output_tensors: &CxxVector<Tensor>,
            input_tensors: &CxxVector<Tensor>,
            opts: AllToAllOptions,
        ) -> Result<Box<BoxedWork>>;
    }

    unsafe extern "C++" {
        ///////////////////////////////////////////////////////////////////////
        /// NOTE: If you are implementing a new binding, please review the
        /// safety discussion in `lib.rs`. Then, please include a "Safety"
        /// section in your docblock, discussing how mutability/aliasing
        /// restrictions apply to your binding.
        ///////////////////////////////////////////////////////////////////////
        include!("ATen/cuda/CUDAEvent.h");
        include!("torch/csrc/distributed/c10d/Types.hpp");
        include!("monarch/torch-sys/src/bridge.h");
        #[namespace = "c10"]
        type IValue = crate::IValue;
        #[namespace = "torch"]
        type Tensor = crate::Tensor;
        #[namespace = "c10"]
        type Device = crate::Device;
        #[namespace = "c10"]
        type MemoryFormat = crate::MemoryFormat;
        #[namespace = "c10"]
        type Layout = crate::Layout;
        #[namespace = "monarch"]
        type FFIPyObject = crate::pyobject::FFIPyObject;
        #[namespace = "c10"]
        type TypePtr = crate::call_op::TypePtr;

        // Creates a Python callback to be passed to `Backend.register_backend`.
        fn create_monarch_backend() -> FFIPyObject;
        fn create_null_backend() -> FFIPyObject;

        // Device
        fn device_from_py_object(obj: FFIPyObject) -> Result<Device>;
        fn device_to_py_object(device: Device) -> FFIPyObject;

        // Layout
        fn layout_from_py_object(obj: FFIPyObject) -> Result<Layout>;
        fn layout_to_py_object(layout: Layout) -> FFIPyObject;

        // MemoryFormat
        fn memory_format_from_py_object(obj: FFIPyObject) -> Result<MemoryFormat>;
        fn memory_format_to_py_object(memory_format: MemoryFormat) -> FFIPyObject;

        // Tensor
        fn tensor_from_py_object(obj: FFIPyObject) -> Result<Tensor>;
        fn tensor_to_py_object(tensor: Tensor) -> FFIPyObject;

        // Methods on Tensor
        fn device(self: &Tensor) -> Device;
        fn scalar_type(self: &Tensor) -> ScalarType;
        fn is_cuda(self: &Tensor) -> bool;
        fn cpu(self: &Tensor) -> Tensor;
        fn is_sparse(self: &Tensor) -> bool;
        fn is_contiguous(self: &Tensor, memory_format: MemoryFormat) -> bool;
        fn numel(self: &Tensor) -> i64;
        fn nbytes(self: &Tensor) -> usize;
        fn suggest_memory_format(t: &Tensor) -> MemoryFormat;
        fn equal(self: &Tensor, other: &Tensor) -> bool;
        fn defined(self: &Tensor) -> bool;

        /// binding for `torch.zeros`
        fn factory_zeros(
            sizes: &[i64],
            dtype: ScalarType,
            layout: Layout,
            device: Device,
        ) -> Tensor;
        /// binding for `torch.empty`
        fn factory_empty(
            sizes: &[i64],
            dtype: ScalarType,
            layout: Layout,
            device: Device,
        ) -> Tensor;
        /// Creates a new one-dimensional f32 Tensor with the provided data.
        /// Mostly used for testing; basically equivalent to a limited version
        /// of the raw `torch.tensor` constructor.
        fn factory_float_tensor(data: &[f32], device: Device) -> Tensor;
        /// Return a clone of this tensor. The semantics of clone are like
        /// `torch.clone`: it will copy the the underlying tensor storage.
        ///
        /// # Safety
        /// This function is guaranteed to produce a fresh (non-aliasing) tensor.
        fn deep_clone(t: &Tensor) -> Tensor;

        /// Bindings for `load`/`save` for `Tensor`.
        fn load_tensor(buf: &[u8]) -> Result<Tensor>;
        fn save_tensor(tensor: &Tensor) -> Result<Vec<u8>>;

        fn copy_(tensor: &mut Tensor, src: &Tensor);
        fn sizes(tensor: &Tensor) -> Vec<i32>;

        // ScalarType
        #[namespace = "c10"]
        type ScalarType = crate::ScalarType;
        #[namespace = "at"]
        #[rust_name = "is_float8_type"]
        fn isFloat8Type(t: ScalarType) -> bool;

        // Convert to Python object.
        fn scalar_type_from_py_object(obj: FFIPyObject) -> Result<ScalarType>;
        fn scalar_type_to_py_object(scalar_type: ScalarType) -> FFIPyObject;

        /// # Safety
        /// - **Mutability**:
        /// `call_op` may mutate the provided arguments (for example, if you
        /// called `aten::add_`), so `args` and `kwargs` require a mutable slice.
        ///
        /// - **Aliasing**:
        /// `call_op` may return aliases of the provided arguments, so it is
        /// marked as `unsafe`. The caller is responsible for using the aliasing
        /// info returned by `call_op` to ensure that Rust's aliasing rules are
        /// respected.
        //
        // TODO this fn ends up making a bunch of small copies to marshall
        // arguments across the FFI boundary. This could probably be improved,
        // at the cost of a less straightforward calling convention.
        unsafe fn call_op_raw(
            op_name: &str,
            overload: &str,
            args: &mut [IValue],
            kwargs: &mut [Kwarg],
            flatten_results: bool,
        ) -> Result<OpCallResult>;

        /// Give information about which arguments can be mutated by the
        /// provided operator.
        /// TODO:
        ///   - This returns results for all arguments, even ones not provided
        ///     by the caller.
        fn get_schema_args_info<'a>(
            op_name: &'a str,
            overload: &'a str,
        ) -> Result<Vec<SchemaArgInfo<'a>>>;

        // Constructors for IValue
        fn ivalue_from_int(val: i64) -> IValue;
        fn ivalue_from_int_list(val: &[i64]) -> IValue;
        fn ivalue_from_double(val: f64) -> IValue;
        fn ivalue_from_bool(val: bool) -> IValue;
        fn ivalue_from_string(val: &String) -> IValue;
        fn ivalue_from_tensor(val: Tensor) -> IValue;
        fn ivalue_from_tensor_list(val: Vec<Tensor>) -> IValue;
        fn ivalue_from_device(val: Device) -> IValue;
        fn ivalue_from_layout(val: Layout) -> IValue;
        fn ivalue_from_scalar_type(val: ScalarType) -> IValue;
        fn ivalue_from_none() -> IValue;

        // Interop with Python object.
        fn arbitrary_ivalue_to_py_object(val: IValue) -> Result<FFIPyObject>;
        fn ivalue_from_arbitrary_py_object(obj: FFIPyObject) -> Result<IValue>;
        /// Converts the provided Python object to an `IValue` with the provided
        /// type. If the object is not convertible to the provided type, an
        /// exception will be thrown.
        fn ivalue_from_py_object_with_type(
            obj: FFIPyObject,
            type_: &TypePtr,
            num_elements: i32,
            allow_nums_as_tensors: bool,
        ) -> Result<IValue>;

        // Equality
        /// Allows comparing ivalues for equality using `operator==` on `IValue`.
        fn ivalues_equal_operator(a: &IValue, b: &IValue) -> bool;

        // Serde for IValue
        fn serialize_ivalue(val: &IValue) -> Result<Vec<u8>>;
        fn deserialize_ivalue(buf: &[u8]) -> Result<IValue>;

        /// Clones the `IValue` with copying data over. Can throw an exception
        /// if the `IValue` is not cloneable.
        fn ivalue_deepcopy(iv: &IValue) -> Result<IValue>;

        // These are methods on the C++ IValue type.
        /// Prints a human-readable representation of the `IValue` to stdout.
        fn dump(self: &IValue) -> ();

        #[doc(hidden)]
        fn isBool(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toBool(self: &IValue) -> Result<bool>;

        #[doc(hidden)]
        fn isInt(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toInt(self: &IValue) -> Result<i64>;

        #[doc(hidden)]
        fn isDouble(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toDouble(self: &IValue) -> Result<f64>;

        #[doc(hidden)]
        fn isIntList(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toIntList(iv: &IValue) -> Result<Vec<i64>>;

        #[doc(hidden)]
        fn isString(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toString(iv: &IValue) -> Result<String>;

        #[doc(hidden)]
        fn isTensor(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toTensor(iv: IValue) -> Result<Tensor>;

        #[doc(hidden)]
        fn isTensorList(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toTensorList(iv: IValue) -> Result<Vec<Tensor>>;

        #[doc(hidden)]
        fn isDevice(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toDevice(self: &IValue) -> Result<Device>;

        //#[doc(hidden)]
        //fn isLayout(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toLayout(self: &IValue) -> Result<Layout>;

        //#[doc(hidden)]
        //fn isScalarType(self: &IValue) -> bool;
        #[doc(hidden)]
        fn toScalarType(self: &IValue) -> Result<ScalarType>;

        #[doc(hidden)]
        fn isNone(self: &IValue) -> bool;
        //#[doc(hidden)]
        //fn toNone(self: &IValue) -> Result<String>;

        // TODO: support the rest of IValues stuff

        // Utility functions on TypePtr
        fn type_ptr_is_tensor(t: &TypePtr) -> bool;
        fn type_ptr_is_tensor_list(t: &TypePtr) -> bool;
        fn type_ptr_is_optional_tensor(t: &TypePtr) -> bool;
        fn type_ptr_is_optional_tensor_list(t: &TypePtr) -> bool;

        // Helpers for debugging
        fn debug_type_str(val: &IValue) -> Result<String>;
        fn debug_print(val: &IValue) -> Result<String>;

        #[namespace = "monarch::test"]
        fn test_make_undefined_tensor_ivalue() -> IValue;

        #[namespace = "monarch::test"]
        fn test_make_opaque_ivalue() -> IValue;

        #[namespace = "monarch::test"]
        fn test_make_tensor() -> Tensor;

        #[namespace = "monarch::test"]
        fn cuda_full(size: &[i64], value: f32) -> Tensor;

        #[namespace = "monarch::test"]
        unsafe fn test_make_alias(t: &Tensor) -> Tensor;

        #[namespace = "monarch::test"]
        fn allclose(a: &Tensor, b: &Tensor) -> Result<bool>;

        #[namespace = "monarch::test"]
        fn repr(t: &Tensor) -> String;

        #[namespace = "monarch::test"]
        fn stack(tensor: &[Tensor]) -> Tensor;

        fn is_alias(lhs: &Tensor, rhs: &Tensor) -> bool;

        // CUDA Event APIs
        #[namespace = "at::cuda"]
        type CUDAEvent;
        fn create_cuda_event(
            enable_timing: bool,
            blocking: bool,
            interprocess: bool,
        ) -> UniquePtr<CUDAEvent>;
        fn record(self: Pin<&mut CUDAEvent>, stream: &CUDAStream);
        fn block(self: Pin<&mut CUDAEvent>, stream: &CUDAStream);
        fn query(self: &CUDAEvent) -> bool;
        fn elapsed_time(self: &CUDAEvent, end_event: &CUDAEvent) -> f32;
        fn synchronize(self: &CUDAEvent);

        // CUDA Stream APIs
        #[namespace = "c10::cuda"]
        type CUDAStream;
        #[namespace = ""]
        type CUstream_st = nccl_sys::CUstream_st;
        fn get_current_stream(device: i8) -> SharedPtr<CUDAStream>;
        fn set_current_stream(stream: &CUDAStream);
        fn create_stream(device: i8, priority: i32) -> SharedPtr<CUDAStream>;
        fn query(self: &CUDAStream) -> bool;
        fn synchronize(self: &CUDAStream);
        fn device_index(self: &CUDAStream) -> i8;
        fn stream(self: &CUDAStream) -> *mut CUstream_st;

        // nccl helpers
        #[namespace = ""]
        type ncclConfig_t = nccl_sys::ncclConfig_t;
        fn make_nccl_config() -> ncclConfig_t;
    }

    // Allow accessing `Tensor` from `CxxVector` in the `BoxedBackend` impl.
    impl CxxVector<Tensor> {}
}

// SAFETY: CUDAStream is thread safe
unsafe impl Send for ffi::CUDAStream {}
// SAFETY: see above
unsafe impl Sync for ffi::CUDAStream {}

impl Debug for ffi::CUDAStream {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("CUDAStream")
            .field("device", &format!("{}", self.device_index()))
            .field("stream", &format!("{:p}", self))
            .finish()
    }
}

impl Debug for ffi::CUDAEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_struct("CUDAEvent")
            .field("ptr", &format!("{:p}", self))
            .finish()
    }
}

// SAFETY: CUDAEvent is thread safe. The comments on `c10::Event` say it isn't, but in
// Rust we would consider it Sync because shared references are fine to access
// across threads.
unsafe impl Send for ffi::CUDAEvent {}
// SAFETY: see above
unsafe impl Sync for ffi::CUDAEvent {}

extern "C" {
    pub(crate) fn cpp_decref(ptr: *mut std::ffi::c_void);
    pub(crate) fn drop(this: *mut IValue);
    pub(crate) fn clone_iv(this: *const IValue, new: *mut IValue);
    pub(crate) fn cpp_incref(ptr: *mut std::ffi::c_void);

    pub(crate) fn const_data_ptr(tensor: *mut std::ffi::c_void) -> *const std::ffi::c_void;
    pub(crate) fn mut_data_ptr(tensor: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
}
