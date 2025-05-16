// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <ATen/cuda/CUDAEvent.h> // @manual=//caffe2:torch-cpp
#include <nccl.h> // @manual
#include <rust/cxx.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/torch.h> // @manual=//caffe2:torch-cpp
#include "monarch/torch-sys/src/bridge.rs.h"

// Attest that c10::IValue is trivially relocatable; that is, it can be moved by
// just doing a memcpy.
//
// IValue contains a tag and a payload.
//   - The tag is trivially copyable, which implies that it is trivially
//     relocatable.
//   - The payload is either trivially copyable OR a Tensor.
//
// So, is Tensor trivially relocatable? Yes. It only holds a
// `c10::intrusive_ptr<TensorImpl>`. `intrusive_ptr` is trivially relocatable.
// The logic is similar to why `shared_ptr` is trivially relocatable:
//    - Moving does not bump the refcount, so there is no special relocation
//      handling beyond what memcpy would do anyway.
//    - Its methods/destructor do not depend on the intrusive_ptr's memory
//      location.
template <>
struct rust::IsRelocatable<c10::IValue> : std::true_type {};
template <>
struct rust::IsRelocatable<at::Tensor> : std::true_type {};

// Enforce that alignment and size of these types are what we expect, as they
// are bound to Rust.
static_assert(sizeof(c10::IValue) == 2 * sizeof(void*), "");
static_assert(alignof(c10::IValue) == alignof(void*), "");
static_assert(sizeof(at::Tensor) == sizeof(void*), "");
static_assert(alignof(at::Tensor) == alignof(void*), "");

namespace monarch {
using Tensor = torch::Tensor;
using IValue = c10::IValue;
using FFIPyObject = PyObject*;
using TypePtr = c10::TypePtr;

OpCallResult call_op_raw(
    rust::Str ns,
    rust::Str op,
    rust::Slice<IValue> args,
    rust::Slice<Kwarg> kwargs,
    bool flatten_results);
rust::Vec<SchemaArgInfo> get_schema_args_info(
    rust::Str name,
    rust::Str overload_name);

rust::String debug_print(const IValue& iv);

inline Tensor deep_clone(const Tensor& t) {
  return t.clone();
}

template <typename T>
rust::Vec<uint8_t> save(const T& t) {
  rust::Vec<uint8_t> out;
  ::torch::save(t, [&](const void* b, size_t len) {
    const uint8_t* buf = reinterpret_cast<const uint8_t*>(b);
    out.reserve(len);
    std::copy(buf, buf + len, std::back_inserter(out));
    return len;
  });
  return out;
}

rust::Vec<uint8_t> save_tensor(const Tensor& t);

template <typename T>
T load(rust::Slice<const uint8_t> buf) {
  T tensor;
  torch::load(tensor, reinterpret_cast<const char*>(buf.data()), buf.size());
  return tensor;
}

Tensor load_tensor(rust::Slice<const uint8_t> buf);

void copy_(Tensor& tensor, const Tensor& src);
rust::Vec<int32_t> sizes(const Tensor& tensor);

FFIPyObject arbitrary_ivalue_to_py_object(IValue val);
IValue ivalue_from_arbitrary_py_object(FFIPyObject obj);

inline IValue ivalue_from_py_object_with_type(
    FFIPyObject obj,
    const TypePtr& t,
    int32_t num_elements,
    bool allow_nums_as_tensors) {
  auto pyObj = py::reinterpret_steal<py::object>(obj);
  torch::jit::ToIValueAllowNumbersAsTensors g(allow_nums_as_tensors);
  return torch::jit::toIValue(
      pyObj.ptr(),
      t,
      num_elements >= 0 ? std::optional(num_elements) : std::nullopt);
}

c10::Device device_from_py_object(FFIPyObject obj);
FFIPyObject device_to_py_object(c10::Device device);

c10::ScalarType scalar_type_from_py_object(FFIPyObject obj);
FFIPyObject scalar_type_to_py_object(c10::ScalarType scalar_type);

c10::Layout layout_from_py_object(FFIPyObject obj);
FFIPyObject layout_to_py_object(c10::Layout layout);

c10::MemoryFormat memory_format_from_py_object(FFIPyObject obj);
FFIPyObject memory_format_to_py_object(c10::MemoryFormat memory_format);

FFIPyObject tensor_to_py_object(Tensor tensor);
Tensor tensor_from_py_object(FFIPyObject obj);

inline Tensor factory_zeros(
    rust::Slice<const int64_t> sizesRs,
    c10::ScalarType dtype,
    c10::Layout layout,
    c10::Device device) {
  std::vector<int64_t> sizes;
  std::copy(sizesRs.begin(), sizesRs.end(), std::back_inserter(sizes));
  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).layout(layout).device(device);
  return torch::zeros(sizes, options);
}

inline Tensor factory_empty(
    rust::Slice<const int64_t> sizesRs,
    c10::ScalarType dtype,
    c10::Layout layout,
    c10::Device device) {
  std::vector<int64_t> sizes;
  std::copy(sizesRs.begin(), sizesRs.end(), std::back_inserter(sizes));
  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).layout(layout).device(device);
  return torch::empty(sizes, options);
}

inline Tensor factory_float_tensor(
    rust::Slice<const float> dataRs,
    c10::Device device) {
  std::vector<float> data;
  std::copy(dataRs.begin(), dataRs.end(), std::back_inserter(data));
  torch::TensorOptions options = torch::TensorOptions().device(device);
  return torch::tensor(data, options);
}

inline bool is_alias(const Tensor& lhs, const Tensor& rhs) {
  return lhs.is_alias_of(rhs);
}

inline IValue ivalue_from_int(int64_t value) noexcept {
  return IValue(value);
}

inline IValue ivalue_from_int_list(rust::Slice<const int64_t> value) noexcept {
  std::vector<int64_t> ret;
  std::copy(value.begin(), value.end(), std::back_inserter(ret));
  return IValue(ret);
}

inline IValue ivalue_from_double(double value) noexcept {
  return IValue(value);
}

inline IValue ivalue_from_string(const rust::String& value) noexcept {
  return IValue(std::string(value));
}

inline IValue ivalue_from_none() noexcept {
  return IValue();
}

inline IValue ivalue_from_bool(bool value) noexcept {
  return IValue(value);
}

inline IValue ivalue_from_tensor(Tensor value) noexcept {
  return IValue(value);
}

inline IValue ivalue_from_tensor_list(rust::Vec<Tensor> value) noexcept {
  c10::List<Tensor> ret;
  std::copy(value.begin(), value.end(), std::back_inserter(ret));
  return IValue(ret);
}

inline IValue ivalue_from_device(c10::Device value) noexcept {
  return IValue(value);
}

inline IValue ivalue_from_layout(at::Layout value) noexcept {
  return IValue(value);
}

inline IValue ivalue_from_scalar_type(at::ScalarType value) noexcept {
  return IValue(value);
}

inline bool ivalues_equal_operator(const IValue& a, const IValue& b) {
  return a == b;
}

rust::Vec<uint8_t> serialize_ivalue(const IValue& iv);

IValue deserialize_ivalue(rust::Slice<const uint8_t> buf);

inline IValue ivalue_deepcopy(const IValue& iv) {
  return iv.deepcopy();
}

inline bool isBool(const IValue& iv) {
  return iv.isBool();
}

inline bool toBool(const IValue& iv) {
  return iv.toBool();
}

inline bool isInt(const IValue& iv) {
  return iv.isInt();
}

inline bool isTensorList(const IValue& iv) {
  return iv.isTensorList();
}

rust::Vec<Tensor> toTensorList(IValue iv);

inline int64_t toInt(const IValue& iv) {
  return iv.toInt();
}

inline rust::Vec<int64_t> toIntList(const IValue& iv) {
  rust::Vec<int64_t> ret;
  const auto& cppVec = iv.toIntVector();
  std::copy(cppVec.cbegin(), cppVec.cend(), std::back_inserter(ret));
  return ret;
}

inline rust::String toString(const IValue& iv) {
  return rust::String(iv.toStringRef());
}

inline at::Tensor toTensor(IValue iv) {
  return iv.toTensor();
}

inline rust::String debug_type_str(const IValue& iv) {
  return rust::String(iv.type()->str());
}

inline at::MemoryFormat suggest_memory_format(const at::Tensor& t) {
  return t.suggest_memory_format();
}

inline bool type_ptr_is_tensor(const TypePtr& type) {
  return type->kind() == c10::TypeKind::TensorType;
}

inline bool type_ptr_is_tensor_list(const TypePtr& type) {
  return type->kind() == c10::TypeKind::ListType &&
      type->expectRef<c10::ListType>().getElementType()->kind() ==
      c10::TypeKind::TensorType;
}

inline bool type_ptr_is_optional_tensor(const TypePtr& type) {
  return type->kind() == c10::TypeKind::OptionalType &&
      type_ptr_is_tensor(type->expectRef<c10::OptionalType>().getElementType());
}

inline bool type_ptr_is_optional_tensor_list(const TypePtr& type) {
  return type->kind() == c10::TypeKind::OptionalType &&
      type_ptr_is_tensor_list(
             type->expectRef<c10::OptionalType>().getElementType());
}

std::unique_ptr<at::cuda::CUDAEvent>
create_cuda_event(bool enable_timing, bool blocking, bool interprocess);

std::shared_ptr<c10::cuda::CUDAStream> get_current_stream(
    c10::DeviceIndex device);

std::shared_ptr<c10::cuda::CUDAStream> create_stream(
    c10::DeviceIndex device,
    int32_t priority);

void set_current_stream(const c10::cuda::CUDAStream& stream);

extern "C" {
void cpp_incref(void* ptr);
void cpp_decref(void* ptr);
void drop(IValue* self);
void clone_iv(const IValue* self, IValue* result) noexcept;
const void* const_data_ptr(void* ptr);
void* mut_data_ptr(void* ptr);
}

namespace test {
inline IValue test_make_undefined_tensor_ivalue() {
  return IValue(at::Tensor());
}

inline IValue test_make_opaque_ivalue() {
  return IValue(c10::complex<double>(1, 2));
}

inline Tensor test_make_tensor() {
  return at::rand({2, 2});
}

inline Tensor test_make_alias(const Tensor& t) {
  return t.view({2, 2});
}

inline Tensor cuda_full(rust::Slice<const int64_t> sizesRs, float value) {
  std::vector<int64_t> sizes;
  std::copy(sizesRs.begin(), sizesRs.end(), std::back_inserter(sizes));
  return at::full(sizes, value).cuda();
}

inline bool allclose(const Tensor& a, const Tensor& b) {
  return a.allclose(b);
}

inline rust::String repr(const Tensor& a) {
  std::stringstream ss;
  ss << a;
  return rust::String(ss.str());
}

inline Tensor stack(rust::Slice<const Tensor> tensor_list) {
  std::vector<Tensor> tensors;
  std::copy(
      tensor_list.begin(), tensor_list.end(), std::back_inserter(tensors));
  return torch::stack(tensors);
}
} // namespace test

FFIPyObject create_monarch_backend();
FFIPyObject create_null_backend();

/// This function exists because ncclConfig initialization requires the use of
/// a macro. We cannot reference the macro directly from Rust code, so we wrap
/// the macro use in a function and bind that to Rust instead.
inline ncclConfig_t make_nccl_config() {
  ncclConfig_t ret = NCCL_CONFIG_INITIALIZER;
  return ret;
}

} // namespace monarch
