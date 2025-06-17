/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "monarch/torch-sys/src/bridge.h"

#include <torch/csrc/distributed/c10d/Backend.hpp> // @manual=//caffe2:torch-cpp
#include <torch/csrc/jit/serialization/export.h> // @manual=//caffe2:torch-cpp
#include <torch/csrc/jit/serialization/pickle.h> // @manual=//caffe2:torch-cpp

namespace monarch {

namespace {
struct OperatorInfo {
  c10::OperatorHandle handle;
  std::unordered_map<c10::Symbol, std::vector<size_t>> inputSets;
  std::unordered_map<c10::Symbol, std::vector<size_t>> outputSets;
  std::optional<size_t> chunkLikeIdx;
  bool allowNumbersAsTensors;
  bool resultsNeedFlattening;
  std::vector<SchemaArgInfo> argsInfo;

  explicit OperatorInfo(const c10::OperatorName& op)
      : handle(getSchemaOrThrow(op)),
        allowNumbersAsTensors(opAllowsNumbersAsTensors(op.name)),
        resultsNeedFlattening(getResultsNeedFlattening()) {
    auto& schema = handle.schema();
    for (size_t i = 0; i < schema.arguments().size(); ++i) {
      const auto& arg = schema.arguments()[i];
      auto aliasInfo = arg.alias_info();

      argsInfo.push_back({
          .name = arg.name(),
          .is_mutable = aliasInfo && aliasInfo->isWrite(),
          .type_ = arg.type(),
          .num_elements = arg.N().value_or(-1),
          .allows_number_as_tensor = allowNumbersAsTensors,
      });

      if (!aliasInfo) {
        continue;
      }

      // For list types its the actual contained type that is being
      // mutated without need to track alias.
      // We just mark that it is being mutated above.
      // We assume here that the aliased values in the list dont show up
      // else where.
      if (aliasInfo->beforeSets().empty() && arg.type() &&
          arg.type()->kind() == c10::TypeKind::ListType) {
        continue;
      }

      // TODO: this will throw if `aliasInfo` has multiple before-sets. We have
      // no ops that have this, but it is technically expressible in the
      // annotation language.
      inputSets[aliasInfo->beforeSet()].push_back(i);
      if (aliasInfo->isWildcardAfter()) {
        TORCH_CHECK(!chunkLikeIdx.has_value());
        chunkLikeIdx = i;
      }
    }

    if (chunkLikeIdx.has_value()) {
      // Do some checking that the schema confirms to what we expect out of
      // chunk-like operators.
      // 1. Single return
      TORCH_CHECK(schema.returns().size() == 1);
      const auto& ret = schema.returns()[0];
      // 2. List<Tensor>
      TORCH_CHECK(
          ret.type()->kind() == c10::TypeKind::ListType &&
          ret.type()->cast<c10::ListType>()->getElementType()->kind() ==
              c10::TypeKind::TensorType);
      // 3. With aliasing set corresponding to the correct input.
      const auto& isChunkLikeSymbol =
          schema.arguments()[*chunkLikeIdx].alias_info()->beforeSet();
      TORCH_CHECK(
          ret.alias_info() &&
          ret.alias_info()->containedTypes().at(0).beforeSet() ==
              isChunkLikeSymbol);
      return;
    }

    for (size_t i = 0; i < schema.returns().size(); ++i) {
      const auto& ret = schema.returns()[i];
      auto aliasInfo = ret.alias_info();
      if (!aliasInfo) {
        continue;
      }
      outputSets[aliasInfo->beforeSet()].push_back(i);
    }
    for (const auto& [aliasSet, outputIndices] : outputSets) {
      if (inputSets.find(aliasSet) == inputSets.end()) {
        TORCH_INTERNAL_ASSERT(
            false,
            "Alias set",
            aliasSet.toUnqualString(),
            " is not present in the inputs, but is present in the outputs");
      }
    }
  }

 private:
  bool getResultsNeedFlattening() {
    for (auto& ret : handle.schema().returns()) {
      if (ret.type()->isSubtypeOf(*c10::AnyListType::get()) ||
          ret.type()->isSubtypeOf(*c10::AnyTupleType::get()) ||
          ret.type()->kind() == c10::TypeKind::DictType) {
        return true;
      }
    }
    return false;
  }

  c10::OperatorHandle getSchemaOrThrow(const c10::OperatorName& op) {
    auto& dispatcher = c10::Dispatcher::singleton();
    const auto it = dispatcher.findSchema(op);
    if (!it.has_value()) {
      std::stringstream ss;
      ss << "Could not find schema for " << op.name << "." << op.overload_name;
      throw std::runtime_error(ss.str());
    }
    return it.value();
  }

  bool opAllowsNumbersAsTensors(const std::string& name) {
    auto symbol = c10::Symbol::fromQualString(name);
    return symbol.is_prims() || symbol.is_nvprims() ||
        (symbol.is_aten() &&
         torch::should_allow_numbers_as_tensors(symbol.toUnqualString()));
  }
};
} // namespace

static std::unordered_map<c10::OperatorName, const OperatorInfo> operatorInfos;
// We could use a better data structure if we have too many writes happening
// to the map but ideally writes should be rare after first iteration.
static std::shared_mutex operatorInfosMutex;

const OperatorInfo& getOperatorInfo(const c10::OperatorName& op) {
  {
    std::shared_lock<std::shared_mutex> readLock(operatorInfosMutex);
    auto it = operatorInfos.find(op);
    if (it != operatorInfos.end()) {
      return it->second;
    }
  }

  auto opInfo = OperatorInfo(op);
  std::unique_lock<std::shared_mutex> writeLock(operatorInfosMutex);
  auto it = operatorInfos.find(op);
  if (it == operatorInfos.end()) {
    it = operatorInfos.emplace(op, std::move(opInfo)).first;
  }
  return it->second;
}

// TODO: this is not a fully general handling of our aliasing language. It
// should cover all cases in native_functions.yaml though.
void computeAliases(
    rust::Vec<AliasInfo>& aliasInfos,
    const OperatorInfo& opInfo,
    const std::vector<IValue>& inputs,
    const std::vector<IValue>& outputs) {
  auto& schema = opInfo.handle.schema();
  TORCH_CHECK(schema.arguments().size() == inputs.size());
  TORCH_CHECK(schema.returns().size() == outputs.size());

  // If there are no alias annotations on the inputs, just return.
  if (opInfo.inputSets.empty()) {
    return;
  }

  // Special handling for chunk-like ops.
  // If we have an input with the alias annotation `(a -> *)`, then we expect a
  // "chunk-like" aliasing pattern:
  //
  //     chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]
  //
  // This pattern means that the output is a list of tensors, each of which is a
  // view into the `self` tensor.
  if (opInfo.chunkLikeIdx.has_value()) {
    auto& aliasInfo = aliasInfos[0];
    aliasInfo.kind = AliasKind::Alias;
    aliasInfo.arg_idx = *opInfo.chunkLikeIdx;
    aliasInfo.arg_name = schema.arguments()[*opInfo.chunkLikeIdx].name();
    return;
  }

  for (const auto& [aliasSet, inputIndices] : opInfo.inputSets) {
    auto it = opInfo.outputSets.find(aliasSet);
    if (it == opInfo.outputSets.end()) {
      continue;
    }

    const auto& outputsIndicesToCheck = it->second;
    for (const auto& inputIndex : inputIndices) {
      const auto& input = inputs.at(inputIndex);
      for (const auto& outputIndex : outputsIndicesToCheck) {
        const auto& output = outputs.at(outputIndex);
        if (input.isAliasOf(output)) {
          aliasInfos[outputIndex].kind = AliasKind::Alias;
          aliasInfos[outputIndex].arg_idx = inputIndex;
          aliasInfos[outputIndex].arg_name =
              schema.arguments().at(inputIndex).name();
        }
      }
    }
  }
}

void flatten_output(
    const IValue& output,
    const AliasInfo& alias_info,
    const std::function<void(const IValue&, const AliasInfo&)>& add_output) {
  if (output.isTuple()) {
    const auto& tupleElements = output.toTuple()->elements();
    for (size_t i = 0; i < tupleElements.size(); ++i) {
      const auto& element = tupleElements[i];
      flatten_output(element, alias_info, add_output);
    }
  } else if (output.isList()) {
    const auto& list = output.toList();
    for (size_t i = 0; i < list.size(); ++i) {
      const auto& element = list.get(i);
      flatten_output(element, alias_info, add_output);
    }
  } else if (output.isGenericDict()) {
    const auto& dict = output.toGenericDict();
    for (const auto& entry : dict) {
      flatten_output(entry.value(), alias_info, add_output);
    }
  } else {
    add_output(output, alias_info);
  }
}

OpCallResult call_op_raw(
    rust::Str name,
    rust::Str overload_name,
    rust::Slice<IValue> args,
    rust::Slice<Kwarg> kwargs,
    bool flatten_results) {
  auto& opInfo =
      getOperatorInfo({std::string(name), std::string(overload_name)});
  const auto& schema = opInfo.handle.schema();

  // Construct a stack, based on the schema
  std::vector<IValue> stack;
  for (size_t idx = 0; idx < args.size(); idx++) {
    const auto& arg = args[idx];
    const auto& argSchema = schema.arguments()[idx];
    if (opInfo.allowNumbersAsTensors &&
        argSchema.type()->cast<at::TensorType>() != nullptr && arg.isScalar()) {
      Tensor tensor = at::scalar_to_tensor(arg.toScalar());
      tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
      stack.emplace_back(std::move(tensor));
    } else {
      stack.push_back(arg);
    }
  }
  // Make the kwargs.
  std::unordered_map<std::string, IValue> kwargsForSchema;
  for (const auto& kwarg : kwargs) {
    kwargsForSchema[std::string(kwarg.name)] = kwarg.arg;
  }
  schema.checkAndNormalizeInputs(stack, kwargsForSchema);

  rust::Vec<AliasInfo> aliasInfos;
  // Initialize alias infos.
  for (size_t i = 0; i < schema.returns().size(); ++i) {
    AliasInfo aliasInfo;
    aliasInfo.kind = AliasKind::NewValue;
    // Initialized, but this value is meaningless.
    aliasInfo.arg_idx = 0;
    aliasInfo.arg_name = "";
    aliasInfos.push_back(aliasInfo);
  }

  if (opInfo.inputSets.empty()) {
    opInfo.handle.callBoxed(stack);
  } else {
    const auto stackCopy = stack;
    opInfo.handle.callBoxed(stack);
    computeAliases(aliasInfos, opInfo, stackCopy, stack);
  }

  if (flatten_results && opInfo.resultsNeedFlattening) {
    rust::Vec<IValue> flattened_outputs;
    rust::Vec<AliasInfo> flattened_alias_infos;
    auto add_output = [&](const IValue& iv, const AliasInfo& ai) {
      flattened_outputs.push_back(iv);
      flattened_alias_infos.push_back(ai);
    };
    for (int i = 0; i < stack.size(); ++i) {
      const auto& output = stack[i];
      const auto& alias_info = aliasInfos[i];
      flatten_output(output, alias_info, add_output);
    }
    return OpCallResult{
        .outputs = std::move(flattened_outputs),
        .alias_infos = std::move(flattened_alias_infos)};
  }

  rust::Vec<IValue> outputs;
  outputs.reserve(stack.size());
  std::copy(stack.begin(), stack.end(), std::back_inserter(outputs));
  return OpCallResult{
      .outputs = std::move(outputs), .alias_infos = std::move(aliasInfos)};
}

rust::Vec<SchemaArgInfo> get_schema_args_info(
    rust::Str name,
    rust::Str overload_name) {
  auto& opInfo =
      getOperatorInfo({std::string(name), std::string(overload_name)});
  rust::Vec<SchemaArgInfo> res;
  res.reserve(opInfo.argsInfo.size());
  std::copy(
      opInfo.argsInfo.begin(), opInfo.argsInfo.end(), std::back_inserter(res));
  return res;
}

rust::String debug_print(const IValue& iv) {
  std::ostringstream oss;
  oss << iv;
  return rust::String(oss.str());
}

rust::Vec<Tensor> toTensorList(IValue iv) {
  rust::Vec<Tensor> ret;
  const auto& cppVec = iv.toTensorList();
  std::copy(cppVec.begin(), cppVec.end(), std::back_inserter(ret));
  return ret;
}

rust::Vec<uint8_t> save_tensor(const Tensor& t) {
  return save(t);
}

Tensor load_tensor(rust::Slice<const uint8_t> buf) {
  return load<Tensor>(buf);
}

void copy_(Tensor& tensor, const Tensor& src) {
  tensor.copy_(src);
}

rust::Vec<int32_t> sizes(const Tensor& tensor) {
  const auto sizes = tensor.sizes().vec();
  rust::Vec<int32_t> result;
  result.reserve(sizes.size());
  std::copy(sizes.begin(), sizes.end(), std::back_inserter(result));
  return result;
}

PyObject* arbitrary_ivalue_to_py_object(IValue val) {
  return torch::jit::toPyObject(std::move(val)).release().ptr();
}

IValue ivalue_from_arbitrary_py_object(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  auto inferredType = torch::jit::tryToInferType(obj);
  if (!inferredType.success()) {
    throw std::runtime_error(inferredType.reason());
  }
  // TODO(agallagher): Arbitrary Python objects -- which we can't and don't want
  // to package into an `IValue` -- will be inferred as a `Class` type.  Throw
  // here so that we'll fallback to parsing into `RValue::PyObject`.
  if (inferredType.type()->cast<at::ClassType>()) {
    throw std::runtime_error("refusing to convert class py object");
  }
  return torch::jit::toIValue(obj, inferredType.type());
}

bool py_object_is_ivalue(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  auto inferredType = torch::jit::tryToInferType(obj);
  if (!inferredType.success()) {
    return false;
  }
  // TODO(agallagher): Arbitrary Python objects -- which we can't and don't want
  // to package into an `IValue` -- will be inferred as a `Class` type.  Throw
  // here so that we'll fallback to parsing into `RValue::PyObject`.
  return !inferredType.type()->cast<at::ClassType>();
}

c10::Device device_from_py_object(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  if (!THPDevice_Check(obj.ptr())) {
    throw std::runtime_error("object is not Device");
  }
  auto device = reinterpret_cast<THPDevice*>(obj.ptr());
  return device->device;
}

PyObject* device_to_py_object(c10::Device device) {
  return THPDevice_New(device);
}

c10::ScalarType scalar_type_from_py_object(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  if (!THPDtype_Check(obj.ptr())) {
    throw std::runtime_error("object is not ScalarType");
  }
  auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
  return dtype->scalar_type;
}

PyObject* scalar_type_to_py_object(c10::ScalarType scalar_type) {
  auto dtype = torch::getTHPDtype(scalar_type);
  return Py_NewRef(dtype);
}

bool py_object_is_scalar_type(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  return THPDtype_Check(obj.ptr());
}

c10::Layout layout_from_py_object(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  if (!THPLayout_Check(obj.ptr())) {
    throw std::runtime_error("object is not Layout");
  }
  auto layout = reinterpret_cast<THPLayout*>(obj.ptr());
  return layout->layout;
}

PyObject* layout_to_py_object(c10::Layout layout) {
  auto thp_layout = torch::getTHPLayout(layout);
  return Py_NewRef(thp_layout);
}

bool py_object_is_layout(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  return THPLayout_Check(obj.ptr());
}

c10::MemoryFormat memory_format_from_py_object(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  if (!THPMemoryFormat_Check(obj.ptr())) {
    throw std::runtime_error("object is not MemoryFormat");
  }
  auto memory_format = reinterpret_cast<THPMemoryFormat*>(obj.ptr());
  return memory_format->memory_format;
}

PyObject* memory_format_to_py_object(c10::MemoryFormat memory_format) {
  auto thp_memory_format = torch::utils::getTHPMemoryFormat(memory_format);
  return Py_NewRef(thp_memory_format);
}

bool py_object_is_memory_format(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  return THPMemoryFormat_Check(obj.ptr());
}

PyObject* tensor_to_py_object(Tensor tensor) {
  torch::jit::guardAgainstNamedTensor<Tensor>(tensor);
  return py::cast(std::move(tensor)).release().ptr();
}

Tensor tensor_from_py_object(PyObject* unowned) {
  auto obj = py::reinterpret_steal<py::object>(unowned);
  if (!THPVariable_Check(obj.ptr())) {
    throw std::runtime_error("object is not Tensor");
  }
  auto var = py::cast<torch::autograd::Variable>(obj);
  torch::jit::guardAgainstNamedTensor<torch::autograd::Variable>(var);
  return var;
}

// TODO: We can do better for IValue serde as we dont need pickle compat here.
const char kIValueStart = '\x01';
const char kTensorsStart = '\x02';
const char kWrappedNumberStart = '\x03';
rust::Vec<uint8_t> serialize_ivalue(const IValue& iv) {
  if (iv.isTensor() && !iv.toTensor().defined()) {
    // Special case for undefined tensors as pickle doesnt
    // support them.
    return rust::Vec<uint8_t>{0};
  }
  std::vector<Tensor> tensors;
  auto pickle_data = torch::jit::pickle(iv, &tensors);
  rust::Vec<uint8_t> out;
  if (!tensors.empty()) {
    out.push_back(kTensorsStart);
    auto tensors_data = save(tensors);
    // Use uint32_t for tensors size
    uint32_t tensors_size = tensors_data.size();
    out.reserve(out.size() + sizeof(tensors_size) + tensors_size);
    // Serialize the size as a 4-byte integer
    for (int j = 0; j < 4; ++j) {
      out.push_back((tensors_size >> (j * 8)) & 0xFF);
    }
    std::copy(
        tensors_data.begin(), tensors_data.end(), std::back_inserter(out));
    // Tensor serialization doesn't maintain the wrapped number flag, so we
    // need to manually serialize it. This is important to maintain because
    // it has implications for the output type of torch ops.
    out.push_back(kWrappedNumberStart);
    for (size_t i = 0; i < tensors.size(); ++i) {
      uint8_t offset = i % sizeof(uint8_t);
      if (offset == 0) {
        out.push_back(0);
      }
      out.back() |= static_cast<uint8_t>(
          tensors.at(i).unsafeGetTensorImpl()->is_wrapped_number() << offset);
    }
  }
  out.push_back(kIValueStart);
  out.reserve(out.size() + pickle_data.size());
  std::copy(pickle_data.begin(), pickle_data.end(), std::back_inserter(out));
  return out;
}

IValue deserialize_ivalue(rust::Slice<const uint8_t> buf) {
  if (buf.size() == 1 && buf.at(0) == 0) {
    // Special case for undefined tensors as pickle doesnt
    // support them.
    return Tensor();
  }
  std::vector<Tensor> tensors;
  size_t i = 0;
  if (i < buf.size() && buf.at(i) == kTensorsStart) {
    i++;
    // Ensure there are enough bytes for the size
    if (i + 4 > buf.size()) {
      throw std::runtime_error(
          "Invalid IValue serialization: tensor data size truncated");
    }
    // Deserialize the size as a 4-byte integer
    uint32_t tensors_size = 0;
    for (int j = 0; j < 4; ++j) {
      tensors_size |= (static_cast<uint32_t>(buf.at(i + j)) << (j * 8));
    }
    i += 4;
    // Ensure there are enough bytes for the tensor data
    if (i + tensors_size > buf.size()) {
      throw std::runtime_error(
          "Invalid IValue serialization: tensor data truncated");
    }
    rust::Slice<const uint8_t> tensor_data(buf.data() + i, tensors_size);
    tensors = load<std::vector<Tensor>>(tensor_data);
    i += tensors_size;
    if (i >= buf.size() || buf.at(i) != kWrappedNumberStart) {
      throw std::runtime_error(
          "Invalid IValue serialization: missing wrapped number start byte");
    }
    for (size_t tensor_index = 0; tensor_index < tensors.size();
         tensor_index++) {
      uint8_t offset = tensor_index % sizeof(uint8_t);
      if (offset == 0) {
        i++;
      }
      if (i >= buf.size()) {
        throw std::runtime_error(
            "Invalid IValue serialization: wrapped number data truncated");
      }
      bool wrapped_number = (buf.at(i) >> offset) & 0x01;
      if (wrapped_number) {
        // You would think we could just call
        // set_wrapped_number(wrapped_number), but you'd be wrong. Internally,
        // set_wrapped_number asserts a 0-dim tensor regardless of whether its
        // argument is true or false, so we can only call set_wrapped_number
        // safely when wrapped_number == true.
        tensors.at(tensor_index)
            .unsafeGetTensorImpl()
            ->set_wrapped_number(true);
      }
    }
    i++;
  }
  if (i >= buf.size() || buf.at(i++) != kIValueStart) {
    throw std::runtime_error(
        "Invalid IValue serialization: missing start byte");
  }

  return torch::jit::unpickle(
      reinterpret_cast<const char*>(buf.data() + i),
      buf.size() - i,
      nullptr,
      tensors);
}

extern "C" {
void cpp_incref(void* ptr) {
  if (ptr) {
    c10::raw::intrusive_ptr::incref(
        static_cast<c10::intrusive_ptr_target*>(ptr));
  }
}

void cpp_decref(void* ptr) {
  if (ptr) {
    c10::raw::intrusive_ptr::decref(
        static_cast<c10::intrusive_ptr_target*>(ptr));
  }
}

void drop(IValue* self) {
  self->~IValue();
}

void clone_iv(const IValue* self, IValue* result) noexcept {
  new (result) IValue(*self);
}

const void* const_data_ptr(void* ptr) {
  return static_cast<c10::TensorImpl*>(ptr)->data();
}
void* mut_data_ptr(void* ptr) {
  return static_cast<c10::TensorImpl*>(ptr)->mutable_data();
}
}

namespace {

bool check_same_size(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

ReduceOp get_reduce_op(const c10d::ReduceOp::RedOpType reduceOp) {
  switch (reduceOp) {
    case c10d::ReduceOp::RedOpType::SUM:
      return ReduceOp::Sum;
    case c10d::ReduceOp::RedOpType::MAX:
      return ReduceOp::Max;
    case c10d::ReduceOp::RedOpType::MIN:
      return ReduceOp::Min;
    case c10d::ReduceOp::RedOpType::AVG:
      return ReduceOp::Avg;
    default:
      throw std::runtime_error("unsupported op");
  }
}

// An impl of `c10d::Work` to wrap and delegate to the Rust side to support
// async backend operations, via ffi through `BoxedWork`.
class MonarchWorkWrapper : public c10d::Work {
 public:
  explicit MonarchWorkWrapper(rust::Box<BoxedWork>&& work)
      : Work(), work_(std::move(work)) {}

  ~MonarchWorkWrapper() override = default;

  bool isCompleted() override {
    return work_->is_completed();
  }

  bool isSuccess() const override {
    TORCH_CHECK(false, "MonarchWorkWrapper::isSuccess() not implemented");
  }

  std::exception_ptr exception() const override {
    TORCH_CHECK(false, "MonarchWorkWrapper::exception() not implemented");
  }

  int sourceRank() const override {
    TORCH_CHECK(false, "MonarchWorkWrapper::sourceRank() not implemented");
  }

  std::vector<at::Tensor> result() override {
    TORCH_CHECK(false, "MonarchWorkWrapper::result() not implemented");
  }

  bool wait(std::chrono::milliseconds timeout) override {
    // FIXME
    TORCH_CHECK(
        timeout == kNoTimeout,
        "FutureWrappingWork::wait() with finite timeout not implemented");
    work_->wait();
    return true;
  }

  void abort() override {
    TORCH_CHECK(false, "MonarchWorkWrapper::abort() not implemented");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    TORCH_CHECK(false, "MonarchWorkWrapper::getFuture() not implemented");
  }

 private:
  rust::Box<BoxedWork> work_;
};

// An impl of `c10d::Backend` to wrap and delegate to the Rust side, via ffi
// through `BoxedBackend`.
class MonarchBackendWrapper : public c10d::Backend {
 public:
  MonarchBackendWrapper(int rank, int size, rust::Box<BoxedBackend>&& backend)
      : c10d::Backend(rank, size), backend_(std::move(backend)) {}

  const std::string getBackendName() const override {
    return "monarch";
  }

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    TORCH_CHECK(
        !isFloat8Type(tensors.back().scalar_type()),
        "Float8 dtypes are not currently supported for NCCL reductions");
    AllreduceOptions rustOpts;
    rustOpts.reduce_op = get_reduce_op(opts.reduceOp);
    auto work = backend_->allreduce(tensors, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override {
    TORCH_CHECK(inputTensors.size() == 1, "Cannot support multiple inputs");
    auto inputTensor = inputTensors.back();

    GatherOptions rustOpts{};
    rustOpts.root_rank = opts.rootRank;
    rustOpts.timeout = opts.timeout.count();

    std::vector<at::Tensor> outputs;

    if (getRank() == opts.rootRank) {
      TORCH_CHECK(
          outputTensors.size() == 1,
          "gather requires a single-element output list");
      outputs = outputTensors[0];
    } else {
      TORCH_CHECK(
          outputTensors.empty(),
          "gather requires non-root rank must not have outputs");
    }

    auto work = backend_->gather(outputs, inputTensor, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override {
    TORCH_CHECK(outputTensors.size() == 1, "Cannot support multiple outputs");
    auto outputTensor = outputTensors.back();
    ScatterOptions rustOpts{};
    rustOpts.root_rank = opts.rootRank;

    std::vector<at::Tensor> inputs;

    if (getRank() == opts.rootRank) {
      TORCH_CHECK(
          inputTensors.size() == 1,
          "scatter requires a single-element input list");
      inputs = inputTensors[0];
    } else {
      TORCH_CHECK(
          inputTensors.empty(),
          "scatter requires empty input on non-root ranks");
    }

    auto work = backend_->scatter(outputTensor, inputs, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& output,
      at::Tensor& input,
      const c10d::AllgatherOptions& /* opts */ =
          c10d::AllgatherOptions()) override {
    auto work = backend_->_allgather_base(output, input);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) {
    TORCH_CHECK(inputTensors.size() == 1, "Cannot support multiple inputs");
    auto inputTensor = inputTensors.back();
    auto outputTensors_ = outputTensors.back();
    TORCH_CHECK(outputTensors_.size() > 0, "Must pass at least one output");
    auto work = backend_->allgather(outputTensors_, inputTensor);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override {
    BarrierOptions rustOpts;
    rustOpts.timeout = opts.timeout.count();
    auto work = backend_->barrier(rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts) override {
    TORCH_CHECK(tensors.size() == 1, "Cannot support multiple inputs");
    auto tensor = tensors.back();
    ReduceOptions rustOpts{};
    rustOpts.reduce_op = get_reduce_op(opts.reduceOp);
    rustOpts.root_rank = opts.rootRank;
    auto work = backend_->reduce(tensor, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output,
      at::Tensor& input,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override {
    ReduceScatterOptions rustOpts;
    rustOpts.reduce_op = get_reduce_op(opts.reduceOp);
    TORCH_CHECK(
        !isFloat8Type(output.scalar_type()),
        "Float8 dtypes are not currently supported for NCCL reductions");
    auto work = backend_->_reduce_scatter_base(output, input, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override {
    TORCH_CHECK(
        outputTensors.size() == 1,
        "Expecting one tensor only but got multiple");
    TORCH_CHECK(
        inputTensors.size() == 1,
        "Expecting one vector of tensors only but got multiple");
    ReduceScatterOptions rustOpts{};
    rustOpts.reduce_op = get_reduce_op(opts.reduceOp);
    auto inputTensors_ = inputTensors.back();
    TORCH_CHECK(
        inputTensors_.size() > 0,
        "Must pass at least one input tensor to reduce_scatter");
    TORCH_CHECK(
        inputTensors_.size() == size_,
        "reduce_scatter input must be one per rank");
    bool same_size = check_same_size(inputTensors_);
    // TODO: If the input tensors are not the same size, it requires coalescing
    // support.
    TORCH_CHECK(same_size, "input tensors must have the same size");
    // Concatenate all tensors together to make a single large tensor. This
    // requires all individual tensors to have the same size.
    at::Tensor inputFlattened = at::cat(inputTensors_);
    auto work = backend_->_reduce_scatter_base(
        outputTensors.back(), inputFlattened, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work>
  send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override {
    auto work = backend_->send(tensors, dstRank, tag);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work>
  recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override {
    auto work = backend_->recv(tensors, srcRank, tag);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    BroadcastOptions rustOpts;
    rustOpts.root_rank = opts.rootRank;
    auto work = backend_->broadcast(tensors, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    if (outputSplitSizes.size() != 0) {
      throw std::runtime_error(
          "outputSplitSizes not yet supported in alltoall_base");
    } else if (inputSplitSizes.size() != 0) {
      throw std::runtime_error(
          "inputSplitSizes not yet supported in alltoall_base");
    }
    AllToAllOptions rustOpts{};
    rustOpts.timeout = opts.timeout.count();
    auto work = backend_->alltoall_base(outputBuffer, inputBuffer, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    AllToAllOptions rustOpts{};
    rustOpts.timeout = opts.timeout.count();
    auto work = backend_->alltoall(outputTensors, inputTensors, rustOpts);
    return c10::make_intrusive<MonarchWorkWrapper>(std::move(work));
  }

 private:
  rust::Box<BoxedBackend> backend_;
};

// An impl of `c10d::Backend` to that does nothing.  We use this when calling
// `init_process_group`.
class NullBackend : public c10d::Backend {
 public:
  NullBackend(int rank, int size) : c10d::Backend(rank, size) {}

  const std::string getBackendName() const override {
    // Use a overly descriptive name, as it'll show up in exceptions if ppl try
    // to use is.
    return "default-process-group-disabled-for-monarch";
  }
};

} // namespace

// Helper to create a Python function callback to pass to
// `Backend.register_backend()`.
PyObject* create_null_backend() {
  return py::cpp_function(
             [](const c10d::DistributedBackendOptions& options,
                const py::none&) -> c10::intrusive_ptr<c10d::Backend> {
               return c10::make_intrusive<NullBackend>(
                   options.group_rank, options.group_size);
             })
      .release()
      .ptr();
}

// Helper to create a Python function callback to pass to
// `Backend.register_backend()`.
// NOTE(agallagher): We use two levels of `Box`ing here:
// 1) The out `Box` is used to ship the `BoxedBackend` type from the Python/Rust
//    side -- via `Box::into_raw()` -- to here where we re-claim it via
//    `Box::from_raw()`.
// 2) The inner `Box` is inside `BoxedBackend` and is just used to wrap the
//    `dyn Backend` trait, as I don't think the Rust/CXX bridge code can handle
//    this otherwise.
// NOTE(agallagher): We case raw pointer to `BoxedBackend` to `uintptr_t` as I
// wasn't sure how to get it from Rust, into Python (via `pg_options`) and then
// here into C++ otherwise.
PyObject* create_monarch_backend() {
  return py::cpp_function(
             [](const c10d::DistributedBackendOptions& options,
                uintptr_t backend) -> c10::intrusive_ptr<c10d::Backend> {
               return c10::make_intrusive<MonarchBackendWrapper>(
                   options.group_rank,
                   options.group_size,
                   rust::Box<BoxedBackend>::from_raw(
                       reinterpret_cast<BoxedBackend*>(backend)));
             })
      .release()
      .ptr();
}
} // namespace monarch
