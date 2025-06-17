#include "monarch/torch-sys-cuda/src/bridge.h"

namespace monarch {
std::unique_ptr<at::cuda::CUDAEvent>
create_cuda_event(bool enable_timing, bool blocking, bool interprocess) {
  unsigned int flags = (blocking ? cudaEventBlockingSync : cudaEventDefault) |
      (enable_timing ? cudaEventDefault : cudaEventDisableTiming) |
      (interprocess ? cudaEventInterprocess : cudaEventDefault);

  return std::make_unique<at::cuda::CUDAEvent>(flags);
}

std::shared_ptr<c10::cuda::CUDAStream> get_current_stream(
    c10::DeviceIndex device) {
  return std::make_shared<c10::cuda::CUDAStream>(
      c10::cuda::getCurrentCUDAStream(device));
}

std::shared_ptr<c10::cuda::CUDAStream> create_stream(
    c10::DeviceIndex device,
    int32_t priority) {
  return std::make_shared<c10::cuda::CUDAStream>(
      c10::cuda::getStreamFromPool((const int)priority, device));
}

void set_current_stream(const c10::cuda::CUDAStream& stream) {
  auto device = c10::cuda::current_device();
  if (device != stream.device_index()) {
    c10::cuda::set_device(stream.device_index());
  }
  at::cuda::setCurrentCUDAStream(stream);
}
} // namespace monarch
