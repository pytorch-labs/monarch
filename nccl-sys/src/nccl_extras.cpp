#include "nccl_extras.h"

namespace monarch::nccl {

ncclResult_t nccl_comm_dump(
    [[maybe_unused]] ncclComm_t comm,
    rust::Vec<rust::String>& out) {
  std::unordered_map<std::string, std::string> map;

#if (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
  ncclResult_t result = ncclCommDump(comm, map);
  if (result != ncclSuccess) {
    return result;
  }
#endif
  for (auto& [k, v] : map) {
    out.emplace_back(k);
    out.emplace_back(v);
  }
  return ncclSuccess;
}

} // namespace monarch::nccl
