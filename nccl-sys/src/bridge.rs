use std::collections::HashMap;
use std::pin::Pin;
use std::pin::pin;

use nccl_sys_bindgen::ncclComm_t;
use nccl_sys_bindgen::ncclResult_t;

#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("monarch/nccl-sys/src/nccl_extras.h");
        #[namespace = ""]
        type ncclComm = nccl_sys_bindgen::ncclComm;

        #[namespace = ""]
        type ncclResult_t = nccl_sys_bindgen::ncclResult_t;

        #[namespace = "monarch::nccl"]
        unsafe fn nccl_comm_dump(comm: *mut ncclComm, map: &mut Vec<String>) -> ncclResult_t;
    }
}

pub fn nccl_comm_dump(comm: ncclComm_t) -> Result<HashMap<String, String>, ncclResult_t> {
    let mut map = Vec::<String>::new();
    // SAFETY: We know this comm type is safe.
    let result = unsafe { crate::bridge::ffi::nccl_comm_dump(comm, &mut map) };
    if result.0 == 0 {
        Ok(map
            .chunks(2)
            .map(|s| (s[0].clone(), s[1].clone()))
            .collect())
    } else {
        Err(result)
    }
}
