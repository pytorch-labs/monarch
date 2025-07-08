/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[cfg(test)]
pub mod test_utils {
    use std::time::Duration;
    use std::time::Instant;

    use hyperactor::ActorRef;
    use hyperactor::Mailbox;
    use hyperactor::clock::Clock;
    use hyperactor::clock::RealClock;
    use hyperactor_mesh::Mesh;
    use hyperactor_mesh::ProcMesh;
    use hyperactor_mesh::RootActorMesh;
    use hyperactor_mesh::alloc::AllocSpec;
    use hyperactor_mesh::alloc::Allocator;
    use hyperactor_mesh::alloc::LocalAllocator;
    use ndslice::shape;

    use crate::IbverbsConfig;
    use crate::RdmaBuffer;
    use crate::cu_check;
    use crate::ibverbs_primitives::get_all_devices;
    use crate::rdma_components::RdmaQueuePair;
    use crate::rdma_manager_actor::RdmaManagerActor;
    use crate::rdma_manager_actor::RdmaManagerMessageClient;

    // Waits for the completion of an RDMA operation.

    // This function polls for the completion of an RDMA operation by repeatedly
    // sending a `PollCompletion` message to the specified actor mesh and checking
    // the returned work completion status. It continues polling until the operation
    // completes or the specified timeout is reached.

    pub async fn wait_for_completion(
        qp: &RdmaQueuePair,
        timeout_secs: u64,
    ) -> Result<bool, anyhow::Error> {
        let timeout = Duration::from_secs(timeout_secs);
        let start_time = Instant::now();
        while start_time.elapsed() < timeout {
            match qp.poll_completion() {
                Ok(Some(wc)) => {
                    if wc.wr_id() == 0 {
                        return Ok(true);
                    }
                }
                Ok(None) => {
                    RealClock.sleep(Duration::from_millis(1)).await;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(e));
                }
            }
        }

        Err(anyhow::Error::msg("Timeout while waiting for completion"))
    }

    pub struct RdmaManagerTestEnv<'a> {
        buffer_1: Buffer,
        buffer_2: Buffer,
        pub client_1: &'a Mailbox,
        pub client_2: &'a Mailbox,
        pub actor_1: ActorRef<RdmaManagerActor>,
        pub actor_2: ActorRef<RdmaManagerActor>,
        pub rdma_handle_1: RdmaBuffer,
        pub rdma_handle_2: RdmaBuffer,
        handle_1_cuda: bool,
        handle_2_cuda: bool,
    }

    #[derive(Debug, Clone)]
    pub struct Buffer {
        ptr: u64,
        len: usize,
        cpu_ref: Option<Box<[u8]>>,
    }
    impl RdmaManagerTestEnv<'_> {
        /// Sets up the RDMA test environment.
        ///
        /// This function initializes the RDMA test environment by setting up two actor meshes
        /// with their respective RDMA configurations. It also prepares two buffers for testing
        /// RDMA operations and fills the first buffer with test data.
        ///
        /// # Arguments
        ///
        /// * `buffer_size` - The size of the buffers to be used in the test.
        /// * `devices` - Optional tuple specifying the indices of RDMA devices to use. If not provided, then
        ///   both RDMAManagerActors will default to the first indexed RDMA device.
        pub async fn setup(
            buffer_size: usize,
            nics: (&str, &str),
            devices: (&str, &str),
        ) -> Result<Self, anyhow::Error> {
            let all_devices = get_all_devices();
            let mut config1 = None;
            let mut config2 = None;

            for device in all_devices.iter() {
                if device.name == nics.0 {
                    config1 = Some(IbverbsConfig {
                        device: device.clone(),
                        ..Default::default()
                    });
                }
                if device.name == nics.1 {
                    config2 = Some(IbverbsConfig {
                        device: device.clone(),
                        ..Default::default()
                    });
                }
            }
            assert!(config1.is_some() && config2.is_some());

            let device_str1 = (String::new(), 0);
            let device_str2 = (String::new(), 0);

            if let Some((backend, idx)) = devices.0.split_once(':') {
                assert!(backend == "cuda");
                let parsed_idx = idx
                    .parse::<usize>()
                    .expect("Device index is not a valid integer");
                let _device_str1 = (backend.to_string(), parsed_idx.to_string());
            } else {
                assert!(devices.0 == "cpu");
                let _device_str1 = (devices.0.to_string(), 0);
            }

            if let Some((backend, idx)) = devices.1.split_once(':') {
                assert!(backend == "cuda");
                let parsed_idx = idx
                    .parse::<usize>()
                    .expect("Device index is not a valid integer");
                let _device_str1 = (backend.to_string(), parsed_idx.to_string());
            } else {
                assert!(devices.1 == "cpu");
                let _device_str2 = (devices.1.to_string(), 0);
            }

            let alloc_1 = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { proc = 1 },
                    constraints: Default::default(),
                })
                .await
                .unwrap();

            let proc_mesh_1 = Box::leak(Box::new(ProcMesh::allocate(alloc_1).await.unwrap()));
            let actor_mesh_1: RootActorMesh<'_, RdmaManagerActor> = proc_mesh_1
                .spawn("rdma_manager", &(config1.unwrap()))
                .await
                .unwrap();

            let alloc_2 = LocalAllocator
                .allocate(AllocSpec {
                    shape: shape! { proc = 1 },
                    constraints: Default::default(),
                })
                .await
                .unwrap();

            let proc_mesh_2 = Box::leak(Box::new(ProcMesh::allocate(alloc_2).await.unwrap()));
            let actor_mesh_2: RootActorMesh<'_, RdmaManagerActor> = proc_mesh_2
                .spawn("rdma_manager", &(config2.unwrap()))
                .await
                .unwrap();

            let mut buf_vec = Vec::new();

            for device_str in [device_str1.clone(), device_str2.clone()] {
                if device_str.0 != "cpu" {
                    let mut buffer = vec![0u8; buffer_size].into_boxed_slice();
                    buf_vec.push(Buffer {
                        ptr: buffer.as_mut_ptr() as u64,
                        len: buffer.len(),
                        cpu_ref: Some(buffer),
                    });
                    continue;
                }
                // CUDA case
                unsafe {
                    cu_check!(cuda_sys::cuInit(0));

                    let mut dptr: cuda_sys::CUdeviceptr = std::mem::zeroed();
                    let mut handle: cuda_sys::CUmemGenericAllocationHandle = std::mem::zeroed();
                    let /*mut*/ padded_size: usize;

                    let mut device: cuda_sys::CUdevice = std::mem::zeroed();
                    cu_check!(cuda_sys::cuDeviceGet(&mut device, device_str.1));

                    let mut context: cuda_sys::CUcontext = std::mem::zeroed();
                    cu_check!(cuda_sys::cuCtxCreate_v2(&mut context, 0, device_str.1));
                    cu_check!(cuda_sys::cuCtxSetCurrent(context));

                    let mut granularity: usize = 0;
                    let mut prop: cuda_sys::CUmemAllocationProp = std::mem::zeroed();
                    prop.type_ = cuda_sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
                    prop.location.type_ = cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                    prop.location.id = device;
                    prop.allocFlags.gpuDirectRDMACapable = 1;
                    prop.requestedHandleTypes =
                        cuda_sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

                    cu_check!(cuda_sys::cuMemGetAllocationGranularity(
                        &mut granularity as *mut usize,
                        &prop,
                        cuda_sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                    ));

                    println!("granularity: {}", granularity);
                    // ensure our size is aligned
                    padded_size = ((buffer_size - 1) / granularity + 1) * granularity;
                    assert!(padded_size == buffer_size);

                    cu_check!(cuda_sys::cuMemCreate(
                        &mut handle as *mut cuda_sys::CUmemGenericAllocationHandle,
                        padded_size,
                        &prop,
                        0
                    ));
                    println!("cuMemCreate done");
                    // reserve and map the memory
                    // let mut dptr: cuda_sys::CUdeviceptr = std::mem::zeroed();
                    cu_check!(cuda_sys::cuMemAddressReserve(
                        &mut dptr as *mut cuda_sys::CUdeviceptr,
                        padded_size,
                        0,
                        0,
                        0,
                    ));
                    println!("cuMemAddressReserve done");
                    println!("dptr: 0x{:x}", dptr);
                    println!("padded_size: {:?}", padded_size);
                    println!("handle: {:?}", handle);

                    assert!(dptr as usize % granularity == 0);
                    assert!(padded_size % granularity == 0);

                    // fails if a add cu_check macro; but passes if we don't
                    let err = cuda_sys::cuMemMap(
                        dptr as cuda_sys::CUdeviceptr,
                        padded_size,
                        0,
                        handle as cuda_sys::CUmemGenericAllocationHandle,
                        0,
                    );
                    if err != cuda_sys::CUresult::CUDA_SUCCESS {
                        panic!("failed reserving and mapping memory {:?}", err);
                    }

                    // set access
                    let mut access_desc: cuda_sys::CUmemAccessDesc = std::mem::zeroed();
                    access_desc.location.type_ =
                        cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
                    access_desc.location.id = device;
                    access_desc.flags =
                        cuda_sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    println!("mem set access");
                    cu_check!(cuda_sys::cuMemSetAccess(dptr, padded_size, &access_desc, 1));
                    println!("mem set access completed");
                    buf_vec.push(Buffer {
                        ptr: dptr,
                        len: padded_size,
                        cpu_ref: None,
                    });
                }
            }

            // Fill buffer1 with test data
            if device_str1.0 == "cuda" {
                let mut temp_buffer = vec![0u8; buffer_size].into_boxed_slice();
                for (i, val) in temp_buffer.iter_mut().enumerate() {
                    *val = (i % 256) as u8;
                }
                unsafe {
                    cu_check!(cuda_sys::cuMemcpyHtoD_v2(
                        buf_vec[0].ptr,
                        temp_buffer.as_ptr() as *const std::ffi::c_void,
                        temp_buffer.len()
                    ));
                }
            } else {
                unsafe {
                    let ptr = buf_vec[0].ptr as *mut u8; // or *const u8
                    for i in 0..buf_vec[0].len {
                        *ptr.add(i) = (i % 256) as u8;
                    }
                }
            }
            let actor_1 = actor_mesh_1.get(0).unwrap();
            let actor_2 = actor_mesh_2.get(0).unwrap();

            let rdma_handle_1 = actor_1
                .request_buffer(proc_mesh_1.client(), buf_vec[0].ptr as usize, buffer_size)
                .await?;
            let rdma_handle_2 = actor_2
                .request_buffer(proc_mesh_2.client(), buf_vec[1].ptr as usize, buffer_size)
                .await?;
            // Get keys from both actors.

            let buffer_2 = buf_vec.remove(1);
            let buffer_1 = buf_vec.remove(0);
            Ok(Self {
                buffer_1,
                buffer_2,
                client_1: proc_mesh_1.client(),
                client_2: proc_mesh_2.client(),
                actor_1,
                actor_2,
                rdma_handle_1,
                rdma_handle_2,
                handle_1_cuda: device_str1.0 == "cuda",
                handle_2_cuda: device_str2.0 == "cuda",
            })
        }

        pub async fn cleanup(self) -> Result<(), anyhow::Error> {
            self.actor_1
                .release_buffer(self.client_1, self.rdma_handle_1.clone())
                .await?;
            self.actor_2
                .release_buffer(self.client_2, self.rdma_handle_2.clone())
                .await?;
            if self.handle_1_cuda {
                unsafe {
                    cu_check!(cuda_sys::cuMemUnmap(
                        self.buffer_1.ptr as cuda_sys::CUdeviceptr,
                        self.buffer_1.len
                    ));
                    cu_check!(cuda_sys::cuMemAddressFree(
                        self.buffer_1.ptr as cuda_sys::CUdeviceptr,
                        self.buffer_1.len
                    ));
                }
            }
            if self.handle_2_cuda {
                unsafe {
                    cu_check!(cuda_sys::cuMemUnmap(
                        self.buffer_2.ptr as cuda_sys::CUdeviceptr,
                        self.buffer_2.len
                    ));
                    cu_check!(cuda_sys::cuMemAddressFree(
                        self.buffer_2.ptr as cuda_sys::CUdeviceptr,
                        self.buffer_2.len
                    ));
                }
            }
            Ok(())
        }

        pub async fn verify_buffers(&self, size: usize) -> Result<(), anyhow::Error> {
            let mut buf_vec = Vec::new();
            for (handle, is_cuda) in [
                (self.rdma_handle_1.clone(), self.handle_1_cuda),
                (self.rdma_handle_2.clone(), self.handle_2_cuda),
            ] {
                if is_cuda {
                    let mut temp_buffer = vec![0u8; size].into_boxed_slice();
                    // SAFETY: The buffer is allocated with the correct size and the pointer is valid.
                    unsafe {
                        cu_check!(cuda_sys::cuMemcpyDtoH_v2(
                            temp_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                            handle.addr as cuda_sys::CUdeviceptr,
                            size
                        ));
                    }
                    buf_vec.push(Buffer {
                        ptr: temp_buffer.as_mut_ptr() as u64,
                        len: size,
                        cpu_ref: Some(temp_buffer),
                    });
                } else {
                    buf_vec.push(Buffer {
                        ptr: handle.addr as u64,
                        len: size,
                        cpu_ref: None,
                    });
                }
            }
            // SAFETY: The pointers are valid and the buffers have the same length.
            unsafe {
                let ptr1 = buf_vec[0].ptr as *mut u8;
                let ptr2: *mut u8 = buf_vec[1].ptr as *mut u8;
                for i in 0..buf_vec[0].len {
                    if *ptr1.add(i) != *ptr2.add(i) {
                        return Err(anyhow::anyhow!("Buffers are not equal at index {}", i));
                    }
                }
            }
            Ok(())
        }
    }
}
