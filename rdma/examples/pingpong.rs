//! A simple RDMA ping-pong example that demonstrates:
//! 1. Setting up RDMA buffers
//! 2. Establishing connections between endpoints
//! 3. Performing RDMA write operations (ping)
//! 4. Performing RDMA read operations (pong)
//! 5. Verifying data transfer correctness
use std::time::Duration;

use anyhow::Result;
use hyperactor::clock::Clock;
use hyperactor::clock::RealClock;
use hyperactor_mesh::ProcMesh;
use hyperactor_mesh::alloc::AllocConstraints;
use hyperactor_mesh::alloc::AllocSpec;
use hyperactor_mesh::alloc::Allocator;
use hyperactor_mesh::alloc::LocalAllocator;
use ndslice::shape;
use rdma::RdmaConnectionConfig;
use rdma::RdmaManager;
use rdma::RdmaManagerArgs;
use rdma::RdmaMemoryRegion;
use rdma::get_all_devices;

// Helper function to fill a buffer with pseudo-random values
fn generate_random_data(buffer: &mut [u8], multiplier: u64) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    for (i, val) in buffer.iter_mut().enumerate() {
        // Mix timestamp with index and optional multiplier to create different values
        *val = ((timestamp.wrapping_add(i as u64).wrapping_mul(multiplier)) % 256) as u8;
    }
}

// Helper function to create test data with pseudo-random values
fn create_test_data(size: usize) -> Box<[u8]> {
    let mut data = vec![0u8; size].into_boxed_slice();
    generate_random_data(&mut data, 1);
    data
}

// Helper function to verify buffer contents match
fn verify_buffers(buffer1: &[u8], buffer2: &[u8]) {
    assert_eq!(buffer1.len(), buffer2.len(), "Buffer lengths don't match");

    for i in 0..buffer1.len() {
        assert_eq!(
            buffer1[i], buffer2[i],
            "Data mismatch at position {}: {} != {}",
            i, buffer1[i], buffer2[i]
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    println!("Starting RDMA ping-pong example");
    const BUFFER_SIZE: usize = 1024;

    // TODO - migrate this to ProcessAllocator.
    let alloc1 = LocalAllocator
        .allocate(AllocSpec {
            shape: shape! {replica=1, host=1, gpu=1},
            constraints: AllocConstraints::none(),
        })
        .await?;

    let alloc2 = LocalAllocator
        .allocate(AllocSpec {
            shape: shape! {replica=1, host=1, gpu=1},
            constraints: AllocConstraints::none(),
        })
        .await?;

    let proc_mesh1 = ProcMesh::allocate(alloc1).await?;
    let proc_mesh2 = ProcMesh::allocate(alloc2).await?;

    // Create and initialize buffers with test data
    let mut buffer1_data = create_test_data(BUFFER_SIZE);
    let mut buffer2_data = vec![0u8; BUFFER_SIZE].into_boxed_slice();

    println!("Creating RDMA buffers...");

    let devices = get_all_devices();
    println!("{} available RDMA devices:", devices.len());
    for (index, device) in devices.iter().enumerate() {
        println!("Device {}: {:?}", index, device);
    }

    if devices.len() < 5 {
        return Err(anyhow::anyhow!(
            "This example expects at least 5 RDMA devices, but {} were found",
            devices.len()
        ));
    }

    let config1 = RdmaConnectionConfig {
        device: devices.clone().into_iter().next().unwrap(),
        ..Default::default()
    };

    let config2 = RdmaConnectionConfig {
        device: devices.clone().into_iter().nth(4).unwrap(),
        ..Default::default()
    };

    // Create RDMA buffers
    let mut buffer1 = RdmaManager::new(RdmaManagerArgs {
        name: "sender".to_string(),
        proc_mesh: &proc_mesh1,
        config: config1,
        memory_region: RdmaMemoryRegion::from(&mut buffer1_data[..]),
    })
    .await?;

    let buffer2 = RdmaManager::new(RdmaManagerArgs {
        name: "receiver".to_string(),
        proc_mesh: &proc_mesh2,
        config: config2,
        memory_region: RdmaMemoryRegion::from(&mut buffer2_data[..]),
    })
    .await?;

    println!("Establishing RDMA connections...");

    // Connect the buffers (this establishes the RDMA connection)
    buffer1.connect(&buffer2).await?;
    buffer2.connect(&buffer1).await?;

    println!("Connection established successfully!");

    // PING: Write data from buffer1 to buffer2
    println!("Performing RDMA write (PING)...");
    let write_id = buffer1.write(&buffer2).await?;

    // Wait for the write operation to complete
    let completed = buffer1.wait_for_completion(&buffer2, 5, write_id).await?;
    if completed {
        println!("PING completed successfully!");
    } else {
        println!("PING timed out!");
        return Err(anyhow::anyhow!("RDMA write operation timed out"));
    }

    // Verify the data was transferred correctly
    verify_buffers(&buffer1_data, &buffer2_data);
    println!("PING data verification successful!");

    // Modify buffer2 data for the PONG with new pseudo-random values
    generate_random_data(&mut buffer2_data, 42);

    // Wait a moment before the PONG
    RealClock.sleep(Duration::from_millis(500)).await;

    // PONG: Read data from buffer2 into buffer1
    println!("Performing RDMA read (PONG)...");
    let read_id = buffer1.read_into(&buffer2).await?;

    // Wait for the read operation to complete
    let completed = buffer1.wait_for_completion(&buffer2, 5, read_id).await?;
    if completed {
        println!("PONG completed successfully!");
    } else {
        println!("PONG timed out!");
        return Err(anyhow::anyhow!("RDMA read operation timed out"));
    }

    // Verify the data was transferred correctly
    verify_buffers(&buffer1_data, &buffer2_data);
    println!("PONG data verification successful!");

    // Demonstrate multiple operations
    println!("Performing multiple RDMA operations...");

    // Generate new pseudo-random data for multiple operations
    generate_random_data(&mut buffer1_data, 17);

    // Perform 5 consecutive write operations with new random data each time
    for i in 1..=5 {
        println!("Write operation {}/5", i);

        // Generate new pseudo-random data for each iteration
        generate_random_data(&mut buffer1_data, i as u64 + 1);

        let write_id = buffer1.write(&buffer2).await?;
        let completed = buffer1.wait_for_completion(&buffer2, 5, write_id).await?;
        if !completed {
            return Err(anyhow::anyhow!("RDMA write operation {} timed out", i));
        }
        verify_buffers(&buffer1_data, &buffer2_data);
        println!("Write operation {}/5 verification successful!", i);

        // Small delay between operations
        RealClock.sleep(Duration::from_millis(100)).await;
    }

    println!("RDMA ping-pong example completed successfully!");
    Ok(())
}
