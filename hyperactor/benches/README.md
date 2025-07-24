# Hyperactor Channel Benchmarks

This directory contains performance benchmarks for the Hyperactor channel system, specifically testing message passing throughput across different transport types and message sizes.

## Overview

The benchmark suite measures the performance of Hyperactor's channel communication system by testing:

- **Transport Types**: Local, TCP, MetaTLS, and Unix socket transports
- **Message Sizes**: Ranging from 10 bytes to 1 GB (10^1 to 10^9 bytes)
- **Metrics**: Throughput measured in bytes per second

## Benchmark Details

### Transport Types Tested

1. **Local**: In-memory transport for same-process communication
2. **TCP**: Network transport using TCP sockets
3. **MetaTLS**: Secure transport using Meta's TLS implementation
4. **Unix**: Unix domain socket transport for inter-process communication

### Message Size Range

The benchmark tests message sizes across multiple orders of magnitude:
- Small messages: 10B, 100B, 1KB
- Medium messages: 10KB, 100KB, 1MB
- Large messages: 10MB, 100MB, 1GB

Each message contains:
- An ID field (`u64`)
- A payload of the specified size (filled with zeros)

## Running the Benchmarks

### Prerequisites

Ensure you have the Rust toolchain and required dependencies installed.

### Running All Benchmarks

```bash
# From the hyperactor directory
buck run @//mode/opt //monarch/hyperactor/benches:channel_benchmarks -- --bench
```

### Running with Cargo (if available)

```bash
# From the hyperactor directory
cargo bench --bench channel_benchmarks
```

## Understanding the Results

The benchmark output will show results in the format:
```
message_sizes/message_<transport>_<size><unit>
```

For example:
- `message_sizes/message_local_1kb`: Local transport with 1KB messages
- `message_sizes/message_tcp_10mb`: TCP transport with 10MB messages
- `message_sizes/message_metatls_100b`: MetaTLS transport with 100 byte messages

### Interpreting Performance

- **Higher throughput** (bytes/sec) indicates better performance
- **Local transport** typically shows the highest throughput
- **Larger messages** generally achieve higher throughput due to amortized overhead
- **Secure transports** (MetaTLS) may show lower throughput due to encryption overhead

## Benchmark Implementation

The benchmark uses the [Criterion](https://docs.rs/criterion/) framework for statistical analysis and uses Tokio's async runtime for handling the asynchronous channel operations.

Each benchmark iteration:
1. Sets up a server listening on the specified transport
2. Creates a client connection to the server
3. Sends messages of the specified size
4. Measures the time taken for round-trip communication
5. Calculates throughput based on bytes transferred and time elapsed

## Use Cases

These benchmarks are useful for:
- **Performance regression testing**: Ensuring changes don't degrade performance
- **Transport selection**: Choosing the appropriate transport for your use case
- **Capacity planning**: Understanding throughput limits for different message sizes
- **Optimization validation**: Verifying that performance improvements are effective
