## Brief Introduction
The Simulator can act as a backend, similar to ProcessBackend, or as a standalone object to receive messages from a pre-stored file. Its primary function is to simulate the execution time and memory usage based on the input messages.

### Execution model
The Simulator consists of multiple worker objects, each maintaining several stream objects. The Simulator forwards messages to the workers, which in turn forward them to the streams. A Stream object maintains a task queue and executes the first task when its dependencies are fulfilled. The task will be marked as finished immediately after executing if it is a computation op. If the task is a collective op, it will only be marked as finished after all other tasks participating in the collective op have been executed. A trace event will be created for the task after it is finished.

### Memory model
Currently, only GPU memory is recorded. A GPU tensor must be created by some task, so a GPU tensor is created when a task is created. However, its memory will only be allocated after the task is executed. To avoid double-counting the memory usage of tensors that share the same storage, a WorkerStorageTracker is used to track unique storage. The memory usage is increased only when a new storage is created, and decreased only when an existing storage is deleted. The memory usage of a storage is attributed to the stream that creates the storage.

## Current Status, Implemented Features
* Concurrent Task Execution: Traces concurrent tasks across different streams and workers, including collective operations.
* Memory Tracking: Traces memory usage without overcounting, particularly for views.
* Controller Message Tracing: Logs messages from the controller for better oversight and debugging.

## Pending Features
* Deduplication: Many workers behave the same. The Simulator should group them to make the trace easier to read and the simulation faster.
* Profiling: The current runtime of each op is hardcoded and incorrect. The Simulator should take the profiling result as data to simulate. We would need a feature to support cached propagation of remote functions.
* Remote Function: The Simulator will fail with the new cache propagation remote function or cause a hang.
* Fetch Shard: Not implemented yet.
* Trace of CPU operations: The current design assumes CPU ops have zero overheads, so CPU tensors will just be created without taking time. This is not accurate and can be an issue if users perform optimizer CPU offloading.
