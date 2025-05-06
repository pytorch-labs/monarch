# RL with Monarch

This folder includes examples for RL training with Monarch:
- [`train_async.py`](train_async.py): an example for asynchronous training
- [`train_sync.py`](train_sync.py): an example for "off-by-1" semi-synchronous training.

The RL formulation is trivial - a policy gradient that tries to make the policy output really large positive values. This is really intended to ensure that weights are being transferred between the learner and generators successfully.

This prototype aims to showcase infrastructural features of Monarch, including:
1. Orchestration of dataflow with Monarch's single controller architecture
2. The ease of wrapping existing multi-controller, SPMD-based PyTorch workloads with Monarch
3. [TODO] Mixing heterogeneous sharding types
4. [TODO] High performance component<>component data transfer, i.e. policy weights and generated trajectories.


## Caveats
1. Only support for single host
