Monarch Examples
================

This directory contains examples demonstrating how to use Monarch for distributed computing.

Python Script Examples
---------------------

These examples are formatted for sphinx-gallery and will be automatically converted to HTML documentation:

- ``ping_pong.py``: Demonstrates the basics of Monarch's Actor/endpoint API with a ping-pong communication example
- ``spmd_ddp.py``: Shows how to run PyTorch's Distributed Data Parallel (DDP) within Monarch actors
- ``grpo_actor.py``: Implements a distributed PPO-like reinforcement learning algorithm using the Monarch actor framework

Running Examples
---------------

To run any example:

.. code-block:: bash

    python examples/example_name.py
