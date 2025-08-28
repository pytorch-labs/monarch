# Monarch 🦋

**Monarch** is a distributed execution engine for PyTorch. Our overall goal is
to deliver the high-quality user experience that people get from single-GPU
PyTorch, but at cluster scale.

> ⚠️ **Early Development Warning** Monarch is currently in an experimental
> stage. You should expect bugs, incomplete features, and APIs that may change
> in future versions. The project welcomes bugfixes, but to make sure things are
> well coordinated you should discuss any significant change before starting the
> work. It's recommended that you signal your intention to contribute in the
> issue tracker, either by filing a new issue or by claiming an existing one.

## What is Monarch?

Monarch extends PyTorch's capabilities to efficiently run on distributed systems. It maintains the familiar PyTorch API while handling the complexities of distributed execution, making it easier to scale your deep learning workloads across multiple GPUs and nodes.

Key features:
- **Familiar PyTorch API** - Use the same PyTorch code you're already familiar with
- **Efficient Distribution** - Scale your models across multiple GPUs and nodes
- **Simplified Communication** - Built-in primitives for distributed communication
- **Performance Optimized** - Designed for high performance at scale

**Note:** Monarch is currently only supported on Linux systems.

## Getting Started

Here are some suggested steps to get started with Monarch:

1. **Learn the Basics**: Check out the [Getting Started](get_started) guide to learn the basics of Monarch
2. **Explore Examples**: Review the [Examples](./generated/examples/index) to see Monarch in action
3. **Dive Deeper**: Explore the API Documentation for more detailed information:
    - [Python API](api/index)
    - [Rust API](rust-api)


```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

get_started
./generated/examples/index
books/books
api/index
rust-api
```

## License

Monarch is BSD-3 licensed, as found in the [LICENSE](https://github.com/meta-pytorch/monarch/blob/main/LICENSE) file.

## Community

We welcome contributions from the community! If you're interested in contributing, please:

1. Check the [GitHub repository](https://github.com/meta-pytorch/monarch)
2. Review existing issues or create a new one
3. Discuss your proposed changes before starting work
4. Submit a pull request with your changes
