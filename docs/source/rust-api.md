# Rust API Documentation

This page provides access to the Rust API documentation for Monarch.

The Rust API documentation is automatically generated from the source code using Rustdoc.


## Accessing the Rust API Documentation

You can access the full Rust API documentation here:

<a href="../index.html" class="btn btn-primary">View Rust API Documentation</a>

<script>
// Check if the help.html file exists
fetch('../index.html')
  .then(response => {
    if (!response.ok) {
      // If the file doesn't exist, try a different path
      document.querySelector('a[href="../index.html"]').href = './index.html';
    }
  })
  .catch(error => {
    // If there's an error, try a different path
    document.querySelector('a[href="../index.html"]').href = './index.html';
  });
</script>


## Key Modules

The Rust implementation includes several key modules:

- **Core**: Core functionality and data structures
- **Runtime**: The execution runtime for distributed operations
- **Communication**: Network communication primitives
- **Serialization**: Data serialization and deserialization utilities

For complete details on all modules, classes, and functions, please refer to the full API documentation linked above.
