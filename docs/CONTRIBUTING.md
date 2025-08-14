# Documentation Contributing Guide

This guide outlines how to contribute to the Monarch documentation, including building docs locally, adding new documentation, books, examples, and API documentation.

## Overview

The Monarch documentation uses [Sphinx](https://www.sphinx-doc.org/) with the [PyTorch theme](https://github.com/pytorch/pytorch_sphinx_theme/blob/pytorch_sphinx_theme2/pytorch_sphinx_theme2_migration_guide.md) and supports multiple content types:
- **Markdown files** (`.md`) using MyST parser
- **Python examples** using Sphinx Gallery
- **Books** using mdBook format
- **API documentation** for both Python and Rust APIs

## Directory Structure

```
docs/
├── source/                    # Sphinx source files
│   ├── _ext/                 # Custom Sphinx extensions
│   ├── _static/              # Static assets (CSS, images, etc.)
│   ├── _templates/           # Custom HTML templates
│   ├── books/                # Books documentation
│   ├── examples/             # Python example scripts
│   ├── generated/            # Auto-generated content
│   ├── python-api/           # Python API documentation
│   ├── conf.py               # Sphinx configuration
│   ├── index.md              # Main documentation index
│   ├── get_started.md        # Getting started guide
│   └── rust-api.md           # Rust API documentation index
├── build/                    # Built documentation output
├── Makefile                  # Build commands
├── requirements.txt          # Python dependencies
└── make.bat                  # Windows build script
```

## Building Documentation Locally

### Prerequisites

1. **Install Python dependencies**:
   ```bash
   cd docs
   pip install -r requirements.txt
   ```

2. **Install Rust and build Rust documentation** (for API docs):
   ```bash
   # Install Rust (if not already installed)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Generate Rust API documentation
   cd ..  # Go to project root
   cargo doc --workspace --no-deps
   ```

3. **Install build dependencies** (from project root):
   ```bash
   pip install -r build-requirements.txt
   ```

### Building the Documentation

#### Quick Build (HTML only)
```bash
cd docs
make html
```

#### Clean Build (recommended for first build)
```bash
cd docs
make clean
make html
```

#### Preview the Documentation
After building, open `docs/build/html/index.html` in your browser or serve it locally:

```bash
cd build/html
python -m http.server 8000
# Then open http://localhost:8000 in your browser
```

#### Available Make Commands
- `make html` - Build HTML documentation
- `make clean` - Clean build and generated directories
- `make help` - Show all available commands

## Adding Documentation

### Adding Regular Documentation Pages

1. **Create a new `.md` file** in `/docs/source/`
2. **Use MyST Markdown syntax** for advanced features:
   ```markdown
   # Page Title

   ## Section

   ```{note}
   This is a note admonition.
   ```

   ```{code-block} python
   # Python code example
   import monarch
   ```
   ```

3. **Add the page to the table of contents** in `docs/source/index.md`:
   ```markdown
   ```{toctree}
   :maxdepth: 2
   :caption: Contents:

   get_started
   your_new_page
   ```
   ```

### MyST Markdown Features

The documentation supports these MyST extensions:
- **Admonitions**: `{note}`, `{warning}`, `{tip}`, etc.
- **Code blocks**: `{code-block} python`
- **Cross-references**: `{doc}page_name` or `{ref}label`
- **Math**: `$inline$` or `$$block$$`
- **HTML**: Direct HTML embedding

## Adding Books

Books are multi-page documentation using mdBook format, ideal for comprehensive guides or tutorials.

### Creating a New Book

1. **Create book directory**:
   ```bash
   mkdir -p docs/source/books/your-book-name/src
   ```

2. **Create `book.toml` configuration**:
   ```toml
   [book]
   authors = ["Your Name"]
   language = "en"
   multilingual = false
   src = "src"
   title = "Your Book Title"

   [output.html]
   ```

3. **Create `SUMMARY.md` (table of contents)**:
   ```markdown
   # Summary

   [Introduction](introduction.md)

   - [Chapter 1](chapter1.md)
   - [Chapter 2](chapter2.md)
     - [Section 2.1](chapter2/section1.md)
   ```

4. **Create content files** in the `src/` directory following your `SUMMARY.md` structure.

5. **Add book to main documentation** by editing `docs/source/books/books.md`:
   ```markdown
   ## Your Book Title

   Description of your book.

   [Read the book](your-book-name/src/introduction.md)
   ```

### Example: Existing HyperActor Book

The HyperActor book demonstrates the structure:
```
docs/source/books/hyperactor-book/
├── book.toml
├── README.md
└── src/
    ├── SUMMARY.md
    ├── introduction.md
    ├── actors/
    │   ├── index.md
    │   ├── actor.md
    │   └── ...
    └── ...
```

## Adding Examples

Examples are Python scripts that demonstrate Monarch functionality and are automatically processed by Sphinx Gallery.

### Creating Python Examples

1. **Add Python script** to `docs/source/examples/`:
   ```python
   """
   Example Title
   =============

   Brief description of what this example demonstrates.

   This example shows how to...
   """

   import monarch

   # Your example code here
   def main():
       # Demonstration code
       pass

   if __name__ == "__main__":
       main()
   ```

2. **Follow naming conventions**:
   - Use descriptive filenames (e.g., `distributed_training.py`)
   - Avoid `__init__.py` (it's automatically excluded)

3. **Include proper docstring**:
   - Start with a title using `=` underline
   - Provide clear description
   - Include usage examples in comments

### Example File Structure

Current examples include:
- `grpo_actor.py` - GRPO (Generalized Robust Policy Optimization) actor example
- `ping_pong.py` - Basic actor communication example
- `spmd_ddp.py` - SPMD (Single Program, Multiple Data) distributed data parallel example

### Sphinx Gallery Configuration

Examples are automatically processed with these settings (in `conf.py`):
```python
sphinx_gallery_conf = {
    "examples_dirs": ["./examples"],
    "gallery_dirs": "./generated/examples",
    "filename_pattern": r".*\.py$",
    "ignore_pattern": r"__init__\.py",
    "plot_gallery": "False",
}
```

## Adding and Building API Documentation

### Python API Documentation

1. **Auto-generated from docstrings**: Python API docs are automatically generated from the source code docstrings.

2. **Add modules to API docs**:
   - Create or edit files in `docs/source/python-api/`
   - Use autodoc directives:
     ```rst
     .. automodule:: monarch.your_module
        :members:
        :undoc-members:
        :show-inheritance:
     ```

3. **Improve docstrings** in Python source code:
   ```python
   def your_function(param1: str, param2: int = 0) -> bool:
       """Brief description of the function.

       Longer description if needed.

       Args:
           param1: Description of param1
           param2: Description of param2, defaults to 0

       Returns:
           Description of return value

       Raises:
           ValueError: When param1 is invalid

       Example:
           >>> your_function("test", 42)
           True
       """
       pass
   ```

### Rust API Documentation

1. **Generate Rust docs**:
   ```bash
   # From project root
   cargo doc --workspace --no-deps
   ```

2. **Rust docs are automatically included** in the build process and copied to the final documentation.

3. **Improve Rust documentation**:
   ```rust
   /// Brief description of the function
   ///
   /// Longer description with more details.
   ///
   /// # Arguments
   ///
   /// * `param1` - Description of param1
   /// * `param2` - Description of param2
   ///
   /// # Returns
   ///
   /// Description of return value
   ///
   /// # Examples
   ///
   /// ```
   /// use your_crate::your_function;
   /// let result = your_function("test", 42);
   /// ```
   pub fn your_function(param1: &str, param2: i32) -> bool {
       // Implementation
   }
   ```

## Configuration Details

### Sphinx Configuration (`conf.py`)

Key configuration settings:
- **Theme**: PyTorch Sphinx Theme 2
- **Extensions**: MyST parser, Sphinx Gallery, Sphinx Design, Mermaid diagrams
- **API Integration**: Automatic inclusion of Rust API docs
- **Gallery**: Automatic Python example processing

### GitHub Actions Integration

Documentation is automatically built and deployed via GitHub Actions (`.github/workflows/doc_build.yml`):
- **Triggers**: Push to main, PRs, manual dispatch
- **Process**: Builds both Rust and Python APIs, processes examples, builds Sphinx docs
- **Deployment**: Automatically deploys to GitHub Pages on main branch

## Best Practices

### Documentation Writing
- **Use clear, concise language**
- **Include code examples** where appropriate
- **Add cross-references** to related documentation
- **Test all code examples** before committing

### File Organization
- **Group related content** in subdirectories
- **Use descriptive filenames**
- **Keep the main index clean** and well-organized
- **Update tables of contents** when adding new pages

### API Documentation
- **Write comprehensive docstrings**
- **Include examples** in docstrings
- **Document all parameters and return values**
- **Use type hints** in Python code

### Testing Documentation
- **Build locally** before submitting PRs
- **Check for broken links** and references
- **Verify examples work** as expected
- **Review generated output** for formatting issues

## Troubleshooting

### Common Issues

1. **Build failures**:
   - Run `make clean` then `make html`
   - Check for missing dependencies in `requirements.txt`
   - Verify Rust toolchain is installed for API docs

2. **Missing examples**:
   - Ensure Python files follow naming conventions
   - Check that files have proper docstring format
   - Verify `sphinx_gallery_conf` in `conf.py`

3. **Broken cross-references**:
   - Use correct MyST syntax for references
   - Ensure target files/sections exist
   - Check file paths are relative to source directory

4. **Theme/styling issues**:
   - Clear browser cache
   - Check `_static/` directory for custom CSS
   - Verify PyTorch theme is properly installed

### Getting Help

- **Sphinx documentation**: https://www.sphinx-doc.org/
- **MyST parser**: https://myst-parser.readthedocs.io/
- **PyTorch theme**: https://github.com/pytorch/pytorch_sphinx_theme
- **Sphinx Gallery**: https://sphinx-gallery.github.io/

For project-specific issues, refer to the main project documentation or create an issue in the repository.
