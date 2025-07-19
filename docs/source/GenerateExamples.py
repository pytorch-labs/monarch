import os

# Configuration
EXAMPLES_DIR = "../../examples"  # Path to examples directory
RST_DIR = "./examples"  # Where to output RST files (relative to Sphinx source dir)


def find_python_files(directory):
    """Find all Python files in the directory and its subdirectories, excluding __init__.py"""
    python_files = []

    # Walk through the directory tree
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                # Get the full path
                full_path = os.path.join(root, file)
                # Get the path relative to the examples directory
                rel_path = os.path.relpath(full_path, directory)
                python_files.append(rel_path)

    return python_files


def generate_rst_files():
    """Generate RST files for all Python examples"""
    # Create the output directory if it doesn't exist
    os.makedirs(RST_DIR, exist_ok=True)

    # Find all Python files
    example_files = find_python_files(EXAMPLES_DIR)

    # Generate RST files for each Python file
    for rel_path in example_files:
        # Create subdirectories in the RST directory if needed
        rel_dir = os.path.dirname(rel_path)
        if rel_dir:
            os.makedirs(os.path.join(RST_DIR, rel_dir), exist_ok=True)

        # Get the base name without extension
        base = os.path.splitext(os.path.basename(rel_path))[0]

        # Capitalize and replace underscores with spaces for nicer titles
        title = base.replace("_", " ").title()

        # Create the RST file path
        if rel_dir:
            rst_rel_path = os.path.join(rel_dir, f"{base}.rst")
        else:
            rst_rel_path = f"{base}.rst"

        rst_path = os.path.join(RST_DIR, rst_rel_path)

        # Write the RST file
        with open(rst_path, "w") as f:
            f.write(
                f"""{title}
{'=' * len(title)}

.. literalinclude:: {os.path.join('..', EXAMPLES_DIR, rel_path)}
   :language: python
   :linenos:
"""
            )

        print(f"Generated RST file for {rel_path}")

    # Generate a Python examples section in the examples.rst file
    examples_rst_path = "./examples.rst"
    with open(examples_rst_path, "r") as f:
        content = f.read()

    # Check if the Python Examples section already exists
    if "Python Examples" not in content:
        # Add the Python Examples section
        python_examples_section = """
Python Examples
--------------

These Python scripts demonstrate how to use Monarch's APIs directly in Python code:

"""
        # Add a list of Python examples
        for rel_path in example_files:
            base = os.path.splitext(os.path.basename(rel_path))[0]
            title = base.replace("_", " ").title()
            python_examples_section += f"- **{title}**: :doc:`examples/{base}`\n"

        # Append the section to the examples.rst file
        with open(examples_rst_path, "a") as f:
            f.write(python_examples_section)

        print("Added Python Examples section to examples.rst")


if __name__ == "__main__":
    generate_rst_files()
    print("RST generation complete!")
