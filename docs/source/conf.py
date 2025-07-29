# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Monarch"
copyright = "2025"
author = ""
release = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxcontrib.mermaid",
    "pytorch_sphinx_theme2",
    "sphinxext.opengraph",
    "myst_parser",
    # "nbsphinx",
    "sphinx_gallery.gen_gallery",
    # "myst_nb",
    #"sphinx_rust",
    
]

sphinx_gallery_conf = {
    "examples_dirs": [
        "./examples",
    ],  # path to your example scripts
    "gallery_dirs": "./generated/examples",  # path to where to save gallery generated output
    "filename_pattern": ".*\.py$",  # Include all Python files
    "ignore_pattern": "__init__\.py",  # Exclude __init__.py files
    "plot_gallery": "False",  # Don't run the examples
    # "thumbnail_size": (400, 400),  # Thumbnail size
    # "download_all_examples": True,  # Allow downloading all examples
    # "line_numbers": True,  # Show line numbers in code blocks
    # "remove_config_comments": True,  # Remove configuration comments
    # "show_memory": False,  # Don't show memory usage
    # "show_signature": True,  # Show function signatures
}



templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import os
import sys

# Add the repository root to the path so Sphinx can find the notebook files
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))
import pytorch_sphinx_theme2

html_theme = "pytorch_sphinx_theme2"
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]

ogp_site_url = "http://pytorch.org/monarch"
ogp_image = "https://pytorch.org/assets/images/social-share.jpg"

html_theme_options = {
    "navigation_with_keys": False,
    "analytics_id": "GTM-T8XT4PS",
    "logo": {
        "text": "",
    },
    "icon_links": [
        {
            "name": "X",
            "url": "https://x.com/PyTorch",
            "icon": "fa-brands fa-x-twitter",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/pytorch-labs/monarch",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Discourse",
            "url": "https://dev-discuss.pytorch.org/",
            "icon": "fa-brands fa-discourse",
        },
        {
            "name": "PyPi",
            "url": "https://pypi.org/project/monarch/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": True,
    "navbar_center": "navbar-nav",
}

theme_variables = pytorch_sphinx_theme2.get_theme_variables()
templates_path = [
    "_templates",
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), "templates"),
]

html_context = {
    "theme_variables": theme_variables,
    "display_github": True,
    "github_url": "https://github.com",
    "github_user": "pytorch-labs",
    "github_repo": "monarch",
    "feedback_url": "https://github.com/pytorch-labs/monarch",
    "github_version": "main",
    "doc_path": "docs/source",
    "library_links": theme_variables.get("library_links", []),
    "community_links": theme_variables.get("community_links", []),
    "language_bindings_links": html_theme_options.get("language_bindings_links", []),
}

# Enable MyST extensions for markdown files including those in books
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Add the books directory to the Sphinx path
books_path = os.path.abspath("../../books")

# Configure Sphinx to include the books directory
exclude_patterns = []  # Reset exclude_patterns to ensure books are included

# Add the books directory to the Sphinx path
import sphinx.util.matching
html_extra_path = []
html_static_path = ["_static"]

# Configure MyST-Parser to find markdown files in the books directory
myst_update_mathjax = False
myst_heading_anchors = 3

# Add the books directory to the Sphinx search path
import os
books_path = os.path.abspath("../../books")


# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Configure MyST-Parser to properly handle relative links in the books directory
myst_url_schemes = ["http", "https", "mailto"]

# Add the books directory to the Sphinx extensions path
import os
import sys
sys.path.insert(0, os.path.abspath("../../books"))

# Allow errors in notebook execution
nbsphinx_allow_errors = True


def truncate_index_file_at_raw_html(file_path):
    """
    Truncate the Sphinx-Gallery index file at the first occurrence of the
    raw HTML div with class 'sphx-glr-thumbnails'.

    Parameters:
    - file_path (str): The path to the index file to be truncated.
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Find the index of the first occurrence of the target lines
        target_lines = [
            ".. raw:: html\n",
            "\n",
            '    <div class="sphx-glr-thumbnails">\n',
        ]

        # Search for the sequence in the lines
        truncate_index = None
        for i in range(len(lines) - len(target_lines) + 1):
            if lines[i : i + len(target_lines)] == target_lines:
                truncate_index = i
                break

        if truncate_index is not None:
            truncated_lines = lines[:truncate_index]
            with open(file_path, "w") as file:
                file.writelines(truncated_lines)
            print(f"File {file_path} truncated at line {truncate_index}.")
        else:
            print(
                f"Target raw HTML block not found in {file_path}. No truncation done."
            )

    except Exception as e:
        print(f"An error occurred while truncating the file: {e}")


# Truncate the Sphinx-Gallery index file at the first occurrence of raw HTML
def truncate_gallery_index_file(app):
    """
    This function runs at the beginning of the build process to truncate the index.rst file.
    It first checks if the file exists, and if not, it runs sphinx-gallery to generate it.
    """
    # Use the source directory path
    index_file = os.path.join(app.srcdir, "examples", "index.rst")

    # Check if the file exists
    if os.path.exists(index_file):
        # Truncate the file
        truncate_index_file_at_raw_html(index_file)
        print(f"Truncated existing file: {index_file}")
    else:
        print(
            f"File {index_file} does not exist yet. It will be generated during the build process."
        )


def setup(app):
    # Connect to the builder-inited event, which runs at the beginning of the build process
    app.connect("builder-inited", truncate_gallery_index_file)

    # Also connect to the build-finished event as a backup
    app.connect(
        "build-finished",
        lambda app, exception: (
            truncate_index_file_at_raw_html(
                os.path.join(app.srcdir, "examples", "index.rst")
            )
            if exception is None
            and os.path.exists(os.path.join(app.srcdir, "examples", "index.rst"))
            else None
        ),
    )
