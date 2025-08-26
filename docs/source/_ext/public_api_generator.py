# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simplified Sphinx extension to generate public API documentation using templates.
No runtime imports or complex inspection - lets Sphinx handle everything.
"""

import os
from typing import Any, Dict, List, Tuple

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def get_public_api_info(
    package_name: str,
) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
    """
    Extract public API information from a package's __init__.py file.
    Only imports the package to get static data - no deep inspection.

    Returns:
        Tuple of (__all__ list, _public_api dict)
    """
    try:
        # Import the package to get __all__ and _public_api
        package = __import__(package_name)

        # Get __all__ list
        all_list = getattr(package, "__all__", [])

        # Get _public_api dict
        public_api_dict = getattr(package, "_public_api", {})

        return all_list, public_api_dict

    except Exception as e:
        logger.warning(f"Failed to import {package_name}: {e}")
        return [], {}


def categorize_apis(
    all_list: List[str], public_api_dict: Dict[str, Tuple[str, str]]
) -> Dict[str, List[str]]:
    """
    Categorize APIs into logical groups for better navigation.
    Simple categorization based on naming patterns and module paths.
    """
    categories = {
        "Core Components": [],
        "Mesh Creation & Management": [],
        "Data Operations": [],
        "Remote Execution": [],
        "System & Utilities": [],
    }

    for api_name in all_list:
        if api_name not in public_api_dict:
            categories["System & Utilities"].append(api_name)
            continue

        module_path, _ = public_api_dict[api_name]

        # Categorize based on module path and API name patterns
        if any(
            keyword in module_path
            for keyword in [
                "device_mesh",
                "stream",
                "pipe",
                "future",
                "shape",
                "selection",
            ]
        ):
            categories["Core Components"].append(api_name)
        elif any(keyword in module_path for keyword in ["mesh", "notebook", "world"]):
            categories["Mesh Creation & Management"].append(api_name)
        elif any(keyword in module_path for keyword in ["tensor", "fetch", "gradient"]):
            categories["Data Operations"].append(api_name)
        elif any(keyword in module_path for keyword in ["remote", "invocation"]):
            categories["Remote Execution"].append(api_name)
        else:
            categories["System & Utilities"].append(api_name)

    # Remove empty categories and sort APIs within each category
    return {k: sorted(v) for k, v in categories.items() if v}


def generate_api_rst_content(
    all_list: List[str], public_api_dict: Dict[str, Tuple[str, str]], package_name: str
) -> str:
    """
    Generate RST with navigation and categorized content.
    """

    # Categorize the APIs
    categories = categorize_apis(all_list, public_api_dict)

    content = f"""Python API
==========

.. note::
   This documents {package_name.title()}'s **public APIs** - the stable, supported interfaces.

   All can be imported as: ``from {package_name} import <name>``

.. contents:: Quick Navigation
   :local:
   :depth: 2

"""

    # Generate content for each category
    for category_name, apis in categories.items():
        if not apis:
            continue

        content += f"{category_name}\n"
        content += "=" * len(category_name) + "\n\n"

        # Add a brief category description
        if category_name == "Core Components":
            content += "Core building blocks for distributed computing: meshes, streams, futures, and data structures.\n\n"
        elif category_name == "Mesh Creation & Management":
            content += "Functions for creating and managing different types of compute meshes.\n\n"
        elif category_name == "Data Operations":
            content += "Operations for data processing, fetching, and gradient computation.\n\n"
        elif category_name == "Remote Execution":
            content += (
                "Remote function execution and distributed computing primitives.\n\n"
            )
        elif category_name == "System & Utilities":
            content += (
                "System utilities, allocators, configuration, and helper functions.\n\n"
            )

        for api_name in apis:
            if api_name in public_api_dict:
                module_path, attr_name = public_api_dict[api_name]
                # Remove :noindex: so public APIs become primary cross-reference targets
                content += f".. autofunction:: {module_path}.{attr_name}\n\n"
            else:
                content += f".. autofunction:: {package_name}.{api_name}\n\n"

    # Add alphabetical index at the end
    content += """Alphabetical Index
==================

All APIs in alphabetical order:

"""

    for api_name in sorted(all_list):
        content += f"* :py:func:`~{package_name}.{api_name}`\n"

    content += "\n"

    return content


def generate_public_api_docs(app: Sphinx, config: Any) -> None:
    """
    Generate public API documentation based on __all__ and _public_api.
    """
    package_name = getattr(config, "public_api_source_module", "monarch")
    output_file = getattr(config, "public_api_output_file", "api/index.rst")

    logger.info(f"Generating public API documentation for {package_name}")

    # Get public API information
    all_list, public_api_dict = get_public_api_info(package_name)

    if not all_list:
        logger.warning(f"No public API found in {package_name}.__all__")
        return

    logger.info(f"Found {len(all_list)} public APIs in {package_name}")

    # Generate RST content with navigation and categories
    rst_content = generate_api_rst_content(all_list, public_api_dict, package_name)

    # Write to output file
    output_path = os.path.join(app.srcdir, output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write(rst_content)

    logger.info(f"Generated public API documentation: {output_path}")


def setup(app: Sphinx) -> Dict[str, Any]:
    """Setup the Sphinx extension."""

    # Add configuration values
    app.add_config_value("public_api_source_module", "monarch", "env")
    app.add_config_value("public_api_output_file", "api/index.rst", "env")

    # Connect to the config-inited event to generate docs early in the build process
    app.connect("config-inited", generate_public_api_docs)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
