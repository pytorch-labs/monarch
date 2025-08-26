#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import sys


def get_public_api_info():
    """Get the public API information from monarch.__init__.py"""
    try:
        # Add python directory to path
        sys.path.insert(0, "python")
        import monarch

        all_list = getattr(monarch, "__all__", [])
        public_api_dict = getattr(monarch, "_public_api", {})

        return all_list, public_api_dict
    except Exception as e:
        print(f"Warning: Could not import monarch to get public API: {e}")
        return [], {}


def main():
    print("=========================================")
    print("Verifying Public API Documentation Generation")
    print("=========================================")

    # Get public API information
    all_list, public_api_dict = get_public_api_info()

    if all_list:
        print(f"Found {len(all_list)} public APIs in monarch.__all__")
    else:
        print("Warning: Could not determine public API from monarch.__all__")

    # Primary API files that should be generated
    api_files = [
        "docs/build/html/api/index.html",  # The main public API documentation
    ]

    total_functions = 0
    total_classes = 0
    total_methods = 0
    missing_files = []
    empty_files = []
    success_files = []

    print("Checking API documentation files...")

    for file_path in api_files:
        filename = os.path.basename(file_path)

        if not os.path.exists(file_path):
            print(f"ERROR: {filename} was not generated")
            missing_files.append(filename)
            continue

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"ERROR: {filename} could not be read: {e}")
            continue

        # Look for signs of successful autodoc generation
        has_content = any(
            [
                'class="sig-name descname"' in content,
                'class="sig-name"' in content,
                '<dt class="sig' in content,
                '<dl class="py' in content,  # Modern Sphinx structure
            ]
        )

        if has_content:
            print(f"{filename} contains API content")
            success_files.append(filename)
        else:
            print(f"WARNING: {filename} may be missing API content")
            empty_files.append(filename)

        # Count function/class/method signatures with multiple patterns
        func_patterns = [
            r'<dt class="sig sig-object py"[^>]*function',
            r'<dl class="py function"',
            r'class="py function"',
        ]

        class_patterns = [
            r'<dt class="sig sig-object py"[^>]*class',
            r'<dl class="py class"',
            r'class="py class"',
        ]

        method_patterns = [
            r'<dt class="sig sig-object py"[^>]*method',
            r'<dl class="py method"',
            r'class="py method"',
        ]

        # Count using all patterns and take the maximum
        func_count = max(
            [len(re.findall(pattern, content)) for pattern in func_patterns]
        )
        class_count = max(
            [len(re.findall(pattern, content)) for pattern in class_patterns]
        )
        method_count = max(
            [len(re.findall(pattern, content)) for pattern in method_patterns]
        )

        total_functions += func_count
        total_classes += class_count
        total_methods += method_count

        print(
            f"   Functions: {func_count}, Classes: {class_count}, Methods: {method_count}"
        )

        # Check for specific content that indicates successful autodoc
        if content and len(content) > 1000:  # Reasonable size check
            signature_count = len(re.findall(r'<dt class="sig', content))
            if signature_count > 0:
                print(f"   Found {signature_count} API signatures")
            else:
                print("   WARNING: No API signatures detected")

    # Check if public API documentation contains the expected APIs
    if all_list and os.path.exists("docs/build/html/api/index.html"):
        with open("docs/build/html/api/index.html", "r", encoding="utf-8") as f:
            public_api_content = f.read()

        documented_apis = 0
        missing_apis = []

        for api_name in all_list:
            # Look for the API name in the documentation
            if api_name in public_api_content:
                documented_apis += 1
            else:
                missing_apis.append(api_name)

        print("\nPublic API Coverage:")
        print(f"   Expected APIs: {len(all_list)}")
        print(f"   Documented APIs: {documented_apis}")
        print(f"   Missing APIs: {len(missing_apis)}")

        if missing_apis and len(missing_apis) <= 5:  # Show a few missing APIs
            print(f"   Some missing: {missing_apis[:5]}")
        elif missing_apis:
            print(f"   Many APIs missing (showing first 5): {missing_apis[:5]}")

    # Summary report
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)

    print(f"Total files checked: {len(api_files)}")
    print(f"Files with content: {len(success_files)}")
    print(f"Files possibly empty: {len(empty_files)}")
    print(f"Files missing: {len(missing_files)}")

    print("\nTotal API elements found:")
    print(f"   Functions: {total_functions}")
    print(f"   Classes: {total_classes}")
    print(f"   Methods: {total_methods}")
    print(f"   TOTAL: {total_functions + total_classes + total_methods}")

    if missing_files:
        print(f"\nERROR: Missing files: {missing_files}")

    if empty_files:
        print(f"\nWARNING: Possibly empty files: {empty_files}")
        print(
            "   These files exist but may not contain properly generated API content."
        )

    if success_files:
        print(f"\nSuccessfully generated: {success_files}")

    # Determine overall success
    if missing_files:
        print("\nVERIFICATION FAILED: Some documentation files were not generated")
        return 1

    if len(empty_files) == len(api_files):
        print("\nVERIFICATION FAILED: All API files appear to be empty")
        return 1

    if total_functions + total_classes + total_methods == 0:
        print("\nVERIFICATION FAILED: No API elements found in any files")
        return 1

    if len(success_files) >= len(api_files) // 2:  # At least half should have content
        print("\nVERIFICATION PASSED: API documentation successfully generated!")
        return 0
    else:
        print("\nVERIFICATION PARTIAL: Some API documentation may be incomplete")
        return 0  # Don't fail on partial success


def check_import_fallback():
    """Fallback check if files don't exist - test imports directly"""
    print("\nRunning fallback import verification...")

    import sys

    sys.path.insert(0, "python")

    modules = [
        "monarch.fetch",
        "monarch.gradient_generator",
        "monarch.notebook",
        "monarch.rust_local_mesh",
    ]

    for mod in modules:
        try:
            __import__(mod)
            print(f"{mod} imports OK")
        except Exception as e:
            print(f"ERROR: {mod} import error: {e}")


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as e:
        print(f"\nERROR: Verification script failed: {e}")
        check_import_fallback()
        exit_code = 1

    sys.exit(exit_code)
