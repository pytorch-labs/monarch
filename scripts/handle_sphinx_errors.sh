#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

echo "Diagnosing Sphinx build issues..."

# Show specific autodoc errors
echo "=== Checking for import errors ==="
python -c "
import sys
sys.path.insert(0, '../python')
problematic_modules = [
    'monarch.fetch',
    'monarch.gradient_generator',
    'monarch.notebook',
    'monarch.rust_local_mesh'
]
for mod in problematic_modules:
    try:
        __import__(mod)
        print(f'{mod} imports OK')
    except Exception as e:
        print(f'ERROR: {mod} import error: {e}')
" || true

# Show autodoc-specific errors
echo "=== Running minimal autodoc test ==="
python -c "
try:
    from sphinx.ext.autodoc import ModuleDocumenter
    from sphinx.application import Sphinx
    from sphinx.util.docutils import docutils_namespace
    import tempfile
    import os
    # Try to autodoc our problematic modules
    modules_to_test = ['monarch.fetch', 'monarch.gradient_generator']
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f'Can import {module_name} for autodoc')
        except Exception as e:
            print(f'ERROR: Cannot import {module_name}: {e}')
except Exception as e:
    print(f'ERROR: Autodoc test setup failed: {e}')
" || true

echo "INFO: Continuing with partial build..."
