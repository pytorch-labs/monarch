# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import types

from pathlib import Path

import monarch


def load_module_from_path(base_path, module_specifier):
    parts = module_specifier.split(".")
    file_path = str(Path(base_path).joinpath(*parts).with_suffix(".pyi"))
    loader = importlib.machinery.SourceFileLoader(module_specifier, file_path)
    spec = importlib.util.spec_from_file_location(
        module_specifier, file_path, loader=loader
    )

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError:
        return None
    return module


def patch_class(rust_entry, python_entry):
    for name, implementation in python_entry.__dict__.items():
        if hasattr(rust_entry, name):
            # do not patch in the stub methods that
            # are already defined by the rust implementation
            continue
        if not callable(implementation):
            continue
        setattr(rust_entry, name, implementation)


def patch_module(rust, python):
    for name in dir(rust):
        python_entry = getattr(python, name, None)
        if python_entry is None:
            continue
        rust_entry = getattr(rust, name)
        if not isinstance(rust_entry, type):
            continue
        patch_class(rust_entry, python_entry)


def add_extension_methods(bindings: types.ModuleType):
    """
    When we bind a rust struct into Python, it is sometimes faster to implement
    parts of the desired Python API in Python. It is also easier to understand
    what the class does in terms of these methods.

    We also want to avoid having to wrap rust objects in another layer of python objects
    because:
    * wrappers double the python overhead
    * it is easy to confuse which level of wrappers and API takes, especially
      along the python<->rust boundary.

    To avoid wrappers we first define the class in pyo3.
    We then write the python typing stubs in the pyi file for the functions rust defined.
    We also add any python extension methods, including their implementation,
    to the stub files.

    This function then loads the stub files and patch the real rust implementation
    with those typing methods.

    Using the stub files themselves can seem like an odd choice but has a lot of
    desirable properties:

    * we get accurate typechecking in:
       - the implementation of extension methods
       - the use of rust methods
       - the use of extension methods
    * go to definition in the IDE will go to the stub file, so it is easy to find
      the python impelmentations compared to putting them somewhere else
    * With no wrappers, any time rust code returns a class defined this way,
      it automatically gains its extension methods.
    """
    base_path = str(Path(monarch.__file__).parent.parent)

    def scan(module):
        for item in dir(module):
            value = getattr(module, item, None)
            if isinstance(value, types.ModuleType):
                scan(value)

        python = load_module_from_path(base_path, module.__name__)
        if python is not None:
            patch_module(module, python)

    scan(bindings)
