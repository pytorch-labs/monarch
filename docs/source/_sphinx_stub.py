# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stub module to help Sphinx import monarch modules during documentation generation.
This patches the import system to allow monarch modules to be imported while
mocking heavy dependencies like torch.
"""

import sys
from unittest import mock


class MockModule:
    """A mock module that can be used as a placeholder for complex dependencies."""
    
    def __init__(self, name):
        self.name = name
        
    def __getattr__(self, name):
        # Return a mock object for any attribute access
        return mock.MagicMock()
        
    def __call__(self, *args, **kwargs):
        # Allow the module to be called like a function
        return mock.MagicMock()


# Mock the heavy dependencies before any monarch modules try to import them
sys.modules['torch'] = MockModule('torch')
sys.modules['torch.utils'] = MockModule('torch.utils')
sys.modules['torch.utils.cpp_extension'] = MockModule('torch.utils.cpp_extension')
sys.modules['torch.utils._python_dispatch'] = MockModule('torch.utils._python_dispatch')  
sys.modules['torch.utils._pytree'] = MockModule('torch.utils._pytree')
sys.modules['torch.utils.weak'] = MockModule('torch.utils.weak')
sys.modules['numpy'] = MockModule('numpy')

# Mock torch attributes that are commonly accessed
torch_mock = sys.modules['torch']
torch_mock.Tensor = mock.MagicMock()
torch_mock.tensor = mock.MagicMock()
torch_mock.full = mock.MagicMock()
torch_mock.long = mock.MagicMock()
torch_mock._C = MockModule('torch._C')
torch_mock._C._GLIBCXX_USE_CXX11_ABI = 1

# Mock torch.utils.cpp_extension functions
cpp_ext_mock = sys.modules['torch.utils.cpp_extension']
cpp_ext_mock.BuildExtension = mock.MagicMock()
cpp_ext_mock.CppExtension = mock.MagicMock()
cpp_ext_mock.CUDA_HOME = None
cpp_ext_mock.include_paths = mock.MagicMock(return_value=[])
cpp_ext_mock.TORCH_LIB_PATH = ""

# Mock other dispatch-related modules
sys.modules['torch.utils._python_dispatch'].TorchDispatchMode = mock.MagicMock()
sys.modules['torch.utils._pytree'].tree_map = mock.MagicMock()
sys.modules['torch.utils.weak'].weakref = mock.MagicMock()
