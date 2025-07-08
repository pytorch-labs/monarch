/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Python.h>

PyObject* patch_cuda(PyObject*, PyObject*);
PyObject* mock_cuda(PyObject*, PyObject*);
PyObject* unmock_cuda(PyObject*, PyObject*);
