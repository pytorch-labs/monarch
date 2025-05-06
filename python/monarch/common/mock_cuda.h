#pragma once

#include <Python.h>

PyObject* patch_cuda(PyObject*, PyObject*);
PyObject* mock_cuda(PyObject*, PyObject*);
PyObject* unmock_cuda(PyObject*, PyObject*);
