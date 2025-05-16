#include <Python.h>
// @lint-ignore CLANGTIDY facebook-hte-RelativeInclude
#include "mock_cuda.h"

static PyMethodDef _C_methods[] = {
    {"patch_cuda",
     patch_cuda,
     METH_NOARGS,
     "Initialize the monarch cuda patch."},
    {"mock_cuda", mock_cuda, METH_NOARGS, "Enable cuda mocking."},
    {"unmock_cuda", unmock_cuda, METH_NOARGS, "Disable cuda mocking."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _C_module = {
    PyModuleDef_HEAD_INIT,
    "_C",
    "A module containing monarch C++ functionality.",
    -1,
    _C_methods,
    NULL,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit__C(void) {
  return PyModule_Create(&_C_module);
}
