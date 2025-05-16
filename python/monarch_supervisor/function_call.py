# pyre-unsafe
import importlib.util
import sys

from monarch_supervisor import _FunctionCall, get_message_queue

if __name__ == "__main__":
    q = get_message_queue()
    _, call = q.recv()
    assert isinstance(call, _FunctionCall)
    filename, *rest = call.target.split(":", 1)
    if not rest:
        modulename, funcname = filename.rsplit(".", 1)
        module = importlib.import_module(modulename)
    else:
        spec = importlib.util.spec_from_file_location("__entry__", filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # pyre-ignore[16]
        spec.loader.exec_module(module)
        sys.modules["__entry__"] = module
        funcname = rest[0]
    func = getattr(module, funcname)
    func(*call.args, **call.kwargs)
