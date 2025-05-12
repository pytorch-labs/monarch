# pyre-unsafe
import os
import pickle
import sys
import traceback
from asyncio.subprocess import create_subprocess_exec
from subprocess import CalledProcessError, Popen

from monarch.common.function import maybe_resolvable_function
from monarch_supervisor.python_executable import PYTHON_EXECUTABLE


class SubprocessError(Exception):
    def __init__(self, the_result, tb):
        self.exe = the_result
        self.tb = tb

    def __str__(self):
        context = "\n".join(self.tb.format())
        return f"A subprocess call threw an exception.\n\nTraceback from subprocess (most recent call last):\n{context}\n{type(self.exe).__name__}: {str(self.exe)}"


def call_configured(**popen_kwargs):
    async def call(fn, *args, **kwargs):
        resolvable = maybe_resolvable_function(fn)
        if resolvable is None:
            raise ValueError(f"Unsupported target for a remote call: {fn!r}")
        argread, argwrite = os.pipe()
        returnread, returnwrite = os.pipe()

        process = await create_subprocess_exec(
            PYTHON_EXECUTABLE,
            "-m",
            "morpho.subprocess",
            str(argread),
            str(returnwrite),
            pass_fds=(argread, returnwrite),
            **popen_kwargs,
        )

        os.close(argread)
        os.close(returnwrite)

        with os.fdopen(argwrite, "wb") as f:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump((resolvable, args, kwargs), f)

        return_code = await process.wait()
        if return_code != 0:
            raise CalledProcessError(return_code, ["morpho.call"])

        with os.fdopen(returnread, "rb") as f:
            # @lint-ignore PYTHONPICKLEISBAD
            success, the_result, tb = pickle.load(f)

        if success:
            return the_result
        else:
            raise SubprocessError(the_result, tb)

    return call


async def call(fn, *args, **kwargs):
    return await call_configured()(fn, *args, **kwargs)


if __name__ == "__main__":
    argread, returnwrite = sys.argv[1:]
    with os.fdopen(int(argread), "rb") as f:
        try:
            # @lint-ignore PYTHONPICKLEISBAD
            resolvable, args, kwargs = pickle.load(f)
            fn = resolvable.resolve()
            result = fn(*args, **kwargs)
            with os.fdopen(int(returnwrite), "wb") as f:
                # @lint-ignore PYTHONPICKLEISBAD
                pickle.dump((True, result, None), f)
        except Exception as e:
            with os.fdopen(int(returnwrite), "wb") as f:
                tb = traceback.extract_tb(e.__traceback__)
                # @lint-ignore PYTHONPICKLEISBAD
                pickle.dump((False, e, tb), f)
