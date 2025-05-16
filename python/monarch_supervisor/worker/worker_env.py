# pyre-unsafe
import runpy
import sys

__MONARCH_WORKER_ENV__ = True


def main() -> None:
    assert sys.argv[1] == "-m"
    main_module = sys.argv[2]

    # Remove the -m and the main module from the command line arguments before
    # forwarding
    sys.argv[1:] = sys.argv[3:]
    # pyre-fixme[16]: Module `runpy` has no attribute `_run_module_as_main`.
    runpy._run_module_as_main(main_module, alter_argv=False)


if __name__ == "__main__":
    main()
