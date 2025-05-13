#!/usr/bin/env python3
# pyre-strict

"""
A simple binary that calls the sleep_indefinitely_for_unit_tests function from the monarch extension.
This is used to test the signal handling behavior of signal_safe_block_on.
"""

import sys

from monarch._monarch.runtime import sleep_indefinitely_for_unit_tests  # @manual


def main() -> None:
    print("Starting sleep_binary. Process will sleep indefinitely until interrupted.")
    sys.stdout.flush()  # Ensure the message is printed before we sleep

    try:
        # This will sleep indefinitely until interrupted by a signal
        sleep_indefinitely_for_unit_tests()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
