from .._lib import runtime  # @manual=//monarch/monarch_extension:monarch_extension

# Re-export the sleep_indefinitely_for_unit_tests function
sleep_indefinitely_for_unit_tests = runtime.sleep_indefinitely_for_unit_tests

__all__ = ["sleep_indefinitely_for_unit_tests"]
