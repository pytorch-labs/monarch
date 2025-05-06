"""Test suite for ExecutionTimer class."""
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import time
import unittest
from unittest.mock import MagicMock, patch

from monarch.timer.execution_timer import ExecutionTimer


class TestExecutionTimer(unittest.TestCase):
    """Test suite for the ExecutionTimer class."""

    def setUp(self) -> None:
        """Reset the profiler state before each test."""
        ExecutionTimer.reset()

    def test_basic_timing(self) -> None:
        """Test basic CUDA timing functionality."""
        with ExecutionTimer.time("test_section"):
            time.sleep(0.01)  # Sleep for 10ms

        # Get the stats
        stats = ExecutionTimer.summary()

        # Check that our section exists
        self.assertIn("test_section", stats)

        # Check timing (should be at least 10ms, allow some overhead)
        section_stats = stats["test_section"]
        self.assertEqual(section_stats["count"], 1)
        self.assertGreaterEqual(section_stats["mean_ms"], 10)  # At least 10ms
        self.assertLess(section_stats["mean_ms"], 50)  # Reasonable upper bound

    def test_multiple_timing_same_section(self) -> None:
        """Test timing the same section multiple times."""
        for _ in range(5):
            with ExecutionTimer.time("repeated_section"):
                time.sleep(0.01)

        stats = ExecutionTimer.summary()
        self.assertIn("repeated_section", stats)

        section_stats = stats["repeated_section"]
        self.assertEqual(section_stats["count"], 5)
        self.assertGreaterEqual(section_stats["mean_ms"], 10)
        self.assertGreaterEqual(section_stats["total_ms"], 50)

    def test_reset(self) -> None:
        """Test that reset clears all timing data."""
        with ExecutionTimer.time("before_reset"):
            time.sleep(0.01)

        # Verify the section exists
        stats_before = ExecutionTimer.summary()
        self.assertIn("before_reset", stats_before)

        # Reset the profiler
        ExecutionTimer.reset()

        # Timing section should be gone
        stats_after = ExecutionTimer.summary()
        self.assertNotIn("before_reset", stats_after)

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.Event")
    def test_cuda_mocked(
        self, mock_event: MagicMock, mock_is_available: MagicMock
    ) -> None:
        """Test CUDA timing with mocked CUDA functions."""

        mock_is_available.return_value = True

        mock_event_instance = MagicMock()
        mock_event_instance.elapsed_time.return_value = 15.0  # 15ms
        mock_event.return_value = mock_event_instance

        with ExecutionTimer.time("mocked_cuda"):
            time.sleep(0.01)

        stats = ExecutionTimer.summary()

        # Should have CUDA timings
        self.assertIn("mocked_cuda", stats)
        self.assertEqual(stats["mocked_cuda"]["mean_ms"], 15.0)

    def test_get_latest_measurement(self) -> None:
        """Test get_latest_measurement."""
        with ExecutionTimer.time("latest_measurement_test"):
            time.sleep(0.01)  # Sleep for 10ms

        # Get the latest measurement
        latest_measurement = ExecutionTimer.get_latest_measurement(
            "latest_measurement_test"
        )

        self.assertGreaterEqual(latest_measurement, 5)
        self.assertLess(latest_measurement, 50)  # Reasonable upper bound

        # Test for a non-existent section
        non_existent_measurement = ExecutionTimer.get_latest_measurement(
            "non_existent_section"
        )
        self.assertEqual(non_existent_measurement, 0.0)

    def test_cpu_timing(self) -> None:
        """Test CPU timing functionality."""
        with ExecutionTimer.time("cpu_section", use_cpu=True):
            time.sleep(0.01)  # Sleep for 10ms
        # Get the stats
        stats = ExecutionTimer.summary()

        # Check that our section exists
        self.assertIn("cpu_section", stats)

        # Check timing (should be at least 10ms, allow some overhead)
        section_stats = stats["cpu_section"]
        self.assertEqual(section_stats["count"], 1)
        self.assertGreaterEqual(section_stats["mean_ms"], 10)  # At least 10ms
        self.assertLess(section_stats["mean_ms"], 50)  # Reasonable upper bound


if __name__ == "__main__":
    unittest.main()
