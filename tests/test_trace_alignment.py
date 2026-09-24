from __future__ import annotations

import unittest

import numpy as np

from analysis_worker import (
    combine_signals_by_time,
    indexed_signal,
    lower_cuda_batch_size,
    next_cuda_batch_size,
)


class TraceAlignmentTests(unittest.TestCase):
    def test_cuda_batch_probe_doubles_to_the_requested_ceiling(self):
        self.assertEqual(next_cuda_batch_size(16, 64), 32)
        self.assertEqual(next_cuda_batch_size(128, 4096), 256)
        self.assertEqual(next_cuda_batch_size(256, 4096), 512)
        self.assertEqual(next_cuda_batch_size(48, 49), 49)
        self.assertEqual(next_cuda_batch_size(64, 64), 64)
        self.assertEqual(lower_cuda_batch_size(64), 32)
        self.assertEqual(lower_cuda_batch_size(24), 12)
        self.assertEqual(lower_cuda_batch_size(3), 2)

    def test_mixed_frame_rates_keep_each_signals_original_timestamps(self):
        time_20_fps = np.arange(227, dtype=float) / 20.0
        time_22_fps = np.arange(228, dtype=float) / 22.0
        signal_20_fps = indexed_signal(time_20_fps, np.arange(227, dtype=float))
        signal_22_fps = indexed_signal(time_22_fps, np.arange(228, dtype=float))

        combined = combine_signals_by_time({
            "B-1_Force_uN": signal_20_fps,
            "C-3_Force_uN": signal_22_fps,
        })

        b1_rows = combined.dropna(subset=["B-1_Force_uN"])
        c3_rows = combined.dropna(subset=["C-3_Force_uN"])
        np.testing.assert_allclose(b1_rows["time_s"], np.round(time_20_fps, 9))
        np.testing.assert_allclose(c3_rows["time_s"], np.round(time_22_fps, 9))
        self.assertEqual(len(b1_rows), 227)
        self.assertEqual(len(c3_rows), 228)
        self.assertAlmostEqual(float(b1_rows["time_s"].iloc[-1]), 11.3)
        self.assertAlmostEqual(float(c3_rows["time_s"].iloc[-1]), 227 / 22)


if __name__ == "__main__":
    unittest.main()
