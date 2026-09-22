from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from pixel_calibration import calibration_summary, ellipse_measurements, micrometers_per_pixel
from pixel_calibration import PixelCalibrationDialog
from desktop_app import choose_calibration_videos


class PixelCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_known_width_calibration(self):
        measurements = ellipse_measurements(1310.0, 900.0)
        self.assertAlmostEqual(
            micrometers_per_pixel(2500.0, measurements["width"]),
            2500.0 / 1310.0,
        )

    def test_circle_perimeter_uses_full_circumference(self):
        measurements = ellipse_measurements(100.0, 100.0)
        self.assertAlmostEqual(measurements["perimeter"], math.pi * 100.0)

    def test_invalid_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            ellipse_measurements(0.0, 10.0)
        with self.assertRaises(ValueError):
            micrometers_per_pixel(100.0, 0.0)

    def test_nine_measurements_are_summarized_with_variability(self):
        values = [2.0, 2.1, 1.9] * 3
        mean, standard_deviation, variation = calibration_summary(values)
        self.assertAlmostEqual(mean, 2.0)
        self.assertGreater(standard_deviation, 0.0)
        self.assertAlmostEqual(variation, standard_deviation / mean * 100.0)

    def test_calibration_uses_three_unique_analysis_videos(self):
        available = [Path(f"recording_{index}.avi") for index in range(8)]
        selected = choose_calibration_videos(available)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertTrue(set(selected).issubset(available))

    def test_calibration_defers_to_file_selection_with_too_few_videos(self):
        self.assertIsNone(choose_calibration_videos([]))
        self.assertIsNone(choose_calibration_videos([Path("one.avi"), Path("two.avi")]))

    def test_dialog_does_not_override_qwidget_metric_method(self):
        dialog = PixelCalibrationDialog(4.35)
        self.assertTrue(callable(dialog.metric))
        self.assertIsInstance(dialog.metric_selector, QWidget)
        dialog.show_frame(np.zeros((120, 160, 3), dtype=np.uint8), reset_ellipse=True)
        self.app.processEvents()
        self.assertIsNotNone(dialog.selected_measurements())
        original_artists = tuple(dialog.selector.artists)
        dialog.show_frame(np.zeros((120, 160, 3), dtype=np.uint8), reset_ellipse=True)
        axes_children = set(dialog.axes.get_children())
        self.assertTrue(all(artist not in axes_children for artist in original_artists))
        dialog.zoom_on_scroll(SimpleNamespace(
            inaxes=dialog.axes,
            xdata=80.0,
            ydata=60.0,
            button="up",
        ))
        zoomed_xlim = dialog.axes.get_xlim()
        self.assertLess(abs(zoomed_xlim[1] - zoomed_xlim[0]), 160.0)
        dialog.show_frame(np.ones((120, 160, 3), dtype=np.uint8), reset_ellipse=False)
        self.assertEqual(dialog.axes.get_xlim(), zoomed_xlim)
        dialog.discard_recorded_ellipse()
        self.assertIsNone(dialog.selector)
        dialog.show_frame(np.ones((120, 160, 3), dtype=np.uint8), reset_ellipse=False)
        self.assertIsNone(dialog.selected_measurements())
        self.assertFalse(dialog.record_button.isEnabled())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
