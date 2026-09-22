from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from colab_worker import (
    CenterTrailRenderer,
    KalmanCenter,
    kalman_innovation_gate_threshold,
    kalman_track,
    track_centers,
    trace_from_tracking,
    video_fps,
)


class ScientificCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "pixel_to_um": 2.0,
            "force_slope_un_per_um": 3.0,
            "bending_axis_mode": "fixed_angle",
            "bending_axis_angle_deg": 0.0,
            "legacy_area_normalization": False,
            "kalman_q_pos": 2.0,
            "kalman_q_vel": 36.0,
            "kalman_r": 8.0,
            "kalman_innovation_gate_enabled": True,
            "kalman_innovation_gate_confidence": 0.99,
            "kalman_innovation_gate_min_radius_px": 120.0,
        }

    def test_force_uses_one_real_diastolic_reference_and_calibrated_axis(self):
        fits = [
            (10.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (12.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (14.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
        ]
        trace = trace_from_tracking(
            fits,
            [(10.0, 5.0), (12.0, 5.0), (14.0, 5.0)],
            [20, 20, 20],
            20.0,
            self.settings,
        )
        self.assertTrue((trace["reference_center_xy"] == "10,5").all())
        np.testing.assert_allclose(trace["Force_uN"], [0.0, 12.0, 24.0])

    def test_missing_segmentation_is_not_exported_as_a_kalman_measurement(self):
        fits = [
            (10.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (np.nan, np.nan, np.nan, np.nan, np.nan, "none"),
            (14.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
        ]
        trace = trace_from_tracking(
            fits,
            [(10.0, 5.0), (12.0, 5.0), (14.0, 5.0)],
            [20, 0, 20],
            20.0,
            self.settings,
        )
        self.assertEqual(trace.loc[1, "tracking_state"], "missing")
        self.assertTrue(np.isnan(trace.loc[1, "Force_uN"]))

    def test_unfiltered_tracking_uses_raw_centers_and_preserves_missing_frames(self):
        fits = [
            (10.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (13.5, 4.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (np.nan, np.nan, np.nan, np.nan, np.nan, "none"),
        ]
        settings = {**self.settings, "tracking_filter_mode": "none"}

        centers = kalman_track(fits, 20.0, settings)
        np.testing.assert_allclose(centers[:2], [(10.0, 5.0), (13.5, 4.0)])
        self.assertTrue(np.isnan(centers[2][0]))
        self.assertTrue(np.isnan(centers[2][1]))

        trace = trace_from_tracking(fits, centers, [20, 20, 0], 20.0, settings)
        np.testing.assert_allclose(trace.loc[:1, "smooth_center_x"], [10.0, 13.5])
        self.assertTrue(np.isnan(trace.loc[2, "smooth_center_x"]))
        self.assertTrue((trace["tracking_filter_mode"] == "none").all())

    def test_covariance_gate_rejects_outlier_without_exporting_force(self):
        fits = [
            (10.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (11.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (500.0, 400.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (12.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
        ]

        tracking = track_centers(fits, 20.0, self.settings)

        self.assertEqual(
            tracking.states,
            ["measured", "measured", "innovation_rejected", "measured"],
        )
        self.assertGreater(
            tracking.innovation_mahalanobis_d2[2],
            tracking.innovation_threshold_d2,
        )
        self.assertLess(tracking.centers[2][0], 20.0)

        trace = trace_from_tracking(
            fits,
            tracking.centers,
            [20, 20, 20, 20],
            20.0,
            self.settings,
            tracking_states=tracking.states,
            innovation_mahalanobis_d2=tracking.innovation_mahalanobis_d2,
        )
        self.assertEqual(trace.loc[2, "tracking_state"], "innovation_rejected")
        self.assertEqual(trace.loc[2, "raw_center_x"], 500.0)
        self.assertTrue(np.isnan(trace.loc[2, "smooth_center_x"]))
        self.assertTrue(np.isnan(trace.loc[2, "Force_uN"]))
        self.assertTrue(trace.loc[2, "kalman_innovation_gate_enabled"])

    def test_innovation_gate_can_be_disabled_explicitly(self):
        fits = [
            (10.0, 5.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (500.0, 400.0, 4.0, 4.0, 0.0, "ellipsefit"),
        ]
        settings = {**self.settings, "kalman_innovation_gate_enabled": False}

        tracking = track_centers(fits, 20.0, settings)

        self.assertEqual(tracking.states, ["measured", "measured"])
        self.assertIsNone(tracking.innovation_threshold_d2)
        self.assertGreater(tracking.centers[1][0], 100.0)

    def test_99_percent_two_dimensional_gate_uses_chi_square_threshold(self):
        self.assertAlmostEqual(
            kalman_innovation_gate_threshold(self.settings),
            9.210340371976182,
        )

    def test_original_120_px_radius_is_the_default_covariance_gate_floor(self):
        fits = [
            (0.0, 0.0, 4.0, 4.0, 0.0, "ellipsefit"),
            (20.0, 0.0, 4.0, 4.0, 0.0, "ellipsefit"),
        ]

        compatible = track_centers(fits, 20.0, self.settings)
        pure_covariance = track_centers(
            fits,
            20.0,
            {**self.settings, "kalman_innovation_gate_min_radius_px": 0.0},
        )

        self.assertGreater(
            compatible.innovation_mahalanobis_d2[1],
            compatible.innovation_threshold_d2,
        )
        self.assertEqual(compatible.states[1], "measured")
        self.assertEqual(pure_covariance.states[1], "innovation_rejected")

    def test_innovation_covariance_expands_during_unobserved_predictions(self):
        filter_state = KalmanCenter(0.0, 0.0, 0.05, 2.0, 36.0, 8.0)
        filter_state.predict()
        early_distance = filter_state.innovation_statistics(20.0, 0.0)[2]
        for _ in range(20):
            filter_state.predict()
        late_distance = filter_state.innovation_statistics(20.0, 0.0)[2]

        self.assertLess(late_distance, early_distance)

    def test_invalid_video_fps_is_rejected(self):
        capture = Mock()
        capture.get.return_value = float("nan")
        with self.assertRaisesRegex(ValueError, "no valid frame-rate metadata"):
            video_fps(capture, "recording.avi")

    def test_segmentation_overlay_keeps_path_without_bridging_missing_centers(self):
        renderer = CenterTrailRenderer()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        renderer.render(frame, (10.0, 20.0), radius=4)
        moved = renderer.render(frame, (20.0, 20.0), radius=4)
        self.assertGreater(np.count_nonzero(moved[18:23, 10:21]), 0)

        renderer.render(frame, (np.nan, np.nan), radius=4)
        reacquired = renderer.render(frame, (50.0, 20.0), radius=4)
        self.assertGreater(np.count_nonzero(reacquired[18:23, 10:21]), 0)
        self.assertEqual(np.count_nonzero(reacquired[18:23, 30:42]), 0)


if __name__ == "__main__":
    unittest.main()
