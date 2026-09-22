import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from point_tracking import (
    AsyncTrackingOverlayWriter,
    TrackingTrailRenderer,
    TrackingOptions,
    VideoSelection,
    absolute_distance,
    build_distance_table,
    calculate_measurements,
    combine_manual_and_automatic_points,
    detect_features,
    derive_projection_axis,
    relative_distance,
    write_combined_force_traces,
    write_tracking_overlay,
    write_tracking_outputs,
)


def options(**overrides):
    values = {
        "strategy": "manual",
        "measurement": "absolute",
        "baseline": "auto_relaxed",
        "px_per_um": 2.0,
        "force_slope_un_per_um": 6.0,
        "force_model_name": "Default pillar",
        "fps_override": None,
        "threshold_percentile": 80.0,
        "save_coordinates": True,
        "save_plots": False,
    }
    values.update(overrides)
    return TrackingOptions(**values)


class PointTrackingTests(unittest.TestCase):
    def test_hybrid_points_keep_manual_first_and_skip_nearby_auto_duplicates(self):
        points, sources = combine_manual_and_automatic_points(
            [(10.0, 10.0)],
            np.array([[12.0, 11.0], [40.0, 50.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(points, [[10.0, 10.0], [40.0, 50.0]])
        self.assertEqual(sources, ["manual", "automatic"])

    def test_automatic_features_are_restricted_to_roi(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (5, 5), (35, 35), (255, 255, 255), -1)
        cv2.rectangle(frame, (65, 45), (78, 58), (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 75), (113, 88), (255, 255, 255), -1)
        points = detect_features(frame, roi=(50, 35, 80, 65), minimum=2)
        self.assertGreaterEqual(len(points), 2)
        self.assertLessEqual(len(points), 10)
        self.assertTrue(np.all((points[:, 0] >= 50) & (points[:, 0] < 130)))
        self.assertTrue(np.all((points[:, 1] >= 35) & (points[:, 1] < 100)))

    def test_auto_relaxed_uses_one_real_trajectory_position(self):
        trajectories = np.array([[[10, 0], [8, 0], [6, 0], [8, 0], [10, 0]]], dtype=float)
        distances, references = absolute_distance(trajectories, "auto_relaxed")
        self.assertEqual(references, [2])
        np.testing.assert_allclose(distances[0], [4, 2, 0, 2, 4])

    def test_relative_distance_is_euclidean_separation(self):
        trajectories = np.array(
            [
                [[0, 0], [1, 1], [2, 2]],
                [[3, 4], [1, 5], [2, 8]],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(relative_distance(trajectories), [5, 4, 6])

    def test_measurements_are_calibrated_to_micrometers(self):
        trajectories = np.array([[[0, 0], [2, 0], [4, 0]]], dtype=float)
        measurements, metadata = calculate_measurements(
            trajectories,
            options(baseline="first_frame"),
        )
        np.testing.assert_allclose(measurements["Point_1_distance_um"], [0, 1, 2])
        np.testing.assert_allclose(measurements["Point_1_Force_uN"], [0, 6, 12])
        self.assertEqual(metadata["baseline_indices"], [0])
        self.assertEqual(metadata["force_axis_projection"], "automatic_pca")
        np.testing.assert_allclose([metadata["force_axis_x"], metadata["force_axis_y"]], [1.0, 0.0])
        self.assertEqual(metadata["force_additional_zeroing"], "none")

    def test_hybrid_motion_cutoff_never_removes_manual_points(self):
        trajectories = np.array(
            [
                [[0, 0], [1, 0], [2, 0]],
                [[0, 0], [2, 0], [4, 0]],
                [[0, 0], [5, 0], [10, 0]],
            ],
            dtype=float,
        )
        measurements, metadata = calculate_measurements(
            trajectories,
            options(
                strategy="hybrid",
                baseline="first_frame",
                threshold_percentile=80.0,
            ),
            manual_point_count=1,
        )

        self.assertIn("Point_1_distance_um", measurements)
        self.assertNotIn("Point_2_distance_um", measurements)
        self.assertIn("Point_3_distance_um", measurements)
        self.assertEqual(metadata["selected_point_indices"], [0, 2])

    def test_axis_projection_removes_perpendicular_motion_from_absolute_force(self):
        trajectories = np.array([[[0, 0], [2, 0.8], [4, -0.7]]], dtype=float)
        measurements, metadata = calculate_measurements(
            trajectories, options(baseline="first_frame"),
        )
        axis = np.array([metadata["force_axis_x"], metadata["force_axis_y"]])
        np.testing.assert_allclose(axis, derive_projection_axis(trajectories))
        # The PCA axis is horizontal here; perpendicular vertical jitter is excluded.
        np.testing.assert_allclose(measurements["Point_1_distance_um"], [0.0, 1.0, 2.0], atol=0.15)

    def test_distance_table_has_analyzer_compatible_time_column(self):
        table = build_distance_table({"Point_1_distance_um": np.array([0, 2, 4])}, 20.0)
        self.assertEqual(list(table.columns), ["time_s", "Point_1_distance_um"])
        np.testing.assert_allclose(table["time_s"], [0, 0.05, 0.1])

    def test_export_writes_one_detailed_trace_with_all_point_signals(self):
        trajectories = np.array([[[0, 1], [2, 3], [4, 5]]], dtype=float)
        selection = VideoSelection(Path("recording.avi"), 20.0, 3, 10, 10, points=[(0, 1)])
        folder = Path.cwd() / f".test-point-tracking-{uuid.uuid4().hex}"
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder, True)
        outputs = write_tracking_outputs(
            folder,
            selection,
            trajectories,
            {
                "Point_1_Force_uN": np.array([0, 6, 12]),
                "Point_1_distance_um": np.array([0, 1, 2]),
            },
            {"baseline_indices": [0]},
            20.0,
            options(),
        )
        trace = pd.read_csv(outputs["trace_csv"])
        self.assertTrue(outputs["trace_csv"].endswith("traces\\recording_point_tracking.csv"))
        self.assertIn("Point_1_x_px", trace.columns)
        self.assertIn("Point_1_distance_um", trace.columns)
        self.assertIn("Point_1_Force_uN", trace.columns)
        self.assertIn("Point_1_tracking_state", trace.columns)
        self.assertEqual(outputs["force_columns"], ["Point_1_Force_uN"])

    def test_combined_force_traces_align_mixed_frame_rates(self):
        folder = Path.cwd() / f".test-point-tracking-{uuid.uuid4().hex}"
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder, True)
        traces = folder / "traces"
        traces.mkdir()
        first = traces / "A_point_tracking.csv"
        second = traces / "B_point_tracking.csv"
        pd.DataFrame({
            "time_s": [0.0, 0.05, 0.1],
            "Point_1_Force_uN": [1.0, 2.0, 3.0],
        }).to_csv(first, index=False)
        pd.DataFrame({
            "time_s": [0.0, 1 / 22, 2 / 22],
            "Point_1_Force_uN": [4.0, 5.0, 6.0],
        }).to_csv(second, index=False)
        combined_path = write_combined_force_traces(
            [
                {"recording_name": "A", "trace_csv": str(first), "force_columns": ["Point_1_Force_uN"]},
                {"recording_name": "B", "trace_csv": str(second), "force_columns": ["Point_1_Force_uN"]},
            ],
            folder,
        )
        combined = pd.read_csv(combined_path)
        self.assertEqual(
            list(combined.columns),
            ["time_s", "A_Point_1_Force_uN", "B_Point_1_Force_uN"],
        )
        self.assertEqual(len(combined), 5)

    def test_overlay_preserves_effective_fps_and_trace_length(self):
        folder = Path.cwd() / f".test-point-tracking-{uuid.uuid4().hex}"
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder, True)
        source = folder / "source.avi"
        writer = cv2.VideoWriter(
            str(source),
            cv2.VideoWriter_fourcc(*"MJPG"),
            20.0,
            (64, 48),
        )
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        selection = VideoSelection(source, 20.0, 4, 64, 48, points=[(10, 10)])
        trajectories = np.array([[[10, 10], [11, 10], [12, 10], [13, 10]]], dtype=float)
        overlay = write_tracking_overlay(
            folder / "overlays" / "source_point_tracking_overlay.avi",
            selection,
            trajectories,
            20.0,
        )
        capture = cv2.VideoCapture(str(overlay))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
        self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 20.0, places=2)
        ok, rendered = capture.read()
        self.assertTrue(ok)
        self.assertGreater(np.count_nonzero(rendered), 0)
        capture.release()

    def test_async_overlay_writer_encodes_all_submitted_frames(self):
        folder = Path.cwd() / f".test-point-tracking-{uuid.uuid4().hex}"
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder, True)
        overlay = folder / "parallel_overlay.avi"
        writer = AsyncTrackingOverlayWriter(overlay, 25.0, (64, 48))
        for index in range(5):
            writer.submit(
                np.zeros((48, 64, 3), dtype=np.uint8),
                np.array([[10 + index, 12]], dtype=float),
            )
        writer.close()
        capture = cv2.VideoCapture(str(overlay))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
        self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 25.0, places=2)
        ok, rendered = capture.read()
        self.assertTrue(ok)
        self.assertGreater(np.count_nonzero(rendered), 0)
        capture.release()

    def test_tracking_trails_persist_without_bridging_lost_points(self):
        renderer = TrackingTrailRenderer()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        renderer.render(frame.copy(), np.array([[10.0, 20.0]]), draw_points=False)
        moved = renderer.render(
            frame.copy(),
            np.array([[20.0, 20.0]]),
            draw_points=False,
        )
        self.assertGreater(np.count_nonzero(moved[18:23, 10:21]), 0)

        renderer.render(
            frame.copy(),
            np.array([[np.nan, np.nan]]),
            draw_points=False,
        )
        reacquired = renderer.render(
            frame.copy(),
            np.array([[50.0, 20.0]]),
            draw_points=False,
        )
        self.assertGreater(np.count_nonzero(reacquired[18:23, 10:21]), 0)
        self.assertEqual(np.count_nonzero(reacquired[18:23, 30:46]), 0)


if __name__ == "__main__":
    unittest.main()
