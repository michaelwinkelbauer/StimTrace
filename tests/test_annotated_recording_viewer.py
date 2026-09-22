from __future__ import annotations

import csv
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from annotated_recording_viewer import (
    AnnotatedRecordingViewer,
    find_annotated_recording,
    playback_interval_ms,
)


class AnnotatedRecordingDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / f".test-annotated-viewer-{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def touch(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_finds_stimtrace_overlay_from_combined_trace(self):
        source = self.touch(self.root / "results" / "stimtrace_force_traces.csv")
        expected = self.touch(self.root / "results" / "overlays" / "B-2_overlay.avi")

        found = find_annotated_recording("StimTrace | B-2_Force_uN", source)

        self.assertEqual(found, expected)

    def test_prefers_point_tracking_overlay_for_point_tracking_trace(self):
        source = self.touch(self.root / "results" / "point_tracking_force_traces.csv")
        self.touch(self.root / "results" / "overlays" / "B-2_overlay.avi")
        expected = self.touch(
            self.root / "results" / "overlays" / "B-2_point_tracking_overlay.avi"
        )

        found = find_annotated_recording(
            "Point Tracking | B-2_Point_1_Force_uN",
            source,
        )

        self.assertEqual(found, expected)

    def test_finds_overlay_from_detailed_trace_subfolder(self):
        source = self.touch(
            self.root / "results" / "traces" / "C-1_stimtrace_tracking.csv"
        )
        expected = self.touch(self.root / "results" / "overlays" / "C-1_overlay.avi")

        found = find_annotated_recording("force_un", source)

        self.assertEqual(found, expected)

    def test_finds_legacy_all_overlaid_video(self):
        source = self.touch(self.root / "D7" / "P3_B-4_pillar_displacement.csv")
        expected = self.touch(
            self.root / "D7" / "all_overlaid_videos" / "P3_B-4_overlay.avi"
        )

        found = find_annotated_recording("force_un", source)

        self.assertEqual(found, expected)

    def test_finds_benchmark_video_using_settings_mapping(self):
        benchmark = self.root / "results" / "kalman_benchmark"
        source = self.touch(benchmark / "kalman_benchmark_all_traces.csv")
        settings = benchmark / "kalman_benchmark_settings.csv"
        settings.parent.mkdir(parents=True, exist_ok=True)
        with settings.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("name", "folder"))
            writer.writeheader()
            writer.writerow({"name": "Balanced", "folder": "balanced"})
        expected = self.touch(benchmark / "balanced" / "videos" / "B-2_tracked_trace.avi")

        found = find_annotated_recording("Balanced | B-2_Force_uN", source)

        self.assertEqual(found, expected)

    def test_returns_none_when_no_matching_recording_exists(self):
        source = self.touch(self.root / "results" / "stimtrace_force_traces.csv")
        self.touch(self.root / "results" / "overlays" / "C-1_overlay.avi")

        self.assertIsNone(find_annotated_recording("StimTrace | B-2_Force_uN", source))

    def test_ambiguous_benchmark_configuration_does_not_open_wrong_video(self):
        benchmark = self.root / "results" / "kalman_benchmark"
        source = self.touch(benchmark / "kalman_benchmark_all_traces.csv")
        settings = benchmark / "kalman_benchmark_settings.csv"
        settings.parent.mkdir(parents=True, exist_ok=True)
        with settings.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("name", "folder"))
            writer.writeheader()
            writer.writerow({"name": "Balanced", "folder": "balanced"})
            writer.writerow({"name": "Balanced", "folder": "balanced_2"})
        self.touch(benchmark / "balanced" / "videos" / "B-2_tracked_trace.avi")
        self.touch(benchmark / "balanced_2" / "videos" / "B-2_tracked_trace.avi")

        self.assertIsNone(
            find_annotated_recording("Balanced | B-2_Force_uN", source)
        )


class PlaybackIntervalTests(unittest.TestCase):
    def test_playback_rates_adjust_frame_interval(self):
        self.assertEqual(playback_interval_ms(20.0, 0.5), 100)
        self.assertEqual(playback_interval_ms(20.0, 1.0), 50)
        self.assertEqual(playback_interval_ms(20.0, 2.0), 25)

    def test_invalid_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            playback_interval_ms(0.0, 1.0)
        with self.assertRaises(ValueError):
            playback_interval_ms(20.0, 0.0)


class RecordingViewerLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_recording_is_fitted_without_scrollbars(self):
        root = Path.cwd() / f".test-recording-layout-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        video = root / "wide_overlay.avi"
        writer = cv2.VideoWriter(
            str(video),
            cv2.VideoWriter_fourcc(*"MJPG"),
            20.0,
            (640, 360),
        )
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
        writer.release()

        viewer = AnnotatedRecordingViewer(video)
        self.addCleanup(viewer.close)
        viewer.resize(480, 360)
        viewer.show()
        self.app.processEvents()

        pixmap = viewer.frame_label.pixmap()
        viewport = viewer.scroll_area.viewport().size()
        self.assertEqual(
            viewer.scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarAlwaysOff,
        )
        self.assertEqual(
            viewer.scroll_area.verticalScrollBarPolicy(),
            Qt.ScrollBarAlwaysOff,
        )
        self.assertFalse(viewer.scroll_area.horizontalScrollBar().isVisible())
        self.assertFalse(viewer.scroll_area.verticalScrollBar().isVisible())
        self.assertLessEqual(pixmap.width(), viewport.width())
        self.assertLessEqual(pixmap.height(), viewport.height())

    def test_viewer_can_save_current_frame_and_recording_copy(self):
        root = Path.cwd() / f".test-recording-save-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        video = root / "overlay.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        writer.write(np.full((48, 64, 3), 127, dtype=np.uint8))
        writer.release()
        viewer = AnnotatedRecordingViewer(video)
        self.addCleanup(viewer.close)
        frame_target = root / "saved_frame.png"
        recording_target = root / "saved_recording.avi"

        with patch(
            "annotated_recording_viewer.QFileDialog.getSaveFileName",
            return_value=(str(frame_target), "PNG image (*.png)"),
        ):
            viewer.save_current_frame()
        self.assertTrue(frame_target.is_file())

        with patch(
            "annotated_recording_viewer.QFileDialog.getSaveFileName",
            return_value=(str(recording_target), "Video files (*.avi)"),
        ):
            viewer.save_recording()
        self.assertTrue(recording_target.is_file())


if __name__ == "__main__":
    unittest.main()
