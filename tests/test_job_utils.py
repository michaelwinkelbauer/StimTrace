from __future__ import annotations

import unittest
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from job_utils import (
    duplicate_output_stems,
    analysis_trace_file,
    elapsed_seconds,
    format_elapsed,
    job_history_identity,
    parse_timestamp,
    result_download_target,
    safe_path_component,
    video_files_in_folder,
)


class JobUtilsTests(unittest.TestCase):
    def test_duplicate_output_stems_are_case_insensitive_and_stable(self):
        paths = [
            Path("first") / "B-1.avi",
            Path("second") / "b-1.AVI",
            Path("third") / "C-1.avi",
            Path("fourth") / "B-1.avi",
        ]
        self.assertEqual(duplicate_output_stems(paths), ["b-1.AVI"])

    def test_duplicate_output_stems_detect_different_video_extensions(self):
        paths = [Path("first") / "sample.avi", Path("second") / "SAMPLE.mp4"]
        self.assertEqual(duplicate_output_stems(paths), ["SAMPLE.mp4"])

    def test_video_folder_finds_supported_direct_files_in_stable_order(self):
        folder = Path.cwd() / f".test-video-folder-{uuid.uuid4().hex}"
        folder.mkdir()
        self.addCleanup(shutil.rmtree, folder, True)
        for name in ("zeta.MP4", "Alpha.avi", "notes.txt", "movie.m4v"):
            (folder / name).touch()
        nested = folder / "results"
        nested.mkdir()
        (nested / "generated_overlay.avi").touch()

        videos = video_files_in_folder(folder)

        self.assertEqual([path.name for path in videos], ["Alpha.avi", "movie.m4v", "zeta.MP4"])

    def test_job_history_identity_prefers_drive_folder(self):
        self.assertEqual(
            job_history_identity({"folder_id": "drive-folder", "job_id": "job-1"}),
            "drive-folder",
        )
        self.assertEqual(job_history_identity({"job_id": "local-job"}), "local-job")

    def test_elapsed_seconds_accepts_local_offset(self):
        start = "2026-08-04T10:00:00+02:00"
        end = datetime(2026, 8, 4, 8, 1, 5, tzinfo=timezone.utc)
        self.assertEqual(elapsed_seconds(start, end), 65.0)

    def test_invalid_timestamp_returns_none(self):
        self.assertIsNone(parse_timestamp("not-a-date"))
        self.assertIsNone(elapsed_seconds("not-a-date"))

    def test_elapsed_format(self):
        self.assertEqual(format_elapsed(5.4), "5s")
        self.assertEqual(format_elapsed(65), "1m 05s")
        self.assertEqual(format_elapsed(3661), "1h 01m 01s")
        self.assertEqual(format_elapsed(None), "")

    def test_cloud_target_is_beside_recordings(self):
        job = {
            "job_id": "study_123",
            "source_paths": [str(Path("recordings") / "a.avi")],
        }
        self.assertEqual(
            result_download_target(job),
            Path("recordings") / "results" / "study_123",
        )

    def test_local_target_must_exist(self):
        job = {"backend": "local", "local_results_path": "missing-results"}
        self.assertIsNone(result_download_target(job))

    def test_path_component_normalization(self):
        self.assertEqual(safe_path_component(" Study A/B ", fallback="study"), "Study_A_B")
        self.assertEqual(safe_path_component("***", fallback="study"), "study")

    def test_trace_preference_and_single_traversal_result(self):
        root = Path("results")
        legacy = root / "kalman_benchmark_all_traces.xlsx"
        preferred = root / "nested" / "kalman_benchmark_all_traces.csv"
        combined = root / "combined_force_results.csv"
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "rglob", return_value=iter([legacy, preferred, combined])),
        ):
            self.assertEqual(analysis_trace_file(root), preferred)

    def test_new_stimtrace_force_file_is_preferred_over_legacy_combined_file(self):
        root = Path("results")
        preferred = root / "stimtrace_force_traces.csv"
        legacy = root / "combined_force_results.csv"
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "rglob", return_value=iter([legacy, preferred])),
        ):
            self.assertEqual(analysis_trace_file(root), preferred)


if __name__ == "__main__":
    unittest.main()
