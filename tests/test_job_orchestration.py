from __future__ import annotations

import unittest
from pathlib import Path

from job_orchestration import (
    CloudSubmissionRequest,
    fetch_cloud_job_progress,
    make_submitted_segmentation_job,
    upload_cloud_job,
)


class FakeDrive:
    def __init__(self, progress: dict | None = None, terminal: dict | None = None):
        self.progress = progress or {"state": "queued"}
        self.terminal = terminal or {}
        self.updated = None
        self.submitted = None

    def submit(self, videos, study, settings, callback, parameters):
        self.submitted = (videos, study, settings, parameters)
        callback(0, 1.0, videos[0].name)
        return "job-1", "folder-1", "progress-1", "preview-1", "results-1"

    def job_progress(self, _progress_file_id):
        return dict(self.progress)

    def job_terminal_status(self, _folder_id):
        return dict(self.terminal)

    def update_job_progress(self, progress_file_id, payload):
        self.updated = (progress_file_id, payload)


class JobOrchestrationTests(unittest.TestCase):
    def test_upload_returns_ui_identifiers_and_reports_progress(self):
        drive = FakeDrive()
        updates = []
        request = CloudSubmissionRequest([Path("B-1.avi")], "study", object(), {"batch": 8})
        result = upload_cloud_job(drive, request, lambda *args: updates.append(args))
        self.assertEqual(result["job_id"], "job-1")
        self.assertEqual(updates, [(0, 1.0, "B-1.avi")])
        self.assertEqual(drive.submitted[3], {"batch": 8})

    def test_cancelling_job_is_resolved_and_persisted(self):
        drive = FakeDrive({"state": "cancelling"}, {"state": "complete"})
        result = fetch_cloud_job_progress(drive, "progress-1", "folder-1")
        self.assertEqual(result["progress"]["state"], "complete")
        self.assertEqual(result["progress"]["job_progress_fraction"], 1.0)
        self.assertEqual(drive.updated[0], "progress-1")

    def test_submitted_job_keeps_model_and_source_paths(self):
        job = make_submitted_segmentation_job(
            {
                "job_id": "job-1", "folder_id": "folder-1", "progress_file_id": "progress-1",
                "preview_file_id": "", "result_archive_file_id": "results-1",
            },
            {
                "study": "study", "video_names": ["B-1.avi"], "videos": [Path("C:/data/B-1.avi")],
                "submitted_at": "2026-08-13T10:00:00+00:00", "model_name": "Pillar v2",
                "overlay_mode": "local",
                "kalman_benchmark": [{"name": "Default"}],
            },
        )
        self.assertEqual(job["model_name"], "Pillar v2")
        self.assertEqual(job["source_paths"], ["C:\\data\\B-1.avi"])
        self.assertEqual(job["kalman_benchmark"], ["Default"])
        self.assertEqual(job["overlay_mode"], "local")
