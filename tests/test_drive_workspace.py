import unittest
from unittest.mock import Mock

from colab_worker import WORKER_VERSION
from desktop_app import DriveClient, MainWindow, NOTEBOOK_VERSION, Settings


class FakeRequest:
    def __init__(self, result):
        self.result = result

    def execute(self, **_kwargs):
        return self.result


class FakeFiles:
    def __init__(self):
        self.get_calls = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return FakeRequest(
            {
                "id": kwargs["fileId"],
                "name": "StimTrace existing",
                "parents": ["root"],
                "trashed": False,
            }
        )

    def create(self, **_kwargs):
        raise AssertionError("A trusted existing workspace must not recreate assets.")

    def update(self, **_kwargs):
        raise AssertionError("A trusted existing workspace must not update assets.")

    def list(self, **_kwargs):
        raise AssertionError("A trusted existing workspace must not search for known assets.")


class FakePermissions:
    def create(self, **_kwargs):
        raise AssertionError("Existing model permissions must not be recreated on every sign-in.")


class FakeService:
    def __init__(self):
        self.files_resource = FakeFiles()

    def files(self):
        return self.files_resource

    def permissions(self):
        return FakePermissions()


class DriveWorkspaceTests(unittest.TestCase):
    def test_completed_study_opens_its_combined_trace_in_signal_viewer(self):
        window = Mock()
        trace_file = Mock()
        window.result_download_target.return_value = Mock(is_dir=Mock(return_value=True))
        window.analysis_trace_file.return_value = trace_file

        MainWindow.open_job_in_signal_viewer(
            window,
            {"type": "segmentation", "state": "complete", "backend": "local"},
        )

        window.open_signal_analysis_files.assert_called_once_with([trace_file])
        window.start_result_download.assert_not_called()

    def test_cloud_study_download_continues_into_signal_viewer(self):
        window = Mock()
        window.result_download_target.return_value = None
        window.drive.service = object()
        job = {"type": "segmentation", "state": "complete", "backend": "cloud"}

        MainWindow.open_job_in_signal_viewer(window, job)

        window.start_result_download.assert_called_once_with(
            job,
            force=True,
            notify=True,
            action="analyze",
        )

    def test_interrupted_media_download_retries_from_a_clean_buffer(self):
        client = DriveClient()
        attempts = 0

        def download(buffer):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                buffer.write(b"partial")
                raise ConnectionResetError(10054, "connection reset")
            buffer.write(b"complete")

        self.assertEqual(client._download_with_retry(download, attempts=2), b"complete")
        self.assertEqual(attempts, 2)

    def test_desktop_notebook_version_tracks_worker_version(self):
        self.assertEqual(NOTEBOOK_VERSION, WORKER_VERSION)

    def test_processed_frame_count_is_monotonic_and_formatted(self):
        job = {"processed_frames": 80}
        MainWindow.update_job_processed_frames(
            job,
            {"processed_frames": 75, "total_job_frames": 227},
        )
        self.assertEqual(job["processed_frames"], 80)
        self.assertEqual(job["total_job_frames"], 227)
        self.assertEqual(MainWindow.format_processed_frames(80), "80")
        self.assertEqual(MainWindow.format_processed_frames(None), "")

    def test_processing_start_is_recorded_only_once(self):
        job = {}
        MainWindow.mark_job_processing_started(job, "2026-01-01T12:00:00+00:00")
        MainWindow.mark_job_processing_started(job, "2026-01-01T12:01:00+00:00")
        self.assertEqual(job["processing_started_at"], "2026-01-01T12:00:00+00:00")

    def test_cloud_manifest_records_the_selected_model(self):
        # The manifest field is deliberately top-level: Drive discovery reads it
        # without inferring the model from segmentation performance settings.
        settings = Settings(selected_model_name="My custom pillar", model_file_id="model-id")
        manifest_fields = {
            "model_name": settings.selected_model_name,
            "model_file_id": settings.model_file_id,
        }
        self.assertEqual(manifest_fields["model_name"], "My custom pillar")
        self.assertEqual(manifest_fields["model_file_id"], "model-id")

    def test_queued_retry_clears_previous_attempt_metrics(self):
        job = {
            "state": "failed",
            "processing_started_at": "2026-01-01T12:00:00+00:00",
            "actual_total_seconds": 45,
            "processed_frames": 80,
            "total_job_frames": 227,
        }
        MainWindow.update_job_history_metrics(job, {"state": "queued"})
        self.assertNotIn("processing_started_at", job)
        self.assertNotIn("actual_total_seconds", job)
        self.assertNotIn("processed_frames", job)
        self.assertNotIn("total_job_frames", job)

    def test_new_worker_attempt_replaces_old_frame_count(self):
        job = {
            "processing_started_at": "2026-01-01T12:00:00+00:00",
            "processed_frames": 180,
        }
        MainWindow.update_job_history_metrics(job, {
            "state": "analyzing",
            "processing_started_at": "2026-01-01T12:05:00+00:00",
            "processed_frames": 12,
            "total_job_frames": 227,
        })
        self.assertEqual(job["processed_frames"], 12)
        self.assertEqual(job["total_job_frames"], 227)
        self.assertEqual(MainWindow.format_processed_frames(12, 227), "12 / 227")

    def test_trusted_workspace_only_checks_root_folder(self):
        settings = Settings(
            drive_root_folder_id="workspace",
            workspace_folder_name="StimTrace existing",
            google_account_email="researcher@example.org",
            model_file_id="model",
            default_model_file_id="model",
            worker_file_id="worker",
            notebook_file_id="notebook",
            marker_file_id="marker",
            worker_status_file_id="status",
            notebook_version=NOTEBOOK_VERSION,
            selected_model_name="Default pillar",
            model_profiles={"Default pillar": "model"},
        )
        settings.save = lambda: None
        client = DriveClient()
        client.account_email = "researcher@example.org"
        client.service = FakeService()

        self.assertEqual(client.ensure_workspace(settings), "workspace")
        self.assertEqual(len(client.service.files_resource.get_calls), 1)
        self.assertEqual(
            client.service.files_resource.get_calls[0]["fileId"],
            "workspace",
        )


if __name__ == "__main__":
    unittest.main()
