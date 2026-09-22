from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from background_tasks import BackgroundFunctionThread, LocalProcessThread, is_external_service_error
from desktop_app import MainWindow, local_overlay_warning_message


class BackgroundTaskTests(unittest.TestCase):
    def test_workflow_cards_delegate_to_video_picker_and_signal_analyzer(self):
        window = type("WindowStub", (), {})()
        window.add_videos = MagicMock()
        window.open_signal_analysis = MagicMock()

        MainWindow.activate_input_videos_step(window)
        MainWindow.activate_analysis_step(window)

        window.add_videos.assert_called_once_with()
        window.open_signal_analysis.assert_called_once_with()

    def test_missing_local_overlay_sources_produce_actionable_warning(self):
        title, message = local_overlay_warning_message(
            [
                "A-1: original video is unavailable",
                "B-2: original video is unavailable",
            ],
            Path("results") / "job-1",
        )

        self.assertEqual(title, "Original recordings unavailable")
        self.assertIn("A-1", message)
        self.assertIn("B-2", message)
        self.assertIn("traces are safe", message)
        self.assertIn("Jobs → Download results", message)

    def test_new_submission_clears_previous_progress_display(self):
        window = type("WindowStub", (), {})()
        for name in (
            "monitoring_label",
            "file_progress_label",
            "file_progress",
            "job_progress_label",
            "job_progress_bar",
            "upload_progress_label",
            "upload_progress",
        ):
            setattr(window, name, MagicMock())
        window.stop_upload_attention = lambda: MainWindow.stop_upload_attention(window)

        MainWindow.prepare_new_submission_progress(window)

        window.monitoring_label.setText.assert_called_once_with("Preparing new submission...")
        window.monitoring_label.setVisible.assert_called_once_with(True)
        for progress in (window.file_progress, window.job_progress_bar, window.upload_progress):
            progress.setValue.assert_called_once_with(0)
            progress.setFormat.assert_called_once_with("0%")
            progress.setVisible.assert_called_once_with(False)
        window.file_progress_label.setVisible.assert_called_once_with(False)
        window.job_progress_label.setVisible.assert_called_once_with(False)
        window.upload_progress_label.setVisible.assert_called_with(False)

    def test_shutdown_returns_without_waiting_for_running_thread(self):
        class RunningThread:
            interrupted = False

            def isRunning(self):
                return True

            def objectName(self):
                return "slow-network-operation"

            def requestInterruption(self):
                self.interrupted = True

        class Status:
            text = ""

            def setText(self, value):
                self.text = value

        window = type("WindowStub", (), {})()
        window.shutdown_started = False
        window.shutdown_started_at = 0.0
        window.active_threads = {RunningThread()}
        window.retired_threads = []
        window.training_page = None
        window.point_tracking_page = None
        window.hardware_probe = None
        window.local_process = None
        window.status = Status()
        window.settings = type("Settings", (), {"remember_google_sign_in": True})()
        window.drive = type("Drive", (), {"sign_out": lambda self: None})()
        self.assertFalse(MainWindow.shutdown_threads(window))
        self.assertIn("slow-network-operation", window.status.text)

    def test_shutdown_terminates_stuck_background_thread_after_deadline(self):
        class StuckThread:
            running = True

            def isRunning(self):
                return self.running

            def objectName(self):
                return "stuck-drive-request"

            def terminate(self):
                self.running = False

            def wait(self, _milliseconds):
                return True

        thread = StuckThread()
        window = type("WindowStub", (), {})()
        window.shutdown_started = True
        window.shutdown_started_at = 0.0
        window.active_threads = {thread}
        window.retired_threads = []
        window.training_page = None
        window.point_tracking_page = None
        window.hardware_probe = None
        window.local_process = None
        window.settings = type("Settings", (), {"remember_google_sign_in": True})()
        window.drive = type("Drive", (), {"sign_out": lambda self: None})()
        self.assertTrue(MainWindow.shutdown_threads(window))
        self.assertFalse(thread.running)

    def test_function_worker_reports_success(self):
        results = []
        worker = BackgroundFunctionThread(lambda: 42)
        worker.succeeded.connect(results.append)
        worker.run()
        self.assertEqual(results, [42])

    def test_function_worker_reports_failure(self):
        errors = []

        def fail():
            raise ValueError("expected failure")

        worker = BackgroundFunctionThread(fail)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertEqual(errors, ["expected failure"])

    def test_external_service_errors_are_classified_without_hiding_local_errors(self):
        self.assertTrue(is_external_service_error(ConnectionError("offline")))
        self.assertFalse(is_external_service_error(ValueError("invalid input")))

    def test_function_worker_can_report_progress_without_touching_the_ui(self):
        progress = []
        worker = BackgroundFunctionThread(lambda: "done")
        worker.progress.connect(progress.append)
        worker.progress.emit((1, 0.5, "recording.avi"))
        worker.run()
        self.assertEqual(progress, [(1, 0.5, "recording.avi")])

    def test_local_process_separates_events_from_output_tail(self):
        events = []
        command = (
            "import json; "
            "print('STIMTRACE_EVENT '+json.dumps({'event':'progress','value':3})); "
            "print('worker diagnostic')"
        )
        worker = LocalProcessThread(sys.executable, ["-c", command])
        worker.event_received.connect(events.append)
        worker.run()
        self.assertEqual(worker.exit_code, 0)
        self.assertEqual(events, [{"event": "progress", "value": 3}])
        self.assertEqual(worker.output_tail, "worker diagnostic")


if __name__ == "__main__":
    unittest.main()
