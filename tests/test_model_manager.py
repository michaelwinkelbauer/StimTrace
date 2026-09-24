import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QGroupBox, QWidget

from desktop_app import (
    ModelManagerPage,
    Settings,
    SettingsDialog,
    WorkflowStep,
    saved_google_session_was_revoked,
)


class ModelManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_model_manager_page_can_be_constructed(self):
        """The Models navigation page must not fail while wiring its settings action."""
        class MainWindowStub(QWidget):
            def __init__(self):
                super().__init__()
                self.settings = Settings()
                self.worker_hardware = {}

            def open_settings(self):
                pass

        main_window = MainWindowStub()
        page = ModelManagerPage(main_window)

        self.assertIs(page.main_window, main_window)
        self.assertGreaterEqual(page.table.rowCount(), 1)

    def test_only_proven_revoked_google_sessions_are_discarded(self):
        self.assertTrue(saved_google_session_was_revoked("invalid_grant: Token expired"))
        self.assertFalse(saved_google_session_was_revoked("getaddrinfo failed"))

    def test_workflow_step_responds_to_mouse_and_keyboard(self):
        step = WorkflowStep(1, "Google account", "Sign in")
        activations = []
        step.clicked.connect(lambda: activations.append(True))
        step.set_clickable(True)
        step.show()
        QTest.mouseClick(step, Qt.LeftButton)
        step.setFocus()
        QTest.keyClick(step, Qt.Key_Return)
        self.assertEqual(len(activations), 2)

    def test_normal_analysis_settings_include_tracking_and_local_overlays(self):
        settings = Settings()
        self.assertEqual(settings.cloud_overlay_mode, "local")
        dialog = SettingsDialog(settings)
        boxes = {box.title(): box for box in dialog.findChildren(QGroupBox)}
        analysis = boxes["Analysis settings"]
        expert = boxes["Expert settings"]
        self.assertEqual(dialog.cloud_overlays.currentData(), "local")
        self.assertTrue(analysis.isAncestorOf(dialog.cloud_overlays))
        self.assertTrue(analysis.isAncestorOf(dialog.tracking_filter))
        self.assertFalse(expert.isAncestorOf(dialog.cloud_overlays))
        self.assertFalse(expert.isAncestorOf(dialog.tracking_filter))
        dialog.close()

    def test_innovation_gate_controls_are_not_user_settings(self):
        dialog = SettingsDialog(Settings())

        self.assertNotIn("kalman_innovation_gate_min_radius_px", dialog.fields)
        self.assertFalse(hasattr(dialog, "innovation_gate"))
        dialog.close()

    def test_pixel_calibration_receives_random_segment_videos(self):
        videos = [Path(f"recording_{index}.avi") for index in range(6)]
        selected = [videos[4], videos[1], videos[5]]
        settings_dialog = SettingsDialog(Settings(), calibration_videos=videos)
        with patch("desktop_app.random.sample", return_value=selected), patch(
            "pixel_calibration.PixelCalibrationDialog"
        ) as calibration_dialog:
            calibration_dialog.return_value.exec.return_value = QDialog.Rejected
            settings_dialog.open_pixel_calibration()
        self.assertEqual(
            calibration_dialog.call_args.kwargs["initial_videos"],
            selected,
        )
        settings_dialog.close()
