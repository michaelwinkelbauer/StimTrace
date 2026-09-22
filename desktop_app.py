"""Desktop client for the Drive-backed StimTrace segmentation worker."""
from __future__ import annotations

import importlib
import faulthandler
import logging
import json
import io
import os
import random
import sys
import uuid
import webbrowser
import zipfile
import shutil
import time
import threading
import httplib2
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from google.auth.transport.requests import Request
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaInMemoryUpload
from PySide6.QtCore import QLocale, QSize, Qt, QThread, QTimer, QUrl, Slot, qInstallMessageHandler
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QFrame, QHBoxLayout, QLabel, QLayout, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QDoubleSpinBox, QHeaderView, QInputDialog, QProgressBar, QSizePolicy, QSpinBox, QStackedWidget,
    QStyle, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget, QScrollArea,
)

from background_tasks import BackgroundFunctionThread, LocalProcessThread
from app_logging import configure_app_logging, get_logger
from ui_actions import configure_row_delete, set_action_icon
from ui_components import WorkflowStep
from app_metadata import APPLICATION_NAME, APPLICATION_VERSION
from job_utils import (
    ACTIVE_JOB_STATES,
    RUNNING_JOB_STATES,
    TERMINAL_JOB_STATES,
    analysis_trace_file as find_analysis_trace_file,
    elapsed_seconds,
    duplicate_output_stems,
    format_elapsed as format_job_elapsed,
    job_history_identity,
    local_now_iso,
    result_download_target as find_result_download_target,
    safe_path_component as normalize_path_component,
    video_files_in_folder,
)
from job_orchestration import (
    CloudSubmissionRequest,
    fetch_cloud_job_progress,
    make_submitted_segmentation_job,
    upload_cloud_job,
)

SOURCE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = (
    Path(getattr(sys, "_MEIPASS")).resolve()
    if getattr(sys, "frozen", False)
    else SOURCE_DIR
)
APP_ICON_PATH = RESOURCE_DIR / "assets" / "stimtrace.ico"
APP_LOGO_PATH = RESOURCE_DIR / "assets" / "stimtrace-logo.png"
USER_DATA_DIR = (
    Path(os.environ.get("LOCALAPPDATA", Path.home())) / "StimTrace"
    if getattr(sys, "frozen", False)
    else SOURCE_DIR
)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
JOB_HISTORY_PATH = USER_DATA_DIR / "jobs.json"
CRASH_LOG_PATH = USER_DATA_DIR / "stimtrace_crash.log"
APP_LOG_PATH = USER_DATA_DIR / "stimtrace.log"
LOGGER = get_logger("desktop")
_FAULT_LOG_HANDLE = None
NOTEBOOK_VERSION = 63
WORKER_HEARTBEAT_GRACE_SECONDS = 120


def choose_calibration_videos(paths: list[Path]) -> list[Path] | None:
    """Return three random unique analysis videos, or defer to file selection."""
    unique_paths = list(dict.fromkeys(Path(path) for path in paths))
    return random.sample(unique_paths, 3) if len(unique_paths) >= 3 else None


def saved_google_session_was_revoked(message: str) -> bool:
    """Return whether an OAuth refresh failure proves that the local token is unusable."""
    normalized = str(message).lower()
    return "invalid_grant" in normalized or "expired or revoked" in normalized


def local_overlay_warning_message(warnings: list[str], target: Path) -> tuple[str, str]:
    """Build an actionable warning for incomplete local overlay rendering."""
    missing_suffix = ": original video is unavailable"
    missing = [warning.removesuffix(missing_suffix) for warning in warnings if warning.endswith(missing_suffix)]
    other = [warning for warning in warnings if not warning.endswith(missing_suffix)]
    if missing:
        shown = "\n".join(f"• {name}" for name in missing[:12])
        if len(missing) > 12:
            shown += f"\n• …and {len(missing) - 12} more"
        body = (
            "The result traces were downloaded successfully, but local overlay videos could not "
            "be created because these original recordings were not found:\n\n"
            f"{shown}\n\n"
            f"Your downloaded traces are safe in:\n{target}\n\n"
            "Restore the recordings, or use Jobs → Download results and select the folder "
            "containing the original videos to render the overlays again."
        )
        if other:
            body += "\n\nOther overlay warnings:\n" + "\n".join(f"• {item}" for item in other[:8])
        return "Original recordings unavailable", body
    return (
        "Some overlays were not created",
        "The result traces were downloaded successfully, but some local overlays were skipped:\n\n"
        + "\n".join(f"• {item}" for item in warnings[:12])
        + f"\n\nDownloaded results are in:\n{target}",
    )


def configure_numeric_locale() -> None:
    """Use a period as the decimal separator throughout the application UI."""
    QLocale.setDefault(QLocale.c())


def install_crash_logging() -> None:
    """Persist Python and Qt diagnostics, including native fault stack dumps."""
    global _FAULT_LOG_HANDLE
    configure_app_logging(APP_LOG_PATH)
    LOGGER.info(
        "Starting %s %s with Python %s",
        APPLICATION_NAME,
        APPLICATION_VERSION,
        sys.version.split()[0],
    )
    _FAULT_LOG_HANDLE = CRASH_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    _FAULT_LOG_HANDLE.write(
        f"\n=== StimTrace session {datetime.now(timezone.utc).isoformat()} "
        f"| Python {sys.version.split()[0]} ===\n"
    )
    faulthandler.enable(_FAULT_LOG_HANDLE, all_threads=True)

    def report_unhandled(error_type, error, traceback) -> None:
        _FAULT_LOG_HANDLE.write("Unhandled Python exception:\n")
        import traceback as traceback_module

        traceback_module.print_exception(
            error_type,
            error,
            traceback,
            file=_FAULT_LOG_HANDLE,
        )
        _FAULT_LOG_HANDLE.flush()
        sys.__excepthook__(error_type, error, traceback)

    def report_qt_message(_message_type, context, message) -> None:
        location = ""
        if context is not None and getattr(context, "file", None):
            location = f" ({context.file}:{context.line})"
        line = f"Qt: {message}{location}\n"
        _FAULT_LOG_HANDLE.write(line)
        _FAULT_LOG_HANDLE.flush()
        try:
            sys.__stderr__.write(line)
            sys.__stderr__.flush()
        except Exception:
            pass

    sys.excepthook = report_unhandled
    qInstallMessageHandler(report_qt_message)
# Limit access to files and job folders created by this application.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]
MODEL_PARAMETER_KEYS = (
    "pixel_to_um",
    "force_slope_un_per_um",
    "bending_axis_mode",
    "bending_axis_angle_deg",
    "legacy_area_normalization",
    "mask_threshold",
    "refine_iterations",
    "tracking_filter_mode",
    "kalman_q_pos",
    "kalman_q_vel",
    "kalman_r",
    "kalman_innovation_gate_enabled",
    "kalman_innovation_gate_confidence",
    "kalman_innovation_gate_min_radius_px",
    "inference_batch_size",
    "cpu_postprocess_workers",
    "progress_interval_seconds",
)
FACTORY_MODEL_PARAMETERS: dict[str, Any] = {
    "pixel_to_um": 4.35,
    "force_slope_un_per_um": 6.14,
    "bending_axis_mode": "automatic",
    "bending_axis_angle_deg": 0.0,
    "legacy_area_normalization": False,
    "mask_threshold": 0.5,
    "refine_iterations": 8,
    "tracking_filter_mode": "kalman",
    "kalman_q_pos": 2.0,
    "kalman_q_vel": 36.0,
    "kalman_r": 8.0,
    "kalman_innovation_gate_enabled": True,
    "kalman_innovation_gate_confidence": 0.99,
    "kalman_innovation_gate_min_radius_px": 120.0,
    "inference_batch_size": 8,
    "cpu_postprocess_workers": 1,
    "progress_interval_seconds": 5.0,
}
DEFAULT_KALMAN_BENCHMARK = [
    {"name": "Default", "kalman_q_pos": 2.0, "kalman_q_vel": 36.0, "kalman_r": 8.0},
    {"name": "Smooth", "kalman_q_pos": 1.0, "kalman_q_vel": 18.0, "kalman_r": 16.0},
    {"name": "Responsive", "kalman_q_pos": 4.0, "kalman_q_vel": 72.0, "kalman_r": 4.0},
    {"name": "Measurement trusting", "kalman_q_pos": 2.0, "kalman_q_vel": 36.0, "kalman_r": 2.0},
    {"name": "Measurement skeptical", "kalman_q_pos": 2.0, "kalman_q_vel": 36.0, "kalman_r": 32.0},
]
SETTING_DESCRIPTIONS = {
    "pixel_to_um": (
        "Physical size represented by one image pixel. This converts tracked pillar "
        "displacement from pixels to micrometers."
    ),
    "force_slope_un_per_um": (
        "Slope measured by the uniaxial pillar calibration. Active contraction force is "
        "slope x displacement relative to the recording's diastolic position."
    ),
    "bending_axis_mode": (
        "Automatic estimates the uniaxial motion direction from the tracked centers. "
        "Fixed angle projects motion onto a direction you define in image coordinates."
    ),
    "bending_axis_angle_deg": (
        "Direction of pillar bending in the image: 0 degrees points right and 90 degrees "
        "points down. The trace is projected onto this axis instead of using 2D distance."
    ),
    "legacy_area_normalization": (
        "Legacy notebook correction: corrected displacement = projected displacement x "
        "(maximum segmented area / current segmented area). A smaller or unstable mask can "
        "therefore amplify displacement and noise. This is not part of the mechanical force "
        "calibration; leave it disabled unless independently validated for your imaging system."
    ),
    "mask_threshold": (
        "Minimum model probability classified as foreground. Higher values create "
        "smaller, stricter masks; lower values include more uncertain pixels."
    ),
    "kalman_q_pos": (
        "Position process-noise variance in px^2, added at each prediction step. Higher "
        "values react faster to movement but can make the trace less smooth."
    ),
    "tracking_filter_mode": (
        "Kalman smoothing combines each segmented center with a constant-velocity motion "
        "prediction. None uses the raw center measured from every valid segmentation; "
        "frames without a valid mask remain missing."
    ),
    "kalman_q_vel": (
        "Velocity process-noise variance in px^2/s^2, added at each prediction step. Higher "
        "values follow rapid contractions more readily but reduce temporal smoothing."
    ),
    "kalman_r": (
        "Position measurement-noise variance in px^2. Higher values trust each segmented "
        "position less and produce a smoother track."
    ),
    "kalman_innovation_gate_enabled": (
        "Before a segmented center updates the Kalman filter, compare its innovation with "
        "the predicted innovation covariance. Rejected measurements remain visible in raw "
        "QC columns but are excluded from displacement and force."
    ),
    "kalman_innovation_gate_confidence": (
        "Statistical acceptance region for two-dimensional center measurements. A higher "
        "percentage accepts larger deviations; 99% is the default. This depends on "
        "appropriately tuned Kalman process and measurement variances."
    ),
    "kalman_innovation_gate_min_radius_px": (
        "Always accept an innovation within this pixel radius. The 120 px default preserves "
        "the original notebook's validated operating boundary while the covariance model "
        "remains empirically tuned. Set this to zero only after calibrating Q and R for a "
        "pure covariance gate."
    ),
    "progress_interval_seconds": (
        "How often Colab writes progress to Google Drive for the desktop application. "
        "Shorter intervals increase Drive traffic but do not speed up segmentation."
    ),
    "refine_iterations": (
        "Number of ellipse-refinement passes applied to each segmented pillar. More passes "
        "can improve boundary fitting but increase CPU processing time; zero disables refinement."
    ),
    "inference_batch_size": (
        "Maximum number of frames evaluated together on the GPU. StimTrace starts from a "
        "memory-appropriate batch, probes larger batches up to this value, and retains the "
        "largest size that fits. GPU memory, not system RAM, determines the result."
    ),
    "cpu_postprocess_workers": (
        "Parallel CPU workers used for mask cleanup and ellipse fitting. The cloud worker "
        "caps this at the runtime's logical CPU count minus one."
    ),
}


def colab_notebook_url(notebook_file_id: str) -> str:
    return f"https://colab.research.google.com/drive/{notebook_file_id}"


APP_STYLESHEET = """
* {
    font-family: "Segoe UI";
    font-size: 10pt;
    color: #e9eef2;
}
QMainWindow, QDialog, QWidget {
    background-color: #15191d;
}
QMenuBar {
    background: #15191d;
    border-bottom: 1px solid #303840;
    padding: 3px 6px;
}
QMenuBar::item {
    padding: 6px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected, QMenu::item:selected {
    background: #29323a;
}
QMenu {
    background: #20262c;
    border: 1px solid #3a444d;
    padding: 5px;
}
QToolBar#primaryNavigation {
    background: #1b2025;
    border: none;
    border-bottom: 1px solid #303840;
    spacing: 4px;
    padding: 5px 8px;
}
QToolBar#primaryNavigation QToolButton {
    min-height: 32px;
    padding: 0 12px;
    border: none;
    border-radius: 5px;
    color: #aeb9c2;
}
QToolBar#primaryNavigation QToolButton:hover {
    background: #29323a;
    color: #ffffff;
}
QToolBar#primaryNavigation QToolButton:checked {
    background: #3a2229;
    color: #f0a1b1;
}
QLabel[role="title"] {
    font-size: 22pt;
    font-weight: 600;
    color: #f5f8fa;
}
QLabel[role="subtitle"], QLabel[role="muted"] {
    color: #9daab4;
}
QLabel[authState="signedOut"] {
    color: #ff7482;
    font-weight: 600;
}
QLabel[role="instruction"] {
    color: #ead5da;
    background: #2b1d22;
    border-left: 3px solid #b54860;
    padding: 9px 11px;
}
QGroupBox {
    font-weight: 600;
    border: 1px solid #333c44;
    border-radius: 6px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    background: #191e23;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #cbd4da;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 32px;
    background: #20262c;
    border: 1px solid #3a444d;
    border-radius: 5px;
    padding: 0 8px;
    selection-background-color: #713244;
}
QSpinBox, QDoubleSpinBox {
    padding-right: 32px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 26px;
    border-left: 1px solid #3a444d;
    border-bottom: 1px solid #303840;
    border-top-right-radius: 4px;
    background: #252c32;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 26px;
    border-left: 1px solid #3a444d;
    border-bottom-right-radius: 4px;
    background: #252c32;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #364149;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #1a201f;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("__SPIN_UP_ICON__");
    width: 12px;
    height: 12px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("__SPIN_DOWN_ICON__");
    width: 12px;
    height: 12px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #c6536b;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QPushButton {
    min-height: 32px;
    background: #272e35;
    border: 1px solid #3b454e;
    border-radius: 5px;
    padding: 0 12px;
}
QPushButton:hover {
    background: #313a42;
    border-color: #53616c;
}
QPushButton:pressed {
    background: #20262c;
}
QPushButton:disabled {
    color: #69747d;
    background: #1c2126;
    border-color: #2b3238;
}
QPushButton[role="primary"] {
    color: #fff7f9;
    background: #b9435b;
    border-color: #b9435b;
    font-weight: 600;
}
QPushButton[role="primary"]:hover {
    background: #d05a71;
    border-color: #d05a71;
}
QPushButton[role="primary"]:pressed {
    background: #963449;
    border-color: #963449;
}
QPushButton[role="mode"] {
    min-width: 110px;
    color: #aeb9c2;
    background: #1b2025;
}
QPushButton[role="mode"]:checked {
    color: #fff7f9;
    background: #713244;
    border-color: #c6536b;
    font-weight: 600;
}
QPushButton[role="danger"] {
    color: #ffb4b4;
    border-color: #704348;
}
QPushButton[role="dangerIcon"] {
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    padding: 0;
    color: #ffffff;
    background: #8f3038;
    border: 1px solid #d45b64;
    font-size: 11pt;
    font-weight: 700;
}
QPushButton[role="dangerIcon"]:hover {
    color: #ffffff;
    background: #c4434d;
    border-color: #ff7b83;
}
QPushButton[role="dangerIcon"]:disabled {
    color: #73545a;
    background: #252126;
    border-color: #49383c;
}
QPushButton[role="traceControl"] {
    min-width: 0;
    min-height: 0;
    padding: 0;
    margin: 0;
}
QCheckBox#autoMaskNextFrame {
    color: #ead5da;
    background: #251a1e;
    border: 1px solid #713244;
    border-radius: 6px;
    padding: 8px 12px;
    spacing: 8px;
    font-weight: 600;
}
QCheckBox#autoMaskNextFrame:hover {
    color: #fff7f9;
    background: #321f25;
    border-color: #c6536b;
}
QCheckBox#autoMaskNextFrame:checked {
    color: #fff7f9;
    background: #3a2229;
    border-color: #d05a71;
}
QCheckBox#autoMaskNextFrame:focus {
    border-color: #f08ba0;
}
QCheckBox#autoMaskNextFrame::indicator {
    width: 18px;
    height: 18px;
}
QListWidget, QTableWidget {
    background: #171c20;
    alternate-background-color: #1c2227;
    border: 1px solid #333c44;
    border-radius: 5px;
    padding: 4px;
    outline: none;
}
QListWidget::item, QTableWidget::item {
    padding: 6px;
}
QListWidget::item:selected, QTableWidget::item:selected {
    background: #653040;
    color: #ffffff;
}
QListWidget#signalTraceList::item {
    padding: 0;
    margin: 1px 0;
}
QListWidget#signalTraceList::item:selected {
    background: transparent;
}
QHeaderView::section {
    background: #252c32;
    color: #cdd5da;
    border: none;
    border-right: 1px solid #364049;
    padding: 7px;
}
QProgressBar {
    min-height: 18px;
    max-height: 18px;
    border: none;
    border-radius: 4px;
    background: #2a3137;
    text-align: center;
}
QProgressBar::chunk {
    border-radius: 4px;
    background: #b9435b;
}
QLabel[role="uploadNotice"] {
    min-height: 24px;
    color: #ff667d;
    background: #351b22;
    border: 1px solid #b9435b;
    border-left: 4px solid #ff667d;
    border-radius: 4px;
    padding: 7px 10px;
    font-size: 11pt;
    font-weight: 700;
}
QLabel[role="uploadNotice"][blinkPhase="dim"] {
    color: #c6536b;
    background: #251a1e;
    border-color: #713244;
}
QProgressBar[uploadState="active"] {
    min-height: 26px;
    max-height: 26px;
    color: #ffffff;
    border: 1px solid #8f354a;
    font-weight: 700;
}
QProgressBar[uploadState="active"]::chunk {
    background: #d34763;
}
QStatusBar {
    color: #9daab4;
    border-top: 1px solid #303840;
}
QToolTip {
    color: #f2f5f7;
    background: #252c32;
    border: 1px solid #46525c;
    padding: 5px;
}
QFrame#workflowStep {
    background: #1b2126;
    border: 1px solid #303941;
    border-radius: 6px;
}
QFrame#workflowStep[state="active"] {
    border-color: #c6536b;
    background: #2b1d22;
}
QFrame#workflowStep[state="complete"] {
    border-color: #39765e;
}
QFrame#workflowStep[clickable="true"]:hover {
    border-color: #6f8493;
    background: #242c32;
}
QFrame#workflowStep[clickable="true"]:focus {
    border-color: #e06a82;
}
QFrame#stickyFooter {
    background: #1b2025;
    border-top: 1px solid #303840;
}
QLabel[role="stepNumber"] {
    min-width: 26px;
    max-width: 26px;
    min-height: 26px;
    max-height: 26px;
    border-radius: 13px;
    background: #323b43;
    color: #cbd4da;
    font-weight: 600;
    qproperty-alignment: AlignCenter;
}
QFrame#workflowStep[state="active"] QLabel[role="stepNumber"] {
    background: #c6536b;
    color: #fff7f9;
}
QFrame#workflowStep[state="complete"] QLabel[role="stepNumber"] {
    background: #4aa878;
    color: #07150e;
}
"""
APP_STYLESHEET = (
    APP_STYLESHEET
    .replace(
        "__SPIN_UP_ICON__",
        (RESOURCE_DIR / "assets" / "chevron-up.xpm").as_posix(),
    )
    .replace(
        "__SPIN_DOWN_ICON__",
        (RESOURCE_DIR / "assets" / "chevron-down.xpm").as_posix(),
    )
)


@dataclass
class Settings:
    drive_root_folder_id: str = ""
    workspace_folder_name: str = ""
    google_account_email: str = ""
    oauth_client_config_path: str = ""
    model_file_id: str = ""
    default_model_file_id: str = ""
    selected_model_name: str = "Default pillar"
    model_profiles: dict[str, str] = field(default_factory=dict)
    model_parameter_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_default_parameter_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    worker_file_id: str = ""
    notebook_file_id: str = ""
    marker_file_id: str = ""
    worker_status_file_id: str = ""
    notebook_version: int = 0
    pixel_to_um: float = 4.35
    force_slope_un_per_um: float = 6.14
    bending_axis_mode: str = "automatic"
    bending_axis_angle_deg: float = 0.0
    legacy_area_normalization: bool = False
    mask_threshold: float = 0.5
    refine_iterations: int = 8
    tracking_filter_mode: str = "kalman"
    kalman_q_pos: float = 2.0
    kalman_q_vel: float = 36.0
    kalman_r: float = 8.0
    kalman_innovation_gate_enabled: bool = True
    kalman_innovation_gate_confidence: float = 0.99
    kalman_innovation_gate_min_radius_px: float = 120.0
    inference_batch_size: int = 8
    cpu_postprocess_workers: int = 1
    progress_interval_seconds: float = 5.0
    remember_google_sign_in: bool = True
    compute_mode: str = "cloud"
    local_device_preference: str = "auto"
    local_cpu_threads: int = 0
    local_inference_batch_size: int = 0
    local_postprocess_workers: int = 0
    local_generate_overlays: bool = True
    cloud_overlay_mode: str = "local"
    local_model_paths: dict[str, str] = field(default_factory=dict)
    point_tracking_settings: dict[str, Any] = field(default_factory=dict)
    hidden_job_ids: list[str] = field(default_factory=list)

    def parameter_values(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in MODEL_PARAMETER_KEYS}

    def store_model_parameters(self, model_name: str | None = None) -> None:
        self.model_parameter_profiles[model_name or self.selected_model_name] = self.parameter_values()

    def ensure_model_parameters(self, model_name: str) -> None:
        initial_values = (
            dict(FACTORY_MODEL_PARAMETERS)
            if model_name == "Default pillar"
            else self.parameter_values()
        )
        current_values = dict(initial_values)
        current_values.update(self.model_parameter_profiles.get(model_name, {}))
        self.model_parameter_profiles[model_name] = current_values
        if model_name == "Default pillar":
            self.model_default_parameter_profiles[model_name] = dict(FACTORY_MODEL_PARAMETERS)
        else:
            default_values = dict(current_values)
            default_values.update(self.model_default_parameter_profiles.get(model_name, {}))
            self.model_default_parameter_profiles[model_name] = default_values

    def model_default_parameters(self, model_name: str) -> dict[str, Any]:
        self.ensure_model_parameters(model_name)
        return dict(self.model_default_parameter_profiles[model_name])

    def apply_model_parameters(self, model_name: str) -> None:
        self.ensure_model_parameters(model_name)
        for key, value in self.model_parameter_profiles[model_name].items():
            if key in MODEL_PARAMETER_KEYS:
                setattr(self, key, value)

    @classmethod
    def load(cls) -> "Settings":
        path = USER_DATA_DIR / "config.json"
        if not path.exists():
            settings = cls()
            settings.store_model_parameters()
            settings.ensure_model_parameters("Default pillar")
            settings.model_metadata["Default pillar"] = {
                "object_type": "Pillar",
                "source": "Bundled",
                "created_at": "",
                "dataset_size": "",
            }
            return settings
        values = json.loads(path.read_text(encoding="utf-8"))
        if values.get("notebook_version", 0) < 14:
            # Version 13 defaulted to four workers, which oversubscribed the
            # small CPU allocation on common Colab runtimes.
            values["cpu_postprocess_workers"] = 1
        allowed = set(cls.__dataclass_fields__)
        settings = cls(**{key: value for key, value in values.items() if key in allowed})
        settings.model_parameter_profiles = {
            name: {
                key: value
                for key, value in profile.items()
                if key in MODEL_PARAMETER_KEYS
            }
            for name, profile in settings.model_parameter_profiles.items()
        }
        settings.model_default_parameter_profiles = {
            name: {
                key: value
                for key, value in profile.items()
                if key in MODEL_PARAMETER_KEYS
            }
            for name, profile in settings.model_default_parameter_profiles.items()
        }
        for model_name in settings.model_profiles or [settings.selected_model_name]:
            settings.ensure_model_parameters(model_name)
        settings.ensure_model_parameters(settings.selected_model_name)
        settings.apply_model_parameters(settings.selected_model_name)
        settings.model_metadata.setdefault(
            "Default pillar",
            {
                "object_type": "Pillar",
                "source": "Bundled",
                "created_at": "",
                "dataset_size": "",
            },
        )
        return settings

    def save(self) -> None:
        (USER_DATA_DIR / "config.json").write_text(
            json.dumps(asdict(self), indent=2),
            encoding="utf-8",
        )


class DriveClient:
    def __init__(self) -> None:
        self.service = None
        self.account_email = ""
        # googleapiclient/httplib2 reuses one native HTTP connection and is not safe
        # for simultaneous calls from the Qt background workers.
        self._request_lock = threading.RLock()

    @staticmethod
    def _build_service(credentials: Credentials):
        """Build a Drive client whose individual HTTP calls cannot hang indefinitely."""
        http = AuthorizedHttp(credentials, http=httplib2.Http(timeout=30))
        return build("drive", "v3", http=http, cache_discovery=False)

    def _refresh_service_connection(self) -> None:
        """Do not reuse an httplib2 TLS socket across independent background tasks."""
        http = getattr(self.service, "_http", None)
        credentials = getattr(http, "credentials", None)
        if credentials is not None:
            self.service = self._build_service(credentials)

    @staticmethod
    def _execute_with_backoff(request, attempts: int = 6):
        for attempt in range(attempts):
            try:
                return request.execute(num_retries=2)
            except HttpError as error:
                content = getattr(error, "content", b"")
                content_text = (
                    content.decode("utf-8", errors="replace")
                    if isinstance(content, bytes)
                    else str(content)
                )
                text = f"{error} {content_text}"
                retryable = (
                    error.resp.status in {429, 500, 502, 503, 504}
                    or (
                        error.resp.status == 403
                        and (
                            "userRateLimitExceeded" in text
                            or "rateLimitExceeded" in text
                        )
                    )
                )
                if not retryable or attempt == attempts - 1:
                    raise
                LOGGER.warning(
                    "Google Drive request failed with retryable status %s; retry %s/%s",
                    error.resp.status,
                    attempt + 1,
                    attempts,
                )
                time.sleep(min(16, 2 ** attempt))

    def sign_in(
        self,
        *,
        interactive: bool = True,
        oauth_client_config_path: Path | None = None,
    ) -> str:
        token_path = USER_DATA_DIR / "token.json"
        credentials: Credentials | None = None
        if token_path.exists():
            credentials = Credentials.from_authorized_user_file(token_path, SCOPES)
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        if not credentials or not credentials.valid:
            if not interactive:
                raise RuntimeError(
                    "The saved Google session cannot be restored without authorization."
                )
            if oauth_client_config_path is None or not oauth_client_config_path.is_file():
                raise FileNotFoundError(
                    "A Google OAuth Desktop client JSON file is required to sign in. "
                    "Select your own configuration file and try again."
                )
            credentials = InstalledAppFlow.from_client_secrets_file(
                oauth_client_config_path,
                SCOPES,
            ).run_local_server(port=0)
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        self.service = self._build_service(credentials)
        user = self.service.about().get(fields="user(emailAddress)").execute().get("user", {})
        self.account_email = user.get("emailAddress", "")
        return self.account_email

    def sign_out(self) -> None:
        """Revoke the app token and remove the local session on window close."""
        token_path = USER_DATA_DIR / "token.json"
        try:
            if token_path.exists():
                credentials = Credentials.from_authorized_user_file(token_path, SCOPES)
                token = credentials.refresh_token or credentials.token
                if token:
                    request = UrlRequest(
                        "https://oauth2.googleapis.com/revoke",
                        data=urlencode({"token": token}).encode("ascii"),
                        method="POST",
                    )
                    urlopen(request, timeout=5).close()
        except Exception:
            # Closing the application must not be blocked by a network failure.
            LOGGER.debug("Google token revocation failed during sign-out", exc_info=True)
        finally:
            token_path.unlink(missing_ok=True)
            self.service = None
            self.account_email = ""

    @staticmethod
    def saved_session_json(file_id: str) -> dict:
        """Read one Drive JSON file with an isolated HTTP client for UI background work."""
        from googleapiclient.http import MediaIoBaseDownload

        token_path = USER_DATA_DIR / "token.json"
        if not token_path.exists():
            raise RuntimeError("The saved Google session is no longer available.")
        credentials = Credentials.from_authorized_user_file(token_path, SCOPES)
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            token_path.write_text(credentials.to_json(), encoding="utf-8")
        service = DriveClient._build_service(credentials)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(
            buffer,
            service.files().get_media(fileId=file_id),
        )
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=3)
        return json.loads(buffer.getvalue().decode("utf-8"))

    def _create_folder(self, name: str, parent: str | None = None) -> str:
        body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if parent:
            body["parents"] = [parent]
        result = self._execute_with_backoff(self.service.files().create(
            body=body,
            fields="id",
        ))
        return result["id"]

    def _replace_with_local_file(
        self,
        file_id: str,
        source: Path,
        mime_type: str | None,
    ) -> None:
        self._execute_with_backoff(self.service.files().update(
            fileId=file_id,
            media_body=MediaFileUpload(
                str(source),
                mimetype=mime_type,
                resumable=True,
            ),
            fields="id",
        ))

    @staticmethod
    def _clear_workspace_settings(settings: Settings) -> None:
        settings.drive_root_folder_id = ""
        settings.workspace_folder_name = ""
        settings.model_file_id = ""
        settings.worker_file_id = ""
        settings.notebook_file_id = ""
        settings.marker_file_id = ""
        settings.worker_status_file_id = ""

    def ensure_workspace(self, settings: Settings) -> str:
        with self._request_lock:
            self._refresh_service_connection()
            return self._ensure_workspace_unlocked(settings)

    def _ensure_workspace_unlocked(self, settings: Settings) -> str:
        if not self.service:
            raise RuntimeError("Sign in before creating a workspace.")
        trusted_existing_assets = (
            settings.google_account_email == self.account_email
            and settings.notebook_version == NOTEBOOK_VERSION
            and bool(settings.drive_root_folder_id)
            and bool(settings.model_file_id)
            and bool(settings.worker_file_id)
            and bool(settings.notebook_file_id)
            and bool(settings.worker_status_file_id)
        )
        if settings.drive_root_folder_id:
            try:
                metadata = self.service.files().get(
                    fileId=settings.drive_root_folder_id,
                    fields="id,name,parents,trashed",
                ).execute()
                workspace_name = metadata["name"]
                if workspace_name.startswith("EMT Segmentation"):
                    workspace_name = f"StimTrace{workspace_name[len('EMT Segmentation'):]}"
                    self.service.files().update(
                        fileId=settings.drive_root_folder_id,
                        body={"name": workspace_name},
                        fields="id,name",
                    ).execute()
                settings.workspace_folder_name = workspace_name
                if metadata.get("trashed"):
                    self.service.files().update(
                        fileId=settings.drive_root_folder_id,
                        body={"trashed": False},
                        fields="id,trashed",
                    ).execute()
                if not metadata.get("parents"):
                    # Older API-created workspaces can become orphaned. Attach
                    # them to My Drive without changing their ID or job links.
                    self.service.files().update(
                        fileId=settings.drive_root_folder_id,
                        addParents="root",
                        fields="id,parents",
                    ).execute()
            except HttpError as error:
                error_text = str(error)
                unusable_workspace = (
                    error.resp.status == 404
                    or (
                        error.resp.status == 403
                        and "appNotAuthorizedToChild" in error_text
                    )
                )
                if not unusable_workspace:
                    raise
                # A manually selected, deleted, or partially authorized folder
                # cannot reliably be repaired with the drive.file scope.
                self._clear_workspace_settings(settings)
        if not settings.drive_root_folder_id:
            settings.workspace_folder_name = f"StimTrace {uuid.uuid4().hex[:8]}"
            settings.drive_root_folder_id = self._create_folder(settings.workspace_folder_name)
        elif settings.workspace_folder_name == "StimTrace":
            # Migrate the initial prototype name so it cannot collide with an
            # older manually created folder in the Colab Drive mount.
            settings.workspace_folder_name = f"StimTrace {settings.drive_root_folder_id[-8:]}"
            self.service.files().update(
                fileId=settings.drive_root_folder_id,
                body={"name": settings.workspace_folder_name}, fields="id"
            ).execute()
        settings.google_account_email = self.account_email
        if settings.notebook_version != NOTEBOOK_VERSION:
            settings.notebook_file_id = ""
        needs_notebook_update = settings.notebook_version != NOTEBOOK_VERSION
        if needs_notebook_update:
            settings.notebook_file_id = ""
        if not trusted_existing_assets:
            for setting_name in (
                "model_file_id",
                "worker_file_id",
                "notebook_file_id",
                "marker_file_id",
                "worker_status_file_id",
            ):
                file_id = getattr(settings, setting_name)
                if not file_id:
                    continue
                try:
                    parents = self.service.files().get(
                        fileId=file_id,
                        fields="parents",
                    ).execute().get("parents", [])
                    if settings.drive_root_folder_id not in parents:
                        setattr(settings, setting_name, "")
                except HttpError:
                    setattr(settings, setting_name, "")
        if not settings.worker_status_file_id:
            worker_status = self.service.files().create(
                body={"name": "worker_status.json", "parents": [settings.drive_root_folder_id]},
                media_body=MediaInMemoryUpload(
                    json.dumps({"state": "offline", "updated_at": 0}).encode("utf-8"),
                    mimetype="application/json",
                ),
                fields="id",
            ).execute()
            settings.worker_status_file_id = worker_status["id"]
        assets = [
            (
                "model_file_id",
                "model.pth",
                [
                    RESOURCE_DIR / "model.pth",
                    RESOURCE_DIR / "unet_multitask_center_ellipse_512x512 - Backup.pth",
                    SOURCE_DIR.parent / "unet_multitask_center_ellipse_512x512 - Backup.pth",
                ],
                None,
            ),
            ("worker_file_id", "colab_worker.py", [RESOURCE_DIR / "colab_worker.py"], "text/x-python"),
            (
                "notebook_file_id",
                "StimTrace Compute.ipynb",
                [
                    RESOURCE_DIR / "stimtrace_colab_worker.ipynb",
                    RESOURCE_DIR / "emt_colab_worker.ipynb",
                ],
                "application/vnd.google.colaboratory",
            ),
        ]
        for setting_name, drive_name, candidates, mime_type in assets:
            current_file_id = getattr(settings, setting_name)
            if current_file_id:
                if (
                    setting_name == "worker_file_id"
                    and needs_notebook_update
                    and candidates[0].exists()
                ):
                    self._replace_with_local_file(current_file_id, candidates[0], mime_type)
                continue
            search_names = [drive_name]
            if setting_name == "notebook_file_id":
                search_names.append("EMT Compute.ipynb")
            existing = []
            for search_name in search_names:
                existing = self.service.files().list(
                    q=(
                        f"'{settings.drive_root_folder_id}' in parents and "
                        f"name = '{search_name}' and trashed = false"
                    ),
                    spaces="drive",
                    fields="files(id)",
                    pageSize=1,
                ).execute().get("files", [])
                if existing:
                    break
            if existing and not (setting_name == "notebook_file_id" and needs_notebook_update):
                setattr(settings, setting_name, existing[0]["id"])
                if (
                    setting_name == "worker_file_id"
                    and needs_notebook_update
                    and candidates[0].exists()
                ):
                    self._replace_with_local_file(existing[0]["id"], candidates[0], mime_type)
                continue
            source = next((path for path in candidates if path.exists()), None)
            if not source:
                continue
            body = {"name": drive_name, "parents": [settings.drive_root_folder_id]}
            if mime_type:
                body["mimeType"] = mime_type
            if setting_name == "notebook_file_id":
                notebook = source.read_text(encoding="utf-8")
                notebook = notebook.replace("__STIMTRACE_WORKSPACE_FOLDER__", settings.workspace_folder_name)
                notebook = notebook.replace("__STIMTRACE_WORKSPACE_ID__", settings.drive_root_folder_id)
                notebook = notebook.replace("__STIMTRACE_GOOGLE_ACCOUNT__", settings.google_account_email)
                notebook = notebook.replace("__STIMTRACE_WORKER_FILE_ID__", settings.worker_file_id)
                notebook = notebook.replace("__STIMTRACE_MODEL_FILE_ID__", settings.model_file_id)
                notebook = notebook.replace(
                    "__STIMTRACE_WORKER_STATUS_FILE_ID__",
                    settings.worker_status_file_id,
                )
                media = MediaInMemoryUpload(notebook.encode("utf-8"), mimetype=mime_type, resumable=True)
            else:
                media = MediaFileUpload(str(source), mimetype=mime_type, resumable=True)
            if setting_name == "notebook_file_id" and existing and needs_notebook_update:
                uploaded = self.service.files().update(
                    fileId=existing[0]["id"],
                    body={"name": drive_name},
                    media_body=media,
                    fields="id",
                ).execute()
            else:
                uploaded = self.service.files().create(
                    body=body,
                    media_body=media,
                    fields="id",
                ).execute()
            setattr(settings, setting_name, uploaded["id"])
        if settings.model_file_id and (
            not settings.default_model_file_id
            or settings.selected_model_name == "Default pillar"
        ):
            settings.default_model_file_id = settings.model_file_id
        if settings.default_model_file_id:
            settings.model_profiles.setdefault("Default pillar", settings.default_model_file_id)
        if settings.selected_model_name not in settings.model_profiles:
            settings.selected_model_name = "Default pillar"
            settings.model_file_id = settings.default_model_file_id or settings.model_file_id
            settings.apply_model_parameters(settings.selected_model_name)
        if settings.marker_file_id and needs_notebook_update:
            try:
                self.service.files().update(
                    fileId=settings.marker_file_id,
                    body={"name": "STIMTRACE_WORKSPACE.json"},
                    fields="id,name",
                ).execute()
            except HttpError:
                LOGGER.debug("Could not rename the existing workspace marker", exc_info=True)
        if not settings.marker_file_id:
            marker = json.dumps({
                "folder_id": settings.drive_root_folder_id,
                "workspace_name": settings.workspace_folder_name,
            }).encode("utf-8")
            uploaded = self.service.files().create(
                body={"name": "STIMTRACE_WORKSPACE.json", "parents": [settings.drive_root_folder_id]},
                media_body=MediaInMemoryUpload(marker, mimetype="application/json"),
                fields="id",
            ).execute()
            settings.marker_file_id = uploaded["id"]
        settings.notebook_version = NOTEBOOK_VERSION
        settings.save()
        return settings.drive_root_folder_id

    def submit(
        self,
        videos: list[Path],
        study: str,
        settings: Settings,
        progress_callback=None,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[str, str, str, str, str]:
        with self._request_lock:
            self._refresh_service_connection()
            return self._submit_unlocked(videos, study, settings, progress_callback, parameters)

    def _submit_unlocked(
        self,
        videos: list[Path],
        study: str,
        settings: Settings,
        progress_callback=None,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[str, str, str, str, str]:
        if not self.service:
            raise RuntimeError("Sign in before submitting a job.")
        if progress_callback:
            progress_callback({"phase": "preparing"})
        root_folder_id = self.ensure_workspace(settings)
        job_id = f"{study.strip().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        incoming = self._create_folder(job_id, root_folder_id)
        for index, video in enumerate(videos):
            # Recreate the resumable request after a TLS/proxy redirect failure.
            # A resumable session URL itself cannot safely be reused after that
            # kind of transport break, but the already-created job folder and
            # previously finished videos remain intact.
            last_error: Exception | None = None
            for attempt in range(4):
                try:
                    request = self.service.files().create(
                        body={"name": video.name, "parents": [incoming]},
                        # The Drive default chunk is so large that normal recordings
                        # show 0% until most/all of the file has transferred.
                        media_body=MediaFileUpload(
                            str(video), resumable=True, chunksize=8 * 1024 * 1024
                        ), fields="id",
                    )
                    # Drive uses HTTP 308 without a Location header to mean
                    # "resumable upload incomplete; send the next chunk".
                    # httplib2 otherwise mistakes that valid response for a
                    # browser redirect and raises RedirectMissingLocation before
                    # googleapiclient can advance the upload session.
                    upload_http = getattr(request, "http", None)
                    previous_follow_redirects = (
                        getattr(upload_http, "follow_redirects", True)
                        if upload_http is not None else True
                    )
                    if upload_http is not None:
                        upload_http.follow_redirects = False
                    response = None
                    try:
                        while response is None:
                            status, response = request.next_chunk(num_retries=3)
                            if progress_callback and status:
                                progress_callback(index, status.progress(), video.name)
                    finally:
                        if upload_http is not None:
                            upload_http.follow_redirects = previous_follow_redirects
                    last_error = None
                    break
                except (OSError, httplib2.HttpLib2Error) as error:
                    last_error = error
                    if attempt == 3:
                        break
                    wait_seconds = 2 ** attempt
                    LOGGER.warning(
                        "Drive upload transport failed for %s; retrying %s/4 in %ss: %s",
                        video.name, attempt + 1, wait_seconds, error,
                    )
                    if progress_callback:
                        progress_callback({
                            "phase": "retrying", "video": video.name,
                            "attempt": attempt + 1, "wait_seconds": wait_seconds,
                        })
                    time.sleep(wait_seconds)
                    self._refresh_service_connection()
            if last_error is not None:
                raise RuntimeError(
                    f"Google Drive could not upload {video.name} after four secure retries. "
                    "Previously uploaded videos remain in the draft job folder; submit again "
                    "after checking the network connection."
                ) from last_error
            if progress_callback:
                progress_callback(index, 1.0, video.name)
        progress = {"state": "queued", "message": "Waiting for Colab compute"}
        progress_file = self.service.files().create(
            body={"name": "progress.json", "parents": [incoming]},
            media_body=MediaInMemoryUpload(json.dumps(progress).encode("utf-8"), mimetype="application/json"),
            fields="id",
        ).execute()
        empty_archive = io.BytesIO()
        with zipfile.ZipFile(empty_archive, "w"):
            pass
        result_archive = self.service.files().create(
            body={"name": "results.zip", "parents": [incoming]},
            media_body=MediaInMemoryUpload(
                empty_archive.getvalue(),
                mimetype="application/zip",
            ),
            fields="id",
        ).execute()
        manifest = {
            "job_id": job_id,
            "study_name": study,
            # Keep this outside the runtime settings so Job History can always
            # report the exact model used, including after a desktop reinstall.
            "model_name": settings.selected_model_name,
            "model_file_id": settings.model_file_id,
            "settings": parameters or settings.parameter_values(),
            "videos": [p.name for p in videos],
            "progress_file_id": progress_file["id"],
            "result_archive_file_id": result_archive["id"],
        }
        # A manifest is the job-ready marker. Upload it last so Colab never sees
        # a job before its complete set of videos has reached Drive.
        self.service.files().create(
            body={"name": "manifest.json", "parents": [incoming]},
            media_body=MediaInMemoryUpload(json.dumps(manifest).encode("utf-8"), mimetype="application/json"), fields="id",
        ).execute()
        return job_id, incoming, progress_file["id"], "", result_archive["id"]

    def submit_training(
        self,
        archive: Path,
        model_name: str,
        epochs: int,
        batch_size: int,
        settings: Settings,
    ) -> dict:
        with self._request_lock:
            self._refresh_service_connection()
            return self._submit_training_unlocked(
                archive, model_name, epochs, batch_size, settings
            )

    def _submit_training_unlocked(
        self,
        archive: Path,
        model_name: str,
        epochs: int,
        batch_size: int,
        settings: Settings,
    ) -> dict:
        if not self.service:
            raise RuntimeError("Sign in before submitting model training.")
        root_folder_id = self.ensure_workspace(settings)
        job_id = f"training_{uuid.uuid4().hex[:8]}"
        folder_id = self._create_folder(job_id, root_folder_id)
        dataset = self._execute_with_backoff(self.service.files().create(
            body={"name": "training_dataset.zip", "parents": [folder_id]},
            media_body=MediaFileUpload(
                str(archive),
                mimetype="application/zip",
                # Training datasets are modest archives. A single multipart upload
                # avoids intermittent proxy/Drive resumable-upload redirects that
                # can fail before returning the required session Location header.
                resumable=False,
            ),
            fields="id",
        ))
        progress = self._execute_with_backoff(self.service.files().create(
            body={"name": "progress.json", "parents": [folder_id]},
            media_body=MediaInMemoryUpload(
                json.dumps({"state": "queued", "message": "Training queued. Waiting for Colab compute."}).encode("utf-8"),
                mimetype="application/json",
            ),
            fields="id",
        ))
        model_output = self._execute_with_backoff(self.service.files().create(
            body={"name": f"{Path(model_name).stem}.pth", "parents": [folder_id]},
            media_body=MediaInMemoryUpload(
                b"",
                mimetype="application/octet-stream",
            ),
            fields="id",
        ))
        manifest = {
            "job_id": job_id,
            "job_type": "training",
            "dataset_file_id": dataset["id"],
            "progress_file_id": progress["id"],
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.0001,
            "model_output_file_id": model_output["id"],
        }
        self._execute_with_backoff(self.service.files().create(
            body={"name": "training_manifest.json", "parents": [folder_id]},
            media_body=MediaInMemoryUpload(
                json.dumps(manifest).encode("utf-8"),
                mimetype="application/json",
            ),
            fields="id",
        ))
        return {
            "job_id": job_id,
            "folder_id": folder_id,
            "progress_file_id": progress["id"],
            "model_name": model_name,
            "model_output_file_id": model_output["id"],
        }

    def import_local_model(self, path: Path, settings: Settings) -> str:
        if not self.service:
            raise RuntimeError("Sign in before importing a segmentation model.")
        if path.suffix.lower() != ".pth" or not path.is_file():
            raise ValueError("Choose a PyTorch model file with the .pth extension.")
        root_folder_id = self.ensure_workspace(settings)
        uploaded = self._execute_with_backoff(self.service.files().create(
            body={"name": path.name, "parents": [root_folder_id]},
            media_body=MediaFileUpload(
                str(path),
                mimetype="application/octet-stream",
                resumable=True,
            ),
            fields="id",
        ))
        profile_name = path.stem
        settings.store_model_parameters()
        settings.model_profiles[profile_name] = uploaded["id"]
        settings.ensure_model_parameters(profile_name)
        inferred_type = "Pillar" if "pillar" in profile_name.lower() else "Other"
        settings.model_metadata[profile_name] = {
            "object_type": inferred_type,
            "source": "Imported",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "dataset_size": "",
        }
        settings.selected_model_name = profile_name
        settings.apply_model_parameters(profile_name)
        settings.model_file_id = uploaded["id"]
        settings.notebook_version = 0
        settings.notebook_file_id = ""
        self.ensure_workspace(settings)
        settings.save()
        return profile_name

    def activate_trained_model(self, source_file_id: str, model_name: str, settings: Settings) -> str:
        if not self.service:
            raise RuntimeError("Sign in before selecting a trained model.")
        root_folder_id = self.ensure_workspace(settings)
        copied = self.service.files().copy(
            fileId=source_file_id,
            body={
                "name": f"{Path(model_name).stem}.pth",
                "parents": [root_folder_id],
            },
            fields="id",
        ).execute()
        profile_name = Path(model_name).stem
        settings.store_model_parameters()
        settings.model_profiles[profile_name] = copied["id"]
        settings.ensure_model_parameters(profile_name)
        settings.model_metadata.setdefault(profile_name, {
            "object_type": (
                "Pillar" if "pillar" in profile_name.lower() else "Other"
            ),
            "source": "Trained",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "dataset_size": "",
        })
        settings.selected_model_name = profile_name
        settings.apply_model_parameters(profile_name)
        settings.model_file_id = copied["id"]
        settings.notebook_version = 0
        settings.notebook_file_id = ""
        self.ensure_workspace(settings)
        settings.save()
        return copied["id"]

    def job_progress(self, progress_file_id: str) -> dict:
        from googleapiclient.http import MediaIoBaseDownload

        with self._request_lock:
            self._refresh_service_connection()
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, self.service.files().get_media(fileId=progress_file_id))
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return json.loads(buffer.getvalue().decode("utf-8"))

    def job_terminal_status(self, folder_id: str) -> dict:
        with self._request_lock:
            self._refresh_service_connection()
            return self._job_terminal_status_unlocked(folder_id)

    def _job_terminal_status_unlocked(self, folder_id: str) -> dict:
        files = self.service.files().list(
            q=f"'{folder_id}' in parents and name = 'status.json' and trashed = false",
            fields="files(id)",
            pageSize=1,
        ).execute().get("files", [])
        if not files:
            return {}
        content = self.service.files().get_media(fileId=files[0]["id"]).execute()
        status = json.loads(content.decode("utf-8"))
        return status if status.get("state") in TERMINAL_JOB_STATES else {}

    def update_job_progress(self, progress_file_id: str, payload: dict) -> None:
        with self._request_lock:
            self._refresh_service_connection()
            self._update_job_progress_unlocked(progress_file_id, payload)

    def _update_job_progress_unlocked(self, progress_file_id: str, payload: dict) -> None:
        self.service.files().update(
            fileId=progress_file_id,
            media_body=MediaInMemoryUpload(
                json.dumps(payload).encode("utf-8"),
                mimetype="application/json",
            ),
        ).execute()

    def file_bytes(self, file_id: str) -> bytes:
        with self._request_lock:
            self._refresh_service_connection()
            return self._file_bytes_unlocked(file_id)

    def _file_bytes_unlocked(self, file_id: str, progress_callback=None) -> bytes:
        """Download a Drive file into memory, retrying transient connection resets."""
        from googleapiclient.http import MediaIoBaseDownload

        def download(buffer: io.BytesIO) -> None:
            downloader = MediaIoBaseDownload(
                buffer, self.service.files().get_media(fileId=file_id)
            )
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if progress_callback and status:
                    progress_callback({
                        "phase": "downloading",
                        "fraction": float(status.progress()),
                        "current_bytes": int(status.resumable_progress),
                        "total_bytes": int(status.total_size or 0),
                    })

        return self._download_with_retry(download)

    @staticmethod
    def _retryable_download_error(error: Exception) -> bool:
        if isinstance(error, HttpError):
            return error.resp.status in {408, 429, 500, 502, 503, 504}
        return isinstance(error, (OSError, httplib2.HttpLib2Error))

    def _download_with_retry(self, download, attempts: int = 5) -> bytes:
        """Restart an interrupted Drive media download from the beginning safely."""
        last_error: Exception | None = None
        for attempt in range(attempts):
            buffer = io.BytesIO()
            try:
                download(buffer)
                return buffer.getvalue()
            except Exception as error:
                last_error = error
                if not self._retryable_download_error(error) or attempt == attempts - 1:
                    raise
                delay = min(12, 2 ** attempt)
                LOGGER.warning(
                    "Drive download interrupted (%s); retrying %s/%s in %ss",
                    error, attempt + 1, attempts, delay,
                )
                time.sleep(delay)
        raise last_error or RuntimeError("Drive download failed.")

    def _download_file_with_retry(self, file_id: str, destination: Path) -> None:
        """Write a complete file atomically; never leave a partial result as final output."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.part")
        partial.unlink(missing_ok=True)
        try:
            data = self.file_bytes(file_id)
            partial.write_bytes(data)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)

    def download_job_results(
        self,
        job_folder_id: str,
        target: Path,
        progress_callback=None,
        result_archive_file_id: str = "",
    ) -> int:
        with self._request_lock:
            self._refresh_service_connection()
            return self._download_job_results_unlocked(
                job_folder_id, target, progress_callback, result_archive_file_id
            )

    def _download_job_results_unlocked(
        self,
        job_folder_id: str,
        target: Path,
        progress_callback=None,
        result_archive_file_id: str = "",
    ) -> int:
        from googleapiclient.http import MediaIoBaseDownload

        if result_archive_file_id:
            archive_data = self._file_bytes_unlocked(
                result_archive_file_id, progress_callback
            )
            try:
                with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
                    members = [item for item in archive.infolist() if not item.is_dir()]
                    target_root = target.resolve()
                    for index, member in enumerate(members, start=1):
                        destination = (target / member.filename).resolve()
                        if target_root not in destination.parents:
                            raise ValueError("The result archive contains an unsafe path.")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, destination.open("wb") as output:
                            shutil.copyfileobj(source, output)
                        if progress_callback:
                            progress_callback({
                                "phase": "extracting", "current": index,
                                "total": len(members),
                                "filename": Path(member.filename).name,
                            })
                    if not members:
                        raise FileNotFoundError("The Drive result archive is empty.")
                    return len(members)
            except zipfile.BadZipFile as error:
                raise ValueError("The Drive result archive is incomplete or invalid.") from error

        children = self.service.files().list(
            q=f"'{job_folder_id}' in parents and name = 'results' and trashed = false",
            fields="files(id,name,mimeType)",
            pageSize=10,
        ).execute().get("files", [])
        result_folder = next(
            (
                item for item in children
                if item["mimeType"] == "application/vnd.google-apps.folder"
            ),
            None,
        )
        if not result_folder:
            raise FileNotFoundError("The completed job has no Drive results folder.")

        files: list[tuple[dict, Path]] = []

        def collect(folder_id: str, relative: Path):
            items = self.service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="files(id,name,mimeType)",
                pageSize=1000,
            ).execute().get("files", [])
            for item in items:
                safe_name = Path(item["name"]).name
                child_relative = relative / safe_name
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    collect(item["id"], child_relative)
                else:
                    files.append((item, child_relative))

        collect(result_folder["id"], Path())
        if not files:
            raise FileNotFoundError("The Drive results folder is empty.")
        target.mkdir(parents=True, exist_ok=True)
        for index, (item, relative) in enumerate(files, start=1):
            destination = target / relative
            self._download_file_with_retry(item["id"], destination)
            if progress_callback:
                progress_callback({
                    "phase": "files", "current": index,
                    "total": len(files), "filename": relative.name,
                })
        return len(files)

    def download_training_package(
        self,
        job_folder_id: str,
        target: Path,
        model_output_file_id: str = "",
    ) -> list[str]:
        with self._request_lock:
            self._refresh_service_connection()
            return self._download_training_package_unlocked(
                job_folder_id, target, model_output_file_id
            )

    def _download_training_package_unlocked(
        self,
        job_folder_id: str,
        target: Path,
        model_output_file_id: str = "",
    ) -> list[str]:
        """Download the portable model-training deliverables from one Drive job."""
        if not self.service:
            raise RuntimeError("Sign in before downloading the offline training package.")
        from googleapiclient.http import MediaIoBaseDownload

        wanted = {
            "training_manifest.json",
            "training_history.csv",
            "training_qc_summary.json",
            "training_history.png",
            "dataset_phase_qc.png",
            "training_report.html",
        }
        children = self.service.files().list(
            q=f"'{job_folder_id}' in parents and trashed = false",
            fields="files(id,name,mimeType)",
            pageSize=100,
        ).execute().get("files", [])
        selected = [item for item in children if item.get("name") in wanted]
        if model_output_file_id:
            model_item = next(
                (item for item in children if item.get("id") == model_output_file_id),
                None,
            )
            if model_item is not None:
                selected.append(model_item)
        # Preserve backwards compatibility with jobs created before the output ID was saved.
        if not any(str(item.get("name", "")).lower().endswith(".pth") for item in selected):
            selected.extend(
                item for item in children
                if str(item.get("name", "")).lower().endswith(".pth")
            )
        unique = {str(item["id"]): item for item in selected}
        if not unique:
            raise FileNotFoundError("The training job has no downloadable model or QC files yet.")
        target.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        for item in unique.values():
            filename = Path(str(item.get("name", ""))).name
            if not filename:
                continue
            destination = target / filename
            self._download_file_with_retry(item["id"], destination)
            downloaded.append(filename)
        return sorted(downloaded, key=str.lower)

    def cancel_job(self, folder_id: str, progress_file_id: str) -> dict:
        with self._request_lock:
            self._refresh_service_connection()
            return self._cancel_job_unlocked(folder_id, progress_file_id)

    def _cancel_job_unlocked(self, folder_id: str, progress_file_id: str) -> dict:
        try:
            current = self.job_progress(progress_file_id)
            if current.get("state") in TERMINAL_JOB_STATES:
                return current
        except Exception:
            LOGGER.debug("Could not read the job state before cancellation", exc_info=True)
        children = self.service.files().list(
            q=f"'{folder_id}' in parents and name = 'cancel.json' and trashed = false",
            fields="files(id)",
            pageSize=1,
        ).execute().get("files", [])
        if not children:
            self.service.files().create(
                body={"name": "cancel.json", "parents": [folder_id]},
                media_body=MediaInMemoryUpload(
                    json.dumps({"requested_at": datetime.now(timezone.utc).isoformat()}).encode("utf-8"),
                    mimetype="application/json",
                ),
                fields="id",
            ).execute()
        try:
            latest = self.job_progress(progress_file_id)
            if latest.get("state") in TERMINAL_JOB_STATES:
                return latest
        except Exception:
            LOGGER.debug("Could not re-read the job state during cancellation", exc_info=True)
        cancelling = {
            "state": "cancelling",
            "message": "Cancellation requested. Waiting for Colab.",
        }
        self.service.files().update(
            fileId=progress_file_id,
            media_body=MediaInMemoryUpload(
                json.dumps(cancelling).encode("utf-8"),
                mimetype="application/json",
            ),
        ).execute()
        return cancelling

    def retry_job(self, folder_id: str, progress_file_id: str, preview_file_id: str) -> None:
        with self._request_lock:
            last_error: OSError | None = None
            for attempt in range(4):
                try:
                    self._refresh_service_connection()
                    self._retry_job_unlocked(folder_id, progress_file_id, preview_file_id)
                    return
                except OSError as error:
                    last_error = error
                    if attempt == 3:
                        break
                    delay = 2 ** attempt
                    LOGGER.warning(
                        "Drive retry request lost its TLS connection; retrying %s/4 in %ss: %s",
                        attempt + 1, delay, error,
                    )
                    time.sleep(delay)
            raise RuntimeError(
                "StimTrace could not reach Google Drive securely after several attempts. "
                "Your job was not changed; check the internet connection and try Retry again."
            ) from last_error

    def _retry_job_unlocked(
        self, folder_id: str, progress_file_id: str, preview_file_id: str
    ) -> None:
        children = self.service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id,name,mimeType)",
            pageSize=1000,
        ).execute().get("files", [])
        for child in children:
            if child["name"] in {"status.json", "cancel.json", "results"}:
                try:
                    self.service.files().update(fileId=child["id"], body={"trashed": True}).execute()
                except HttpError:
                    LOGGER.warning(
                        "Could not remove stale Drive job artifact %s",
                        child.get("name", child.get("id", "unknown")),
                        exc_info=True,
                    )
            elif child["name"] == "results.zip":
                empty_archive = io.BytesIO()
                with zipfile.ZipFile(empty_archive, "w"):
                    pass
                self.service.files().update(
                    fileId=child["id"],
                    media_body=MediaInMemoryUpload(
                        empty_archive.getvalue(),
                        mimetype="application/zip",
                    ),
                ).execute()
        self.service.files().update(
            fileId=progress_file_id,
            media_body=MediaInMemoryUpload(
                json.dumps({"state": "queued", "message": "Retry queued. Waiting for Colab compute."}).encode("utf-8"),
                mimetype="application/json",
            ),
        ).execute()
        if preview_file_id:
            self.service.files().update(
                fileId=preview_file_id,
                media_body=MediaInMemoryUpload(b"", mimetype="image/jpeg"),
            ).execute()

    def discover_jobs(self, root_folder_id: str) -> list[dict]:
        with self._request_lock:
            self._refresh_service_connection()
            return self._discover_jobs_unlocked(root_folder_id)

    def _discover_jobs_unlocked(self, root_folder_id: str) -> list[dict]:
        folders = self.service.files().list(
            q=(
                f"'{root_folder_id}' in parents and "
                "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            ),
            fields="files(id,name,createdTime)",
            pageSize=1000,
        ).execute().get("files", [])
        discovered = []
        for folder in folders:
            children = self.service.files().list(
                q=f"'{folder['id']}' in parents and trashed = false",
                fields="files(id,name,modifiedTime)",
                pageSize=1000,
            ).execute().get("files", [])
            by_name = {child["name"]: child for child in children}
            manifest_name = (
                "manifest.json" if "manifest.json" in by_name
                else "training_manifest.json" if "training_manifest.json" in by_name
                else ""
            )
            if not manifest_name:
                continue
            try:
                manifest = json.loads(self.file_bytes(by_name[manifest_name]["id"]).decode("utf-8"))
                progress_id = manifest.get("progress_file_id", by_name.get("progress.json", {}).get("id", ""))
                preview_id = manifest.get("preview_file_id", by_name.get("preview.jpg", {}).get("id", ""))
                result_archive_id = manifest.get(
                    "result_archive_file_id",
                    by_name.get("results.zip", {}).get("id", ""),
                )
                progress = self.job_progress(progress_id) if progress_id else {}
                progress_modified_at = by_name.get("progress.json", {}).get("modifiedTime", "")
                history_fields = {
                    key: progress[key]
                    for key in (
                        "processing_started_at",
                        "processing_finished_at",
                        "actual_total_seconds",
                        "processed_frames",
                        "total_job_frames",
                    )
                    if key in progress
                }
                if (
                    progress.get("state") in TERMINAL_JOB_STATES
                    and "processing_finished_at" not in history_fields
                    and progress_modified_at
                ):
                    history_fields["processing_finished_at"] = progress_modified_at
                if (
                    "actual_total_seconds" not in history_fields
                    and history_fields.get("processing_started_at")
                    and history_fields.get("processing_finished_at")
                ):
                    finished = datetime.fromisoformat(
                        str(history_fields["processing_finished_at"]).replace("Z", "+00:00")
                    )
                    duration = elapsed_seconds(history_fields["processing_started_at"], finished)
                    if duration is not None:
                        history_fields["actual_total_seconds"] = duration
                discovered.append({
                    "job_id": manifest.get("job_id", folder["name"]),
                    "study": manifest.get("study_name", manifest.get("model_name", folder["name"])),
                    "type": manifest.get("job_type", "segmentation"),
                    "model_name": (
                        manifest.get("model_name")
                        or manifest.get("settings", {}).get("selected_model_name", "")
                    ),
                    "folder_id": folder["id"],
                    "progress_file_id": progress_id,
                    "preview_file_id": preview_id,
                    "result_archive_file_id": result_archive_id,
                    "videos": manifest.get("videos", []),
                    "state": progress.get("state", "unknown"),
                    "message": progress.get("message", ""),
                    "progress_modified_at": progress_modified_at,
                    "created_at": folder.get("createdTime", ""),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    **history_fields,
                })
            except Exception:
                LOGGER.warning(
                    "Ignoring an unreadable Drive job folder %s",
                    folder.get("id", "unknown"),
                    exc_info=True,
                )
                continue
        return discovered


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings: Settings,
        parent: QWidget | None = None,
        cloud_hardware: dict[str, Any] | None = None,
        kalman_benchmark_combinations: list[dict[str, Any]] | None = None,
        on_kalman_benchmark_change=None,
        calibration_videos: list[Path] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced settings")
        self.setObjectName("advancedSettingsDialog")
        self.setMinimumWidth(640)
        self.setStyleSheet(
            """
            QDialog#advancedSettingsDialog QComboBox,
            QDialog#advancedSettingsDialog QSpinBox,
            QDialog#advancedSettingsDialog QDoubleSpinBox {
                min-height: 27px;
                max-height: 27px;
                padding-top: 0;
                padding-bottom: 0;
            }
            """
        )
        self.settings = settings
        self.calibration_videos = [Path(path) for path in (calibration_videos or [])]
        self.kalman_benchmark_combinations = list(kalman_benchmark_combinations or [])
        self.on_kalman_benchmark_change = on_kalman_benchmark_change
        for model_name in settings.model_profiles or [settings.selected_model_name]:
            settings.ensure_model_parameters(model_name)
        self.parameter_profiles = {
            name: dict(values)
            for name, values in settings.model_parameter_profiles.items()
        }
        self.editing_model_name = settings.selected_model_name
        layout = QVBoxLayout(self)
        workspace_layout = QFormLayout()
        hardware = cloud_hardware or {}
        if hardware:
            device_name = hardware.get("device_name", hardware.get("device", "Unknown"))
            gpu_memory = hardware.get("gpu_memory_gb", 0)
            system_memory = hardware.get("system_memory_gb", 0)
            cpu_threads = hardware.get("cpu_threads", 0)
            utilization = hardware.get("gpu_utilization_pct")
            runtime_parts = [str(device_name)]
            if gpu_memory:
                runtime_parts.append(f"{gpu_memory:g} GB GPU memory")
            if system_memory:
                runtime_parts.append(f"{system_memory:g} GB system RAM")
            if cpu_threads:
                runtime_parts.append(f"{cpu_threads} logical CPU threads")
            if utilization is not None:
                runtime_parts.append(f"{utilization:g}% GPU at last sample")
            runtime = QLabel(" | ".join(runtime_parts))
        else:
            runtime = QLabel("Start the Colab worker to detect its GPU, RAM, and CPU capacity.")
        runtime.setWordWrap(True)
        runtime.setProperty("role", "muted")
        workspace_layout.addRow("Connected Colab runtime", runtime)
        layout.addLayout(workspace_layout)

        analysis_box = QGroupBox("Analysis settings")
        analysis_layout = QFormLayout(analysis_box)
        analysis_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        analysis_note = QLabel(
            "Select the segmentation model and enter the calibration values measured for "
            "this experimental setup. Review the tracking and output settings for each analysis."
        )
        analysis_note.setWordWrap(True)
        analysis_note.setProperty("role", "muted")
        analysis_layout.addRow(analysis_note)
        self.model_selector = QComboBox()
        self.model_selector.addItems(list(settings.model_profiles) or ["Default pillar"])
        current_index = self.model_selector.findText(settings.selected_model_name)
        self.model_selector.setCurrentIndex(max(0, current_index))
        self.model_selector.setToolTip(
            "The default pillar model remains available. Select a custom binary model only for matching image types."
        )
        analysis_layout.addRow("Segmentation model", self.model_selector)
        self.fields: dict[str, Any] = {}
        for key, label, minimum, maximum, decimals in [
            ("pixel_to_um", "Pixel calibration (um/px)", 0.001, 100, 4),
            ("force_slope_un_per_um", "Force slope (uN/um)", 0, 1000, 4),
        ]:
            field = QDoubleSpinBox()
            field.setRange(minimum, maximum)
            field.setDecimals(decimals)
            field.setValue(getattr(settings, key))
            if key == "pixel_to_um":
                calibration_row = QWidget()
                calibration_layout = QHBoxLayout(calibration_row)
                calibration_layout.setContentsMargins(0, 0, 0, 0)
                calibration_layout.setSpacing(8)
                calibration_layout.addWidget(field, 1)
                calibrate = QPushButton("Measure from video")
                set_action_icon(calibrate, "open")
                calibrate.setToolTip(
                    "Draw an ellipse around a known object in a video frame and calculate um/px."
                )
                calibrate.clicked.connect(self.open_pixel_calibration)
                calibration_layout.addWidget(calibrate)
                self.add_described_row(analysis_layout, label, key, calibration_row)
            else:
                self.add_described_row(analysis_layout, label, key, field)
            self.fields[key] = field
        force_note = QLabel(
            "StimTrace reports active contraction force relative to a diastolic position "
            "from the recording. Absolute passive or preload force is not calculated."
        )
        force_note.setWordWrap(True)
        force_note.setProperty("role", "muted")
        analysis_layout.addRow(force_note)
        self.tracking_filter = QComboBox()
        self.tracking_filter.addItem("Kalman smoothing", "kalman")
        self.tracking_filter.addItem("None (raw segmentation centers)", "none")
        self.add_described_row(
            analysis_layout,
            "Tracking filter",
            "tracking_filter_mode",
            self.tracking_filter,
        )
        self.cloud_overlays = QComboBox()
        self.cloud_overlays.addItem("Create overlays locally on CPU after results download", "local")
        self.cloud_overlays.addItem("Create overlays in Colab", "cloud")
        self.cloud_overlays.setCurrentIndex(
            max(0, self.cloud_overlays.findData(settings.cloud_overlay_mode))
        )
        self.cloud_overlays.setToolTip(
            "Local CPU overlays save Colab runtime. StimTrace uses the downloaded tracking CSV "
            "and the original videos still on this computer."
        )
        analysis_layout.addRow("Cloud overlay rendering", self.cloud_overlays)
        layout.addWidget(analysis_box)

        expert_box = QGroupBox("Expert settings")
        expert_layout = QFormLayout(expert_box)
        expert_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        expert_note = QLabel(
            "These options change segmentation, tracking, or compute behavior. Keep the "
            "model defaults unless a controlled benchmark supports a different value."
        )
        expert_note.setWordWrap(True)
        expert_note.setProperty("role", "muted")
        expert_layout.addRow(expert_note)
        if on_kalman_benchmark_change is not None:
            benchmark_row = QWidget()
            benchmark_layout = QHBoxLayout(benchmark_row)
            benchmark_layout.setContentsMargins(0, 0, 0, 0)
            benchmark_button = QPushButton("Configure Kalman benchmark")
            set_action_icon(benchmark_button, "settings")
            benchmark_button.setToolTip(
                "Expert tool: segment once and compare several Kalman tracking configurations."
            )
            benchmark_button.clicked.connect(self.configure_kalman_benchmark)
            self.benchmark_summary = QLabel()
            self.benchmark_summary.setProperty("role", "muted")
            self.update_benchmark_summary()
            benchmark_layout.addWidget(benchmark_button)
            benchmark_layout.addWidget(self.benchmark_summary, 1)
            expert_layout.addRow("Kalman benchmark", benchmark_row)
        self.axis_mode = QComboBox()
        self.axis_mode.addItem("Automatic from tracked motion", "automatic")
        self.axis_mode.addItem("Fixed image angle", "fixed_angle")
        self.add_described_row(
            expert_layout,
            "Pillar bending axis",
            "bending_axis_mode",
            self.axis_mode,
        )
        self.axis_angle = QDoubleSpinBox()
        self.axis_angle.setRange(-360.0, 360.0)
        self.axis_angle.setDecimals(1)
        self.axis_angle.setSuffix(" deg")
        self.add_described_row(
            expert_layout,
            "Bending-axis angle",
            "bending_axis_angle_deg",
            self.axis_angle,
        )
        self.area_normalization = QCheckBox(
            "Scale displacement by maximum area / current area"
        )
        self.add_described_row(
            expert_layout,
            "Legacy area normalization",
            "legacy_area_normalization",
            self.area_normalization,
        )
        self.innovation_gate = QCheckBox(
            "Reject statistically implausible segmented centers"
        )
        self.innovation_gate_label = self.add_described_row(
            expert_layout,
            "Innovation gate",
            "kalman_innovation_gate_enabled",
            self.innovation_gate,
        )
        self.innovation_gate_confidence = QDoubleSpinBox()
        self.innovation_gate_confidence.setRange(90.0, 99.99)
        self.innovation_gate_confidence.setDecimals(2)
        self.innovation_gate_confidence.setSingleStep(0.1)
        self.innovation_gate_confidence.setSuffix(" %")
        self.innovation_gate_confidence_label = self.add_described_row(
            expert_layout,
            "Innovation-gate confidence",
            "kalman_innovation_gate_confidence",
            self.innovation_gate_confidence,
        )
        self.kalman_labels: dict[str, QLabel] = {}
        for key, label, minimum, maximum, decimals in [
            ("mask_threshold", "Mask threshold", 0.01, 0.99, 2),
            ("kalman_q_pos", "Kalman position variance (px^2)", 0, 1000, 2),
            ("kalman_q_vel", "Kalman velocity variance (px^2/s^2)", 0, 1000, 2),
            ("kalman_r", "Kalman measurement variance (px^2)", 0.01, 1000, 2),
            ("kalman_innovation_gate_min_radius_px", "Minimum gate radius (px)", 0, 2000, 1),
            ("progress_interval_seconds", "Progress update interval (seconds)", 2, 30, 1),
        ]:
            field = QDoubleSpinBox()
            field.setRange(minimum, maximum)
            field.setDecimals(decimals)
            field.setValue(getattr(settings, key))
            field_label = self.add_described_row(expert_layout, label, key, field)
            if key in (
                "kalman_q_pos",
                "kalman_q_vel",
                "kalman_r",
                "kalman_innovation_gate_min_radius_px",
            ):
                self.kalman_labels[key] = field_label
            self.fields[key] = field
        self.refine = QSpinBox()
        self.refine.setRange(0, 30)
        self.refine.setValue(settings.refine_iterations)
        self.add_described_row(
            expert_layout,
            "Ellipse refinement iterations",
            "refine_iterations",
            self.refine,
        )
        self.batch_size = QSpinBox()
        # This is a requested ceiling, not a hardware claim. The cloud worker
        # probes upward and backs off on CUDA OOM, so high-memory GPUs (for
        # example an A100) must not be artificially limited by the desktop UI.
        self.batch_size.setRange(1, 4096)
        self.batch_size.setValue(settings.inference_batch_size)
        self.add_described_row(
            expert_layout,
            "Maximum GPU inference batch size",
            "inference_batch_size",
            self.batch_size,
        )
        self.cpu_workers = QSpinBox()
        self.cpu_workers.setRange(1, 32)
        self.cpu_workers.setValue(settings.cpu_postprocess_workers)
        self.add_described_row(
            expert_layout,
            "CPU postprocess workers",
            "cpu_postprocess_workers",
            self.cpu_workers,
        )
        compact_editors = [
            self.model_selector,
            self.axis_mode,
            self.axis_angle,
            self.tracking_filter,
            self.innovation_gate,
            self.innovation_gate_confidence,
            *self.fields.values(),
            self.refine,
            self.batch_size,
            self.cpu_workers,
            self.cloud_overlays,
        ]
        for editor in compact_editors:
            # Use the complete form field column. A fixed width left substantial
            # unused space on wider displays and truncated longer dropdown text.
            editor.setMinimumWidth(0)
            editor.setMaximumWidth(16777215)
            editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        applies_note = QLabel(
            "Performance changes apply to the next submitted cloud job. Restarting the "
            "Colab worker is not required."
        )
        applies_note.setWordWrap(True)
        applies_note.setProperty("role", "muted")
        expert_layout.addRow(applies_note)
        layout.addWidget(expert_box)
        self.model_selector.currentTextChanged.connect(self.change_model_profile)
        self.axis_mode.currentIndexChanged.connect(self.update_axis_fields)
        self.tracking_filter.currentIndexChanged.connect(self.update_tracking_filter_fields)
        self.innovation_gate.toggled.connect(self.update_tracking_filter_fields)
        self.load_parameter_values(self.parameter_profiles[self.editing_model_name])
        buttons = QHBoxLayout()
        restore = QPushButton("Restore model defaults")
        set_action_icon(restore, "clear")
        restore.setToolTip(
            "Load this model's immutable default values into the fields. "
            "Click Save to apply them, or Cancel to discard the reset."
        )
        restore.clicked.connect(self.restore_model_defaults)
        save = QPushButton("Save")
        set_action_icon(save, "save")
        save.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        set_action_icon(cancel, "cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(restore)
        buttons.addStretch(1)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

    @staticmethod
    def add_described_row(
        layout: QFormLayout,
        text: str,
        key: str,
        field: QWidget,
    ) -> QLabel:
        description = SETTING_DESCRIPTIONS[key]
        label = QLabel(text)
        label.setToolTip(description)
        label.setAccessibleDescription(description)
        field.setToolTip(description)
        field.setAccessibleDescription(description)
        layout.addRow(label, field)
        return label

    def restore_model_defaults(self) -> None:
        model_name = self.model_selector.currentText() or self.editing_model_name
        defaults = self.settings.model_default_parameters(model_name)
        self.load_parameter_values(defaults)

    def open_pixel_calibration(self) -> None:
        from pixel_calibration import PixelCalibrationDialog

        # Sample across the current analysis selection so repeated calibrations
        # are not always biased toward the first recordings in the list.  The
        # dialog still offers Select 3 videos when fewer than three are loaded or
        # when a different matched-camera set is needed.
        initial_videos = choose_calibration_videos(self.calibration_videos)
        dialog = PixelCalibrationDialog(
            self.fields["pixel_to_um"].value(),
            self,
            initial_videos=initial_videos,
        )
        if dialog.exec() == QDialog.Accepted:
            self.fields["pixel_to_um"].setValue(dialog.calibration_um_per_px)

    def update_benchmark_summary(self) -> None:
        count = len(self.kalman_benchmark_combinations)
        self.benchmark_summary.setText(
            f"{count} configuration{'s' if count != 1 else ''} active"
            if count else "Disabled"
        )

    def configure_kalman_benchmark(self) -> None:
        dialog = KalmanBenchmarkDialog(
            self.kalman_benchmark_combinations or DEFAULT_KALMAN_BENCHMARK,
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.kalman_benchmark_combinations = dialog.combinations
        self.update_benchmark_summary()
        self.on_kalman_benchmark_change(self.kalman_benchmark_combinations)

    def ui_parameter_values(self) -> dict[str, Any]:
        values = {key: field.value() for key, field in self.fields.items()}
        values.update({
            "bending_axis_mode": self.axis_mode.currentData(),
            "bending_axis_angle_deg": self.axis_angle.value(),
            "legacy_area_normalization": self.area_normalization.isChecked(),
            "tracking_filter_mode": self.tracking_filter.currentData(),
            "kalman_innovation_gate_enabled": self.innovation_gate.isChecked(),
            "kalman_innovation_gate_confidence": (
                self.innovation_gate_confidence.value() / 100.0
            ),
            "refine_iterations": self.refine.value(),
            "inference_batch_size": self.batch_size.value(),
            "cpu_postprocess_workers": self.cpu_workers.value(),
        })
        return values

    def load_parameter_values(self, values: dict[str, Any]) -> None:
        for key, field in self.fields.items():
            field.setValue(values.get(key, getattr(self.settings, key)))
        axis_index = self.axis_mode.findData(values.get("bending_axis_mode", "automatic"))
        self.axis_mode.setCurrentIndex(max(0, axis_index))
        self.axis_angle.setValue(float(values.get("bending_axis_angle_deg", 0.0) or 0.0))
        self.area_normalization.setChecked(bool(values.get("legacy_area_normalization", False)))
        filter_index = self.tracking_filter.findData(
            values.get("tracking_filter_mode", "kalman")
        )
        self.tracking_filter.setCurrentIndex(max(0, filter_index))
        self.innovation_gate.setChecked(
            bool(values.get("kalman_innovation_gate_enabled", True))
        )
        self.innovation_gate_confidence.setValue(
            100.0 * float(values.get("kalman_innovation_gate_confidence", 0.99))
        )
        self.refine.setValue(int(values.get("refine_iterations", self.settings.refine_iterations)))
        self.batch_size.setValue(int(values.get("inference_batch_size", self.settings.inference_batch_size)))
        self.cpu_workers.setValue(int(values.get("cpu_postprocess_workers", self.settings.cpu_postprocess_workers)))
        self.update_axis_fields()
        self.update_tracking_filter_fields()

    def update_axis_fields(self) -> None:
        fixed_axis = self.axis_mode.currentData() == "fixed_angle"
        self.axis_angle.setEnabled(fixed_axis)

    def update_tracking_filter_fields(self) -> None:
        kalman_enabled = self.tracking_filter.currentData() == "kalman"
        for key in (
            "kalman_q_pos",
            "kalman_q_vel",
            "kalman_r",
            "kalman_innovation_gate_min_radius_px",
        ):
            self.fields[key].setEnabled(kalman_enabled)
            self.kalman_labels[key].setEnabled(kalman_enabled)
        self.fields["kalman_innovation_gate_min_radius_px"].setEnabled(
            kalman_enabled and self.innovation_gate.isChecked()
        )
        self.kalman_labels["kalman_innovation_gate_min_radius_px"].setEnabled(
            kalman_enabled and self.innovation_gate.isChecked()
        )
        self.innovation_gate.setEnabled(kalman_enabled)
        self.innovation_gate_label.setEnabled(kalman_enabled)
        confidence_enabled = kalman_enabled and self.innovation_gate.isChecked()
        self.innovation_gate_confidence.setEnabled(confidence_enabled)
        self.innovation_gate_confidence_label.setEnabled(confidence_enabled)

    def change_model_profile(self, model_name: str) -> None:
        if not model_name or model_name == self.editing_model_name:
            return
        try:
            current_values = self.ui_parameter_values()
        except ValueError:
            current_values = dict(self.parameter_profiles[self.editing_model_name])
        self.parameter_profiles[self.editing_model_name] = current_values
        self.parameter_profiles.setdefault(model_name, dict(current_values))
        self.load_parameter_values(self.parameter_profiles[model_name])
        self.editing_model_name = model_name

    def accept(self) -> None:
        selected_model_name = self.model_selector.currentText()
        selected_values = self.ui_parameter_values()
        self.parameter_profiles[selected_model_name] = selected_values
        self.settings.model_parameter_profiles = self.parameter_profiles
        self.settings.cloud_overlay_mode = str(self.cloud_overlays.currentData())
        for key, value in selected_values.items():
            setattr(self.settings, key, value)
        selected_model_id = self.settings.model_profiles.get(selected_model_name, "")
        if (
            selected_model_name != self.settings.selected_model_name
            or selected_model_id != self.settings.model_file_id
        ):
            self.settings.selected_model_name = selected_model_name
            self.settings.model_file_id = selected_model_id
            self.settings.notebook_version = 0
            self.settings.notebook_file_id = ""
        else:
            self.settings.selected_model_name = selected_model_name
        self.settings.save()
        super().accept()


class KalmanBenchmarkDialog(QDialog):
    """Expert editor for replaying one segmentation with several Kalman settings."""

    def __init__(
        self,
        combinations: list[dict[str, Any]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Kalman benchmark")
        self.resize(790, 390)
        layout = QVBoxLayout(self)
        description = QLabel(
            "StimTrace segments each frame once, then replays the raw pillar measurements "
            "through every configuration below. Each configuration receives its own traces "
            "and synchronized tracked-video output."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        units = QLabel(
            "Position and measurement values are variances in px^2. Velocity is a variance in px^2/s^2."
        )
        units.setProperty("role", "muted")
        units.setWordWrap(True)
        layout.addWidget(units)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Configuration", "Position variance (px^2)",
            "Velocity variance (px^2/s^2)", "Measurement variance (px^2)",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 4):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, 1)
        table_actions = QHBoxLayout()
        add = QPushButton("Add configuration")
        set_action_icon(add, "apply")
        add.clicked.connect(lambda: self.add_row())
        remove = QPushButton("Remove selected")
        set_action_icon(remove, "delete")
        remove.setProperty("role", "danger")
        remove.clicked.connect(self.remove_selected)
        reset = QPushButton("Restore five presets")
        set_action_icon(reset, "clear")
        reset.clicked.connect(lambda: self.load_combinations(DEFAULT_KALMAN_BENCHMARK))
        table_actions.addWidget(add)
        table_actions.addWidget(remove)
        table_actions.addWidget(reset)
        table_actions.addStretch(1)
        layout.addLayout(table_actions)
        buttons = QHBoxLayout()
        disable = QPushButton("Disable benchmark")
        set_action_icon(disable, "clear")
        disable.clicked.connect(self.disable_benchmark)
        use = QPushButton("Use benchmark")
        set_action_icon(use, "apply")
        use.setProperty("role", "primary")
        use.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        set_action_icon(cancel, "cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(disable)
        buttons.addStretch(1)
        buttons.addWidget(use)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self.combinations: list[dict[str, Any]] = []
        self.load_combinations(combinations or DEFAULT_KALMAN_BENCHMARK)

    def add_row(self, values: dict[str, Any] | None = None) -> None:
        values = values or {
            "name": f"Configuration {self.table.rowCount() + 1}",
            "kalman_q_pos": 2.0,
            "kalman_q_vel": 36.0,
            "kalman_r": 8.0,
        }
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, key in enumerate(("name", "kalman_q_pos", "kalman_q_vel", "kalman_r")):
            self.table.setItem(row, column, QTableWidgetItem(str(values[key])))

    def load_combinations(self, combinations: list[dict[str, Any]]) -> None:
        self.table.setRowCount(0)
        for values in combinations:
            self.add_row(values)

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def disable_benchmark(self) -> None:
        self.combinations = []
        super().accept()

    def accept(self) -> None:
        combinations = []
        names: set[str] = set()
        try:
            for row in range(self.table.rowCount()):
                items = [self.table.item(row, column) for column in range(4)]
                if any(item is None for item in items):
                    raise ValueError(f"Configuration {row + 1} is incomplete.")
                name = items[0].text().strip()
                if not name:
                    raise ValueError(f"Configuration {row + 1} needs a name.")
                normalized = name.casefold()
                if normalized in names:
                    raise ValueError(f"Configuration names must be unique: {name}")
                names.add(normalized)
                q_pos, q_vel, measurement = (float(item.text()) for item in items[1:])
                if q_pos < 0 or q_vel < 0 or measurement <= 0:
                    raise ValueError("Position and velocity must be non-negative; measurement must be greater than zero.")
                combinations.append({
                    "name": name,
                    "kalman_q_pos": q_pos,
                    "kalman_q_vel": q_vel,
                    "kalman_r": measurement,
                })
        except ValueError as error:
            QMessageBox.warning(self, "Invalid benchmark configuration", str(error))
            return
        if not combinations:
            QMessageBox.warning(self, "No configurations", "Add at least one configuration or disable the benchmark.")
            return
        self.combinations = combinations
        super().accept()


class LocalComputeSettingsDialog(QDialog):
    def __init__(
        self,
        settings: Settings,
        hardware: dict[str, Any],
        running: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Local compute settings - running job"
            if running
            else "Local compute settings"
        )
        self.settings = settings
        self.running = running
        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        cuda_available = bool(hardware.get("cuda_available"))
        cuda_name = hardware.get("cuda_name", "")
        cpu_threads = max(1, int(hardware.get("cpu_threads", 1)))
        detected = (
            f"CPU: {cpu_threads} logical processors"
            + (f"\nGPU: {cuda_name} (CUDA available)" if cuda_available else "\nGPU: no compatible CUDA device detected")
        )
        hardware_label = QLabel(detected)
        hardware_label.setWordWrap(True)
        hardware_label.setToolTip(
            "StimTrace detects hardware through the installed PyTorch environment. "
            "Unsupported or unavailable GPUs automatically fall back to the CPU in Auto mode."
        )
        layout.addRow("Detected hardware", hardware_label)

        self.device = QComboBox()
        self.device.addItem("Automatic (recommended)", "auto")
        self.device.addItem("CPU", "cpu")
        if cuda_available:
            self.device.addItem(f"NVIDIA CUDA - {cuda_name}", "cuda")
        current = self.device.findData(settings.local_device_preference)
        self.device.setCurrentIndex(max(0, current))
        self.device.setToolTip(
            "Automatic uses an NVIDIA CUDA GPU when this Python environment supports it, "
            "otherwise it uses the CPU."
        )
        self.device.setEnabled(not running)
        if running:
            self.device.setToolTip(
                "The processing device cannot change after the model is loaded. "
                "This setting becomes available when the current job finishes."
            )
        layout.addRow("Processing device", self.device)

        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(0, cpu_threads)
        self.cpu_threads.setSpecialValueText("Automatic")
        self.cpu_threads.setValue(min(settings.local_cpu_threads, cpu_threads))
        self.cpu_threads.setToolTip(
            "Maximum PyTorch CPU threads. Automatic uses the environment default. "
            "Lower this if processing makes the computer unresponsive."
        )
        self.cpu_threads.setEnabled(not running)
        if running:
            self.cpu_threads.setToolTip(
                "PyTorch CPU threads are fixed when a job starts. This setting becomes "
                "available when the current job finishes."
            )
        layout.addRow("CPU threads", self.cpu_threads)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(0, 4096)
        self.batch_size.setSpecialValueText("Automatic")
        self.batch_size.setValue(settings.local_inference_batch_size)
        self.batch_size.setToolTip(
            "Frames evaluated together. Automatic uses 1 on CPU and 8 on CUDA. "
            "Larger values can improve throughput on high-memory GPUs but use more "
            "GPU memory. Reduce this if GPU memory is exhausted."
        )
        layout.addRow("Inference batch size", self.batch_size)

        self.postprocess_workers = QSpinBox()
        self.postprocess_workers.setRange(0, min(16, cpu_threads))
        self.postprocess_workers.setSpecialValueText("Automatic")
        self.postprocess_workers.setValue(
            min(settings.local_postprocess_workers, min(16, cpu_threads))
        )
        self.postprocess_workers.setToolTip(
            "Parallel CPU workers for mask cleanup and ellipse fitting. More workers can be "
            "faster but increase memory use."
        )
        layout.addRow("Postprocess workers", self.postprocess_workers)

        self.overlays = QCheckBox("Create overlay videos")
        self.overlays.setChecked(settings.local_generate_overlays)
        self.overlays.setToolTip(
            "Write a video with the tracked center overlaid. Disabling this saves a second "
            "video-reading pass and disk space; force traces are still generated."
        )
        layout.addRow("Outputs", self.overlays)

        memory_note = QLabel(
            (
                "Batch size, postprocess workers, and overlay generation are sent to the running "
                "worker and take effect after its current inference batch. Processing device and "
                "CPU thread count apply to the next job."
            )
            if running
            else (
                "Frames are streamed from disk in bounded batches. StimTrace does not retain "
                "the full recording in RAM."
            )
        )
        memory_note.setProperty("role", "muted")
        memory_note.setWordWrap(True)
        layout.addRow(memory_note)

        buttons = QHBoxLayout()
        restore = QPushButton("Restore recommended")
        set_action_icon(restore, "clear")
        restore.clicked.connect(self.restore_recommended)
        save = QPushButton("Apply" if running else "Save")
        set_action_icon(save, "apply" if running else "save")
        save.setProperty("role", "primary")
        save.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        set_action_icon(cancel, "cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(restore)
        buttons.addStretch(1)
        buttons.addWidget(save)
        buttons.addWidget(cancel)
        layout.addRow(buttons)

    def restore_recommended(self) -> None:
        if not self.running:
            self.device.setCurrentIndex(0)
            self.cpu_threads.setValue(0)
        self.batch_size.setValue(0)
        self.postprocess_workers.setValue(0)
        self.overlays.setChecked(True)

    def accept(self) -> None:
        self.settings.local_device_preference = str(self.device.currentData())
        self.settings.local_cpu_threads = self.cpu_threads.value()
        self.settings.local_inference_batch_size = self.batch_size.value()
        self.settings.local_postprocess_workers = self.postprocess_workers.value()
        self.settings.local_generate_overlays = self.overlays.isChecked()
        self.settings.save()
        super().accept()


class ModelManagerPage(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)

        title_row = QHBoxLayout()
        title = QLabel("Segmentation Models")
        title.setProperty("role", "title")
        help_button = QPushButton("Help")
        set_action_icon(help_button, "help")
        help_button.clicked.connect(self.show_help)
        self.settings_button = QPushButton("Advanced settings")
        set_action_icon(self.settings_button, "settings")
        self.settings_button.clicked.connect(self.main_window.open_settings)
        title_row.addWidget(title, 1)
        title_row.addWidget(self.settings_button)
        title_row.addWidget(help_button)
        layout.addLayout(title_row)
        header = QHBoxLayout()
        description = QLabel(
            "Manage segmentation models and their independent calibration, tracking, "
            "and performance parameters."
        )
        description.setWordWrap(True)
        header.addWidget(description, 1)
        layout.addLayout(header)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Model", "Active", "Object type", "Source", "Created", "Dataset", "Parameters"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 6):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.doubleClicked.connect(self.activate)
        layout.addWidget(self.table, 1)

        primary_actions = QHBoxLayout()
        for label, callback, primary, action in [
            ("Activate", self.activate, True, "apply"),
            ("Edit parameters", self.edit_parameters, False, "settings"),
            ("Train new model", self.train_model, False, "run"),
            ("Import .pth", self.import_model, False, "open"),
        ]:
            button = QPushButton(label)
            set_action_icon(button, action)
            if primary:
                button.setProperty("role", "primary")
            button.clicked.connect(callback)
            primary_actions.addWidget(button)
        primary_actions.addStretch(1)
        layout.addLayout(primary_actions)

        secondary_actions = QHBoxLayout()
        for label, callback, action in [
            ("Export", self.export_model, "save"),
            ("Rename", self.rename_model, "settings"),
            ("Set object type", self.set_object_type, "settings"),
            ("Remove", self.remove_model, "delete"),
        ]:
            button = QPushButton(label)
            set_action_icon(button, action)
            if action == "delete":
                button.setProperty("role", "danger")
            button.clicked.connect(callback)
            secondary_actions.addWidget(button)
        secondary_actions.addStretch(1)
        layout.addLayout(secondary_actions)
        self.populate()

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "Segmentation model help",
            "Activate selects the model used for new segmentation jobs. Running jobs keep "
            "the model with which they were submitted.\n\n"
            "Edit parameters opens the model-specific calibration, mask, tracking, and "
            "performance settings.\n\n"
            "Train new model opens the annotation and Colab training workspace. Import adds "
            "an existing .pth checkpoint locally and also uploads it when Google Drive is "
            "connected. Export saves the selected checkpoint locally.\n\n"
            "Rename changes the display name. Set object type records what the model segments. "
            "Remove deletes the local catalog entry but does not delete the checkpoint from Drive. "
            "The bundled Default pillar model cannot be renamed or removed.\n\n"
            "After activating a different model, reopen Colab and run both cells so the worker "
            "loads that checkpoint.",
        )

    def selected_model_name(self) -> str | None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.text() if item else None

    def populate(self, select_name: str | None = None) -> None:
        settings = self.main_window.settings
        settings.model_profiles.setdefault(
            "Default pillar",
            settings.default_model_file_id or settings.model_file_id,
        )
        names = sorted(
            settings.model_profiles,
            key=lambda name: (name != settings.selected_model_name, name.lower()),
        )
        self.table.setRowCount(len(names))
        selected_row = 0
        for row, name in enumerate(names):
            metadata = settings.model_metadata.get(name, {})
            parameters = settings.model_parameter_profiles.get(name, settings.parameter_values())
            created = str(metadata.get("created_at", ""))
            if "T" in created:
                created = created.split("T", 1)[0]
            threshold = float(parameters.get("mask_threshold", settings.mask_threshold))
            pixel_size = float(parameters.get("pixel_to_um", settings.pixel_to_um))
            values = [
                name,
                "Yes" if name == settings.selected_model_name else "",
                metadata.get("object_type", "Other"),
                metadata.get("source", "Unknown"),
                created or "-",
                str(metadata.get("dataset_size", "") or "-"),
                f"threshold {threshold:.2f}, {pixel_size:.2f} um/px",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    drive_id = settings.model_profiles.get(name, "")
                    local_path = settings.local_model_paths.get(name, "")
                    item.setToolTip(
                        f"Drive file ID: {drive_id or 'not uploaded'}\n"
                        f"Local checkpoint: {local_path or 'not cached'}"
                    )
                self.table.setItem(row, column, item)
            if name == (select_name or settings.selected_model_name):
                selected_row = row
        if names:
            self.table.selectRow(selected_row)

    def activate(self, *_args) -> None:
        name = self.selected_model_name()
        if not name:
            return
        try:
            self.main_window.activate_model_profile(name)
            self.populate(name)
        except Exception as error:
            QMessageBox.critical(self, "Could not activate model", str(error))

    def edit_parameters(self) -> None:
        name = self.selected_model_name()
        if not name:
            return
        try:
            self.main_window.activate_model_profile(name, notify=False)
            dialog = SettingsDialog(
                self.main_window.settings,
                self,
                cloud_hardware=self.main_window.worker_hardware,
                calibration_videos=self.main_window.videos,
            )
            if dialog.exec() == QDialog.Accepted:
                self.main_window.model_status.setText(
                    f"Segmentation model: {self.main_window.settings.selected_model_name}"
                )
                if (
                    self.main_window.drive.service
                    and self.main_window.settings.model_file_id
                ):
                    self.main_window.drive.ensure_workspace(self.main_window.settings)
                self.populate(self.main_window.settings.selected_model_name)
        except Exception as error:
            QMessageBox.critical(self, "Could not edit parameters", str(error))

    def train_model(self) -> None:
        self.main_window.open_model_training()

    def import_model(self) -> None:
        imported_name = self.main_window.import_trained_model(open_colab=False, parent=self)
        self.populate(imported_name)

    def export_model(self) -> None:
        name = self.selected_model_name()
        if not name:
            return
        local_source_text = self.main_window.settings.local_model_paths.get(name, "")
        local_source = Path(local_source_text) if local_source_text else None
        if name == "Default pillar" and not (local_source and local_source.is_file()):
            bundled = self.main_window.bundled_model_path()
            local_source = bundled if bundled.is_file() else None
        file_id = self.main_window.settings.model_profiles.get(name, "")
        if not (local_source and local_source.is_file()) and not (
            self.main_window.drive.service and file_id
        ):
            QMessageBox.information(
                self,
                "Model unavailable",
                "This checkpoint is not available locally. Sign in to Google Drive to retrieve it.",
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export segmentation model",
            f"{name}.pth",
            "PyTorch model (*.pth)",
        )
        if not filename:
            return
        target = Path(filename)
        if target.suffix.lower() != ".pth":
            target = target.with_suffix(".pth")
        try:
            if local_source and local_source.is_file():
                shutil.copy2(local_source, target)
            else:
                target.write_bytes(self.main_window.drive.file_bytes(file_id))
            QMessageBox.information(self, "Model exported", f"Saved the model to:\n{target}")
        except Exception as error:
            QMessageBox.critical(self, "Could not export model", str(error))

    def rename_model(self) -> None:
        old_name = self.selected_model_name()
        if not old_name:
            return
        if old_name == "Default pillar":
            QMessageBox.information(self, "Rename unavailable", "The bundled model name is fixed.")
            return
        new_name, accepted = QInputDialog.getText(
            self,
            "Rename model",
            "Model name",
            text=old_name,
        )
        new_name = new_name.strip()
        settings = self.main_window.settings
        if not accepted or not new_name or new_name == old_name:
            return
        if new_name in settings.model_profiles:
            QMessageBox.warning(self, "Name already used", "Choose a unique model name.")
            return
        settings.model_profiles[new_name] = settings.model_profiles.pop(old_name)
        if old_name in settings.model_parameter_profiles:
            settings.model_parameter_profiles[new_name] = settings.model_parameter_profiles.pop(old_name)
        if old_name in settings.model_default_parameter_profiles:
            settings.model_default_parameter_profiles[new_name] = (
                settings.model_default_parameter_profiles.pop(old_name)
            )
        if old_name in settings.model_metadata:
            settings.model_metadata[new_name] = settings.model_metadata.pop(old_name)
        if old_name in settings.local_model_paths:
            settings.local_model_paths[new_name] = settings.local_model_paths.pop(old_name)
        if settings.selected_model_name == old_name:
            settings.selected_model_name = new_name
            self.main_window.model_status.setText(f"Segmentation model: {new_name}")
        settings.save()
        self.populate(new_name)

    def set_object_type(self) -> None:
        name = self.selected_model_name()
        if not name:
            return
        settings = self.main_window.settings
        current = settings.model_metadata.get(name, {}).get("object_type", "Other")
        options = ["Pillar", "Tissue", "Other"]
        value, accepted = QInputDialog.getItem(
            self,
            "Model object type",
            "Object type",
            options,
            options.index(current) if current in options else len(options) - 1,
            False,
        )
        if not accepted:
            return
        settings.model_metadata.setdefault(name, {})["object_type"] = value
        settings.save()
        self.populate(name)

    def remove_model(self) -> None:
        name = self.selected_model_name()
        if not name:
            return
        if name == "Default pillar":
            QMessageBox.information(self, "Remove unavailable", "The bundled model cannot be removed.")
            return
        if QMessageBox.question(
            self,
            "Remove model",
            f"Remove {name} from the local model catalog?\n\n"
            "The checkpoint will remain in Google Drive.",
        ) != QMessageBox.Yes:
            return
        settings = self.main_window.settings
        was_active = settings.selected_model_name == name
        settings.model_profiles.pop(name, None)
        settings.model_parameter_profiles.pop(name, None)
        settings.model_default_parameter_profiles.pop(name, None)
        settings.model_metadata.pop(name, None)
        settings.local_model_paths.pop(name, None)
        if was_active:
            self.main_window.activate_model_profile("Default pillar", notify=False)
        else:
            settings.save()
        self.populate()


class JobHistoryPage(QWidget):
    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__(main_window)
        self.main_window = main_window
        self.visible_jobs: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title_row = QHBoxLayout()
        title = QLabel("Job History")
        title.setProperty("role", "title")
        help_button = QPushButton("Help")
        set_action_icon(help_button, "help")
        help_button.clicked.connect(self.show_help)
        title_row.addWidget(title, 1)
        title_row.addWidget(help_button)
        layout.addLayout(title_row)
        header = QHBoxLayout()
        description = QLabel(
            "Review previous segmentation and training jobs, resume monitoring, retry failures, "
            "or download completed results."
        )
        description.setWordWrap(True)
        header.addWidget(description, 1)
        layout.addLayout(header)
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels(
            [
                "Type", "Compute", "Study", "Model", "Job ID", "State", "Submitted",
                "Total processing time", "Processed frames", "Videos", "Last message",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        for label, callback, action in [
            ("Refresh", self.refresh_remote, "refresh"),
            ("Resume monitoring", self.resume, "apply"),
            ("Retry failed job", self.retry, "refresh"),
            ("Cancel job", self.cancel, "cancel"),
            ("Download results", self.download_results, "save"),
            ("Open Drive folder", self.open_drive, "open_folder"),
        ]:
            button = QPushButton(label)
            set_action_icon(button, action)
            button.clicked.connect(callback)
            row.addWidget(button)
        self.open_signal_button = QPushButton("Open in Signal Viewer")
        set_action_icon(self.open_signal_button, "open")
        self.open_signal_button.setToolTip(
            "Open the selected completed study's combined traces in Signal Viewer."
        )
        self.open_signal_button.clicked.connect(self.open_signal_viewer)
        row.addWidget(self.open_signal_button)
        self.open_local_button = QPushButton("Open local folder")
        set_action_icon(self.open_local_button, "open_folder")
        self.open_local_button.clicked.connect(self.open_local)
        row.addWidget(self.open_local_button)
        row.addStretch(1)
        clear_button = QPushButton("Clear history")
        set_action_icon(clear_button, "delete")
        clear_button.setProperty("role", "danger")
        clear_button.clicked.connect(self.clear_history)
        row.addWidget(clear_button)
        clear_logs_button = QPushButton("Clear logs")
        set_action_icon(clear_logs_button, "delete")
        clear_logs_button.setProperty("role", "danger")
        clear_logs_button.clicked.connect(self.clear_logs)
        row.addWidget(clear_logs_button)
        layout.addLayout(row)
        self.table.itemSelectionChanged.connect(self.update_selected_actions)
        self.populate()

    def show_help(self):
        QMessageBox.information(
            self,
            "Job history help",
            "Select one job in the table, then choose an action.\n\n"
            "Refresh updates cloud states while retaining local history. Resume monitoring "
            "returns the selected job to the main progress display. Retry creates a new attempt "
            "for a failed or cancelled job. Cancel stops the active local worker or asks Colab "
            "to stop. Cloud results are downloaded beside the original videos; local results "
            "are already stored there. Total processing time is measured from when the worker "
            "starts the job (including cloud video download) until Colab finishes uploading "
            "the completed result package. Queueing and desktop upload/download time are excluded. "
            "Processed frames reports source recording frames analyzed; it excludes extra frames "
            "written while rendering overlays or benchmark videos. "
            "Clear history removes finished entries from this list without deleting recordings, "
            "downloaded results, or Google Drive folders. Active jobs are retained. "
            "Open study in Signal Viewer loads the combined traces for a completed segmentation "
            "study, downloading cloud results first when needed. "
            "Open Drive folder applies only to cloud jobs. Open local folder opens an existing "
            "downloaded result directory for either local or cloud jobs.",
        )

    def selected_job(self) -> dict | None:
        row = self.table.currentRow()
        return self.visible_jobs[row] if 0 <= row < len(self.visible_jobs) else None

    def selected_local_folder(self) -> Path | None:
        job = self.selected_job()
        if not job:
            return None
        saved = str(job.get("local_results_path", "")).strip()
        target = Path(saved) if saved else self.main_window.result_download_target(job)
        return target if target is not None and target.is_dir() else None

    def update_selected_actions(self) -> None:
        job = self.selected_job()
        self.open_local_button.setEnabled(self.selected_local_folder() is not None)
        self.open_signal_button.setEnabled(
            bool(
                job
                and job.get("type", "segmentation") == "segmentation"
                and job.get("state") == "complete"
            )
        )

    def populate(self):
        self.visible_jobs = list(reversed(self.main_window.jobs))
        self.table.setRowCount(len(self.visible_jobs))
        for row, job in enumerate(self.visible_jobs):
            values = [
                job.get("type", "segmentation"),
                "Local" if job.get("backend") == "local" else "Cloud",
                job.get("study", ""),
                job.get("model_name", "") or "Not recorded (legacy job)",
                job.get("job_id", ""),
                job.get("state", "unknown"),
                job.get("created_at", ""),
                self.main_window.format_elapsed(job.get("actual_total_seconds")),
                self.main_window.format_processed_frames(
                    job.get("processed_frames"), job.get("total_job_frames")
                ),
                ", ".join(job.get("videos", [])),
                job.get("message", ""),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        if self.visible_jobs:
            self.table.selectRow(0)
        self.update_selected_actions()

    def refresh_remote(self):
        try:
            self.main_window.refresh_job_history()
            self.populate()
        except Exception as error:
            QMessageBox.critical(self, "Could not refresh jobs", str(error))

    def clear_history(self) -> None:
        clearable = [
            job for job in self.main_window.jobs
            if job.get("state") in TERMINAL_JOB_STATES
        ]
        if not clearable:
            QMessageBox.information(
                self,
                "Nothing to clear",
                "There are no finished job-history entries to clear. Active jobs are retained.",
            )
            return
        if QMessageBox.question(
            self,
            "Clear job history",
            f"Remove {len(clearable)} finished job-history entr"
            f"{'y' if len(clearable) == 1 else 'ies'}?\n\n"
            "This does not delete recordings, downloaded results, or Google Drive folders. "
            "Active jobs remain available.",
        ) != QMessageBox.Yes:
            return
        removed = self.main_window.clear_finished_job_history()
        self.populate()
        QMessageBox.information(
            self,
            "Job history cleared",
            f"Removed {removed} finished job-history entr"
            f"{'y' if removed == 1 else 'ies'}.",
        )

    def clear_logs(self) -> None:
        if QMessageBox.question(
            self,
            "Clear diagnostic logs",
            "Clear the StimTrace application log and crash report?\n\n"
            "This does not affect jobs, settings, recordings, or results.",
        ) != QMessageBox.Yes:
            return
        try:
            self.main_window.clear_diagnostic_logs()
            QMessageBox.information(
                self,
                "Diagnostic logs cleared",
                "The application log and crash report were cleared.",
            )
        except OSError as error:
            QMessageBox.critical(self, "Could not clear logs", str(error))

    def resume(self):
        job = self.selected_job()
        if job:
            self.main_window.activate_job(job)
            self.main_window.show_page("segment")

    def retry(self):
        job = self.selected_job()
        if not job:
            return
        if job.get("state") not in {"failed", "cancelled"}:
            QMessageBox.information(self, "Retry unavailable", "Only failed or cancelled jobs can be retried.")
            return
        self.main_window.request_job_retry(job)
        self.populate()

    def cancel(self):
        job = self.selected_job()
        if not job:
            return
        if job.get("state") in TERMINAL_JOB_STATES:
            QMessageBox.information(self, "Already finished", "This job is no longer running.")
            return
        if QMessageBox.question(self, "Cancel job", f"Cancel {job.get('job_id', 'this job')}?") != QMessageBox.Yes:
            return
        try:
            self.main_window.cancel_job(job)
            self.populate()
        except Exception as error:
            QMessageBox.critical(self, "Cancellation failed", str(error))

    def open_drive(self):
        job = self.selected_job()
        if job and job.get("folder_id"):
            webbrowser.open(f"https://drive.google.com/drive/folders/{job['folder_id']}")
        elif job and job.get("backend") == "local":
            QMessageBox.information(
                self,
                "Local job",
                "This job ran on this computer and has no Google Drive folder.",
            )

    def open_local(self):
        target = self.selected_local_folder()
        if target is None:
            QMessageBox.information(
                self,
                "Local folder unavailable",
                "This job does not currently have an existing local results folder. "
                "Download the completed results first.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def open_signal_viewer(self) -> None:
        job = self.selected_job()
        if job:
            self.main_window.open_job_in_signal_viewer(job)

    def download_results(self):
        job = self.selected_job()
        if not job:
            return
        if job.get("backend") == "local":
            target_text = job.get("local_results_path", "")
            target = Path(target_text) if target_text else None
            if target and target.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
            else:
                QMessageBox.warning(
                    self,
                    "Results unavailable",
                    "The local results folder no longer exists at its recorded location.",
                )
            return
        if not self.main_window.drive.service:
            QMessageBox.information(
                self,
                "Sign in required",
                "Sign in to Google Drive on the main screen before downloading results.",
            )
            return
        if job.get("state") != "complete":
            QMessageBox.information(self, "Results unavailable", "This job has not completed.")
            return
        source_paths = [Path(path) for path in job.get("source_paths", []) if path]
        if not source_paths or any(not path.is_file() for path in source_paths):
            folder = QFileDialog.getExistingDirectory(
                self,
                "Choose the folder containing the original recordings",
            )
            if not folder:
                return
            job["source_paths"] = [
                str(Path(folder) / video_name)
                for video_name in job.get("videos", [])
            ]
            job.pop("local_results_path", None)
        self.main_window.start_result_download(job, force=True, notify=True)
        self.populate()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        app_icon = QIcon(str(APP_ICON_PATH))
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
            QApplication.instance().setWindowIcon(app_icon)
        QApplication.instance().setStyleSheet(APP_STYLESHEET)
        self.settings = Settings.load(); self.drive = DriveClient(); self.videos: list[Path] = []
        self.jobs = self._load_jobs()
        for saved_job in self.jobs:
            if (
                saved_job.get("backend") == "local"
                and saved_job.get("state") in ACTIVE_JOB_STATES
            ):
                saved_job["state"] = "failed"
                saved_job["message"] = (
                    "Local processing was interrupted when StimTrace closed. Retry the job."
                )
        self._save_jobs()
        self.signal_page = None
        self.training_page = None
        self.point_tracking_page = None
        self.model_page = None
        self.job_history_page = None
        self.loaded_module_keys: set[str] = set()
        self.module_load_errors: dict[str, str] = {}
        self.requested_page_key = "segment"
        self.pending_signal_path: Path | None = None
        self.pending_signal_paths: list[Path] = []
        self.pending_signal_handoff_paths: list[Path] = []
        self.active_threads: set[QThread] = set()
        # Retain recently completed QThreads beyond their finished signal. PySide can
        # otherwise destroy the Python wrapper while Qt is still unwinding the signal.
        self.retired_threads: list[QThread] = []
        self.active_progress_file_id = ""
        self.monitored_job_id = ""
        self.submission_in_progress = False
        self.pending_cloud_submission: tuple[list[str], str] | None = None
        self.colab_opened = False
        self.worker_online = False
        self.worker_outdated = False
        self.worker_last_heartbeat_at = 0.0
        self.colab_crash_prompted_job_ids: set[str] = set()
        self.worker_device = ""
        self.worker_hardware: dict[str, Any] = {}
        self.colab_connection_thread: BackgroundFunctionThread | None = None
        self.colab_connection_in_progress = False
        self.resume_job_poll_after_colab = False
        self.worker_poll_thread: BackgroundFunctionThread | None = None
        self.restore_session_thread: BackgroundFunctionThread | None = None
        self.sign_in_thread: BackgroundFunctionThread | None = None
        self.drive_discovery_thread: BackgroundFunctionThread | None = None
        self.last_drive_discovery_started_at = 0.0
        self.job_progress_thread: BackgroundFunctionThread | None = None
        self.job_retry_thread: BackgroundFunctionThread | None = None
        self.cloud_submission_thread: BackgroundFunctionThread | None = None
        self.cloud_submission_context: dict[str, Any] = {}
        self.result_download_thread: BackgroundFunctionThread | None = None
        self.result_download_context: dict[str, Any] = {}
        self.signal_load_thread: BackgroundFunctionThread | None = None
        self.signal_dependency_thread: BackgroundFunctionThread | None = None
        self.signal_dependencies_ready = False
        self.local_process: LocalProcessThread | None = None
        self.local_cancel_file: Path | None = None
        self.local_control_file: Path | None = None
        self.local_terminal_event = ""
        self.local_hardware: dict[str, Any] = {
            "cuda_available": False,
            "cuda_name": "",
            "cpu_threads": os.cpu_count() or 1,
        }
        self.hardware_probe: LocalProcessThread | None = None
        self.shutdown_started = False
        self.shutdown_started_at = 0.0
        self.shutdown_retry_timer = QTimer(self)
        self.shutdown_retry_timer.setSingleShot(True)
        self.shutdown_retry_timer.timeout.connect(self.retry_close_after_workers)
        self.kalman_benchmark_combinations: list[dict[str, Any]] = []
        self.setWindowTitle(APPLICATION_NAME); self.setMinimumSize(640, 480)
        self.navigation = QToolBar("Primary navigation", self)
        self.navigation.setObjectName("primaryNavigation")
        self.navigation.setMovable(False)
        self.navigation.setIconSize(QSize(18, 18))
        self.navigation.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.TopToolBarArea, self.navigation)
        self.nav_action_group = QActionGroup(self)
        self.nav_action_group.setExclusive(True)
        self.nav_actions: dict[str, QAction] = {}
        segment_nav = QAction(
            self.style().standardIcon(QStyle.SP_ComputerIcon),
            "Segment",
            self,
        )
        segment_nav.setCheckable(True)
        segment_nav.setChecked(True)
        segment_nav.triggered.connect(lambda _checked=False: self.show_page("segment"))
        self.navigation.addAction(segment_nav)
        self.nav_action_group.addAction(segment_nav)
        self.nav_actions["segment"] = segment_nav
        analyze_nav = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogContentsView),
            "Analyze signals",
            self,
        )
        analyze_nav.setCheckable(True)
        analyze_nav.triggered.connect(self.open_signal_analysis)
        self.navigation.addAction(analyze_nav)
        self.nav_action_group.addAction(analyze_nav)
        self.nav_actions["signals"] = analyze_nav
        models_nav = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            "Models",
            self,
        )
        models_nav.setCheckable(True)
        models_nav.triggered.connect(self.open_model_manager)
        self.navigation.addAction(models_nav)
        self.nav_action_group.addAction(models_nav)
        self.nav_actions["models"] = models_nav
        training_nav = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogNewFolder),
            "Train models",
            self,
        )
        training_nav.setCheckable(True)
        training_nav.triggered.connect(self.open_model_training)
        self.navigation.addAction(training_nav)
        self.nav_action_group.addAction(training_nav)
        self.nav_actions["training"] = training_nav
        point_tracking_nav = QAction(
            self.style().standardIcon(QStyle.SP_ArrowRight),
            "Track points",
            self,
        )
        point_tracking_nav.setCheckable(True)
        point_tracking_nav.triggered.connect(self.open_point_tracking)
        self.navigation.addAction(point_tracking_nav)
        self.nav_action_group.addAction(point_tracking_nav)
        self.nav_actions["point_tracking"] = point_tracking_nav
        jobs_nav = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogListView),
            "Jobs",
            self,
        )
        jobs_nav.setCheckable(True)
        jobs_nav.triggered.connect(self.open_job_history)
        self.navigation.addAction(jobs_nav)
        self.nav_action_group.addAction(jobs_nav)
        self.nav_actions["jobs"] = jobs_nav
        self.navigation.clear()
        for key in ("segment", "signals", "point_tracking", "training", "models", "jobs"):
            self.navigation.addAction(self.nav_actions[key])
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)
        # Let the scroll area's viewport determine the page width.  Forcing the
        # layout's calculated minimum made long status/path text expand the
        # entire main window beyond the current screen.
        layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        self.page = QScrollArea()
        self.page.setWidgetResizable(True)
        self.page.setFrameShape(QFrame.NoFrame)
        self.page.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.page.setWidget(root)
        shell_layout.addWidget(self.page, 1)
        self.page_stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {"segment": shell}
        self.page_stack.addWidget(shell)
        self.loading_page = QWidget()
        loading_layout = QVBoxLayout(self.loading_page)
        loading_layout.setContentsMargins(18, 14, 18, 14)
        loading_layout.addStretch(1)
        self.loading_title = QLabel("Loading")
        self.loading_title.setProperty("role", "title")
        self.loading_title.setAlignment(Qt.AlignCenter)
        self.loading_message = QLabel("Preparing the requested workspace...")
        self.loading_message.setProperty("role", "muted")
        self.loading_message.setAlignment(Qt.AlignCenter)
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 0)
        self.loading_progress.setMaximumWidth(480)
        loading_layout.addWidget(self.loading_title)
        loading_layout.addWidget(self.loading_message)
        loading_layout.addWidget(self.loading_progress, 0, Qt.AlignHCenter)
        loading_layout.addStretch(1)
        self.page_stack.addWidget(self.loading_page)
        self.setCentralWidget(self.page_stack)
        title_row = QHBoxLayout()
        title = QLabel("StimTrace")
        title.setProperty("role", "title")
        self.status = QLabel("Not signed in")
        self.status.setProperty("role", "muted")
        self.status.setProperty("authState", "signedOut")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(0)
        self.status.setMaximumHeight(54)
        self.status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.sign_button = QPushButton("Sign in")
        self.sign_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        set_action_icon(self.sign_button, "apply")
        self.sign_button.clicked.connect(self.sign_in)
        self.sign_out_button = QPushButton("Sign out")
        self.sign_out_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        set_action_icon(self.sign_out_button, "cancel")
        self.sign_out_button.clicked.connect(self.sign_out)
        self.sign_out_button.setEnabled(False)
        help_button = QPushButton("Help")
        help_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        set_action_icon(help_button, "help")
        help_button.clicked.connect(self.show_help)
        title_row.addWidget(title)
        title_row.addWidget(self.status, 1)
        # Preserve right-aligned compact actions when the setup/status widgets
        # are temporarily hidden during an active job.
        title_row.addStretch(1)
        title_row.addWidget(self.sign_button)
        title_row.addWidget(self.sign_out_button)
        title_row.addWidget(help_button)
        layout.addLayout(title_row)
        self.purpose_label = QLabel(
            "Analyze engineered-tissue contraction videos with Google Colab compute and download overlays, "
            "force traces, and a combined CSV."
        )
        self.purpose_label.setProperty("role", "subtitle")
        self.purpose_label.setWordWrap(True)
        layout.addWidget(self.purpose_label)

        compute_mode_row = QHBoxLayout()
        compute_mode_label = QLabel("Compute")
        compute_mode_label.setStyleSheet("font-weight: 600;")
        self.compute_mode_group = QButtonGroup(self)
        self.compute_mode_group.setExclusive(True)
        self.cloud_mode_button = QPushButton("Cloud")
        self.cloud_mode_button.setCheckable(True)
        self.cloud_mode_button.setProperty("role", "mode")
        self.local_mode_button = QPushButton("This computer")
        self.local_mode_button.setCheckable(True)
        self.local_mode_button.setProperty("role", "mode")
        self.compute_mode_group.addButton(self.cloud_mode_button)
        self.compute_mode_group.addButton(self.local_mode_button)
        self.cloud_mode_button.clicked.connect(
            lambda _checked=False: self.set_compute_mode("cloud")
        )
        self.local_mode_button.clicked.connect(
            lambda _checked=False: self.set_compute_mode("local")
        )
        self.settings_button = QPushButton("Advanced settings")
        set_action_icon(self.settings_button, "settings")
        self.settings_button.setToolTip(
            "Configure the segmentation model, pixel calibration, force conversion, "
            "tracking, and performance settings."
        )
        self.settings_button.clicked.connect(self.open_settings)
        compute_mode_row.addWidget(compute_mode_label)
        compute_mode_row.addWidget(self.cloud_mode_button)
        compute_mode_row.addWidget(self.local_mode_button)
        compute_mode_row.addStretch(1)
        compute_mode_row.addWidget(self.settings_button)
        layout.addLayout(compute_mode_row)

        step_row = QHBoxLayout()
        step_row.setSpacing(8)
        self.workflow_steps = [
            WorkflowStep(1, "Google account", "Sign in"),
            WorkflowStep(2, "Colab compute", "Open and run"),
            WorkflowStep(3, "Input videos", "Add recordings"),
            WorkflowStep(4, "Analysis", "Submit job"),
        ]
        self.workflow_steps[0].clicked.connect(self.activate_google_account_step)
        self.workflow_steps[1].clicked.connect(self.activate_colab_compute_step)
        self.workflow_steps[2].clicked.connect(self.activate_input_videos_step)
        self.workflow_steps[3].clicked.connect(self.activate_analysis_step)
        for step in self.workflow_steps:
            step_row.addWidget(step, 1)
        layout.addLayout(step_row)

        self.next_step = QLabel("Step 1: Sign in to Google Drive to prepare your personal StimTrace workspace.")
        self.next_step.setProperty("role", "instruction")
        self.next_step.setWordWrap(True)
        self.next_step.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.next_step.setMaximumHeight(72)
        layout.addWidget(self.next_step)

        self.remember_sign_in = QCheckBox("Stay signed in on this computer")
        self.remember_sign_in.setChecked(self.settings.remember_google_sign_in)
        self.remember_sign_in.toggled.connect(self.set_remember_sign_in)
        self.model_status = QLabel(f"Segmentation model: {self.settings.selected_model_name}")
        self.model_status.setProperty("role", "muted")
        self.model_status.setWordWrap(True)
        self.model_status.setMinimumWidth(0)
        self.model_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.model_status)

        action_row = QHBoxLayout()
        self.compute_button = QPushButton("Open Colab")
        self.compute_button.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.compute_button.setToolTip(
            "Open the StimTrace Compute notebook. When its worker is already online, reconnect to "
            "that runtime without running the notebook cells again."
        )
        self.compute_button.clicked.connect(self.open_colab)
        self.compute_button.setEnabled(False)
        # Colab is a compute-mode action, so keep it beside the Cloud selector.
        compute_mode_row.insertWidget(3, self.compute_button)
        self.colab_wait_progress = QProgressBar()
        self.colab_wait_progress.setRange(0, 0)
        self.colab_wait_progress.setTextVisible(False)
        self.colab_wait_progress.setFixedWidth(120)
        self.colab_wait_progress.setFixedHeight(10)
        self.colab_wait_progress.setVisible(False)
        self.colab_wait_label = QLabel("Preparing Colab...")
        self.colab_wait_label.setProperty("role", "muted")
        self.colab_wait_label.setVisible(False)
        action_row.addWidget(self.colab_wait_progress)
        action_row.addWidget(self.colab_wait_label)
        action_row.addStretch(1)
        action_row.addWidget(self.remember_sign_in)
        layout.addLayout(action_row)

        self.study_box = QGroupBox("New segmentation job")
        self.study_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        job_layout = QVBoxLayout(self.study_box)
        form = QGridLayout()
        self.study = QLineEdit()
        self.study.setPlaceholderText("e.g. Isoproterenol_High")
        form.addWidget(QLabel("Study name"), 0, 0)
        form.addWidget(self.study, 0, 1)
        job_layout.addLayout(form)
        file_header = QHBoxLayout()
        self.video_count_label = QLabel("No videos selected")
        self.video_count_label.setProperty("role", "muted")
        self.add_button = QPushButton("Add videos")
        set_action_icon(self.add_button, "open")
        self.add_button.clicked.connect(self.add_videos)
        self.add_button.setEnabled(False)
        self.add_folder_button = QPushButton("Add video folder")
        set_action_icon(self.add_folder_button, "open_folder")
        self.add_folder_button.setToolTip(
            "Add supported video files directly inside a selected folder."
        )
        self.add_folder_button.clicked.connect(self.add_video_folder)
        self.add_folder_button.setEnabled(False)
        self.clear_button = QPushButton("Clear all videos")
        set_action_icon(self.clear_button, "delete")
        self.clear_button.setProperty("role", "danger")
        self.clear_button.clicked.connect(self.clear_videos)
        file_header.addWidget(self.video_count_label, 1)
        file_header.addWidget(self.add_button)
        file_header.addWidget(self.add_folder_button)
        file_header.addWidget(self.clear_button)
        job_layout.addLayout(file_header)
        self.video_table = QTableWidget(0, 6)
        self.video_table.setHorizontalHeaderLabels(
            ["Recording", "Duration", "Frames", "Size", "Status", ""]
        )
        self.video_table.setAlternatingRowColors(True)
        self.video_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.video_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3, 4):
            self.video_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.video_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Fixed)
        self.video_table.setColumnWidth(5, 38)
        job_layout.addWidget(self.video_table, 1)
        self.update_video_table_height()
        layout.addWidget(self.study_box)

        self.progress_box = QGroupBox("Submitted job progress")
        self.progress_box.setMinimumHeight(162)
        self.progress_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        progress_layout = QVBoxLayout(self.progress_box)
        summary_row = QHBoxLayout()
        self.active_job_summary = QLabel("No active job")
        self.active_job_summary.setProperty("role", "muted")
        self.active_job_summary.setWordWrap(True)
        self.setup_toggle_button = QPushButton("Show setup")
        set_action_icon(self.setup_toggle_button, "settings")
        self.setup_toggle_button.clicked.connect(self.toggle_setup_details)
        # Keep the active-job layout control with the cloud compute actions.
        compute_mode_row.insertWidget(4, self.setup_toggle_button)
        summary_row.addWidget(self.active_job_summary, 1)
        progress_layout.addLayout(summary_row)
        self.monitoring_label = QLabel("No job is currently being monitored.")
        self.monitoring_label.setMinimumHeight(20)
        self.monitoring_label.setVisible(False)
        progress_layout.addWidget(self.monitoring_label)
        self.file_progress_label = QLabel("Current file: waiting for frame information")
        self.file_progress_label.setMinimumHeight(20)
        self.file_progress_label.setVisible(False)
        progress_layout.addWidget(self.file_progress_label)
        self.file_progress = QProgressBar(); self.file_progress.setVisible(False)
        self.file_progress.setRange(0, 100); self.file_progress.setValue(0); self.file_progress.setFormat("%p%")
        progress_layout.addWidget(self.file_progress)
        self.job_progress_label = QLabel("Total job: waiting for Colab")
        self.job_progress_label.setMinimumHeight(20)
        self.job_progress_label.setVisible(False)
        progress_layout.addWidget(self.job_progress_label)
        self.job_progress_bar = QProgressBar(); self.job_progress_bar.setVisible(False)
        self.job_progress_bar.setRange(0, 1000); self.job_progress_bar.setValue(0); self.job_progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.job_progress_bar)
        self.upload_progress_label = QLabel("Uploading videos to Google Drive...")
        self.upload_progress_label.setProperty("role", "uploadNotice")
        self.upload_progress_label.setProperty("blinkPhase", "bright")
        self.upload_progress_label.setWordWrap(True)
        self.upload_progress_label.setVisible(False)
        progress_layout.addWidget(self.upload_progress_label)
        self.upload_progress = QProgressBar()
        self.upload_progress.setProperty("uploadState", "active")
        self.upload_progress.setVisible(False)
        progress_layout.addWidget(self.upload_progress)
        job_actions = QHBoxLayout()
        self.cancel_current_button = QPushButton("Cancel")
        set_action_icon(self.cancel_current_button, "cancel")
        self.cancel_current_button.setProperty("role", "danger")
        self.cancel_current_button.clicked.connect(self.cancel_current_job)
        self.retry_current_button = QPushButton("Retry")
        set_action_icon(self.retry_current_button, "refresh")
        self.retry_current_button.clicked.connect(self.retry_current_job)
        self.retry_current_button.setVisible(False)
        self.open_current_drive_button = QPushButton("Open Drive")
        set_action_icon(self.open_current_drive_button, "open_folder")
        self.open_current_drive_button.clicked.connect(self.open_current_job_drive)
        self.open_results_button = QPushButton("Open results")
        set_action_icon(self.open_results_button, "open_folder")
        self.open_results_button.clicked.connect(self.open_current_results)
        self.analyze_results_button = QPushButton("Analyze signals")
        set_action_icon(self.analyze_results_button, "open")
        self.analyze_results_button.clicked.connect(self.analyze_current_results)
        job_actions.addWidget(self.cancel_current_button)
        job_actions.addWidget(self.retry_current_button)
        job_actions.addStretch(1)
        job_actions.addWidget(self.open_current_drive_button)
        job_actions.addWidget(self.open_results_button)
        job_actions.addWidget(self.analyze_results_button)
        progress_layout.addLayout(job_actions)
        self.progress_box.setVisible(False)
        layout.addWidget(self.progress_box)
        layout.addStretch(1)

        footer_bar = QFrame()
        footer_bar.setObjectName("stickyFooter")
        footer = QHBoxLayout(footer_bar)
        footer.setContentsMargins(18, 9, 18, 9)
        self.footer_logo = QLabel()
        self.footer_logo.setAccessibleName("StimTrace by Stimulatrix")
        self.footer_logo.setToolTip("StimTrace by Stimulatrix")
        self.footer_logo.setFixedSize(46, 32)
        self.footer_logo.setAlignment(Qt.AlignCenter)
        logo = QPixmap(str(APP_LOGO_PATH))
        if not logo.isNull():
            self.footer_logo.setPixmap(
                logo.scaled(QSize(42, 30), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self.local_settings_button = QPushButton("Local compute settings")
        set_action_icon(self.local_settings_button, "settings")
        self.local_settings_button.clicked.connect(self.open_local_settings)
        self.submit_button = QPushButton("Submit analysis")
        set_action_icon(self.submit_button, "run")
        self.submit_button.setProperty("role", "primary")
        self.submit_button.clicked.connect(self.submit)
        self.submit_button.setEnabled(False)
        footer.addWidget(self.footer_logo)
        footer.addWidget(self.local_settings_button)
        footer.addStretch(1)
        footer.addWidget(self.submit_button)
        shell_layout.addWidget(footer_bar)

        self.setup_widgets = [
            self.purpose_label,
            *self.workflow_steps,
            self.status,
            self.remember_sign_in,
            self.sign_out_button,
            self.model_status,
            self.sign_button,
            self.compute_button,
            self.study_box,
        ]
        self.setup_expanded_during_job = False
        self.set_compute_mode(self.settings.compute_mode, save=False)
        self.update_workflow_steps()
        self.poll_timer = QTimer(self); self.poll_timer.setInterval(3000); self.poll_timer.timeout.connect(self.poll_job_progress)
        self.worker_timer = QTimer(self)
        self.worker_timer.setInterval(5000)
        self.worker_timer.timeout.connect(self.poll_worker_status)
        self.upload_attention_timer = QTimer(self)
        self.upload_attention_timer.setInterval(650)
        self.upload_attention_timer.timeout.connect(self.toggle_upload_attention)
        self.upload_blink_phase = False
        remembered_job = self._best_active_job()
        if (
            self.settings.compute_mode == "cloud"
            and remembered_job
            and remembered_job.get("type", "segmentation") == "segmentation"
        ):
            self.monitored_job_id = remembered_job.get("job_id", "")
            # Startup always begins with an empty new-job recording list. The
            # previous cloud job remains visible in the progress panel and can be
            # restored explicitly from Job History when its inputs are needed.
            self.monitoring_label.setText(
                f"Restoring active job: {remembered_job.get('job_id', '')} - "
                f"{remembered_job.get('state', 'queued')}"
            )
            self.monitoring_label.setVisible(True)
            self.progress_box.setVisible(True)
            self.set_active_job_focus(remembered_job, remembered_job.get("state", "queued"))
            QTimer.singleShot(0, lambda: self.page.ensureWidgetVisible(self.progress_box, 0, 16))
        self.restore_session_timer = QTimer(self)
        self.restore_session_timer.setSingleShot(True)
        self.restore_session_timer.timeout.connect(self.restore_google_session)
        self.hardware_probe_start_timer = QTimer(self)
        self.hardware_probe_start_timer.setSingleShot(True)
        self.hardware_probe_start_timer.timeout.connect(self.start_local_hardware_probe)
        if self.settings.remember_google_sign_in and (USER_DATA_DIR / "token.json").exists():
            # Let the maximized window paint before performing saved-session network calls.
            self.sign_button.setEnabled(False)
            self.sign_button.setText("Restoring...")
            self.restore_session_timer.start(350)
        self.hardware_probe_start_timer.start(0)

    def register_page(self, key: str, page: QWidget) -> None:
        self.pages[key] = page
        self.page_stack.addWidget(page)

    def register_thread(self, thread: QThread, name: str) -> QThread:
        """Keep a worker alive until Qt confirms native thread cleanup is complete."""
        thread.setObjectName(name)
        self.active_threads.add(thread)
        thread.finished.connect(
            self.release_finished_thread,
            Qt.QueuedConnection,
        )
        return thread

    @Slot()
    def release_finished_thread(self) -> None:
        thread = self.sender()
        if not isinstance(thread, QThread):
            return
        # This slot is explicitly queued onto the GUI thread. Keep a bounded set of
        # completed wrappers so none is destroyed while Qt is still dispatching its
        # completion signals.
        thread.wait()
        self.active_threads.discard(thread)
        if thread not in self.retired_threads:
            self.retired_threads.append(thread)
        while len(self.retired_threads) > 64:
            retired = self.retired_threads.pop(0)
            if not retired.isRunning():
                retired.deleteLater()

    def configure_kalman_benchmark(self) -> None:
        dialog = KalmanBenchmarkDialog(
            self.kalman_benchmark_combinations or DEFAULT_KALMAN_BENCHMARK,
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.set_kalman_benchmark_combinations(dialog.combinations)

    def set_kalman_benchmark_combinations(
        self, combinations: list[dict[str, Any]]
    ) -> None:
        self.kalman_benchmark_combinations = list(combinations)

    def effective_job_parameters(self) -> dict[str, Any]:
        parameters = self.settings.parameter_values()
        if self.kalman_benchmark_combinations:
            parameters["kalman_benchmark"] = [
                dict(combination)
                for combination in self.kalman_benchmark_combinations
            ]
        parameters["generate_overlays"] = self.settings.cloud_overlay_mode == "cloud"
        return parameters

    def show_page(self, key: str) -> None:
        self.requested_page_key = key
        page = self.pages.get(key)
        if page is None:
            return
        self.page_stack.setCurrentWidget(page)
        action = self.nav_actions.get(key)
        if action:
            action.setChecked(True)
        titles = {
            "segment": "StimTrace",
            "signals": "StimTrace - Signal Analysis",
            "models": "StimTrace - Models",
            "training": "StimTrace - Model Training",
            "point_tracking": "StimTrace - Point Tracking",
            "jobs": "StimTrace - Jobs",
        }
        self.setWindowTitle(titles.get(key, "StimTrace"))

    def show_module_loading(self, key: str, title: str) -> None:
        self.requested_page_key = key
        action = self.nav_actions.get(key)
        if action:
            action.setChecked(True)
        self.loading_title.setText(title)
        self.loading_message.setText(
            "Preparing scientific libraries."
        )
        self.page_stack.setCurrentWidget(self.loading_page)
        self.setWindowTitle(f"StimTrace - {title}")

    def load_scientific_module(self, key: str) -> None:
        """Import Qt-backed scientific pages on the GUI thread, as Qt requires."""
        if self.shutdown_started or key in self.loaded_module_keys:
            return
        if key == "signals" and not self.signal_dependencies_ready:
            if self.signal_dependency_thread is not None:
                return

            def preload_signal_dependencies() -> bool:
                # These packages do not create Qt objects and are safe to import away
                # from the GUI thread. The Qt-backed Matplotlib module remains below.
                for module_name in (
                    "numpy", "pandas", "scipy.sparse", "scipy.sparse.linalg", "scipy.signal"
                ):
                    importlib.import_module(module_name)
                return True

            thread = BackgroundFunctionThread(preload_signal_dependencies, self)
            self.register_thread(thread, "signal-numeric-dependency-loader")
            thread.succeeded.connect(self.finish_signal_dependency_load, Qt.QueuedConnection)
            thread.failed.connect(
                lambda message: self.module_preload_failed("signals", message),
                Qt.QueuedConnection,
            )
            thread.finished.connect(self.finish_signal_dependency_thread, Qt.QueuedConnection)
            self.signal_dependency_thread = thread
            thread.start()
            return
        module_names = {
            "signals": "signal_analysis",
            "training": "model_training",
            "point_tracking": "point_tracking",
        }
        module_name = module_names.get(key)
        if module_name is None:
            return
        try:
            importlib.import_module(module_name)
        except Exception as error:
            self.module_preload_failed(key, str(error))
            return
        self.loaded_module_keys.add(key)
        if self.requested_page_key == "signals" and key == "signals":
            self._show_signal_analysis()
        elif self.requested_page_key == "training" and key == "training":
            self.open_model_training()
        elif self.requested_page_key == "point_tracking" and key == "point_tracking":
            self.open_point_tracking()

    @Slot(object)
    def finish_signal_dependency_load(self, _result: object) -> None:
        self.signal_dependencies_ready = True
        QTimer.singleShot(0, lambda: self.load_scientific_module("signals"))

    @Slot()
    def finish_signal_dependency_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.signal_dependency_thread is thread:
            self.signal_dependency_thread = None

    def module_preload_failed(self, key: str, error: str) -> None:
        self.module_load_errors[key] = error
        if self.requested_page_key != key:
            return
        titles = {
            "signals": "Signal analysis unavailable",
            "training": "Model training unavailable",
            "point_tracking": "Point tracking unavailable",
        }
        title = titles.get(key, "Scientific workspace unavailable")
        QMessageBox.critical(
            self,
            title,
            f"{error}\n\nInstall the desktop requirements and restart StimTrace.",
        )
        self.show_page("segment")

    def start_local_hardware_probe(self) -> None:
        if self.shutdown_started:
            return
        if self.hardware_probe is not None:
            return
        program, arguments = self.local_worker_command(["--hardware-json"])
        process = LocalProcessThread(program, arguments, self)
        self.register_thread(process, "local-hardware-probe")
        process.finished.connect(
            self.finish_local_hardware_probe,
            Qt.QueuedConnection,
        )
        self.hardware_probe = process
        process.start()

    @staticmethod
    def local_worker_command(arguments: list[str]) -> tuple[str, list[str]]:
        if getattr(sys, "frozen", False):
            return sys.executable, ["--local-worker", *arguments]
        return sys.executable, [str(SOURCE_DIR / "local_runner.py"), *arguments]

    @Slot()
    def finish_local_hardware_probe(self) -> None:
        process = self.sender()
        if not isinstance(process, LocalProcessThread):
            return
        process.wait()
        try:
            candidate = json.loads(process.output_tail.splitlines()[-1])
            if isinstance(candidate, dict):
                self.local_hardware.update(candidate)
        except (IndexError, json.JSONDecodeError):
            LOGGER.warning("Local hardware probe returned invalid output: %s", process.output_tail)
        if self.hardware_probe is process:
            self.hardware_probe = None
        if self.settings.compute_mode == "local" and self.local_process is None:
            self.status.setText(self.local_device_description())
            self.update_workflow_steps()

    def local_device_description(self) -> str:
        preference = self.settings.local_device_preference
        cuda_available = bool(self.local_hardware.get("cuda_available"))
        if preference == "cuda" and cuda_available:
            return f"Local compute ready: {self.local_hardware.get('cuda_name', 'NVIDIA CUDA GPU')}."
        if preference == "cuda":
            return (
                "NVIDIA CUDA is selected but is not available in this Python environment. "
                "Choose Automatic or CPU in Local compute settings."
            )
        if preference == "auto" and cuda_available:
            return (
                f"Local compute ready: {self.local_hardware.get('cuda_name', 'NVIDIA CUDA GPU')} "
                "(automatic selection)."
            )
        threads = self.settings.local_cpu_threads or self.local_hardware.get("cpu_threads", 1)
        return f"Local compute ready: CPU ({threads} logical processors available)."

    def cloud_runtime_description(self, compact: bool = False) -> str:
        hardware = self.worker_hardware
        if not hardware:
            return self.worker_device or "Colab runtime"
        parts = [str(hardware.get("device_name", hardware.get("device", "Colab runtime")))]
        utilization = hardware.get("gpu_utilization_pct")
        if utilization is not None:
            parts.append(f"{utilization:g}% GPU")
        if compact:
            return " | ".join(parts)
        gpu_memory = hardware.get("gpu_memory_gb", 0)
        used_memory = hardware.get("gpu_memory_used_gb")
        if gpu_memory:
            memory_text = f"{gpu_memory:g} GB GPU memory"
            if used_memory is not None:
                memory_text = f"{used_memory:g}/{gpu_memory:g} GB GPU memory"
            parts.append(memory_text)
        if hardware.get("system_memory_gb"):
            parts.append(f"{hardware['system_memory_gb']:g} GB system RAM")
        if hardware.get("cpu_threads"):
            parts.append(f"{hardware['cpu_threads']} logical CPU threads")
        if hardware.get("gpu_power_watts") is not None:
            parts.append(f"{hardware['gpu_power_watts']:g} W GPU power")
        return " | ".join(parts)

    def colab_worker_state_text(self) -> str:
        if self.worker_outdated:
            return "outdated worker detected - restart required"
        if self.worker_online and self.worker_device:
            return f"{self.worker_device} online"
        if self.worker_online:
            return "online"
        return "heartbeat not detected"

    def set_compute_mode(self, mode: str, save: bool = True) -> None:
        mode = "local" if mode == "local" else "cloud"
        if self.local_process is not None and self.local_process.isRunning():
            mode = "local"
        self.settings.compute_mode = mode
        if save:
            self.settings.save()
        self.cloud_mode_button.setChecked(mode == "cloud")
        self.local_mode_button.setChecked(mode == "local")
        cloud = mode == "cloud"
        for widget in (
            self.remember_sign_in,
            self.sign_out_button,
            self.sign_button,
            self.compute_button,
        ):
            widget.setVisible(cloud)
        if not cloud:
            self.set_colab_waiting(False)
        elif self.colab_opened and not self.worker_online and not self.worker_outdated:
            self.set_colab_waiting(
                True,
                "Waiting for Colab worker. Run both notebook cells.",
            )
        self.local_settings_button.setVisible(not cloud)
        self.purpose_label.setText(
            (
                "Analyze engineered-tissue contraction videos with Google Colab compute and "
                "download overlays, force traces, and a combined CSV."
            )
            if cloud
            else (
                "Analyze engineered-tissue contraction videos on this computer and save overlays, "
                "force traces, and a combined CSV beside the recordings."
            )
        )
        if cloud:
            self.set_auth_status(
                (
                    f"Signed in as {self.drive.account_email}. Your personal StimTrace Drive "
                    "workspace is ready."
                    if self.drive.service
                    else "Not signed in"
                ),
                signed_in=bool(self.drive.service),
            )
        elif self.local_process is None:
            self.set_auth_status(self.local_device_description(), signed_in=True)
        ready = bool(self.videos) and (
            not cloud or (bool(self.drive.service) and not self.worker_outdated)
        )
        videos_enabled = not cloud or bool(self.drive.service)
        self.add_button.setEnabled(videos_enabled)
        self.add_folder_button.setEnabled(videos_enabled)
        self.submit_button.setEnabled(ready)
        self.submit_button.setText("Submit analysis" if cloud else "Run locally")
        self.open_current_drive_button.setVisible(cloud)
        self.update_workflow_steps()

    def open_local_settings(self) -> None:
        running = bool(self.local_process and self.local_process.isRunning())
        dialog = LocalComputeSettingsDialog(
            self.settings,
            self.local_hardware,
            running=running,
            parent=self,
        )
        if dialog.exec() == QDialog.Accepted:
            if running:
                self.write_local_runtime_control()
                self.monitoring_label.setText(
                    "Local settings update sent; it will apply after the current inference batch."
                )
            else:
                self.status.setText(self.local_device_description())
                self.update_workflow_steps()

    def write_local_runtime_control(self) -> None:
        if not self.local_control_file:
            return
        values = {
            "cpu_threads": self.settings.local_cpu_threads,
            "inference_batch_size": self.settings.local_inference_batch_size,
            "postprocess_workers": self.settings.local_postprocess_workers,
            "generate_overlays": self.settings.local_generate_overlays,
        }
        temporary = self.local_control_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(values), encoding="utf-8")
        temporary.replace(self.local_control_file)

    def show_help(self):
        QMessageBox.information(
            self,
            "StimTrace help",
            "Choose Cloud to process with Google Colab, or This computer to process locally. "
            "Local mode works without Google sign-in and automatically uses a compatible NVIDIA "
            "CUDA GPU when available, with CPU fallback.\n\n"
            "Local workflow:\n\n"
            "Choose This computer, optionally review Local compute settings, add recordings and "
            "a study name, then choose Run locally. StimTrace remains responsive while the "
            "separate worker processes each frame. Results are saved under results/<job ID> "
            "beside the recordings.\n\n"
            "Cloud workflow:\n\n"
            "1. Sign in with the Google account that will run Colab. StimTrace creates a private "
            "workspace in that account's My Drive.\n\n"
            "2. Open Colab, select a GPU runtime, and run both notebook cells. Continue when "
            "Colab says that the StimTrace worker is running and watching the queue.\n\n"
            "3. Add one or more videos and enter a study name.\n\n"
            "4. Submit the job. Keep the Colab tab and worker cell running. StimTrace uploads the "
            "videos and displays per-file and total progress.\n\n"
            "5. When processing completes, results are downloaded to a Results folder beside "
            "the original recordings. Use Jobs > Job history to resume, retry, or download again.\n\n"
            "Download results in Job History performs an explicit download and reports the destination. "
            "For a job discovered from Drive, StimTrace first asks for the original recording folder.\n\n"
            "Stay signed in keeps a limited Google refresh token on this computer. Use Sign out "
            "to remove it or when changing accounts.\n\n"
            "Advanced settings contains model-specific calibration, tracking, and segmentation "
            "values. Use Measure from video beside Pixel calibration to draw an ellipse around "
            "a known object and calculate um/px from its width, height, mean diameter, or perimeter. "
            "The selected pixel measure and entered physical distance must represent the same dimension. "
            "Active force projects the tracked center onto the pillar bending axis and "
            "uses a real diastolic frame as zero; the calibration intercept is therefore not "
            "applied. StimTrace does not report absolute passive or preload force because no "
            "unloaded optical reference is available. Automatic axis estimation can be replaced "
            "with a known fixed image angle. Legacy area normalization "
            "is disabled unless explicitly enabled.\n\n"
            "Local compute settings controls the device and local performance limits.",
        )

    def set_remember_sign_in(self, enabled: bool) -> None:
        self.settings.remember_google_sign_in = enabled
        self.settings.save()

    def activate_google_account_step(self) -> None:
        if self.settings.compute_mode != "cloud":
            return
        if self.drive.service:
            account = self.settings.google_account_email or self.drive.account_email
            self.set_auth_status(
                f"Signed in as {account}" if account else "Google Drive is connected",
                signed_in=True,
            )
            return
        self.sign_in()

    def activate_colab_compute_step(self) -> None:
        if self.settings.compute_mode != "cloud":
            return
        if not self.drive.service:
            self.next_step.setText("Sign in to Google Drive before opening the Colab notebook.")
            self.activate_google_account_step()
            return
        self.open_colab()

    def activate_input_videos_step(self) -> None:
        self.add_videos()

    def activate_analysis_step(self) -> None:
        self.open_signal_analysis()

    def update_workflow_steps(self, job_state: str = "") -> None:
        if self.settings.compute_mode == "local":
            self.workflow_steps[0].set_clickable(False)
            self.workflow_steps[1].set_clickable(False)
            self.workflow_steps[2].set_clickable(True, "Select recordings to analyze")
            self.workflow_steps[3].set_clickable(True, "Open the Signal Analyzer")
            has_videos = bool(self.videos)
            running = bool(
                self.local_process is not None
                and self.local_process.isRunning()
            )
            device = (
                self.local_hardware.get("cuda_name", "CUDA GPU")
                if (
                    self.local_hardware.get("cuda_available")
                    and self.settings.local_device_preference != "cpu"
                )
                else "CPU"
            )
            titles = (
                ("This computer", device),
                ("Local processing", "Running" if running else "Ready"),
                ("Input videos", f"{len(self.videos)} selected" if has_videos else "Add recordings"),
                ("Analysis", job_state.replace("_", " ").title() if job_state else "Run locally"),
            )
            for step, (title, detail) in zip(self.workflow_steps, titles):
                step.title.setText(title)
                step.set_state("active" if (running and step is self.workflow_steps[1]) else "complete", detail)
            if not has_videos:
                self.workflow_steps[2].set_state("active", "Add recordings")
                self.workflow_steps[3].set_state("pending", "Run locally")
            elif not running and not job_state:
                self.workflow_steps[3].set_state("active", "Ready to run")
            if job_state in TERMINAL_JOB_STATES:
                self.workflow_steps[3].set_state(
                    "complete" if job_state == "complete" else "active",
                    job_state.title(),
                )
            self.video_count_label.setText(
                f"{len(self.videos)} video{'s' if len(self.videos) != 1 else ''} selected"
                if self.videos
                else "No videos selected"
            )
            if not running:
                self.next_step.setText(
                    "Add recordings and a study name, then run the analysis on this computer."
                    if not has_videos
                    else "Review the selected videos and model, then choose Run locally."
                )
            return

        for step, title in zip(
            self.workflow_steps,
            ("Google account", "Colab compute", "Input videos", "Analysis"),
        ):
            step.title.setText(title)
        signed_in = bool(self.drive.service)
        has_videos = bool(self.videos)
        self.compute_button.setText(
            "Preparing Colab..."
            if self.colab_connection_in_progress
            else "Update Colab"
            if self.worker_outdated
            else "View Colab"
            if self.worker_online
            else "Open Colab"
        )
        self.compute_button.setEnabled(signed_in and not self.colab_connection_in_progress)
        self.compute_button.setToolTip(
            (
                "Open the same StimTrace notebook used by the running worker. If Colab shows Connect, "
                "use it to reattach this browser tab; do not run the cells again."
            )
            if self.worker_online
            else "Open the StimTrace Compute notebook, choose a GPU runtime, and run both cells once."
        )
        self.workflow_steps[0].set_state(
            "complete" if signed_in else "active",
            "Connected" if signed_in else "Sign in",
        )
        self.workflow_steps[0].set_clickable(
            self.sign_in_thread is None and self.restore_session_thread is None,
            "Sign in to Google Drive" if not signed_in else "Google Drive account connected",
        )
        self.workflow_steps[1].set_state(
            "complete" if self.worker_online else ("active" if signed_in else "pending"),
            (
                f"Worker online ({self.cloud_runtime_description(compact=True)})"
                if self.worker_online and self.worker_device
                else "Worker online" if self.worker_online
                else "Restart with updated notebook" if self.worker_outdated
                else "Waiting for worker" if self.colab_opened
                else "Open and run"
            ),
        )
        self.workflow_steps[1].set_clickable(
            signed_in and not self.colab_connection_in_progress,
            self.cloud_runtime_description()
            if self.worker_online
            else "Open the StimTrace Colab compute notebook",
        )
        self.workflow_steps[2].set_state(
            "complete" if has_videos else ("active" if signed_in and self.colab_opened else "pending"),
            f"{len(self.videos)} selected" if has_videos else "Add recordings",
        )
        self.workflow_steps[2].set_clickable(True, "Select recordings to analyze")
        terminal = job_state in TERMINAL_JOB_STATES
        active = bool(self.active_progress_file_id) and not terminal
        self.workflow_steps[3].set_state(
            "complete" if job_state == "complete" else ("active" if active or has_videos else "pending"),
            (
                job_state.replace("_", " ").title()
                if job_state
                else "Update Colab first" if self.worker_outdated and has_videos
                else "Ready to submit" if has_videos
                else "Submit job"
            ),
        )
        self.workflow_steps[3].set_clickable(True, "Open the Signal Analyzer")
        self.video_count_label.setText(
            f"{len(self.videos)} video{'s' if len(self.videos) != 1 else ''} selected"
            if self.videos
            else "No videos selected"
        )

    def reset_job_progress(self, message: str) -> None:
        self.file_progress.setRange(0, 100)
        self.file_progress.setValue(0)
        self.file_progress.setFormat("0%")
        self.job_progress_bar.setRange(0, 1000)
        self.job_progress_bar.setValue(0)
        self.job_progress_bar.setFormat("0%")
        self.upload_progress.setValue(0)
        self.upload_progress.setVisible(False)
        self.stop_upload_attention()
        self.file_progress_label.setText(message)
        self.job_progress_label.setText(message)

    def prepare_new_submission_progress(self) -> None:
        """Clear progress left by the previously displayed job."""
        self.stop_upload_attention()
        self.monitoring_label.setText("Preparing new submission...")
        self.monitoring_label.setVisible(True)

        self.file_progress_label.setText("Current file: waiting for processing to start")
        self.file_progress_label.setVisible(False)
        self.file_progress.setRange(0, 100)
        self.file_progress.setValue(0)
        self.file_progress.setFormat("0%")
        self.file_progress.setVisible(False)

        self.job_progress_label.setText("Total job: waiting for submission")
        self.job_progress_label.setVisible(False)
        self.job_progress_bar.setRange(0, 1000)
        self.job_progress_bar.setValue(0)
        self.job_progress_bar.setFormat("0%")
        self.job_progress_bar.setVisible(False)

        self.upload_progress_label.setText("Preparing video upload...")
        self.upload_progress_label.setVisible(False)
        self.upload_progress.setRange(0, 100)
        self.upload_progress.setValue(0)
        self.upload_progress.setFormat("0%")
        self.upload_progress.setVisible(False)

    def start_upload_attention(self, text: str) -> None:
        self.upload_progress_label.setText(text)
        self.upload_progress_label.setVisible(True)
        self.upload_blink_phase = False
        self.upload_progress_label.setProperty("blinkPhase", "bright")
        self.upload_progress_label.style().unpolish(self.upload_progress_label)
        self.upload_progress_label.style().polish(self.upload_progress_label)
        self.upload_attention_timer.start()

    def update_upload_attention(self, text: str) -> None:
        self.upload_progress_label.setText(text)

    def toggle_upload_attention(self) -> None:
        self.upload_blink_phase = not self.upload_blink_phase
        self.upload_progress_label.setProperty(
            "blinkPhase",
            "dim" if self.upload_blink_phase else "bright",
        )
        self.upload_progress_label.style().unpolish(self.upload_progress_label)
        self.upload_progress_label.style().polish(self.upload_progress_label)

    def stop_upload_attention(self) -> None:
        if hasattr(self, "upload_attention_timer"):
            self.upload_attention_timer.stop()
        if hasattr(self, "upload_progress_label"):
            self.upload_progress_label.setVisible(False)

    def current_job(self) -> dict | None:
        if self.monitored_job_id:
            job = next(
                (item for item in self.jobs if item.get("job_id") == self.monitored_job_id),
                None,
            )
            if job:
                return job
        return self._job_by_progress_id(self.active_progress_file_id)

    def set_active_job_focus(self, job: dict | None, state: str) -> None:
        active_states = {"queued", "downloading", "analyzing", "uploading_results", "cancelling"}
        active = bool(job) and state in active_states
        show_setup = not active or self.setup_expanded_during_job
        for widget in self.setup_widgets:
            widget.setVisible(show_setup)
        # Colab is the active cloud-job companion, not merely setup. Keep it
        # available while a job is running even when setup details are collapsed.
        self.compute_button.setVisible(self.settings.compute_mode == "cloud")
        self.setup_toggle_button.setVisible(active)
        self.setup_toggle_button.setText("Hide setup" if show_setup and active else "Show setup")
        if job:
            model_name = job.get("model_name", self.settings.selected_model_name)
            video_count = len(job.get("videos", []))
            local_job = job.get("backend") == "local"
            compute_state = (
                job.get("device_name", "this computer")
                if local_job
                else self.colab_worker_state_text()
            )
            self.active_job_summary.setText(
                f"Job: {job.get('job_id', 'unknown')}   |   Study: {job.get('study', '')}   |   "
                f"Model: {model_name}   |   "
                f"{video_count} video{'s' if video_count != 1 else ''}   |   "
                f"{'Local' if local_job else 'Colab'}: {compute_state}"
            )
        else:
            self.active_job_summary.setText("No submitted job selected")
        self.cancel_current_button.setEnabled(active and state != "cancelling")
        self.retry_current_button.setEnabled(bool(job) and state in {"failed", "cancelled"})
        self.open_current_drive_button.setEnabled(bool(job and job.get("folder_id")))
        complete = bool(job) and state == "complete"
        self.open_results_button.setEnabled(complete)
        self.analyze_results_button.setEnabled(complete)
        if self.settings.compute_mode == "local":
            for widget in (
                self.remember_sign_in,
                self.sign_out_button,
                self.sign_button,
                self.compute_button,
            ):
                widget.setVisible(False)
        self.update_video_table_height()

    def set_pending_submission_focus(self) -> None:
        video_count = len(self.videos)
        compute_state = self.colab_worker_state_text()
        self.active_job_summary.setText(
            f"New submission   |   Study: {self.study.text().strip()}   |   "
            f"Model: {self.settings.selected_model_name}   |   "
            f"{video_count} video{'s' if video_count != 1 else ''}   |   "
            f"Uploading to Drive   |   Colab: {compute_state}"
        )
        for button in (
            self.cancel_current_button,
            self.retry_current_button,
            self.open_current_drive_button,
            self.open_results_button,
            self.analyze_results_button,
        ):
            button.setEnabled(False)
        self.update_video_table_height()

    def toggle_setup_details(self) -> None:
        self.setup_expanded_during_job = not self.setup_expanded_during_job
        job = self.current_job()
        self.set_active_job_focus(job, job.get("state", "queued") if job else "")

    def cancel_current_job(self) -> None:
        job = self.current_job()
        if not job:
            return
        if QMessageBox.question(
            self,
            "Cancel analysis",
            f"Cancel {job.get('job_id', 'this job')}?",
        ) == QMessageBox.Yes:
            try:
                self.cancel_job(job)
            except Exception as error:
                QMessageBox.critical(self, "Cancellation failed", str(error))

    def retry_current_job(self) -> None:
        job = self.current_job()
        if not job:
            return
        self.request_job_retry(job)

    def open_current_job_drive(self) -> None:
        job = self.current_job()
        if job and job.get("folder_id"):
            webbrowser.open(f"https://drive.google.com/drive/folders/{job['folder_id']}")

    def open_current_results(self) -> None:
        job = self.current_job()
        if not job:
            return
        target = Path(job["local_results_path"]) if job.get("local_results_path") else None
        if not target or not target.exists():
            self.start_result_download(job, force=True, notify=True, action="open")
        elif target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def analyze_current_results(self) -> None:
        job = self.current_job()
        if not job:
            return
        self.open_job_in_signal_viewer(job)

    def open_job_in_signal_viewer(self, job: dict) -> None:
        """Open a completed segmentation study, downloading cloud results if required."""
        if job.get("type", "segmentation") != "segmentation":
            QMessageBox.information(
                self,
                "Signal Viewer unavailable",
                "Training jobs do not contain force traces for Signal Viewer.",
            )
            return
        if job.get("state") != "complete":
            QMessageBox.information(
                self,
                "Study not complete",
                "Signal Viewer can open this study after its analysis has completed.",
            )
            return
        target = self.result_download_target(job)
        if target is not None and target.is_dir():
            trace_file = self.analysis_trace_file(target)
            if trace_file is not None:
                self.open_signal_analysis_files([trace_file])
                return
            if job.get("backend") == "local":
                QMessageBox.information(
                    self,
                    "Force traces unavailable",
                    "The local results do not contain a combined force-trace file.",
                )
                return
        elif job.get("backend") == "local":
            QMessageBox.warning(
                self,
                "Results unavailable",
                "The local results folder no longer exists at its recorded location.",
            )
            return
        if not self.drive.service:
            QMessageBox.information(
                self,
                "Sign in required",
                "Sign in to Google Drive before downloading this study's results.",
            )
            return
        self.start_result_download(job, force=True, notify=True, action="analyze")

    @staticmethod
    def analysis_trace_file(results_folder: Path) -> Path | None:
        """Return the preferred signal-analysis input for a completed job."""
        return find_analysis_trace_file(results_folder)

    def restore_google_session(self) -> None:
        if (
            self.shutdown_started
            or self.restore_session_thread is not None
            or self.sign_in_thread is not None
        ):
            return
        self.sign_button.setEnabled(False)
        self.sign_button.setText("Restoring...")
        self.set_auth_status("Restoring saved Google session...", signed_in=False)

        def restore() -> str:
            email = self.drive.sign_in(interactive=False)
            self.drive.ensure_workspace(self.settings)
            return email

        thread = BackgroundFunctionThread(restore, self)
        self.register_thread(thread, "google-session-restore")
        thread.succeeded.connect(self.finish_restored_google_session, Qt.QueuedConnection)
        thread.failed.connect(self.fail_restored_google_session, Qt.QueuedConnection)
        thread.finished.connect(self.finish_restore_session_thread, Qt.QueuedConnection)
        self.restore_session_thread = thread
        thread.start()

    @Slot(object)
    def finish_restored_google_session(self, email: object) -> None:
        self.apply_signed_in_state(str(email), open_colab=False)

    @Slot(str)
    def fail_restored_google_session(self, message: str) -> None:
        LOGGER.warning("Saved Google session could not be restored: %s", message)
        token_is_invalid = saved_google_session_was_revoked(message)
        if token_is_invalid:
            (USER_DATA_DIR / "token.json").unlink(missing_ok=True)
        self.drive.service = None
        self.drive.account_email = ""
        self.set_auth_status(
            "Saved Google session expired. Sign in again."
            if token_is_invalid
            else "Could not reach Google Drive. Check your connection and try Sign in again.",
            signed_in=False,
        )

    @Slot()
    def finish_restore_session_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.restore_session_thread is thread:
            self.restore_session_thread = None
        if self.sign_in_thread is None:
            self.sign_button.setText("Sign in")
            self.sign_button.setEnabled(not bool(self.drive.service))

    def set_auth_status(self, message: str, signed_in: bool) -> None:
        self.status.setText(message)
        self.status.setProperty("authState", "signedIn" if signed_in else "signedOut")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def poll_worker_status(self) -> None:
        if not self.drive.service or not self.settings.worker_status_file_id:
            self.worker_online = False
            self.worker_outdated = False
            self.worker_device = ""
            self.worker_hardware = {}
            self.update_workflow_steps()
            return
        if self.colab_connection_in_progress:
            return
        if self.worker_poll_thread is not None:
            return
        thread = BackgroundFunctionThread(
            lambda: DriveClient.saved_session_json(
                self.settings.worker_status_file_id
            ),
            self,
        )
        self.register_thread(thread, "drive-worker-status-poll")
        thread.succeeded.connect(self.apply_worker_status, Qt.QueuedConnection)
        thread.failed.connect(self.worker_status_unavailable, Qt.QueuedConnection)
        thread.finished.connect(
            self.finish_worker_poll_thread,
            Qt.QueuedConnection,
        )
        self.worker_poll_thread = thread
        thread.start()

    def apply_worker_status(self, status: object) -> None:
        if not isinstance(status, dict):
            self.worker_status_unavailable("Invalid worker status response.")
            return
        heartbeat_at = float(status.get("updated_at", 0) or 0)
        age = max(0.0, time.time() - heartbeat_at)
        heartbeat_online = (
            status.get("state") == "online"
            and age < WORKER_HEARTBEAT_GRACE_SECONDS
        )
        if heartbeat_online:
            self.worker_last_heartbeat_at = heartbeat_at
        worker_version = int(status.get("worker_version", 0) or 0)
        self.worker_outdated = heartbeat_online and worker_version != NOTEBOOK_VERSION
        self.worker_online = heartbeat_online and not self.worker_outdated
        self.worker_hardware = status if self.worker_online else {}
        self.worker_device = (
            status.get("device_name", status.get("device", ""))
            if self.worker_online
            else ""
        )
        if self.worker_online:
            self.follow_worker_reported_job(status)
        self.finish_worker_status_update()

    def follow_worker_reported_job(self, status: dict) -> None:
        """Show the segmentation job that the live Colab worker says it is running."""
        if (
            self.submission_in_progress
            or self.settings.compute_mode != "cloud"
            or status.get("current_job_type") != "segmentation"
        ):
            return
        progress_file_id = str(status.get("current_progress_file_id", ""))
        job_id = str(status.get("current_job_id", ""))
        folder_id = str(status.get("current_job_folder_id", ""))
        target = next(
            (
                job for job in self.jobs
                if job.get("type", "segmentation") == "segmentation"
                and (
                    (progress_file_id and job.get("progress_file_id") == progress_file_id)
                    or (folder_id and job.get("folder_id") == folder_id)
                    or (job_id and job.get("job_id") == job_id)
                )
            ),
            None,
        )
        if target is None:
            # The job may have been submitted from another StimTrace installation.
            # Discover it without blocking the interface, then the next heartbeat
            # can select it using the authoritative worker IDs.
            self.start_drive_job_discovery()
            return
        if target.get("job_id") != self.monitored_job_id:
            # The worker heartbeat makes a full folder scan unnecessary here. Poll
            # this job's progress.json immediately after switching the panel.
            self.last_drive_discovery_started_at = time.monotonic()
            self.activate_job(target, quiet=True)
        elif self.job_progress_thread is None and self.drive_discovery_thread is None:
            self.poll_job_progress()

    def worker_status_unavailable(self, message: str) -> None:
        last_seen_age = time.time() - self.worker_last_heartbeat_at
        if (
            self.worker_last_heartbeat_at
            and last_seen_age < WORKER_HEARTBEAT_GRACE_SECONDS
            and (self.worker_online or self.worker_outdated)
        ):
            LOGGER.warning(
                "Worker heartbeat read failed; retaining the last confirmed runtime "
                "for %.0f more seconds: %s",
                WORKER_HEARTBEAT_GRACE_SECONDS - last_seen_age,
                message,
            )
            return
        self.worker_online = False
        self.worker_outdated = False
        self.worker_last_heartbeat_at = 0.0
        self.worker_device = ""
        self.worker_hardware = {}
        self.finish_worker_status_update()

    @Slot()
    def finish_worker_poll_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.worker_poll_thread is thread:
            self.worker_poll_thread = None

    def finish_worker_status_update(self) -> None:
        self.update_workflow_steps()
        if self.worker_online:
            self.set_colab_waiting(False)
        elif self.colab_opened and not self.worker_outdated:
            self.set_colab_waiting(
                True,
                "Waiting for Colab worker. Run both notebook cells.",
            )
        if self.worker_outdated:
            self.set_colab_waiting(False)
            self.next_step.setText(
                "The running Colab worker is outdated. Open Colab, restart the session, "
                "and run the updated notebook before retrying this job."
            )
            self.submit_button.setText("Update Colab to submit")
            self.submit_button.setToolTip(
                "The current Colab worker cannot process this version of StimTrace. "
                "Open the updated notebook and restart its runtime first."
            )
            self.submit_button.setEnabled(bool(self.videos) and bool(self.drive.service))
        elif self.settings.compute_mode == "cloud":
            self.submit_button.setText("Submit analysis")
            self.submit_button.setToolTip("")
        job = self.current_job()
        if job and not self.submission_in_progress:
            self.set_active_job_focus(job, job.get("state", "queued"))

    def sign_out(self) -> None:
        self.poll_timer.stop()
        self.worker_timer.stop()
        self.drive.sign_out()
        self.active_progress_file_id = ""
        self.monitored_job_id = ""
        self.set_auth_status("Not signed in", signed_in=False)
        self.sign_out_button.setEnabled(False)
        self.sign_button.setText("Sign in")
        self.sign_button.setEnabled(True)
        if self.training_page is not None:
            self.training_page.update_training_account_controls()
            self.training_page.update_training_workflow()
        self.compute_button.setEnabled(False)
        local_mode = self.settings.compute_mode == "local"
        self.add_button.setEnabled(local_mode)
        self.add_folder_button.setEnabled(local_mode)
        self.submit_button.setEnabled(local_mode and bool(self.videos))
        self.monitoring_label.setVisible(False)
        self.file_progress_label.setVisible(False)
        self.file_progress.setVisible(False)
        self.job_progress_label.setVisible(False)
        self.job_progress_bar.setVisible(False)
        self.upload_progress.setVisible(False)
        self.progress_box.setVisible(False)
        self.update_video_table_height()
        self.colab_opened = False
        self.worker_online = False
        self.worker_outdated = False
        self.worker_device = ""
        self.worker_hardware = {}
        self.set_colab_waiting(False)
        self.setup_expanded_during_job = False
        for widget in self.setup_widgets:
            widget.setVisible(True)
        self.next_step.setText(
            "Add recordings and a study name, then run the analysis on this computer."
            if local_mode
            else "Step 1: Sign in to Google Drive to prepare your personal StimTrace workspace."
        )
        self.update_workflow_steps()

    def add_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose videos",
            "",
            "Video files (*.avi *.mp4 *.mov *.mkv *.m4v)",
        )
        self.add_video_paths(map(Path, files))

    def add_video_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose video folder")
        if not folder:
            return
        try:
            files = video_files_in_folder(Path(folder))
        except ValueError as error:
            QMessageBox.warning(self, "Video folder unavailable", str(error))
            return
        if not files:
            QMessageBox.information(
                self,
                "No supported videos",
                "The selected folder contains no AVI, MP4, MOV, MKV, or M4V files.",
            )
            return
        self.add_video_paths(files)

    def add_video_paths(self, paths) -> None:
        existing = {path.resolve() for path in self.videos}
        for path in map(Path, paths):
            if path.resolve() not in existing:
                self.videos.append(path)
                existing.add(path.resolve())
        self.refresh_video_table()
        if self.videos:
            self.submit_button.setEnabled(
                self.settings.compute_mode == "local"
                or (bool(self.drive.service) and not self.worker_outdated)
            )
            self.next_step.setText(
                "Review the selected videos and model, then choose Run locally."
                if self.settings.compute_mode == "local"
                else "Step 4: Enter a study name, review the selected videos, then submit the analysis job."
            )
        self.update_workflow_steps()

    def clear_videos(self) -> None:
        self.videos.clear()
        self.video_table.setRowCount(0)
        self.submit_button.setEnabled(False)
        self.update_video_table_height()
        self.next_step.setText("Step 3: Add the videos you want to analyze.")
        self.update_workflow_steps()

    @staticmethod
    def video_metadata(path: Path) -> tuple[str, str, str]:
        if not path.exists():
            return "-", "-", "Missing"
        size_mb = path.stat().st_size / (1024 * 1024)
        size_text = f"{size_mb:.1f} MB"
        try:
            import cv2

            capture = cv2.VideoCapture(str(path))
            frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            duration = frames / fps if fps > 0 else 0
            duration_text = (
                f"{int(duration // 60)}m {int(duration % 60):02d}s"
                if duration >= 60
                else f"{duration:.1f}s"
            )
            return duration_text, str(frames) if frames else "-", size_text
        except Exception:
            LOGGER.debug("Could not read video metadata for %s", path, exc_info=True)
            return "-", "-", size_text

    def refresh_video_table(self, remote_job: bool = False) -> None:
        self.video_table.setRowCount(len(self.videos))
        for row, path in enumerate(self.videos):
            duration, frames, size_or_status = self.video_metadata(path)
            missing = size_or_status == "Missing"
            size_text = "-" if missing else size_or_status
            status = "Drive job" if remote_job and missing else "Missing" if missing else "Ready"
            name_item = QTableWidgetItem(path.name)
            name_item.setToolTip(str(path))
            self.video_table.setItem(row, 0, name_item)
            for column, value in enumerate((duration, frames, size_text, status), start=1):
                self.video_table.setItem(row, column, QTableWidgetItem(value))
            remove = QPushButton("X")
            configure_row_delete(remove, f"Remove {path.name}")
            remove.setAccessibleName(f"Remove {path.name}")
            remove.clicked.connect(lambda _checked=False, video=path: self.remove_video(video))
            remove.setEnabled(not bool(self.active_progress_file_id))
            self.video_table.setCellWidget(row, 5, remove)
        self.video_table.resizeRowsToContents()
        self.update_video_table_height()

    def update_video_table_height(self) -> None:
        row_count = self.video_table.rowCount()
        progress_visible = bool(
            getattr(self, "progress_box", None) and self.progress_box.isVisible()
        )
        minimum_rows = 4 if progress_visible else 6
        visible_rows = min(12, max(minimum_rows, row_count))
        default_row_height = max(38, self.video_table.verticalHeader().defaultSectionSize())
        content_height = sum(
            max(default_row_height, self.video_table.rowHeight(row))
            for row in range(min(row_count, visible_rows))
        )
        content_height += max(0, visible_rows - row_count) * default_row_height
        header_height = max(32, self.video_table.horizontalHeader().sizeHint().height())
        frame_height = self.video_table.frameWidth() * 2
        # Keep several rows usable on compact windows, but allow the recording
        # list to consume additional space when the main window is enlarged.
        self.video_table.setMinimumHeight(header_height + content_height + frame_height + 4)
        self.video_table.setMaximumHeight(16777215)

    def set_video_status(self, row: int, status: str) -> None:
        if 0 <= row < self.video_table.rowCount():
            item = self.video_table.item(row, 4)
            if item:
                item.setText(status)

    def remove_video(self, video: Path) -> None:
        self.videos = [path for path in self.videos if path != video]
        self.refresh_video_table()
        self.submit_button.setEnabled(
            bool(self.videos)
            and (self.settings.compute_mode == "local" or bool(self.drive.service))
        )
        self.update_workflow_steps()

    @staticmethod
    def _load_jobs() -> list[dict]:
        try:
            data = json.loads(JOB_HISTORY_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_jobs(self) -> None:
        JOB_HISTORY_PATH.write_text(json.dumps(self.jobs, indent=2), encoding="utf-8")

    def _job_by_progress_id(self, progress_file_id: str) -> dict | None:
        return next(
            (job for job in self.jobs if job.get("progress_file_id") == progress_file_id),
            None,
        )

    def _best_active_job(self, running_only: bool = False) -> dict | None:
        # Cancellation requests can remain pending indefinitely while Colab is
        # offline. Keep them in Job History, but do not take over the main
        # screen with a stale request on the next application start.
        candidates = [
            job for job in self.jobs
            if job.get("progress_file_id")
            and (
                job.get("state") in RUNNING_JOB_STATES
                or (not running_only and job.get("state") == "queued")
            )
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda job: (
                1 if job.get("state") in RUNNING_JOB_STATES else 0,
                job.get("progress_modified_at", ""),
                job.get("created_at", ""),
            ),
        )

    def follow_active_cloud_segmentation_job(self) -> None:
        """Keep the Segment progress panel on the job Colab is actually processing."""
        if self.submission_in_progress or self.settings.compute_mode != "cloud":
            return
        candidates = [
            job for job in self.jobs
            if job.get("type", "segmentation") == "segmentation"
            and job.get("progress_file_id")
            and job.get("state") in ACTIVE_JOB_STATES
        ]
        if not candidates:
            return
        running = [job for job in candidates if job.get("state") in RUNNING_JOB_STATES]
        # A running job takes priority; otherwise retain the current queued job if it
        # is still eligible, falling back to the oldest queued submission.
        current = self.current_job()
        target = (
            max(running, key=lambda job: (job.get("progress_modified_at", ""), job.get("created_at", "")))
            if running
            else current if current in candidates
            else min(candidates, key=lambda job: job.get("created_at", ""))
        )
        if target.get("job_id") != self.monitored_job_id:
            self.activate_job(target, quiet=True)

    def open_signal_analysis(self) -> None:
        self._show_signal_analysis()

    def _show_signal_analysis(self, path: Path | None = None) -> None:
        self.requested_page_key = "signals"
        if path is not None:
            self.pending_signal_path = path
        if "signals" in self.module_load_errors:
            self.module_preload_failed("signals", self.module_load_errors["signals"])
            return
        if "signals" not in self.loaded_module_keys:
            self.show_module_loading("signals", "Signal Analysis")
            # Paint the loading page first, then import on the GUI thread. Importing
            # Matplotlib's Qt backend from QThread can crash Qt6Core outright.
            QTimer.singleShot(0, lambda: self.load_scientific_module("signals"))
            return
        try:
            from signal_analysis import SignalAnalysisPage
        except ImportError as error:
            QMessageBox.critical(
                self,
                "Signal analysis unavailable",
                f"{error}\n\nInstall the desktop requirements and restart StimTrace.",
            )
            return
        if self.signal_page is None:
            self.signal_page = SignalAnalysisPage(self)
            self.register_page("signals", self.signal_page)
        self.show_page("signals")
        if self.pending_signal_handoff_paths:
            pending_paths = self.pending_signal_handoff_paths
            self.pending_signal_handoff_paths = []
            self.open_signal_handoff(pending_paths)
        elif self.pending_signal_path is not None:
            pending_path = self.pending_signal_path
            self.pending_signal_path = None
            self.load_signal_file_async(pending_path, sheet_name="Force traces")
        elif self.pending_signal_paths:
            pending_paths = self.pending_signal_paths
            self.pending_signal_paths = []
            self.load_signal_files_async(pending_paths)

    def open_signal_analysis_files(self, paths: list[Path]) -> None:
        """Offer to add or replace external traces in the shared signal workspace."""
        self.pending_signal_handoff_paths = [Path(path) for path in paths]
        self._show_signal_analysis()

    def open_signal_handoff(self, paths: list[Path]) -> None:
        """Load completed-job traces without silently discarding the current workspace."""
        if self.signal_page is None:
            self.pending_signal_handoff_paths = [Path(path) for path in paths]
            return
        paths = [Path(path) for path in paths]
        if not paths:
            return
        if not self.signal_page.signal_columns():
            self.load_signal_files_async(paths)
            return

        question = QMessageBox(self)
        question.setIcon(QMessageBox.Question)
        question.setWindowTitle("Add or replace traces?")
        question.setText(
            f"{len(self.signal_page.signal_columns())} trace(s) are currently loaded.\n\n"
            "Add the completed-job traces to the current workspace, or replace the current workspace?"
        )
        add_button = question.addButton("Add traces", QMessageBox.AcceptRole)
        replace_button = question.addButton("Replace traces", QMessageBox.DestructiveRole)
        question.addButton(QMessageBox.Cancel)
        question.setDefaultButton(add_button)
        question.exec()
        selected = question.clickedButton()
        if selected is add_button:
            combined: list[Path] = []
            seen: set[str] = set()
            for path in [*self.signal_page.paths, *paths]:
                identity = str(path.resolve()).casefold()
                if identity not in seen:
                    seen.add(identity)
                    combined.append(path)
            self.load_signal_files_async(combined)
            return
        if selected is not replace_button:
            return

        save_question = QMessageBox(self)
        save_question.setIcon(QMessageBox.Question)
        save_question.setWindowTitle("Save current analysis?")
        save_question.setText(
            "Export the currently selected processed traces before replacing the Signal Analysis workspace?"
        )
        save_button = save_question.addButton("Export and replace", QMessageBox.AcceptRole)
        discard_button = save_question.addButton("Replace without saving", QMessageBox.DestructiveRole)
        save_question.addButton(QMessageBox.Cancel)
        save_question.setDefaultButton(save_button)
        save_question.exec()
        selected = save_question.clickedButton()
        if selected is save_button:
            exported = self.signal_page.export_processed(
                lambda _saved: self.load_signal_files_async(paths)
            )
            if not exported:
                self.signal_page.set_status("Replacement cancelled; current traces were kept.")
            return
        if selected is discard_button:
            self.load_signal_files_async(paths)

    def load_signal_file_async(self, path: Path, sheet_name: str | None = None) -> None:
        self.load_signal_files_async(
            [Path(path)],
            {Path(path): sheet_name} if sheet_name else None,
        )

    def load_signal_files_async(
        self,
        paths: list[Path],
        sheet_names: dict[Path, str | None] | None = None,
    ) -> None:
        if self.signal_page is None:
            return
        paths = [Path(path) for path in paths]
        if not paths:
            return
        if self.signal_load_thread is not None:
            self.signal_page.set_status("Another trace file is still loading.")
            return
        if self.signal_page.operation_thread is not None:
            self.signal_page.set_status(
                "Finish the current signal-analysis operation before loading new traces."
            )
            return
        label = paths[0].name if len(paths) == 1 else f"{len(paths)} trace files"
        self.signal_page.set_operation_busy(True, f"Loading {label} in the background...")

        def preload() -> dict:
            loaded_files = []
            errors = []
            for path in paths:
                try:
                    loaded = self.signal_page._read_trace_file(
                        path,
                        (sheet_names or {}).get(path),
                        allow_dialog=False,
                    )
                    if loaded is not None:
                        loaded_files.append((str(path), loaded))
                except Exception as error:
                    errors.append(f"{path.name}: {error}")
            return {
                "paths": [path for path, _loaded in loaded_files],
                "loaded_files": loaded_files,
                "errors": errors,
            }

        thread = BackgroundFunctionThread(
            preload,
            self,
        )
        self.register_thread(thread, "signal-trace-file-loader")
        thread.succeeded.connect(self.finish_signal_file_load, Qt.QueuedConnection)
        thread.failed.connect(self.fail_signal_file_load, Qt.QueuedConnection)
        thread.finished.connect(self.finish_signal_load_thread, Qt.QueuedConnection)
        self.signal_load_thread = thread
        thread.start()

    @Slot(object)
    def finish_signal_file_load(self, result: object) -> None:
        if self.signal_page is None or not isinstance(result, dict):
            return
        loaded_files = result.get("loaded_files", [])
        paths = [Path(path) for path in result.get("paths", [])]
        if not loaded_files:
            self.signal_page.set_status("No trace data was loaded.")
            return
        for path_text, loaded in loaded_files:
            self.signal_page.cache_preloaded_trace_file(Path(path_text), loaded)
        self.signal_page.load_files(paths)
        errors = result.get("errors", [])
        if errors:
            QMessageBox.warning(self, "Some files were skipped", "\n".join(errors[:10]))

    @Slot(str)
    def fail_signal_file_load(self, message: str) -> None:
        if self.signal_page is not None:
            self.signal_page.set_status(f"Could not open traces: {message}")
        QMessageBox.critical(self, "Could not open traces", message)

    @Slot()
    def finish_signal_load_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.signal_load_thread is thread:
            self.signal_load_thread = None
        if self.signal_page is not None:
            self.signal_page.set_operation_busy(False)
        if self.pending_signal_path is not None and self.requested_page_key == "signals":
            pending = self.pending_signal_path
            self.pending_signal_path = None
            QTimer.singleShot(0, lambda: self.load_signal_file_async(pending, "Force traces"))

    def offer_signal_analysis(self, job: dict) -> None:
        if job.get("type", "segmentation") != "segmentation" or job.get("analysis_prompted"):
            return
        local_results = job.get("local_results_path", "")
        if not local_results:
            return
        trace_file = self.analysis_trace_file(Path(local_results))
        if trace_file is None:
            return
        self.record_actual_processing_time(job)
        job["analysis_prompted"] = True
        self._save_jobs()
        if QMessageBox.question(
            self,
            "Open signal analysis?",
            "Segmentation and force inference are complete.\n\n"
            "Open the combined force traces in StimTrace Signal Analysis now?",
        ) == QMessageBox.Yes:
            self.open_signal_analysis_files([trace_file])

    def open_model_training(self) -> None:
        self.requested_page_key = "training"
        if "training" in self.module_load_errors:
            self.module_preload_failed("training", self.module_load_errors["training"])
            return
        if "training" not in self.loaded_module_keys:
            self.show_module_loading("training", "Model Training")
            QTimer.singleShot(0, lambda: self.load_scientific_module("training"))
            return
        try:
            from model_training import ModelTrainingPage
        except ImportError as error:
            QMessageBox.critical(
                self,
                "Model training unavailable",
                f"{error}\n\nInstall the desktop requirements and restart StimTrace.",
            )
            return
        if self.training_page is None:
            self.training_page = ModelTrainingPage(self)
            self.register_page("training", self.training_page)
        self.show_page("training")

    def open_point_tracking(self) -> None:
        self.requested_page_key = "point_tracking"
        if "point_tracking" in self.module_load_errors:
            self.module_preload_failed(
                "point_tracking",
                self.module_load_errors["point_tracking"],
            )
            return
        if "point_tracking" not in self.loaded_module_keys:
            self.show_module_loading("point_tracking", "Point Tracking")
            QTimer.singleShot(0, lambda: self.load_scientific_module("point_tracking"))
            return
        try:
            from point_tracking import PointTrackingPage
        except ImportError as error:
            QMessageBox.critical(
                self,
                "Point tracking unavailable",
                f"{error}\n\nInstall the desktop requirements and restart StimTrace.",
            )
            return
        if self.point_tracking_page is None:
            self.point_tracking_page = PointTrackingPage(self)
            self.register_page("point_tracking", self.point_tracking_page)
        self.point_tracking_page.refresh_force_calibration()
        self.show_page("point_tracking")

    def open_model_manager(self) -> None:
        if self.model_page is None:
            self.model_page = ModelManagerPage(self)
            self.register_page("models", self.model_page)
        self.model_page.populate()
        self.show_page("models")

    def activate_model_profile(self, model_name: str, notify: bool = True) -> None:
        if model_name not in self.settings.model_profiles:
            raise KeyError(f"Unknown segmentation model: {model_name}")
        model_file_id = self.settings.model_profiles.get(model_name, "")
        local_path = self.settings.local_model_paths.get(model_name, "")
        local_only = bool(local_path and Path(local_path).is_file()) or (
            model_name == "Default pillar" and self.bundled_model_path().is_file()
        )
        if not model_file_id and not local_only:
            raise RuntimeError(
                "This checkpoint is not available in Google Drive or on this computer."
            )
        changed = (
            model_name != self.settings.selected_model_name
            or model_file_id != self.settings.model_file_id
        )
        self.settings.store_model_parameters()
        self.settings.selected_model_name = model_name
        self.settings.model_file_id = model_file_id
        self.settings.apply_model_parameters(model_name)
        if changed:
            self.settings.notebook_version = 0
            self.settings.notebook_file_id = ""
            self.worker_online = False
            self.worker_outdated = False
            self.worker_device = ""
            self.worker_hardware = {}
        if self.drive.service and model_file_id:
            self.drive.ensure_workspace(self.settings)
        else:
            self.settings.save()
        self.model_status.setText(f"Segmentation model: {model_name}")
        self.update_workflow_steps()
        if changed:
            self.next_step.setText(
                f"{model_name} is selected for local analysis."
                if local_only and not model_file_id
                else (
                    f"{model_name} is selected. Open the regenerated Colab notebook and run both "
                    "cells before submitting a new analysis."
                )
            )
        if notify:
            QMessageBox.information(
                self,
                "Model activated",
                (
                    f"{model_name} is now used for local segmentation jobs."
                    if local_only and not model_file_id
                    else (
                        f"{model_name} is now used for new segmentation jobs.\n\n"
                        "Open Colab and run both cells so the worker loads this checkpoint."
                    )
                    if changed
                    else f"{model_name} is already the active segmentation model."
                ),
            )

    def import_trained_model(
        self,
        open_colab: bool = True,
        parent: QWidget | None = None,
    ) -> str | None:
        dialog_parent = parent or self
        filename, _ = QFileDialog.getOpenFileName(
            dialog_parent,
            "Import trained segmentation model",
            "",
            "PyTorch model (*.pth)",
        )
        if not filename:
            return None
        try:
            source = Path(filename).resolve()
            if self.drive.service:
                profile_name = self.drive.import_local_model(source, self.settings)
            else:
                base_name = source.stem
                profile_name = base_name
                suffix = 2
                while profile_name in self.settings.model_profiles:
                    profile_name = f"{base_name} ({suffix})"
                    suffix += 1
                self.settings.store_model_parameters()
                self.settings.model_profiles[profile_name] = ""
                self.settings.ensure_model_parameters(profile_name)
                self.settings.selected_model_name = profile_name
                self.settings.model_file_id = ""
                self.settings.model_metadata[profile_name] = {
                    "object_type": "Other",
                    "source": "Imported locally",
                    "created_at": datetime.now(timezone.utc).astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "dataset_size": "",
                }
            self.settings.local_model_paths[profile_name] = str(Path(filename).resolve())
            self.settings.save()
            self.model_status.setText(f"Segmentation model: {profile_name}")
            QMessageBox.information(
                dialog_parent,
                "Model imported",
                (
                    f"{profile_name} is now selected for local segmentation."
                    if not self.drive.service
                    else (
                        f"{profile_name} is now selected. Stop the old Colab worker, open the "
                        "regenerated notebook, and run both cells before submitting segmentation jobs."
                    )
                ),
            )
            if open_colab and self.drive.service:
                self.open_colab()
            return profile_name
        except Exception as error:
            QMessageBox.critical(dialog_parent, "Could not import model", str(error))
            return None

    def prepare_training_job(
        self, archive: Path, model_name: str, epochs: int, batch_size: int
    ) -> dict:
        """Upload a training job without mutating Qt-owned job-list state."""
        with zipfile.ZipFile(archive) as bundle:
            dataset_size = sum(
                1
                for name in bundle.namelist()
                if name.replace("\\", "/").startswith("masks/")
                and name.lower().endswith(".png")
            )
        result = self.drive.submit_training(
            archive,
            model_name,
            epochs,
            batch_size,
            self.settings,
        )
        return {**result, "dataset_size": dataset_size}

    def record_training_job(
        self,
        result: dict,
        model_name: str,
        project_dir: Path | None = None,
    ) -> None:
        """Apply a remotely prepared training job on the GUI thread."""
        self.jobs.append({
            "job_id": result["job_id"],
            "study": model_name,
            "type": "training",
            "folder_id": result["folder_id"],
            "progress_file_id": result["progress_file_id"],
            "preview_file_id": "",
            "model_output_file_id": result["model_output_file_id"],
            "model_name": model_name,
            "annotation_project_path": str(project_dir) if project_dir else "",
            "dataset_size": result.get("dataset_size", ""),
            "videos": [],
            "state": "queued",
            "message": "Training queued. Waiting for Colab compute.",
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        self._save_jobs()

    def activate_trained_model(self, model_file_id: str, model_name: str) -> None:
        self.drive.activate_trained_model(model_file_id, model_name, self.settings)
        profile_name = self.settings.selected_model_name
        training_job = next(
            (
                job for job in reversed(self.jobs)
                if job.get("type") == "training"
                and job.get("model_output_file_id") == model_file_id
            ),
            None,
        )
        metadata = self.settings.model_metadata.setdefault(profile_name, {})
        metadata["source"] = "Trained"
        if training_job:
            metadata["dataset_size"] = training_job.get("dataset_size", "")
        self.settings.save()
        self.model_status.setText(f"Segmentation model: {self.settings.selected_model_name}")
        QMessageBox.information(
            self,
            "Model selected",
            f"{Path(model_name).stem} is now the segmentation model. The default pillar model remains "
            "available under Advanced settings. Stop the current Colab worker, "
            "open the regenerated notebook, and run both cells before submitting segmentation jobs.",
        )
        self.open_colab()

    def open_job_history(self) -> None:
        if self.job_history_page is None:
            self.job_history_page = JobHistoryPage(self)
            self.register_page("jobs", self.job_history_page)
        self.job_history_page.populate()
        self.show_page("jobs")

    def activate_job(self, job: dict, quiet: bool = False) -> None:
        self.active_progress_file_id = job.get("progress_file_id", "")
        self.monitored_job_id = job.get("job_id", "")
        if job.get("backend") == "local":
            self.set_compute_mode("local")
            self.restore_job_inputs(job)
            state = job.get("state", "unknown")
            self.progress_box.setVisible(True)
            self.monitoring_label.setText(
                f"Local job: {self.monitored_job_id} - {state.replace('_', ' ')}"
            )
            self.monitoring_label.setVisible(True)
            self.set_active_job_focus(job, state)
            self.next_step.setText(job.get("message", "Local job selected."))
            self.status.setText(job.get("message", "Local job selected."))
            self.file_progress_label.setVisible(True)
            self.file_progress.setVisible(True)
            self.job_progress_label.setVisible(True)
            self.job_progress_bar.setVisible(True)
            if state == "complete":
                self.file_progress_label.setText("All selected videos processed")
                self.file_progress.setValue(100)
                self.job_progress_label.setText("Local analysis complete")
                self.job_progress_bar.setValue(1000)
            elif state in {"failed", "cancelled"}:
                self.reset_job_progress(job.get("message", state.title()))
            return
        if not self.active_progress_file_id:
            if not quiet:
                QMessageBox.warning(self, "Cannot resume", "This history entry has no progress file.")
            return
        self.restore_job_inputs(job)
        self.setup_expanded_during_job = False
        self.set_active_job_focus(job, job.get("state", "queued"))
        self.monitoring_label.setText(f"Monitoring job: {self.monitored_job_id}")
        self.monitoring_label.setVisible(True)
        self.progress_box.setVisible(True)
        self.update_video_table_height()
        self.file_progress_label.setVisible(True); self.file_progress.setVisible(True)
        self.job_progress_label.setVisible(True); self.job_progress_bar.setVisible(True)
        self.upload_progress.setVisible(False)
        self.update_workflow_steps(job.get("state", "queued"))
        QTimer.singleShot(0, lambda: self.page.ensureWidgetVisible(self.progress_box, 0, 16))
        self.poll_timer.start()
        self.poll_job_progress()

    def restore_job_inputs(self, job: dict) -> None:
        if job.get("type", "segmentation") != "segmentation":
            return
        study_name = job.get("study", "")
        if study_name:
            self.study.setText(study_name)
        source_paths = [path for path in job.get("source_paths", []) if path]
        display_paths = source_paths or job.get("videos", [])
        self.videos = [Path(path) for path in display_paths]
        self.refresh_video_table(remote_job=not bool(source_paths))
        self.submit_button.setEnabled(False)
        self.update_workflow_steps(job.get("state", "queued"))

    def refresh_job_history(self) -> None:
        if not self.drive.service:
            return
        for job in self.jobs:
            if job.get("backend") == "local":
                continue
            progress_file_id = job.get("progress_file_id")
            if not progress_file_id:
                continue
            try:
                progress = self.drive.job_progress(progress_file_id)
            except Exception:
                LOGGER.warning(
                    "Could not refresh cloud job %s",
                    job.get("job_id", progress_file_id),
                    exc_info=True,
                )
                continue
            job["state"] = progress.get("state", job.get("state", "unknown"))
            job["message"] = progress.get("message", "")
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.update_job_history_metrics(job, progress)
        self._save_jobs()

    def clear_finished_job_history(self) -> int:
        retained = []
        removed = []
        for job in self.jobs:
            if job.get("state") in TERMINAL_JOB_STATES:
                removed.append(job)
            else:
                retained.append(job)
        if not removed:
            return 0
        hidden = set(self.settings.hidden_job_ids)
        hidden.update(
            identity
            for identity in (job_history_identity(job) for job in removed)
            if identity
        )
        self.settings.hidden_job_ids = sorted(hidden)
        self.jobs = retained
        if self.monitored_job_id and not any(
            job.get("job_id") == self.monitored_job_id for job in retained
        ):
            self.monitored_job_id = ""
        self._save_jobs()
        self.settings.save()
        return len(removed)

    @staticmethod
    def clear_diagnostic_logs() -> None:
        root_logger = logging.getLogger("stimtrace")
        for handler in tuple(root_logger.handlers):
            stream = getattr(handler, "stream", None)
            if stream is None or getattr(stream, "closed", True):
                continue
            handler.acquire()
            try:
                stream.seek(0)
                stream.truncate(0)
                stream.flush()
            finally:
                handler.release()
        if _FAULT_LOG_HANDLE is not None and not _FAULT_LOG_HANDLE.closed:
            _FAULT_LOG_HANDLE.seek(0)
            _FAULT_LOG_HANDLE.truncate(0)
            _FAULT_LOG_HANDLE.flush()
        for base_path in (APP_LOG_PATH, CRASH_LOG_PATH):
            for rotated in base_path.parent.glob(base_path.name + ".*"):
                if rotated.is_file():
                    rotated.unlink(missing_ok=True)

    def discover_drive_jobs(self) -> None:
        discovered = self.drive.discover_jobs(self.settings.drive_root_folder_id)
        self.merge_discovered_jobs(discovered)

    def merge_discovered_jobs(self, discovered: list[dict]) -> None:
        hidden = set(self.settings.hidden_job_ids)
        discovered = [
            job for job in discovered
            if job_history_identity(job) not in hidden
        ]
        existing = {job.get("folder_id"): job for job in self.jobs}
        for remote_job in discovered:
            local = existing.get(remote_job["folder_id"])
            if local:
                # Early cloud manifests did not carry a model name. Do not erase
                # the name recorded locally at submission when refreshing them.
                if not remote_job.get("model_name") and local.get("model_name"):
                    remote_job["model_name"] = local["model_name"]
                local.update(remote_job)
                self.update_job_history_metrics(local, remote_job)
            else:
                self.jobs.append(remote_job)
        self._save_jobs()

    def prompt_for_colab_runtime_failure(self) -> None:
        """Explain a worker-reported runtime stop once, with an immediate restart action."""
        failed_jobs = [
            job for job in self.jobs
            if job.get("state") == "failed"
            and "previous colab runtime stopped" in str(job.get("message", "")).casefold()
            and job.get("job_id") not in self.colab_crash_prompted_job_ids
        ]
        if not failed_jobs:
            return
        self.colab_crash_prompted_job_ids.update(
            str(job.get("job_id", "")) for job in failed_jobs
        )
        details = "\n".join(
            f"• {job.get('study') or job.get('job_id')}: {job.get('message', 'Unknown error')}"
            for job in failed_jobs[:6]
        )
        more = f"\n• …and {len(failed_jobs) - 6} more" if len(failed_jobs) > 6 else ""
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Critical)
        message.setWindowTitle("Colab runtime stopped")
        message.setText(
            "The Colab worker stopped unexpectedly while processing your job. "
            "This is a Colab runtime/hardware failure, not an annotation or input-file error."
        )
        message.setInformativeText(
            "Reported error(s):\n"
            f"{details}{more}\n\n"
            "Restart the Colab runtime, then run both StimTrace cells. "
            "After the worker is online, retry these failed jobs from Jobs. Queued jobs remain safe."
        )
        restart_button = message.addButton("Open Colab to restart", QMessageBox.AcceptRole)
        message.addButton("Close", QMessageBox.RejectRole)
        message.exec()
        if message.clickedButton() is restart_button:
            self.open_colab()

    def start_drive_job_discovery(self) -> None:
        if (
            self.shutdown_started
            or not self.drive.service
            or not self.settings.drive_root_folder_id
            or self.drive_discovery_thread is not None
        ):
            return
        thread = BackgroundFunctionThread(
            lambda: self.drive.discover_jobs(self.settings.drive_root_folder_id),
            self,
        )
        self.register_thread(thread, "drive-job-discovery")
        thread.succeeded.connect(self.finish_drive_job_discovery, Qt.QueuedConnection)
        thread.failed.connect(
            lambda message: LOGGER.warning("Drive job discovery failed: %s", message),
            Qt.QueuedConnection,
        )
        thread.finished.connect(self.finish_drive_discovery_thread, Qt.QueuedConnection)
        self.drive_discovery_thread = thread
        self.last_drive_discovery_started_at = time.monotonic()
        thread.start()

    @Slot(object)
    def finish_drive_job_discovery(self, discovered: object) -> None:
        if isinstance(discovered, list):
            self.merge_discovered_jobs(discovered)
            self.prompt_for_colab_runtime_failure()
            self.follow_active_cloud_segmentation_job()
            if self.job_history_page is not None:
                self.job_history_page.populate()

    @Slot()
    def finish_drive_discovery_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.drive_discovery_thread is thread:
            self.drive_discovery_thread = None

    def retry_job(self, job: dict) -> None:
        if job.get("backend") == "local":
            source_paths = [Path(path) for path in job.get("source_paths", [])]
            missing = [str(path) for path in source_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "These original recordings are unavailable:\n" + "\n".join(missing[:8])
                )
            self.set_compute_mode("local")
            self.study.setText(job.get("study", ""))
            self.videos = source_paths
            self.refresh_video_table()
            self.show_page("segment")
            self.submit_local()
            return
        if not self.drive.service:
            raise RuntimeError("Sign in before retrying a job.")
        self.drive.retry_job(
            job["folder_id"],
            job["progress_file_id"],
            job.get("preview_file_id", ""),
        )
        job["state"] = "queued"
        job["message"] = "Retry queued. Waiting for Colab compute."
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.reset_job_attempt_metrics(job)
        self._save_jobs()
        self.activate_job(job)

    def request_job_retry(self, job: dict) -> None:
        """Retry a job without ever performing Drive I/O on the GUI thread."""
        if job.get("backend") == "local":
            try:
                self.retry_job(job)
            except Exception as error:
                QMessageBox.critical(self, "Retry failed", str(error))
            return
        if self.job_retry_thread is not None:
            QMessageBox.information(
                self,
                "Retry already in progress",
                "StimTrace is already sending a retry request. You can continue using the app while it finishes.",
            )
            return
        if not self.drive.service:
            QMessageBox.information(self, "Sign in required", "Sign in before retrying a job.")
            return

        job_id = str(job.get("job_id", ""))
        folder_id = str(job.get("folder_id", ""))
        progress_file_id = str(job.get("progress_file_id", ""))
        preview_file_id = str(job.get("preview_file_id", ""))
        job["message"] = "Sending retry request to Google Drive..."
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_jobs()
        self.status.setText(job["message"])
        self.poll_timer.stop()

        def send_retry() -> str:
            self.drive.retry_job(folder_id, progress_file_id, preview_file_id)
            return job_id

        thread = BackgroundFunctionThread(send_retry, self)
        self.register_thread(thread, "drive-job-retry")
        thread.succeeded.connect(self.finish_job_retry, Qt.QueuedConnection)
        thread.failed.connect(self.fail_job_retry, Qt.QueuedConnection)
        thread.finished.connect(self.finish_job_retry_thread, Qt.QueuedConnection)
        self.job_retry_thread = thread
        thread.start()

    @Slot(object)
    def finish_job_retry(self, retried_job_id: object) -> None:
        job = next(
            (item for item in self.jobs if item.get("job_id") == str(retried_job_id)),
            None,
        )
        if job is None:
            return
        job["state"] = "queued"
        job["message"] = "Retry queued. Waiting for Colab compute."
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_jobs()
        self.activate_job(job)
        if self.job_history_page is not None:
            self.job_history_page.populate()

    @Slot(str)
    def fail_job_retry(self, message: str) -> None:
        self.poll_timer.start()
        QMessageBox.critical(self, "Retry failed", message)

    @Slot()
    def finish_job_retry_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.job_retry_thread is thread:
            self.job_retry_thread = None

    def cancel_job(self, job: dict) -> None:
        if job.get("backend") == "local":
            if (
                self.local_process is not None
                and job.get("job_id") == self.monitored_job_id
            ):
                if self.local_cancel_file:
                    self.local_cancel_file.write_text("cancel", encoding="ascii")
                job["state"] = "cancelling"
                job["message"] = "Cancellation requested. Waiting for the current batch to stop."
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save_jobs()
                self.reset_job_progress("Cancellation requested. Stopping local processing.")
                self.status.setText(job["message"])
                self.next_step.setText(job["message"])
                self.set_active_job_focus(job, "cancelling")
                return
            job["state"] = "cancelled"
            job["message"] = "Local job is no longer running."
            self._save_jobs()
            return
        if not self.drive.service:
            raise RuntimeError("Sign in before cancelling a job.")
        cancellation = self.drive.cancel_job(
            job["folder_id"],
            job["progress_file_id"],
        )
        state = cancellation.get("state", "cancelling")
        job["state"] = state
        job["message"] = cancellation.get("message", "Cancellation requested.")
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_jobs()
        if job.get("progress_file_id") == self.active_progress_file_id:
            if state in TERMINAL_JOB_STATES:
                self.poll_job_progress()
                return
            self.reset_job_progress("Cancellation requested. Waiting for Colab.")
            self.next_step.setText("Cancellation requested. Waiting for Colab to stop the job.")
            self.update_workflow_steps("cancelling")
            self.poll_timer.start()

    def sign_in(self) -> None:
        if self.sign_in_thread is not None or self.restore_session_thread is not None:
            return
        oauth_client_config = self.select_oauth_client_config()
        if oauth_client_config is None:
            return
        self.restore_session_timer.stop()
        self.sign_button.setEnabled(False)
        self.sign_button.setText("Signing in...")
        self.set_auth_status("Opening Google authorization...", signed_in=False)
        self.set_colab_waiting(True, "Waiting for Google authorization...")
        self.next_step.setText(
            "Complete the Google authorization in your browser. StimTrace will then prepare Colab."
        )

        def authorize_and_prepare() -> str:
            email = self.drive.sign_in(oauth_client_config_path=oauth_client_config)
            self.drive.ensure_workspace(self.settings)
            return email

        thread = BackgroundFunctionThread(authorize_and_prepare, self)
        self.register_thread(thread, "google-sign-in")
        thread.succeeded.connect(self.finish_manual_sign_in, Qt.QueuedConnection)
        thread.failed.connect(self.fail_manual_sign_in, Qt.QueuedConnection)
        thread.finished.connect(self.finish_manual_sign_in_thread, Qt.QueuedConnection)
        self.sign_in_thread = thread
        thread.start()

    def select_oauth_client_config(self) -> Path | None:
        """Return a user-managed OAuth Desktop client config without copying it."""
        configured_path = Path(self.settings.oauth_client_config_path).expanduser()
        if configured_path.is_file():
            return configured_path

        dialog = QDialog(self)
        dialog.setWindowTitle("Set up optional Google Drive")
        dialog.setMinimumWidth(650)
        layout = QVBoxLayout(dialog)
        title = QLabel("Set up Google Drive and Colab in five steps")
        title.setStyleSheet("font-size: 17px; font-weight: 700;")
        layout.addWidget(title)
        introduction = QLabel(
            "Cloud processing is optional. StimTrace does not distribute a shared Google "
            "OAuth configuration, so this one-time setup keeps your Google Cloud project "
            "and Drive access under your control."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        setup_steps = (
            (
                "1. Create or select a Google Cloud project",
                "Open Google Cloud Console and create a project, for example ‘StimTrace’. "
                "Use that project for the remaining steps.",
                "Open Google Cloud Console",
                "https://console.cloud.google.com/projectselector2/home/dashboard",
            ),
            (
                "2. Configure the OAuth audience",
                "Open Google Auth Platform. Configure Branding, keep the audience External, "
                "and add the Google account you will use as a test user.",
                "Open Google Auth Platform",
                "https://console.cloud.google.com/auth/overview",
            ),
            (
                "3. Enable Google Drive API",
                "Open the Google Drive API page and select Enable for the project you created.",
                "Open Google Drive API",
                "https://console.cloud.google.com/apis/library/drive.googleapis.com",
            ),
            (
                "4. Create and download a Desktop-app OAuth client",
                "Open Clients, create an OAuth client, choose Desktop app, then download the "
                "generated JSON file. Keep it in a private location; do not place it in the "
                "StimTrace folder or share it.",
                "Open OAuth clients",
                "https://console.cloud.google.com/auth/clients",
            ),
        )
        for heading, text, button_text, url in setup_steps:
            card = QFrame()
            card.setObjectName("workflowStep")
            card_layout = QHBoxLayout(card)
            description = QLabel(f"<b>{heading}</b><br>{text}")
            description.setWordWrap(True)
            card_layout.addWidget(description, 1)
            open_page = QPushButton(button_text)
            open_page.clicked.connect(
                lambda _checked=False, target=url: QDesktopServices.openUrl(QUrl(target))
            )
            card_layout.addWidget(open_page)
            layout.addWidget(card)

        step_five = QLabel("<b>5. Select the downloaded JSON file</b><br>"
                            "StimTrace stores only the path on this computer and never copies "
                            "or uploads the file.")
        step_five.setWordWrap(True)
        layout.addWidget(step_five)
        buttons = QHBoxLayout()
        select_file = QPushButton("Select OAuth Desktop-client JSON")
        select_file.setProperty("role", "primary")
        cancel = QPushButton("Use local analysis only")
        buttons.addWidget(select_file)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        selected_path: Path | None = None

        def choose_file() -> None:
            nonlocal selected_path
            filename, _ = QFileDialog.getOpenFileName(
                dialog,
                "Select your Google OAuth Desktop client JSON file",
                str(Path.home()),
                "JSON files (*.json)",
            )
            if not filename:
                return
            candidate = Path(filename)
            try:
                configuration = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                QMessageBox.warning(
                    dialog,
                    "Invalid OAuth configuration",
                    f"StimTrace could not read the selected JSON file:\n{error}",
                )
                return
            installed = configuration.get("installed") if isinstance(configuration, dict) else None
            if not isinstance(installed, dict) or not installed.get("client_id"):
                QMessageBox.warning(
                    dialog,
                    "OAuth Desktop client required",
                    "Select the JSON file for a Google OAuth client created with the "
                    "Desktop app application type.",
                )
                return
            selected_path = candidate.resolve()
            self.settings.oauth_client_config_path = str(selected_path)
            self.settings.save()
            dialog.accept()

        select_file.clicked.connect(choose_file)
        cancel.clicked.connect(dialog.reject)
        return selected_path if dialog.exec() == QDialog.Accepted else None

    @Slot(object)
    def finish_manual_sign_in(self, email: object) -> None:
        self.apply_signed_in_state(str(email), open_colab=True)
        QTimer.singleShot(1200, self.start_drive_job_discovery)

    @Slot(str)
    def fail_manual_sign_in(self, message: str) -> None:
        LOGGER.error("Google sign-in failed: %s", message)
        self.set_colab_waiting(False)
        self.set_auth_status("Google sign-in failed.", signed_in=False)
        if self.training_page is not None:
            self.training_page.update_training_account_controls()
            self.training_page.update_training_workflow()
        QMessageBox.critical(self, "Sign-in failed", message)

    @Slot()
    def finish_manual_sign_in_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.sign_in_thread is thread:
            self.sign_in_thread = None
        self.sign_button.setText("Sign in")
        self.sign_button.setEnabled(not bool(self.drive.service))

    def apply_signed_in_state(self, email: str, *, open_colab: bool) -> None:
        self.set_auth_status(
            f"Signed in as {email}. Your personal StimTrace Drive workspace is ready.",
            signed_in=True,
        )
        self.sign_button.setText("Sign in")
        self.sign_button.setEnabled(False)
        self.sign_out_button.setEnabled(True)
        if self.training_page is not None:
            self.training_page.update_training_account_controls()
            self.training_page.update_training_workflow()
        self.model_status.setText(f"Segmentation model: {self.settings.selected_model_name}")
        self.compute_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.add_folder_button.setEnabled(True)
        self.submit_button.setEnabled(bool(self.videos))
        self.update_workflow_steps()
        self.next_step.setText(
            "Step 2: Open the Colab notebook, select a GPU, and run both cells. "
            "Wait for the worker-ready message."
        )
        active = self._best_active_job()
        if active and self.settings.compute_mode == "cloud":
            self.activate_job(active, quiet=True)
        if self.settings.compute_mode == "local":
            self.set_compute_mode("local", save=False)
        elif open_colab:
            self.open_colab()
        self.poll_worker_status()
        self.worker_timer.start()

    def open_settings(self) -> None:
        dialog = SettingsDialog(
            self.settings,
            self,
            cloud_hardware=self.worker_hardware if self.settings.compute_mode == "cloud" else None,
            kalman_benchmark_combinations=self.kalman_benchmark_combinations,
            on_kalman_benchmark_change=self.set_kalman_benchmark_combinations,
            calibration_videos=self.videos,
        )
        if dialog.exec() == QDialog.Accepted:
            self.model_status.setText(f"Segmentation model: {self.settings.selected_model_name}")
            if self.drive.service:
                try:
                    self.drive.ensure_workspace(self.settings)
                except Exception as error:
                    QMessageBox.critical(self, "Could not update settings", str(error))
    def set_colab_waiting(self, active: bool, message: str = "") -> None:
        self.colab_wait_progress.setVisible(active)
        self.colab_wait_label.setVisible(active)
        if message:
            self.colab_wait_label.setText(message)

    def open_colab(self) -> None:
        if self.colab_connection_in_progress:
            return
        try:
            if not self.settings.model_profiles.get(
                self.settings.selected_model_name, ""
            ):
                raise RuntimeError(
                    f"{self.settings.selected_model_name} is available only on this computer. "
                    "Select a Drive-backed model on the Models page before opening Colab."
                )
            needs_workspace_update = (
                not self.settings.drive_root_folder_id
                or not self.settings.notebook_file_id
                or self.settings.notebook_version != NOTEBOOK_VERSION
            )
            if not needs_workspace_update:
                self.finish_open_colab()
                return
            self.colab_connection_in_progress = True
            self.resume_job_poll_after_colab = self.poll_timer.isActive()
            self.poll_timer.stop()
            self.worker_timer.stop()
            self.compute_button.setEnabled(False)
            self.compute_button.setText("Preparing Colab...")
            self.set_colab_waiting(True, "Updating the Colab notebook...")
            self.status.setText("Contacting Google Drive and preparing the Colab notebook...")
            self.next_step.setText(
                "StimTrace is updating your Colab notebook. This can take a few seconds."
            )
            thread = BackgroundFunctionThread(
                lambda: self.drive.ensure_workspace(self.settings),
                self,
            )
            self.register_thread(thread, "colab-workspace-update")
            thread.succeeded.connect(
                self.workspace_update_succeeded,
                Qt.QueuedConnection,
            )
            thread.failed.connect(self.fail_open_colab, Qt.QueuedConnection)
            thread.finished.connect(
                self.finish_colab_connection_thread,
                Qt.QueuedConnection,
            )
            self.colab_connection_thread = thread
            thread.start()
        except Exception as error:
            QMessageBox.critical(self, "Could not open Colab", str(error))

    @Slot(object)
    def workspace_update_succeeded(self, _result: object) -> None:
        self.finish_open_colab()

    def finish_open_colab(self) -> None:
        try:
            if not self.settings.notebook_file_id:
                raise FileNotFoundError(
                    "The StimTrace Compute notebook could not be uploaded. "
                    "Restart the application from its installation folder."
                )
            webbrowser.open(colab_notebook_url(self.settings.notebook_file_id))
        except Exception as error:
            self.fail_open_colab(str(error))
            return
        self.colab_opened = True
        self.colab_connection_in_progress = False
        if self.worker_online:
            self.set_colab_waiting(False)
            self.status.setText(
                f"Colab worker is online on {self.worker_device or 'the selected runtime'}. "
                "If the notebook shows Connect, reconnect the tab; do not run the cells again."
            )
            self.next_step.setText(
                "The existing Colab worker is still running. Progress remains visible in StimTrace."
            )
        else:
            self.set_colab_waiting(
                True,
                "Waiting for Colab worker. Run both notebook cells.",
            )
            self.status.setText("Colab opened. Waiting for the StimTrace worker...")
            self.next_step.setText(
                "In Colab, select a GPU and run both cells. Continue when StimTrace reports that the worker is online."
            )
        self.update_workflow_steps()

    def fail_open_colab(self, message: str) -> None:
        self.colab_connection_in_progress = False
        self.set_colab_waiting(False)
        self.compute_button.setEnabled(bool(self.drive.service))
        self.update_workflow_steps()
        QMessageBox.critical(self, "Could not open Colab", message)

    @Slot()
    def finish_colab_connection_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.colab_connection_thread is thread:
            self.colab_connection_thread = None
        self.colab_connection_in_progress = False
        if self.drive.service:
            self.worker_timer.start()
        if self.resume_job_poll_after_colab:
            self.poll_timer.start()
        self.resume_job_poll_after_colab = False
        self.compute_button.setEnabled(bool(self.drive.service))
        self.update_workflow_steps()

    @staticmethod
    def bundled_model_path() -> Path:
        candidates = [
            RESOURCE_DIR / "model.pth",
            RESOURCE_DIR / "unet_multitask_center_ellipse_512x512 - Backup.pth",
            SOURCE_DIR.parent / "unet_multitask_center_ellipse_512x512 - Backup.pth",
        ]
        return next((path for path in candidates if path.is_file()), candidates[0])

    @staticmethod
    def safe_path_component(value: str) -> str:
        return normalize_path_component(value, fallback="study")

    def resolve_local_model(self) -> Path:
        model_name = self.settings.selected_model_name
        if model_name == "Default pillar":
            bundled = self.bundled_model_path()
            if bundled.is_file():
                return bundled
        saved = self.settings.local_model_paths.get(model_name, "")
        if saved and Path(saved).is_file():
            return Path(saved)
        model_file_id = self.settings.model_profiles.get(model_name, "")
        if not model_file_id or not self.drive.service:
            raise FileNotFoundError(
                f"The checkpoint for {model_name} is not stored on this computer.\n\n"
                "Sign in once so StimTrace can download it from your Drive model catalog, "
                "or import the .pth checkpoint from the Models page."
            )
        cache = USER_DATA_DIR / "model_cache"
        cache.mkdir(exist_ok=True)
        target = cache / (
            f"{self.safe_path_component(model_name)}_{model_file_id[-8:]}.pth"
        )
        self.status.setText(f"Downloading {model_name} for local processing...")
        QApplication.processEvents()
        target.write_bytes(self.drive.file_bytes(model_file_id))
        self.settings.local_model_paths[model_name] = str(target)
        self.settings.save()
        return target

    def submit_local(self, submitted_at: str | None = None) -> None:
        if self.local_process is not None and self.local_process.isRunning():
            QMessageBox.information(
                self,
                "Local analysis already running",
                "Wait for the current local job to finish or cancel it before starting another.",
            )
            return
        stems = [video.stem.lower() for video in self.videos]
        if len(stems) != len(set(stems)):
            QMessageBox.warning(
                self,
                "Duplicate recording names",
                "Local analysis requires unique recording file names because result files use "
                "the recording name. Rename the duplicates and add them again.",
            )
            return
        try:
            model_path = self.resolve_local_model()
        except Exception as error:
            if (
                self.settings.selected_model_name != "Default pillar"
                and QMessageBox.question(
                    self,
                    "Local model unavailable",
                    f"{error}\n\nLocate the matching .pth checkpoint on this computer?",
                ) == QMessageBox.Yes
            ):
                filename, _ = QFileDialog.getOpenFileName(
                    self,
                    f"Locate {self.settings.selected_model_name}",
                    "",
                    "PyTorch model (*.pth)",
                )
                if not filename:
                    return
                model_path = Path(filename)
                self.settings.local_model_paths[
                    self.settings.selected_model_name
                ] = str(model_path.resolve())
                self.settings.save()
            else:
                QMessageBox.critical(self, "Local model unavailable", str(error))
                return

        study_name = self.study.text().strip()
        job_id = f"{self.safe_path_component(study_name)}_{uuid.uuid4().hex[:8]}"
        source_folders = {video.parent for video in self.videos}
        recording_folder = (
            next(iter(source_folders))
            if len(source_folders) == 1
            else self.videos[0].parent
        )
        output = recording_folder / "results" / job_id
        cancel_file = USER_DATA_DIR / f".cancel_{job_id}"
        cancel_file.unlink(missing_ok=True)
        self.local_cancel_file = cancel_file
        control_file = USER_DATA_DIR / f".control_{job_id}.json"
        control_file.unlink(missing_ok=True)
        self.local_control_file = control_file
        self.write_local_runtime_control()
        video_names = [video.name for video in self.videos]
        submitted_job = {
            "job_id": job_id,
            "study": study_name,
            "type": "segmentation",
            "backend": "local",
            "folder_id": "",
            "progress_file_id": "",
            "videos": video_names,
            "source_paths": [str(video) for video in self.videos],
            "local_results_path": str(output),
            "state": "queued",
            "message": "Starting local analysis.",
            "created_at": submitted_at or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "submitted_at": submitted_at or datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.settings.selected_model_name,
            "device_name": "Detecting hardware",
            "kalman_benchmark": [
                combination["name"]
                for combination in self.kalman_benchmark_combinations
            ],
        }
        self.jobs.append(submitted_job)
        self._save_jobs()
        self.monitored_job_id = job_id
        self.active_progress_file_id = ""
        self.local_terminal_event = ""

        parameters = self.effective_job_parameters()
        arguments = [
            "--model", str(model_path),
            "--output", str(output),
            "--parameters-json", json.dumps(parameters, separators=(",", ":")),
            "--device", self.settings.local_device_preference,
            "--cpu-threads", str(self.settings.local_cpu_threads),
            "--batch-size", str(self.settings.local_inference_batch_size),
            "--postprocess-workers", str(self.settings.local_postprocess_workers),
            "--cancel-file", str(cancel_file),
            "--control-file", str(control_file),
            "--job-id", job_id,
        ]
        if not self.settings.local_generate_overlays and not self.kalman_benchmark_combinations:
            arguments.append("--no-overlays")
        arguments.extend(str(video) for video in self.videos)
        program, arguments = self.local_worker_command(arguments)
        process = LocalProcessThread(program, arguments, self)
        self.register_thread(process, "local-segmentation-process")
        process.event_received.connect(self.handle_local_event, Qt.QueuedConnection)
        process.finished.connect(self.finish_local_process, Qt.QueuedConnection)
        self.local_process = process

        self.prepare_new_submission_progress()
        self.progress_box.setVisible(True)
        self.setup_expanded_during_job = False
        self.monitoring_label.setText(f"Local job: {job_id} - starting")
        self.monitoring_label.setVisible(True)
        self.file_progress_label.setText("Loading segmentation model...")
        self.file_progress_label.setVisible(True)
        self.file_progress.setRange(0, 100)
        self.file_progress.setValue(0)
        self.file_progress.setVisible(True)
        self.job_progress_label.setText("Total job: starting local worker")
        self.job_progress_label.setVisible(True)
        self.job_progress_bar.setRange(0, 1000)
        self.job_progress_bar.setValue(0)
        self.job_progress_bar.setVisible(True)
        self.upload_progress.setVisible(False)
        self.set_active_job_focus(submitted_job, "queued")
        self.submit_button.setEnabled(False)
        self.cloud_mode_button.setEnabled(False)
        self.local_mode_button.setEnabled(False)
        for row in range(self.video_table.rowCount()):
            self.set_video_status(row, "Queued")
        self.status.setText("Starting local analysis...")
        self.next_step.setText(
            "Local analysis is running in the background. Progress appears below."
        )
        self.update_workflow_steps("analyzing")
        process.start()

    def handle_local_event(self, event: dict[str, Any]) -> None:
        job = self.current_job()
        if not job:
            return
        event_name = event.get("event", "")
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        if event_name == "starting":
            self.mark_job_processing_started(job, event.get("processing_started_at"))
            job["state"] = "analyzing"
            job["device_name"] = event.get("device_name", event.get("device", "local"))
            job["message"] = f"Running locally on {job['device_name']}."
            self.status.setText(job["message"])
            self.monitoring_label.setText(
                f"Local job: {job.get('job_id', '')} - analyzing on {job['device_name']}"
            )
            self.set_active_job_focus(job, "analyzing")
        elif event_name == "progress":
            stage = event.get("stage", "segmenting")
            current = int(event.get("current_frame", 0))
            total = max(1, int(event.get("current_file_frames", 1)))
            file_name = event.get("file_name", "Current video")
            file_index = event.get("file_index", "?")
            file_count = event.get("file_count", "?")
            stage_text = {
                "segmenting": "Segmenting",
                "overlay": "Writing overlay",
                "workbook": "Creating combined results",
            }.get(stage, stage.replace("_", " ").title())
            file_fraction = max(0.0, min(1.0, float(event.get("file_progress_fraction", 0))))
            job_fraction, processed_frames, total_frames = self.frame_progress_fraction(event)
            self.file_progress_label.setText(
                f"{stage_text}: file {file_index}/{file_count} - {file_name}, "
                f"frame {current}/{total}"
            )
            self.file_progress.setValue(round(file_fraction * 100))
            self.job_progress_bar.setValue(round(job_fraction * 1000))
            eta = self._format_eta(event.get("estimated_remaining_seconds"))
            self.job_progress_label.setText(
                self.frame_progress_text(
                    job_fraction,
                    processed_frames,
                    total_frames,
                    stage,
                    eta,
                )
            )
            job["state"] = "analyzing"
            job["message"] = self.file_progress_label.text()
            self.update_job_processed_frames(job, event)
            for row, video in enumerate(self.videos):
                if video.name == file_name:
                    self.set_video_status(row, f"{round(file_fraction * 100)}%")
                    break
        elif event_name == "settings_applied":
            job["runtime_settings"] = {
                "cpu_threads": event.get("cpu_threads"),
                "inference_batch_size": event.get("inference_batch_size"),
                "postprocess_workers": event.get("cpu_postprocess_workers"),
                "generate_overlays": event.get("generate_overlays"),
            }
            self.monitoring_label.setText(
                "Runtime settings applied: "
                f"{event.get('cpu_threads', '?')} CPU threads, "
                f"batch {event.get('inference_batch_size', '?')}, "
                f"{event.get('cpu_postprocess_workers', '?')} postprocess workers."
            )
        elif event_name == "complete":
            self.local_terminal_event = "complete"
            job["state"] = "complete"
            job["local_results_path"] = event.get(
                "output", job.get("local_results_path", "")
            )
            job["device_name"] = event.get("device_name", job.get("device_name", "local"))
            job["message"] = f"Complete. Results saved to {job['local_results_path']}."
            self.update_job_processed_frames(job, event)
        elif event_name == "cancelled":
            self.local_terminal_event = "cancelled"
            job["state"] = "cancelled"
            job["message"] = "Local analysis cancelled."
        elif event_name == "failed":
            self.local_terminal_event = "failed"
            job["state"] = "failed"
            job["message"] = f"Local analysis failed: {event.get('error', 'Unknown error')}"
        self._save_jobs()

    @Slot()
    def finish_local_process(self) -> None:
        process = self.sender()
        if not isinstance(process, LocalProcessThread):
            return
        process.wait()
        exit_code = process.exit_code
        output_tail = process.output_tail
        job = self.current_job()
        if job and not self.local_terminal_event:
            self.local_terminal_event = "failed"
            job["state"] = "failed"
            job["message"] = (
                f"Local worker stopped unexpectedly (exit code {exit_code})."
                + (f"\n{output_tail}" if output_tail else "")
            )
            self._save_jobs()
        if self.local_process is process:
            self.local_process = None
        if self.local_cancel_file:
            self.local_cancel_file.unlink(missing_ok=True)
        self.local_cancel_file = None
        if self.local_control_file:
            self.local_control_file.unlink(missing_ok=True)
        self.local_control_file = None
        self.cloud_mode_button.setEnabled(True)
        self.local_mode_button.setEnabled(True)
        state = job.get("state", "failed") if job else "failed"
        self.submit_button.setEnabled(bool(self.videos))
        if state == "complete":
            self.file_progress.setValue(100)
            self.job_progress_bar.setValue(1000)
            self.file_progress_label.setText("All selected videos processed")
            self.job_progress_label.setText("Local analysis complete")
            self.status.setText(job["message"])
            self.next_step.setText(job["message"])
            for row in range(self.video_table.rowCount()):
                self.set_video_status(row, "Complete")
            self.offer_signal_analysis(job)
        elif state == "cancelled":
            self.reset_job_progress("Local analysis cancelled")
            self.status.setText("Local analysis cancelled.")
            self.next_step.setText("Local analysis cancelled. You can change the inputs and run again.")
            for row in range(self.video_table.rowCount()):
                self.set_video_status(row, "Cancelled")
        else:
            self.status.setText(job.get("message", "Local analysis failed.") if job else "Local analysis failed.")
            self.next_step.setText(self.status.text())
            for row in range(self.video_table.rowCount()):
                self.set_video_status(row, "Failed")
            QMessageBox.critical(self, "Local analysis failed", self.status.text())
        self.set_active_job_focus(job, state)
        self.update_workflow_steps(state)
        if self.job_history_page is not None:
            self.job_history_page.populate()

    def submit(self) -> None:
        if not self.validate_submission_inputs():
            return
        submitted_at = local_now_iso()
        if self.settings.compute_mode == "local":
            self.submit_local(submitted_at)
            return
        if not self.validate_cloud_submission():
            return
        video_names = [video.name for video in self.videos]
        duplicate = self.find_duplicate_submission(video_names)
        if duplicate:
            if not duplicate.get("source_paths"):
                duplicate["source_paths"] = [str(video) for video in self.videos]
                self._save_jobs()
            self.activate_job(duplicate)
            QMessageBox.information(
                self,
                "Job already submitted",
                f"{duplicate['job_id']} already contains these videos and is "
                f"{duplicate.get('state', 'active')}. StimTrace resumed monitoring it instead of uploading a duplicate.",
            )
            return
        self.start_cloud_submission_when_safe(video_names, submitted_at)

    def start_cloud_submission_when_safe(
        self, video_names: list[str], submitted_at: str
    ) -> None:
        """Do not begin an upload while the shared Drive client is checking a job."""
        self.prepare_new_submission_progress()
        if (
            self.cloud_submission_thread is not None
            or self.job_progress_thread is not None
            or self.drive_discovery_thread is not None
        ):
            self.pending_cloud_submission = (video_names, submitted_at)
            self.submission_in_progress = True
            self.submit_button.setEnabled(False)
            self.status.setText("Waiting for the current Drive status check before uploading...")
            self.next_step.setText("Preparing the new queued job. Existing job monitoring will resume after upload.")
            QTimer.singleShot(150, self.resume_pending_cloud_submission)
            return
        self.pending_cloud_submission = None
        self.submit_cloud_job(video_names, submitted_at)

    def resume_pending_cloud_submission(self) -> None:
        pending = self.pending_cloud_submission
        if pending is None:
            return
        if self.job_progress_thread is not None or self.drive_discovery_thread is not None:
            QTimer.singleShot(150, self.resume_pending_cloud_submission)
            return
        self.pending_cloud_submission = None
        self.submit_cloud_job(*pending)

    def validate_submission_inputs(self) -> bool:
        if not self.study.text().strip():
            QMessageBox.warning(
                self,
                "Study name required",
                "Enter a study name before submitting the analysis.",
            )
            self.study.setFocus()
            return False
        if not self.videos:
            QMessageBox.warning(
                self,
                "No videos selected",
                "Choose at least one video before submitting the analysis.",
            )
            return False
        missing = [str(video) for video in self.videos if not video.is_file()]
        if missing:
            QMessageBox.warning(
                self,
                "Video files unavailable",
                "These recordings cannot be found:\n\n" + "\n".join(missing[:8]),
            )
            return False
        duplicate_names = duplicate_output_stems(self.videos)
        if duplicate_names:
            QMessageBox.warning(
                self,
                "Duplicate recording names",
                "Selected recordings must have unique filename stems because result files are "
                "identified without the video extension. Rename these recordings before submitting:\n\n"
                + "\n".join(duplicate_names[:8]),
            )
            return False
        return True

    def validate_cloud_submission(self) -> bool:
        if self.worker_outdated:
            message = QMessageBox(self)
            message.setIcon(QMessageBox.Warning)
            message.setWindowTitle("Colab update required")
            message.setText(
                "The running Colab worker is an older version and cannot accept this job."
            )
            message.setInformativeText(
                "Open Colab, restart the runtime, and run both cells in the updated notebook. "
                "Submit the analysis after StimTrace reports that the worker is online."
            )
            open_button = message.addButton("Open Colab", QMessageBox.AcceptRole)
            message.addButton(QMessageBox.Cancel)
            message.exec()
            if message.clickedButton() is open_button:
                self.open_colab()
            return False
        if not self.settings.model_profiles.get(
            self.settings.selected_model_name, ""
        ):
            QMessageBox.warning(
                self,
                "Cloud model unavailable",
                f"{self.settings.selected_model_name} is available only on this computer. "
                "Select a Drive-backed model on the Models page before submitting to Colab.",
            )
            return False
        return True

    def find_duplicate_submission(self, video_names: list[str]) -> dict | None:
        return next(
            (
                job for job in reversed(self.jobs)
                if job.get("study", "").strip().lower() == self.study.text().strip().lower()
                and job.get("videos", []) == video_names
                and job.get("state") not in TERMINAL_JOB_STATES
            ),
            None,
        )

    def submit_cloud_job(self, video_names: list[str], submitted_at: str) -> None:
        previous_job = self.current_job()
        previous_state = previous_job.get("state", "queued") if previous_job else ""
        self.submission_in_progress = True
        self.poll_timer.stop()
        # A status read for the previous job may have completed while this
        # submission waited for the shared Drive client. Clear it again at the
        # actual upload boundary so stale 100% bars cannot reappear.
        self.prepare_new_submission_progress()
        videos = list(self.videos)
        study = self.study.text().strip()
        parameters = self.effective_job_parameters()
        model_name = self.settings.selected_model_name
        overlay_mode = self.settings.cloud_overlay_mode
        kalman_benchmark = list(self.kalman_benchmark_combinations)
        self.progress_box.setVisible(True)
        self.page.ensureWidgetVisible(self.progress_box, 0, 16)
        self.set_pending_submission_focus()
        self.upload_progress.setRange(0, len(videos) * 100)
        self.upload_progress.setValue(0); self.upload_progress.setVisible(True)
        self.start_upload_attention(
            f"Uploading {len(videos)} video{'s' if len(videos) != 1 else ''} to Google Drive. Keep StimTrace open."
        )
        self.status.setText("Uploading videos in the background...")
        self.next_step.setText("Uploading videos to Google Drive. StimTrace remains usable while this finishes.")

        request = CloudSubmissionRequest(videos, study, self.settings, parameters)

        def upload() -> dict:
            def show_progress(*payload):
                if len(payload) == 1 and isinstance(payload[0], dict):
                    thread.progress.emit(payload[0])
                elif len(payload) == 3:
                    thread.progress.emit(tuple(payload))
            return upload_cloud_job(self.drive, request, show_progress)

        thread = BackgroundFunctionThread(upload, self)
        self.register_thread(thread, "drive-cloud-submission")
        thread.progress.connect(self.update_cloud_upload_progress, Qt.QueuedConnection)
        thread.succeeded.connect(self.finish_cloud_submission, Qt.QueuedConnection)
        thread.failed.connect(self.fail_cloud_submission, Qt.QueuedConnection)
        thread.finished.connect(self.finish_cloud_submission_thread, Qt.QueuedConnection)
        self.cloud_submission_context = {
            "video_names": video_names, "videos": videos, "study": study,
            "submitted_at": submitted_at, "model_name": model_name,
            "overlay_mode": overlay_mode,
            "kalman_benchmark": kalman_benchmark, "previous_job": previous_job,
            "previous_state": previous_state,
        }
        self.cloud_submission_thread = thread
        thread.start()

    @Slot(object)
    def update_cloud_upload_progress(self, payload: object) -> None:
        if isinstance(payload, dict) and payload.get("phase") == "preparing":
            self.upload_progress.setRange(0, 0)
            self.upload_progress.setFormat("Preparing Google Drive workspace...")
            self.update_upload_attention(
                "Preparing the Google Drive workspace before video upload. Keep StimTrace open."
            )
            self.status.setText("Preparing Google Drive workspace...")
            self.next_step.setText("Preparing Google Drive workspace, then uploading videos.")
            return
        if isinstance(payload, dict) and payload.get("phase") == "retrying":
            video = str(payload.get("video", "video"))
            attempt = payload.get("attempt", "?")
            seconds = payload.get("wait_seconds", "?")
            self.upload_progress.setRange(0, 0)
            self.upload_progress.setFormat(f"Reconnecting to Drive for {video}...")
            self.update_upload_attention(
                f"Drive connection interrupted while uploading {video}. Retrying {attempt}/4 in {seconds}s."
            )
            return
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        index, fraction, filename = payload
        context = self.cloud_submission_context
        videos = context.get("videos", [])
        try:
            index, fraction = int(index), float(fraction)
        except (TypeError, ValueError):
            return
        self.upload_progress.setRange(0, max(1, len(videos) * 100))
        percent = round(fraction * 100)
        self.upload_progress.setValue(index * 100 + percent)
        self.upload_progress.setFormat(f"Video {index + 1}/{len(videos)}: {filename} - {percent}%")
        self.update_upload_attention(
            f"Uploading video {index + 1} of {len(videos)}: {filename} - {percent}%. Keep StimTrace open."
        )
        self.set_video_status(index, "Uploaded" if fraction >= 1 else f"Uploading {percent}%")

    @Slot(object)
    def finish_cloud_submission(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        context = dict(self.cloud_submission_context)
        self.active_progress_file_id = str(result["progress_file_id"])
        submitted_job = make_submitted_segmentation_job(result, context)
        self.jobs.append(submitted_job)
        self._save_jobs()
        self.monitored_job_id = submitted_job["job_id"]
        self.monitoring_label.setText(f"Monitoring job: {submitted_job['job_id']} (queued)")
        self.monitoring_label.setVisible(True)
        self.file_progress_label.setText("Current file: waiting for this job to reach Colab")
        self.file_progress_label.setVisible(True)
        self.file_progress.setRange(0, 100); self.file_progress.setValue(0); self.file_progress.setFormat("%p%")
        self.file_progress.setVisible(True)
        self.job_progress_label.setText("Total job: queued - an earlier Drive job may be processed first")
        self.job_progress_label.setVisible(True)
        self.job_progress_bar.setRange(0, 1000); self.job_progress_bar.setValue(0); self.job_progress_bar.setFormat("%p%")
        self.job_progress_bar.setVisible(True)
        self.upload_progress.setValue(len(context["videos"]) * 100)
        self.upload_progress.setVisible(False)
        self.stop_upload_attention()
        for row in range(self.video_table.rowCount()):
            self.set_video_status(row, "Queued")
        self.status.setText(f"Submitted {submitted_job['job_id']}. Waiting for Colab compute.")
        self.next_step.setText("Job queued: keep Colab running. StimTrace will update this screen as processing advances.")
        self.update_workflow_steps("queued")
        self.setup_expanded_during_job = False
        self.submission_in_progress = False
        self.set_active_job_focus(submitted_job, "queued")
        self.submit_button.setEnabled(False); self.poll_timer.start()

    @Slot(str)
    def fail_cloud_submission(self, error: str) -> None:
        context = dict(self.cloud_submission_context)
        self.submission_in_progress = False
        self.stop_upload_attention()
        self.upload_progress.setFormat("Upload failed")
        for row in range(self.video_table.rowCount()):
            if self.video_table.item(row, 4).text() != "Uploaded":
                self.set_video_status(row, "Upload failed")
        previous_job = context.get("previous_job")
        previous_state = context.get("previous_state", "")
        self.set_active_job_focus(previous_job, previous_state)
        if previous_job and previous_state in ACTIVE_JOB_STATES:
            self.poll_timer.start()
        QMessageBox.critical(self, "Submission failed", error)

    @Slot()
    def finish_cloud_submission_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.cloud_submission_thread is thread:
            self.cloud_submission_thread = None
            self.cloud_submission_context = {}

    def poll_job_progress(self) -> None:
        if (
            self.submission_in_progress
            or not self.active_progress_file_id
            or self.job_progress_thread is not None
            or self.job_retry_thread is not None
        ):
            return
        # A full Drive history scan can take minutes when many jobs exist. It must
        # never precede or repeatedly starve this three-second active-job poll.
        # The worker heartbeat identifies the exact running job; history discovery
        # is reserved for an unknown worker job or an explicit history refresh.
        if self.drive_discovery_thread is not None:
            return
        progress_file_id = self.active_progress_file_id
        active_job = self._job_by_progress_id(progress_file_id)
        folder_id = active_job.get("folder_id", "") if active_job else ""
        thread = BackgroundFunctionThread(
            lambda: self.fetch_remote_job_progress(progress_file_id, folder_id),
            self,
        )
        self.register_thread(thread, "drive-job-progress-poll")
        thread.succeeded.connect(self.receive_remote_job_progress, Qt.QueuedConnection)
        thread.failed.connect(self.ignore_remote_job_progress_error, Qt.QueuedConnection)
        thread.finished.connect(self.finish_job_progress_thread, Qt.QueuedConnection)
        self.job_progress_thread = thread
        thread.start()

    def fetch_remote_job_progress(self, progress_file_id: str, folder_id: str) -> dict:
        return fetch_cloud_job_progress(self.drive, progress_file_id, folder_id)

    @Slot(object)
    def receive_remote_job_progress(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        if result.get("progress_file_id") != self.active_progress_file_id:
            return
        progress = result.get("progress")
        if isinstance(progress, dict):
            self.apply_job_progress(progress)

    @Slot(str)
    def ignore_remote_job_progress_error(self, _message: str) -> None:
        LOGGER.warning("Cloud job progress poll failed: %s", _message)

    @Slot()
    def finish_job_progress_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.job_progress_thread is thread:
            self.job_progress_thread = None

    def apply_job_progress(self, progress: dict) -> None:
        active_job = self._job_by_progress_id(self.active_progress_file_id)
        state = progress.get("state", "queued")
        is_terminal = state in TERMINAL_JOB_STATES
        message = progress.get("message", state.replace("_", " ").title())
        self.progress_box.setVisible(True)
        self.update_workflow_steps(state)
        if active_job:
            active_job["state"] = state
            active_job["message"] = message
            active_job["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.update_job_history_metrics(active_job, progress)
            self._save_jobs()
        self.set_active_job_focus(active_job, state)
        self.status.setText(message)
        self.next_step.setText(message)
        display_job_id = self.monitored_job_id or (active_job.get("job_id", "") if active_job else "")
        runtime_text = self.progress_runtime_text(progress, is_terminal)
        self.monitoring_label.setText(
            f"Monitoring job: {display_job_id} - {state.replace('_', ' ')}{runtime_text}"
        )
        self.monitoring_label.setVisible(True)
        self.update_remote_file_progress(progress, state, is_terminal)
        self.update_remote_total_progress(progress, state, is_terminal)
        if is_terminal:
            self.finish_remote_job(active_job, state, progress)

    @staticmethod
    def progress_runtime_text(progress: dict, is_terminal: bool) -> str:
        if is_terminal:
            return ""
        stage = progress.get("stage", "segmenting")
        if stage == "benchmark_video":
            return " | CPU video encoding"
        if stage == "overlay":
            return " | CPU overlay encoding"
        if stage != "segmenting" or not progress.get("inference_batch_size"):
            return ""
        effective_batch = progress["inference_batch_size"]
        requested_batch = progress.get("requested_inference_batch_size")
        batch_text = f"batch {effective_batch}"
        if requested_batch and int(requested_batch) != int(effective_batch):
            batch_text += f" (requested {requested_batch}; adjusted for GPU memory)"
        runtime_text = (
            f" | {batch_text} | {progress.get('cpu_postprocess_workers', '?')} CPU workers"
        )
        throughput = progress.get("throughput_fps")
        if throughput is not None:
            runtime_text += f" | {float(throughput):.1f} frames/s"
        gpu_seconds = progress.get("gpu_inference_seconds")
        postprocess_seconds = progress.get("postprocess_seconds")
        if gpu_seconds is not None and postprocess_seconds is not None:
            runtime_text += (
                f" (GPU {float(gpu_seconds):.2f}s; "
                f"postprocess {float(postprocess_seconds):.2f}s/batch)"
            )
        return runtime_text

    def update_remote_file_progress(
        self,
        progress: dict,
        state: str,
        is_terminal: bool,
    ) -> None:
        current_frame = progress.get("current_frame")
        current_file_frames = progress.get("current_file_frames")
        if not is_terminal and current_frame is not None and current_file_frames:
            self.file_progress.setVisible(True)
            self.file_progress.setRange(0, int(current_file_frames))
            self.file_progress.setValue(min(int(current_frame), int(current_file_frames)))
            file_name = progress.get("current_file_name", "Current video")
            file_index = progress.get("current_file_index", "?")
            file_count = progress.get("file_count", "?")
            stage = progress.get("stage", "segmenting")
            if stage == "benchmark_video":
                file_text = (
                    f"File {file_index}/{file_count} - {file_name}: rendering benchmark "
                    f"videos, output frame {current_frame}/{current_file_frames}"
                )
            elif stage == "overlay":
                file_text = (
                    f"File {file_index}/{file_count} - {file_name}: creating overlay, "
                    f"frame {current_frame}/{current_file_frames}"
                )
            elif stage == "workbook":
                file_text = "Creating combined result files"
            else:
                file_text = (
                    f"File {file_index}/{file_count} - {file_name}: "
                    f"frame {current_frame}/{current_file_frames}"
                )
            self.file_progress_label.setText(file_text)
            self.file_progress_label.setVisible(True)
            self.file_progress.setFormat("%p%")
        elif state in {"queued", "downloading"}:
            self.file_progress_label.setText(
                "Current file: waiting for Colab" if state == "queued"
                else "Current file: Colab is downloading videos"
            )
            self.file_progress_label.setVisible(True)

    def update_remote_total_progress(
        self,
        progress: dict,
        state: str,
        is_terminal: bool,
    ) -> None:
        fraction, processed_frames, total_frames = self.frame_progress_fraction(progress)
        if not is_terminal and fraction is not None:
            self.job_progress_bar.setVisible(True)
            self.job_progress_bar.setRange(0, 1000)
            self.job_progress_bar.setValue(max(0, min(1000, round(float(fraction) * 1000))))
            eta = self._format_eta(progress.get("estimated_remaining_seconds"))
            self.job_progress_label.setText(
                self.frame_progress_text(
                    fraction,
                    processed_frames,
                    total_frames,
                    str(progress.get("stage", "segmenting")),
                    eta,
                )
            )
            self.job_progress_label.setVisible(True)
            self.job_progress_bar.setFormat("%p%")
        elif state == "queued":
            self.job_progress_label.setText("Total job: queued - waiting for available Colab compute")
            self.job_progress_label.setVisible(True)

    def finish_remote_job(self, active_job: dict | None, state: str, progress: dict) -> None:
        terminal_text = {
            "complete": "Analysis complete",
            "failed": "Analysis failed",
            "cancelled": "Analysis cancelled",
        }
        self.poll_timer.stop()
        self.upload_progress.setVisible(False)
        self.stop_upload_attention()
        if state == "complete":
            self.file_progress.setVisible(True)
            self.file_progress.setRange(0, 100)
            self.file_progress.setValue(100)
            self.file_progress.setFormat("%p%")
            self.file_progress_label.setVisible(True)
            self.file_progress_label.setText("All selected videos processed")
            self.job_progress_bar.setVisible(True)
            self.job_progress_bar.setRange(0, 1000)
            self.job_progress_bar.setValue(1000)
            self.job_progress_bar.setFormat("%p%")
            self.job_progress_label.setVisible(True)
            self.job_progress_label.setText("Analysis complete")
        else:
            self.reset_job_progress(terminal_text[state])
        if state == "complete" and active_job:
            self.start_result_download(active_job, action="offer")
            self.job_progress_label.setText("Analysis complete - downloading results")
        self.submit_button.setEnabled(bool(self.videos))
        self.active_progress_file_id = ""
        self.refresh_video_table(
            remote_job=bool(active_job and not active_job.get("source_paths"))
        )
        for row in range(self.video_table.rowCount()):
            self.set_video_status(row, terminal_text[state].removeprefix("Analysis "))
        self.update_workflow_steps(state)
        self.set_active_job_focus(active_job, state)
        # The worker heartbeat contains the next active job and will hand the panel
        # over directly. Do not scan the full Drive history here because that blocks
        # detailed progress polling for the next job.
        if state == "complete":
            local_path = active_job.get("local_results_path", "") if active_job else ""
            self.next_step.setText(
                "Analysis is complete. Downloading the result files in the background."
                if self.result_download_thread is not None
                else f"Analysis complete. Results downloaded to {local_path}."
                if local_path
                else "Analysis complete. Results remain available in the job's Google Drive folder."
            )
        elif state == "cancelled":
            self.next_step.setText(
                "Analysis cancelled. Open Job history to retry it without uploading the videos again."
            )
        else:
            self.next_step.setText(
                f"Analysis failed: {progress.get('error', 'Check the Colab output for details.')}"
            )

    def result_download_target(
        self,
        job: dict,
        force: bool = False,
    ) -> Path | None:
        return find_result_download_target(job, force=force)

    def start_result_download(
        self,
        job: dict,
        force: bool = False,
        notify: bool = False,
        action: str = "",
    ) -> None:
        target = self.result_download_target(job, force)
        if job.get("backend") == "local":
            if target is None:
                if notify:
                    QMessageBox.warning(
                        self,
                        "Results unavailable",
                        "The local results folder no longer exists at its recorded location.",
                    )
                return
            if notify and not action:
                QMessageBox.information(self, "Local results", f"Results are stored at:\n{target}")
            self.finish_result_action(job, target, action)
            return
        if target is None:
            if notify:
                QMessageBox.warning(
                    self,
                    "Download location unavailable",
                    "Choose the original recording folder from Job History and try again.",
                )
            return
        if job.get("local_results_path") and target.exists() and not force:
            self.finish_result_action(job, target, action)
            return
        if self.result_download_thread is not None:
            if notify:
                QMessageBox.information(
                    self,
                    "Download in progress",
                    "StimTrace is already downloading a completed result set.",
                )
            return
        self.status.setText(f"Downloading results to {target}...")
        self.next_step.setText("Downloading completed results. StimTrace remains usable while this finishes.")
        self.job_progress_label.setText("Analysis complete - downloading results...")
        self.job_progress_bar.setRange(0, 1000)
        self.job_progress_bar.setValue(0)
        self.job_progress_bar.setFormat("Downloading results: 0%")
        job_id = job.get("job_id", "")
        result_archive_file_id = job.get("result_archive_file_id", "")
        folder_id = job.get("folder_id", "")

        def download() -> dict:
            def report_progress(payload: object) -> None:
                thread.progress.emit(payload)

            count = self.drive.download_job_results(
                folder_id,
                target,
                report_progress,
                result_archive_file_id,
            )
            overlays_created: list[str] = []
            overlays_skipped: list[str] = []
            if job.get("overlay_mode") == "local":
                # Import the scientific video stack only in this worker thread,
                # after the small trace archive is safely downloaded.
                from local_runner import render_downloaded_overlays

                thread.progress.emit({"phase": "overlays"})
                overlays_created, overlays_skipped = render_downloaded_overlays(
                    target,
                    list(job.get("source_paths", [])),
                    lambda payload: thread.progress.emit(payload),
                )
            return {
                "job_id": job_id, "target": str(target), "count": count,
                "overlays_created": overlays_created,
                "overlays_skipped": overlays_skipped,
            }

        thread = BackgroundFunctionThread(download, self)
        self.register_thread(thread, "drive-result-download")
        thread.progress.connect(self.update_result_download_progress, Qt.QueuedConnection)
        thread.succeeded.connect(self.finish_result_download, Qt.QueuedConnection)
        thread.failed.connect(self.fail_result_download, Qt.QueuedConnection)
        thread.finished.connect(self.finish_result_download_thread, Qt.QueuedConnection)
        self.result_download_context = {
            "job_id": job_id,
            "target": str(target),
            "notify": notify,
            "action": action,
            "folder_id": folder_id,
        }
        self.result_download_thread = thread
        thread.start()

    @Slot(object)
    def update_result_download_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        phase = str(payload.get("phase", ""))
        if phase == "downloading":
            fraction = min(1.0, max(0.0, float(payload.get("fraction", 0.0))))
            percent = round(fraction * 100)
            self.job_progress_bar.setRange(0, 1000)
            self.job_progress_bar.setValue(round(fraction * 800))
            self.job_progress_bar.setFormat(f"Downloading results: {percent}%")
            total_bytes = int(payload.get("total_bytes", 0) or 0)
            current_bytes = int(payload.get("current_bytes", 0) or 0)
            size_text = (
                f" ({current_bytes / 1048576:.1f}/{total_bytes / 1048576:.1f} MB)"
                if total_bytes else ""
            )
            self.job_progress_label.setText(
                f"Downloading result archive: {percent}%{size_text}"
            )
        elif phase in {"extracting", "files"}:
            current = int(payload.get("current", 0) or 0)
            total = max(1, int(payload.get("total", 1) or 1))
            filename = str(payload.get("filename", ""))
            fraction = min(1.0, current / total)
            self.job_progress_bar.setRange(0, 1000)
            self.job_progress_bar.setValue(round(800 + fraction * 200))
            verb = "Extracting" if phase == "extracting" else "Downloading"
            self.job_progress_bar.setFormat(f"{verb} files: {current}/{total}")
            self.job_progress_label.setText(
                f"{verb} result {current}/{total}: {filename}"
            )
        elif phase == "overlays":
            self.file_progress_label.setText("Preparing local overlay rendering...")
            self.file_progress_label.setVisible(True)
            self.file_progress.setRange(0, 1000)
            self.file_progress.setValue(0)
            self.file_progress.setFormat("0%")
            self.file_progress.setVisible(True)
            self.job_progress_bar.setRange(0, 1000)
            self.job_progress_bar.setValue(0)
            self.job_progress_bar.setFormat("Local overlays: 0%")
            self.job_progress_label.setText("Results downloaded - creating overlay videos locally")
            self.status.setText("Creating overlay videos locally...")
            self.next_step.setText("Rendering overlays from downloaded tracking coordinates and original videos.")
        elif phase == "overlay_progress":
            file_index = int(payload.get("file_index", 0) or 0)
            file_count = max(1, int(payload.get("file_count", 1) or 1))
            current = int(payload.get("current_frame", 0) or 0)
            file_total = max(1, int(payload.get("current_file_frames", 1) or 1))
            filename = str(payload.get("filename", ""))
            overall = min(1.0, max(0.0, float(payload.get("overall_fraction", 0.0))))
            file_fraction = min(1.0, max(0.0, current / file_total))
            self.file_progress.setRange(0, 1000)
            self.file_progress.setValue(round(file_fraction * 1000))
            self.file_progress.setFormat(f"{round(file_fraction * 100)}%")
            self.file_progress_label.setText(
                f"Overlay {file_index}/{file_count}: {filename}, frame {current}/{file_total}"
            )
            self.job_progress_bar.setRange(0, 1000)
            self.job_progress_bar.setValue(round(overall * 1000))
            self.job_progress_bar.setFormat(f"Local overlays: {round(overall * 100)}%")
            self.job_progress_label.setText(
                f"Creating local overlays: {file_index}/{file_count} recordings"
            )

    @Slot(object)
    def finish_result_download(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        job = next(
            (item for item in self.jobs if item.get("job_id") == result.get("job_id")),
            None,
        )
        if job is None:
            return
        target = Path(str(result.get("target", "")))
        count = int(result.get("count", 0))
        overlays_created = list(result.get("overlays_created", []))
        overlays_skipped = list(result.get("overlays_skipped", []))
        job["local_results_path"] = str(target)
        job["message"] = f"Complete. Downloaded {count} result files to {target}."
        if overlays_created:
            job["message"] += f" Created {len(overlays_created)} overlay video(s) locally."
        if overlays_skipped:
            job["message"] += f" {len(overlays_skipped)} overlay(s) could not be created."
            job["local_overlay_warnings"] = overlays_skipped
        job["download_finished_at"] = datetime.now(timezone.utc).isoformat()
        self._save_jobs()
        self.status.setText(job["message"])
        self.next_step.setText(job["message"])
        self.job_progress_bar.setRange(0, 1000)
        self.job_progress_bar.setValue(1000)
        self.job_progress_bar.setFormat("100%")
        self.job_progress_label.setText(
            "Analysis complete - results downloaded; local overlays incomplete"
            if overlays_skipped
            else "Analysis complete - results downloaded"
        )
        if overlays_created or overlays_skipped:
            self.file_progress.setRange(0, 1000)
            self.file_progress.setValue(1000)
            self.file_progress.setFormat("100%")
            self.file_progress_label.setText(
                f"Local overlay rendering finished: {len(overlays_created)} created, "
                f"{len(overlays_skipped)} skipped"
                if overlays_skipped
                else "Local overlay rendering complete"
            )
        context = dict(self.result_download_context)
        if overlays_skipped:
            title, message = local_overlay_warning_message(overlays_skipped, target)
            QMessageBox.warning(self, title, message)
        elif context.get("notify"):
            QMessageBox.information(
                self,
                "Results downloaded",
                f"Downloaded {count} result files to:\n{target}",
            )
        self.finish_result_action(job, target, str(context.get("action", "")))
        if self.job_history_page is not None:
            self.job_history_page.populate()

    @Slot(str)
    def fail_result_download(self, error: str) -> None:
        context = dict(self.result_download_context)
        job = next(
            (item for item in self.jobs if item.get("job_id") == context.get("job_id")),
            None,
        )
        if job is not None:
            job["message"] = f"Analysis complete, but local result download failed: {error}"
            self._save_jobs()
            self.status.setText(job["message"])
        self.job_progress_bar.setRange(0, 1000)
        self.job_progress_bar.setValue(1000)
        self.job_progress_label.setText("Analysis complete - result download needs attention")
        if context.get("notify"):
            if "no Drive results folder" in error:
                if QMessageBox.question(
                    self,
                    "Older result format",
                    "This job was completed with an older StimTrace worker. Its Colab-created result "
                    "folder cannot be downloaded through the restricted desktop authorization.\n\n"
                    "Open the job in Google Drive to download the results manually?",
                ) == QMessageBox.Yes and context.get("folder_id"):
                    webbrowser.open(
                        f"https://drive.google.com/drive/folders/{context['folder_id']}"
                    )
            else:
                QMessageBox.critical(self, "Result download failed", error)

    @Slot()
    def finish_result_download_thread(self) -> None:
        thread = self.sender()
        if isinstance(thread, BackgroundFunctionThread):
            thread.wait()
        if self.result_download_thread is thread:
            self.result_download_thread = None
        self.result_download_context = {}

    def finish_result_action(self, job: dict, target: Path, action: str) -> None:
        if action == "offer":
            self.offer_signal_analysis(job)
        elif action == "open":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
        elif action == "analyze":
            trace_file = self.analysis_trace_file(target)
            if trace_file is not None:
                self.open_signal_analysis_files([trace_file])
            else:
                QMessageBox.information(
                    self,
                    "Force traces unavailable",
                    "The downloaded results do not contain a combined force-trace file.",
                )

    def record_actual_processing_time(self, job: dict) -> None:
        if job.get("actual_total_seconds") is not None:
            return
        start_text = (
            job.get("processing_started_at")
            or job.get("submitted_at")
            or job.get("created_at")
        )
        if not start_text:
            return
        finished = datetime.now(timezone.utc)
        duration = elapsed_seconds(start_text, finished)
        if duration is None:
            return
        job["results_prompted_at"] = finished.isoformat()
        job["actual_total_seconds"] = duration
        self._save_jobs()

    @staticmethod
    def mark_job_processing_started(job: dict, timestamp: object = None) -> None:
        if job.get("processing_started_at"):
            return
        job["processing_started_at"] = str(timestamp or datetime.now(timezone.utc).isoformat())

    @staticmethod
    def format_elapsed(seconds: object) -> str:
        return format_job_elapsed(seconds)

    @staticmethod
    def format_processed_frames(value: object, total: object = None) -> str:
        try:
            frames = int(value)
        except (TypeError, ValueError):
            return ""
        if frames < 0:
            return ""
        try:
            total_frames = int(total)
        except (TypeError, ValueError):
            total_frames = -1
        return f"{frames} / {total_frames}" if total_frames >= 0 else str(frames)

    @staticmethod
    def reset_job_attempt_metrics(job: dict) -> None:
        for key in (
            "processing_started_at",
            "processing_finished_at",
            "actual_total_seconds",
            "processed_frames",
            "total_job_frames",
            "results_prompted_at",
        ):
            job.pop(key, None)

    @classmethod
    def update_job_history_metrics(cls, job: dict, progress: dict) -> None:
        state = str(progress.get("state", job.get("state", "")))
        incoming_start = progress.get("processing_started_at")
        previous_start = job.get("processing_started_at")
        if state == "queued":
            cls.reset_job_attempt_metrics(job)
            return
        if incoming_start and previous_start and str(incoming_start) != str(previous_start):
            # The same Drive folder is reused by Retry. A new worker start marks a
            # new attempt, so partial counts and duration from the prior attempt
            # must not leak into this one.
            cls.reset_job_attempt_metrics(job)
        if incoming_start:
            job["processing_started_at"] = str(incoming_start)
        elif state in RUNNING_JOB_STATES and not job.get("processing_started_at"):
            cls.mark_job_processing_started(job)
        cls.update_job_processed_frames(job, progress)
        for key in ("processing_finished_at", "actual_total_seconds"):
            if progress.get(key) is not None:
                job[key] = progress[key]
        if (
            job.get("actual_total_seconds") is None
            and job.get("processing_started_at")
            and job.get("processing_finished_at")
        ):
            try:
                finished = datetime.fromisoformat(
                    str(job["processing_finished_at"]).replace("Z", "+00:00")
                )
            except (TypeError, ValueError):
                return
            duration = elapsed_seconds(job["processing_started_at"], finished)
            if duration is not None:
                job["actual_total_seconds"] = duration

    @staticmethod
    def update_job_processed_frames(job: dict, progress: dict) -> None:
        try:
            processed = max(0, int(progress["processed_frames"]))
        except (KeyError, TypeError, ValueError):
            return
        try:
            previous = max(0, int(job.get("processed_frames", 0)))
        except (TypeError, ValueError):
            previous = 0
        job["processed_frames"] = max(previous, processed)
        try:
            job["total_job_frames"] = max(0, int(progress["total_job_frames"]))
        except (KeyError, TypeError, ValueError):
            pass

    @staticmethod
    def frame_progress_fraction(progress: dict) -> tuple[float, int | None, int | None]:
        """Return frame-based progress, retaining legacy fraction only as a fallback."""
        try:
            total = max(0, int(progress.get("total_job_frames", 0)))
            processed = max(0, int(progress.get("processed_frames", 0)))
        except (TypeError, ValueError):
            total = 0
            processed = 0
        if total:
            return min(1.0, processed / total), min(processed, total), total
        try:
            return max(0.0, min(1.0, float(progress.get("job_progress_fraction", 0)))), None, None
        except (TypeError, ValueError):
            return 0.0, None, None

    @staticmethod
    def frame_progress_text(
        fraction: float,
        processed_frames: int | None,
        total_frames: int | None,
        stage: str,
        eta: str,
    ) -> str:
        if processed_frames is not None and total_frames is not None:
            text = f"Frames processed: {processed_frames}/{total_frames} ({fraction * 100:.1f}%)"
            if fraction >= 1.0 and stage != "segmenting":
                return text + " - generating result files"
            if eta:
                return text + f" - estimated time remaining: {eta}"
            return text
        return f"Total job: {fraction * 100:.1f}%" + (f" - estimated time remaining: {eta}" if eta else "")

    @staticmethod
    def _format_eta(seconds) -> str:
        if seconds is None:
            return ""
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def shutdown_threads(self, allow_cancel: bool = True) -> bool:
        if not self.shutdown_started:
            self.shutdown_started = True
            self.shutdown_started_at = time.monotonic()
            for timer_name in (
                "restore_session_timer",
                "hardware_probe_start_timer",
                "poll_timer",
                "worker_timer",
                "upload_attention_timer",
            ):
                timer = getattr(self, timer_name, None)
                if timer is not None:
                    timer.stop()
            if self.training_page is not None:
                self.training_page.training_timer.stop()
            if (
                self.point_tracking_page is not None
                and self.point_tracking_page.worker is not None
            ):
                self.point_tracking_page.worker.request_cancel()
            if self.hardware_probe is not None and self.hardware_probe.isRunning():
                self.hardware_probe.terminate_process()
            if self.local_process is not None and self.local_process.isRunning():
                if allow_cancel and self.local_cancel_file:
                    try:
                        self.local_cancel_file.write_text("cancel", encoding="ascii")
                    except OSError:
                        LOGGER.exception("Could not write local cancellation marker during shutdown")
                job = self.current_job()
                if allow_cancel and job:
                    self.local_terminal_event = "cancelled"
                    job["state"] = "cancelled"
                    job["message"] = "Local analysis stopped because StimTrace closed."
                    self._save_jobs()
                self.local_process.terminate_process()
            for thread in self.active_threads:
                if thread.isRunning():
                    thread.requestInterruption()

        elapsed = time.monotonic() - self.shutdown_started_at
        if elapsed >= 5.0:
            if self.hardware_probe is not None and self.hardware_probe.isRunning():
                self.hardware_probe.kill_process()
            if self.local_process is not None and self.local_process.isRunning():
                self.local_process.kill_process()
            # A blocking SSL read cannot observe Qt's interruption request. Do not
            # leave an invisible or uncloseable application waiting indefinitely
            # for a remote host during final shutdown.
            for thread in list(self.active_threads):
                if not thread.isRunning():
                    continue
                LOGGER.warning(
                    "Terminating background task during shutdown after %.1fs: %s",
                    elapsed,
                    thread.objectName() or "background task",
                )
                thread.terminate()
                thread.wait(500)
        running = [thread for thread in self.active_threads if thread.isRunning()]
        if running:
            names = ", ".join(thread.objectName() or "background task" for thread in running[:3])
            if hasattr(self, "status"):
                self.status.setText(f"Closing StimTrace after background work finishes: {names}")
            return False
        self.active_threads.clear()
        self.retired_threads.clear()
        if not self.settings.remember_google_sign_in:
            self.drive.sign_out()
        return True

    def retry_close_after_workers(self) -> None:
        if self.shutdown_threads():
            self.close()
        else:
            self.shutdown_retry_timer.start(150)

    def closeEvent(self, event) -> None:
        if self.shutdown_threads():
            event.accept()
        else:
            event.ignore()
            # Closing should take the window away immediately. A short-lived
            # background cleanup may continue, with a five-second hard limit.
            self.hide()
            if not self.shutdown_retry_timer.isActive():
                self.shutdown_retry_timer.start(150)


if __name__ == "__main__":
    if "--local-worker" in sys.argv:
        sys.argv.remove("--local-worker")
        from local_runner import main as run_local_worker

        run_local_worker()
    else:
        install_crash_logging()
        configure_numeric_locale()
        app = QApplication(sys.argv)
        app_icon = QIcon(str(APP_ICON_PATH))
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
        window = MainWindow()
        app.aboutToQuit.connect(lambda: window.shutdown_threads(allow_cancel=False))
        # Start at a comfortable size instead of unconditionally occupying the
        # whole display.  Clamp to the available desktop so Windows scaling,
        # taskbars, and smaller secondary monitors cannot place part of the
        # application off-screen.  The user can still maximize normally.
        screen = window.screen() or app.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = min(1600, max(window.minimumWidth(), int(available.width() * 0.90)))
            target_height = min(1000, max(window.minimumHeight(), int(available.height() * 0.90)))
            target_width = min(target_width, available.width())
            target_height = min(target_height, available.height())
            window.resize(target_width, target_height)
            window.move(
                available.x() + (available.width() - target_width) // 2,
                available.y() + (available.height() - target_height) // 2,
            )
        window.show()
        exit_code = app.exec()
        window.shutdown_threads(allow_cancel=False)
        sys.exit(exit_code)
