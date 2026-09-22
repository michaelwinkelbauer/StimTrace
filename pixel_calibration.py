"""Interactive spatial calibration from a known object in a video frame."""
from __future__ import annotations

import math
from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import EllipseSelector
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui_actions import configure_row_delete, set_action_icon


VIDEO_FILTER = "Video files (*.avi *.mp4 *.mov *.mkv *.m4v);;All files (*.*)"


def ellipse_measurements(width_px: float, height_px: float) -> dict[str, float]:
    """Return explicit distance measures for an axis-aligned ellipse."""
    width = abs(float(width_px))
    height = abs(float(height_px))
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("Ellipse width and height must be positive finite values.")
    semi_major = width / 2.0
    semi_minor = height / 2.0
    h = ((semi_major - semi_minor) / (semi_major + semi_minor)) ** 2
    perimeter = math.pi * (semi_major + semi_minor) * (
        1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h))
    )
    return {
        "width": width,
        "height": height,
        "mean_diameter": (width + height) / 2.0,
        "perimeter": perimeter,
    }


def micrometers_per_pixel(known_distance_um: float, measured_distance_px: float) -> float:
    """Calculate spatial calibration from matching physical and pixel distances."""
    known = float(known_distance_um)
    measured = float(measured_distance_px)
    if not math.isfinite(known) or known <= 0:
        raise ValueError("Known distance must be greater than zero.")
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError("Measured pixel distance must be greater than zero.")
    return known / measured


def calibration_summary(values: list[float]) -> tuple[float, float, float]:
    """Return mean, sample standard deviation, and coefficient of variation."""
    calibrations = np.asarray(values, dtype=float)
    if calibrations.size == 0 or not np.isfinite(calibrations).all() or np.any(calibrations <= 0):
        raise ValueError("Calibration measurements must be positive finite values.")
    mean = float(np.mean(calibrations))
    standard_deviation = float(np.std(calibrations, ddof=1)) if calibrations.size > 1 else 0.0
    coefficient_of_variation = standard_deviation / mean * 100.0
    return mean, standard_deviation, coefficient_of_variation


def read_video_frame(path: Path, frame_index: int) -> tuple[np.ndarray, int]:
    """Read one RGB frame and return it with the recording's frame count."""
    capture = cv.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        frame_count = max(1, int(capture.get(cv.CAP_PROP_FRAME_COUNT)))
        index = min(max(0, int(frame_index)), frame_count - 1)
        capture.set(cv.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not decode frame {index + 1} from {path.name}.")
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB), frame_count
    finally:
        capture.release()


class PixelCalibrationDialog(QDialog):
    """Measure a known ellipse dimension and return its spatial calibration."""

    def __init__(
        self,
        current_um_per_px: float,
        parent=None,
        initial_videos: list[Path] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate pixel size from video")
        available = QGuiApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(960, 680)
        self.resize(
            min(1280, max(960, available.width() - 80)),
            min(820, max(680, available.height() - 120)),
        )
        self.video_paths: list[Path] = []
        self.video_frame_counts: dict[str, int] = {}
        self.measurements: dict[str, list[dict[str, float | int]]] = {}
        self.video_path: Path | None = None
        self.image_artist = None
        self.selector: EllipseSelector | None = None
        self.pan_state = None
        self.needs_manual_ellipse = False
        self.calibration_um_per_px = float(current_um_per_px)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Select three recordings, then measure the same known object in three different frames "
            "per recording. All recordings must use the same camera and zoom."
        )
        description.setWordWrap(True)
        description.setProperty("role", "subtitle")
        help_button = QPushButton("Help")
        set_action_icon(help_button, "help")
        help_button.setToolTip("Show instructions for pixel calibration")
        help_button.clicked.connect(self.show_help)
        help_row = QHBoxLayout()
        help_row.addWidget(description, 1)
        help_row.addWidget(help_button, 0, Qt.AlignTop)
        layout.addLayout(help_row)

        file_row = QHBoxLayout()
        self.open_button = QPushButton("Select 3 videos")
        set_action_icon(self.open_button, "open")
        self.open_button.clicked.connect(self.select_videos)
        self.video_label = QLabel("No videos selected")
        self.video_label.setProperty("role", "muted")
        file_row.addWidget(self.open_button)
        file_row.addWidget(self.video_label, 1)
        layout.addLayout(file_row)

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Recording"))
        self.video_selector = QComboBox()
        self.video_selector.setEnabled(False)
        self.video_selector.currentIndexChanged.connect(self.change_video)
        video_row.addWidget(self.video_selector, 1)
        self.measurement_progress = QLabel("Measurements: 0 / 9")
        video_row.addWidget(self.measurement_progress)
        layout.addLayout(video_row)

        # These controls are placed in the compact left side panel below. Keeping
        # them out of the header preserves as much vertical area as possible for
        # the frame being calibrated.
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.metric_selector = QComboBox()
        self.metric_selector.addItem("Ellipse width", "width")
        self.metric_selector.addItem("Ellipse height", "height")
        self.metric_selector.addItem("Mean of width and height", "mean_diameter")
        self.metric_selector.addItem("Ellipse perimeter", "perimeter")
        self.metric_selector.currentIndexChanged.connect(self.update_measurement)
        form.addRow("Known dimension corresponds to", self.metric_selector)
        self.measured_label = QLabel("Draw an ellipse to measure pixels")
        form.addRow("Selected distance", self.measured_label)
        self.known_distance = QDoubleSpinBox()
        self.known_distance.setRange(0.001, 1_000_000.0)
        self.known_distance.setDecimals(3)
        self.known_distance.setSuffix(" um")
        self.known_distance.setValue(1000.0)
        self.known_distance.valueChanged.connect(self.update_measurement)
        form.addRow("Known distance", self.known_distance)
        self.current_result_label = QLabel("-")
        form.addRow("Current measurement", self.current_result_label)
        self.result_label = QLabel("Complete all 9 measurements")
        self.result_label.setStyleSheet("font-weight: 600;")
        form.addRow("Mean calibration", self.result_label)
        note = QLabel(
            "Calibration = known distance / selected pixels."
        )
        note.setWordWrap(True)
        note.setProperty("role", "muted")

        workspace_splitter = QSplitter(Qt.Horizontal)
        workspace_splitter.setChildrenCollapsible(False)

        measurement_panel = QWidget()
        measurement_panel.setMinimumWidth(300)
        measurement_panel.setMaximumWidth(380)
        measurement_layout = QVBoxLayout(measurement_panel)
        measurement_layout.setContentsMargins(0, 0, 8, 0)
        measurement_heading = QLabel("Recorded measurements")
        measurement_heading.setStyleSheet("font-weight: 600;")
        measurement_layout.addWidget(measurement_heading)
        measurement_layout.addLayout(form)
        measurement_layout.addWidget(note)
        self.measurement_table = QTableWidget(0, 4)
        self.measurement_table.setHorizontalHeaderLabels(
            ["Measurement", "Pixels", "um/px", ""]
        )
        self.measurement_table.verticalHeader().setVisible(False)
        self.measurement_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.measurement_table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.measurement_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        measurement_layout.addWidget(self.measurement_table, 1)
        workspace_splitter.addWidget(measurement_panel)

        self.record_button = QPushButton("Record this measurement")
        set_action_icon(self.record_button, "save")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.record_measurement)
        self.undo_button = QPushButton("Undo last for this video")
        set_action_icon(self.undo_button, "previous")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_last_measurement)
        self.reset_button = QPushButton("Delete all measurements")
        set_action_icon(self.reset_button, "delete")
        self.reset_button.setProperty("role", "danger")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.confirm_reset_measurements)
        measurement_actions = QVBoxLayout()
        measurement_actions.setSpacing(5)
        measurement_actions.addWidget(self.record_button)
        action_row = QHBoxLayout()
        action_row.addWidget(self.undo_button)
        action_row.addWidget(self.reset_button)
        measurement_actions.addLayout(action_row)
        measurement_layout.addLayout(measurement_actions)

        frame_panel = QWidget()
        frame_layout = QVBoxLayout(frame_panel)
        frame_layout.setContentsMargins(8, 0, 0, 0)
        self.figure = Figure(facecolor="#14191d")
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor("#14191d")
        self.axes.set_axis_off()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(560, 360)
        self.canvas.setToolTip(
            "Left-drag to draw or resize the ellipse. Scroll to zoom. "
            "Middle-drag or right-drag to move the frame."
        )
        self.canvas.mpl_connect("scroll_event", self.zoom_on_scroll)
        self.canvas.mpl_connect("button_press_event", self.start_pan)
        self.canvas.mpl_connect("motion_notify_event", self.pan_frame)
        self.canvas.mpl_connect("button_release_event", self.stop_pan)
        frame_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        frame_layout.addWidget(self.canvas, 1)
        mouse_legend = QLabel(
            "<b>Mouse controls</b>&nbsp;&nbsp; "
            "● Left-drag: draw or resize ellipse &nbsp;&nbsp; "
            "✥ Right- or middle-button drag: pan &nbsp;&nbsp; "
            "↕ Wheel: zoom"
        )
        mouse_legend.setTextFormat(Qt.RichText)
        mouse_legend.setTextInteractionFlags(Qt.NoTextInteraction)
        mouse_legend.setAlignment(Qt.AlignCenter)
        mouse_legend.setWordWrap(True)
        mouse_legend.setProperty("role", "muted")
        mouse_legend.setToolTip(
            "Left-drag draws or resizes the ellipse. Drag with the right or middle mouse button "
            "to pan; use the wheel to zoom."
        )
        frame_layout.addWidget(mouse_legend)
        workspace_splitter.addWidget(frame_panel)
        workspace_splitter.setStretchFactor(0, 0)
        workspace_splitter.setStretchFactor(1, 1)
        workspace_splitter.setSizes([340, 1240])
        layout.addWidget(workspace_splitter, 1)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Video frame"))
        self.previous_frame_button = QPushButton("Previous")
        set_action_icon(self.previous_frame_button, "previous")
        self.previous_frame_button.setEnabled(False)
        self.previous_frame_button.clicked.connect(lambda: self.step_frame(-1))
        frame_row.addWidget(self.previous_frame_button)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.schedule_frame_load)
        self.frame_slider.sliderReleased.connect(self.load_selected_frame)
        self.frame_label = QLabel("-")
        frame_row.addWidget(self.frame_slider, 1)
        self.next_frame_button = QPushButton("Next")
        set_action_icon(self.next_frame_button, "next")
        self.next_frame_button.setEnabled(False)
        self.next_frame_button.clicked.connect(lambda: self.step_frame(1))
        frame_row.addWidget(self.next_frame_button)
        frame_row.addWidget(self.frame_label)
        layout.addLayout(frame_row)
        self.frame_timer = QTimer(self)
        self.frame_timer.setSingleShot(True)
        self.frame_timer.setInterval(80)
        self.frame_timer.timeout.connect(self.load_selected_frame)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.apply_button = self.buttons.button(QDialogButtonBox.Ok)
        self.apply_button.setText("Use calibration")
        set_action_icon(self.apply_button, "apply")
        self.apply_button.setEnabled(False)
        set_action_icon(self.buttons.button(QDialogButtonBox.Cancel), "cancel")
        self.buttons.accepted.connect(self.accept_calibration)
        self.buttons.rejected.connect(self.reject)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.buttons)
        layout.addLayout(button_row)
        if initial_videos:
            # Delay frame decoding until the dialog has painted, keeping its opening responsive.
            QTimer.singleShot(0, lambda: self.load_videos(initial_videos))

    def select_videos(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Select exactly three calibration videos", "", VIDEO_FILTER
        )
        if not filenames:
            return
        self.load_videos([Path(filename) for filename in filenames])

    def load_videos(self, paths: list[Path]) -> bool:
        """Validate and load the three recordings used for the nine measurements."""
        unique_paths = list(dict.fromkeys(str(Path(path).resolve()) for path in paths))
        if len(unique_paths) != 3:
            QMessageBox.warning(
                self,
                "Three videos required",
                f"Select exactly three different videos. You selected {len(unique_paths)}.",
            )
            return False
        try:
            paths = [Path(path) for path in unique_paths]
            frame_counts = {}
            for path in paths:
                _, frame_count = read_video_frame(path, 0)
                if frame_count < 3:
                    raise ValueError(f"{path.name} has fewer than three readable frames.")
                frame_counts[str(path)] = frame_count
        except Exception as error:
            QMessageBox.critical(self, "Could not open video", str(error))
            return False
        self.video_paths = paths
        self.video_frame_counts = frame_counts
        self.measurements = {str(path): [] for path in paths}
        self.needs_manual_ellipse = False
        self.video_label.setText(" | ".join(path.name for path in paths))
        self.video_selector.blockSignals(True)
        self.video_selector.clear()
        for index, path in enumerate(paths, start=1):
            self.video_selector.addItem(f"{index}. {path.name}", str(path))
        self.video_selector.blockSignals(False)
        self.video_selector.setEnabled(True)
        self.video_selector.setCurrentIndex(0)
        self.change_video(0)
        self.update_measurement_progress()
        return True

    def suggested_frame(self, path: Path) -> int:
        count = len(self.measurements.get(str(path), []))
        frame_count = self.video_frame_counts[str(path)]
        fraction = (0.25, 0.50, 0.75)[min(count, 2)]
        return min(frame_count - 1, max(0, round((frame_count - 1) * fraction)))

    def change_video(self, index: int) -> None:
        if index < 0 or index >= len(self.video_paths):
            return
        self.frame_timer.stop()
        self.video_path = self.video_paths[index]
        frame_count = self.video_frame_counts[str(self.video_path)]
        frame_index = self.suggested_frame(self.video_path)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, frame_count - 1)
        self.frame_slider.setValue(frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_slider.setEnabled(True)
        self.previous_frame_button.setEnabled(True)
        self.next_frame_button.setEnabled(True)
        self.frame_label.setText(f"{frame_index + 1} / {frame_count}")
        try:
            image, _ = read_video_frame(self.video_path, frame_index)
            self.show_frame(image, reset_ellipse=True)
        except Exception as error:
            QMessageBox.critical(self, "Could not read frame", str(error))
        self.update_measurement_progress()

    def schedule_frame_load(self, frame_index: int) -> None:
        self.frame_label.setText(f"{frame_index + 1} / {self.frame_slider.maximum() + 1}")
        self.frame_timer.start()

    def step_frame(self, step: int) -> None:
        target = min(
            self.frame_slider.maximum(),
            max(self.frame_slider.minimum(), self.frame_slider.value() + step),
        )
        if target == self.frame_slider.value():
            return
        self.frame_slider.setValue(target)
        self.load_selected_frame()

    def load_selected_frame(self) -> None:
        if self.video_path is None:
            return
        try:
            self.frame_timer.stop()
            image, _ = read_video_frame(self.video_path, self.frame_slider.value())
            self.show_frame(image, reset_ellipse=False)
        except Exception as error:
            QMessageBox.critical(self, "Could not read frame", str(error))

    def show_frame(self, image: np.ndarray, *, reset_ellipse: bool) -> None:
        previous_xlim = self.axes.get_xlim() if self.image_artist is not None else None
        previous_ylim = self.axes.get_ylim() if self.image_artist is not None else None
        if self.image_artist is None:
            self.image_artist = self.axes.imshow(image, origin="upper")
        else:
            self.image_artist.set_data(image)
        height, width = image.shape[:2]
        if reset_ellipse or previous_xlim is None or previous_ylim is None:
            self.axes.set_xlim(0, width)
            self.axes.set_ylim(height, 0)
        else:
            self.axes.set_xlim(previous_xlim)
            self.axes.set_ylim(previous_ylim)
        if reset_ellipse or self.selector is None:
            if self.selector is not None:
                self.remove_selector()
            self.selector = EllipseSelector(
                self.axes,
                self.ellipse_changed,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                spancoords="data",
                interactive=True,
                drag_from_anywhere=True,
                props={"facecolor": "none", "edgecolor": "#d45a76", "linewidth": 2.0},
                handle_props={"color": "#f3a2b5", "alpha": 0.9},
            )
            if not self.needs_manual_ellipse:
                ellipse_width = width * 0.25
                ellipse_height = height * 0.25
                center_x, center_y = width / 2.0, height / 2.0
                self.selector.extents = (
                    center_x - ellipse_width / 2.0,
                    center_x + ellipse_width / 2.0,
                    center_y - ellipse_height / 2.0,
                    center_y + ellipse_height / 2.0,
                )
        self.canvas.draw_idle()
        self.update_measurement()

    def zoom_on_scroll(self, event) -> None:
        if (
            event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
            or self.image_artist is None
        ):
            return
        factor = 0.80 if event.button == "up" else 1.25
        image = np.asarray(self.image_artist.get_array())
        height, width = image.shape[:2]
        self.axes.set_xlim(
            self.scaled_limits(self.axes.get_xlim(), event.xdata, factor, 0.0, float(width))
        )
        self.axes.set_ylim(
            self.scaled_limits(self.axes.get_ylim(), event.ydata, factor, 0.0, float(height))
        )
        self.canvas.draw_idle()

    def start_pan(self, event) -> None:
        """Start grab-style image panning with the middle or right mouse button."""
        if (
            event.button not in (MouseButton.MIDDLE, MouseButton.RIGHT)
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        toolbar = getattr(self.canvas, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return
        self.pan_state = (
            float(event.xdata),
            float(event.ydata),
            self.axes.get_xlim(),
            self.axes.get_ylim(),
        )
        self.canvas.setCursor(Qt.ClosedHandCursor)

    def pan_frame(self, event) -> None:
        if (
            self.pan_state is None
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
            or self.image_artist is None
        ):
            return
        start_x, start_y, original_xlim, original_ylim = self.pan_state
        image = np.asarray(self.image_artist.get_array())
        height, width = image.shape[:2]
        self.axes.set_xlim(self.shifted_limits(
            original_xlim,
            start_x - float(event.xdata),
            0.0,
            float(width),
        ))
        self.axes.set_ylim(self.shifted_limits(
            original_ylim,
            start_y - float(event.ydata),
            0.0,
            float(height),
        ))
        self.canvas.draw_idle()

    def stop_pan(self, event) -> None:
        if self.pan_state is None:
            return
        if event.button in (MouseButton.MIDDLE, MouseButton.RIGHT):
            self.pan_state = None
            self.canvas.setCursor(Qt.ArrowCursor)

    @staticmethod
    def shifted_limits(
        limits: tuple[float, float],
        shift: float,
        bound_low: float,
        bound_high: float,
    ) -> tuple[float, float]:
        """Translate limits while keeping the visible image within its bounds."""
        start, end = map(float, limits)
        ascending = end >= start
        low, high = (start, end) if ascending else (end, start)
        span = high - low
        full_span = bound_high - bound_low
        if span >= full_span:
            result = (bound_low, bound_high)
        else:
            new_low = low + float(shift)
            new_high = high + float(shift)
            if new_low < bound_low:
                new_high += bound_low - new_low
                new_low = bound_low
            if new_high > bound_high:
                new_low -= new_high - bound_high
                new_high = bound_high
            result = (new_low, new_high)
        return result if ascending else (result[1], result[0])

    @staticmethod
    def scaled_limits(
        limits: tuple[float, float],
        cursor: float,
        factor: float,
        bound_low: float,
        bound_high: float,
    ) -> tuple[float, float]:
        """Scale one plot axis around the cursor while retaining image bounds."""
        start, end = map(float, limits)
        ascending = end >= start
        low, high = (start, end) if ascending else (end, start)
        full_span = max(bound_high - bound_low, 1.0)
        span = max(high - low, 1e-9)
        new_span = min(full_span, max(full_span / 50.0, span * factor))
        fraction = min(1.0, max(0.0, (float(cursor) - low) / span))
        new_low = float(cursor) - fraction * new_span
        new_high = new_low + new_span
        if new_low < bound_low:
            new_high += bound_low - new_low
            new_low = bound_low
        if new_high > bound_high:
            new_low -= new_high - bound_high
            new_high = bound_high
        result = (max(bound_low, new_low), min(bound_high, new_high))
        return result if ascending else (result[1], result[0])

    def ellipse_changed(self, _press=None, _release=None) -> None:
        self.needs_manual_ellipse = False
        self.update_measurement()

    def selected_measurements(self) -> dict[str, float] | None:
        if self.selector is None or self.needs_manual_ellipse:
            return None
        x1, x2, y1, y2 = self.selector.extents
        try:
            return ellipse_measurements(x2 - x1, y2 - y1)
        except ValueError:
            return None

    def update_measurement(self, _value=None) -> None:
        measurements = self.selected_measurements()
        if not measurements:
            self.measured_label.setText("Draw an ellipse to measure pixels")
            self.current_result_label.setText("-")
            self.apply_button.setEnabled(False)
            return
        metric = str(self.metric_selector.currentData())
        measured = measurements[metric]
        self.measured_label.setText(
            f"{measured:.3f} px  (width {measurements['width']:.3f} px, "
            f"height {measurements['height']:.3f} px)"
        )
        current_calibration = micrometers_per_pixel(
            self.known_distance.value(), measured
        )
        self.current_result_label.setText(f"{current_calibration:.6f} um/px")
        self.record_button.setEnabled(
            self.video_path is not None
            and len(self.measurements.get(str(self.video_path), [])) < 3
        )

    def all_measurements(self) -> list[dict[str, float | int]]:
        return [
            measurement
            for path in self.video_paths
            for measurement in self.measurements.get(str(path), [])
        ]

    def record_measurement(self) -> None:
        measurements = self.selected_measurements()
        if self.video_path is None or not measurements:
            return
        records = self.measurements[str(self.video_path)]
        frame_index = self.frame_slider.value()
        if any(int(record["frame_index"]) == frame_index for record in records):
            QMessageBox.information(
                self,
                "Different frame required",
                "This frame has already been measured for the current video. Choose another frame.",
            )
            return
        metric = str(self.metric_selector.currentData())
        measured_px = measurements[metric]
        records.append({
            "frame_index": frame_index,
            "measured_px": measured_px,
            "metric": metric,
            "calibration_um_per_px": micrometers_per_pixel(
                self.known_distance.value(), measured_px
            ),
        })
        self.discard_recorded_ellipse()
        self.update_measurement_progress()
        if len(records) < 3:
            self.frame_slider.setValue(self.suggested_frame(self.video_path))
            self.load_selected_frame()
            return
        for index, path in enumerate(self.video_paths):
            if len(self.measurements[str(path)]) < 3:
                self.video_selector.setCurrentIndex(index)
                return

    def discard_recorded_ellipse(self) -> None:
        """Remove a completed placement so the next measurement starts independently."""
        if self.selector is not None:
            self.remove_selector()
        self.needs_manual_ellipse = True
        self.canvas.draw_idle()
        self.measured_label.setText("Draw a new ellipse to measure pixels")
        self.current_result_label.setText("-")
        self.record_button.setEnabled(False)

    def remove_selector(self) -> None:
        """Disconnect and remove every artist belonging to the current ellipse selector."""
        if self.selector is None:
            return
        selector = self.selector
        self.selector = None
        selector.set_active(False)
        selector.set_visible(False)
        selector.disconnect_events()
        for artist in tuple(selector.artists):
            try:
                artist.remove()
            except ValueError:
                pass

    def undo_last_measurement(self) -> None:
        if self.video_path is None:
            return
        records = self.measurements.get(str(self.video_path), [])
        if records:
            records.pop()
            self.update_measurement_progress()

    def confirm_reset_measurements(self) -> None:
        if not self.all_measurements():
            return
        response = QMessageBox.question(
            self,
            "Delete all measurements",
            "Delete all recorded calibration measurements?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response == QMessageBox.Yes:
            self.reset_measurements()

    def reset_measurements(self) -> None:
        for records in self.measurements.values():
            records.clear()
        self.update_measurement_progress()

    def delete_measurement(self, video_path: str, frame_index: int) -> None:
        records = self.measurements.get(video_path, [])
        self.measurements[video_path] = [
            record
            for record in records
            if int(record["frame_index"]) != int(frame_index)
        ]
        self.update_measurement_progress()

    def refresh_measurement_table(self) -> None:
        rows = [
            (path, record)
            for path in self.video_paths
            for record in self.measurements.get(str(path), [])
        ]
        self.measurement_table.setRowCount(len(rows))
        for row, (path, record) in enumerate(rows):
            metric = str(record.get("metric", self.metric_selector.currentData()))
            metric_name = {
                "width": "width",
                "height": "height",
                "mean_diameter": "mean diameter",
                "perimeter": "perimeter",
            }.get(metric, metric)
            frame_number = int(record["frame_index"]) + 1
            values = (
                f"{path.name}\nFrame {frame_number} - {metric_name}",
                f"{float(record['measured_px']):.1f}",
                f"{float(record['calibration_um_per_px']):.5f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    f"{path.name}, frame {frame_number}: "
                    f"{float(record['measured_px']):.3f} px ({metric_name}), "
                    f"{float(record['calibration_um_per_px']):.6f} um/px"
                )
                if column in (1, 2):
                    item.setTextAlignment(Qt.AlignCenter)
                self.measurement_table.setItem(row, column, item)
            self.measurement_table.setRowHeight(row, 44)
            delete_button = QPushButton("X")
            configure_row_delete(delete_button, "Delete this calibration measurement")
            delete_button.clicked.connect(
                lambda _checked=False, video_path=str(path), frame_index=int(record["frame_index"]):
                self.delete_measurement(video_path, frame_index)
            )
            self.measurement_table.setCellWidget(row, 3, delete_button)

    def update_measurement_progress(self) -> None:
        records = self.all_measurements()
        self.refresh_measurement_table()
        counts = [len(self.measurements.get(str(path), [])) for path in self.video_paths]
        detail = " | ".join(
            f"Video {index}: {count}/3" for index, count in enumerate(counts, start=1)
        )
        self.measurement_progress.setText(
            f"Measurements: {len(records)} / 9" + (f"  ({detail})" if detail else "")
        )
        locked = bool(records)
        self.known_distance.setEnabled(not locked)
        self.reset_button.setEnabled(locked)
        current_records = (
            self.measurements.get(str(self.video_path), []) if self.video_path else []
        )
        self.undo_button.setEnabled(bool(current_records))
        self.record_button.setEnabled(
            self.video_path is not None
            and self.selected_measurements() is not None
            and len(current_records) < 3
        )
        complete = len(records) == 9 and counts == [3, 3, 3]
        if complete:
            mean, standard_deviation, variation = calibration_summary([
                float(record["calibration_um_per_px"]) for record in records
            ])
            self.calibration_um_per_px = mean
            self.result_label.setText(
                f"{mean:.6f} um/px (SD {standard_deviation:.6f}, CV {variation:.2f}%, n=9)"
            )
        else:
            self.result_label.setText("Complete all 9 measurements")
        self.apply_button.setEnabled(complete)

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "Pixel calibration help",
            "1. Select exactly three videos recorded with the same camera and zoom.\n\n"
            "2. Select the dimension represented by the known physical distance. You may "
            "change this between measurements; each recorded value retains its own setting.\n\n"
            "3. Left-drag around the known object, then adjust the ellipse using its handles. "
            "Scroll to zoom and middle-drag or right-drag to move the frame.\n\n"
            "4. Record three different frames from each video. StimTrace calculates each "
            "calibration as known distance divided by measured pixels.\n\n"
            "5. Review or delete individual measurements in the left list. After all nine "
            "are recorded, use their mean calibration.",
        )

    def accept_calibration(self) -> None:
        if not self.apply_button.isEnabled():
            QMessageBox.warning(
                self,
                "Calibration incomplete",
                "Record three different frames in each of the three selected videos first.",
            )
            return
        self.accept()
