"""Locate and play annotated recordings associated with Signal Analysis traces."""
from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QCursor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


ANNOTATED_SUFFIXES = (
    "_point_tracking_overlay.avi",
    "_overlay.avi",
    "_tracked_trace.avi",
)
TRACE_FILE_SUFFIXES = (
    "_stimtrace_tracking",
    "_pillar_displacement",
    "_point_tracking_force",
    "_point_tracking_displacement",
    "_tracked_distances",
    "_point_tracking",
)
PROCESSING_LABELS = frozenset({"processed", "smoothed", "corrected", "original"})


def playback_interval_ms(fps: float, speed: float) -> int:
    """Return a practical Qt timer interval for the requested playback rate."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("The annotated recording has no valid frame rate.")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be greater than zero.")
    return max(1, round(1000.0 / (fps * speed)))


def _recording_from_trace_file(path: Path) -> str:
    stem = Path(path).stem
    lower = stem.casefold()
    for suffix in TRACE_FILE_SUFFIXES:
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return ""


def _recording_from_video(path: Path) -> str:
    name = Path(path).name
    lower = name.casefold()
    for suffix in ANNOTATED_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _trace_parts(trace_name: str) -> list[str]:
    parts = [part.strip() for part in str(trace_name).split("|") if part.strip()]
    while parts and parts[-1].casefold() in PROCESSING_LABELS:
        parts.pop()
    return parts


def _benchmark_video_directory(source_path: Path, trace_name: str) -> Path | None:
    """Map a benchmark display name to the worker's configuration video folder."""
    settings_path = source_path.parent / "kalman_benchmark_settings.csv"
    parts = _trace_parts(trace_name)
    if not settings_path.is_file() or len(parts) < 2:
        return None
    configuration = parts[0].casefold()
    matches = []
    try:
        with settings_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("name", "")).strip().casefold() != configuration:
                    continue
                folder = str(row.get("folder", "")).strip()
                if folder and Path(folder).name == folder:
                    matches.append(source_path.parent / folder / "videos")
    except (OSError, csv.Error):
        return None
    return matches[0] if len(matches) == 1 else None


def _candidate_directories(source_path: Path, trace_name: str) -> list[Path]:
    source = Path(source_path)
    directories = []
    benchmark = _benchmark_video_directory(source, trace_name)
    if benchmark is not None:
        directories.append(benchmark)
    for base in (source.parent, source.parent.parent):
        directories.extend(
            (
                base / "overlays",
                base / "videos",
                base / "all_overlaid_videos",
            )
        )
    unique = []
    seen = set()
    for directory in directories:
        identity = str(directory.resolve()).casefold()
        if identity not in seen:
            seen.add(identity)
            unique.append(directory)
    return unique


def annotated_recording_candidates(source_path: Path, trace_name: str) -> list[Path]:
    """Return generated videos from only the expected folders beside a trace file."""
    candidates = []
    seen = set()
    for directory in _candidate_directories(Path(source_path), trace_name):
        if not directory.is_dir():
            continue
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            continue
        for candidate in entries:
            lower = candidate.name.casefold()
            if not candidate.is_file() or not any(
                lower.endswith(suffix) for suffix in ANNOTATED_SUFFIXES
            ):
                continue
            identity = str(candidate.resolve()).casefold()
            if identity not in seen:
                seen.add(identity)
                candidates.append(candidate)
    return candidates


def _source_method(source_path: Path, trace_name: str) -> str:
    source_lower = source_path.stem.casefold()
    first_part = _trace_parts(trace_name)[:1]
    first = first_part[0].casefold() if first_part else ""
    if "point_tracking" in source_lower or first == "point tracking":
        return "point_tracking"
    if (
        source_lower in {"stimtrace_force_traces", "combined_force_results"}
        or source_lower.endswith(("_stimtrace_tracking", "_pillar_displacement"))
        or first == "stimtrace"
    ):
        return "stimtrace"
    if source_lower.startswith("kalman_benchmark"):
        return "benchmark"
    return ""


def _candidate_score(
    candidate: Path,
    trace_name: str,
    source_path: Path,
) -> int | None:
    recording = _recording_from_video(candidate).casefold()
    direct_recording = _recording_from_trace_file(source_path).casefold()
    parts = [part.casefold() for part in _trace_parts(trace_name)]
    direct_match = bool(direct_recording and recording == direct_recording)
    trace_match = any(
        part == recording or part.startswith(f"{recording}_")
        for part in parts
    )
    if not direct_match and not trace_match:
        return None

    score = 100 if direct_match else 50
    lower = candidate.name.casefold()
    method = _source_method(source_path, trace_name)
    if method == "point_tracking" and lower.endswith("_point_tracking_overlay.avi"):
        score += 20
    elif method == "stimtrace" and lower.endswith("_overlay.avi") and not lower.endswith(
        "_point_tracking_overlay.avi"
    ):
        score += 20
    elif method == "benchmark" and lower.endswith("_tracked_trace.avi"):
        score += 20
    if "kalman_benchmark" in str(source_path.parent).casefold() and lower.endswith(
        "_tracked_trace.avi"
    ):
        score += 10
    return score


def find_annotated_recording(trace_name: str, source_path: Path | None) -> Path | None:
    """Find the unambiguous generated recording associated with one loaded trace."""
    if source_path is None:
        return None
    source = Path(source_path)
    scored = []
    for candidate in annotated_recording_candidates(source, trace_name):
        score = _candidate_score(candidate, trace_name, source)
        if score is not None:
            scored.append((score, candidate))
    if not scored:
        return None
    best_score = max(score for score, _candidate in scored)
    best = [candidate for score, candidate in scored if score == best_score]
    return best[0] if len(best) == 1 else None


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remainder:05.2f}"
    return f"{minutes}:{remainder:05.2f}"


class AnnotatedRecordingViewer(QWidget):
    """Native-resolution, scrollable OpenCV player for one annotated recording."""

    def __init__(self, video_path: Path, parent=None) -> None:
        super().__init__(parent, Qt.Window)
        self.video_path = Path(video_path)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(f"Annotated recording - {self.video_path.name}")

        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to watch annotated recordings.") from error
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Could not open the annotated recording:\n{self.video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            not math.isfinite(self.fps)
            or self.fps <= 0
            or self.frame_count <= 0
            or self.frame_width <= 0
            or self.frame_height <= 0
        ):
            self.capture.release()
            raise ValueError(
                "The annotated recording has incomplete frame-rate, frame-count, or size metadata."
            )

        self.current_frame = -1
        self.current_image: QImage | None = None
        self.playback_speed = 1.0
        self._updating_slider = False
        self._resume_after_seek = False
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(playback_interval_ms(self.fps, self.playback_speed))
        self.timer.timeout.connect(self._advance_frame)
        if not self._show_frame(0, seek=True):
            self.capture.release()
            raise ValueError(f"The annotated recording contains no readable frames:\n{self.video_path}")
        self._set_initial_size()
        self._render_current_image()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        heading = QLabel(
            f"{self.video_path.name}  |  {self.frame_width} × {self.frame_height}  |  "
            f"{self.fps:.3g} fps"
        )
        heading.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(heading)

        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.frame_label)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.viewport().installEventFilter(self)
        layout.addWidget(self.scroll_area, 1)

        timeline = QHBoxLayout()
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, self.frame_count - 1)
        self.position_slider.setSingleStep(1)
        self.position_slider.setPageStep(max(1, round(self.fps)))
        self.position_slider.sliderPressed.connect(self._begin_seek)
        self.position_slider.sliderReleased.connect(self._finish_seek)
        self.position_slider.valueChanged.connect(self._slider_value_changed)
        timeline.addWidget(self.position_slider, 1)
        self.position_label = QLabel()
        self.position_label.setMinimumWidth(190)
        self.position_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        timeline.addWidget(self.position_label)
        layout.addLayout(timeline)

        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)
        self.save_frame_button = QPushButton("Save frame")
        self.save_frame_button.setToolTip("Save the current full-resolution video frame as a PNG.")
        self.save_frame_button.clicked.connect(self.save_current_frame)
        controls.addWidget(self.save_frame_button)
        self.save_recording_button = QPushButton("Save recording")
        self.save_recording_button.setToolTip("Save a copy of this annotated recording.")
        self.save_recording_button.clicked.connect(self.save_recording)
        controls.addWidget(self.save_recording_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Speed"))
        self.speed_group = QButtonGroup(self)
        self.speed_group.setExclusive(True)
        self.speed_buttons = {}
        for label, speed in (("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0)):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(speed == 1.0)
            button.clicked.connect(
                lambda _checked=False, requested_speed=speed: self.set_playback_speed(
                    requested_speed
                )
            )
            self.speed_group.addButton(button)
            self.speed_buttons[speed] = button
            controls.addWidget(button)
        controls.addStretch(1)
        layout.addLayout(controls)

    def _set_initial_size(self) -> None:
        screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        if screen is None:
            self.resize(min(1200, self.frame_width + 40), min(900, self.frame_height + 150))
            return
        available = screen.availableGeometry()
        maximum_width = max(320, round(available.width() * 0.9))
        maximum_height = max(240, round(available.height() * 0.9))
        desired_width = self.frame_width + 44
        desired_height = self.frame_height + 158
        self.resize(
            min(maximum_width, max(640, desired_width)),
            min(maximum_height, max(420, desired_height)),
        )

    def _show_frame(self, frame_index: int, *, seek: bool) -> bool:
        frame_index = max(0, min(int(frame_index), self.frame_count - 1))
        if seek:
            self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.pause()
            return False
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        self.current_image = image
        self._render_current_image()
        self.current_frame = frame_index
        self._updating_slider = True
        self.position_slider.setValue(frame_index)
        self._updating_slider = False
        self._update_position_label()
        return True

    def _render_current_image(self) -> None:
        """Render the current frame fitted inside the viewport without scrolling."""
        if self.current_image is None or not hasattr(self, "frame_label"):
            return
        pixmap = QPixmap.fromImage(self.current_image)
        viewport_size = self.scroll_area.viewport().size()
        if viewport_size.width() > 1 and viewport_size.height() > 1:
            pixmap = pixmap.scaled(
                viewport_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.frame_label.setPixmap(pixmap)
        self.frame_label.resize(pixmap.size())

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.scroll_area.viewport()
            and event.type() == QEvent.Resize
        ):
            self._render_current_image()
        return super().eventFilter(watched, event)

    def _update_position_label(self) -> None:
        current_seconds = max(0, self.current_frame) / self.fps
        total_seconds = max(0, self.frame_count - 1) / self.fps
        self.position_label.setText(
            f"{_format_time(current_seconds)} / {_format_time(total_seconds)}  "
            f"({max(0, self.current_frame) + 1}/{self.frame_count})"
        )

    def toggle_playback(self) -> None:
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.current_frame >= self.frame_count - 1:
            if not self._show_frame(0, seek=True):
                return
        self.timer.start()
        self.play_button.setText("Pause")

    def pause(self) -> None:
        if hasattr(self, "timer"):
            self.timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("Play")

    def set_playback_speed(self, speed: float) -> None:
        self.playback_speed = float(speed)
        interval = playback_interval_ms(self.fps, self.playback_speed)
        was_playing = self.timer.isActive()
        self.timer.setInterval(interval)
        if was_playing:
            self.timer.start()

    def _advance_frame(self) -> None:
        next_frame = self.current_frame + 1
        if next_frame >= self.frame_count:
            self.pause()
            return
        if not self._show_frame(next_frame, seek=False):
            self.pause()

    def _begin_seek(self) -> None:
        self._resume_after_seek = self.timer.isActive()
        self.timer.stop()

    def _finish_seek(self) -> None:
        self._show_frame(self.position_slider.value(), seek=True)
        if self._resume_after_seek:
            self.play()
        self._resume_after_seek = False

    def _slider_value_changed(self, value: int) -> None:
        if self._updating_slider or self.position_slider.isSliderDown():
            return
        self._show_frame(value, seek=True)

    def save_current_frame(self) -> None:
        if self.current_image is None:
            QMessageBox.warning(self, "No frame available", "There is no video frame to save.")
            return
        default_name = f"{self.video_path.stem}_frame_{self.current_frame + 1:06d}.png"
        target, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save current frame",
            str(self.video_path.with_name(default_name)),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;BMP image (*.bmp)",
        )
        if not target:
            return
        path = Path(target)
        if not path.suffix:
            path = path.with_suffix(".png")
        if not self.current_image.save(str(path)):
            QMessageBox.critical(self, "Could not save frame", f"Could not write:\n{path}")

    def save_recording(self) -> None:
        target, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save annotated recording",
            str(self.video_path),
            "Video files (*.avi *.mp4 *.mov *.mkv);;All files (*.*)",
        )
        if not target:
            return
        destination = Path(target)
        if not destination.suffix:
            destination = destination.with_suffix(self.video_path.suffix)
        try:
            if destination.resolve() == self.video_path.resolve():
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.video_path, destination)
        except OSError as error:
            QMessageBox.critical(
                self,
                "Could not save recording",
                f"Could not copy the annotated recording:\n{error}",
            )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.pause()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        super().closeEvent(event)
