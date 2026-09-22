"""Frame extraction, polygon annotation, and Colab training submission."""
from __future__ import annotations

import json
import hashlib
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF, QShortcut
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app_logging import get_logger
from background_tasks import BackgroundFunctionThread
from job_utils import duplicate_output_stems
from ui_actions import set_action_icon


LOGGER = get_logger("model_training")


DATASET_EFFORTS = {
    "minimum": {
        "label": "Minimum (60 frames)",
        "recordings": 15,
        "frames_per_recording": 4,
        "minimum_train_frames": 20,
        "minimum_validation_frames": 5,
    },
    "recommended": {
        "label": "Recommended (125 frames)",
        "recordings": 25,
        "frames_per_recording": 5,
        "minimum_train_frames": 40,
        "minimum_validation_frames": 10,
    },
    "robust": {
        "label": "Robust (200 frames)",
        "recordings": 40,
        "frames_per_recording": 5,
        "minimum_train_frames": 60,
        "minimum_validation_frames": 15,
    },
}

SUPPORTED_DATASET_STRATEGIES = {"phase_aware_v1", "migrated_existing_v1"}


def source_video_id(path: Path) -> str:
    """Create a stable identifier for one input recording."""
    return hashlib.sha256(str(path.resolve()).casefold().encode("utf-8")).hexdigest()[:16]


def legacy_frame_source(frame: Path) -> tuple[str, int]:
    """Recover a recording key and frame index from the retired extracted-frame naming."""
    match = re.match(r"^(?P<source>.+?)[_-]f(?P<index>\d+)$", frame.stem, re.IGNORECASE)
    if match:
        return match.group("source"), int(match.group("index"))
    return frame.stem, 0


def normalise_values(values: np.ndarray) -> np.ndarray:
    """Return finite values scaled to 0-1 without amplifying a flat signal."""
    values = np.asarray(values, dtype=float)
    low, high = np.nanpercentile(values, (5, 95))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-9:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _best_index(indices: np.ndarray, score: np.ndarray, used: set[int]) -> int | None:
    """Select the highest scoring unused index from a bounded candidate range."""
    if indices.size == 0:
        return None
    for index in indices[np.argsort(score[indices])[::-1]]:
        candidate = int(index)
        if candidate not in used:
            return candidate
    return None


def phase_aware_frame_indices(
    motion: np.ndarray,
    sharpness: np.ndarray,
    state_distance: np.ndarray,
    fps: float,
    requested_count: int,
) -> list[dict[str, float | int | str]]:
    """Choose sharp frames spanning rest, transition, deformation, and recovery.

    Movement locates the dynamic parts of a twitch. The chosen transition frames are
    weighted toward sharpness so annotations do not preferentially contain blur.
    """
    count = len(motion)
    if count == 0:
        return []
    requested_count = max(1, min(int(requested_count), count))
    motion_score = normalise_values(motion)
    sharpness_score = normalise_values(sharpness)
    deformation_score = normalise_values(state_distance)
    quality = 0.75 * sharpness_score + 0.25 * (1.0 - motion_score)
    transition_quality = 0.60 * motion_score + 0.40 * sharpness_score
    peak_quality = 0.70 * deformation_score + 0.30 * sharpness_score
    peak = int(np.argmax(peak_quality))
    search = max(2, min(count // 3, round(max(1.0, float(fps)) * 1.25)))
    before = np.arange(max(0, peak - search), peak + 1)
    after = np.arange(peak, min(count, peak + search + 1))
    used: set[int] = set()

    def choose(indices: np.ndarray, score: np.ndarray, phase: str) -> dict[str, float | int | str] | None:
        index = _best_index(indices, score, used)
        if index is None:
            return None
        used.add(index)
        return {
            "frame_index": index,
            "phase": phase,
            "motion_score": float(motion_score[index]),
            "sharpness_score": float(sharpness_score[index]),
            "deformation_score": float(deformation_score[index]),
        }

    rise = choose(before, transition_quality, "contraction transition")
    fall = choose(after, transition_quality, "relaxation transition")
    before_rest = np.arange(max(0, peak - 2 * search), max(1, peak - search // 3))
    after_rest = np.arange(min(count - 1, peak + search // 3), min(count, peak + 2 * search + 1))
    selected = [
        choose(before_rest, quality, "relaxed before contraction"),
        rise,
        choose(np.arange(max(0, peak - 2), min(count, peak + 3)), peak_quality, "maximum deformation"),
        fall,
        choose(after_rest, quality, "relaxed after contraction"),
    ]
    selected = [item for item in selected if item is not None]

    # Short or weakly sampled recordings cannot support a confident five-state cycle.
    while len(selected) < requested_count:
        candidate = _best_index(np.arange(count), 0.5 * quality + 0.5 * deformation_score, used)
        if candidate is None:
            break
        used.add(candidate)
        selected.append({
            "frame_index": candidate,
            "phase": "additional representative frame",
            "motion_score": float(motion_score[candidate]),
            "sharpness_score": float(sharpness_score[candidate]),
            "deformation_score": float(deformation_score[candidate]),
        })
    selected.sort(key=lambda item: int(item["frame_index"]))
    return selected[:requested_count]


def analyse_video_for_phase_selection(video: Path, requested_count: int) -> dict:
    """Read a video once and derive phase-aware annotation candidates."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open training video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    features: list[np.ndarray] = []
    motion = [0.0]
    sharpness: list[float] = []
    previous = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            preview = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
            central = preview[12:84, 12:84]
            features.append(central.astype(np.float32).reshape(-1))
            sharpness.append(float(cv2.Laplacian(central, cv2.CV_32F).var()))
            if previous is not None:
                motion.append(float(np.mean(np.abs(central.astype(np.float32) - previous))))
            previous = central
    finally:
        capture.release()
    if not features:
        raise ValueError(f"Could not decode any frames from training video: {video.name}")
    feature_array = np.stack(features)
    motion_array = np.asarray(motion[:len(features)], dtype=float)
    sharpness_array = np.asarray(sharpness, dtype=float)
    rest_indices = np.argsort(motion_array)[:max(1, len(feature_array) // 5)]
    rest_reference = np.median(feature_array[rest_indices], axis=0)
    state_distance = np.mean(np.abs(feature_array - rest_reference), axis=1)
    candidates = phase_aware_frame_indices(
        motion_array,
        sharpness_array,
        state_distance,
        fps,
        requested_count,
    )
    descriptor = np.asarray([
        float(np.mean(feature_array)),
        float(np.std(feature_array)),
        float(np.median(motion_array)),
        float(np.percentile(motion_array, 95)),
        float(np.percentile(state_distance, 95)),
        float(np.median(sharpness_array)),
    ])
    return {
        "path": str(video),
        "source_video": video.name,
        "source_video_id": source_video_id(video),
        "fps": fps if fps > 0 else 0.0,
        "frame_count": len(features) or frame_count,
        "descriptor": descriptor.tolist(),
        "candidates": candidates,
    }


def select_diverse_recordings(records: list[dict], maximum: int) -> list[dict]:
    """Select visually diverse recordings using deterministic farthest-point sampling."""
    if len(records) <= maximum:
        return list(records)
    descriptors = np.asarray([record["descriptor"] for record in records], dtype=float)
    scale = np.std(descriptors, axis=0)
    scale[scale < 1e-9] = 1.0
    descriptors = (descriptors - np.mean(descriptors, axis=0)) / scale
    selected = [int(np.argmax(np.linalg.norm(descriptors, axis=1)))]
    while len(selected) < maximum:
        distances = np.min(
            np.linalg.norm(descriptors[:, None, :] - descriptors[selected][None, :, :], axis=2),
            axis=1,
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    return [records[index] for index in sorted(selected)]


def assign_recording_splits(records: list[dict], validation_fraction: float = 0.20) -> None:
    """Assign complete recordings to train or validation with feature-balanced coverage."""
    if not records:
        return
    validation_count = max(1, round(len(records) * validation_fraction))
    descriptors = np.asarray([record["descriptor"] for record in records], dtype=float)
    descriptors -= np.mean(descriptors, axis=0, keepdims=True)
    if len(records) > 1:
        _unused, _singular, vectors = np.linalg.svd(descriptors, full_matrices=False)
        projection = descriptors @ vectors[0]
    else:
        projection = np.zeros(1)
    ordered = np.argsort(projection)
    positions = np.linspace(0, len(records) - 1, validation_count, dtype=int)
    validation = {int(ordered[position]) for position in positions}
    for index, record in enumerate(records):
        record["split"] = "validation" if index in validation else "training"


def write_phase_aware_frames(records: list[dict], frames_dir: Path) -> dict[str, dict]:
    """Write selected source frames and return their immutable provenance metadata."""
    frame_metadata: dict[str, dict] = {}
    for record in records:
        video = Path(record["path"])
        targets = {int(item["frame_index"]): item for item in record["candidates"]}
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Could not reopen training video: {video.name}")
        frame_index = 0
        try:
            while targets:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                selection = targets.pop(frame_index, None)
                if selection is not None:
                    filename = f"{video.stem}_f{frame_index:06d}.png"
                    target = frames_dir / filename
                    if not cv2.imwrite(str(target), frame):
                        raise RuntimeError(f"Could not write {target}")
                    frame_metadata[filename] = {
                        **selection,
                        "source_video": record["source_video"],
                        "source_video_id": record["source_video_id"],
                        "split": record["split"],
                        "time_s": frame_index / record["fps"] if record["fps"] > 0 else None,
                    }
                frame_index += 1
        finally:
            capture.release()
        if targets:
            raise ValueError(f"Could not extract all selected frames from {video.name}")
    return frame_metadata


def propagate_polygons_to_frame(
    source_frame: Path,
    target_frame: Path,
    polygons: list[dict],
) -> tuple[list[dict], float, float, int]:
    """Copy polygons and align them with sparse optical-flow features in the object mask.

    The output is an editable draft for the user to review before leaving the frame.
    """
    source = cv2.imread(str(source_frame), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(target_frame), cv2.IMREAD_GRAYSCALE)
    if source is None or target is None:
        raise ValueError("Could not read the source or target annotation frame.")
    if source.shape != target.shape:
        raise ValueError("Mask propagation requires source and target frames of the same size.")
    if not polygons:
        raise ValueError("The source frame has no saved polygon to copy.")

    mask = np.zeros(source.shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon.get("points", []), dtype=np.float32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    if not np.any(mask):
        raise ValueError("The source annotation does not contain a usable polygon.")
    mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
    features = cv2.goodFeaturesToTrack(
        source,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=4,
        mask=mask,
        blockSize=7,
    )
    if features is None or len(features) < 4:
        raise ValueError(
            "StimTrace could not find enough image detail inside the annotated object to align this mask."
        )
    tracked, status, _errors = cv2.calcOpticalFlowPyrLK(
        source,
        target,
        features,
        None,
        winSize=(41, 41),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    valid = status.reshape(-1).astype(bool) if status is not None else np.zeros(len(features), dtype=bool)
    if tracked is None or valid.sum() < 4:
        raise ValueError("Too few object features could be tracked into the selected frame.")
    displacement = (tracked.reshape(-1, 2) - features.reshape(-1, 2))[valid]
    dx, dy = np.median(displacement, axis=0)
    maximum_shift = max(12.0, min(source.shape) * 0.12)
    if float(np.hypot(dx, dy)) > maximum_shift:
        raise ValueError(
            "The estimated mask movement is implausibly large. Draw this frame manually instead."
        )
    copied = json.loads(json.dumps(polygons))
    for polygon in copied:
        shifted = []
        for x, y in polygon.get("points", []):
            shifted.append((
                float(np.clip(float(x) + dx, 0, source.shape[1] - 1)),
                float(np.clip(float(y) + dy, 0, source.shape[0] - 1)),
            ))
        polygon["points"] = shifted
    return copied, float(dx), float(dy), int(valid.sum())


def create_training_archive(project_dir: Path) -> tuple[Path, int]:
    """Build a training archive without accessing Qt-owned state."""
    frames = project_dir / "frames"
    annotations = project_dir / "annotations"
    masks = project_dir / "masks"
    annotated = [
        frame
        for frame in frames.iterdir()
        if (masks / f"{frame.stem}.png").exists()
        and (annotations / f"{frame.stem}.json").exists()
    ]
    if len(annotated) < 2:
        raise ValueError("Save annotations for at least two frames before training.")
    positive_masks = sum(np_any_mask(masks / f"{frame.stem}.png") for frame in annotated)
    if not positive_masks:
        raise ValueError(
            "Training needs at least one frame with a pillar mask as well as any negative-control frames."
        )
    project_file = project_dir / "project.json"
    try:
        project = json.loads(project_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        project = {}
    if project.get("selection_strategy") not in SUPPORTED_DATASET_STRATEGIES:
        raise ValueError(
            "This is an older annotation project. Create a new training dataset before training."
        )
    frame_metadata = project.get("frame_metadata", {})
    dataset_manifest = {
        "schema_version": 1,
        "selection_strategy": project["selection_strategy"],
        "effort": project.get("effort", ""),
        "frames": [],
    }
    missing_metadata = [frame.name for frame in annotated if frame.name not in frame_metadata]
    if missing_metadata:
        raise ValueError(
                "This training dataset is missing split metadata for annotated frame(s): "
            + ", ".join(missing_metadata[:5])
        )
    source_splits: dict[str, set[str]] = {}
    for frame in annotated:
        metadata = dict(frame_metadata[frame.name])
        split = metadata.get("split")
        source = str(metadata.get("source_video_id", ""))
        if split not in {"training", "validation"} or not source:
            raise ValueError(f"Invalid split metadata for {frame.name}.")
        source_splits.setdefault(source, set()).add(str(split))
        dataset_manifest["frames"].append({"image": frame.name, **metadata})
    leaked_sources = [source for source, splits in source_splits.items() if len(splits) > 1]
    if leaked_sources:
        raise ValueError(
            "A source recording appears in both training and validation. "
                "Create a new training dataset to repair the split."
        )
    training_count = sum(item["split"] == "training" for item in dataset_manifest["frames"])
    validation_count = sum(item["split"] == "validation" for item in dataset_manifest["frames"])
    effort = DATASET_EFFORTS.get(str(project.get("effort", "recommended")), DATASET_EFFORTS["recommended"])
    if training_count < effort["minimum_train_frames"] or validation_count < effort["minimum_validation_frames"]:
        raise ValueError(
            f"Annotate at least {effort['minimum_train_frames']} development and "
            f"{effort['minimum_validation_frames']} quality-check frames before training "
            f"({training_count} development, {validation_count} quality-check currently annotated)."
        )
    dataset_manifest["summary"] = {
        "annotated_frames": len(annotated),
        "positive_mask_frames": positive_masks,
        "negative_control_frames": len(annotated) - positive_masks,
        "training_frames": training_count,
        "validation_frames": validation_count,
        "training_recordings": len({
            item["source_video_id"] for item in dataset_manifest["frames"]
            if item["split"] == "training"
        }),
        "validation_recordings": len({
            item["source_video_id"] for item in dataset_manifest["frames"]
            if item["split"] == "validation"
        }),
    }
    archive = project_dir / "training_dataset.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for folder in (frames, annotations, masks):
            for path in folder.iterdir():
                if path.is_file():
                    bundle.write(path, path.relative_to(project_dir))
        if project_file.exists():
            bundle.write(project_file, project_file.name)
        bundle.writestr("dataset_manifest.json", json.dumps(dataset_manifest, indent=2))
    return archive, len(annotated)


class AnnotationCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.pixmap = QPixmap()
        self.image_path: Path | None = None
        self.polygons: list[dict] = []
        self.current_points: list[tuple[float, float]] = []
        self.current_label = "Pillar"
        self.zoom_factor = 1.0
        self.pan_offset = QPointF()
        self.pan_start: QPointF | None = None
        self.setCursor(Qt.CrossCursor)

    def set_image(self, path: Path, polygons: list[dict]):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError(f"Could not open image: {path}")
        self.image_path = path
        self.pixmap = pixmap
        self.polygons = polygons
        self.current_points = []
        self.update()

    def fit_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        scaled = self.pixmap.size()
        scaled.scale(self.size(), Qt.KeepAspectRatio)
        left = (self.width() - scaled.width()) / 2
        top = (self.height() - scaled.height()) / 2
        return QRectF(left, top, scaled.width(), scaled.height())

    def image_rect(self) -> QRectF:
        fitted = self.fit_rect()
        width = fitted.width() * self.zoom_factor
        height = fitted.height() * self.zoom_factor
        return QRectF(
            (self.width() - width) / 2 + self.pan_offset.x(),
            (self.height() - height) / 2 + self.pan_offset.y(),
            width,
            height,
        )

    def reset_view(self):
        self.zoom_factor = 1.0
        self.pan_offset = QPointF()
        self.update()

    def clamp_pan(self):
        rect = self.image_rect()
        margin = 40.0
        dx = 0.0
        dy = 0.0
        if rect.right() < margin:
            dx = margin - rect.right()
        elif rect.left() > self.width() - margin:
            dx = self.width() - margin - rect.left()
        if rect.bottom() < margin:
            dy = margin - rect.bottom()
        elif rect.top() > self.height() - margin:
            dy = self.height() - margin - rect.top()
        self.pan_offset += QPointF(dx, dy)

    def widget_to_image(self, point: QPointF) -> tuple[float, float] | None:
        rect = self.image_rect()
        if not rect.contains(point) or self.pixmap.isNull():
            return None
        x = (point.x() - rect.left()) * self.pixmap.width() / rect.width()
        y = (point.y() - rect.top()) * self.pixmap.height() / rect.height()
        return float(x), float(y)

    def image_to_widget(self, point: tuple[float, float]) -> QPointF:
        rect = self.image_rect()
        return QPointF(
            rect.left() + point[0] * rect.width() / self.pixmap.width(),
            rect.top() + point[1] * rect.height() / self.pixmap.height(),
        )

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self.pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.RightButton:
            self.undo_point()
            return
        if event.button() != Qt.LeftButton:
            return
        mapped = self.widget_to_image(event.position())
        if mapped:
            self.current_points.append(mapped)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.pan_start is not None:
            movement = event.position() - self.pan_start
            self.pan_offset += movement
            self.pan_start = event.position()
            self.clamp_pan()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            self.pan_start = None
            self.setCursor(Qt.CrossCursor)

    def wheelEvent(self, event):
        if self.pixmap.isNull() or not self.image_rect().contains(event.position()):
            return
        old_rect = self.image_rect()
        image_x = (event.position().x() - old_rect.left()) / old_rect.width()
        image_y = (event.position().y() - old_rect.top()) / old_rect.height()
        step = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        new_zoom = min(12.0, max(1.0, self.zoom_factor * step))
        if new_zoom == self.zoom_factor:
            return
        self.zoom_factor = new_zoom
        fitted = self.fit_rect()
        new_width = fitted.width() * new_zoom
        new_height = fitted.height() * new_zoom
        desired_left = event.position().x() - image_x * new_width
        desired_top = event.position().y() - image_y * new_height
        self.pan_offset = QPointF(
            desired_left - (self.width() - new_width) / 2,
            desired_top - (self.height() - new_height) / 2,
        )
        self.clamp_pan()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            mapped = self.widget_to_image(event.position())
            if mapped and (not self.current_points or self.current_points[-1] != mapped):
                self.current_points.append(mapped)
            self.finish_polygon()

    def finish_polygon(self):
        if len(self.current_points) >= 3:
            self.polygons.append({"label": self.current_label, "points": list(self.current_points)})
        self.current_points = []
        self.update()

    def undo_point(self):
        if self.current_points:
            self.current_points.pop()
        elif self.polygons:
            self.polygons.pop()
        self.update()

    def clear_polygons(self):
        self.current_points = []
        self.polygons = []
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15171a"))
        if self.pixmap.isNull():
            painter.setPen(QColor("#aeb6c0"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open or create an annotation project")
            return
        rect = self.image_rect()
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        fill_colors = [
            QColor(0, 220, 150, 70),
            QColor(255, 190, 0, 70),
            QColor(80, 160, 255, 70),
            QColor(235, 90, 140, 70),
        ]
        for index, polygon in enumerate(self.polygons):
            points = QPolygonF([self.image_to_widget(tuple(point)) for point in polygon["points"]])
            painter.setPen(QPen(fill_colors[index % len(fill_colors)].lighter(150), 2))
            painter.setBrush(fill_colors[index % len(fill_colors)])
            painter.drawPolygon(points)
        if self.current_points:
            points = QPolygonF([self.image_to_widget(point) for point in self.current_points])
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(points)
            for point in points:
                painter.drawEllipse(point, 3, 3)


class ModelTrainingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.project_dir: Path | None = None
        self.project_metadata: dict = {}
        self.frames: list[Path] = []
        self.current_index = -1
        self.propagated_draft = False
        self.training_progress_id = ""
        self.trained_model_file_id = ""
        self.trained_model_name = ""
        self.extraction_thread: BackgroundFunctionThread | None = None
        self.training_submit_thread: BackgroundFunctionThread | None = None
        self.training_poll_thread: BackgroundFunctionThread | None = None
        self.training_download_thread: BackgroundFunctionThread | None = None
        self.training_package_path: Path | None = None
        self.training_timer = QTimer(self)
        self.training_timer.setInterval(4000)
        self.training_timer.timeout.connect(self.poll_training)
        self._compact_annotation_toolbar: bool | None = None
        self._build_ui()
        self._setup_shortcuts()

    def _build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(18, 14, 18, 14)
        header = QHBoxLayout()
        title = QLabel("Model Training")
        title.setProperty("role", "title")
        self.training_account_status = QLabel("Not signed in")
        self.training_account_status.setProperty("role", "muted")
        self.training_account_status.setWordWrap(True)
        self.training_account_status.setMinimumWidth(0)
        self.training_account_status.setMaximumHeight(54)
        self.training_account_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.training_sign_in_button = QPushButton("Sign in")
        set_action_icon(self.training_sign_in_button, "apply")
        self.training_sign_in_button.clicked.connect(self.sign_in_for_training)
        self.training_sign_out_button = QPushButton("Sign out")
        set_action_icon(self.training_sign_out_button, "cancel")
        self.training_sign_out_button.clicked.connect(self.sign_out_for_training)
        help_button = QPushButton("Help")
        set_action_icon(help_button, "help")
        help_button.clicked.connect(self.show_help)
        header.addWidget(title)
        header.addWidget(self.training_account_status, 1)
        header.addWidget(self.training_sign_in_button)
        header.addWidget(self.training_sign_out_button)
        header.addWidget(help_button)
        page_layout.addLayout(header)
        description = QLabel(
            "Extract representative video frames, draw object masks, then fine-tune a new "
            "segmentation model using the running Colab worker."
        )
        description.setWordWrap(True)
        description.setProperty("role", "subtitle")
        page_layout.addWidget(description)

        compute_row = QHBoxLayout()
        compute_label = QLabel("Compute")
        compute_label.setStyleSheet("font-weight: 600;")
        cloud_training = QPushButton("Cloud")
        cloud_training.setCheckable(True)
        cloud_training.setChecked(True)
        cloud_training.setProperty("role", "mode")
        cloud_training.setToolTip("Custom-model training runs in the connected Colab runtime.")
        advanced_settings = QPushButton("Advanced settings")
        set_action_icon(advanced_settings, "settings")
        advanced_settings.setToolTip("Configure the active segmentation model and processing settings.")
        if self.main_window and hasattr(self.main_window, "open_settings"):
            advanced_settings.clicked.connect(self.main_window.open_settings)
        else:
            advanced_settings.setEnabled(False)
        compute_row.addWidget(compute_label)
        compute_row.addWidget(cloud_training)
        compute_row.addStretch(1)
        compute_row.addWidget(advanced_settings)
        page_layout.addLayout(compute_row)

        steps_row = QHBoxLayout()
        steps_row.setSpacing(8)
        self.training_workflow_steps = [
            self._workflow_step(1, "Google account", "Sign in"),
            self._workflow_step(2, "Training data", "Open or create project"),
            self._workflow_step(3, "Annotations", "Save masks"),
            self._workflow_step(4, "Training", "Submit to Colab"),
        ]
        for step in self.training_workflow_steps:
            step.setMinimumWidth(0)
            step.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            steps_row.addWidget(step, 1)
        page_layout.addLayout(steps_row)
        splitter = QSplitter(Qt.Horizontal)
        page_layout.addWidget(splitter, 1)
        controls = QWidget()
        controls.setMinimumWidth(0)
        controls.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        layout = QVBoxLayout(controls)
        # This widget lives inside a scroll area. Its layout must establish the
        # true content minimum; otherwise Qt may shrink the parent while child
        # widgets retain their own minimum heights and paint over later rows.
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        project_actions = QHBoxLayout()
        self.open_project_button = QPushButton("Open annotation project")
        set_action_icon(self.open_project_button, "open_folder")
        self.open_project_button.clicked.connect(self.open_project)
        self.extract_button = QPushButton("Create training dataset")
        set_action_icon(self.extract_button, "open")
        self.extract_button.clicked.connect(self.extract_frames)
        project_actions.addWidget(self.open_project_button)
        project_actions.addWidget(self.extract_button)
        for button in (self.open_project_button, self.extract_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        layout.addLayout(project_actions)
        self.project_label = QLabel("No project selected")
        self.project_label.setWordWrap(True)
        self.project_label.setProperty("role", "muted")
        layout.addWidget(self.project_label)

        extraction_box = QGroupBox("Automatic frame selection")
        extraction_form = QFormLayout(extraction_box)
        extraction_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        extraction_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.dataset_effort = QComboBox()
        for key, values in DATASET_EFFORTS.items():
            self.dataset_effort.addItem(values["label"], key)
        self.dataset_effort.setMinimumWidth(190)
        self.dataset_effort.setCurrentIndex(self.dataset_effort.findData("recommended"))
        self.dataset_effort.currentIndexChanged.connect(self.update_dataset_plan)
        extraction_form.addRow("Dataset size", self.dataset_effort)
        self.dataset_plan_label = QLabel()
        self.dataset_plan_label.setMinimumWidth(190)
        self.dataset_plan_label.setProperty("role", "muted")
        extraction_form.addRow("Frame plan", self.dataset_plan_label)
        self.extraction_progress = QProgressBar()
        self.extraction_progress.setRange(0, 0)
        self.extraction_progress.setTextVisible(False)
        self.extraction_progress.setVisible(False)
        extraction_form.addRow("Activity", self.extraction_progress)
        layout.addWidget(extraction_box)

        self.frame_list = QListWidget()
        self.frame_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.frame_list.setMinimumHeight(260)
        self.frame_list.setToolTip("Green rows are annotated; red rows still need an annotation.")
        self.frame_list.currentRowChanged.connect(self.load_frame)
        layout.addWidget(self.frame_list, 3)
        self.annotation_count = QLabel("Annotated: 0 / 0")
        self.annotation_count.setToolTip(
            "Saved annotations include empty masks for negative-control frames."
        )
        layout.addWidget(self.annotation_count)

        label_box = QGroupBox("Annotation")
        label_layout = QFormLayout(label_box)
        label_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        label_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.class_name = QComboBox()
        self.class_name.setEditable(True)
        self.class_name.addItems(["Pillar", "Tissue", "Other"])
        self.class_name.currentTextChanged.connect(self.set_class_name)
        label_layout.addRow("Object class", self.class_name)
        layout.addWidget(label_box)

        self.undo_button = QPushButton("Undo")
        set_action_icon(self.undo_button, "previous")
        self.undo_button.clicked.connect(self.canvas_undo)
        self.finish_button = QPushButton("Finish polygon")
        set_action_icon(self.finish_button, "apply")
        self.finish_button.clicked.connect(self.canvas_finish)
        self.clear_button = QPushButton("Clear mask")
        set_action_icon(self.clear_button, "clear")
        self.clear_button.clicked.connect(self.canvas_clear)
        self.save_button = QPushButton("Save annotation and open next frame")
        set_action_icon(self.save_button, "save")
        self.save_button.clicked.connect(self.save_annotation)
        self.auto_mask_checkbox = QCheckBox("Auto-mask next frame (recommended)")
        self.auto_mask_checkbox.setObjectName("autoMaskNextFrame")
        self.auto_mask_checkbox.setChecked(True)
        self.auto_mask_checkbox.setMinimumHeight(40)
        self.auto_mask_checkbox.setToolTip(
            "After saving or moving to the next frame, copy the closest saved mask from the "
            "same recording and align it automatically. Review the draft before continuing."
        )

        training_box = QGroupBox("Colab training")
        training_form = QFormLayout(training_box)
        training_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        training_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.model_name = QLineEdit("StimTrace_custom_model")
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(30)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 32)
        self.batch_size.setValue(4)
        training_form.addRow("Model name", self.model_name)
        training_form.addRow("Epochs", self.epochs)
        training_form.addRow("Batch size", self.batch_size)
        layout.addWidget(training_box)
        self.train_button = QPushButton("Submit training to Colab")
        set_action_icon(self.train_button, "run")
        self.train_button.clicked.connect(self.submit_training)
        layout.addWidget(self.train_button)
        self.use_model_button = QPushButton("Use trained model for segmentation")
        set_action_icon(self.use_model_button, "apply")
        self.use_model_button.clicked.connect(self.use_trained_model)
        self.use_model_button.setEnabled(False)
        layout.addWidget(self.use_model_button)
        self.open_training_package_button = QPushButton("Open offline training package")
        set_action_icon(self.open_training_package_button, "open_folder")
        self.open_training_package_button.clicked.connect(self.open_training_package)
        self.open_training_package_button.setEnabled(False)
        layout.addWidget(self.open_training_package_button)
        for button in (
            self.train_button,
            self.use_model_button,
            self.open_training_package_button,
        ):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.training_label = QLabel("No training job submitted")
        self.training_label.setWordWrap(True)
        layout.addWidget(self.training_label)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 1000)
        self.training_progress.setValue(0)
        layout.addWidget(self.training_progress)

        display = QWidget()
        display.setMinimumWidth(0)
        display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        display_layout = QVBoxLayout(display)
        self.canvas = AnnotationCanvas()
        display_layout.addWidget(self.canvas, 1)
        mouse_legend = QLabel(
            "<b>Mouse controls</b>&nbsp;&nbsp; "
            "● Left click: add point &nbsp;&nbsp; "
            "◉ Double-left click: finish polygon &nbsp;&nbsp; "
            "● Right click: undo &nbsp;&nbsp; "
            "✥ Middle-button drag: pan &nbsp;&nbsp; "
            "↕ Wheel: zoom"
        )
        mouse_legend.setTextFormat(Qt.RichText)
        mouse_legend.setTextInteractionFlags(Qt.NoTextInteraction)
        mouse_legend.setAlignment(Qt.AlignCenter)
        mouse_legend.setWordWrap(True)
        mouse_legend.setProperty("role", "muted")
        mouse_legend.setToolTip(
            "Left click adds a polygon point. Double-left click finishes the polygon. "
            "Right click undoes. Drag with the middle mouse button to pan; use the wheel to zoom."
        )
        display_layout.addWidget(mouse_legend)
        previous = QPushButton("Previous frame")
        set_action_icon(previous, "previous")
        previous.clicked.connect(lambda: self.navigate(-1))
        next_button = QPushButton("Next frame")
        set_action_icon(next_button, "next")
        next_button.clicked.connect(lambda: self.navigate(1))
        reset_zoom = QPushButton("Reset zoom")
        set_action_icon(reset_zoom, "clear")
        reset_zoom.clicked.connect(self.canvas.reset_view)
        self.frame_status = QLabel("0/0")
        self.frame_status.setAlignment(Qt.AlignCenter)
        self.frame_status.setMinimumWidth(56)
        self.annotation_toolbar = QGridLayout()
        self.annotation_toolbar.setSpacing(8)
        self.annotation_toolbar_widgets = (
            previous,
            self.frame_status,
            next_button,
            reset_zoom,
            self.undo_button,
            self.finish_button,
            self.clear_button,
            self.save_button,
            self.auto_mask_checkbox,
        )
        for widget in self.annotation_toolbar_widgets:
            widget.setMinimumWidth(0)
        display_layout.addLayout(self.annotation_toolbar)

        # Keep the controls at their natural heights. Without an independent
        # scroll area, short windows forced Qt to compress form rows until
        # labels and spin boxes overlapped. The scroll pane can also be widened
        # with the splitter instead of imposing the previous 440 px ceiling.
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setMinimumWidth(390)
        self.controls_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.controls_scroll.setWidget(controls)
        self.training_controls = controls
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(display)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setMinimumHeight(120)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        splitter.setSizes([410, 1190])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.workspace_splitter = splitter
        self.status_label = QLabel("Create a project or open an existing annotation folder.")
        self.status_label.setProperty("role", "muted")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        page_layout.addWidget(self.status_label)
        self._update_responsive_layout(force=True)
        self.update_dataset_plan()
        self.update_training_account_controls()
        self.update_training_workflow()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _update_responsive_layout(self, force: bool = False) -> None:
        """Wrap the annotation actions before they impose a wide page minimum."""
        if not hasattr(self, "annotation_toolbar"):
            return
        compact = self.width() < 1250
        if not force and compact == self._compact_annotation_toolbar:
            return
        self._compact_annotation_toolbar = compact
        for widget in self.annotation_toolbar_widgets:
            self.annotation_toolbar.removeWidget(widget)
        for column in range(12):
            self.annotation_toolbar.setColumnStretch(column, 0)

        (
            previous,
            frame_status,
            next_button,
            reset_zoom,
            undo,
            finish,
            clear,
            save,
            auto_mask,
        ) = self.annotation_toolbar_widgets
        if compact:
            self.annotation_toolbar.addWidget(previous, 0, 0)
            self.annotation_toolbar.addWidget(frame_status, 0, 1)
            self.annotation_toolbar.addWidget(next_button, 0, 2)
            self.annotation_toolbar.addWidget(reset_zoom, 0, 3)
            self.annotation_toolbar.addWidget(undo, 1, 0)
            self.annotation_toolbar.addWidget(finish, 1, 1)
            self.annotation_toolbar.addWidget(clear, 1, 2)
            self.annotation_toolbar.addWidget(save, 2, 0, 1, 4)
            self.annotation_toolbar.addWidget(auto_mask, 3, 0, 1, 4, Qt.AlignHCenter)
            for column in range(4):
                self.annotation_toolbar.setColumnStretch(column, 1)
        else:
            self.annotation_toolbar.setColumnStretch(0, 1)
            for column, widget in enumerate(self.annotation_toolbar_widgets, start=1):
                self.annotation_toolbar.addWidget(widget, 0, column)
            self.annotation_toolbar.setColumnStretch(len(self.annotation_toolbar_widgets) + 1, 1)

    @staticmethod
    def _workflow_step(number: int, title: str, detail: str) -> QFrame:
        """Create a Segment-page-style workflow card without coupling this page to the main window."""
        step = QFrame()
        step.setObjectName("workflowStep")
        step.setProperty("state", "pending")
        step_layout = QHBoxLayout(step)
        step_layout.setContentsMargins(10, 8, 10, 8)
        step_layout.setSpacing(9)
        number_label = QLabel(str(number))
        number_label.setProperty("role", "stepNumber")
        step_layout.addWidget(number_label)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 600;")
        detail_label = QLabel(detail)
        detail_label.setProperty("role", "muted")
        for label in (title_label, detail_label):
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        text_layout.addWidget(title_label)
        text_layout.addWidget(detail_label)
        step_layout.addLayout(text_layout, 1)
        step.detail_label = detail_label
        return step

    def update_training_workflow(self, annotated: int = 0) -> None:
        """Reflect the currently actionable training step in the shared workflow-card style."""
        if not hasattr(self, "training_workflow_steps"):
            return
        drive = getattr(self.main_window, "drive", None)
        signed_in = bool(drive and getattr(drive, "service", None))
        has_project = bool(self.project_dir and self.frames)
        if not signed_in:
            states = (("active", "Sign in on Segment"), ("pending", "Open or create project"),
                      ("pending", "Save masks"), ("pending", "Submit to Colab"))
        elif not has_project:
            states = (("complete", "Connected"), ("active", "Open or create project"),
                      ("pending", "Save masks"), ("pending", "Submit to Colab"))
        elif annotated < len(self.frames):
            states = (("complete", "Connected"), ("complete", "Project ready"),
                      ("active", f"{annotated} / {len(self.frames)} saved"), ("pending", "Submit to Colab"))
        else:
            states = (("complete", "Connected"), ("complete", "Project ready"),
                      ("complete", f"{annotated} masks saved"), ("active", "Submit to Colab"))
        for step, (state, detail) in zip(self.training_workflow_steps, states):
            step.setProperty("state", state)
            step.detail_label.setText(detail)
            step.style().unpolish(step)
            step.style().polish(step)

    def update_training_account_controls(self) -> None:
        """Mirror the shared Google account state in the training-page header."""
        if not hasattr(self, "training_account_status"):
            return
        drive = getattr(self.main_window, "drive", None)
        signed_in = bool(drive and getattr(drive, "service", None))
        if signed_in:
            email = getattr(drive, "account_email", "")
            self.training_account_status.setText(f"Signed in as {email}" if email else "Signed in")
        else:
            self.training_account_status.setText("Not signed in")
        self.training_sign_in_button.setEnabled(not signed_in)
        self.training_sign_out_button.setEnabled(signed_in)

    def sign_in_for_training(self) -> None:
        """Start the normal Google sign-in flow without leaving model training."""
        if self.main_window and hasattr(self.main_window, "sign_in"):
            self.main_window.sign_in()
            self.training_account_status.setText("Signing in...")
            self.training_sign_in_button.setEnabled(False)
            # The authorization happens in a background thread; refresh once it has a chance to finish.
            QTimer.singleShot(1500, self.update_training_account_controls)

    def sign_out_for_training(self) -> None:
        if self.main_window and hasattr(self.main_window, "sign_out"):
            self.main_window.sign_out()
        self.update_training_account_controls()

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def show_help(self):
        QMessageBox.information(
            self,
            "Model training help",
            "PROJECT AND EXTRACTION\n"
            "Create a training dataset in a new project folder. StimTrace selects visually "
            "diverse recordings, reserves complete recordings for validation, and chooses sharp frames "
            "covering relaxed, transition, maximum-deformation, and recovery appearances. The normal "
            "user does not need to assign training or validation bins. Reopening a project restores "
            "saved masks and the last selected frame. Untouched candidates are excluded from training.\n\n"
            "ANNOTATION\n"
            "Choose or type an object class. Left-click adds polygon points. Double-left-click or "
            "Finish polygon completes the polygon. Right-click removes the latest point, or the "
            "latest completed polygon when no polygon is being drawn. Multiple polygons can be saved "
            "on one frame. Clear mask removes all polygons after confirmation. Saving writes JSON and "
            "PNG mask files and opens the next candidate.\n\n"
            "COPY AND ALIGN\n"
            "Copy and align saved mask finds the closest already annotated frame from the same "
            "recording, tracks image features inside the object, and moves a copy of its polygons. "
            "The result is an editable draft. Inspect its boundary and correct it if needed; it is "
            "saved automatically when you change frames.\n\n"
            "VIEW\n"
            "The mouse wheel zooms around the pointer. Middle-button drag pans. Reset zoom restores "
            "the complete image. Zoom and pan are preserved between frames.\n\n"
            "TRAINING\n"
            "Annotate every selected candidate. Minimum datasets use 60 frames from 15 recordings; the "
            "recommended dataset uses 125 frames from 25 recordings; robust datasets use 200 frames "
            "from 40 recordings. Colab uses the saved recording-level split, saves the checkpoint with "
            "the best validation Dice score, and exports loss, Dice, IoU, and phase-coverage QC reports. "
            "Enter a unique model name, epochs, and batch size, then submit while the Colab worker is "
            "running. After completion, Use trained model for segmentation activates it.\n\n"
            "KEYBOARD SHORTCUTS\n"
            "Ctrl+O  Open project\n"
            "Ctrl+E  Extract frames\n"
            "Ctrl+Enter  Finish polygon\n"
            "Ctrl+Z  Undo latest point or polygon\n"
            "Ctrl+Shift+Delete  Clear mask\n"
            "Ctrl+S  Save annotation and open next frame\n"
            "Auto-mask next frame  Save, advance, and align a mask automatically\n"
            "Page Up / Page Down  Previous / next frame\n"
            "Ctrl+0  Reset zoom",
        )

    def _setup_shortcuts(self):
        shortcuts = [
            ("Ctrl+O", self.open_project),
            ("Ctrl+E", self.extract_frames),
            ("Ctrl+Return", self.canvas_finish),
            ("Ctrl+Z", self.canvas_undo),
            ("Ctrl+Shift+Delete", self.canvas_clear),
            ("Ctrl+S", self.save_annotation),
            ("PgUp", lambda: self.navigate(-1)),
            ("PgDown", lambda: self.navigate(1)),
            ("Ctrl+0", self.canvas.reset_view),
        ]
        self.shortcuts = []
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def set_class_name(self, value: str):
        if hasattr(self, "canvas"):
            self.canvas.current_label = value.strip() or "Object"

    def project_paths(self):
        if not self.project_dir:
            raise RuntimeError("Open or create an annotation project first.")
        frames = self.project_dir / "frames"
        annotations = self.project_dir / "annotations"
        masks = self.project_dir / "masks"
        for path in (frames, annotations, masks):
            path.mkdir(parents=True, exist_ok=True)
        return frames, annotations, masks

    def open_project(self):
        folder = QFileDialog.getExistingDirectory(self, "Open annotation project")
        if folder:
            self.set_project(Path(folder))

    def migrate_existing_dataset(self, folder: Path, frames: list[Path]) -> dict:
        """Preserve annotations from the retired format while adding safe recording groups."""
        grouped: dict[str, list[tuple[Path, int]]] = {}
        for frame in frames:
            source_name, frame_index = legacy_frame_source(frame)
            grouped.setdefault(source_name, []).append((frame, frame_index))
        sources = sorted(grouped)
        validation_count = max(1, round(len(sources) * 0.2)) if len(sources) >= 2 else 0
        validation_positions = {
            round(index * (len(sources) - 1) / max(1, validation_count - 1))
            for index in range(validation_count)
        } if validation_count else set()
        frame_metadata: dict[str, dict] = {}
        for source_position, source_name in enumerate(sources):
            split = "validation" if source_position in validation_positions else "training"
            source_id = hashlib.sha256(source_name.casefold().encode("utf-8")).hexdigest()[:16]
            for frame, frame_index in grouped[source_name]:
                frame_metadata[frame.name] = {
                    "source_video": source_name,
                    "source_video_id": source_id,
                    "frame_index": frame_index,
                    "split": split,
                    "phase": "existing candidate",
                    "time_s": None,
                }
        metadata = {
            "schema_version": 2,
            "selection_strategy": "migrated_existing_v1",
            "effort": "minimum",
            "migrated_from": "retired_extracted_frame_project",
            "frame_metadata": frame_metadata,
            "split_summary": {
                "training_recordings": len(sources) - len(validation_positions),
                "validation_recordings": len(validation_positions),
                "training_candidates": sum(
                    item["split"] == "training" for item in frame_metadata.values()
                ),
                "validation_candidates": sum(
                    item["split"] == "validation" for item in frame_metadata.values()
                ),
            },
        }
        (folder / "project.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def set_project(self, folder: Path):
        self.project_dir = folder
        frames, _, _ = self.project_paths()
        metadata_path = self.project_dir / "project.json"
        try:
            self.project_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            self.project_metadata = {}
        self.frames = sorted(
            path for path in frames.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if self.project_metadata.get("selection_strategy") not in SUPPORTED_DATASET_STRATEGIES:
            self.project_metadata = self.migrate_existing_dataset(folder, self.frames)
        self.frame_list.clear()
        self.refresh_frame_list_status()
        self.project_label.setText(str(folder))
        self.current_index = -1
        self.canvas.reset_view()
        if self.frames:
            resume_index = 0
            try:
                last_frame = self.project_metadata.get("last_frame", "")
                resume_index = next(
                    (index for index, frame in enumerate(self.frames) if frame.name == last_frame),
                    0,
                )
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                pass
            self.frame_list.setCurrentRow(resume_index)
        self.update_annotation_count()
        migrated = self.project_metadata.get("selection_strategy") == "migrated_existing_v1"
        self.set_status(
            f"Opened {len(self.frames)} existing annotation candidates; recording groups were restored."
            if migrated else f"{len(self.frames)} training candidates loaded."
        )

    def save_project_position(self):
        if not self.project_dir or not 0 <= self.current_index < len(self.frames):
            return
        metadata_path = self.project_dir / "project.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            metadata = {}
        metadata["last_frame"] = self.frames[self.current_index].name
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.project_metadata = metadata

    def update_annotation_count(self):
        annotated = 0
        split_counts = {"training": 0, "validation": 0}
        frame_metadata = self.project_metadata.get("frame_metadata", {})
        for frame in self.frames:
            path = self.annotation_path(frame)
            if not path.exists():
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
                annotated += 1
                split = frame_metadata.get(frame.name, {}).get("split")
                if split in split_counts:
                    split_counts[split] += 1
            except (OSError, json.JSONDecodeError):
                continue
        validation_total = sum(
            item.get("split") == "validation" for item in frame_metadata.values()
        )
        self.annotation_count.setText(f"Annotated: {annotated} / {len(self.frames)}")
        self.annotation_count.setToolTip(
            "Saved annotations include empty masks for negative controls. "
            f"Independent quality-check frames: {split_counts['validation']} / {validation_total}."
        )
        self.refresh_frame_list_status()
        self.update_training_workflow(annotated)

    def refresh_frame_list_status(self) -> None:
        """Show only useful frame names and make annotation progress visible at a glance."""
        while self.frame_list.count() < len(self.frames):
            self.frame_list.addItem(QListWidgetItem())
        while self.frame_list.count() > len(self.frames):
            self.frame_list.takeItem(self.frame_list.count() - 1)
        for index, frame in enumerate(self.frames):
            annotated = False
            try:
                json.loads(self.annotation_path(frame).read_text(encoding="utf-8"))
                annotated = True
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                pass
            item = self.frame_list.item(index)
            item.setText(f"✓  {frame.name}" if annotated else f"○  {frame.name}")
            item.setBackground(QColor("#235f43") if annotated else QColor("#6f3440"))
            item.setForeground(QColor("#ffffff"))
            item.setToolTip(
                "Annotated — included in training (an empty mask is a negative control)" if annotated
                else "Not annotated — select this frame to draw a mask"
            )

    def update_dataset_plan(self, _index: int | None = None) -> None:
        effort_key = str(self.dataset_effort.currentData() or "recommended")
        effort = DATASET_EFFORTS[effort_key]
        self.dataset_plan_label.setText(
            f"{effort['recordings']} recordings x {effort['frames_per_recording']} frames"
        )
        self.dataset_plan_label.setToolTip(
            "StimTrace chooses sharp frames across rest, transition, maximum deformation, "
            "and recovery. About 20% of recordings are reserved for quality checking."
        )

    def extract_frames(self):
        videos, _ = QFileDialog.getOpenFileNames(
            self,
            "Select training videos",
            "",
            "Videos (*.avi *.mp4 *.mov *.mkv)",
        )
        if not videos:
            return
        duplicate_names = duplicate_output_stems([Path(name) for name in videos])
        if duplicate_names:
            QMessageBox.warning(
                self,
                "Duplicate video names",
                "Training videos must have unique filenames so extracted frames are not "
                "overwritten. Rename these files first:\n\n" + "\n".join(duplicate_names[:8]),
            )
            return
        effort_key = str(self.dataset_effort.currentData() or "recommended")
        effort = DATASET_EFFORTS[effort_key]
        if len(videos) < effort["recordings"]:
            response = QMessageBox.warning(
                self,
                "Limited recording diversity",
                f"{effort['label']} normally uses {effort['recordings']} independent recordings. "
                f"Only {len(videos)} were selected. StimTrace can continue, but the model will have "
                "less independent quality-check data.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                return
        folder = QFileDialog.getExistingDirectory(self, "Choose annotation project folder")
        if not folder:
            return
        self.project_dir = Path(folder)
        frames_dir, annotations_dir, masks_dir = self.project_paths()
        existing_candidates = [
            path for path in frames_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if existing_candidates:
            choice = QMessageBox.question(
                self,
                "Existing extracted frames",
                f"This project already contains {len(existing_candidates)} extracted frame(s).\n\n"
                "StimTrace creates one fixed development/quality-check split. Choose Yes to "
                "replace this project's existing candidates and annotations, or No to choose a new folder.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return
        project_dir = self.project_dir
        video_paths = [Path(video) for video in videos]

        def extract_snapshot() -> dict:
            if existing_candidates:
                for path in existing_candidates:
                    path.unlink(missing_ok=True)
                for directory, suffix in ((annotations_dir, ".json"), (masks_dir, ".png")):
                    for path in directory.glob(f"*{suffix}"):
                        path.unlink(missing_ok=True)
            analysed = [
                analyse_video_for_phase_selection(video, effort["frames_per_recording"])
                for video in video_paths
            ]
            selected_records = select_diverse_recordings(analysed, effort["recordings"])
            assign_recording_splits(selected_records)
            frame_metadata = write_phase_aware_frames(selected_records, frames_dir)
            training_records = sum(record["split"] == "training" for record in selected_records)
            validation_records = len(selected_records) - training_records
            metadata = {
                "schema_version": 2,
                "selection_strategy": "phase_aware_v1",
                "effort": effort_key,
                "considered_videos": [str(video) for video in video_paths],
                "selected_recordings": selected_records,
                "frame_metadata": frame_metadata,
                "split_summary": {
                    "training_recordings": training_records,
                    "validation_recordings": validation_records,
                    "training_candidates": sum(
                        item["split"] == "training" for item in frame_metadata.values()
                    ),
                    "validation_candidates": sum(
                        item["split"] == "validation" for item in frame_metadata.values()
                    ),
                },
            }
            (project_dir / "project.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            return {
                "project_dir": str(project_dir),
                "written": len(frame_metadata),
                "video_count": len(selected_records),
                "candidate_count": len(frame_metadata),
                "training_records": training_records,
                "validation_records": validation_records,
            }

        thread = BackgroundFunctionThread(extract_snapshot, self.main_window or self)
        register_thread = getattr(self.main_window, "register_thread", None)
        if callable(register_thread):
            register_thread(thread, "training-frame-extraction")
        thread.succeeded.connect(self.finish_frame_extraction, Qt.QueuedConnection)
        thread.failed.connect(self.fail_frame_extraction, Qt.QueuedConnection)
        thread.finished.connect(
            lambda worker=thread: self.finish_frame_extraction_thread(worker),
            Qt.QueuedConnection,
        )
        self.extraction_thread = thread
        self.open_project_button.setEnabled(False)
        self.extract_button.setEnabled(False)
        self.dataset_effort.setEnabled(False)
        self.extraction_progress.setVisible(True)
        self.set_status(
            f"Inspecting {len(video_paths)} recording(s) and selecting training frames in the background..."
        )
        thread.start()

    def finish_frame_extraction(self, result: object) -> None:
        if not isinstance(result, dict):
            self.fail_frame_extraction("Frame extraction returned an invalid result.")
            return
        self.set_project(Path(str(result["project_dir"])))
        self.set_status(
            f"Selected {result['written']} training frames from {result['video_count']} recordings "
            f"({result['training_records']} development, {result['validation_records']} quality-check recordings)."
        )

    def fail_frame_extraction(self, message: str) -> None:
        LOGGER.error("Frame extraction failed: %s", message)
        self.set_status(f"Frame extraction failed: {message}")
        QMessageBox.critical(self, "Frame extraction failed", message)

    def finish_frame_extraction_thread(self, thread: BackgroundFunctionThread) -> None:
        if self.extraction_thread is thread:
            self.extraction_thread = None
        self.open_project_button.setEnabled(True)
        self.extract_button.setEnabled(True)
        self.dataset_effort.setEnabled(True)
        self.extraction_progress.setVisible(False)

    def annotation_path(self, frame: Path) -> Path:
        return self.project_paths()[1] / f"{frame.stem}.json"

    def mask_path(self, frame: Path) -> Path:
        return self.project_paths()[2] / f"{frame.stem}.png"

    def load_frame(self, index: int):
        if not 0 <= index < len(self.frames):
            return
        if self.current_index >= 0:
            current_frame = self.frames[self.current_index]
            if self.propagated_draft:
                # A copied/aligned mask is a usable annotation. Persist it when the
                # user changes frames so normal next/previous navigation never
                # interrupts the review workflow with a save/discard prompt.
                self.save_annotation(silent=True)
            elif self.canvas.polygons or self.annotation_path(current_frame).exists():
                self.save_annotation(silent=True)
        self.current_index = index
        frame = self.frames[index]
        polygons = []
        annotation = self.annotation_path(frame)
        if annotation.exists():
            payload = json.loads(annotation.read_text(encoding="utf-8"))
            polygons = payload.get("shapes", [])
        try:
            self.canvas.set_image(frame, polygons)
            self.canvas.reset_view()
            self.propagated_draft = False
            self.frame_status.setText(f"{index + 1}/{len(self.frames)} - {frame.name}")
            self.save_project_position()
        except Exception as error:
            QMessageBox.critical(self, "Could not load frame", str(error))

    def navigate(self, offset: int):
        if not self.frames:
            return
        target_index = max(0, min(len(self.frames) - 1, self.current_index + offset))
        if target_index == self.current_index:
            return
        self.frame_list.setCurrentRow(target_index)
        if offset > 0 and self.auto_mask_checkbox.isChecked():
            QTimer.singleShot(0, self.apply_auto_mask_to_current_frame)

    def canvas_undo(self):
        self.canvas.undo_point()

    def canvas_finish(self):
        self.canvas.finish_polygon()

    def canvas_clear(self):
        if QMessageBox.question(self, "Clear annotation", "Remove all polygons from this frame?") == QMessageBox.Yes:
            self.canvas.clear_polygons()

    def closest_saved_annotation(self, target: Path) -> tuple[Path, list[dict]] | None:
        target_metadata = self.project_metadata.get("frame_metadata", {}).get(target.name, {})
        target_source = target_metadata.get("source_video_id")
        target_index = int(target_metadata.get("frame_index", -1))
        if not target_source:
            return None
        candidates: list[tuple[int, int, Path, list[dict]]] = []
        for frame in self.frames:
            if frame == target:
                continue
            metadata = self.project_metadata.get("frame_metadata", {}).get(frame.name, {})
            if metadata.get("source_video_id") != target_source:
                continue
            annotation = self.annotation_path(frame)
            if not annotation.exists():
                continue
            try:
                shapes = json.loads(annotation.read_text(encoding="utf-8")).get("shapes", [])
            except (OSError, json.JSONDecodeError):
                continue
            if not shapes:
                continue
            frame_index = int(metadata.get("frame_index", -1))
            is_future = 1 if frame_index > target_index else 0
            candidates.append((is_future, abs(frame_index - target_index), frame, shapes))
        if not candidates:
            return None
        _future, _distance, source, shapes = min(candidates, key=lambda item: (item[0], item[1]))
        return source, shapes

    def apply_auto_mask_to_current_frame(self):
        """Populate the selected frame with an aligned draft when auto-mask is enabled."""
        if self.auto_mask_checkbox.isChecked():
            self.copy_and_align_mask(confirm_replace=False, automatic=True)

    def copy_and_align_mask(self, confirm_replace: bool = True, automatic: bool = False):
        if not 0 <= self.current_index < len(self.frames):
            QMessageBox.information(self, "No frame selected", "Select a frame before copying a mask.")
            return
        target = self.frames[self.current_index]
        source_data = self.closest_saved_annotation(target)
        if source_data is None:
            metadata = self.project_metadata.get("frame_metadata", {}).get(target.name, {})
            same_recording = sum(
                item.get("source_video_id") == metadata.get("source_video_id")
                for item in self.project_metadata.get("frame_metadata", {}).values()
            )
            recording_name = str(metadata.get("source_video", "this recording"))
            message = (
                "Annotate and save one frame, then move to a different candidate from the same "
                f"recording before copying it. This dataset has {same_recording} candidate frame(s) "
                f"for {recording_name}. StimTrace does not copy masks between different recordings."
            )
            if automatic:
                self.set_status(f"Auto-mask skipped: {message}")
            else:
                QMessageBox.information(self, "No saved mask available", message)
            return
        if confirm_replace and (self.canvas.polygons or self.annotation_path(target).exists()):
            if QMessageBox.question(
                self,
                "Replace current annotation?",
                "Replace the current editable annotation with an aligned copy? "
                "The saved file is not changed until you save this frame.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ) != QMessageBox.Yes:
                return
        source, shapes = source_data
        try:
            copied, dx, dy, feature_count = propagate_polygons_to_frame(source, target, shapes)
            self.canvas.set_image(target, copied)
            self.propagated_draft = True
            prefix = "Auto-masked" if automatic else "Copied and aligned"
            self.set_status(
                f"{prefix} from {source.name} using {feature_count} object features "
                f"(shift {dx:.1f}, {dy:.1f} px). Review the mask, then save it."
            )
        except Exception as error:
            LOGGER.info("Mask propagation was not reliable: %s", error)
            message = f"{error}\n\nDraw this frame manually or choose a closer saved frame."
            if automatic:
                self.set_status(f"Auto-mask skipped: {message}")
            else:
                QMessageBox.information(self, "Could not align mask", message)

    def save_annotation(self, silent: bool = False):
        if self.current_index < 0 or self.canvas.pixmap.isNull():
            return
        self.canvas.finish_polygon()
        frame = self.frames[self.current_index]
        payload = {
            "image": frame.name,
            "width": self.canvas.pixmap.width(),
            "height": self.canvas.pixmap.height(),
            "shapes": self.canvas.polygons,
        }
        self.annotation_path(frame).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mask = Image.new("L", (payload["width"], payload["height"]), 0)
        draw = ImageDraw.Draw(mask)
        for shape in self.canvas.polygons:
            points = [(round(point[0]), round(point[1])) for point in shape["points"]]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        mask.save(self.mask_path(frame))
        self.propagated_draft = False
        self.update_annotation_count()
        if not silent:
            if self.current_index + 1 < len(self.frames):
                saved_name = frame.name
                next_index = self.current_index + 1
                # Avoid a nested save/load cycle while this save button handler is still active.
                auto_mask = self.auto_mask_checkbox.isChecked()
                QTimer.singleShot(0, lambda index=next_index: self.frame_list.setCurrentRow(index))
                if auto_mask:
                    QTimer.singleShot(0, self.apply_auto_mask_to_current_frame)
                    self.set_status(f"Saved {saved_name}; moved to the next frame and auto-masked it.")
                else:
                    self.set_status(f"Saved {saved_name}; moved to the next frame.")
            else:
                self.set_status(f"Saved {frame.name}; this is the final frame.")

    def closeEvent(self, event):
        if self.propagated_draft:
            choice = QMessageBox.question(
                self,
                "Save copied mask?",
                "The copied and aligned mask is still a draft. Save it before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if choice == QMessageBox.Save:
                self.save_annotation(silent=True)
            elif choice == QMessageBox.Cancel:
                event.ignore()
                return
            else:
                self.propagated_draft = False
        elif self.current_index >= 0 and (
            self.canvas.polygons or self.annotation_path(self.frames[self.current_index]).exists()
        ):
            self.save_annotation(silent=True)
        self.save_project_position()
        super().closeEvent(event)

    def submit_training(self):
        try:
            if not self.main_window or not self.main_window.drive.service:
                raise RuntimeError("Sign in to Google Drive in the segmentation window first.")
            if not self.project_dir:
                raise RuntimeError("Open an annotation project first.")
            if self.propagated_draft:
                raise RuntimeError(
                    "Review and save or discard the copied mask before submitting training."
                )
            self.save_annotation(silent=True)
            project_dir = self.project_dir
            model_name = self.model_name.text().strip() or "StimTrace_custom_model"
            epochs = self.epochs.value()
            batch_size = self.batch_size.value()
        except Exception as error:
            LOGGER.exception("Training submission failed")
            QMessageBox.critical(self, "Could not submit training", str(error))
            return

        def prepare_and_upload() -> dict:
            archive, _annotated_count = create_training_archive(project_dir)
            return self.main_window.prepare_training_job(
                archive, model_name, epochs, batch_size
            )

        thread = BackgroundFunctionThread(prepare_and_upload, self.main_window)
        register_thread = getattr(self.main_window, "register_thread", None)
        if callable(register_thread):
            register_thread(thread, "training-dataset-upload")
        thread.succeeded.connect(
            lambda result: self.finish_training_submission(result, model_name),
            Qt.QueuedConnection,
        )
        thread.failed.connect(self.fail_training_submission, Qt.QueuedConnection)
        thread.finished.connect(
            lambda worker=thread: self.finish_training_submission_thread(worker),
            Qt.QueuedConnection,
        )
        self.training_submit_thread = thread
        self.train_button.setEnabled(False)
        self.training_progress.setRange(0, 0)
        self.training_label.setText(
            "Preparing and uploading the training dataset in the background..."
        )
        thread.start()

    def finish_training_submission(self, result: object, model_name: str) -> None:
        if not isinstance(result, dict):
            self.fail_training_submission("Training submission returned an invalid result.")
            return
        try:
            self.main_window.record_training_job(result, model_name, self.project_dir)
            self.training_progress_id = result["progress_file_id"]
        except Exception as error:
            LOGGER.exception("Could not record the submitted training job")
            self.fail_training_submission(str(error))
            return
        self.training_label.setText(f"Training job {result['job_id']} queued.")
        self.training_progress.setRange(0, 1000)
        self.training_progress.setValue(0)
        self.training_timer.start()

    def fail_training_submission(self, message: str) -> None:
        LOGGER.error("Training submission failed: %s", message)
        if "userRateLimitExceeded" in message or "rateLimitExceeded" in message:
            message = (
                "Google Drive is still rate-limiting this account after automatic retries. "
                "Your annotations remain saved locally. Wait a few minutes, then submit again."
            )
        self.training_label.setText("Training submission failed. Annotations remain saved.")
        self.training_progress.setRange(0, 1000)
        self.training_progress.setValue(0)
        QMessageBox.critical(self, "Could not submit training", message)

    def finish_training_submission_thread(
        self, thread: BackgroundFunctionThread
    ) -> None:
        if self.training_submit_thread is thread:
            self.training_submit_thread = None
        self.train_button.setEnabled(True)
        if not self.training_timer.isActive():
            self.training_progress.setRange(0, 1000)

    def poll_training(self):
        if not self.training_progress_id or self.training_poll_thread is not None:
            return
        progress_id = self.training_progress_id
        thread = BackgroundFunctionThread(
            lambda: self.main_window.drive.job_progress(progress_id),
            self.main_window,
        )
        register_thread = getattr(self.main_window, "register_thread", None)
        if callable(register_thread):
            register_thread(thread, "training-progress-poll")
        thread.succeeded.connect(self.apply_training_progress, Qt.QueuedConnection)
        thread.failed.connect(
            lambda message: LOGGER.warning("Training progress poll failed: %s", message),
            Qt.QueuedConnection,
        )
        thread.finished.connect(
            lambda worker=thread: self.finish_training_poll_thread(worker),
            Qt.QueuedConnection,
        )
        self.training_poll_thread = thread
        thread.start()

    def finish_training_poll_thread(self, thread: BackgroundFunctionThread) -> None:
        if self.training_poll_thread is thread:
            self.training_poll_thread = None

    def apply_training_progress(self, progress: object) -> None:
        if not isinstance(progress, dict):
            return
        state = progress.get("state", "queued")
        message = progress.get("message", state.replace("_", " "))
        self.training_label.setText(message)
        fraction = progress.get("progress_fraction")
        if fraction is not None:
            self.training_progress.setValue(max(0, min(1000, round(float(fraction) * 1000))))
        if state in {"complete", "failed", "cancelled"}:
            self.training_timer.stop()
            if state == "complete":
                self.training_progress.setValue(1000)
                self.trained_model_file_id = progress.get("model_file_id", "")
                self.trained_model_name = progress.get("model_name", "model.pth")
                self.use_model_button.setEnabled(bool(self.trained_model_file_id))
                self.training_label.setText(
                    f"Training complete. Downloading model and QC package for offline use..."
                )
                self.download_training_package()

    def download_training_package(self) -> None:
        """Mirror the completed checkpoint and QC files into the annotation project."""
        if (
            self.training_download_thread is not None
            or not self.project_dir
            or not self.training_progress_id
        ):
            return
        job = next(
            (
                item for item in reversed(getattr(self.main_window, "jobs", []))
                if item.get("type") == "training"
                and item.get("progress_file_id") == self.training_progress_id
            ),
            None,
        )
        if not job:
            LOGGER.warning("Completed training job is missing from local history.")
            return
        target = self.project_dir / "system" / "training" / str(job["job_id"])
        self.training_label.setText(f"Downloading offline training package to {target}...")

        def download() -> dict:
            files = self.main_window.drive.download_training_package(
                str(job["folder_id"]),
                target,
                str(job.get("model_output_file_id", "")),
            )
            return {"job_id": job["job_id"], "target": str(target), "files": files}

        thread = BackgroundFunctionThread(download, self.main_window)
        register_thread = getattr(self.main_window, "register_thread", None)
        if callable(register_thread):
            register_thread(thread, "training-package-download")
        thread.succeeded.connect(self.finish_training_package_download, Qt.QueuedConnection)
        thread.failed.connect(self.fail_training_package_download, Qt.QueuedConnection)
        thread.finished.connect(
            lambda worker=thread: self.finish_training_package_thread(worker),
            Qt.QueuedConnection,
        )
        self.training_download_thread = thread
        thread.start()

    def finish_training_package_download(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        target = Path(str(result["target"]))
        self.training_package_path = target
        self.open_training_package_button.setEnabled(target.is_dir())
        for job in getattr(self.main_window, "jobs", []):
            if job.get("job_id") == result.get("job_id"):
                job["local_training_package_path"] = str(target)
                break
        save_jobs = getattr(self.main_window, "_save_jobs", None)
        if callable(save_jobs):
            save_jobs()
        self.training_label.setText(
            f"Training complete. Offline package saved to {target} ({len(result.get('files', []))} files)."
        )

    def fail_training_package_download(self, message: str) -> None:
        LOGGER.warning("Could not download offline training package: %s", message)
        self.training_label.setText(
            "Training complete, but the offline package could not be downloaded. "
            "It remains available in the Drive training folder."
        )

    def finish_training_package_thread(self, thread: BackgroundFunctionThread) -> None:
        if self.training_download_thread is thread:
            self.training_download_thread = None

    def open_training_package(self) -> None:
        if self.training_package_path and self.training_package_path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.training_package_path.resolve())))

    def use_trained_model(self):
        if not self.trained_model_file_id:
            return
        try:
            self.main_window.activate_trained_model(
                self.trained_model_file_id,
                self.trained_model_name,
            )
        except Exception as error:
            QMessageBox.critical(self, "Could not select model", str(error))


def np_any_mask(path: Path) -> bool:
    image = Image.open(path)
    return image.getbbox() is not None
