"""Expert optical-flow point tracking for StimTrace.

The distance definitions are adapted from the MIT-licensed Cell Motion Tracker
2.0.0 project. The Qt workflow and export format are native to StimTrace. See
NOTICE.md for attribution.
"""
from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2 as cv
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app_logging import get_logger
from job_utils import duplicate_output_stems, video_files_in_folder
from ui_actions import set_action_icon


LOGGER = get_logger("point_tracking")
VIDEO_FILTER = "Video files (*.avi *.mp4 *.mov *.mkv *.m4v);;All files (*.*)"
AUTOMATIC_FEATURE_MAXIMUM = 10


@dataclass
class VideoSelection:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    initial_frame: int = 0
    baseline_frame: int = 0
    points: list[tuple[float, float]] = field(default_factory=list)
    feature_roi: tuple[int, int, int, int] | None = None

@dataclass(frozen=True)
class TrackingOptions:
    strategy: str
    measurement: str
    baseline: str
    px_per_um: float
    force_slope_un_per_um: float
    force_model_name: str
    fps_override: float | None
    threshold_percentile: float
    save_coordinates: bool
    save_plots: bool
    save_overlay_video: bool = True
    axis_projection: str = "automatic"


def probe_video(path: Path) -> VideoSelection:
    capture = cv.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = float(capture.get(cv.CAP_PROP_FPS))
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0:
            fps = 0.0
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video metadata is incomplete: {path.name}")
        return VideoSelection(path, fps, frame_count, width, height)
    finally:
        capture.release()


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        capture.set(cv.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_index} from {path.name}")
        return frame
    finally:
        capture.release()


def detect_features(
    frame: np.ndarray,
    maximum: int = AUTOMATIC_FEATURE_MAXIMUM,
    roi: tuple[int, int, int, int] | None = None,
    minimum: int = 1,
) -> np.ndarray:
    """Return up to a compact set of strong features for optical-flow tracking."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = None
    if roi is not None:
        x, y, width, height = map(int, roi)
        x = int(np.clip(x, 0, gray.shape[1]))
        y = int(np.clip(y, 0, gray.shape[0]))
        width = int(np.clip(width, 0, gray.shape[1] - x))
        height = int(np.clip(height, 0, gray.shape[0] - y))
        if width < 8 or height < 8:
            raise ValueError("The automatic-feature ROI is too small.")
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y:y + height, x:x + width] = 255
    minimum = max(1, int(minimum))
    for quality in (0.3, 0.15, 0.05, 0.01):
        points = cv.goodFeaturesToTrack(
            gray,
            maxCorners=max(minimum, maximum),
            qualityLevel=quality,
            minDistance=7,
            blockSize=7,
            mask=mask,
        )
        if points is not None and len(points) >= minimum:
            return points.reshape(-1, 2).astype(np.float32)
    location = " inside the selected ROI" if roi is not None else ""
    raise ValueError(
        f"Fewer than {minimum} trackable feature{'s were' if minimum != 1 else ' was'} "
        f"detected{location}. Choose a textured ROI with visible edges."
    )


def combine_manual_and_automatic_points(
    manual_points: list[tuple[float, float]] | np.ndarray,
    automatic_points: np.ndarray,
    minimum_separation_px: float = 5.0,
) -> tuple[np.ndarray, list[str]]:
    """Keep manual points first and append non-duplicate automatic features."""
    manual = np.asarray(manual_points, dtype=np.float32).reshape(-1, 2)
    automatic = np.asarray(automatic_points, dtype=np.float32).reshape(-1, 2)
    accepted = [point for point in manual]
    sources = ["manual"] * len(manual)
    for point in automatic:
        if accepted:
            distances = np.linalg.norm(np.asarray(accepted) - point, axis=1)
            if np.any(distances < minimum_separation_px):
                continue
        accepted.append(point)
        sources.append("automatic")
    if not accepted:
        return np.empty((0, 2), dtype=np.float32), []
    return np.asarray(accepted, dtype=np.float32), sources


def track_lucas_kanade(
    path: Path,
    initial_points: np.ndarray,
    initial_frame: int = 0,
    progress: Callable[[int, int], None] | None = None,
    preview: Callable[[np.ndarray, np.ndarray, int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Track points with forward-backward validated pyramidal Lucas-Kanade flow."""
    points = np.asarray(initial_points, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        raise ValueError("Select at least one point before tracking.")
    capture = cv.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        total_frames = max(1, int(capture.get(cv.CAP_PROP_FRAME_COUNT)) - initial_frame)
        capture.set(cv.CAP_PROP_POS_FRAMES, initial_frame)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read initialization frame from {path.name}")
        previous_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        current = points.copy()
        active = np.ones(len(points), dtype=bool)
        consecutive_failures = np.zeros(len(points), dtype=np.int16)
        samples = [points.copy()]
        lk = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        recovery_lk = {
            "winSize": (41, 41),
            "maxLevel": 4,
            "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.01),
            "minEigThreshold": 1e-6,
        }
        maximum_step = max(25.0, 0.10 * math.hypot(frame.shape[1], frame.shape[0]))
        if progress:
            progress(1, total_frames)
        if preview:
            preview(frame, points, 1, total_frames)
        processed = 1
        while True:
            if cancelled and cancelled():
                raise InterruptedError("Point tracking cancelled.")
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            next_sample = np.full_like(current, np.nan)
            active_indices = np.flatnonzero(active)
            if len(active_indices):
                previous = current[active_indices].reshape(-1, 1, 2)
                forward, status, _error = cv.calcOpticalFlowPyrLK(
                    previous_gray, gray, previous, None, **lk
                )
                if forward is not None and status is not None:
                    backward, backward_status, _ = cv.calcOpticalFlowPyrLK(
                        gray, previous_gray, forward, None, **lk
                    )
                    forward_flat = forward.reshape(-1, 2)
                    forward_status = status.reshape(-1).astype(bool)
                    finite = np.isfinite(forward_flat).all(axis=1)
                    step = np.linalg.norm(
                        forward_flat - previous.reshape(-1, 2), axis=1
                    )
                    plausible_forward = forward_status & finite & (step <= maximum_step)
                    valid = plausible_forward.copy()
                    if backward is None or backward_status is None:
                        valid[:] = False
                    else:
                        back_flat = backward.reshape(-1, 2)
                        fb_error = np.linalg.norm(back_flat - previous.reshape(-1, 2), axis=1)
                        valid &= backward_status.reshape(-1).astype(bool)
                        valid &= fb_error <= 1.5

                    # A manually chosen point can be valid but fail the conservative
                    # forward-backward check in low-contrast microscopy frames. Retry
                    # only rejected points with a larger search window, then accept a
                    # plausible forward estimate when the strict check rejects every
                    # candidate. This remains bounded against large tracking jumps.
                    rejected = np.flatnonzero(~valid)
                    if len(rejected):
                        retry_previous = previous[rejected]
                        retry_forward, retry_status, _ = cv.calcOpticalFlowPyrLK(
                            previous_gray, gray, retry_previous, None, **recovery_lk
                        )
                        if retry_forward is not None and retry_status is not None:
                            retry_flat = retry_forward.reshape(-1, 2)
                            retry_step = np.linalg.norm(
                                retry_flat - retry_previous.reshape(-1, 2), axis=1
                            )
                            retry_valid = (
                                retry_status.reshape(-1).astype(bool)
                                & np.isfinite(retry_flat).all(axis=1)
                                & (retry_step <= maximum_step)
                            )
                            accepted = rejected[retry_valid]
                            forward_flat[accepted] = retry_flat[retry_valid]
                            valid[accepted] = True
                    if not valid.any() and plausible_forward.any():
                        valid = plausible_forward

                    valid_indices = active_indices[valid]
                    next_sample[valid_indices] = forward_flat[valid]
                    current[valid_indices] = forward_flat[valid]
                    consecutive_failures[valid_indices] = 0
                    failed_indices = active_indices[~valid]
                    consecutive_failures[failed_indices] += 1
                    active[failed_indices[consecutive_failures[failed_indices] >= 3]] = False
                else:
                    consecutive_failures[active_indices] += 1
                    active[active_indices[consecutive_failures[active_indices] >= 3]] = False
            samples.append(next_sample)
            previous_gray = gray
            processed += 1
            if progress:
                progress(processed, total_frames)
            if preview:
                preview(frame, next_sample, processed, total_frames)
        if len(samples) < 2:
            raise ValueError(f"{path.name} contains fewer than two readable frames.")
        trajectories = np.stack(samples, axis=1)
        if np.all(~np.isfinite(trajectories[:, 1:, :])):
            raise ValueError(
                "None of the selected points could be tracked. Choose points on visible "
                "edges or textured structures, or use Automatic features."
            )
        return trajectories
    finally:
        capture.release()


def absolute_distance(
    trajectories: np.ndarray,
    baseline: str = "auto_relaxed",
    baseline_index: int = 0,
    projection_axis: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Calculate displacement from a physical position in one frame.

    With a unit projection axis this is signed, one-dimensional bending
    displacement; without it the legacy result is Euclidean distance.
    """
    axis = None
    if projection_axis is not None:
        axis = np.asarray(projection_axis, dtype=float).reshape(2)
        magnitude = float(np.linalg.norm(axis))
        if not np.isfinite(axis).all() or not np.isfinite(magnitude) or magnitude <= 1e-12:
            raise ValueError("Projection axis must contain two finite non-zero values.")
        axis = axis / magnitude
    distances: list[np.ndarray] = []
    references: list[int] = []
    for trajectory in np.asarray(trajectories, dtype=float):
        valid_indices = np.flatnonzero(np.isfinite(trajectory).all(axis=1))
        if not len(valid_indices):
            distances.append(np.full(len(trajectory), np.nan))
            references.append(-1)
            continue
        if baseline == "first_frame":
            reference_index = int(valid_indices[0])
        elif baseline == "selected_frame":
            reference_index = int(np.clip(baseline_index, 0, len(trajectory) - 1))
            if reference_index not in valid_indices:
                reference_index = int(valid_indices[np.argmin(abs(valid_indices - reference_index))])
        elif baseline == "auto_relaxed":
            valid_positions = trajectory[valid_indices]
            pairwise = np.linalg.norm(
                valid_positions[:, None, :] - valid_positions[None, :, :], axis=2
            )
            endpoint_a, endpoint_b = np.unravel_index(np.nanargmax(pairwise), pairwise.shape)
            chosen = endpoint_a if pairwise[endpoint_a].mean() > pairwise[endpoint_b].mean() else endpoint_b
            reference_index = int(valid_indices[chosen])
        else:
            raise ValueError(f"Unknown baseline definition: {baseline}")
        reference = trajectory[reference_index]
        relative = trajectory - reference
        distances.append(relative @ axis if axis is not None else np.linalg.norm(relative, axis=1))
        references.append(reference_index)
    return distances, references


def derive_projection_axis(trajectories: np.ndarray) -> np.ndarray:
    """Estimate one stable bending axis from the selected tracked positions."""
    points = np.asarray(trajectories, dtype=float)
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError("Point trajectories must have shape (points, frames, 2).")
    count = np.isfinite(points).all(axis=2).sum(axis=0)
    centroid = np.full((points.shape[1], 2), np.nan, dtype=float)
    for index in np.flatnonzero(count > 0):
        centroid[index] = np.nanmean(points[:, index, :], axis=0)
    valid = centroid[np.isfinite(centroid).all(axis=1)]
    if len(valid) < 2:
        raise ValueError("At least two valid tracked positions are required to estimate a bending axis.")
    centered = valid - np.nanmedian(valid, axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = np.asarray(vectors[:, int(np.argmax(values))], dtype=float)
    magnitude = float(np.linalg.norm(axis))
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    axis = axis / magnitude
    # Eigenvectors have arbitrary polarity. Use the dominant excursion as the
    # positive direction so standalone point-tracking traces remain intuitive.
    relative = valid - valid[0]
    projected = relative @ axis
    if abs(float(np.nanmin(projected))) > float(np.nanmax(projected)):
        axis = -axis
    return axis


def cumulative_distance(trajectories: np.ndarray) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for trajectory in np.asarray(trajectories, dtype=float):
        steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        steps[~np.isfinite(steps)] = 0.0
        outputs.append(np.concatenate(([0.0], np.cumsum(steps))))
    return outputs


def relative_distance(trajectories: np.ndarray) -> np.ndarray:
    trajectories = np.asarray(trajectories, dtype=float)
    if len(trajectories) < 2:
        raise ValueError("Relative distance requires at least two tracked points.")
    return np.linalg.norm(trajectories[0] - trajectories[1], axis=1)


def calculate_measurements(
    trajectories: np.ndarray,
    options: TrackingOptions,
    baseline_index: int = 0,
    manual_point_count: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    px_per_um = options.px_per_um
    if not math.isfinite(px_per_um) or px_per_um <= 0:
        raise ValueError("Pixels per micrometer must be greater than zero.")
    slope = options.force_slope_un_per_um
    if not math.isfinite(slope) or slope < 0:
        raise ValueError("Force slope must be a finite, non-negative value.")
    metadata: dict = {
        "measurement": options.measurement,
        "baseline": options.baseline,
        "force_model_name": options.force_model_name,
        "force_slope_un_per_um": slope,
        "force_equation": "Force_uN = distance_um * force_slope_uN_per_um",
        "force_axis_projection": options.axis_projection if options.measurement == "absolute" else "not_applicable",
        "force_additional_zeroing": "none",
    }
    if options.measurement == "relative":
        distances_um = {
            "Point_1_to_Point_2_distance_um": relative_distance(trajectories) / px_per_um
        }
    elif options.measurement == "cumulative":
        distances = cumulative_distance(trajectories)
        distances_um = {
            f"Point_{index + 1}_cumulative_distance_um": values / px_per_um
            for index, values in enumerate(distances)
        }
    else:
        legacy_distances, reference_indices = absolute_distance(
            trajectories,
            baseline=options.baseline,
            baseline_index=baseline_index,
        )
        selected_indices = list(range(len(legacy_distances)))
        if options.strategy in {"automatic", "hybrid"}:
            manual_count = (
                int(np.clip(manual_point_count, 0, len(legacy_distances)))
                if options.strategy == "hybrid"
                else 0
            )
            automatic_indices = np.arange(manual_count, len(legacy_distances), dtype=int)
            amplitudes = np.asarray([
                np.nanmax(legacy_distances[index])
                if np.isfinite(legacy_distances[index]).any()
                else 0.0
                for index in automatic_indices
            ])
            selected_automatic: list[int] = []
            threshold = np.nan
            if len(amplitudes):
                threshold = float(
                    np.percentile(amplitudes, options.threshold_percentile)
                )
                selected_automatic = automatic_indices[
                    amplitudes > threshold
                ].tolist()
                if not selected_automatic:
                    selected_automatic = [
                        int(automatic_indices[int(np.argmax(amplitudes))])
                    ]
            selected_indices = [*range(manual_count), *selected_automatic]
            metadata.update({
                "movement_threshold_px": threshold,
                "movement_threshold_percentile": options.threshold_percentile,
                "selected_point_indices": selected_indices,
                "manual_point_count": manual_count,
                "automatic_point_count": len(automatic_indices),
            })
        axis = None
        if options.axis_projection == "automatic":
            axis = derive_projection_axis(np.asarray(trajectories, dtype=float)[selected_indices])
            distances, reference_indices = absolute_distance(
                trajectories,
                baseline=options.baseline,
                baseline_index=baseline_index,
                projection_axis=axis,
            )
            metadata.update({
                "force_axis_projection": "automatic_pca",
                "force_axis_x": float(axis[0]),
                "force_axis_y": float(axis[1]),
            })
        else:
            distances = legacy_distances
            metadata.update({"force_axis_x": np.nan, "force_axis_y": np.nan})
        metadata["baseline_indices"] = reference_indices
        distances_um = {
            f"Point_{point_index + 1}_distance_um": distances[point_index] / px_per_um
            for point_index in selected_indices
        }

    forces = {
        name.removesuffix("_distance_um") + "_Force_uN": np.asarray(values) * slope
        for name, values in distances_um.items()
    }
    return {**forces, **distances_um}, metadata


def build_distance_table(measurements: dict[str, np.ndarray], fps: float) -> pd.DataFrame:
    if not measurements:
        raise ValueError("No distance traces were produced.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be greater than zero.")
    lengths = {len(values) for values in measurements.values()}
    if len(lengths) != 1:
        raise ValueError("Distance traces do not have a common length.")
    sample_count = lengths.pop()
    data = {"time_s": np.arange(sample_count, dtype=float) / fps}
    data.update(measurements)
    return pd.DataFrame(data)


def build_tracking_trace(
    selection: VideoSelection,
    trajectories: np.ndarray,
    measurements: dict[str, np.ndarray],
    measurement_metadata: dict,
    fps: float,
    options: TrackingOptions,
) -> pd.DataFrame:
    """Build one complete per-recording trace without duplicated output files."""
    sample_count = trajectories.shape[1]
    data: dict[str, object] = {
        "frame_idx": selection.initial_frame + np.arange(sample_count, dtype=int),
        "time_s": np.arange(sample_count, dtype=float) / fps,
    }
    for index, trajectory in enumerate(trajectories, start=1):
        valid = np.isfinite(trajectory).all(axis=1)
        data[f"Point_{index}_tracking_state"] = np.where(valid, "valid", "lost")
        data[f"Point_{index}_x_px"] = trajectory[:, 0]
        data[f"Point_{index}_y_px"] = trajectory[:, 1]
        data[f"Point_{index}_x_um"] = trajectory[:, 0] / options.px_per_um
        data[f"Point_{index}_y_um"] = trajectory[:, 1] / options.px_per_um
    data.update(measurements)
    trace = pd.DataFrame(data)
    trace["source_fps"] = selection.fps
    trace["effective_fps"] = fps
    trace["pixels_per_micrometer"] = options.px_per_um
    trace["force_model_name"] = options.force_model_name
    trace["force_slope_un_per_um"] = options.force_slope_un_per_um
    trace["measurement_mode"] = options.measurement
    trace["baseline_mode"] = options.baseline
    trace["point_selection_mode"] = options.strategy
    trace["point_sources"] = json.dumps(
        measurement_metadata.get("point_sources", [])
    )
    trace["movement_threshold_percentile"] = measurement_metadata.get(
        "movement_threshold_percentile", np.nan
    )
    trace["selected_point_indices"] = json.dumps(
        measurement_metadata.get("selected_point_indices", list(range(len(trajectories))))
    )
    trace["baseline_reference_indices"] = json.dumps(
        measurement_metadata.get("baseline_indices", [])
    )
    trace["force_axis_projection"] = measurement_metadata.get("force_axis_projection", "none")
    trace["force_axis_x"] = measurement_metadata.get("force_axis_x", np.nan)
    trace["force_axis_y"] = measurement_metadata.get("force_axis_y", np.nan)
    trace["calibration_equation"] = "Force_uN = distance_um * force_slope_uN_per_um"
    return trace


TRACKING_COLORS = (
    (40, 210, 255),
    (80, 220, 80),
    (255, 140, 60),
    (200, 90, 255),
    (60, 80, 255),
    (255, 220, 80),
)


def tracking_color(point_index: int) -> tuple[int, int, int]:
    return TRACKING_COLORS[point_index % len(TRACKING_COLORS)]


def annotate_tracking_frame(
    frame: np.ndarray,
    points: np.ndarray,
    point_indices: list[int] | None = None,
) -> np.ndarray:
    """Draw stable point labels and tracking-loss information on one frame."""
    height, width = frame.shape[:2]
    visible_indices = point_indices if point_indices is not None else list(range(len(points)))
    font_scale = max(0.6, min(width, height) / 1600.0)
    radius = max(5, round(min(width, height) / 220))
    lost_labels = []
    for point_index in visible_indices:
        point = points[point_index]
        label = f"P{point_index + 1}"
        if not np.isfinite(point).all():
            lost_labels.append(label)
            continue
        color = tracking_color(point_index)
        center = tuple(np.rint(point).astype(int))
        cv.circle(frame, center, radius, color, 3, cv.LINE_AA)
        cv.circle(frame, center, max(2, radius // 3), color, -1, cv.LINE_AA)
        cv.putText(
            frame,
            label,
            (center[0] + radius + 4, center[1] - radius - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv.LINE_AA,
        )
    if lost_labels:
        cv.putText(
            frame,
            "Lost: " + ", ".join(lost_labels),
            (20, 36),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (40, 40, 230),
            2,
            cv.LINE_AA,
        )
    return frame


class TrackingTrailRenderer:
    """Accumulate each point's valid path and composite it over successive frames."""

    def __init__(self) -> None:
        self.trail_layer: np.ndarray | None = None
        self.previous_points: dict[int, np.ndarray] = {}

    def render(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        point_indices: list[int] | None = None,
        *,
        draw_points: bool = True,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if self.trail_layer is None or self.trail_layer.shape != frame.shape:
            self.trail_layer = np.zeros_like(frame)
            self.previous_points.clear()
        visible_indices = point_indices if point_indices is not None else list(range(len(points)))
        thickness = max(2, round(min(frame.shape[:2]) / 480))
        for point_index in visible_indices:
            point = points[point_index]
            previous = self.previous_points.get(point_index)
            if not np.isfinite(point).all():
                self.previous_points.pop(point_index, None)
                continue
            if previous is not None:
                cv.line(
                    self.trail_layer,
                    tuple(np.rint(previous).astype(int)),
                    tuple(np.rint(point).astype(int)),
                    tracking_color(point_index),
                    thickness,
                    cv.LINE_AA,
                )
            self.previous_points[point_index] = point.copy()
        rendered = cv.addWeighted(frame, 1.0, self.trail_layer, 0.9, 0.0)
        return (
            annotate_tracking_frame(rendered, points, point_indices)
            if draw_points
            else rendered
        )


class AsyncTrackingOverlayWriter:
    """Encode overlays concurrently with tracking using a small bounded queue."""

    def __init__(self, destination: Path, fps: float, frame_size: tuple[int, int]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination = destination
        self.writer = cv.VideoWriter(
            str(destination),
            cv.VideoWriter_fourcc(*"XVID"),
            fps,
            frame_size,
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Could not create point-tracking overlay: {destination}")
        self.frames: queue.Queue[tuple[np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=2)
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._encode,
            name=f"point-overlay-{destination.stem}",
            daemon=True,
        )
        self.thread.start()

    def _encode(self) -> None:
        trail_renderer = TrackingTrailRenderer()
        try:
            while True:
                item = self.frames.get()
                if item is None:
                    return
                frame, points = item
                self.writer.write(trail_renderer.render(frame, points))
        except BaseException as error:
            self.error = error
        finally:
            self.writer.release()

    def _raise_error(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Point-tracking overlay encoding failed: {self.error}") from self.error

    def submit(self, frame: np.ndarray, points: np.ndarray) -> None:
        while True:
            self._raise_error()
            try:
                self.frames.put((frame.copy(), np.asarray(points).copy()), timeout=0.1)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        if self.thread.is_alive():
            while True:
                self._raise_error()
                try:
                    self.frames.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self.thread.join()
        self._raise_error()


def write_tracking_overlay(
    destination: Path,
    selection: VideoSelection,
    trajectories: np.ndarray,
    fps: float,
    point_indices: list[int] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Render tracked points over source frames using the trace time base."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(str(selection.path))
    writer = cv.VideoWriter(
        str(destination),
        cv.VideoWriter_fourcc(*"XVID"),
        fps,
        (selection.width, selection.height),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"Could not create point-tracking overlay for: {selection.path.name}")
    capture.set(cv.CAP_PROP_POS_FRAMES, selection.initial_frame)
    trail_renderer = TrackingTrailRenderer()
    try:
        for sample_index in range(trajectories.shape[1]):
            if cancelled and cancelled():
                raise InterruptedError("Point tracking cancelled.")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Overlay rendering stopped at frame {sample_index + 1} of "
                    f"{trajectories.shape[1]} for {selection.path.name}."
                )
            writer.write(
                trail_renderer.render(
                    frame,
                    trajectories[:, sample_index],
                    point_indices,
                )
            )
    finally:
        capture.release()
        writer.release()
    return destination


def write_combined_force_traces(outputs: list[dict[str, object]], output_dir: Path) -> Path:
    """Collate point forces on the union of their native timestamps."""
    signals: list[pd.Series] = []
    for output in outputs:
        trace = pd.read_csv(str(output["trace_csv"]))
        times = pd.to_numeric(trace["time_s"], errors="coerce").round(9)
        recording = str(output["recording_name"])
        for column in output["force_columns"]:
            values = pd.to_numeric(trace[str(column)], errors="coerce")
            series = pd.Series(
                values.to_numpy(),
                index=times,
                name=f"{recording}_{column}",
            )
            series = series[series.index.notna()].groupby(level=0, sort=True).mean()
            signals.append(series)
    if not signals:
        raise ValueError("Point tracking produced no force traces to combine.")
    combined = pd.concat(signals, axis=1).sort_index()
    combined.index.name = "time_s"
    destination = output_dir / "point_tracking_force_traces.csv"
    combined.to_csv(destination)
    return destination


def write_tracking_outputs(
    output_dir: Path,
    selection: VideoSelection,
    trajectories: np.ndarray,
    measurements: dict[str, np.ndarray],
    measurement_metadata: dict,
    fps: float,
    options: TrackingOptions,
) -> dict[str, object]:
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = selection.path.stem
    trace = build_tracking_trace(
        selection,
        trajectories,
        measurements,
        measurement_metadata,
        fps,
        options,
    )
    trace_path = trace_dir / f"{stem}_point_tracking.csv"
    trace.to_csv(trace_path, index=False)
    return {
        "recording_name": stem,
        "trace_csv": str(trace_path),
        "force_columns": [
            column for column in measurements if column.lower().endswith("_force_un")
        ],
    }

class TrackingThread(QThread):
    file_progress = Signal(int, int, str, int, int)
    preview_ready = Signal(object)
    status_changed = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        selections: list[VideoSelection],
        options: TrackingOptions,
        output_dir: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.selections = selections
        self.options = options
        self.output_dir = output_dir
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True

    def run(self) -> None:
        results = []
        try:
            for file_index, selection in enumerate(self.selections, start=1):
                if self.cancel_requested:
                    raise InterruptedError("Point tracking cancelled.")
                initial_frame = read_video_frame(selection.path, selection.initial_frame)
                manual_point_count = len(selection.points)
                point_sources = ["manual"] * manual_point_count
                if self.options.strategy in {"automatic", "hybrid"}:
                    required_features = 2 if self.options.measurement == "relative" else 1
                    if self.options.strategy == "hybrid":
                        required_features = max(1, required_features - manual_point_count)
                    automatic_points = detect_features(
                        initial_frame,
                        roi=selection.feature_roi,
                        minimum=required_features,
                    )
                    if self.options.strategy == "hybrid":
                        initial_points, point_sources = combine_manual_and_automatic_points(
                            selection.points,
                            automatic_points,
                        )
                    else:
                        initial_points = automatic_points
                        manual_point_count = 0
                        point_sources = ["automatic"] * len(initial_points)
                    minimum_points = 2 if self.options.measurement == "relative" else 1
                    if len(initial_points) < minimum_points:
                        raise ValueError(
                            f"{selection.path.name}: combined point selection contains fewer "
                            f"than {minimum_points} usable points. Move the manual points or "
                            "choose a more textured automatic-feature ROI."
                        )
                    selection.points = [
                        tuple(map(float, point)) for point in initial_points
                    ]
                else:
                    initial_points = np.asarray(selection.points, dtype=np.float32)
                fps = self.options.fps_override or selection.fps
                overlay_path = (
                    self.output_dir
                    / "overlays"
                    / f"{selection.path.stem}_point_tracking_overlay.avi"
                    if self.options.save_overlay_video else None
                )
                overlay_writer = (
                    AsyncTrackingOverlayWriter(
                        overlay_path,
                        fps,
                        (selection.width, selection.height),
                    )
                    if overlay_path is not None else None
                )
                self.status_changed.emit(
                    f"Tracking{' and writing video' if overlay_writer else ''} for "
                    f"{selection.path.name}..."
                )
                last_preview_at = 0.0
                preview_trail_renderer = TrackingTrailRenderer()

                def emit_preview(
                    frame: np.ndarray,
                    points: np.ndarray,
                    frame_index: int,
                    frame_total: int,
                ) -> None:
                    nonlocal last_preview_at
                    if overlay_writer is not None:
                        overlay_writer.submit(frame, points)
                    preview_frame = preview_trail_renderer.render(
                        frame.copy(),
                        points,
                        draw_points=False,
                    )
                    now = time.perf_counter()
                    if frame_index not in {1, frame_total} and now - last_preview_at < 0.1:
                        return
                    last_preview_at = now
                    valid_points = np.asarray(points, dtype=float).reshape(-1, 2)
                    valid_points = valid_points[np.isfinite(valid_points).all(axis=1)]
                    self.preview_ready.emit(
                        {
                            "file_index": file_index,
                            "file_total": len(self.selections),
                            "name": selection.path.name,
                            "frame": preview_frame,
                            "points": valid_points.tolist(),
                            "frame_index": frame_index,
                            "frame_total": frame_total,
                        }
                    )

                try:
                    trajectories = track_lucas_kanade(
                        selection.path,
                        initial_points,
                        selection.initial_frame,
                        progress=lambda frame, total, index=file_index, name=selection.path.name: (
                            self.file_progress.emit(index, len(self.selections), name, frame, total)
                        ),
                        preview=emit_preview,
                        cancelled=lambda: self.cancel_requested,
                    )
                    if overlay_writer is not None:
                        overlay_writer.close()
                except BaseException:
                    if overlay_writer is not None:
                        try:
                            overlay_writer.close()
                        except Exception:
                            LOGGER.warning("Could not finalize failed point overlay", exc_info=True)
                    if overlay_path is not None:
                        overlay_path.unlink(missing_ok=True)
                    raise
                baseline_index = max(0, selection.baseline_frame - selection.initial_frame)
                measurements, measurement_metadata = calculate_measurements(
                    trajectories,
                    self.options,
                    baseline_index,
                    manual_point_count=manual_point_count,
                )
                measurement_metadata["point_sources"] = point_sources
                outputs = write_tracking_outputs(
                    self.output_dir,
                    selection,
                    trajectories,
                    measurements,
                    measurement_metadata,
                    fps,
                    self.options,
                )
                if overlay_path is not None:
                    outputs["overlay_video"] = str(overlay_path)
                results.append(outputs)
            self.status_changed.emit("Creating combined point-tracking force traces...")
            combined_path = write_combined_force_traces(results, self.output_dir)
            self.completed.emit({
                "outputs": results,
                "combined_force_csv": str(combined_path),
                "overlay_count": sum("overlay_video" in output for output in results),
            })
        except InterruptedError:
            self.failed.emit("Point tracking cancelled.")
        except Exception as error:
            LOGGER.exception("Point-tracking job failed")
            self.failed.emit(str(error))


class PointCanvas(QWidget):
    points_changed = Signal()
    roi_changed = Signal(object)
    zoom_changed = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.frame: np.ndarray | None = None
        self.pixmap = QPixmap()
        self.points: list[tuple[float, float]] = []
        self.preview_points: list[tuple[float, float]] | None = None
        self.feature_roi: QRectF | None = None
        self.roi_mode = False
        self.allow_points_with_roi = False
        self._roi_start: QPointF | None = None
        self._roi_dragged = False
        self._roi_press_position = QPointF()
        self.zoom_factor = 1.0
        self.view_center = QPointF()
        self._pan_start: QPointF | None = None
        self._pan_center = QPointF()
        self.setMinimumSize(640, 420)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip(
            "Mouse wheel: zoom at the pointer. Middle-button drag: pan. "
            "Left-click: add a point. Right-click: remove the last point."
        )

    def set_frame(
        self,
        frame: np.ndarray | None,
        points: list[tuple[float, float]],
        feature_roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.frame = frame
        self.points = points
        self.preview_points = None
        self.feature_roi = QRectF(*feature_roi) if feature_roi is not None else None
        self._set_pixmap(frame)
        self.update()

    def set_roi_mode(self, enabled: bool, allow_points: bool = False) -> None:
        self.roi_mode = bool(enabled)
        self.allow_points_with_roi = self.roi_mode and bool(allow_points)
        self.setToolTip(
            (
                "Left-click: add a manual point. Left-drag: draw the automatic-feature "
                "ROI. Right-click: remove the last manual point. Mouse wheel: zoom. "
                "Middle-button drag: pan."
                if self.allow_points_with_roi
                else "Left-drag: draw the automatic-feature ROI. Right-click: clear the ROI. "
                "Mouse wheel: zoom. Middle-button drag: pan."
            )
            if self.roi_mode
            else (
                "Mouse wheel: zoom at the pointer. Middle-button drag: pan. "
                "Left-click: add a point. Right-click: remove the last point."
            )
        )
        self.update()

    def roi_tuple(self) -> tuple[int, int, int, int] | None:
        if self.feature_roi is None or self.feature_roi.isEmpty():
            return None
        roi = self.feature_roi.normalized()
        return (
            int(round(roi.x())),
            int(round(roi.y())),
            max(1, int(round(roi.width()))),
            max(1, int(round(roi.height()))),
        )

    def set_tracking_preview(
        self,
        frame: np.ndarray,
        points: list[tuple[float, float]],
    ) -> None:
        self.frame = frame
        self.preview_points = points
        self._set_pixmap(frame)
        self.update()

    def _set_pixmap(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self.pixmap = QPixmap()
            self.reset_zoom()
        else:
            previous_size = self.pixmap.size()
            rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()
            self.pixmap = QPixmap.fromImage(image)
            if previous_size != self.pixmap.size():
                self.reset_zoom()

    def reset_zoom(self) -> None:
        self.zoom_factor = 1.0
        if not self.pixmap.isNull():
            self.view_center = QPointF(self.pixmap.width() / 2, self.pixmap.height() / 2)
        else:
            self.view_center = QPointF()
        self.zoom_changed.emit(100)
        self.update()

    def image_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        scale = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height())
        width = self.pixmap.width() * scale
        height = self.pixmap.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def source_rect(self) -> QRectF:
        if self.pixmap.isNull():
            return QRectF()
        width = self.pixmap.width() / self.zoom_factor
        height = self.pixmap.height() / self.zoom_factor
        self._clamp_view_center(width, height)
        return QRectF(
            self.view_center.x() - width / 2,
            self.view_center.y() - height / 2,
            width,
            height,
        )

    def _clamp_view_center(self, source_width: float, source_height: float) -> None:
        half_width = source_width / 2
        half_height = source_height / 2
        self.view_center.setX(min(max(self.view_center.x(), half_width), self.pixmap.width() - half_width))
        self.view_center.setY(min(max(self.view_center.y(), half_height), self.pixmap.height() - half_height))

    def _image_position(self, widget_position: QPointF) -> QPointF | None:
        target = self.image_rect()
        source = self.source_rect()
        if target.isEmpty() or source.isEmpty() or not target.contains(widget_position):
            return None
        return QPointF(
            source.left() + (widget_position.x() - target.left()) * source.width() / target.width(),
            source.top() + (widget_position.y() - target.top()) * source.height() / target.height(),
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#10161a"))
        target = self.image_rect()
        if target.isEmpty():
            painter.setPen(QColor("#aab6bf"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Add videos to select tracking points")
            return
        source = self.source_rect()
        painter.drawPixmap(target, self.pixmap, source)
        scale_x = target.width() / source.width()
        scale_y = target.height() / source.height()
        displayed_points = self.preview_points if self.preview_points is not None else self.points
        for index, (x, y) in enumerate(displayed_points, start=1):
            display = QPointF(
                target.left() + (x - source.left()) * scale_x,
                target.top() + (y - source.top()) * scale_y,
            )
            if not target.contains(display):
                continue
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QColor("#c43f5e"))
            painter.drawEllipse(display, 6, 6)
            painter.drawText(display + QPointF(9, -7), str(index))
        if self.feature_roi is not None and not self.feature_roi.isEmpty():
            roi = self.feature_roi.normalized()
            display_roi = QRectF(
                target.left() + (roi.left() - source.left()) * scale_x,
                target.top() + (roi.top() - source.top()) * scale_y,
                roi.width() * scale_x,
                roi.height() * scale_y,
            ).intersected(target)
            painter.setPen(QPen(QColor("#d34b67"), 3))
            painter.setBrush(QColor(195, 51, 82, 35))
            painter.drawRect(display_roi)

    def mousePressEvent(self, event) -> None:
        target = self.image_rect()
        if target.isEmpty():
            return
        if event.button() == Qt.MiddleButton and target.contains(event.position()):
            self._pan_start = event.position()
            self._pan_center = QPointF(self.view_center)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.RightButton:
            if self.allow_points_with_roi and self.points:
                self.points.pop()
                self.points_changed.emit()
                self.update()
            elif self.roi_mode:
                self.feature_roi = None
                self.roi_changed.emit(None)
                self.update()
            elif self.points:
                self.points.pop()
                self.points_changed.emit()
                self.update()
            return
        if event.button() != Qt.LeftButton:
            return
        image_position = self._image_position(event.position())
        if image_position is None:
            return
        if self.roi_mode:
            self._roi_start = image_position
            self._roi_dragged = False
            self._roi_press_position = event.position()
            if not self.allow_points_with_roi:
                self.feature_roi = QRectF(image_position, image_position)
            self.update()
            event.accept()
            return
        self.points.append((float(image_position.x()), float(image_position.y())))
        self.points_changed.emit()
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.roi_mode and self._roi_start is not None and (event.buttons() & Qt.LeftButton):
            image_position = self._image_position(event.position())
            if image_position is not None:
                drag_distance = (
                    event.position() - self._roi_press_position
                ).manhattanLength()
                if drag_distance >= 4:
                    self._roi_dragged = True
                    self.feature_roi = QRectF(
                        self._roi_start, image_position
                    ).normalized()
                    self.update()
            event.accept()
            return
        if self._pan_start is None or not (event.buttons() & Qt.MiddleButton):
            return
        target = self.image_rect()
        source = self.source_rect()
        if target.isEmpty() or source.isEmpty():
            return
        delta = event.position() - self._pan_start
        self.view_center = QPointF(
            self._pan_center.x() - delta.x() * source.width() / target.width(),
            self._pan_center.y() - delta.y() * source.height() / target.height(),
        )
        self._clamp_view_center(source.width(), source.height())
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.roi_mode and self._roi_start is not None:
            start = self._roi_start
            self._roi_start = None
            if self.allow_points_with_roi and not self._roi_dragged:
                self.points.append((float(start.x()), float(start.y())))
                self.points_changed.emit()
                self.update()
                event.accept()
                return
            roi = self.roi_tuple()
            if roi is not None and (roi[2] < 8 or roi[3] < 8):
                self.feature_roi = None
                roi = None
            self.roi_changed.emit(roi)
            self.update()
            event.accept()
            return
        if event.button() == Qt.MiddleButton and self._pan_start is not None:
            self._pan_start = None
            self.setCursor(Qt.CrossCursor)
            event.accept()

    def wheelEvent(self, event) -> None:
        anchor = self._image_position(event.position())
        if anchor is None or event.angleDelta().y() == 0:
            event.ignore()
            return
        target = self.image_rect()
        relative_x = (event.position().x() - target.left()) / target.width()
        relative_y = (event.position().y() - target.top()) / target.height()
        step = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        new_zoom = min(20.0, max(1.0, self.zoom_factor * step))
        if math.isclose(new_zoom, self.zoom_factor):
            event.accept()
            return
        self.zoom_factor = new_zoom
        new_width = self.pixmap.width() / new_zoom
        new_height = self.pixmap.height() / new_zoom
        self.view_center = QPointF(
            anchor.x() + (0.5 - relative_x) * new_width,
            anchor.y() + (0.5 - relative_y) * new_height,
        )
        self._clamp_view_center(new_width, new_height)
        self.zoom_changed.emit(int(round(new_zoom * 100)))
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.reset_zoom()
            event.accept()


class PointTrackingPage(QWidget):
    def __init__(self, stimtrace_window=None) -> None:
        super().__init__(stimtrace_window)
        self.stimtrace_window = stimtrace_window
        self.selections: list[VideoSelection] = []
        self.current_frame: np.ndarray | None = None
        self.worker: TrackingThread | None = None
        self.result_paths: list[Path] = []
        self.output_base: Path | None = None
        self.output_dir: Path | None = None
        self._syncing_controls = False
        self.force_slope_un_per_um = 0.0
        self.force_model_name = ""
        self._restoring_preferences = False
        self.build_ui()
        self.refresh_force_calibration()

    def build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        title_row = QHBoxLayout()
        heading = QVBoxLayout()
        title = QLabel("StimTrace Point Tracking")
        title.setProperty("role", "title")
        heading.addWidget(title)
        description = QLabel(
            "Track manually selected or automatically detected image features with Lucas-Kanade "
            "optical flow and export calibrated distance and force traces."
        )
        description.setProperty("role", "subtitle")
        heading.addWidget(description)
        title_row.addLayout(heading, 1)
        help_button = QPushButton("Help")
        set_action_icon(help_button, "help")
        help_button.clicked.connect(self.show_help)
        title_row.addWidget(help_button)
        layout.addLayout(title_row)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        controls = QWidget()
        controls.setObjectName("pointTrackingControlsPanel")
        controls.setMinimumWidth(360)
        controls.setMaximumWidth(520)
        self.controls_panel = controls
        controls_layout = QVBoxLayout(controls)

        videos = QWidget()
        videos.setMinimumWidth(330)
        videos.setMaximumWidth(520)
        videos_layout = QVBoxLayout(videos)
        videos_layout.setContentsMargins(8, 0, 8, 0)
        video_title = QLabel("Videos")
        video_title.setStyleSheet("font-weight: 600;")
        videos_layout.addWidget(video_title)
        video_buttons = QHBoxLayout()
        self.add_video_button = QPushButton("Add videos")
        set_action_icon(self.add_video_button, "open")
        self.add_video_button.clicked.connect(self.add_videos)
        self.add_video_folder_button = QPushButton("Add video folder")
        set_action_icon(self.add_video_folder_button, "open_folder")
        self.add_video_folder_button.setToolTip(
            "Add supported video files directly inside a selected folder."
        )
        self.add_video_folder_button.clicked.connect(self.add_video_folder)
        self.clear_video_button = QPushButton("Clear all videos")
        set_action_icon(self.clear_video_button, "delete")
        self.clear_video_button.setProperty("role", "danger")
        self.clear_video_button.clicked.connect(self.clear_videos)
        video_buttons.addWidget(self.add_video_button)
        video_buttons.addWidget(self.add_video_folder_button)
        video_buttons.addWidget(self.clear_video_button)
        video_buttons.setAlignment(Qt.AlignHCenter)
        videos_layout.addLayout(video_buttons)
        self.video_table = QTableWidget(0, 4)
        self.video_table.setHorizontalHeaderLabels(["Recording", "FPS", "Frames", "Points / ROI"])
        self.video_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.video_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.video_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.video_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.video_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.video_table.itemSelectionChanged.connect(self.video_selected)
        videos_layout.addWidget(self.video_table, 1)

        settings_box = QGroupBox("Tracking and measurement")
        form = QFormLayout(settings_box)
        self.strategy = QComboBox()
        self.strategy.addItem("Manual points", "manual")
        self.strategy.addItem("Automatic features", "automatic")
        self.strategy.addItem("Automatic + manual points", "hybrid")
        self.strategy.setToolTip(
            "Combined mode detects features inside the ROI and adds your manually "
            "selected points to the same tracking run."
        )
        self.strategy.currentIndexChanged.connect(self.update_control_states)
        form.addRow("Point selection", self.strategy)
        self.measurement = QComboBox()
        self.measurement.addItem("Absolute displacement", "absolute")
        self.measurement.addItem("Relative distance (points 1-2)", "relative")
        self.measurement.addItem("Cumulative path length", "cumulative")
        self.measurement.setToolTip(
            "Relative distance is the frame-by-frame straight-line separation between "
            "points 1 and 2. It is not zeroed to a reference frame."
        )
        self.measurement.currentIndexChanged.connect(self.update_control_states)
        form.addRow("Measurement", self.measurement)
        self.baseline = QComboBox()
        self.baseline.addItem("Auto-detected relaxed position", "auto_relaxed")
        self.baseline.addItem("Initialization frame", "first_frame")
        self.baseline.addItem("Selected reference frame", "selected_frame")
        self.baseline.currentIndexChanged.connect(self.update_control_states)
        form.addRow("Absolute reference", self.baseline)
        self.axis_projection = QComboBox()
        self.axis_projection.addItem("Automatic from tracked motion", "automatic")
        self.axis_projection.addItem("Euclidean distance (legacy)", "none")
        self.axis_projection.setToolTip(
            "For absolute displacement, project every selected point onto one PCA bending axis "
            "derived from the tracked motion. This makes force a signed, one-dimensional quantity "
            "and suppresses perpendicular jitter. Legacy Euclidean distance remains available."
        )
        self.axis_projection.currentIndexChanged.connect(self.update_control_states)
        form.addRow("Force projection", self.axis_projection)
        self.px_per_um = QDoubleSpinBox()
        self.px_per_um.setRange(0.0001, 1_000_000.0)
        self.px_per_um.setDecimals(4)
        self.px_per_um.setValue(1.0)
        self.px_per_um.setToolTip("Calibration expressed as image pixels per micrometer.")
        self.force_slope_label = QLabel()
        self.force_slope_label.setWordWrap(True)
        self.force_slope_label.setToolTip(
            "Imported from the active StimTrace model profile. Change it in Advanced settings. "
            "Absolute point tracking can use the automatic bending-axis projection selected above."
        )
        form.addRow("Pixels per µm", self.px_per_um)
        form.addRow("Force conversion", self.force_slope_label)
        self.use_video_fps = QCheckBox("Use each recording's frame rate")
        self.use_video_fps.setChecked(True)
        self.use_video_fps.toggled.connect(self.update_control_states)
        form.addRow("Frame rate", self.use_video_fps)
        self.fps_override = QDoubleSpinBox()
        self.fps_override.setRange(0.01, 100_000.0)
        self.fps_override.setDecimals(3)
        self.fps_override.setValue(20.0)
        form.addRow("FPS override", self.fps_override)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 100.0)
        self.threshold.setDecimals(1)
        self.threshold.setValue(80.0)
        movement_tooltip = (
            "Ranks automatically detected points by their maximum displacement. An "
            "80th-percentile cutoff retains approximately the top 20% most-moving "
            "automatic points. Manual points are always retained. This setting is used "
            "only for automatic or combined absolute-displacement measurements."
        )
        self.threshold.setToolTip(movement_tooltip)
        form.addRow("Automatic motion cutoff", self.threshold)
        form.labelForField(self.threshold).setToolTip(movement_tooltip)
        self.tracking_form = form
        self.save_coordinates = QCheckBox("Export coordinate CSV")
        self.save_coordinates.setChecked(True)
        self.save_plots = QCheckBox("Export plots")
        self.save_plots.setChecked(False)
        self.save_overlay_video = QCheckBox("Export tracked-points video")
        self.save_overlay_video.setChecked(True)
        self.save_overlay_video.setToolTip(
            "Write an annotated video showing the tracked point trajectories."
        )
        standard_output = QLabel(
            "Creates detailed per-video traces and one combined force CSV. "
            "Tracked-points videos are optional."
        )
        standard_output.setWordWrap(True)
        form.addRow("Video output", self.save_overlay_video)
        form.addRow("Standard output", standard_output)
        controls_layout.addWidget(settings_box)

        output_box = QGroupBox("Output")
        output_layout = QVBoxLayout(output_box)
        self.output_label = QLabel("Results will be saved beside the first recording.")
        self.output_label.setWordWrap(True)
        self.choose_output_button = QPushButton("Choose output folder")
        set_action_icon(self.choose_output_button, "open_folder")
        self.choose_output_button.clicked.connect(self.choose_output_folder)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.choose_output_button)
        controls_layout.addWidget(output_box)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Run tracking")
        set_action_icon(self.run_button, "run")
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self.run_tracking)
        self.cancel_button = QPushButton("Cancel")
        set_action_icon(self.cancel_button, "cancel")
        self.cancel_button.setProperty("role", "danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_tracking)
        actions.addWidget(self.run_button)
        actions.addWidget(self.cancel_button)
        actions.setAlignment(Qt.AlignHCenter)
        controls_layout.addLayout(actions)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        frame_controls = QHBoxLayout()
        frame_controls.addWidget(QLabel("Initialization frame"))
        self.initial_frame = QSpinBox()
        self.initial_frame.valueChanged.connect(self.initial_frame_changed)
        frame_controls.addWidget(self.initial_frame)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.frame_slider_changed)
        frame_controls.addWidget(self.frame_slider, 1)
        self.reference_frame_label = QLabel("Reference frame")
        frame_controls.addWidget(self.reference_frame_label)
        self.reference_frame = QSpinBox()
        self.reference_frame.valueChanged.connect(self.reference_frame_changed)
        frame_controls.addWidget(self.reference_frame)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(44)
        self.zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        frame_controls.addWidget(self.zoom_label)
        self.reset_zoom_button = QPushButton("Reset zoom")
        set_action_icon(self.reset_zoom_button, "clear")
        self.reset_zoom_button.setToolTip("Fit the complete video frame in the tracking canvas.")
        frame_controls.addWidget(self.reset_zoom_button)
        self.clear_points_button = QPushButton("Clear points")
        set_action_icon(self.clear_points_button, "delete")
        self.clear_points_button.setProperty("role", "danger")
        self.clear_points_button.setToolTip(
            "Remove every manually selected point from the current video."
        )
        self.clear_points_button.clicked.connect(self.clear_current_points)
        frame_controls.addWidget(self.clear_points_button)
        self.clear_roi_button = QPushButton("Clear ROI")
        set_action_icon(self.clear_roi_button, "clear")
        self.clear_roi_button.setToolTip("Remove the automatic-feature detection region for this video.")
        self.clear_roi_button.clicked.connect(self.clear_feature_roi)
        frame_controls.addWidget(self.clear_roi_button)
        self.apply_roi_button = QPushButton("Apply ROI to all videos")
        set_action_icon(self.apply_roi_button, "apply")
        self.apply_roi_button.setToolTip(
            "Copy this ROI proportionally to every selected recording."
        )
        self.apply_roi_button.clicked.connect(self.apply_feature_roi_to_all)
        frame_controls.addWidget(self.apply_roi_button)
        workspace_layout.addLayout(frame_controls)
        self.canvas = PointCanvas()
        self.canvas.points_changed.connect(self.points_changed)
        self.canvas.roi_changed.connect(self.feature_roi_changed)
        self.canvas.zoom_changed.connect(lambda percent: self.zoom_label.setText(f"{percent}%"))
        self.reset_zoom_button.clicked.connect(self.canvas.reset_zoom)
        workspace_layout.addWidget(self.canvas, 1)
        self.selection_status = QLabel(
            "Choose a video, select its initialization frame, then left-click one or more points."
        )
        workspace_layout.addWidget(self.selection_status)
        self.progress_label = QLabel("Ready")
        self.file_progress = QProgressBar()
        self.total_progress = QProgressBar()
        workspace_layout.addWidget(self.progress_label)
        workspace_layout.addWidget(self.file_progress)
        workspace_layout.addWidget(self.total_progress)

        results_box = QGroupBox("Point Tracking results")
        results_layout = QVBoxLayout(results_box)
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Result", "Analyzer-compatible CSV"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        results_layout.addWidget(self.results_table)
        result_buttons = QHBoxLayout()
        open_folder = QPushButton("Open output folder")
        set_action_icon(open_folder, "open_folder")
        open_folder.clicked.connect(self.open_output_folder)
        analyze = QPushButton("Open combined traces in Signal Analysis")
        set_action_icon(analyze, "open")
        analyze.clicked.connect(self.open_in_signal_analysis)
        result_buttons.addWidget(open_folder)
        result_buttons.addWidget(analyze)
        results_layout.addLayout(result_buttons)
        workspace_layout.addWidget(results_box)

        splitter.addWidget(controls)
        splitter.addWidget(videos)
        splitter.addWidget(workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([500, 420, 900])
        layout.addWidget(splitter, 1)
        self.restore_preferences()
        self.connect_preference_controls()
        self.update_control_states()

    def restore_preferences(self) -> None:
        settings = getattr(self.stimtrace_window, "settings", None)
        preferences = dict(getattr(settings, "point_tracking_settings", {}) or {})
        if not preferences:
            return
        if "drift_compensation" in preferences:
            preferences.pop("drift_compensation", None)
            settings.point_tracking_settings = preferences
            try:
                settings.save()
            except OSError:
                LOGGER.exception("Could not remove obsolete drift-compensation preference")
        self._restoring_preferences = True
        try:
            for control, key in (
                (self.strategy, "strategy"),
                (self.measurement, "measurement"),
                (self.baseline, "baseline"),
                (self.axis_projection, "axis_projection"),
            ):
                index = control.findData(preferences.get(key))
                if index >= 0:
                    control.setCurrentIndex(index)
            self.px_per_um.setValue(float(preferences.get("px_per_um", 1.0)))
            self.use_video_fps.setChecked(bool(preferences.get("use_video_fps", True)))
            self.fps_override.setValue(float(preferences.get("fps_override", 20.0)))
            self.threshold.setValue(float(preferences.get("threshold_percentile", 80.0)))
            self.save_coordinates.setChecked(bool(preferences.get("save_coordinates", True)))
            self.save_plots.setChecked(bool(preferences.get("save_plots", True)))
            self.save_overlay_video.setChecked(
                bool(preferences.get("save_overlay_video", True))
            )
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring invalid saved Point Tracking preferences", exc_info=True)
        finally:
            self._restoring_preferences = False

    def connect_preference_controls(self) -> None:
        for control in (self.strategy, self.measurement, self.baseline, self.axis_projection):
            control.currentIndexChanged.connect(self.persist_preferences)
        for control in (
            self.use_video_fps,
            self.save_coordinates,
            self.save_plots,
            self.save_overlay_video,
        ):
            control.toggled.connect(self.persist_preferences)
        for control in (self.px_per_um, self.fps_override, self.threshold):
            control.editingFinished.connect(self.persist_preferences)

    def persist_preferences(self, *_args) -> None:
        if self._restoring_preferences:
            return
        settings = getattr(self.stimtrace_window, "settings", None)
        if settings is None:
            return
        settings.point_tracking_settings = {
            "strategy": str(self.strategy.currentData()),
            "measurement": str(self.measurement.currentData()),
            "baseline": str(self.baseline.currentData()),
            "axis_projection": str(self.axis_projection.currentData()),
            "px_per_um": float(self.px_per_um.value()),
            "use_video_fps": bool(self.use_video_fps.isChecked()),
            "fps_override": float(self.fps_override.value()),
            "threshold_percentile": float(self.threshold.value()),
            "save_coordinates": bool(self.save_coordinates.isChecked()),
            "save_plots": bool(self.save_plots.isChecked()),
            "save_overlay_video": bool(self.save_overlay_video.isChecked()),
        }
        try:
            settings.save()
        except OSError:
            LOGGER.exception("Could not save Point Tracking preferences")

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "Expert point tracking help",
            "1. Add one or more videos. StimTrace reads FPS from each recording by default. "
            "Use an override only when the file metadata is missing or known to be wrong.\n\n"
            "2. Enter the calibration as pixels per micrometer. For manual tracking, select each "
            "video and left-click its initialization frame to add points; right-click removes the "
            "last point. Use the mouse wheel to zoom at the pointer and middle-button drag to pan; "
            "middle-button double-click or Reset zoom fits the complete frame. Relative distance "
            "uses the first two points.\n\n"
            "3. Absolute displacement can use the initialization frame, a selected real frame, or "
            "the legacy auto-relaxed definition from Cell Motion Tracker. Cumulative mode sums "
            "frame-to-frame motion. In Automatic features mode, left-drag a rectangular ROI around "
            "the pillar or tissue. Detection is restricted to this ROI; use Apply ROI to all videos "
            "when recordings share the same framing. Combined mode uses left-click for manual "
            "points and left-drag for the automatic ROI. The motion cutoff ranks automatic points "
            "by maximum displacement; an 80th-percentile cutoff keeps roughly the top 20%. Manual "
            "points are never removed by this filter.\n\n"
            "4. For absolute displacement, Automatic from tracked motion estimates one PCA bending "
            "axis from the retained points and projects every point onto it before force conversion. "
            "This removes perpendicular jitter; choose legacy Euclidean distance only to reproduce "
            "older outputs. The active model profile supplies the force slope: "
            "Force_uN = projected_distance_um x force_slope_uN_per_um.\n\n"
            "5. Run tracking. Forward-backward Lucas-Kanade checks reject unreliable points. "
            "StimTrace writes one detailed tracking CSV per recording, plus one combined force "
            "CSV aligned using each recording's native timestamps. Enable Export tracked-points "
            "video when you also want a labeled overlay for each recording. The "
            "combined CSV opens directly in StimTrace Signal Analysis.\n\n"
            "This is an expert exploratory tool. Inspect trajectories and lost-point gaps before "
            "using measurements quantitatively.",
        )

    def add_videos(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", VIDEO_FILTER)
        self.add_video_paths(map(Path, names))

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
        existing = {selection.path.resolve() for selection in self.selections}
        errors = []
        for path in map(Path, paths):
            if path.resolve() in existing:
                continue
            try:
                self.selections.append(probe_video(path))
                existing.add(path.resolve())
            except Exception as error:
                errors.append(f"{path.name}: {error}")
        self.populate_videos()
        if self.selections and self.video_table.currentRow() < 0:
            self.video_table.selectRow(0)
        if errors:
            QMessageBox.warning(self, "Some videos were skipped", "\n".join(errors))

    def clear_videos(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.selections.clear()
        self.video_table.setRowCount(0)
        self.canvas.set_frame(None, [])
        self.selection_status.setText("Add videos to begin.")

    def populate_videos(self) -> None:
        selected_row = max(0, self.video_table.currentRow())
        self.video_table.setRowCount(len(self.selections))
        for row, selection in enumerate(self.selections):
            if selection.feature_roi is not None and selection.points:
                point_summary = f"{len(selection.points)} + ROI"
            elif selection.feature_roi is not None:
                point_summary = "ROI set"
            else:
                point_summary = str(len(selection.points))
            values = [
                selection.path.name,
                f"{selection.fps:.3f}" if selection.fps > 0 else "Unknown",
                str(selection.frame_count),
                point_summary,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.video_table.setItem(row, column, item)
        if self.selections:
            self.video_table.selectRow(min(selected_row, len(self.selections) - 1))

    def current_selection(self) -> VideoSelection | None:
        row = self.video_table.currentRow()
        return self.selections[row] if 0 <= row < len(self.selections) else None

    def video_selected(self) -> None:
        selection = self.current_selection()
        if selection is None:
            return
        self._syncing_controls = True
        maximum = max(0, selection.frame_count - 1)
        for control in (self.initial_frame, self.reference_frame):
            control.setRange(0, maximum)
        self.reference_frame.setMinimum(selection.initial_frame)
        self.frame_slider.setRange(0, maximum)
        self.initial_frame.setValue(selection.initial_frame)
        self.frame_slider.setValue(selection.initial_frame)
        self.reference_frame.setValue(selection.baseline_frame)
        self._syncing_controls = False
        self.load_current_frame()

    def load_current_frame(self) -> None:
        selection = self.current_selection()
        if selection is None:
            return
        try:
            self.current_frame = read_video_frame(selection.path, selection.initial_frame)
            self.canvas.set_frame(
                self.current_frame,
                selection.points,
                selection.feature_roi,
            )
            self.update_selection_status()
        except Exception as error:
            QMessageBox.critical(self, "Could not read video", str(error))

    def initial_frame_changed(self, value: int) -> None:
        if self._syncing_controls:
            return
        selection = self.current_selection()
        if selection is None:
            return
        if value != selection.initial_frame and selection.points:
            selection.points.clear()
        selection.initial_frame = value
        selection.baseline_frame = max(selection.baseline_frame, value)
        self._syncing_controls = True
        self.frame_slider.setValue(value)
        self.reference_frame.setMinimum(value)
        self.reference_frame.setValue(selection.baseline_frame)
        self._syncing_controls = False
        self.load_current_frame()
        self.populate_videos()

    def frame_slider_changed(self, value: int) -> None:
        if not self._syncing_controls:
            self.initial_frame.setValue(value)

    def reference_frame_changed(self, value: int) -> None:
        if self._syncing_controls:
            return
        selection = self.current_selection()
        if selection is not None:
            selection.baseline_frame = value

    def points_changed(self) -> None:
        self.populate_videos()
        self.update_selection_status()

    def feature_roi_changed(self, roi: object) -> None:
        selection = self.current_selection()
        if selection is None:
            return
        selection.feature_roi = tuple(map(int, roi)) if roi is not None else None
        self.populate_videos()
        self.update_selection_status()

    def clear_feature_roi(self) -> None:
        selection = self.current_selection()
        if selection is None:
            return
        selection.feature_roi = None
        self.canvas.feature_roi = None
        self.canvas.update()
        self.populate_videos()
        self.update_selection_status()

    def clear_current_points(self) -> None:
        """Remove all manually selected points from the current recording."""
        selection = self.current_selection()
        if selection is None or not selection.points:
            return
        selection.points.clear()
        self.canvas.points = selection.points
        self.canvas.update()
        self.populate_videos()
        self.update_selection_status()

    def apply_feature_roi_to_all(self) -> None:
        source = self.current_selection()
        if source is None or source.feature_roi is None:
            QMessageBox.information(
                self,
                "ROI required",
                "Draw an ROI on the current video before applying it to all videos.",
            )
            return
        x, y, width, height = source.feature_roi
        for selection in self.selections:
            scale_x = selection.width / max(1, source.width)
            scale_y = selection.height / max(1, source.height)
            selection.feature_roi = (
                round(x * scale_x),
                round(y * scale_y),
                round(width * scale_x),
                round(height * scale_y),
            )
        self.load_current_frame()
        self.populate_videos()
        self.selection_status.setText(
            f"Applied the ROI proportionally to {len(self.selections)} videos."
        )

    def update_selection_status(self) -> None:
        selection = self.current_selection()
        if selection is None:
            return
        strategy = self.strategy.currentData()
        if strategy in {"automatic", "hybrid"}:
            roi_text = (
                f"ROI {selection.feature_roi} is ready."
                if selection.feature_roi is not None
                else "Left-drag over the pillar or tissue to define the detection ROI."
            )
            if strategy == "hybrid":
                self.selection_status.setText(
                    f"{selection.path.name}: {len(selection.points)} manual point(s) plus "
                    f"automatic features on frame {selection.initial_frame}. {roi_text}"
                )
            else:
                self.selection_status.setText(
                    f"{selection.path.name}: automatic features on frame "
                    f"{selection.initial_frame}. {roi_text}"
                )
        else:
            self.selection_status.setText(
                f"{selection.path.name}: {len(selection.points)} point(s) selected on frame "
                f"{selection.initial_frame}."
            )

    def update_control_states(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        strategy = self.strategy.currentData()
        automatic = strategy in {"automatic", "hybrid"}
        manual = strategy in {"manual", "hybrid"}
        hybrid = strategy == "hybrid"
        absolute = self.measurement.currentData() == "absolute"
        self.strategy.setEnabled(not running)
        self.measurement.setEnabled(not running)
        self.baseline.setEnabled(absolute and not running)
        self.set_tracking_form_row_visible(self.baseline, absolute)
        self.axis_projection.setEnabled(absolute and not running)
        self.set_tracking_form_row_visible(self.axis_projection, absolute)
        self.reference_frame.setEnabled(
            absolute and self.baseline.currentData() == "selected_frame" and not running
        )
        selected_reference = absolute and self.baseline.currentData() == "selected_frame"
        self.reference_frame_label.setVisible(selected_reference)
        self.reference_frame.setVisible(selected_reference)
        self.threshold.setEnabled(automatic and absolute and not running)
        self.set_tracking_form_row_visible(self.threshold, automatic and absolute)
        self.px_per_um.setEnabled(not running)
        self.use_video_fps.setEnabled(not running)
        self.fps_override.setEnabled(not self.use_video_fps.isChecked() and not running)
        self.set_tracking_form_row_visible(
            self.fps_override, not self.use_video_fps.isChecked()
        )
        self.save_coordinates.setEnabled(not running)
        self.save_plots.setEnabled(not running)
        self.save_overlay_video.setEnabled(not running)
        self.video_table.setEnabled(not running)
        self.initial_frame.setEnabled(not running)
        self.frame_slider.setEnabled(not running)
        self.add_video_button.setEnabled(not running)
        self.add_video_folder_button.setEnabled(not running)
        self.clear_video_button.setEnabled(not running)
        self.choose_output_button.setEnabled(not running)
        self.clear_roi_button.setVisible(automatic)
        self.apply_roi_button.setVisible(automatic)
        self.clear_roi_button.setEnabled(automatic and not running)
        self.apply_roi_button.setEnabled(automatic and not running)
        self.clear_points_button.setVisible(manual)
        self.clear_points_button.setEnabled(manual and not running)
        self.canvas.setEnabled(not running)
        self.canvas.set_roi_mode(automatic and not running, allow_points=hybrid)
        self.update_selection_status()

    def set_tracking_form_row_visible(self, field: QWidget, visible: bool) -> None:
        """Show a form field and its label only when the active mode uses it."""
        label = self.tracking_form.labelForField(field)
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    def choose_output_folder(self) -> None:
        initial = str(self.output_base or (self.selections[0].path.parent if self.selections else Path.home()))
        folder = QFileDialog.getExistingDirectory(self, "Choose point-tracking output folder", initial)
        if folder:
            self.output_base = Path(folder)
            self.output_label.setText(str(self.output_base))

    def options(self) -> TrackingOptions:
        return TrackingOptions(
            strategy=str(self.strategy.currentData()),
            measurement=str(self.measurement.currentData()),
            baseline=str(self.baseline.currentData()),
            px_per_um=float(self.px_per_um.value()),
            force_slope_un_per_um=self.force_slope_un_per_um,
            force_model_name=self.force_model_name,
            fps_override=None if self.use_video_fps.isChecked() else float(self.fps_override.value()),
            threshold_percentile=float(self.threshold.value()),
            save_coordinates=bool(self.save_coordinates.isChecked()),
            save_plots=bool(self.save_plots.isChecked()),
            save_overlay_video=bool(self.save_overlay_video.isChecked()),
            axis_projection=str(self.axis_projection.currentData()),
        )

    def refresh_force_calibration(self) -> None:
        settings = getattr(self.stimtrace_window, "settings", None)
        self.force_slope_un_per_um = float(
            getattr(settings, "force_slope_un_per_um", 0.0)
        )
        self.force_model_name = str(
            getattr(settings, "selected_model_name", "Unknown model")
        )
        if hasattr(self, "force_slope_label"):
            self.force_slope_label.setText(
                f"{self.force_slope_un_per_um:.4f} uN/um ({self.force_model_name})"
            )

    def validate_run(self, options: TrackingOptions) -> None:
        if not self.selections:
            raise ValueError("Add at least one video.")
        duplicate_names = duplicate_output_stems([item.path for item in self.selections])
        if duplicate_names:
            raise ValueError(
                "Point-tracking videos must have unique filenames so result files are not "
                "overwritten. Rename: " + ", ".join(duplicate_names)
            )
        if options.fps_override is None:
            missing = [item.path.name for item in self.selections if item.fps <= 0]
            if missing:
                raise ValueError("FPS metadata is missing for: " + ", ".join(missing))
        if options.strategy == "manual":
            required = 2 if options.measurement == "relative" else 1
            missing = [item.path.name for item in self.selections if len(item.points) < required]
            if missing:
                raise ValueError(
                    f"Select at least {required} point(s) in every video. Missing: "
                    + ", ".join(missing)
                )
        else:
            missing_roi = [
                item.path.name for item in self.selections if item.feature_roi is None
            ]
            if missing_roi:
                raise ValueError(
                    "Draw an automatic-feature ROI around the pillar or tissue for every video. "
                    "Missing: " + ", ".join(missing_roi)
                )

    def run_tracking(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        options = self.options()
        try:
            self.validate_run(options)
        except ValueError as error:
            QMessageBox.warning(self, "Tracking setup incomplete", str(error))
            return
        if self.output_base is None:
            base = self.selections[0].path.parent / "results"
        else:
            base = self.output_base
        job_dir = base / f"point_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = job_dir
        self.output_label.setText(str(job_dir))
        self.result_paths = []
        self.results_table.setRowCount(0)
        self.file_progress.setValue(0)
        self.total_progress.setValue(0)
        worker = TrackingThread(
            [
                VideoSelection(
                    path=selection.path,
                    fps=selection.fps,
                    frame_count=selection.frame_count,
                    width=selection.width,
                    height=selection.height,
                    initial_frame=selection.initial_frame,
                    baseline_frame=selection.baseline_frame,
                    points=list(selection.points),
                    feature_roi=selection.feature_roi,
                )
                for selection in self.selections
            ],
            options,
            job_dir,
            self,
        )
        worker.file_progress.connect(self.update_progress)
        worker.preview_ready.connect(self.update_tracking_preview)
        worker.status_changed.connect(self.progress_label.setText)
        worker.completed.connect(self.tracking_completed)
        worker.failed.connect(self.tracking_failed)
        worker.finished.connect(self.tracking_finished)
        self.worker = worker
        register_thread = getattr(self.stimtrace_window, "register_thread", None)
        if callable(register_thread):
            register_thread(worker, "expert-point-tracking")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.update_control_states()
        self.progress_label.setText("Starting point tracking...")
        worker.start()

    def update_progress(
        self, file_index: int, file_total: int, name: str, frame: int, frame_total: int
    ) -> None:
        file_percent = round(100 * frame / max(1, frame_total))
        total_percent = round(100 * ((file_index - 1) + frame / max(1, frame_total)) / file_total)
        self.progress_label.setText(
            f"Video {file_index}/{file_total}: {name}, frame {frame}/{frame_total}"
        )
        self.file_progress.setValue(file_percent)
        self.total_progress.setValue(total_percent)

    def update_tracking_preview(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        frame = payload.get("frame")
        points = payload.get("points", [])
        if not isinstance(frame, np.ndarray):
            return
        self.canvas.set_tracking_preview(
            frame,
            [tuple(map(float, point)) for point in points],
        )
        self.selection_status.setText(
            f"Live tracking: {payload.get('name', 'video')}, frame "
            f"{payload.get('frame_index', '?')}/{payload.get('frame_total', '?')} - "
            f"{len(points)} valid point(s)"
        )

    def cancel_tracking(self) -> None:
        if self.worker is not None:
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("Cancellation requested. Finishing the current video read...")

    def tracking_completed(self, result: object) -> None:
        if not isinstance(result, dict):
            self.tracking_failed("Point tracking returned an invalid result set.")
            return
        outputs = result.get("outputs", [])
        combined_path = result.get("combined_force_csv", "")
        overlay_count = int(result.get("overlay_count", 0))
        self.result_paths = [Path(str(combined_path))] if combined_path else []
        self.results_table.setRowCount(len(self.result_paths))
        for row, path in enumerate(self.result_paths):
            self.results_table.setItem(
                row,
                0,
                QTableWidgetItem("All recordings - combined force traces"),
            )
            self.results_table.setItem(row, 1, QTableWidgetItem(str(path)))
        if self.result_paths:
            self.results_table.selectRow(0)
        self.file_progress.setValue(100)
        self.total_progress.setValue(100)
        video_text = (
            f" Created {overlay_count} tracked-points video(s)."
            if overlay_count else ""
        )
        self.progress_label.setText(
            f"Tracking complete. Created {len(outputs)} detailed trace set(s), "
            f"plus one combined force CSV.{video_text}"
        )
        if self.result_paths:
            answer = QMessageBox.question(
                self,
                "Open tracked traces?",
                "All videos have finished tracking. Open the combined force traces in "
                "Signal Analysis now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.open_in_signal_analysis()

    def tracking_failed(self, message: str) -> None:
        self.file_progress.setValue(0)
        self.total_progress.setValue(0)
        self.progress_label.setText(message)
        if message != "Point tracking cancelled.":
            QMessageBox.critical(self, "Point tracking failed", message)

    def tracking_finished(self) -> None:
        worker = self.worker
        if worker is not None:
            worker.wait()
        self.worker = None
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.update_control_states()

    @staticmethod
    def find_relocated_result(path: Path) -> Path | None:
        """Find a result moved with its timestamped job folder under the former study root."""
        path = Path(path)
        if path.is_file():
            return path
        # Expected layout is <study>/results/<job>/<file>. Researchers commonly
        # reorganize that complete results folder into a condition subdirectory.
        if len(path.parents) < 3:
            return None
        study_root = path.parents[2]
        if not study_root.is_dir():
            return None
        candidates = [
            candidate
            for candidate in study_root.rglob(path.name)
            if candidate.is_file() and candidate.parent.name == path.parent.name
        ]
        return candidates[0] if len(candidates) == 1 else None

    def open_output_folder(self) -> None:
        if self.output_dir is None or not self.output_dir.exists():
            QMessageBox.information(self, "No output", "Run point tracking first.")
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))

    def open_in_signal_analysis(self) -> None:
        rows = sorted({index.row() for index in self.results_table.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, "No traces selected", "Select one or more exported rows.")
            return
        paths: list[Path] = []
        missing: list[Path] = []
        for row in rows:
            if not 0 <= row < len(self.result_paths):
                continue
            original = self.result_paths[row]
            resolved = self.find_relocated_result(original)
            if resolved is None:
                missing.append(original)
                continue
            if resolved != original:
                self.result_paths[row] = resolved
                self.results_table.setItem(row, 1, QTableWidgetItem(str(resolved)))
            paths.append(resolved)
        if missing:
            QMessageBox.warning(
                self,
                "Trace files not found",
                "The following Point Tracking result files no longer exist at their saved "
                "locations and could not be found elsewhere in the study folder:\n\n"
                + "\n".join(str(path) for path in missing)
                + "\n\nIf the results were moved outside the study folder, open them directly "
                "from Signal Analysis.",
            )
        if not paths:
            return
        if len({path.parent for path in paths}) == 1:
            self.output_dir = paths[0].parent
        opener = getattr(self.stimtrace_window, "open_signal_analysis_files", None)
        if not callable(opener):
            QMessageBox.warning(self, "Signal Analysis unavailable", "Could not reach the main StimTrace window.")
            return
        opener(paths)
        self.stimtrace_window.raise_()
