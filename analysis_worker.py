"""Run this module in a mounted Google Colab runtime to process Drive jobs."""
from __future__ import annotations

import json
import io
import gc
import html
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Reduce CUDA allocator fragmentation before PyTorch initializes its allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Albumentations otherwise performs a network version check while the worker imports.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
WORKER_VERSION = 64


@dataclass
class DeferredOverlayResult:
    """A completed scientific trace plus the CPU overlay work it enables."""

    trace_path: Path
    video_path: Path
    output: Path
    centers: list[tuple[float, float]]
    radius: int
    fps: float
    frame_size: tuple[int, int]


class PillarCenterTrackingError(ValueError):
    """A recording cannot produce a scientifically valid single-pillar trace."""


@dataclass
class CenterTrackingResult:
    """Internal centers plus per-frame measurement acceptance diagnostics."""

    centers: list[tuple[float, float]]
    states: list[str]
    innovation_distance_px: list[float]
    innovation_mahalanobis_d2: list[float]
    innovation_threshold_d2: float | None


def next_cuda_batch_size(current: int, requested: int) -> int:
    """Double the next CUDA probe, bounded only by the user-requested ceiling."""
    current = max(1, int(current))
    requested = max(1, int(requested))
    return min(requested, current * 2)


def lower_cuda_batch_size(current: int) -> int:
    """Halve a failed CUDA allocation while retaining a usable retry size."""
    current = max(1, int(current))
    return max(1, (current + 1) // 2)


def indexed_signal(
    time_values,
    signal_values,
    *,
    time_precision: int = 9,
) -> pd.Series:
    """Return one numeric signal indexed by its own rounded timestamps."""
    frame = pd.DataFrame({
        "time_s": pd.to_numeric(time_values, errors="coerce"),
        "signal": pd.to_numeric(signal_values, errors="coerce"),
    }).dropna(subset=["time_s", "signal"])
    if frame.empty:
        return pd.Series(dtype=float)
    frame["time_s"] = frame["time_s"].round(time_precision)
    return frame.groupby("time_s", sort=True)["signal"].mean()


def combine_signals_by_time(signals: dict[str, pd.Series]) -> pd.DataFrame:
    """Combine signals on the union of their timestamps without resampling."""
    nonempty = {name: signal for name, signal in signals.items() if not signal.empty}
    if not nonempty:
        raise ValueError("No numeric force traces were available for the combined CSV.")
    combined = pd.concat(nonempty, axis=1, join="outer").sort_index()
    combined.index.name = "time_s"
    return combined.reset_index()


class JobCancelled(Exception):
    pass


def runtime_hardware(device: torch.device) -> dict:
    cpu_threads = os.cpu_count() or 1
    details = {
        "device": str(device),
        "device_name": "CPU",
        "cpu_threads": cpu_threads,
        "cpu_postprocess_limit": max(1, cpu_threads - 1),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda or "",
    }
    try:
        details["system_memory_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3,
            1,
        )
    except (AttributeError, OSError, ValueError):
        details["system_memory_gb"] = 0
    if device.type != "cuda":
        return details
    properties = torch.cuda.get_device_properties(device)
    details.update({
        "device_name": properties.name,
        "gpu_memory_gb": round(properties.total_memory / 1024**3, 1),
    })
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        utilization, memory_mib, power_watts = [value.strip() for value in query.split(",")]
        details.update({
            "gpu_utilization_pct": float(utilization),
            "gpu_memory_used_gb": round(float(memory_mib) / 1024, 1),
            "gpu_power_watts": round(float(power_watts), 1),
        })
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        pass
    return details


class MultiTaskUnet(nn.Module):
    # Center head is retained because it is part of the supplied checkpoint.
    def __init__(self):
        super().__init__()
        self.unet = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1, encoder_weights=None)
        channels = self.unet.encoder.out_channels[-1]
        self.center_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 256), nn.ReLU(inplace=True), nn.Linear(256, 2))

    def forward(self, image):
        logits = self.unet(image)
        return logits, self.center_head(self.unet.encoder(image)[-1])


class KalmanCenter:
    def __init__(self, x: float, y: float, dt: float, q_pos: float, q_vel: float, r: float):
        self.dt = dt
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.p = np.eye(4) * 20.0
        self.f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.r = np.eye(2) * r

    def predict(self):
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q

    def innovation_statistics(
        self, x: float, y: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return residual, residual covariance, and normalized innovation squared."""
        residual = np.array([x, y], dtype=float) - self.h @ self.x
        covariance = self.h @ self.p @ self.h.T + self.r
        distance_squared = float(residual @ np.linalg.solve(covariance, residual))
        # Round-off can only make this infinitesimally negative; the statistic is
        # mathematically non-negative for a positive-definite covariance matrix.
        return residual, covariance, max(0.0, distance_squared)

    def update(
        self,
        x: float,
        y: float,
        statistics: tuple[np.ndarray, np.ndarray, float] | None = None,
    ):
        residual, covariance, _ = statistics or self.innovation_statistics(x, y)
        gain = np.linalg.solve(covariance, (self.p @ self.h.T).T).T
        self.x = self.x + gain @ residual
        identity_minus_gain = np.eye(4) - gain @ self.h
        # Joseph form is more numerically stable and preserves covariance
        # symmetry/positive semidefiniteness better than (I-KH)P.
        self.p = (
            identity_minus_gain @ self.p @ identity_minus_gain.T
            + gain @ self.r @ gain.T
        )
        self.p = (self.p + self.p.T) / 2.0
        return float(self.x[0]), float(self.x[1])


def transforms():
    return A.Compose([A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_REFLECT_101, p=1), A.Resize(512, 512), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


def video_fps(capture, video_name: str) -> float:
    """Read a scientifically usable constant frame rate from an OpenCV capture."""
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"{video_name} has no valid frame-rate metadata. StimTrace will not assume an "
            "arbitrary frame rate because that would invalidate the time axis and kinetic "
            "measurements. Transcode the recording with a fixed frame rate and retry."
        )
    return fps


def ellipse(mask: np.ndarray, min_area: int = 80, min_points: int = 40):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return np.nan, np.nan, np.nan, np.nan, np.nan, "none"
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) >= min_area and len(contour) >= min_points:
        try:
            (x, y), (a, b), angle = cv2.fitEllipse(contour)
            return x, y, a, b, angle, "ellipsefit"
        except cv2.error: pass
    moments = cv2.moments(contour)
    if moments["m00"]: return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"], np.nan, np.nan, np.nan, "centroid"
    return np.nan, np.nan, np.nan, np.nan, np.nan, "none"


def refine(mask: np.ndarray, iterations: int):
    result = mask
    fit = ellipse(result)
    if fit[-1] != "ellipsefit" or iterations <= 0:
        return result, fit

    # Repeated contour fitting over a 1080p frame dominates Colab's limited CPU.
    # The refinement only intersects the selected component with its fitted ellipse,
    # so all later work can be performed in that component's bounding region.
    nonzero = cv2.findNonZero(result)
    if nonzero is None:
        return result, fit
    bound_x, bound_y, width, height = cv2.boundingRect(nonzero)
    padding = 3
    x0 = max(0, bound_x - padding)
    y0 = max(0, bound_y - padding)
    x1 = min(result.shape[1], bound_x + width + padding)
    y1 = min(result.shape[0], bound_y + height + padding)
    result = result[y0:y1, x0:x1].copy()
    x, y, a, b, angle, kind = fit
    fit = (x - x0, y - y0, a, b, angle, kind)
    for _ in range(iterations):
        x, y, a, b, angle, kind = fit
        if kind != "ellipsefit": break
        candidate = np.zeros_like(result)
        cv2.ellipse(candidate, (int(round(x)), int(round(y))), (max(1, int(round(a / 2))), max(1, int(round(b / 2)))), angle, 0, 360, 1, -1)
        result = ((result > 0) & (candidate > 0)).astype(np.uint8)
        fit = ellipse(result)
    x, y, a, b, angle, kind = fit
    if np.isfinite(x) and np.isfinite(y):
        fit = (x + x0, y + y0, a, b, angle, kind)
    return result, fit


def postprocess_prediction(bgr: np.ndarray, probability: np.ndarray, threshold: float, refine_iterations: int):
    """Convert one network prediction into a fitted pillar mask."""
    h, w = bgr.shape[:2]
    probability = cv2.GaussianBlur(probability.astype(np.float32), (7, 7), 1.2)
    mask = np.array(
        Image.fromarray((probability > threshold).astype(np.uint8) * 255).resize(
            (w, h), Image.Resampling.NEAREST
        )
    ) > 128
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if components > 1:
        mask = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return refine(mask, refine_iterations)


def tracking_filter_mode(cfg: dict) -> str:
    """Return and validate the configured temporal center-filter mode."""
    mode = str(cfg.get("tracking_filter_mode", "kalman")).strip().lower()
    if mode not in {"kalman", "none"}:
        raise ValueError(f"Unsupported tracking filter mode: {mode!r}.")
    return mode


KALMAN_INNOVATION_GATE_CONFIDENCE = 0.99
KALMAN_INNOVATION_GATE_MIN_RADIUS_PX = 120.0


def kalman_innovation_gate_threshold() -> float:
    """Return the fixed two-dimensional chi-square threshold for Kalman QC.

    For two observed coordinates, the chi-square quantile has the exact closed
    form ``-2 log(1-confidence)``. Avoiding a SciPy dependency keeps the same
    calculation available in the standalone Colab worker.
    """
    return float(-2.0 * np.log1p(-KALMAN_INNOVATION_GATE_CONFIDENCE))


def track_centers(fits: list[tuple], fps: float, cfg: dict) -> CenterTrackingResult:
    """Track raw fits and retain diagnostics for rejected Kalman measurements."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Tracking requires a finite positive frame rate.")
    mode = tracking_filter_mode(cfg)
    threshold = kalman_innovation_gate_threshold() if mode == "kalman" else None
    centers: list[tuple[float, float]] = []
    states: list[str] = []
    innovation_distances: list[float] = []
    innovation_squared: list[float] = []
    minimum_radius = KALMAN_INNOVATION_GATE_MIN_RADIUS_PX
    filter_state = None
    for x, y, *_ in fits:
        measurement_valid = bool(np.isfinite(x) and np.isfinite(y))
        if mode == "none":
            centers.append(
                (float(x), float(y)) if measurement_valid else (np.nan, np.nan)
            )
            states.append("measured" if measurement_valid else "missing")
            innovation_distances.append(np.nan)
            innovation_squared.append(np.nan)
            continue
        if measurement_valid:
            if filter_state is None:
                filter_state = KalmanCenter(
                    x,
                    y,
                    1 / fps,
                    float(cfg["kalman_q_pos"]),
                    float(cfg["kalman_q_vel"]),
                    float(cfg["kalman_r"]),
                )
                smooth = (x, y)
                state = "measured"
                distance_px = np.nan
                distance_squared = np.nan
            else:
                filter_state.predict()
                statistics = filter_state.innovation_statistics(x, y)
                distance_px = float(np.linalg.norm(statistics[0]))
                distance_squared = statistics[2]
                if (
                    threshold is None
                    or distance_px <= minimum_radius
                    or distance_squared <= threshold
                ):
                    smooth = filter_state.update(x, y, statistics)
                    state = "measured"
                else:
                    # Keep the prediction only as internal/overlay state. The
                    # rejected observed center is exported as NaN by trace_from_tracking.
                    smooth = (float(filter_state.x[0]), float(filter_state.x[1]))
                    state = "innovation_rejected"
        elif filter_state is not None:
            filter_state.predict()
            smooth = (float(filter_state.x[0]), float(filter_state.x[1]))
            state = "missing"
            distance_px = np.nan
            distance_squared = np.nan
        else:
            smooth = (np.nan, np.nan)
            state = "missing"
            distance_px = np.nan
            distance_squared = np.nan
        centers.append(smooth)
        states.append(state)
        innovation_distances.append(distance_px)
        innovation_squared.append(distance_squared)
    return CenterTrackingResult(
        centers,
        states,
        innovation_distances,
        innovation_squared,
        threshold,
    )


def kalman_track(fits: list[tuple], fps: float, cfg: dict) -> list[tuple[float, float]]:
    """Compatibility wrapper returning raw or Kalman-filtered center coordinates."""
    return track_centers(fits, fps, cfg).centers


def trace_from_tracking(
    fits: list[tuple],
    centers: list[tuple[float, float]],
    pixel_counts: list[int],
    fps: float,
    cfg: dict,
    benchmark_name: str | None = None,
    tracking_states: list[str] | None = None,
    innovation_distance_px: list[float] | None = None,
    innovation_mahalanobis_d2: list[float] | None = None,
) -> pd.DataFrame:
    if tracking_states is not None and len(tracking_states) != len(fits):
        raise ValueError("Tracking-state diagnostics must match the fitted-frame count.")
    if innovation_mahalanobis_d2 is not None and len(innovation_mahalanobis_d2) != len(fits):
        raise ValueError("Innovation diagnostics must match the fitted-frame count.")
    if innovation_distance_px is not None and len(innovation_distance_px) != len(fits):
        raise ValueError("Innovation-distance diagnostics must match the fitted-frame count.")
    rows = []
    for index, (fit, smooth, count) in enumerate(zip(fits, centers, pixel_counts)):
        x, y, a, b, angle, kind = fit
        detected = bool(np.isfinite(x) and np.isfinite(y) and count > 0)
        tracking_state = (
            tracking_states[index]
            if tracking_states is not None
            else "measured" if detected else "missing"
        )
        measurement_valid = detected and tracking_state == "measured"
        rows.append({
            "frame_idx": index,
            "time_s": index / fps,
            "pillar_pixels": count,
            "raw_center_x": x,
            "raw_center_y": y,
            # Kalman predictions remain available to overlay rendering through
            # ``centers`` but are not exported as observed scientific measurements.
            "smooth_center_x": smooth[0] if measurement_valid else np.nan,
            "smooth_center_y": smooth[1] if measurement_valid else np.nan,
            "tracking_state": tracking_state,
            "innovation_distance_px": (
                innovation_distance_px[index]
                if innovation_distance_px is not None
                else np.nan
            ),
            "innovation_mahalanobis_d2": (
                innovation_mahalanobis_d2[index]
                if innovation_mahalanobis_d2 is not None
                else np.nan
            ),
            "ellipse_axis_a": a,
            "ellipse_axis_b": b,
            "ellipse_angle_deg": angle,
            "ellipse_method": kind,
        })
    trace = apply_force_calibration(pd.DataFrame(rows), cfg)
    filter_mode = tracking_filter_mode(cfg)
    trace["tracking_filter_mode"] = filter_mode
    gate_threshold = (
        kalman_innovation_gate_threshold() if filter_mode == "kalman" else None
    )
    trace["kalman_innovation_gate_enabled"] = gate_threshold is not None
    trace["kalman_innovation_gate_confidence"] = (
        KALMAN_INNOVATION_GATE_CONFIDENCE
        if gate_threshold is not None
        else np.nan
    )
    trace["kalman_innovation_gate_threshold_d2"] = (
        gate_threshold if gate_threshold is not None else np.nan
    )
    trace["kalman_innovation_gate_min_radius_px"] = (
        KALMAN_INNOVATION_GATE_MIN_RADIUS_PX
        if gate_threshold is not None
        else np.nan
    )
    if benchmark_name:
        trace["benchmark_name"] = benchmark_name
        trace["kalman_q_pos_px2"] = cfg["kalman_q_pos"]
        trace["kalman_q_vel_px2_per_s2"] = cfg["kalman_q_vel"]
        trace["kalman_r_px2"] = cfg["kalman_r"]
    return trace


def _motion_axis(coordinates: np.ndarray, cfg: dict) -> np.ndarray:
    """Return a unit image-plane vector for the calibrated uniaxial motion."""
    if cfg.get("bending_axis_mode", "automatic") == "fixed_angle":
        radians = np.deg2rad(float(cfg.get("bending_axis_angle_deg", 0.0)))
        return np.array([np.cos(radians), np.sin(radians)], dtype=float)
    centered = coordinates - np.nanmedian(coordinates, axis=0)
    covariance = np.cov(centered, rowvar=False)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return np.array([1.0, 0.0], dtype=float)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 0 else np.array([1.0, 0.0], dtype=float)


def apply_force_calibration(trace: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Project tracked centers onto the bending axis and apply the force calibration."""
    trace = trace.copy()
    coordinates = trace[["smooth_center_x", "smooth_center_y"]].to_numpy(dtype=float)
    valid = np.isfinite(coordinates).all(axis=1)
    if valid.sum() < 2:
        raise PillarCenterTrackingError(
            "Fewer than two measured pillar centers were available. Kalman predictions "
            "cannot substitute for observed segmentation positions."
        )
    axis = _motion_axis(coordinates[valid], cfg)
    projected = coordinates @ axis
    valid_projection = projected[valid]
    median_projection = float(np.nanmedian(valid_projection))
    lower_excursion = median_projection - float(np.nanpercentile(valid_projection, 5))
    upper_excursion = float(np.nanpercentile(valid_projection, 95)) - median_projection
    if lower_excursion > upper_excursion:
        axis = -axis
        projected = -projected
        valid_projection = -valid_projection
    diastolic_limit = float(np.nanpercentile(valid_projection, 20))
    diastolic_rows = valid & (projected <= diastolic_limit)
    if not diastolic_rows.any():
        diastolic_rows = valid
    diastolic_indices = np.flatnonzero(diastolic_rows)
    median_diastolic_projection = float(np.nanmedian(projected[diastolic_rows]))
    reference_index = int(
        diastolic_indices[
            np.argmin(np.abs(projected[diastolic_indices] - median_diastolic_projection))
        ]
    )
    # Use a center from one real frame rather than independently combining x
    # and y statistics that may never have coexisted physically.
    reference_center = coordinates[reference_index].copy()
    displacement_px = (coordinates - reference_center) @ axis
    reference_description = (
        f"frame {reference_index}, nearest actual frame to the median of the "
        "lowest 20% projected diastolic positions"
    )

    if bool(cfg.get("legacy_area_normalization", False)):
        areas = pd.to_numeric(trace["pillar_pixels"], errors="coerce").replace(0, np.nan)
        displacement_px = displacement_px * float(areas.max()) / areas.to_numpy(dtype=float)

    displacement_um = displacement_px * float(cfg["pixel_to_um"])
    trace["axis_projected_position_px"] = projected
    trace["displacement_px"] = displacement_px
    trace["displacement_um"] = displacement_um
    # Retain these two names for compatibility with older result readers. They now
    # contain the signed, axis-projected displacement (including the legacy area
    # correction only when the user explicitly enabled it).
    trace["XY_combo"] = displacement_px
    trace["XY_combo_norm"] = displacement_px
    trace["Force_uN"] = displacement_um * float(cfg["force_slope_un_per_um"])
    trace["force_reference_mode"] = "active_diastolic"
    trace["force_reference_description"] = reference_description
    trace["bending_axis_xy"] = f"{axis[0]:.9g},{axis[1]:.9g}"
    trace["reference_center_xy"] = (
        f"{reference_center[0]:.9g},{reference_center[1]:.9g}"
    )
    trace["area_normalization"] = (
        "legacy enabled" if bool(cfg.get("legacy_area_normalization", False)) else "disabled"
    )
    trace["calibration_equation"] = "slope * active_displacement_um"
    return trace


def benchmark_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return slug or "configuration"


def benchmark_configurations(cfg: dict) -> list[dict]:
    """Return the raw reference and the user-selected Kalman variants."""
    combinations = cfg.get("kalman_benchmark", [])
    if not isinstance(combinations, list):
        raise ValueError("Kalman benchmark settings must be a list of configurations.")
    # Every smoothing comparison requires the same raw-center reference.
    return [{"name": "Unfiltered", "tracking_filter_mode": "none"}, *combinations]


def prepare_live_trace_panel(
    trace: pd.DataFrame,
    height: int,
    title: str,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Prepare the static chart and coordinates used by every output frame."""
    width = max(560, int(height * 1.05))
    panel = np.full((height, width, 3), (22, 28, 32), dtype=np.uint8)
    left, right, top, bottom = 72, 24, 70, 58
    x0, x1, y0, y1 = left, width - right, top, height - bottom
    cv2.rectangle(panel, (x0, y0), (x1, y1), (59, 70, 78), 1)
    force = trace["Force_uN"].to_numpy(dtype=float)
    time_values = trace["time_s"].to_numpy(dtype=float)
    finite = np.isfinite(force)
    if not finite.any():
        return (
            panel,
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.zeros(len(trace), dtype=np.int32),
            (y0, y1),
        )
    y_min, y_max = float(np.nanmin(force)), float(np.nanmax(force))
    padding = max(0.05, (y_max - y_min) * 0.08)
    y_min -= padding
    y_max += padding
    duration = max(float(time_values[-1]), 1e-9)
    cv2.putText(panel, title, (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 240, 243), 2, cv2.LINE_AA)
    if tracking_filter_mode(cfg) == "none":
        subtitle = "Tracking filter: none (raw segmentation centers)"
    else:
        subtitle = (
            f"q pos {cfg['kalman_q_pos']:g} px^2 | q vel {cfg['kalman_q_vel']:g} px^2/s^2 | "
            f"r {cfg['kalman_r']:g} px^2"
        )
    cv2.putText(panel, subtitle, (left, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (185, 197, 205), 1, cv2.LINE_AA)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(y1 - fraction * (y1 - y0))
        value = y_min + fraction * (y_max - y_min)
        cv2.line(panel, (x0, y), (x1, y), (51, 62, 70), 1)
        cv2.putText(panel, f"{value:.2f}", (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 197, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, "Force (uN)", (5, top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 197, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, "Time (s)", (x1 - 45, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 197, 205), 1, cv2.LINE_AA)
    cursor_positions = np.rint(
        x0 + np.clip(time_values / duration, 0.0, 1.0) * (x1 - x0)
    ).astype(np.int32)
    valid_indices = np.flatnonzero(finite)
    points = np.column_stack((
        cursor_positions[valid_indices],
        np.rint(
            y1
            - (force[valid_indices] - y_min)
            / max(y_max - y_min, 1e-9)
            * (y1 - y0)
        ).astype(np.int32),
    )).astype(np.int32)
    return (
        panel,
        points,
        valid_indices.astype(np.int32),
        cursor_positions,
        (y0, y1),
    )


def draw_prepared_trace_panel(
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
    frame_index: int,
) -> np.ndarray:
    """Reveal a prepared trace up to the video frame and draw its cursor."""
    base, points, valid_indices, cursor_positions, (y0, y1) = prepared
    panel = base.copy()
    point_count = int(np.searchsorted(valid_indices, frame_index, side="right"))
    if point_count > 1:
        cv2.polylines(panel, [points[:point_count]], False, (68, 196, 184), 2, cv2.LINE_AA)
    if len(cursor_positions):
        cursor_x = int(cursor_positions[min(frame_index, len(cursor_positions) - 1)])
        cv2.line(panel, (cursor_x, y0), (cursor_x, y1), (77, 130, 188), 1)
    return panel


class CenterTrailRenderer:
    """Draw one tracked center and its accumulated valid path on video frames."""

    def __init__(self, color: tuple[int, int, int] = (0, 255, 255)) -> None:
        self.color = color
        self.trail_layer: np.ndarray | None = None
        self.previous_center: tuple[int, int] | None = None

    def render(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        radius: int,
    ) -> np.ndarray:
        if self.trail_layer is None or self.trail_layer.shape != frame.shape:
            self.trail_layer = np.zeros_like(frame)
            self.previous_center = None
        coordinates = np.asarray(center, dtype=float).reshape(2)
        current_center: tuple[int, int] | None = None
        if np.isfinite(coordinates).all():
            current_center = tuple(np.rint(coordinates).astype(int))
            if self.previous_center is not None:
                cv2.line(
                    self.trail_layer,
                    self.previous_center,
                    current_center,
                    self.color,
                    max(2, round(min(frame.shape[:2]) / 480)),
                    cv2.LINE_AA,
                )
            self.previous_center = current_center
        else:
            # Keep the established path, but never bridge across missing tracking.
            self.previous_center = None
        rendered = cv2.addWeighted(frame, 1.0, self.trail_layer, 0.9, 0.0)
        if current_center is not None:
            cv2.circle(rendered, current_center, radius, self.color, 4)
            cv2.circle(rendered, current_center, 4, self.color, -1)
        return rendered


def create_benchmark_videos(
    video_path: Path,
    fits: list[tuple],
    fps: float,
    radius: int,
    specifications: list[dict],
    cancel_check=None,
    stage_callback=None,
) -> None:
    """Render all Kalman variants while decoding the source recording once."""
    if not specifications:
        return
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video for benchmark output: {video_path}")
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    chart_width = max(560, int(height * 1.05))
    writers = []
    prepared_panels = []
    trail_renderers = []
    for specification in specifications:
        destination = specification["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width + chart_width, height),
        )
        if not writer.isOpened():
            capture.release()
            for opened_writer in writers:
                opened_writer.release()
            raise RuntimeError(f"Could not create benchmark video: {destination}")
        writers.append(writer)
        prepared_panels.append(prepare_live_trace_panel(
            specification["trace"],
            height,
            specification["title"],
            specification["cfg"],
        ))
        trail_renderers.append(CenterTrailRenderer())
    frame_count = min(
        len(fits),
        *(len(specification["centers"]) for specification in specifications),
    )
    total_output_frames = frame_count * len(specifications)
    started = time.perf_counter()
    print(
        f"Rendering {len(specifications)} Kalman benchmark videos for {video_path.name} "
        f"in one source pass ({width + chart_width}x{height}, CPU video encoding)...",
        flush=True,
    )
    try:
        for index in range(frame_count):
            if cancel_check and cancel_check():
                raise InterruptedError("Processing cancelled.")
            ok, frame = capture.read()
            if not ok:
                break
            for specification_index, (specification, prepared, writer, trail_renderer) in enumerate(
                zip(specifications, prepared_panels, writers, trail_renderers)
            ):
                annotated = trail_renderer.render(
                    frame,
                    specification["centers"][index],
                    radius,
                )
                panel = draw_prepared_trace_panel(prepared, index)
                writer.write(np.hstack((annotated, panel)))
                if stage_callback:
                    completed = index * len(specifications) + specification_index + 1
                    stage_callback(
                        video_path.name,
                        "benchmark_video",
                        completed,
                        total_output_frames,
                    )
    finally:
        capture.release()
        for writer in writers:
            writer.release()
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(
        f"Benchmark videos complete for {video_path.name}: {total_output_frames} output "
        f"frames in {elapsed:.1f}s ({total_output_frames / elapsed:.1f} encoded frames/s).",
        flush=True,
    )


def create_overlay_video(
    video_path: Path,
    output: Path,
    centers: list[tuple[float, float]],
    radius: int,
    fps: float,
    frame_size: tuple[int, int],
    *,
    cancel_check=None,
    stage_callback=None,
) -> Path:
    """Encode one tracking overlay; safe to run in a dedicated CPU thread."""
    overlay = output / "overlays"
    overlay.mkdir(exist_ok=True)
    destination = overlay / f"{video_path.stem}_overlay.avi"
    print(f"Creating overlay video for {video_path.name}...", flush=True)
    capture = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        frame_size,
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"Could not create overlay video for: {video_path}")
    trail_renderer = CenterTrailRenderer()
    try:
        for frame_index, (x, y) in enumerate(centers, start=1):
            if cancel_check and cancel_check():
                raise InterruptedError("Processing cancelled.")
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(trail_renderer.render(frame, (x, y), radius))
            if stage_callback:
                stage_callback(video_path.name, "overlay", frame_index, len(centers))
    finally:
        capture.release()
        writer.release()
    return destination


def render_segmentation_fit_overlay(frame: np.ndarray, fit: tuple) -> np.ndarray:
    """Draw the same fitted-pillar overlay used to inspect normal segmentation.

    The validation QC video is intended to show the measured ellipse and its raw
    centre, not a semi-transparent binary mask that obscures the recording.
    """
    result = frame.copy()
    x, y, axis_a, axis_b, angle, kind = fit
    values = np.asarray((x, y), dtype=float)
    if not np.isfinite(values).all():
        return result
    center = tuple(np.rint(values).astype(int))
    color = (0, 255, 255)  # Yellow in BGR, matching the standard overlay.
    if kind == "ellipsefit" and np.isfinite([axis_a, axis_b, angle]).all():
        axes = tuple(np.maximum(1, np.rint([axis_a / 2.0, axis_b / 2.0]).astype(int)))
        cv2.ellipse(result, center, axes, float(angle), 0, 360, color, 2, cv2.LINE_AA)
    cv2.circle(result, center, 3, color, -1, cv2.LINE_AA)
    return result


def process_video(
    video_path: Path,
    output: Path,
    model,
    device,
    cfg: dict,
    progress_callback=None,
    stage_callback=None,
    cancel_check=None,
    runtime_settings_callback=None,
    defer_overlay: bool = False,
    segmentation_overlay_destination: Path | None = None,
) -> Path | dict[str, Path] | DeferredOverlayResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = video_fps(cap, video_path.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_label = str(total_frames) if total_frames else "?"
    fits, pixel_counts = [], []
    tfm = transforms(); threshold = cfg["mask_threshold"]; last_progress_update = 0.0
    progress_interval = max(
        2.0,
        float(cfg.get("progress_interval_seconds", cfg.get("preview_interval_seconds", 5.0))),
    )
    postprocess_pool = None
    segmentation_overlay_writer = None
    active_cpu_workers = 0
    batch_number = 0
    safe_cuda_batch_size = (
        max(1, int(cfg["_cuda_safe_batch_size"]))
        if device.type == "cuda" and cfg.get("_cuda_safe_batch_size")
        else max(1, int(cfg["_cuda_initial_batch_size"]))
        if device.type == "cuda" and cfg.get("_cuda_initial_batch_size")
        else None
    )
    cuda_tuning_complete = bool(cfg.get("_cuda_batch_tuning_complete"))

    def infer_batch(
        batch_tensors: list[torch.Tensor], preferred_size: int
    ) -> tuple[np.ndarray, int, bool]:
        """Run inference, reducing CUDA batch size after an out-of-memory error."""
        attempt_size = max(1, min(preferred_size, len(batch_tensors)))
        memory_reduced = False
        while True:
            tensor_batch = None
            try:
                chunks = []
                for start in range(0, len(batch_tensors), attempt_size):
                    tensor_batch = torch.stack(
                        batch_tensors[start:start + attempt_size]
                    ).to(device, non_blocking=True)
                    with torch.inference_mode():
                        # Center regression is unused during inference. Calling the U-Net
                        # directly avoids a second encoder pass in MultiTaskUnet.forward().
                        chunk = torch.sigmoid(model.unet(tensor_batch))[:, 0].cpu().numpy()
                    chunks.append(chunk)
                    del tensor_batch
                    tensor_batch = None
                return np.concatenate(chunks, axis=0), attempt_size, memory_reduced
            except RuntimeError as error:
                cuda_oom_type = getattr(torch, "OutOfMemoryError", ())
                is_cuda_oom = device.type == "cuda" and (
                    (bool(cuda_oom_type) and isinstance(error, cuda_oom_type))
                    or "cuda out of memory" in str(error).lower()
                )
                if not is_cuda_oom or attempt_size <= 1:
                    raise
                chunks.clear()
                tensor_batch = None
                gc.collect()
                torch.cuda.empty_cache()
                reduced_size = lower_cuda_batch_size(attempt_size)
                print(
                    f"CUDA memory limit reached at batch {attempt_size}. "
                    f"Retrying automatically with batch {reduced_size}.",
                    flush=True,
                )
                attempt_size = reduced_size
                memory_reduced = True

    if segmentation_overlay_destination is not None:
        segmentation_overlay_destination.parent.mkdir(parents=True, exist_ok=True)
        segmentation_overlay_writer = cv2.VideoWriter(
            str(segmentation_overlay_destination),
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (frame_width, frame_height),
        )
        if not segmentation_overlay_writer.isOpened():
            segmentation_overlay_writer.release()
            cap.release()
            raise RuntimeError(
                f"Could not create segmentation overlay: {segmentation_overlay_destination.name}"
            )

    try:
        while True:
            if cancel_check and cancel_check():
                raise InterruptedError("Processing cancelled.")
            runtime_settings = (
                runtime_settings_callback() or {}
                if runtime_settings_callback
                else {}
            )
            requested_batch_size = max(
                1,
                int(runtime_settings.get(
                    "inference_batch_size",
                    cfg.get("inference_batch_size", 8),
                )),
            )
            batch_size = (
                min(requested_batch_size, safe_cuda_batch_size)
                if safe_cuda_batch_size is not None
                else requested_batch_size
            )
            requested_cpu_workers = max(
                1,
                int(runtime_settings.get(
                    "cpu_postprocess_workers",
                    cfg.get("cpu_postprocess_workers", 1),
                )),
            )
            # Postprocessing is a separate stage, so use all requested cores
            # except one reserved for the runtime and progress reporting.
            cpu_workers = min(
                requested_cpu_workers,
                batch_size,
                max(1, (os.cpu_count() or 1) - 1),
            )
            if cpu_workers != active_cpu_workers:
                if postprocess_pool:
                    postprocess_pool.shutdown(wait=True)
                postprocess_pool = (
                    ThreadPoolExecutor(
                        max_workers=cpu_workers,
                        thread_name_prefix="stimtrace-mask",
                    )
                    if cpu_workers > 1
                    else None
                )
                active_cpu_workers = cpu_workers
            batch_started = time.perf_counter()
            batch_frames = []
            batch_tensors = []
            for _ in range(batch_size):
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                batch_frames.append(bgr)
                batch_tensors.append(tfm(image=rgb)["image"])
            if not batch_frames:
                break

            preprocess_finished = time.perf_counter()
            probabilities, effective_batch_size, memory_reduced = infer_batch(
                batch_tensors, batch_size
            )
            attempted_batch_size = min(batch_size, len(batch_tensors))
            if device.type == "cuda" and (
                memory_reduced or effective_batch_size < attempted_batch_size
            ):
                safe_cuda_batch_size = effective_batch_size
                cfg["_cuda_safe_batch_size"] = effective_batch_size
                cfg["_effective_inference_batch_size"] = effective_batch_size
                cfg["_cuda_batch_tuning_complete"] = True
                cuda_tuning_complete = True
            else:
                # A video's final partial batch is not a GPU-memory reduction.
                # Keep reporting and reusing the established full batch size.
                cfg["_effective_inference_batch_size"] = batch_size
                if (
                    device.type == "cuda"
                    and not cuda_tuning_complete
                    and len(batch_frames) == batch_size
                ):
                    probed_size = next_cuda_batch_size(batch_size, requested_batch_size)
                    if probed_size > batch_size:
                        safe_cuda_batch_size = probed_size
                        cfg["_cuda_safe_batch_size"] = probed_size
                        print(
                            f"CUDA batch {batch_size} succeeded; probing batch "
                            f"{probed_size} on the next batch.",
                            flush=True,
                        )
                    else:
                        cfg["_cuda_batch_tuning_complete"] = True
                        cuda_tuning_complete = True
            inference_finished = time.perf_counter()

            if postprocess_pool:
                processed = postprocess_pool.map(
                    postprocess_prediction,
                    batch_frames,
                    probabilities,
                    [threshold] * len(batch_frames),
                    [cfg["refine_iterations"]] * len(batch_frames),
                )
            else:
                processed = (
                    postprocess_prediction(bgr, probability, threshold, cfg["refine_iterations"])
                    for bgr, probability in zip(batch_frames, probabilities)
                )
            # Executor.map preserves frame order; only the optional stateful Kalman step is sequential.
            for source_frame, (mask, fit) in zip(batch_frames, processed):
                fits.append(fit); pixel_counts.append(int(mask.sum()))
                if segmentation_overlay_writer is not None:
                    segmentation_overlay_writer.write(render_segmentation_fit_overlay(source_frame, fit))
                frame_number = len(fits)
            batch_number += 1
            batch_finished = time.perf_counter()
            batch_elapsed = max(batch_finished - batch_started, 1e-9)
            cfg["_last_batch_performance"] = {
                "throughput_fps": round(len(batch_frames) / batch_elapsed, 2),
                "decode_preprocess_seconds": round(preprocess_finished - batch_started, 3),
                "gpu_inference_seconds": round(inference_finished - preprocess_finished, 3),
                "postprocess_seconds": round(batch_finished - inference_finished, 3),
            }
            frame_number = len(fits)
            now = time.monotonic()
            if progress_callback and (
                frame_number == len(batch_frames)
                or now - last_progress_update >= progress_interval
                or (total_frames and frame_number >= total_frames)
            ):
                progress_callback(video_path.name, frame_number, total_frames)
                last_progress_update = now
            if batch_number == 1 or batch_number % 10 == 0:
                print(
                    f"\n{video_path.name}: batch {batch_number}, frames {len(fits)}/{total_label} "
                    f"| inference batch {effective_batch_size} "
                    f"| decode/preprocess {preprocess_finished - batch_started:.1f}s "
                    f"| GPU {inference_finished - preprocess_finished:.1f}s "
                    f"| postprocess {batch_finished - inference_finished:.1f}s"
                )
    finally:
        if postprocess_pool:
            postprocess_pool.shutdown(wait=True)
        cap.release()
        if segmentation_overlay_writer is not None:
            segmentation_overlay_writer.release()
    if not fits:
        raise ValueError(f"No readable frames found in: {video_path}")
    print(f"Segmentation complete for {video_path.name}.", flush=True)
    tracking = track_centers(fits, fps, cfg)
    centers = tracking.centers
    rejected_count = tracking.states.count("innovation_rejected")
    if rejected_count:
        print(
            f"Innovation gate rejected {rejected_count} center measurement(s) in "
            f"{video_path.name}; rejected frames remain NaN in the scientific trace.",
            flush=True,
        )
    diameters = [(a + b) / 2 for _, _, a, b, _, kind in fits if kind == "ellipsefit" and np.isfinite(a) and np.isfinite(b)]
    radius = max(1, int(round((np.mean(diameters) if diameters else 40) / 2)))
    final_runtime_settings = (
        runtime_settings_callback() or {}
        if runtime_settings_callback
        else {}
    )
    benchmark_combinations = cfg.get("kalman_benchmark", [])
    if benchmark_combinations:
        benchmark_configurations(cfg)  # Validate the submitted benchmark once.
        benchmark_root = output / "kalman_benchmark"
        results: dict[str, Path] = {}
        settings_rows = []
        used_slugs: set[str] = set()
        benchmark_video_specs = []
        for combination in benchmark_configurations(cfg):
            name = str(combination.get("name", "Configuration")).strip() or "Configuration"
            slug = benchmark_slug(name)
            suffix = 2
            original_slug = slug
            while slug in used_slugs:
                slug = f"{original_slug}_{suffix}"
                suffix += 1
            used_slugs.add(slug)
            variant_cfg = dict(cfg)
            variant_cfg["tracking_filter_mode"] = combination.get(
                "tracking_filter_mode", "kalman"
            )
            if variant_cfg["tracking_filter_mode"] == "kalman":
                for key in ("kalman_q_pos", "kalman_q_vel", "kalman_r"):
                    if key not in combination:
                        raise ValueError(f"Benchmark configuration '{name}' is missing {key}.")
                    variant_cfg[key] = float(combination[key])
            variant_tracking = track_centers(fits, fps, variant_cfg)
            variant_centers = variant_tracking.centers
            trace = trace_from_tracking(
                fits,
                variant_centers,
                pixel_counts,
                fps,
                variant_cfg,
                benchmark_name=name,
                tracking_states=variant_tracking.states,
                innovation_distance_px=variant_tracking.innovation_distance_px,
                innovation_mahalanobis_d2=variant_tracking.innovation_mahalanobis_d2,
            )
            variant_root = benchmark_root / slug
            trace_dir = variant_root / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{video_path.stem}_stimtrace_tracking.csv"
            trace.to_csv(trace_path, index=False)
            results[slug] = trace_path
            settings_rows.append({
                "name": name,
                "folder": slug,
                "tracking_filter_mode": variant_cfg["tracking_filter_mode"],
                "kalman_q_pos_px2": (
                    variant_cfg["kalman_q_pos"]
                    if variant_cfg["tracking_filter_mode"] == "kalman" else np.nan
                ),
                "kalman_q_vel_px2_per_s2": (
                    variant_cfg["kalman_q_vel"]
                    if variant_cfg["tracking_filter_mode"] == "kalman" else np.nan
                ),
                "kalman_r_px2": (
                    variant_cfg["kalman_r"]
                    if variant_cfg["tracking_filter_mode"] == "kalman" else np.nan
                ),
                "kalman_innovation_gate_enabled": (
                    variant_tracking.innovation_threshold_d2 is not None
                ),
                "kalman_innovation_gate_confidence": (
                    KALMAN_INNOVATION_GATE_CONFIDENCE
                    if variant_tracking.innovation_threshold_d2 is not None else np.nan
                ),
                "kalman_innovation_gate_threshold_d2": (
                    variant_tracking.innovation_threshold_d2
                ),
                "kalman_innovation_gate_min_radius_px": (
                    KALMAN_INNOVATION_GATE_MIN_RADIUS_PX
                    if variant_tracking.innovation_threshold_d2 is not None else np.nan
                ),
            })
            if final_runtime_settings.get(
                "generate_overlays",
                cfg.get("generate_overlays", True),
            ):
                benchmark_video_specs.append({
                    "destination": (
                        variant_root
                        / "videos"
                        / f"{video_path.stem}_tracked_trace.avi"
                    ),
                    "centers": variant_centers,
                    "trace": trace,
                    "title": name,
                    "cfg": variant_cfg,
                })
        if benchmark_video_specs:
            create_benchmark_videos(
                video_path,
                fits,
                fps,
                radius,
                benchmark_video_specs,
                cancel_check=cancel_check,
                stage_callback=stage_callback,
            )
        settings_path = benchmark_root / "kalman_benchmark_settings.csv"
        settings = pd.DataFrame(settings_rows).drop_duplicates(subset="folder")
        if settings_path.exists():
            existing = pd.read_csv(settings_path)
            settings = pd.concat([existing, settings], ignore_index=True).drop_duplicates(
                subset="folder",
                keep="last",
            )
        settings.to_csv(settings_path, index=False)
        return results
    generate_overlay = final_runtime_settings.get(
        "generate_overlays",
        cfg.get("generate_overlays", True),
    )
    trace = trace_from_tracking(
        fits,
        centers,
        pixel_counts,
        fps,
        cfg,
        tracking_states=tracking.states,
        innovation_distance_px=tracking.innovation_distance_px,
        innovation_mahalanobis_d2=tracking.innovation_mahalanobis_d2,
    )
    print(f"Writing displacement trace for {video_path.name}...", flush=True)
    trace_path = output / "traces"; trace_path.mkdir(exist_ok=True); result = trace_path / f"{video_path.stem}_stimtrace_tracking.csv"; trace.to_csv(result, index=False)
    if generate_overlay and defer_overlay:
        return DeferredOverlayResult(
            trace_path=result,
            video_path=video_path,
            output=output,
            centers=list(centers),
            radius=radius,
            fps=fps,
            frame_size=(frame_width, frame_height),
        )
    if generate_overlay:
        create_overlay_video(
            video_path,
            output,
            centers,
            radius,
            fps,
            (frame_width, frame_height),
            cancel_check=cancel_check,
            stage_callback=stage_callback,
        )
    return result


def recording_name_from_tracking_file(path: Path) -> str:
    """Return the recording name for current and legacy detailed trace files."""
    stem = path.stem
    for suffix in ("_stimtrace_tracking", "_pillar_displacement"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem


def create_master(traces: list[Path], output: Path, cfg: dict) -> None:
    print("Creating combined force CSV...", flush=True)
    all_data: dict[str, pd.Series] = {}
    for trace in traces:
        data = pd.read_csv(trace)
        force = (
            pd.to_numeric(data["Force_uN"], errors="coerce")
            if "Force_uN" in data.columns
            else (
                pd.to_numeric(data["XY_combo_norm"], errors="coerce")
                * cfg["pixel_to_um"]
                * cfg["force_slope_un_per_um"]
            )
        )
        name = recording_name_from_tracking_file(trace)
        all_data[f"{name}_Force_uN"] = indexed_signal(data["time_s"], force)
    master = combine_signals_by_time(all_data)
    master.to_csv(output / "stimtrace_force_traces.csv", index=False)


def create_benchmark_master(
    traces_by_configuration: dict[str, list[Path]],
    output: Path,
    cfg: dict,
) -> Path:
    """Create one CSV containing every recording/configuration trace."""
    print("Creating all-configuration Kalman benchmark CSV...", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    all_data: dict[str, pd.Series] = {}
    configured_names = {}
    used_folders: set[str] = set()
    for combination in benchmark_configurations(cfg):
        folder = benchmark_slug(combination.get("name", "Configuration"))
        suffix = 2
        original_folder = folder
        while folder in used_folders:
            folder = f"{original_folder}_{suffix}"
            suffix += 1
        used_folders.add(folder)
        configured_names[folder] = combination
    for folder_name, traces in traces_by_configuration.items():
        combination = configured_names.get(folder_name, {})
        display_name = str(combination.get("name", folder_name))
        for trace_path in traces:
            data = pd.read_csv(trace_path)
            force = (
                pd.to_numeric(data["Force_uN"], errors="coerce")
                if "Force_uN" in data.columns
                else (
                    pd.to_numeric(data["XY_combo_norm"], errors="coerce")
                    * cfg["pixel_to_um"]
                    * cfg["force_slope_un_per_um"]
                )
            )
            recording = recording_name_from_tracking_file(trace_path)
            # Signal Analysis interprets the left side as condition and the
            # right side as specimen, enabling direct configuration comparisons.
            column_name = f"{display_name} | {recording}_Force_uN"
            suffix = 2
            original_name = column_name
            while column_name in all_data:
                column_name = f"{original_name} ({suffix})"
                suffix += 1
            all_data[column_name] = indexed_signal(data["time_s"], force)
    if not all_data:
        raise ValueError("No Kalman benchmark traces were available for the combined CSV.")
    master = combine_signals_by_time(all_data)
    csv_destination = output / "kalman_benchmark_all_traces.csv"
    master.to_csv(csv_destination, index=False)
    return csv_destination


def load_model(model_path: Path, device):
    model = MultiTaskUnet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # The training notebook writes a raw state dictionary; also accept standard
    # checkpoint wrappers so future training exports remain compatible.
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
    model.eval()
    return model


def process_job(
    job: Path,
    model,
    device,
    progress_callback=None,
    cancel_check=None,
    cuda_batch_limit: int | None = None,
    checkpoint_callback=None,
    existing_trace_paths: list[Path] | None = None,
) -> int:
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8")); cfg = manifest["settings"]
    if cuda_batch_limit:
        # Start conservatively, then probe larger batches until the requested
        # ceiling or the runtime's actual CUDA memory limit is reached.
        cfg["_cuda_initial_batch_size"] = max(1, int(cuda_batch_limit))
    requested_batch_size = max(1, int(cfg.get("inference_batch_size", 8)))
    effective_cpu_workers = min(
        max(1, int(cfg.get("cpu_postprocess_workers", 1))),
        requested_batch_size,
        max(1, (os.cpu_count() or 1) - 1),
    )
    print(
        f"Runtime settings: batch {requested_batch_size}, "
        f"{effective_cpu_workers} CPU postprocess worker(s).",
        flush=True,
    )
    output = job / "results"; output.mkdir(exist_ok=True)
    videos = [job / name for name in manifest["videos"]]
    frame_totals = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        frame_totals.append(max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        cap.release()
    total_job_frames = sum(frame_totals)
    completed_frames = 0
    job_started = time.monotonic()
    benchmark_active = bool(cfg.get("kalman_benchmark"))
    generate_overlays = bool(cfg.get("generate_overlays", True))
    if benchmark_active:
        segmentation_weight, output_weight = 0.70, 0.28
    elif generate_overlays:
        segmentation_weight, output_weight = 0.90, 0.08
    else:
        segmentation_weight, output_weight = 0.98, 0.0
    progress_interval = max(2.0, float(cfg.get("progress_interval_seconds", 5.0)))
    # Resume jobs may already have per-video traces restored from the Drive
    # checkpoint archive. Include them when rebuilding the combined outputs.
    traces = list(existing_trace_paths or [])
    benchmark_traces: dict[str, list[Path]] = {}
    skipped_videos: list[dict[str, str]] = []
    if not videos:
        if not traces:
            raise ValueError("No unfinished videos or valid checkpoint traces were available.")
        print("All recordings were restored from checkpoints; rebuilding combined results.", flush=True)
        create_master(traces, output, cfg)
        (job / "status.json").write_text(
            json.dumps({"state": "complete", "job_id": manifest["job_id"]}),
            encoding="utf-8",
        )
        return 0
    # Colab kernels can terminate without a Python traceback when OpenCV video
    # encoding runs concurrently with CUDA inference. Keep the cloud worker's
    # native-video path serial until it can be isolated in a separate process.
    pipeline_overlays = False
    overlay_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="stimtrace-overlay")
        if pipeline_overlays
        else None
    )
    overlay_futures = []
    segmentation_running = threading.Event()
    if pipeline_overlays:
        print(
            "GPU overlay pipeline enabled: CPU encoding will overlap segmentation "
            "from the second video onward.",
            flush=True,
        )
    for index, video in enumerate(videos, start=1):
        if cancel_check and cancel_check():
            raise JobCancelled("Cancelled by the user.")
        print(f"\nVideo {index}/{len(videos)}: {video.name}", flush=True)
        file_total = frame_totals[index - 1]
        last_stage_progress = 0.0

        def emit_progress(stage: str, current: int, stage_total: int) -> None:
            denominator = max(1, stage_total or file_total)
            stage_fraction = min(1.0, current / denominator)
            if stage == "segmenting":
                file_fraction = segmentation_weight * stage_fraction
                processed_frames = completed_frames + min(current, file_total)
            else:
                file_fraction = segmentation_weight + output_weight * stage_fraction
                # Frame progress measures segmentation only. Rendering is reported as a
                # separate stage and must not make the submitted-frame count ambiguous.
                processed_frames = completed_frames + file_total
            job_fraction = min(
                0.98,
                (completed_frames + file_fraction * file_total) / max(1, total_job_frames),
            )
            elapsed = time.monotonic() - job_started
            eta_seconds = elapsed * (1.0 - job_fraction) / job_fraction if job_fraction > 0 else None
            if progress_callback:
                progress_callback(video.name, current, denominator, {
                    "stage": stage,
                    "current_file_index": index,
                    "file_count": len(videos),
                    "current_frame": current,
                    "current_file_frames": denominator,
                    "processed_frames": processed_frames,
                    "total_job_frames": total_job_frames,
                    "file_progress_fraction": file_fraction,
                    "job_progress_fraction": job_fraction,
                    "estimated_remaining_seconds": round(eta_seconds) if eta_seconds is not None else None,
                    "inference_batch_size": int(
                        cfg.get("_effective_inference_batch_size", requested_batch_size)
                    ),
                    "requested_inference_batch_size": requested_batch_size,
                    "cpu_postprocess_workers": effective_cpu_workers,
                    **cfg.get("_last_batch_performance", {}),
                })

        def report_progress(video_name, frame_number, total_frames):
            emit_progress("segmenting", frame_number, total_frames)

        def report_stage(video_name, stage, current, stage_total):
            nonlocal last_stage_progress
            now = time.monotonic()
            if current < stage_total and now - last_stage_progress < progress_interval:
                return
            emit_progress(stage, current, stage_total)
            last_stage_progress = now

        try:
            segmentation_running.set()
            result = process_video(
                video,
                output,
                model,
                device,
                cfg,
                report_progress if progress_callback else None,
                stage_callback=report_stage if progress_callback else None,
                cancel_check=cancel_check,
                defer_overlay=pipeline_overlays,
            )
        except InterruptedError as error:
            if overlay_executor:
                overlay_executor.shutdown(wait=True, cancel_futures=True)
            raise JobCancelled("Cancelled by the user.") from error
        except PillarCenterTrackingError as error:
            # A missing/ambiguous pillar must never become a Kalman-predicted
            # scientific trace. Preserve the reason, count the recording as
            # complete for progress, and continue with the remaining videos.
            reason = str(error)
            skipped_videos.append({"video": video.name, "reason": reason})
            print(f"Skipping {video.name}: {reason}", flush=True)
            completed_frames += file_total
            if progress_callback:
                emit_progress("segmenting", file_total, file_total)
            continue
        except BaseException:
            if overlay_executor:
                overlay_executor.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            segmentation_running.clear()
        if isinstance(result, DeferredOverlayResult):
            traces.append(result.trace_path)

            def overlay_progress(
                video_name,
                stage,
                current,
                stage_total,
                *,
                file_index=index,
                file_total=file_total,
            ):
                # While GPU inference is active, keep the primary UI focused on
                # segmentation. If encoding remains afterward, expose its progress.
                if segmentation_running.is_set() or not progress_callback:
                    return
                progress_callback(video_name, current, max(1, stage_total), {
                    "stage": stage,
                    "current_file_index": file_index,
                    "file_count": len(videos),
                    "current_frame": current,
                    "current_file_frames": max(1, stage_total),
                    "processed_frames": completed_frames + file_total,
                    "total_job_frames": total_job_frames,
                    "file_progress_fraction": segmentation_weight + output_weight * min(
                        1.0, current / max(1, stage_total)
                    ),
                    "job_progress_fraction": min(
                        0.98, (completed_frames + file_total) / max(1, total_job_frames)
                    ),
                    "estimated_remaining_seconds": None,
                    "inference_batch_size": int(
                        cfg.get("_effective_inference_batch_size", requested_batch_size)
                    ),
                    "requested_inference_batch_size": requested_batch_size,
                    "cpu_postprocess_workers": effective_cpu_workers,
                    **cfg.get("_last_batch_performance", {}),
                })

            overlay_futures.append(overlay_executor.submit(
                create_overlay_video,
                result.video_path,
                result.output,
                result.centers,
                result.radius,
                result.fps,
                result.frame_size,
                cancel_check=cancel_check,
                stage_callback=overlay_progress,
            ))
        elif isinstance(result, dict):
            for name, trace in result.items():
                benchmark_traces.setdefault(name, []).append(trace)
        else:
            traces.append(result)
        if checkpoint_callback:
            checkpoint_callback(video.name)
        completed_frames += file_total
    if overlay_executor:
        try:
            for future in overlay_futures:
                future.result()
        except InterruptedError as error:
            raise JobCancelled("Cancelled by the user.") from error
        finally:
            overlay_executor.shutdown(wait=True, cancel_futures=True)
    if progress_callback:
        progress_callback(videos[-1].name, 1, 1, {
            "stage": "workbook",
            "current_file_index": len(videos),
            "file_count": len(videos),
            "current_frame": 1,
            "current_file_frames": 1,
            "processed_frames": total_job_frames,
            "total_job_frames": total_job_frames,
            "file_progress_fraction": 1.0,
            "job_progress_fraction": 0.98,
            "estimated_remaining_seconds": None,
            "inference_batch_size": int(
                cfg.get("_effective_inference_batch_size", requested_batch_size)
            ),
            "requested_inference_batch_size": requested_batch_size,
            "cpu_postprocess_workers": effective_cpu_workers,
            **cfg.get("_last_batch_performance", {}),
        })
    if skipped_videos:
        (output / "skipped_videos.json").write_text(
            json.dumps(skipped_videos, indent=2), encoding="utf-8"
        )
    if benchmark_traces:
        for name, variant_traces in benchmark_traces.items():
            if cancel_check and cancel_check():
                raise JobCancelled("Cancelled by the user.")
            create_master(variant_traces, output / "kalman_benchmark" / name, cfg)
        if cancel_check and cancel_check():
            raise JobCancelled("Cancelled by the user.")
        create_benchmark_master(
            benchmark_traces,
            output / "kalman_benchmark",
            cfg,
        )
    elif traces:
        if cancel_check and cancel_check():
            raise JobCancelled("Cancelled by the user.")
        create_master(traces, output, cfg)
    else:
        raise PillarCenterTrackingError(
            "No video had enough unambiguous measured pillar centers to create a trace. "
            "See skipped_videos.json in the results folder."
        )
    if cancel_check and cancel_check():
        raise JobCancelled("Cancelled by the user.")
    print("Job processing complete. Preparing results for upload...", flush=True)
    (job / "status.json").write_text(json.dumps({"state": "complete", "job_id": manifest["job_id"]}), encoding="utf-8")
    return total_job_frames


def _drive_children(service, parent_id: str) -> list[dict]:
    files = []
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{parent_id}' in parents and trashed = false",
            fields="nextPageToken,files(id,name,mimeType,modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return files


def _drive_has_named_child(service, parent_id: str, name: str) -> bool:
    escaped_name = name.replace("'", "\\'")
    result = service.files().list(
        q=f"'{parent_id}' in parents and name = '{escaped_name}' and trashed = false",
        fields="files(id)",
        pageSize=1,
    ).execute()
    return bool(result.get("files"))


def _drive_download(service, file_id: str, target: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        downloader = MediaIoBaseDownload(output, service.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = downloader.next_chunk()

def _drive_file_bytes(service, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, service.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def _drive_upload_bytes(service, parent_id: str, name: str, data: bytes, mime_type: str) -> str:
    from googleapiclient.http import MediaInMemoryUpload

    result = service.files().create(
        body={"name": name, "parents": [parent_id]},
        media_body=MediaInMemoryUpload(data, mimetype=mime_type),
        fields="id",
    ).execute()
    return result["id"]


def _drive_update_json(service, file_id: str, payload: dict) -> None:
    from googleapiclient.http import MediaInMemoryUpload

    data = json.dumps(payload).encode("utf-8")
    service.files().update(
        fileId=file_id,
        media_body=MediaInMemoryUpload(data, mimetype="application/json"),
    ).execute()


def _drive_update_file(
    service,
    file_id: str,
    path: Path,
    mime_type: str,
    cancel_check=None,
) -> None:
    from googleapiclient.http import MediaFileUpload

    request = service.files().update(
        fileId=file_id,
        media_body=MediaFileUpload(str(path), mimetype=mime_type, resumable=True),
    )
    response = None
    while response is None:
        if cancel_check and cancel_check():
            raise JobCancelled("Cancelled by the user.")
        _, response = request.next_chunk()


def _zip_results(source: Path, destination: Path, cancel_check=None) -> None:
    with zipfile.ZipFile(destination, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                if cancel_check and cancel_check():
                    raise JobCancelled("Cancelled by the user.")
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(
                    path,
                    path.relative_to(source),
                    compress_type=compression,
                )


class MaskTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], augment: bool):
        self.pairs = pairs
        operations = [A.Resize(512, 512)]
        if augment:
            operations.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        operations.extend([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
        self.transform = A.Compose(operations)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image_bgr = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask is None:
            raise ValueError(f"Could not read training pair: {image_path.name}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"].float()
        mask_tensor = (transformed["mask"].float() > 127).unsqueeze(0).float()
        return image_tensor, mask_tensor


def training_pairs_from_dataset(dataset_root: Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], dict]:
    """Load a phase-aware dataset with recording-level development/QC splits."""
    frames_dir = dataset_root / "frames"
    masks_dir = dataset_root / "masks"
    annotations_dir = dataset_root / "annotations"
    pairs = [
        (image, masks_dir / f"{image.stem}.png")
        for image in sorted(frames_dir.iterdir())
        if image.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and (masks_dir / f"{image.stem}.png").exists()
        and (annotations_dir / f"{image.stem}.json").exists()
    ]
    if len(pairs) < 2:
        raise ValueError("Training requires at least two image/mask pairs.")
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            "Training dataset is missing the phase-aware split manifest. "
            "Create a new annotation project in StimTrace."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection_strategy") not in {"phase_aware_v1", "migrated_existing_v1"}:
        raise ValueError("Unsupported training split manifest.")
    assignments = {str(item.get("image", "")): item for item in manifest.get("frames", [])}
    training_pairs: list[tuple[Path, Path]] = []
    validation_pairs: list[tuple[Path, Path]] = []
    source_splits: dict[str, set[str]] = {}
    for pair in pairs:
        item = assignments.get(pair[0].name)
        if not item:
            raise ValueError(f"Annotated frame {pair[0].name} is missing from the split manifest.")
        source = str(item.get("source_video_id", ""))
        split = str(item.get("split", ""))
        if not source or split not in {"training", "validation"}:
            raise ValueError(f"Invalid split assignment for {pair[0].name}.")
        source_splits.setdefault(source, set()).add(split)
        (training_pairs if split == "training" else validation_pairs).append(pair)
    if any(len(splits) > 1 for splits in source_splits.values()):
        raise ValueError("Training split is invalid: a source recording appears in both sets.")
    if len(training_pairs) < 2 or not validation_pairs:
        raise ValueError("Training requires at least two training pairs and one validation pair.")
    return training_pairs, validation_pairs, manifest


def train_custom_model(
    dataset_zip: Path,
    output_model: Path,
    base_model,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    epoch_callback=None,
) -> pd.DataFrame:
    dataset_root = dataset_zip.parent / "dataset"
    dataset_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(dataset_zip) as archive:
        root = dataset_root.resolve()
        for member in archive.infolist():
            destination = (dataset_root / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(
                    f"Training archive contains an unsafe path: {member.filename}"
                )
        archive.extractall(dataset_root)
    training_pairs, validation_pairs, dataset_manifest = training_pairs_from_dataset(dataset_root)
    training_loader = torch.utils.data.DataLoader(
        MaskTrainingDataset(training_pairs, augment=True),
        batch_size=min(batch_size, len(training_pairs)),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = torch.utils.data.DataLoader(
        MaskTrainingDataset(validation_pairs, augment=False),
        batch_size=min(batch_size, len(validation_pairs)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    trained_model = MultiTaskUnet().to(device)
    trained_model.load_state_dict(base_model.state_dict())
    for parameter in trained_model.center_head.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        (parameter for parameter in trained_model.unet.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    bce = nn.BCEWithLogitsLoss()

    def loss_for(logits, targets):
        bce_loss = bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce_loss + dice_loss

    def evaluate(loader):
        losses = []
        dice_scores = []
        iou_scores = []
        with torch.inference_mode():
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = trained_model.unet(images)
                losses.append(float(loss_for(logits, masks).cpu()))
                predictions = (torch.sigmoid(logits) >= 0.5).float()
                intersection = (predictions * masks).sum(dim=(1, 2, 3))
                union = predictions.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
                dice_scores.extend(((2.0 * intersection + 1.0) /
                                    (predictions.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1.0)).cpu().tolist())
                iou_scores.extend(((intersection + 1.0) / (union + 1.0)).cpu().tolist())
        return float(np.mean(losses)), float(np.mean(dice_scores)), float(np.mean(iou_scores))

    history = []
    best_validation_dice = -float("inf")
    best_epoch = 0
    best_model = output_model.with_suffix(".best.pth")
    for epoch in range(1, epochs + 1):
        trained_model.train()
        training_losses = []
        for images, masks in training_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(trained_model.unet(images), masks)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        trained_model.eval()
        validation_loss, validation_dice, validation_iou = evaluate(validation_loader)
        row = {
            "epoch": epoch,
            "training_loss": float(np.mean(training_losses)),
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
            "validation_iou": validation_iou,
        }
        history.append(row)
        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            best_epoch = epoch
            torch.save(trained_model.state_dict(), best_model)
        print(
            f"Training epoch {epoch}/{epochs}: loss {row['training_loss']:.4f}, "
            f"validation {row['validation_loss']:.4f}, Dice {validation_dice:.4f}, IoU {validation_iou:.4f}",
            flush=True,
        )
        if epoch_callback:
            epoch_callback(epoch, epochs, row)
    if best_model.exists():
        os.replace(best_model, output_model)
    else:
        torch.save(trained_model.state_dict(), output_model)
    del trained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = pd.DataFrame(history)
    result.attrs["dataset_manifest"] = dataset_manifest
    result.attrs["best_epoch"] = best_epoch
    return result


def write_training_qc_report(history: pd.DataFrame, dataset_manifest: dict, output_dir: Path) -> list[Path]:
    """Create portable training/QC summaries for review in Drive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    frames = list(dataset_manifest.get("frames", []))
    summary = dict(dataset_manifest.get("summary", {}))
    summary.setdefault("selection_strategy", dataset_manifest.get("selection_strategy", "phase_aware_v1"))
    summary.setdefault("best_epoch", int(history.attrs.get("best_epoch", len(history))))
    if not history.empty:
        best_row = history.loc[history["validation_dice"].idxmax()] if "validation_dice" in history else history.iloc[-1]
        summary["best_validation_dice"] = float(best_row.get("validation_dice", float("nan")))
        summary["best_validation_iou"] = float(best_row.get("validation_iou", float("nan")))
        summary["best_validation_loss"] = float(best_row.get("validation_loss", float("nan")))
    summary_path = output_dir / "training_qc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs.append(summary_path)
    try:
        import matplotlib.pyplot as plt

        history_plot = output_dir / "training_history.png"
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(history["epoch"], history["training_loss"], label="Training loss")
        axes[0].plot(history["epoch"], history["validation_loss"], label="Validation loss")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training history")
        axes[0].legend()
        if "validation_dice" in history:
            axes[1].plot(history["epoch"], history["validation_dice"], label="Dice")
            axes[1].plot(history["epoch"], history["validation_iou"], label="IoU")
        axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation segmentation quality", ylim=(0, 1))
        axes[1].legend()
        figure.savefig(history_plot, dpi=160)
        plt.close(figure)
        outputs.append(history_plot)

        if frames:
            phase_plot = output_dir / "dataset_phase_qc.png"
            phases = sorted({str(frame.get("phase", "unclassified")) for frame in frames})
            splits = ("training", "validation")
            figure, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
            positions = np.arange(len(phases))
            width = 0.38
            for offset, split in [(-width / 2, splits[0]), (width / 2, splits[1])]:
                counts = [sum(
                    str(frame.get("phase", "unclassified")) == phase and frame.get("split") == split
                    for frame in frames
                ) for phase in phases]
                axis.bar(positions + offset, counts, width, label=split.title())
            axis.set_xticks(positions, phases, rotation=20, ha="right")
            axis.set_ylabel("Annotated frames")
            axis.set_title("Phase and split coverage")
            axis.legend()
            figure.savefig(phase_plot, dpi=160)
            plt.close(figure)
            outputs.append(phase_plot)
    except Exception as error:
        print(f"Could not create optional training QC plots: {error}", flush=True)

    report_path = output_dir / "training_report.html"
    image_tags = "".join(
        f'<figure><img src="{html.escape(path.name)}" style="max-width: 100%;"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
        for path in outputs if path.suffix.lower() == ".png"
    )
    summary_rows = "".join(
        f"<tr><th>{html.escape(str(key).replace('_', ' ').title())}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    report_path.write_text(
        "<html><head><title>StimTrace training report</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1000px;margin:32px auto;}"
        "table{border-collapse:collapse;}th,td{border:1px solid #bbb;padding:7px;text-align:left;}"
        "figure{margin:24px 0;}figcaption{color:#555;}</style></head><body>"
        "<h1>StimTrace training report</h1>"
        "<p>The validation set contains complete recordings held out before frame selection. "
        "A frame from one recording is never used in both training and validation.</p>"
        f"<table>{summary_rows}</table>{image_tags}</body></html>",
        encoding="utf-8",
    )
    outputs.append(report_path)
    return outputs


def _process_drive_training_job(service, job: dict, by_name: dict, base_model, device) -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="stimtrace_training_"))
    progress_file_id = None
    processing_started_at = datetime.now(timezone.utc).isoformat()
    processing_started_monotonic = time.monotonic()

    def terminal_timing() -> dict:
        return {
            "processing_started_at": processing_started_at,
            "processing_finished_at": datetime.now(timezone.utc).isoformat(),
            "actual_total_seconds": max(0.0, time.monotonic() - processing_started_monotonic),
        }

    try:
        _drive_download(
            service,
            by_name["training_manifest.json"]["id"],
            temp_root / "training_manifest.json",
        )
        manifest = json.loads((temp_root / "training_manifest.json").read_text(encoding="utf-8"))
        progress_file_id = manifest.get("progress_file_id")
        if "cancel.json" in by_name:
            raise JobCancelled("Cancelled by the user.")
        _drive_download(
            service,
            manifest["dataset_file_id"],
            temp_root / "training_dataset.zip",
        )
        if progress_file_id:
            _drive_update_json(service, progress_file_id, {
                "state": "training",
                "message": "Preparing the annotated dataset.",
                "progress_fraction": 0.0,
                "processing_started_at": processing_started_at,
            })
        model_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in manifest.get("model_name", "StimTrace_custom_model")
        ).strip("_") or "StimTrace_custom_model"
        output_model = temp_root / f"{model_name}.pth"

        def publish_epoch(epoch, epochs, metrics):
            if _drive_has_named_child(service, job["id"], "cancel.json"):
                raise JobCancelled("Cancelled by the user.")
            if progress_file_id:
                _drive_update_json(service, progress_file_id, {
                    "state": "training",
                    "message": (
                        f"Epoch {epoch}/{epochs}: training loss "
                        f"{metrics['training_loss']:.4f}, validation loss "
                        f"{metrics['validation_loss']:.4f}, Dice "
                        f"{metrics.get('validation_dice', 0):.4f}."
                    ),
                    "current_epoch": epoch,
                    "total_epochs": epochs,
                    "progress_fraction": epoch / epochs * 0.95,
                })

        history = train_custom_model(
            temp_root / "training_dataset.zip",
            output_model,
            base_model,
            device,
            max(1, int(manifest.get("epochs", 30))),
            max(1, int(manifest.get("batch_size", 4))),
            float(manifest.get("learning_rate", 0.0001)),
            publish_epoch,
        )
        history_path = temp_root / "training_history.csv"
        history.to_csv(history_path, index=False)
        qc_outputs = write_training_qc_report(
            history,
            dict(history.attrs.get("dataset_manifest", {})),
            temp_root / "training_qc",
        )
        from googleapiclient.http import MediaFileUpload

        model_output_file_id = manifest.get("model_output_file_id", "")
        if model_output_file_id:
            _drive_update_file(
                service,
                model_output_file_id,
                output_model,
                "application/octet-stream",
            )
            uploaded_model = {"id": model_output_file_id}
        else:
            uploaded_model = service.files().create(
                body={"name": output_model.name, "parents": [job["id"]]},
                media_body=MediaFileUpload(
                    str(output_model),
                    mimetype="application/octet-stream",
                    resumable=True,
                ),
                fields="id",
            ).execute()
        service.files().create(
            body={"name": history_path.name, "parents": [job["id"]]},
            media_body=MediaFileUpload(str(history_path), mimetype="text/csv"),
            fields="id",
        ).execute()
        for qc_output in qc_outputs:
            mime_type = {
                ".csv": "text/csv",
                ".json": "application/json",
                ".html": "text/html",
                ".png": "image/png",
            }.get(qc_output.suffix.lower(), "application/octet-stream")
            service.files().create(
                body={"name": qc_output.name, "parents": [job["id"]]},
                media_body=MediaFileUpload(str(qc_output), mimetype=mime_type),
                fields="id",
            ).execute()
        _drive_upload_bytes(
            service,
            job["id"],
            "status.json",
            json.dumps({"state": "complete", "model_name": output_model.name}).encode("utf-8"),
            "application/json",
        )
        if progress_file_id:
            _drive_update_json(service, progress_file_id, {
                "state": "complete",
                "message": f"Training complete. Saved {output_model.name}.",
                "model_name": output_model.name,
                "model_file_id": uploaded_model["id"],
                "progress_fraction": 1.0,
                **terminal_timing(),
            })
        print(f"Completed training job: {manifest['job_id']}", flush=True)
    except JobCancelled as error:
        _drive_upload_bytes(
            service,
            job["id"],
            "status.json",
            json.dumps({"state": "cancelled", "message": str(error)}).encode("utf-8"),
            "application/json",
        )
        if progress_file_id:
            _drive_update_json(service, progress_file_id, {
                "state": "cancelled",
                "message": "Training cancelled.",
                **terminal_timing(),
            })
    except Exception as error:
        _drive_upload_bytes(
            service,
            job["id"],
            "status.json",
            json.dumps({"state": "failed", "error": str(error)}).encode("utf-8"),
            "application/json",
        )
        if progress_file_id:
            _drive_update_json(service, progress_file_id, {
                "state": "failed",
                "message": "Training failed.",
                "error": str(error),
                **terminal_timing(),
            })
        print(f"Training job {job['name']} failed: {error}", flush=True)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def run_drive_worker(
    root_folder_id: str,
    model_file_id: str,
    worker_status_file_id: str = "",
    poll_seconds: int = 20,
) -> None:
    """Process Drive jobs by file ID, without relying on a mounted Drive path."""
    from google.auth import default
    from googleapiclient.discovery import build

    credentials, _ = default()
    service = build("drive", "v3", credentials=credentials)
    # Verify immediately that Colab authorized the same account as StimTrace.
    service.files().get(fileId=root_folder_id, fields="id,name").execute()

    model_path = Path(f"/content/stimtrace_model_{model_file_id}.pth")
    if not model_path.exists() or model_path.stat().st_size < 1_000_000:
        print("Downloading the approved StimTrace model...")
        _drive_download(service, model_file_id, model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Stable algorithm selection uses less transient workspace than benchmarking
        # several cuDNN algorithms for user-selected batch sizes.
        torch.backends.cudnn.benchmark = False
    print("Loading the StimTrace segmentation model...")
    model = load_model(model_path, device)
    hardware = runtime_hardware(device)
    cloud_cuda_batch_limit = None
    if device.type == "cuda":
        gpu_memory_gb = float(hardware.get("gpu_memory_gb", 0) or 0)
        # This is only the first probe, not a hard cap. Successful batches ramp
        # upward, while a recoverable CUDA OOM establishes the safe ceiling.
        cloud_cuda_batch_limit = (
            16 if gpu_memory_gb <= 16
            else 24 if gpu_memory_gb <= 24
            else 32
        )
    print(
        "StimTrace worker running on "
        f"{hardware['device_name']} with {hardware.get('gpu_memory_gb', 0)} GB GPU memory, "
        f"{hardware.get('system_memory_gb', 0)} GB system memory, and "
        f"{hardware['cpu_threads']} logical CPU threads. Watching the personal Drive queue."
    )

    worker_activity_lock = threading.Lock()
    worker_activity = {
        "current_job_id": "",
        "current_job_folder_id": "",
        "current_progress_file_id": "",
        "current_job_type": "",
    }

    def set_worker_activity(
        job_id: str = "",
        folder_id: str = "",
        progress_file_id: str = "",
        job_type: str = "",
    ) -> None:
        with worker_activity_lock:
            worker_activity.update({
                "current_job_id": job_id,
                "current_job_folder_id": folder_id,
                "current_progress_file_id": progress_file_id,
                "current_job_type": job_type,
            })

    def publish_heartbeat() -> None:
        heartbeat_service = build("drive", "v3", credentials=credentials)
        while True:
            try:
                with worker_activity_lock:
                    activity = dict(worker_activity)
                _drive_update_json(heartbeat_service, worker_status_file_id, {
                    "state": "online",
                    "worker_version": WORKER_VERSION,
                    **runtime_hardware(device),
                    **activity,
                    "updated_at": time.time(),
                })
            except Exception as heartbeat_error:
                print(f"Worker heartbeat update skipped: {heartbeat_error}", flush=True)
            time.sleep(5)

    if worker_status_file_id:
        threading.Thread(
            target=publish_heartbeat,
        name="stimtrace-worker-heartbeat",
            daemon=True,
        ).start()

    while True:
        folders = [
            item for item in _drive_children(service, root_folder_id)
            if item["mimeType"] == "application/vnd.google-apps.folder"
        ]
        for job in folders:
            children = _drive_children(service, job["id"])
            by_name = {item["name"]: item for item in children}
            manifest_name = (
                "manifest.json" if "manifest.json" in by_name
                else "training_manifest.json" if "training_manifest.json" in by_name
                else ""
            )
            if not manifest_name:
                continue
            try:
                existing_manifest = json.loads(
                    _drive_file_bytes(service, by_name[manifest_name]["id"]).decode("utf-8")
                )
                existing_progress_id = existing_manifest.get("progress_file_id", "")
                progress_state = (
                    json.loads(
                        _drive_file_bytes(service, existing_progress_id).decode("utf-8")
                    ).get("state", "")
                    if existing_progress_id
                    else ""
                )
            except Exception as queue_error:
                print(f"Could not inspect queue state for {job['name']}: {queue_error}")
                continue
            if "status.json" in by_name:
                if progress_state == "cancelling" and existing_progress_id:
                    try:
                        terminal_status = json.loads(
                            _drive_file_bytes(service, by_name["status.json"]["id"]).decode("utf-8")
                        )
                        terminal_state = terminal_status.get("state", "")
                        if terminal_state in {"complete", "failed", "cancelled"}:
                            messages = {
                                "complete": "Analysis completed before cancellation was received.",
                                "failed": "Analysis failed before cancellation was received.",
                                "cancelled": "Analysis cancelled.",
                            }
                            payload = {
                                "state": terminal_state,
                                "message": messages[terminal_state],
                                "error": terminal_status.get("error", ""),
                            }
                            if terminal_state == "complete":
                                payload.update({
                                    "job_progress_fraction": 1.0,
                                    "estimated_remaining_seconds": 0,
                                })
                            _drive_update_json(service, existing_progress_id, payload)
                            progress_state = terminal_state
                    except Exception as reconciliation_error:
                        print(f"Could not reconcile late cancellation: {reconciliation_error}")
                if progress_state != "queued":
                    continue
                service.files().delete(fileId=by_name["status.json"]["id"]).execute()
                children = _drive_children(service, job["id"])
                by_name = {item["name"]: item for item in children}
            elif progress_state != "queued":
                active_states = {
                    "downloading", "analyzing", "training", "uploading_results", "cancelling"
                }
                if progress_state in active_states and existing_progress_id:
                    modified_text = by_name.get("progress.json", {}).get("modifiedTime", "")
                    try:
                        modified_at = datetime.fromisoformat(modified_text.replace("Z", "+00:00"))
                        stale_seconds = (datetime.now(timezone.utc) - modified_at).total_seconds()
                    except (TypeError, ValueError):
                        stale_seconds = float("inf")
                    if stale_seconds > 60:
                        terminal_state = "cancelled" if progress_state == "cancelling" else "failed"
                        message = (
                            "Analysis cancelled."
                            if terminal_state == "cancelled"
                            else (
                                "The previous Colab runtime stopped before completion. "
                                "Retry this job manually from Jobs."
                            )
                        )
                        terminal = {
                            "state": terminal_state,
                            "message": message,
                            "error": "" if terminal_state == "cancelled" else message,
                        }
                        _drive_upload_bytes(
                            service,
                            job["id"],
                            "status.json",
                            json.dumps(terminal).encode("utf-8"),
                            "application/json",
                        )
                        _drive_update_json(service, existing_progress_id, terminal)
                        print(f"Marked interrupted job {job['name']} as {terminal_state}.")
                # A worker may only start jobs explicitly placed in the queue.
                continue
            if "training_manifest.json" in by_name:
                set_worker_activity(
                    str(existing_manifest.get("job_id", job["name"])),
                    str(job["id"]),
                    str(existing_progress_id),
                    "training",
                )
                try:
                    _process_drive_training_job(service, job, by_name, model, device)
                finally:
                    set_worker_activity()
                continue
            if "manifest.json" not in by_name:
                continue

            temp_root = Path(tempfile.mkdtemp(prefix="stimtrace_job_"))
            progress_file_id = None
            processing_started_at = ""
            processing_started_monotonic = 0.0
            set_worker_activity(
                str(existing_manifest.get("job_id", job["name"])),
                str(job["id"]),
                str(existing_progress_id),
                "segmentation",
            )
            try:
                _drive_download(service, by_name["manifest.json"]["id"], temp_root / "manifest.json")
                manifest = json.loads((temp_root / "manifest.json").read_text(encoding="utf-8"))
                progress_file_id = manifest.get("progress_file_id")
                processing_started_at = datetime.now(timezone.utc).isoformat()
                processing_started_monotonic = time.monotonic()
                cancel_state = {"checked_at": 0.0, "requested": "cancel.json" in by_name}

                def cancel_requested(force: bool = False) -> bool:
                    now = time.monotonic()
                    if (
                        not force
                        and now - cancel_state["checked_at"] < 1.0
                    ):
                        return bool(cancel_state["requested"])
                    cancel_state["checked_at"] = now
                    cancel_state["requested"] = _drive_has_named_child(
                        service,
                        job["id"],
                        "cancel.json",
                    )
                    return bool(cancel_state["requested"])

                if cancel_requested(force=True):
                    raise JobCancelled("Cancelled by the user.")
                if progress_file_id:
                    _drive_update_json(service, progress_file_id, {
                        "state": "downloading",
                        "message": "Colab is downloading the submitted videos.",
                        "processing_started_at": processing_started_at,
                    })
                # Each successfully completed recording is checkpointed to this
                # archive. A Colab runtime can disappear at any time; on retry
                # we restore those traces and only download/process recordings
                # that do not yet have a checkpoint.
                checkpoint_name = "checkpoint_results.zip"
                checkpoint_file = by_name.get(checkpoint_name, {})
                checkpoint_path = temp_root / checkpoint_name
                if checkpoint_file.get("id"):
                    try:
                        _drive_download(service, checkpoint_file["id"], checkpoint_path)
                        with zipfile.ZipFile(checkpoint_path) as archive:
                            archive.extractall(temp_root / "results")
                        print("Restored completed recording checkpoints from Drive.", flush=True)
                    except Exception as checkpoint_error:
                        # Keep the job recoverable even if an interrupted Drive
                        # upload left a damaged checkpoint archive behind.
                        print(f"Ignoring unreadable checkpoint archive: {checkpoint_error}", flush=True)
                existing_traces = list((temp_root / "results" / "traces").glob("*_stimtrace_tracking.csv"))
                completed_names = {
                    path.name.removesuffix("_stimtrace_tracking.csv")
                    for path in existing_traces
                }
                pending_videos = [
                    name for name in manifest["videos"]
                    if Path(name).stem not in completed_names
                ]
                if completed_names:
                    print(
                        f"Resuming job: {len(completed_names)} recording(s) restored; "
                        f"{len(pending_videos)} remaining.", flush=True,
                    )
                for video_name in pending_videos:
                    if video_name not in by_name:
                        raise FileNotFoundError(f"Video missing from Drive job: {video_name}")
                    _drive_download(service, by_name[video_name]["id"], temp_root / video_name)

                # ``process_job`` consumes the local manifest. Restrict it to
                # unfinished videos while retaining restored traces for master
                # CSV generation at the end of a resumed attempt.
                manifest["videos"] = pending_videos
                (temp_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

                print(f"Processing job: {manifest['job_id']}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if progress_file_id:
                    processing_finished_at = datetime.now(timezone.utc).isoformat()
                    _drive_update_json(service, progress_file_id, {
                        "state": "analyzing",
                        "message": "Colab is segmenting pillars and calculating traces.",
                        "processing_started_at": processing_started_at,
                    })
                def publish_progress(video_name, frame_number, total_frames, details):
                    if cancel_requested(force=True):
                        raise JobCancelled("Cancelled by the user.")
                    try:
                        if progress_file_id:
                            stage = details.get("stage", "segmenting")
                            file_index = details["current_file_index"]
                            file_count = details["file_count"]
                            if stage == "benchmark_video":
                                message = (
                                    f"Rendering Kalman benchmark videos for file {file_index} of "
                                    f"{file_count}: {video_name}, output frame {frame_number} of "
                                    f"{total_frames or '?'}."
                                )
                            elif stage == "overlay":
                                message = (
                                    f"Creating overlay video for file {file_index} of {file_count}: "
                                    f"{video_name}, frame {frame_number} of {total_frames or '?'}."
                                )
                            elif stage == "workbook":
                                message = "Creating combined result files."
                            else:
                                message = (
                                    f"Segmenting file {file_index} of {file_count}: {video_name}, "
                                    f"frame {frame_number} of {total_frames or '?'}."
                                )
                            _drive_update_json(service, progress_file_id, {
                                "state": "analyzing",
                                "message": message,
                                "current_file_name": video_name,
                                "processing_started_at": processing_started_at,
                                **details,
                            })
                    except Exception as progress_error:
                        print(f"Progress update skipped: {progress_error}")

                checkpoint_file_id = checkpoint_file.get("id", "")

                def checkpoint_completed_video(video_name: str) -> None:
                    nonlocal checkpoint_file_id
                    checkpoint_destination = temp_root / checkpoint_name
                    _zip_results(temp_root / "results", checkpoint_destination, cancel_requested)
                    if checkpoint_file_id:
                        _drive_update_file(
                            service, checkpoint_file_id, checkpoint_destination,
                            "application/zip", cancel_requested,
                        )
                    else:
                        checkpoint_file_id = _drive_upload_bytes(
                            service, job["id"], checkpoint_name,
                            checkpoint_destination.read_bytes(), "application/zip",
                        )
                    print(f"Checkpoint saved after {video_name}.", flush=True)

                processed_frame_count = process_job(
                    temp_root,
                    model,
                    device,
                    publish_progress,
                    cancel_requested,
                    cloud_cuda_batch_limit,
                    checkpoint_completed_video,
                    existing_traces,
                )
                if cancel_requested(force=True):
                    raise JobCancelled("Cancelled by the user.")
                if progress_file_id:
                    _drive_update_json(service, progress_file_id, {
                        "state": "uploading_results",
                        "message": "Analysis finished. Colab is packaging and uploading the result files.",
                        "stage": "packaging",
                        "current_file_name": "",
                        "current_file_index": None,
                        "current_frame": None,
                        "current_file_frames": None,
                        "file_progress_fraction": 1.0,
                        "job_progress_fraction": 0.98,
                        "processing_started_at": processing_started_at,
                        "processed_frames": processed_frame_count,
                        "total_job_frames": processed_frame_count,
                    })
                result_archive_file_id = manifest.get("result_archive_file_id", "")
                if not result_archive_file_id:
                    raise RuntimeError(
                    "This job has no authorized result archive. Retry it from the updated StimTrace application."
                    )
                result_archive = temp_root / "results.zip"
                _zip_results(
                    temp_root / "results",
                    result_archive,
                    cancel_requested,
                )
                if cancel_requested(force=True):
                    raise JobCancelled("Cancelled by the user.")
                _drive_update_file(
                    service,
                    result_archive_file_id,
                    result_archive,
                    "application/zip",
                    cancel_requested,
                )
                if cancel_requested(force=True):
                    raise JobCancelled("Cancelled by the user.")
                status = (temp_root / "status.json").read_bytes()
                _drive_upload_bytes(service, job["id"], "status.json", status, "application/json")
                if progress_file_id:
                    _drive_update_json(service, progress_file_id, {
                        "state": "complete",
                        "message": "Analysis complete. Results are available in Google Drive.",
                        "stage": "complete",
                        "current_file_name": "",
                        "current_file_index": None,
                        "current_frame": None,
                        "current_file_frames": None,
                        "file_progress_fraction": 1.0,
                        "job_progress_fraction": 1.0,
                        "processing_started_at": processing_started_at,
                        "processed_frames": processed_frame_count,
                        "total_job_frames": processed_frame_count,
                        "estimated_remaining_seconds": 0,
                        "processing_finished_at": processing_finished_at,
                        "actual_total_seconds": max(
                            0.0, time.monotonic() - processing_started_monotonic
                        ),
                    })
                print(f"Completed job: {manifest['job_id']}")
            except JobCancelled as error:
                cancellation = json.dumps({"state": "cancelled", "message": str(error)}).encode("utf-8")
                _drive_upload_bytes(service, job["id"], "status.json", cancellation, "application/json")
                if progress_file_id:
                    processing_finished_at = datetime.now(timezone.utc).isoformat()
                    _drive_update_json(service, progress_file_id, {
                        "state": "cancelled",
                        "message": "Analysis cancelled.",
                        "stage": "cancelled",
                        "current_file_name": "",
                        "current_file_index": None,
                        "current_frame": None,
                        "current_file_frames": None,
                        "file_progress_fraction": 0.0,
                        "job_progress_fraction": 0.0,
                        "error": "",
                        "processing_started_at": processing_started_at,
                        "processing_finished_at": processing_finished_at,
                        "actual_total_seconds": max(
                            0.0, time.monotonic() - processing_started_monotonic
                        ) if processing_started_monotonic else 0.0,
                    })
                print(f"Cancelled job: {job['name']}")
            except Exception as error:
                failure = json.dumps({"state": "failed", "error": str(error)}).encode("utf-8")
                _drive_upload_bytes(service, job["id"], "status.json", failure, "application/json")
                if progress_file_id:
                    processing_finished_at = datetime.now(timezone.utc).isoformat()
                    _drive_update_json(service, progress_file_id, {
                        "state": "failed",
                        "message": "Analysis failed.",
                        "stage": "failed",
                        "current_file_name": "",
                        "current_file_index": None,
                        "current_frame": None,
                        "current_file_frames": None,
                        "file_progress_fraction": 0.0,
                        "job_progress_fraction": 0.0,
                        "error": str(error),
                        "processing_started_at": processing_started_at,
                        "processing_finished_at": processing_finished_at,
                        "actual_total_seconds": max(
                            0.0, time.monotonic() - processing_started_monotonic
                        ) if processing_started_monotonic else 0.0,
                    })
                print(f"Job {job['name']} failed: {error}")
            finally:
                set_worker_activity()
                shutil.rmtree(temp_root, ignore_errors=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        time.sleep(poll_seconds)
