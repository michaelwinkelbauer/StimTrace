"""Interactive force-trace processing for StimTrace segmentation outputs."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QItemSelectionModel, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, savgol_filter

from app_logging import get_logger
from annotated_recording_viewer import AnnotatedRecordingViewer, find_annotated_recording
from background_tasks import BackgroundFunctionThread
from ui_actions import set_action_icon, set_polarity_icon
from ui_components import FrozenFirstColumnTable


LOGGER = get_logger("signal_analysis")


BASELINE_METHODS = (
    ("None", "none"),
    ("Constant percentile", "constant_percentile"),
    ("Rolling percentile", "rolling_percentile"),
    ("Asymmetric least squares", "asymmetric_least_squares"),
)

REFERENCE_METRICS = (
    "Beat_rate_BPM",
    "Peak_force_mean",
    "Amplitude_mean",
    "Time_to_peak_ms_mean",
    "Rise_10_90_ms_mean",
    "Relaxation_50_ms_mean",
    "Relaxation_90_ms_mean",
    "CTD50_ms_mean",
    "CTD90_ms_mean",
    "Max_contraction_slope_per_s_mean",
    "Max_relaxation_slope_per_s_mean",
)

BEAT_METRIC_COLUMNS = (
    "Peak_force",
    "Diastolic_force_pre",
    "Amplitude",
    "Time_to_peak_ms",
    "Rise_10_90_ms",
    "Relaxation_50_ms",
    "Relaxation_90_ms",
    "CTD50_ms",
    "CTD90_ms",
    "Signal_to_noise",
    "Max_contraction_slope_per_s",
    "Max_relaxation_slope_per_s",
)


def summary_display_columns(summary: pd.DataFrame) -> list[str]:
    """Return the compact Summary-table columns, including reference metrics when available."""
    preferred = [
        "Trace", "Specimen", "Analysis_status", "Analysis_error",
        "Reference_trace", "Reference_match_status",
        "N_beats", "Beat_rate_BPM", "Beat_rate_BPM_pct_baseline",
        "Peak_force_mean", "Peak_force_mean_pct_baseline",
        "Amplitude_mean", "Amplitude_mean_pct_baseline",
        "Time_to_peak_ms_mean", "Time_to_peak_ms_mean_pct_baseline",
        "Rise_10_90_ms_mean", "Rise_10_90_ms_mean_pct_baseline",
        "Relaxation_50_ms_mean",
        "Relaxation_50_ms_mean_pct_baseline",
        "Relaxation_90_ms_mean", "Relaxation_90_ms_mean_pct_baseline",
        "CTD50_ms_mean", "CTD50_ms_mean_pct_baseline",
        "CTD90_ms_mean", "CTD90_ms_mean_pct_baseline",
        "Max_contraction_slope_per_s_mean",
        "Max_contraction_slope_per_s_mean_pct_baseline",
        "Max_relaxation_slope_per_s_mean",
        "Max_relaxation_slope_per_s_mean_pct_baseline",
        "Beat_interval_CV_pct",
    ]
    return [column for column in preferred if column in summary.columns]


SUMMARY_COLUMN_LABELS = {
    "Analysis_status": "Analysis status",
    "Analysis_error": "Analysis note",
    "Beat_rate_BPM_pct_baseline": "Beat rate (% reference)",
    "Peak_force_mean_pct_baseline": "Peak force (% reference)",
    "Amplitude_mean_pct_baseline": "Amplitude (% reference)",
    "Time_to_peak_ms_mean_pct_baseline": "Time to peak (% reference)",
    "Rise_10_90_ms_mean_pct_baseline": "Rise 10–90% (% reference)",
    "Relaxation_50_ms_mean_pct_baseline": "Relaxation 50% (% reference)",
    "Relaxation_90_ms_mean_pct_baseline": "Relaxation 90% (% reference)",
    "CTD50_ms_mean_pct_baseline": "CTD50 (% reference)",
    "CTD90_ms_mean_pct_baseline": "CTD90 (% reference)",
    "Max_contraction_slope_per_s_mean_pct_baseline": "Max contraction slope (% reference)",
    "Max_relaxation_slope_per_s_mean_pct_baseline": "Max relaxation slope (% reference)",
}


@dataclass
class AnalysisSettings:
    min_interval_s: float = 0.10
    prominence_pct: float = 20.0
    prominence_mode: str = "percent_range"
    prominence_absolute: float = 10.0
    smoothing_s: float = 0.15
    start_s: float | None = None
    end_s: float | None = None
    invert: bool = False
    baseline_method: str = "none"
    baseline_percentile: float = 10.0
    baseline_window_s: float = 5.0
    baseline_asls_log10: float = 6.0
    time_shift_frames: int = 0


@dataclass(frozen=True)
class TraceProcessingSettings:
    invert: bool = False
    baseline_method: str = "none"
    baseline_percentile: float = 10.0
    baseline_window_s: float = 5.0
    baseline_asls_log10: float = 6.0
    time_shift_frames: int = 0


@dataclass
class PreparedSignal:
    t: np.ndarray
    original: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    smooth: np.ndarray
    dt: float
    smoothing_window: int


def _odd_window(seconds: float, dt: float, count: int) -> int:
    if seconds <= 0 or not np.isfinite(dt) or dt <= 0 or count < 5:
        return 0
    window = max(3, int(round(seconds / dt)))
    if window % 2 == 0:
        window += 1
    maximum = count if count % 2 else count - 1
    return min(window, maximum)


def _crossing_time(t: np.ndarray, y: np.ndarray, level: float, rising: bool) -> float:
    if len(t) < 2:
        return np.nan
    delta = y - level
    hits = (
        np.where((delta[:-1] <= 0) & (delta[1:] >= 0))[0]
        if rising
        else np.where((delta[:-1] >= 0) & (delta[1:] <= 0))[0]
    )
    if not len(hits):
        return np.nan
    index = int(hits[-1] if rising else hits[0])
    dy = y[index + 1] - y[index]
    if dy == 0:
        return float(t[index])
    return float(t[index] + (level - y[index]) * (t[index + 1] - t[index]) / dy)


def parse_trace_name(name: str) -> tuple[str, str, str]:
    parts = str(name).split("|")
    timestamp, condition, specimen = "", "", str(name)
    if len(parts) >= 3:
        middle = parts[1].strip()
        if " : " in middle:
            clock, condition = middle.rsplit(" : ", 1)
            timestamp = f"{parts[0].strip()} {clock.strip()}"
        else:
            timestamp, condition = parts[0].strip(), middle
        specimen = re.sub(r"_(Force|force).*", "", parts[2].strip()).strip()
    elif len(parts) == 2:
        left, right = parts[0].strip(), parts[1].strip()
        condition = left
        specimen = re.sub(r"_(Force|force).*", "", right).strip()
        if re.fullmatch(
            r"(?:force|xy_combo|displacement|signal)(?:[_ ].*)?",
            right,
            flags=re.IGNORECASE,
        ):
            # Separate one-trace files are prefixed with their filename when loaded
            # together. In that case the filename identifies the specimen.
            specimen = left
    else:
        specimen = re.sub(r"_(Force|force|XY_combo).*", "", str(name)).strip()
    return condition, specimen, timestamp


def reference_match_key(name: str) -> str:
    """Return a condition-independent specimen key for reference matching.

    StimTrace recording names commonly use ``CONDITION_SPECIMEN`` (for example,
    ``BL_B-1`` and ``H_B-1``). Keep the full parsed specimen for display, while
    matching references by the portion after the first condition prefix.
    """
    # Validation traces are several independent methods for *one recording*,
    # rather than conditions for one specimen.  Match those by the original
    # recording stem before applying the normal biological specimen rule.
    validation_key = validation_recording_match_key(name)
    if validation_key:
        return f"recording::{validation_key}"

    # Processed-trace exports append a display-only column suffix, e.g.
    # ``StimTrace | BL_B-1_Force_uN | processed``.  The generic parser treats
    # the last component as the specimen, which would make every selected
    # processed trace look like the same reference.  For matching,
    # deliberately recover the actual measurement column before parsing it.
    parts = [part.strip() for part in str(name).split("|")]
    # Keep the retired labels readable when older exports are reopened.
    processing_labels = {"processed", "smoothed", "corrected"}
    if len(parts) >= 3 and parts[-1].casefold() in processing_labels:
        measurement_name = re.sub(r"_(Force|force|XY_combo).*", "", parts[-2]).strip()
        specimen = measurement_name or parse_trace_name(name)[1].strip()
    else:
        specimen = parse_trace_name(name)[1].strip()
    prefix, separator, specimen_key = specimen.partition("_")
    if separator and prefix and specimen_key:
        return specimen_key.casefold()
    return specimen.casefold()


def validation_recording_match_key(name: str) -> str:
    """Return a stable original-recording key for validation method traces.

    A validation recording may appear as a source filename plus a force column
    (``recording_manual_ground_truth_trace | manual_ground_truth_force_uN``), or as
    a normalised display label.  In either case, manual-ellipse, raw-DL, and
    point-tracking forces must resolve to the same recording—not to their
    measurement-method names.
    """
    parts = [part.strip() for part in str(name).split("|")]
    while parts and parts[-1].casefold() in {"processed", "smoothed", "corrected"}:
        parts.pop()
    for part in parts:
        token = re.sub(r"\s*\(\d+\)\s*$", "", part).strip()
        lowered = token.casefold()
        for suffix in (
            "_manual_ellipse_validation", "_point_tracking_validation",
            "_manual_ellipse_trace", "_manual_ground_truth_trace", "_dl_raw_ellipse_trace", "_point_tracking_trace",
        ):
            if lowered.endswith(suffix):
                recording = token[: -len(suffix)].strip(" _-")
                if recording:
                    return recording.casefold()
        # This handles Signal Analysis labels where the recording stem and
        # numeric method column have already been joined into one display name.
        match = re.match(
            r"^(?P<recording>.+?)_(?:manual_ellipse|manual_ground_truth|dl_raw|point(?:_tracking)?(?:_\d+)?)"
            r"(?:_force(?:_u[nN])?)?$",
            token,
            flags=re.IGNORECASE,
        )
        if match:
            recording = match.group("recording").strip(" _-")
            if recording:
                return recording.casefold()
    return ""


def reference_match_description(match_key: str) -> str:
    """Human-readable matching method for results and exports."""
    return "Matched by recording" if str(match_key).startswith("recording::") else "Matched by specimen"


def is_auxiliary_trace_column(name: str) -> bool:
    """Identify exported baseline/original columns without matching words such as baselineClean."""
    return bool(
        re.search(
            r"(?:^|[|_\s-])(baseline|original|uncorrected)(?:$|[|_\s-])",
            str(name),
            flags=re.IGNORECASE,
        )
    )


def selected_result_frames(
    summary: pd.DataFrame,
    beats: pd.DataFrame,
    selected_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict previously calculated results to the current trace-list selection."""
    selected = set(map(str, selected_columns))
    selected_summary = (
        summary[summary["Trace"].astype(str).isin(selected)].copy()
        if not summary.empty and "Trace" in summary
        else pd.DataFrame(columns=summary.columns)
    )
    selected_beats = (
        beats[beats["Trace"].astype(str).isin(selected)].copy()
        if not beats.empty and "Trace" in beats
        else pd.DataFrame(columns=beats.columns)
    )
    return selected_summary, selected_beats


def estimate_baseline(
    values: np.ndarray,
    dt: float,
    settings: AnalysisSettings,
) -> np.ndarray:
    method = settings.baseline_method
    if method == "none":
        return np.zeros_like(values)
    if method == "constant_percentile":
        return np.full_like(values, np.nanpercentile(values, settings.baseline_percentile))
    if method == "rolling_percentile":
        window = _odd_window(settings.baseline_window_s, dt, len(values))
        if window < 3:
            raise ValueError("The rolling baseline window needs at least three samples.")
        baseline = (
            pd.Series(values)
            .rolling(
                window=window,
                center=True,
                min_periods=max(3, (window + 1) // 4),
            )
            .quantile(settings.baseline_percentile / 100.0)
            .to_numpy(dtype=float)
        )
        # A centered rolling window has incomplete support at both edges. Preserve
        # the requested method without feeding NaNs into peak detection.
        return (
            pd.Series(baseline)
            .interpolate(limit_direction="both")
            .to_numpy(dtype=float)
        )
    if method == "asymmetric_least_squares":
        if len(values) < 5:
            raise ValueError("Asymmetric least-squares correction needs at least five samples.")
        difference = diags(
            (np.ones(len(values) - 2), -2 * np.ones(len(values) - 2), np.ones(len(values) - 2)),
            (0, 1, 2),
            shape=(len(values) - 2, len(values)),
            format="csc",
        )
        smoothness = 10.0 ** settings.baseline_asls_log10
        asymmetry = 0.01
        weights = np.ones(len(values))
        baseline = values.copy()
        penalty = smoothness * (difference.T @ difference)
        for _ in range(10):
            weight_matrix = diags(weights, 0, format="csc")
            baseline = spsolve(weight_matrix + penalty, weights * values)
            weights = np.where(values > baseline, asymmetry, 1.0 - asymmetry)
        return np.asarray(baseline, dtype=float)
    raise ValueError(f"Unknown baseline correction method: {method}")


def prepare_signal(time_values, force_values, settings: AnalysisSettings) -> PreparedSignal:
    t = np.asarray(time_values, dtype=float)
    original = np.asarray(force_values, dtype=float)
    if settings.invert:
        original = -original
    valid = np.isfinite(t) & np.isfinite(original)
    t, original = t[valid], original[valid]
    if len(t) < 5:
        raise ValueError("Fewer than five valid samples are available.")
    order = np.argsort(t)
    t, original = t[order], original[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, original = t[unique], original[unique]
    source_dt = float(np.median(np.diff(t)))
    if not np.isfinite(source_dt) or source_dt <= 0:
        raise ValueError("The time axis does not have a valid frame interval.")
    t = t + int(settings.time_shift_frames) * source_dt
    window = np.ones(len(t), dtype=bool)
    if settings.start_s is not None:
        window &= t >= settings.start_s
    if settings.end_s is not None:
        window &= t <= settings.end_s
    t, original = t[window], original[window]
    if len(t) < 5:
        raise ValueError("Fewer than five valid samples remain in the analysis window.")
    dt = float(np.median(np.diff(t)))
    baseline = estimate_baseline(original, dt, settings)
    corrected = original - baseline
    window = _odd_window(settings.smoothing_s, dt, len(corrected))
    smooth = (
        savgol_filter(corrected, window, min(3, window - 1), mode="interp")
        if window >= 3
        else corrected.copy()
    )
    return PreparedSignal(t, original, baseline, corrected, smooth, dt, window)


def detect_signal_peaks(
    smooth: np.ndarray,
    dt: float,
    settings: AnalysisSettings,
) -> np.ndarray:
    """Detect peaks using the same thresholds used for trace analysis and alignment."""
    signal_range = float(np.nanpercentile(smooth, 99) - np.nanpercentile(smooth, 1))
    if signal_range <= 0:
        raise ValueError("The trace has no measurable dynamic range.")
    if settings.prominence_mode == "absolute":
        prominence = float(settings.prominence_absolute)
        if not np.isfinite(prominence) or prominence <= 0:
            raise ValueError("Absolute peak prominence must be greater than zero.")
    elif settings.prominence_mode == "percent_range":
        prominence = signal_range * settings.prominence_pct / 100.0
    else:
        raise ValueError(f"Unknown peak prominence mode: {settings.prominence_mode}")
    distance = max(1, int(round(settings.min_interval_s / dt)))
    peaks, _ = find_peaks(smooth, distance=distance, prominence=prominence)
    return peaks


def first_detected_peak_time(
    time_values,
    force_values,
    settings: AnalysisSettings,
) -> tuple[float, float]:
    """Return the first detected peak time and its source frame interval."""
    prepared = prepare_signal(time_values, force_values, settings)
    peaks = detect_signal_peaks(prepared.smooth, prepared.dt, settings)
    if not len(peaks):
        raise ValueError("No peaks detected. Reduce prominence or invert the signal.")
    return float(prepared.t[int(peaks[0])]), prepared.dt


def analyze_trace(time_values, force_values, trace_name: str, settings: AnalysisSettings):
    prepared = prepare_signal(time_values, force_values, settings)
    t, raw, smooth, dt = prepared.t, prepared.corrected, prepared.smooth, prepared.dt
    signal_range = float(np.nanpercentile(smooth, 99) - np.nanpercentile(smooth, 1))
    peaks = detect_signal_peaks(smooth, dt, settings)
    if settings.prominence_mode == "absolute":
        prominence = float(settings.prominence_absolute)
    else:
        prominence = signal_range * settings.prominence_pct / 100.0
    if not len(peaks):
        raise ValueError("No beats detected. Reduce prominence or invert the signal.")

    condition, specimen, timestamp = parse_trace_name(trace_name)
    residual = raw - smooth
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    rows = []
    for peak_index, peak in enumerate(peaks):
        left_edge = 0 if peak_index == 0 else int((peaks[peak_index - 1] + peak) // 2)
        right_edge = len(smooth) - 1 if peak_index == len(peaks) - 1 else int((peak + peaks[peak_index + 1]) // 2)
        left = int(left_edge + np.argmin(smooth[left_edge:peak + 1]))
        right = int(peak + np.argmin(smooth[peak:right_edge + 1]))
        if not left < peak < right:
            continue
        amp_left = float(smooth[peak] - smooth[left])
        amp_right = float(smooth[peak] - smooth[right])
        amplitude = (amp_left + amp_right) / 2.0
        if amplitude <= 0:
            continue
        rise_levels = {q: smooth[left] + q * amp_left for q in (0.1, 0.5, 0.9)}
        fall_levels = {q: smooth[right] + q * amp_right for q in (0.1, 0.5, 0.9)}
        rise = {q: _crossing_time(t[left:peak + 1], smooth[left:peak + 1], value, True) for q, value in rise_levels.items()}
        fall = {q: _crossing_time(t[peak:right + 1], smooth[peak:right + 1], value, False) for q, value in fall_levels.items()}
        rows.append({
            "Trace": trace_name,
            "Condition": condition,
            "Specimen": specimen,
            "Timestamp": timestamp,
            "Beat": len(rows) + 1,
            "Peak_time_s": t[peak],
            "Peak_force": smooth[peak],
            "Diastolic_force_pre": smooth[left],
            "Diastolic_force_post": smooth[right],
            "Amplitude": amplitude,
            "Time_to_peak_ms": (t[peak] - t[left]) * 1000,
            "Rise_10_90_ms": (rise[0.9] - rise[0.1]) * 1000,
            "Relaxation_50_ms": (fall[0.5] - t[peak]) * 1000,
            "Relaxation_90_ms": (fall[0.1] - t[peak]) * 1000,
            "CTD50_ms": (fall[0.5] - rise[0.5]) * 1000,
            "CTD90_ms": (fall[0.1] - rise[0.1]) * 1000,
            "Signal_to_noise": amplitude / noise if noise > 0 else np.nan,
            "Max_contraction_slope_per_s": float(np.max(np.gradient(smooth[left:peak + 1], t[left:peak + 1]))),
            "Max_relaxation_slope_per_s": float(np.min(np.gradient(smooth[peak:right + 1], t[peak:right + 1]))),
        })
    beats = pd.DataFrame(rows)
    if beats.empty:
        raise ValueError("Peaks were found, but no complete beats could be measured.")
    intervals = np.diff(beats["Peak_time_s"].to_numpy(float))
    duration = float(t[-1] - t[0])
    # A rate requires at least one complete inter-beat interval. Estimating BPM
    # from a single peak and the window duration depends on arbitrary crop edges.
    rate = float(60.0 / np.mean(intervals)) if len(intervals) else np.nan
    interval_cv = float(np.std(intervals, ddof=1) / np.mean(intervals) * 100) if len(intervals) > 1 else np.nan
    summary = {
        "Trace": trace_name,
        "Condition": condition,
        "Specimen": specimen,
        "Timestamp": timestamp,
        "N_beats": len(beats),
        "Beat_rate_BPM": rate,
        "Beat_interval_CV_pct": interval_cv,
        "Analyzed_duration_s": duration,
        "Sampling_interval_ms": dt * 1000,
        "Smoothing_window_samples": prepared.smoothing_window,
        "Inverted": settings.invert,
        "Baseline_method": settings.baseline_method,
        "Signal_range_1_99": signal_range,
        "Peak_prominence_mode": settings.prominence_mode,
        "Peak_prominence_threshold": prominence,
        "Analysis_status": "Complete",
        "Analysis_error": "",
    }
    for column in BEAT_METRIC_COLUMNS:
        values = pd.to_numeric(beats[column], errors="coerce")
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_SD"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
    return summary, beats, {
        "t": t,
        "original": prepared.original,
        "baseline": prepared.baseline,
        "raw": raw,
        "smooth": smooth,
        "peaks": peaks,
        "baseline_method": settings.baseline_method,
        "prominence": prominence,
        "prominence_mode": settings.prominence_mode,
    }


def failed_trace_summary(
    trace_name: str,
    settings: AnalysisSettings,
    error: Exception | str,
) -> dict:
    """Create an explicit NaN summary row for a trace without measurable beats."""
    condition, specimen, timestamp = parse_trace_name(trace_name)
    message = str(error)
    if "No beats detected" in message:
        status = "No beats detected"
    elif "no complete beats" in message.lower():
        status = "No complete beats"
    else:
        status = "Analysis failed"
    summary = {
        "Trace": trace_name,
        "Condition": condition,
        "Specimen": specimen,
        "Timestamp": timestamp,
        "N_beats": 0,
        "Beat_rate_BPM": np.nan,
        "Beat_interval_CV_pct": np.nan,
        "Analyzed_duration_s": np.nan,
        "Sampling_interval_ms": np.nan,
        "Smoothing_window_samples": np.nan,
        "Inverted": settings.invert,
        "Baseline_method": settings.baseline_method,
        "Signal_range_1_99": np.nan,
        "Peak_prominence_mode": settings.prominence_mode,
        "Peak_prominence_threshold": np.nan,
        "Analysis_status": status,
        "Analysis_error": message,
    }
    for column in BEAT_METRIC_COLUMNS:
        summary[f"{column}_mean"] = np.nan
        summary[f"{column}_SD"] = np.nan
    return summary


def analyze_trace_batch(
    time_values: np.ndarray,
    signals: dict[str, np.ndarray],
    settings_by_column: dict[str, AnalysisSettings],
    selected_columns: list[str],
    reference_columns: list[str],
    baseline_columns: set[str],
) -> dict:
    """Analyze an immutable trace snapshot without touching Qt objects."""
    summaries: list[dict] = []
    beat_frames: list[pd.DataFrame] = []
    errors: list[str] = []
    plot_cache: dict[str, dict] = {}
    reference_by_specimen = {
        reference_match_key(column): column for column in reference_columns
    }
    reference_summaries: dict[str, dict] = {}
    failed_references: set[str] = set()
    analysis_columns = list(dict.fromkeys([*selected_columns, *reference_columns]))
    for column in analysis_columns:
        try:
            summary, beats, plot_info = analyze_trace(
                time_values,
                signals[column],
                column,
                settings_by_column[column],
            )
            if column in selected_columns:
                summaries.append(summary)
                beat_frames.append(beats)
                plot_cache[column] = plot_info
            if column in baseline_columns:
                reference_summaries[reference_match_key(column)] = summary
        except Exception as error:
            LOGGER.warning("Trace analysis failed for %s: %s", column, error)
            errors.append(f"{column}: {error}")
            if column in selected_columns:
                summaries.append(
                    failed_trace_summary(
                        column,
                        settings_by_column.get(column, AnalysisSettings()),
                        error,
                    )
                )
            if column in baseline_columns:
                failed_references.add(column)

    summary_frame = pd.DataFrame(summaries)
    beats_frame = (
        pd.concat(beat_frames, ignore_index=True) if beat_frames else pd.DataFrame()
    )
    if not summary_frame.empty:
        for index, row in summary_frame.iterrows():
            match_key = reference_match_key(str(row["Trace"]))
            reference_column = reference_by_specimen.get(match_key, "")
            summary_frame.loc[index, "Is_reference_trace"] = (
                "Yes" if row["Trace"] in baseline_columns else "No"
            )
            summary_frame.loc[index, "Reference_trace"] = reference_column
            summary_frame.loc[index, "Reference_match_status"] = (
                "Reference analysis failed"
                if reference_column in failed_references
                else reference_match_description(match_key)
                if reference_column
                else "No matching reference"
            )
            reference = reference_summaries.get(match_key)
            if reference:
                for metric in REFERENCE_METRICS:
                    denominator = reference.get(metric, np.nan)
                    summary_frame.loc[index, f"{metric}_pct_baseline"] = (
                        100 * row.get(metric, np.nan) / denominator
                        if np.isfinite(denominator) and denominator != 0
                        else np.nan
                    )
        if not beats_frame.empty:
            beats_frame["Is_reference_trace"] = beats_frame["Trace"].map(
                lambda trace: "Yes" if trace in baseline_columns else "No"
            )
            beats_frame["Reference_trace"] = beats_frame["Trace"].map(
                lambda trace: reference_by_specimen.get(reference_match_key(str(trace)), "")
            )
            beats_frame["Reference_match_status"] = beats_frame["Reference_trace"].map(
                lambda reference: reference_match_description(reference_match_key(reference))
                if reference else "No matching reference"
            )
    return {
        "summary": summary_frame,
        "beats": beats_frame,
        "plot_cache": plot_cache,
        "errors": errors,
        "selected_count": len(selected_columns),
    }


def freeze_export_headers(writer: pd.ExcelWriter) -> None:
    """Keep the first identifier column and header row visible in every sheet."""
    for worksheet in writer.sheets.values():
        worksheet.freeze_panes = "B2"


def write_metrics_workbook(
    filename: str,
    summary: pd.DataFrame,
    beats: pd.DataFrame,
    references: pd.DataFrame,
    settings: pd.DataFrame,
    processing: pd.DataFrame,
) -> str:
    """Write a metrics workbook in a worker thread."""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Trace summary", index=False)
        beats.to_excel(writer, sheet_name="Beat metrics", index=False)
        references.to_excel(writer, sheet_name="Reference traces", index=False)
        settings.to_excel(writer, sheet_name="Settings", index=False)
        processing.to_excel(writer, sheet_name="Trace processing", index=False)
        freeze_export_headers(writer)
    return filename


def write_processed_workbook(
    filename: str,
    time_column: str,
    time_values: np.ndarray,
    signals: dict[str, np.ndarray],
    settings_by_column: dict[str, AnalysisSettings],
    references: pd.DataFrame,
    settings: pd.DataFrame,
    processing: pd.DataFrame,
) -> str:
    """Calculate and write selected processed traces away from the GUI thread."""
    output_frames: list[pd.DataFrame] = []
    for column, values in signals.items():
        prepared = prepare_signal(time_values, values, settings_by_column[column])
        exported_signals = {
            f"{column} | original": prepared.original,
            # This is the final baseline-corrected and smoothed signal used by
            # peak detection and beat-metric calculations.
            f"{column} | processed": prepared.smooth,
        }
        output_frames.append(
            pd.DataFrame({time_column: prepared.t, **exported_signals}).set_index(time_column)
        )
    output = pd.concat(output_frames, axis=1).sort_index().reset_index()
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="Processed traces", index=False)
        references.to_excel(writer, sheet_name="Reference traces", index=False)
        settings.to_excel(writer, sheet_name="Settings", index=False)
        processing.to_excel(writer, sheet_name="Trace processing", index=False)
        freeze_export_headers(writer)
    return filename


class SheetDialog(QDialog):
    def __init__(self, sheets: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose worksheet")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Worksheet to load"))
        self.list = QListWidget()
        self.list.addItems(sheets)
        self.list.setCurrentRow(0)
        layout.addWidget(self.list)
        buttons = QDialogButtonBox(QDialogButtonBox.Open | QDialogButtonBox.Cancel)
        set_action_icon(buttons.button(QDialogButtonBox.Open), "open")
        set_action_icon(buttons.button(QDialogButtonBox.Cancel), "cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_sheet(self) -> str:
        return self.list.currentItem().text()


class TraceListRow(QWidget):
    """Compact trace row with direct inversion and frame-alignment controls."""

    row_clicked = Signal(object)
    row_dragged = Signal(object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("traceListRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("selected", False)

    def set_selected(self, selected: bool) -> None:
        """Paint one continuous selection band behind controls and labels."""
        selected = bool(selected)
        if self.property("selected") == selected:
            return
        self.setProperty("selected", selected)
        self.setStyleSheet(
            """
            QWidget#traceListRow {
                background: #653040;
                border-radius: 3px;
            }
            QWidget#traceListRow QLabel {
                color: #ffffff;
                background: #653040;
            }
            QWidget#traceListRow QPushButton[role="traceControl"] {
                background: #653040;
                border-color: #a65369;
                border-radius: 3px;
            }
            QWidget#traceListRow QPushButton[role="traceControl"]:hover {
                background: #774052;
            }
            QWidget#traceListRow QPushButton[role="traceControl"]:checked {
                background: #963b55;
            }
            """
            if selected
            else ""
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.row_clicked.emit(event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton:
            self.row_dragged.emit(
                event.globalPosition().toPoint(),
                event.modifiers(),
            )
            event.accept()
            return
        super().mouseMoveEvent(event)


class SignalAnalysisPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        parent_settings = getattr(parent, "settings", None)
        try:
            self.force_calibration_available = (
                parent_settings is not None
                and np.isfinite(float(parent_settings.force_slope_un_per_um))
                and float(parent_settings.force_slope_un_per_um) != 0
            )
        except (AttributeError, TypeError, ValueError):
            self.force_calibration_available = False
        self.data: pd.DataFrame | None = None
        self.path: Path | None = None
        self.paths: list[Path] = []
        self.trace_source_paths: dict[str, Path] = {}
        self.time_column = ""
        self.visible_columns: list[str] = []
        self.plot_cache: dict[str, dict] = {}
        self.summary = pd.DataFrame()
        self.beats = pd.DataFrame()
        self.baseline_columns: set[str] = set()
        self.auxiliary_traces_hidden = False
        self.trace_processing: dict[str, TraceProcessingSettings] = {}
        # The first-peak action is intentionally reversible.  Keep the offsets
        # that existed immediately before the automatic alignment so a second
        # click restores the exact timing state rather than assuming zero.
        self._first_peak_alignment_restore: dict[str, int] = {}
        self.trace_row_controls: dict[str, tuple[QPushButton, QLabel]] = {}
        self._pending_plot_view: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._trace_drag_anchor_item: QListWidgetItem | None = None
        self._trace_drag_base_rows: set[int] = set()
        self._trace_drag_additive = False
        self._trace_drag_selecting = True
        self._loading_processing_controls = False
        self._preloaded_trace_files: dict[str, tuple[pd.DataFrame, str]] = {}
        self.operation_thread: BackgroundFunctionThread | None = None
        self.recording_viewers: set[AnnotatedRecordingViewer] = set()
        self._selected_recording_path: Path | None = None
        self._operation_busy = False
        self._build_ui()
        self.plot_timer = QTimer(self)
        self.plot_timer.setSingleShot(True)
        self.plot_timer.setInterval(40)
        self.plot_timer.timeout.connect(self.plot_selected)

    def _build_ui(self):
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(18, 14, 18, 14)
        header = QHBoxLayout()
        title = QLabel("StimTrace Signal Analysis")
        title.setProperty("role", "title")
        help_button = QPushButton("Help")
        set_action_icon(help_button, "help")
        help_button.clicked.connect(self.show_help)
        header.addWidget(title, 1)
        header.addWidget(help_button)
        page_layout.addLayout(header)
        description = QLabel(
            "Load generated force traces, inspect and process selected signals, calculate "
            "contraction metrics, and export the results."
        )
        description.setWordWrap(True)
        description.setProperty("role", "subtitle")
        page_layout.addWidget(description)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        page_layout.addWidget(splitter, 1)

        self.source_label = QLabel("No trace file loaded — drop CSV/Excel files anywhere on this page")
        self.source_label.setWordWrap(True)
        self.source_label.setProperty("role", "muted")
        self.source_path = QLineEdit()
        self.source_path.setReadOnly(True)
        self.source_path.setPlaceholderText("Source folder")
        self.source_path.setToolTip(
            "Folder containing the opened trace file(s). Select and copy this path if needed."
        )
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter traces")
        self.search.textChanged.connect(self.populate_traces)
        self.trace_list = QListWidget()
        self.trace_list.setObjectName("signalTraceList")
        self.trace_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.trace_list.itemSelectionChanged.connect(self.trace_selection_changed)

        controls = QWidget()
        controls.setMinimumWidth(290)
        controls.setMaximumWidth(340)
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(0, 0, 8, 0)
        self.open_button = QPushButton("Open or add CSV/Excel files")
        set_action_icon(self.open_button, "open")
        self.open_button.setToolTip("Choose trace files, or drag CSV/Excel files onto this page")
        self.open_button.clicked.connect(self.open_file)
        control_layout.addWidget(self.open_button)
        self.open_trace_location_button = QPushButton("Open file location")
        set_action_icon(self.open_trace_location_button, "open_folder")
        self.open_trace_location_button.setToolTip(
            "Open the folder containing the selected trace's source file."
        )
        self.open_trace_location_button.setEnabled(False)
        self.open_trace_location_button.clicked.connect(self.open_selected_trace_location)
        control_layout.addWidget(self.open_trace_location_button)

        selection_box = QGroupBox("Reference traces")
        selection_layout = QVBoxLayout(selection_box)
        baseline_row = QHBoxLayout()
        set_baseline = QPushButton("Set reference")
        set_action_icon(set_baseline, "apply")
        set_baseline.clicked.connect(self.set_baselines)
        clear_baseline = QPushButton("Clear references")
        set_action_icon(clear_baseline, "clear")
        clear_baseline.clicked.connect(self.clear_baselines)
        baseline_row.addWidget(set_baseline)
        baseline_row.addWidget(clear_baseline)
        selection_layout.addLayout(baseline_row)
        control_layout.addWidget(selection_box)

        settings_box = QGroupBox("Detection and processing")
        form = QFormLayout(settings_box)
        self.processing_target = QLabel("Select one or more traces")
        self.processing_target.setWordWrap(True)
        self.processing_target.setProperty("role", "muted")
        form.addRow("Processing target", self.processing_target)
        self.min_interval = self._double(0.01, 20, 0.10, 2)
        self.prominence_mode = QComboBox()
        self.prominence_mode.addItem("Relative (% range)", "percent_range")
        self.prominence_mode.addItem("Absolute", "absolute")
        self.prominence_mode.setToolTip(
            "Relative uses the processed signal's 99th–1st percentile range. Absolute uses "
            "the displayed signal unit directly (µN for force traces)."
        )
        self.prominence = self._double(0.1, 100, 20.0, 1)
        self.prominence.setSuffix(" %")
        self.prominence.setToolTip(
            "Percentage of the processed signal's robust range (99th percentile minus 1st percentile)."
        )
        self.prominence_absolute = self._double(0.001, 1000000, 10.0, 3)
        self.prominence_absolute.setToolTip(
            "Required prominence in the displayed y-axis unit after inversion, baseline correction, "
            "and smoothing. This is µN for force traces."
        )
        self.smoothing = self._double(0, 20, 0.15, 2)
        smoothing_tooltip = (
            "Savitzky–Golay smoothing-window duration. StimTrace converts this duration "
            "to an odd number of samples and fits a third-order local polynomial "
            "(second order when the window contains only three samples). Set this to 0 "
            "to disable smoothing. Peak detection and beat metrics use the smoothed "
            "signal; the original signal remains unchanged."
        )
        self.smoothing.setToolTip(smoothing_tooltip)
        self.start_time = QLineEdit()
        self.start_time.setPlaceholderText("Beginning")
        self.end_time = QLineEdit()
        self.end_time.setPlaceholderText("End")
        self.baseline_method = QComboBox()
        for label, value in BASELINE_METHODS:
            self.baseline_method.addItem(label, value)
        self.baseline_method.setToolTip(
            "Choose how the resting baseline is estimated before contraction analysis."
        )
        self.baseline_percentile = self._double(0, 50, 10.0, 1)
        self.baseline_percentile.setSuffix("%")
        self.baseline_percentile.setToolTip(
            "Lower signal values used to estimate rest. Ten percent is a robust starting value "
            "for traces with upward contraction peaks."
        )
        self.baseline_window = self._double(0.1, 600, 5.0, 1)
        self.baseline_window.setSuffix(" s")
        self.baseline_window.setToolTip(
            "Duration of the moving window. It should normally span at least five contraction cycles."
        )
        self.baseline_asls_smoothness = self._double(2, 10, 6.0, 1)
        self.baseline_asls_smoothness.setToolTip(
            "Logarithmic smoothness of the fitted baseline. Higher values produce a flatter, "
            "more slowly changing baseline."
        )
        for field in (
            self.min_interval,
            self.prominence_mode,
            self.prominence,
            self.prominence_absolute,
            self.smoothing,
            self.start_time,
            self.end_time,
            self.baseline_method,
            self.baseline_percentile,
            self.baseline_window,
            self.baseline_asls_smoothness,
        ):
            field.setMaximumWidth(150)
        form.addRow("Min. beat interval (s)", self.min_interval)
        form.addRow("Prominence mode", self.prominence_mode)
        form.addRow("Relative prominence", self.prominence)
        form.addRow("Absolute prominence", self.prominence_absolute)
        form.addRow("Smoothing window (s)", self.smoothing)
        form.labelForField(self.smoothing).setToolTip(smoothing_tooltip)
        form.addRow("Start time (s)", self.start_time)
        form.addRow("End time (s)", self.end_time)
        form.addRow("Baseline method", self.baseline_method)
        form.addRow("Baseline percentile", self.baseline_percentile)
        form.addRow("Baseline window (s)", self.baseline_window)
        form.addRow("ALS smoothness", self.baseline_asls_smoothness)
        self.baseline_parameter_rows = {
            self.baseline_percentile: form.labelForField(self.baseline_percentile),
            self.baseline_window: form.labelForField(self.baseline_window),
            self.baseline_asls_smoothness: form.labelForField(self.baseline_asls_smoothness),
        }
        self.prominence_parameter_rows = {
            self.prominence: form.labelForField(self.prominence),
            self.prominence_absolute: form.labelForField(self.prominence_absolute),
        }
        self.prominence_mode.currentIndexChanged.connect(self.update_prominence_controls)
        self.prominence_mode.currentIndexChanged.connect(self.refresh_selected_trace_plot)
        self.min_interval.valueChanged.connect(self.refresh_selected_trace_plot)
        self.prominence.valueChanged.connect(self.refresh_selected_trace_plot)
        self.prominence_absolute.valueChanged.connect(self.refresh_selected_trace_plot)
        self.smoothing.valueChanged.connect(self.refresh_selected_trace_plot)
        self.start_time.editingFinished.connect(self.refresh_selected_trace_plot)
        self.end_time.editingFinished.connect(self.refresh_selected_trace_plot)
        self.baseline_method.currentIndexChanged.connect(self.update_baseline_controls)
        self.baseline_method.currentIndexChanged.connect(self.store_selected_processing)
        self.baseline_percentile.valueChanged.connect(self.store_selected_processing)
        self.baseline_window.valueChanged.connect(self.store_selected_processing)
        self.baseline_asls_smoothness.valueChanged.connect(self.store_selected_processing)
        self.update_prominence_controls()
        self.update_baseline_controls()
        control_layout.addWidget(settings_box)
        self.analyze_button = QPushButton("Analyze selected traces")
        set_action_icon(self.analyze_button, "run")
        self.analyze_button.setProperty("role", "primary")
        self.analyze_button.clicked.connect(self.analyze_selected)
        control_layout.addWidget(self.analyze_button)
        export_row = QHBoxLayout()
        self.export_metrics_button = QPushButton("Export metrics")
        set_action_icon(self.export_metrics_button, "save")
        self.export_metrics_button.clicked.connect(self.export_metrics)
        export_row.addWidget(self.export_metrics_button)
        self.export_processed_button = QPushButton("Export processed traces")
        set_action_icon(self.export_processed_button, "save")
        self.export_processed_button.clicked.connect(self.export_processed)
        export_row.addWidget(self.export_processed_button)
        export_row.setAlignment(Qt.AlignHCenter)
        control_layout.addLayout(export_row)

        control_layout.addStretch(1)

        traces = QWidget()
        traces.setMinimumWidth(400)
        traces.setMaximumWidth(650)
        trace_layout = QVBoxLayout(traces)
        trace_layout.setContentsMargins(8, 0, 8, 0)
        trace_title = QLabel("Traces · invert and one-frame shift controls")
        trace_title.setStyleSheet("font-weight: 600;")
        trace_layout.addWidget(trace_title)
        trace_layout.addWidget(self.source_label)
        trace_layout.addWidget(self.source_path)
        trace_layout.addWidget(self.search)
        trace_selection_row = QHBoxLayout()
        self.select_all_traces_button = QPushButton("Select all")
        set_action_icon(self.select_all_traces_button, "apply")
        self.select_all_traces_button.setToolTip(
            "Select every trace currently visible in the filtered list."
        )
        self.select_all_traces_button.clicked.connect(self.select_all)
        self.clear_trace_selection_button = QPushButton("Clear selection")
        set_action_icon(self.clear_trace_selection_button, "clear")
        self.clear_trace_selection_button.clicked.connect(
            self.trace_list.clearSelection
        )
        self.align_first_peak_button = QPushButton("Align first peak")
        set_action_icon(self.align_first_peak_button, "apply")
        self.align_first_peak_button.setToolTip(
            "Align the first detected peak of all selected traces to the earliest selected peak."
        )
        self.align_first_peak_button.setEnabled(False)
        self.align_first_peak_button.clicked.connect(
            self.align_selected_traces_to_first_peak
        )
        self.watch_recording_button = QPushButton("Watch recording")
        set_action_icon(self.watch_recording_button, "run")
        self.watch_recording_button.setToolTip(
            "Select one trace with an available annotated recording."
        )
        self.watch_recording_button.setEnabled(False)
        self.watch_recording_button.clicked.connect(self.watch_selected_recording)
        self.remove_traces_button = QPushButton("Remove selected from list")
        set_action_icon(self.remove_traces_button, "delete")
        self.remove_traces_button.setProperty("role", "danger")
        self.remove_traces_button.setToolTip(
            "Remove selected traces from this analysis workspace. Source files are not deleted."
        )
        self.remove_traces_button.setEnabled(False)
        self.remove_traces_button.clicked.connect(self.remove_selected_traces)
        trace_selection_row.addWidget(self.select_all_traces_button)
        trace_selection_row.addWidget(self.clear_trace_selection_button)
        trace_selection_row.addWidget(self.align_first_peak_button)
        trace_selection_row.setAlignment(Qt.AlignHCenter)
        trace_layout.addLayout(trace_selection_row)
        trace_layout.addWidget(self.remove_traces_button)
        trace_layout.addWidget(self.watch_recording_button)
        self.toggle_auxiliary_traces_button = QPushButton(
            "Hide baselines/originals"
        )
        self.toggle_auxiliary_traces_button.setCheckable(True)
        self.toggle_auxiliary_traces_button.setToolTip(
            "Hide loaded baseline, original, and uncorrected columns from the trace list, "
            "and hide generated baseline/original curves in the plot."
        )
        self.toggle_auxiliary_traces_button.toggled.connect(
            self.toggle_auxiliary_traces
        )
        trace_layout.addWidget(self.toggle_auxiliary_traces_button)
        trace_layout.addWidget(self.trace_list, 1)
        self.remove_trace_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.Delete),
            self.trace_list,
        )
        self.remove_trace_shortcut.setContext(Qt.WidgetShortcut)
        self.remove_trace_shortcut.activated.connect(self.remove_selected_traces)

        tabs = QTabWidget()
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setToolTip(
            "Point at the plot and use the mouse wheel to zoom. Use Home in the toolbar to reset."
        )
        self.axes = self.figure.add_subplot(111)
        self.scroll_connection_id = self.canvas.mpl_connect(
            "scroll_event",
            self.zoom_with_mouse_wheel,
        )
        self.style_plot()
        plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        plot_legend = QVBoxLayout()
        plot_legend.setContentsMargins(0, 0, 0, 0)
        plot_legend.setSpacing(2)
        mouse_legend_row = QHBoxLayout()
        mouse_legend_row.addStretch(1)
        self.mouse_controls_label = QLabel(
            "Mouse controls  •  Click: select trace  •  Ctrl+click: add/remove  •  "
            "Drag: select range  •  Plot wheel: zoom  •  Home: reset view"
        )
        self.mouse_controls_label.setProperty("role", "muted")
        self.mouse_controls_label.setAlignment(Qt.AlignCenter)
        mouse_legend_row.addWidget(self.mouse_controls_label)
        mouse_legend_row.addStretch(1)
        plot_legend.addLayout(mouse_legend_row)
        trace_legend_row = QHBoxLayout()
        trace_legend_row.setSpacing(4)
        trace_legend_row.addStretch(1)
        trace_controls = QLabel("Selected trace controls")
        trace_controls.setProperty("role", "muted")
        trace_legend_row.addWidget(trace_controls)
        legend_control_size = max(16, trace_controls.fontMetrics().height() + 2)
        legend_icon_size = max(8, legend_control_size - 6)
        self.legend_polarity_button = QPushButton()
        self.legend_polarity_button.setObjectName("selectedTracePolarityButton")
        set_polarity_icon(self.legend_polarity_button)
        self.legend_polarity_button.setProperty("role", "traceControl")
        self.legend_polarity_button.setFixedSize(legend_control_size, legend_control_size)
        self.legend_polarity_button.setIconSize(QSize(legend_icon_size, legend_icon_size))
        self.legend_polarity_button.setStyleSheet(
            "QPushButton { padding: 0; margin: 0; font-size: 14px; font-weight: 600; }"
        )
        self.legend_polarity_button.setToolTip("Toggle polarity for all selected traces")
        self.legend_polarity_button.clicked.connect(self.toggle_selected_trace_polarity)
        trace_legend_row.addWidget(self.legend_polarity_button, 0, Qt.AlignCenter)
        self.legend_shift_earlier_button = QPushButton()
        self.legend_shift_earlier_button.setObjectName("selectedTraceShiftEarlierButton")
        self.legend_shift_earlier_button.setProperty("role", "traceControl")
        self.legend_shift_earlier_button.setFixedSize(legend_control_size, legend_control_size)
        self.legend_shift_earlier_button.setStyleSheet(
            "QPushButton { padding: 0; margin: 0; }"
        )
        set_action_icon(self.legend_shift_earlier_button, "previous")
        self.legend_shift_earlier_button.setIconSize(QSize(legend_icon_size, legend_icon_size))
        self.legend_shift_earlier_button.setToolTip("Move all selected traces one frame earlier")
        self.legend_shift_earlier_button.clicked.connect(
            lambda: self.shift_selected_traces_by_frames(-1)
        )
        trace_legend_row.addWidget(self.legend_shift_earlier_button, 0, Qt.AlignCenter)
        self.legend_shift_later_button = QPushButton()
        self.legend_shift_later_button.setObjectName("selectedTraceShiftLaterButton")
        self.legend_shift_later_button.setProperty("role", "traceControl")
        self.legend_shift_later_button.setFixedSize(legend_control_size, legend_control_size)
        self.legend_shift_later_button.setStyleSheet(
            "QPushButton { padding: 0; margin: 0; }"
        )
        set_action_icon(self.legend_shift_later_button, "next")
        self.legend_shift_later_button.setIconSize(QSize(legend_icon_size, legend_icon_size))
        self.legend_shift_later_button.setToolTip("Move all selected traces one frame later")
        self.legend_shift_later_button.clicked.connect(
            lambda: self.shift_selected_traces_by_frames(1)
        )
        trace_legend_row.addWidget(self.legend_shift_later_button, 0, Qt.AlignCenter)
        trace_legend_row.addStretch(1)
        plot_legend.addLayout(trace_legend_row)
        plot_layout.addLayout(plot_legend)
        tabs.addTab(plot_tab, "Signals and QC")
        self.results_table = FrozenFirstColumnTable()
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSortingEnabled(True)
        tabs.addTab(self.results_table, "Summary")
        splitter.addWidget(controls)
        splitter.addWidget(traces)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([320, 500, 1000])
        self.status_label = QLabel("Open generated trace CSV or Excel files.")
        self.status_label.setProperty("role", "muted")
        page_layout.addWidget(self.status_label)
        self.busy_progress = QProgressBar()
        self.busy_progress.setRange(0, 0)
        self.busy_progress.setTextVisible(False)
        self.busy_progress.setFixedHeight(5)
        self.busy_progress.setVisible(False)
        page_layout.addWidget(self.busy_progress)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_operation_busy(self, busy: bool, message: str = "") -> None:
        """Protect mutable analysis state while a background operation is active."""
        self._operation_busy = busy
        self.busy_progress.setVisible(busy)
        for control in (
            self.open_button,
            self.analyze_button,
            self.export_metrics_button,
            self.export_processed_button,
            self.trace_list,
            self.remove_traces_button,
            self.select_all_traces_button,
            self.clear_trace_selection_button,
            self.align_first_peak_button,
            self.open_trace_location_button,
            self.watch_recording_button,
            self.toggle_auxiliary_traces_button,
        ):
            control.setEnabled(not busy)
        self.update_trace_actions()
        if message:
            self.set_status(message)

    def start_background_operation(
        self,
        name: str,
        function,
        on_success,
        failure_title: str,
    ) -> bool:
        if self.operation_thread is not None:
            self.set_status("Another signal-analysis operation is still running.")
            return False
        thread = BackgroundFunctionThread(function, self.window())
        register_thread = getattr(self.window(), "register_thread", None)
        if callable(register_thread):
            register_thread(thread, f"signal-analysis-{name}")
        thread.succeeded.connect(on_success, Qt.QueuedConnection)
        thread.failed.connect(
            lambda message: self.background_operation_failed(failure_title, message),
            Qt.QueuedConnection,
        )
        thread.finished.connect(
            lambda worker=thread: self.background_operation_finished(worker),
            Qt.QueuedConnection,
        )
        self.operation_thread = thread
        self.set_operation_busy(True, f"{name.replace('-', ' ').title()} in progress...")
        thread.start()
        return True

    def background_operation_failed(self, title: str, message: str) -> None:
        LOGGER.error("%s: %s", title, message)
        self.set_status(f"{title}: {message}")
        QMessageBox.critical(self, title, message)

    def background_operation_finished(self, thread: QThread) -> None:
        if self.operation_thread is thread:
            self.operation_thread = None
            self.set_operation_busy(False)

    def show_help(self):
        QMessageBox.information(
            self,
            "Signal processing help",
            "1. Open one or more pillar-displacement CSV files or combined force-results "
            "workbooks generated by segmentation. Multiple files are aligned by time and their "
            "trace names are prefixed with the source filename. To load files from another folder, "
            "use Open or add again and choose Add traces when prompted. For workbooks, select the "
            "sheet containing the traces.\n\n"
            "2. Select one or more signals from the list. Use the plot toolbar to inspect them. "
            "Use Remove selected or press Delete while the trace list is focused to remove "
            "signals from the current workspace; the source CSV/Excel files are never changed. "
            "When one generated trace is selected, use Watch recording to open its annotated "
            "video in a separate scrollable player. "
            "Optionally select a control signal and choose Set as reference trace. References "
            "are matched to analyzed traces by specimen name (for example, B-1). Choose no more "
            "than one reference for each specimen. A reference is analyzed normally and is used "
            "only to calculate beat rate, amplitude, time-to-peak, and CTD90 as a percentage of "
            "the matching reference. It is not subtracted from the signal.\n\n"
            "3. Adjust the beat interval, peak prominence, smoothing, and time range as "
            "needed. Use the circular-arrow button beside a trace to invert only that trace. Use "
            "the previous/next arrow buttons in the same row to move that recording exactly one "
            "frame earlier or later. Baseline correction is saved separately for the currently "
            "selected traces; selecting another trace restores its settings. Choose a "
            "relative prominence to threshold peaks by a percentage of the processed signal's "
            "99th–1st percentile range, or absolute prominence to require a fixed vertical "
            "prominence in the displayed unit (µN for force traces). Choose a "
            "baseline correction method: Constant percentile "
            "shifts the complete trace by one resting value; Rolling percentile follows gradual "
            "drift within a moving window; Asymmetric least squares fits a smooth baseline while "
            "ignoring upward contraction peaks.\n\n"
            "4. Analyze selected traces to calculate contraction and relaxation metrics and "
            "review the original signal, estimated baseline, corrected signal, detected peaks, "
            "and the Summary tab. Point at the plot and use the mouse wheel to zoom around that "
            "location; use the toolbar Home button to restore the complete view.\n\n"
            "5. Export metrics for the numerical summary or export processed traces for the "
            "adjusted original and final processed data. Exports contain only traces "
            "currently selected in the list. Use Hide baselines/originals to remove auxiliary "
            "columns from the list and auxiliary curves from the plot. Processed-trace exports "
            "retain the shifted time positions, and both workbooks save each trace's frame offset. "
            "Both workbooks include a "
            "Reference traces sheet stating whether references were selected, which traces were "
            "used, and which analyzed traces they matched.\n\n"
            "Baseline correction and reference traces are different: baseline correction removes "
            "the resting offset or drift within each signal; a reference trace provides a control "
            "value for between-trace percentage comparisons.",
        )

    @staticmethod
    def _double(minimum, maximum, value, decimals):
        field = QDoubleSpinBox()
        field.setRange(minimum, maximum)
        field.setDecimals(decimals)
        field.setValue(value)
        return field

    def update_baseline_controls(self) -> None:
        method = self.baseline_method.currentData()
        visible_fields = {
            "constant_percentile": {self.baseline_percentile},
            "rolling_percentile": {self.baseline_percentile, self.baseline_window},
            "asymmetric_least_squares": {self.baseline_asls_smoothness},
        }.get(method, set())
        for field, label in self.baseline_parameter_rows.items():
            field.setVisible(field in visible_fields)
            if label:
                label.setVisible(field in visible_fields)

    def update_prominence_controls(self) -> None:
        absolute = self.prominence_mode.currentData() == "absolute"
        for field, label in self.prominence_parameter_rows.items():
            visible = field is self.prominence_absolute if absolute else field is self.prominence
            field.setVisible(visible)
            if label:
                label.setVisible(visible)

    def processing_from_controls(self) -> TraceProcessingSettings:
        method = self.baseline_method.currentData()
        return TraceProcessingSettings(
            invert=False,
            baseline_method=str(method) if method is not None else "none",
            baseline_percentile=self.baseline_percentile.value(),
            baseline_window_s=self.baseline_window.value(),
            baseline_asls_log10=self.baseline_asls_smoothness.value(),
        )

    def settings_for_column(
        self,
        column: str,
        shared: AnalysisSettings | None = None,
    ) -> AnalysisSettings:
        shared = shared or self.settings()
        processing = self.trace_processing.get(column, TraceProcessingSettings())
        return replace(
            shared,
            invert=processing.invert,
            baseline_method=processing.baseline_method,
            baseline_percentile=processing.baseline_percentile,
            baseline_window_s=processing.baseline_window_s,
            baseline_asls_log10=processing.baseline_asls_log10,
            time_shift_frames=processing.time_shift_frames,
        )

    def store_selected_processing(self, *_args) -> None:
        if self._loading_processing_controls:
            return
        columns = self.selected_columns()
        if not columns:
            return
        processing = self.processing_from_controls()
        for column in columns:
            current = self.trace_processing.get(column, TraceProcessingSettings())
            self.trace_processing[column] = replace(
                processing,
                invert=current.invert,
                time_shift_frames=current.time_shift_frames,
            )
            self.plot_cache.pop(column, None)
        self.processing_target.setText(
            f"Applied separately to {len(columns)} selected trace"
            f"{'s' if len(columns) != 1 else ''}"
        )
        self.schedule_plot(preserve_view=True)

    def refresh_selected_trace_plot(self, *_args) -> None:
        """Refresh changed processing while retaining the analyst's current graph view."""
        if self._loading_processing_controls:
            return
        for column in self.selected_columns():
            self.plot_cache.pop(column, None)
        self.schedule_plot(preserve_view=True)

    def load_selected_processing(self) -> None:
        columns = self.selected_columns()
        if not columns:
            self.processing_target.setText("Select one or more traces")
            return
        profiles = [
            self.trace_processing.get(column, TraceProcessingSettings())
            for column in columns
        ]
        profile = profiles[0]
        # Inversion is controlled independently by the symbol button in each list row.
        # Do not report otherwise-identical shared controls as mixed merely because
        # the selected traces have different inversion states.
        mixed = any(
            replace(
                candidate,
                invert=profile.invert,
                time_shift_frames=profile.time_shift_frames,
            ) != profile
            for candidate in profiles[1:]
        )
        self._loading_processing_controls = True
        try:
            index = self.baseline_method.findData(profile.baseline_method)
            self.baseline_method.setCurrentIndex(max(0, index))
            self.baseline_percentile.setValue(profile.baseline_percentile)
            self.baseline_window.setValue(profile.baseline_window_s)
            self.baseline_asls_smoothness.setValue(profile.baseline_asls_log10)
        finally:
            self._loading_processing_controls = False
        self.update_baseline_controls()
        if mixed:
            self.processing_target.setText(
                f"{len(columns)} traces selected with different settings; controls show the "
                "first trace. A change applies the displayed settings to all selected traces."
            )
        else:
            self.processing_target.setText(
                f"{len(columns)} selected trace{'s' if len(columns) != 1 else ''}"
            )

    def update_trace_row_processing(self, column: str) -> None:
        controls = self.trace_row_controls.get(column)
        if controls is None:
            return
        invert_button, shift_label = controls
        processing = self.trace_processing.get(column, TraceProcessingSettings())
        invert_button.blockSignals(True)
        try:
            invert_button.setChecked(processing.invert)
        finally:
            invert_button.blockSignals(False)
        shift_label.setText(f"{processing.time_shift_frames:+d} f")

    def shift_trace_by_frames(self, column: str, frames: int) -> None:
        maximum = max(1, (len(self.data) - 1) if self.data is not None else 1)
        current = self.trace_processing.get(column, TraceProcessingSettings())
        shifted = max(-maximum, min(maximum, current.time_shift_frames + int(frames)))
        if shifted == current.time_shift_frames:
            return
        self.trace_processing[column] = replace(current, time_shift_frames=shifted)
        if column in getattr(self, "_first_peak_alignment_restore", {}):
            self._first_peak_alignment_restore.clear()
        self.plot_cache.pop(column, None)
        self.update_trace_row_processing(column)
        self.schedule_plot(preserve_view=True)
        self.set_status(
            f"Moved {column} one frame {'later' if frames > 0 else 'earlier'} "
            f"(current shift: {shifted:+d} frames). Analyze again to refresh metrics."
        )

    def toggle_selected_trace_polarity(self) -> None:
        selected = self.selected_columns()
        if not selected:
            return
        enable_inversion = not all(
            self.trace_processing.get(column, TraceProcessingSettings()).invert
            for column in selected
        )
        for column in selected:
            current = self.trace_processing.get(column, TraceProcessingSettings())
            self.trace_processing[column] = replace(current, invert=enable_inversion)
            self.plot_cache.pop(column, None)
            self.update_trace_row_processing(column)
        self.schedule_plot(preserve_view=True)
        self.set_status(
            f"Polarity {'inverted' if enable_inversion else 'restored'} for "
            f"{len(selected)} selected trace{'s' if len(selected) != 1 else ''}."
        )

    def shift_selected_traces_by_frames(self, frames: int) -> None:
        selected = self.selected_columns()
        if not selected:
            return
        maximum = max(1, (len(self.data) - 1) if self.data is not None else 1)
        changed = 0
        for column in selected:
            current = self.trace_processing.get(column, TraceProcessingSettings())
            shifted = max(-maximum, min(maximum, current.time_shift_frames + int(frames)))
            if shifted == current.time_shift_frames:
                continue
            self.trace_processing[column] = replace(current, time_shift_frames=shifted)
            self.plot_cache.pop(column, None)
            self.update_trace_row_processing(column)
            changed += 1
        if not changed:
            return
        if set(selected) & set(getattr(self, "_first_peak_alignment_restore", {})):
            self._first_peak_alignment_restore.clear()
            getattr(self, "update_first_peak_alignment_action", lambda: None)()
        self.schedule_plot(preserve_view=True)
        self.set_status(
            f"Moved {changed} selected trace{'s' if changed != 1 else ''} one frame "
            f"{'later' if frames > 0 else 'earlier'}. Analyze again to refresh metrics."
        )

    def align_selected_traces_to_first_peak(self) -> None:
        """Toggle first-peak alignment without ever changing source time data."""
        if self.data is None:
            return
        columns = self.selected_columns()
        if len(columns) < 2:
            QMessageBox.warning(
                self,
                "Select more traces",
                "Select at least two traces to align their first detected peaks.",
            )
            return
        restore_offsets = getattr(self, "_first_peak_alignment_restore", {})
        if restore_offsets and set(restore_offsets) == set(columns):
            changed = 0
            for column, original_shift in restore_offsets.items():
                current = self.trace_processing.get(column, TraceProcessingSettings())
                if current.time_shift_frames == original_shift:
                    continue
                self.trace_processing[column] = replace(
                    current, time_shift_frames=original_shift
                )
                self.plot_cache.pop(column, None)
                self.update_trace_row_processing(column)
                changed += 1
            self._first_peak_alignment_restore = {}
            getattr(self, "update_first_peak_alignment_action", lambda: None)()
            if changed:
                self.schedule_plot(preserve_view=True)
            self.set_status(
                "Restored the timing offsets that were present before first-peak alignment."
            )
            return
        # A different selection starts a new alignment operation; it must not
        # accidentally restore offsets for another group of traces.
        self._first_peak_alignment_restore = {}
        try:
            shared_settings = self.settings()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid processing settings", str(error))
            return

        time_values = pd.to_numeric(
            self.data[self.time_column], errors="coerce"
        ).to_numpy(float)
        first_peaks: dict[str, tuple[float, float]] = {}
        errors: list[str] = []
        for column in columns:
            values = pd.to_numeric(self.data[column], errors="coerce").to_numpy(float)
            try:
                first_peaks[column] = first_detected_peak_time(
                    time_values,
                    values,
                    self.settings_for_column(column, shared_settings),
                )
            except ValueError as error:
                errors.append(f"{column}: {error}")
        if errors:
            QMessageBox.warning(
                self,
                "Could not align first peaks",
                "No alignment was applied. Each selected trace needs a detectable peak with the "
                "current polarity and detection settings:\n\n" + "\n".join(errors),
            )
            return

        target_time = min(time for time, _dt in first_peaks.values())
        maximum = max(1, len(self.data) - 1)
        changed = 0
        original_offsets = {
            column: self.trace_processing.get(
                column, TraceProcessingSettings()
            ).time_shift_frames
            for column in columns
        }
        for column, (peak_time, dt) in first_peaks.items():
            current = self.trace_processing.get(column, TraceProcessingSettings())
            frame_delta = int(round((target_time - peak_time) / dt))
            shifted = max(
                -maximum,
                min(maximum, current.time_shift_frames + frame_delta),
            )
            if shifted == current.time_shift_frames:
                continue
            self.trace_processing[column] = replace(current, time_shift_frames=shifted)
            self.plot_cache.pop(column, None)
            self.update_trace_row_processing(column)
            changed += 1
        if not changed:
            self.set_status("Selected first peaks are already aligned to the nearest frame.")
            return
        self._first_peak_alignment_restore = original_offsets
        getattr(self, "update_first_peak_alignment_action", lambda: None)()
        self.schedule_plot(preserve_view=True)
        self.set_status(
            f"Aligned the first detected peak of {len(columns)} selected traces. "
            "Analyze again to refresh metrics."
        )

    def update_first_peak_alignment_action(self) -> None:
        """Present one clear align/restore action for the current selection."""
        button = getattr(self, "align_first_peak_button", None)
        if button is None:
            return
        selected = set(self.selected_columns())
        can_restore = bool(self._first_peak_alignment_restore) and selected == set(
            self._first_peak_alignment_restore
        )
        button.setText("Restore timing" if can_restore else "Align first peak")
        button.setToolTip(
            "Restore the timing offsets present before automatic peak alignment."
            if can_restore
            else "Align the first detected peak of all selected traces to the earliest selected peak."
        )

    def settings(self) -> AnalysisSettings:
        def optional(field: QLineEdit):
            text = field.text().strip()
            return float(text) if text else None

        settings = AnalysisSettings(
            min_interval_s=self.min_interval.value(),
            prominence_pct=self.prominence.value(),
            prominence_mode=str(self.prominence_mode.currentData()),
            prominence_absolute=self.prominence_absolute.value(),
            smoothing_s=self.smoothing.value(),
            start_s=optional(self.start_time),
            end_s=optional(self.end_time),
            invert=False,
            baseline_method=str(self.baseline_method.currentData()),
            baseline_percentile=self.baseline_percentile.value(),
            baseline_window_s=self.baseline_window.value(),
            baseline_asls_log10=self.baseline_asls_smoothness.value(),
        )
        if settings.start_s is not None and settings.end_s is not None and settings.start_s >= settings.end_s:
            raise ValueError("Analysis start must be before analysis end.")
        return settings

    def open_file(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Open one or more signal-trace files",
            "",
            "Data files (*.csv *.xlsx *.xls)",
        )
        if not filenames:
            return
        self.offer_trace_files([Path(filename) for filename in filenames])

    @staticmethod
    def trace_paths_from_urls(urls) -> tuple[list[Path], list[Path]]:
        """Split dropped local files into supported traces and rejected paths."""
        selected: list[Path] = []
        rejected: list[Path] = []
        seen: set[str] = set()
        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            identity = str(path.resolve()).casefold()
            if identity in seen:
                continue
            seen.add(identity)
            target = selected if path.suffix.lower() in {".csv", ".xlsx", ".xls"} else rejected
            target.append(path)
        return selected, rejected

    def dragEnterEvent(self, event) -> None:
        selected, _ = self.trace_paths_from_urls(event.mimeData().urls())
        if selected and not self._operation_busy:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        selected, _ = self.trace_paths_from_urls(event.mimeData().urls())
        if selected and not self._operation_busy:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        selected_paths, rejected = self.trace_paths_from_urls(event.mimeData().urls())
        if not selected_paths:
            event.ignore()
            return
        event.acceptProposedAction()
        if rejected:
            names = "\n".join(f"• {path.name}" for path in rejected[:10])
            QMessageBox.warning(
                self,
                "Unsupported files skipped",
                "StimTrace can open CSV and Excel trace files. These dropped files were skipped:\n\n"
                + names,
            )
        self.offer_trace_files(selected_paths)

    def offer_trace_files(self, selected_paths: list[Path]) -> None:
        """Apply the common add-or-replace flow to selected or dropped files."""
        if self._operation_busy:
            self.set_status("Finish the current signal-analysis operation before loading new traces.")
            return
        selected_paths = [Path(path) for path in selected_paths]
        if not selected_paths:
            return
        # A source path is meaningful only while at least one loaded trace remains.
        # Repair stale entries left by older sessions before describing the workspace.
        active_sources = {path.resolve() for path in self.trace_source_paths.values()}
        if active_sources:
            self.paths = [path for path in self.paths if path.resolve() in active_sources]
            self.path = self.paths[0] if self.paths else None
        if self.paths and self.signal_columns():
            question = QMessageBox(self)
            question.setIcon(QMessageBox.Question)
            question.setWindowTitle("Add or replace traces?")
            question.setText(
                f"{len(self.paths)} source file(s) ({len(self.signal_columns())} visible trace(s)) "
                f"are already loaded. Add the newly selected "
                f"{len(selected_paths)} file(s), or replace the current traces?\n\n"
                "Changing the loaded files clears the current analysis results and reference selections."
            )
            add_button = question.addButton("Add traces", QMessageBox.AcceptRole)
            replace_button = question.addButton("Replace traces", QMessageBox.DestructiveRole)
            question.addButton(QMessageBox.Cancel)
            question.setDefaultButton(add_button)
            question.exec()
            clicked = question.clickedButton()
            if clicked is add_button:
                combined: list[Path] = []
                seen: set[str] = set()
                for path in [*self.paths, *selected_paths]:
                    identity = str(path.resolve()).casefold()
                    if identity not in seen:
                        seen.add(identity)
                        combined.append(path)
                if len(combined) == len(self.paths):
                    self.set_status("The selected files are already loaded.")
                    return
                selected_paths = combined
            elif clicked is not replace_button:
                return
        async_loader = getattr(self.window(), "load_signal_files_async", None)
        if callable(async_loader):
            async_loader(selected_paths)
        else:
            self.load_files(selected_paths)

    def _read_trace_file(
        self,
        path: Path,
        sheet_name: str | None = None,
        allow_dialog: bool = True,
    ) -> tuple[pd.DataFrame, str] | None:
        if not path.is_file():
            raise FileNotFoundError(
                "The trace file does not exist at its saved location. It may have been moved "
                f"or deleted:\n{path}"
            )
        cache_key = str(path.resolve()).casefold()
        cached = self._preloaded_trace_files.pop(cache_key, None)
        if cached is not None:
            return cached
        if path.suffix.lower() == ".csv":
            data = pd.read_csv(path, sep=None, engine="python")
        else:
            with pd.ExcelFile(path) as book:
                sheet = sheet_name
                if sheet is None:
                    sheet = next(
                        (
                            preferred
                            for preferred in ("Force traces", "Processed traces")
                            if preferred in book.sheet_names
                        ),
                        None,
                    )
                if sheet is None and len(book.sheet_names) == 1:
                    sheet = book.sheet_names[0]
                if sheet is None:
                    if not allow_dialog:
                        raise ValueError(
                            "This workbook has multiple worksheets and no recognized signal sheet "
                            "('Force traces' or 'Processed traces'). Export the desired worksheet "
                            "as CSV, then open that CSV."
                        )
                    dialog = SheetDialog(book.sheet_names, self)
                    dialog.setWindowTitle(f"Choose worksheet - {path.name}")
                    if dialog.exec() != QDialog.Accepted:
                        return None
                    sheet = dialog.selected_sheet()
                data = pd.read_excel(book, sheet_name=sheet)
            if (
                path.name.lower() == "kalman_benchmark_all_traces.xlsx"
                and sheet == "Force traces"
            ):
                # Preserve compatibility with benchmark jobs created before the
                # worker began exporting a companion CSV.
                try:
                    data.to_csv(path.with_suffix(".csv"), index=False)
                except OSError:
                    pass
        data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")
        data.columns = [str(column).strip() for column in data.columns]
        if data.shape[1] < 2:
            raise ValueError("Expected one time column and at least one signal column.")
        candidates = [column for column in data.columns if re.search(r"time|sec", column, re.I)]
        return data, candidates[0] if candidates else data.columns[0]

    def cache_preloaded_trace_file(
        self,
        path: Path,
        loaded: tuple[pd.DataFrame, str],
    ) -> None:
        self._preloaded_trace_files[str(path.resolve()).casefold()] = loaded

    @staticmethod
    def _source_prefix(path: Path, duplicate_stems: set[str]) -> str:
        if path.stem.lower() in duplicate_stems:
            return f"{path.parent.name}_{path.stem}"
        return path.stem

    @staticmethod
    def trace_source_identity(path: Path) -> tuple[str, str]:
        """Identify StimTrace-generated files while accepting legacy filenames."""
        stem = path.stem
        lower = stem.lower()
        if lower in {"stimtrace_force_traces", "combined_force_results"}:
            return "StimTrace", ""
        if lower == "point_tracking_force_traces":
            return "Point Tracking", ""
        for suffix in (
            "_manual_ellipse_validation", "_point_tracking_validation",
            "_manual_ellipse_trace", "_manual_ground_truth_trace", "_dl_raw_ellipse_trace", "_point_tracking_trace",
        ):
            if lower.endswith(suffix):
                return "Validation", stem[: -len(suffix)]
        for suffix in ("_stimtrace_tracking", "_pillar_displacement"):
            if lower.endswith(suffix):
                return "StimTrace", stem[: -len(suffix)]
        for suffix in (
            "_point_tracking_force",
            "_point_tracking_displacement",
            "_tracked_distances",
        ):
            if lower.endswith(suffix):
                return "Point Tracking", stem[: -len(suffix)]
        return "", ""

    @classmethod
    def trace_display_name(
        cls,
        path: Path,
        column: str,
        *,
        multiple: bool,
        duplicate_stems: set[str],
    ) -> str:
        method, recording = cls.trace_source_identity(path)
        if method:
            signal_name = f"{recording}_{column}" if recording else column
            return f"{method} | {signal_name}"
        if multiple:
            return f"{cls._source_prefix(path, duplicate_stems)} | {column}"
        return column

    def load_files(
        self,
        paths: list[Path],
        sheet_names: dict[Path, str | None] | None = None,
    ) -> None:
        try:
            paths = [Path(path) for path in paths]
            if not paths:
                return
            stem_counts: dict[str, int] = {}
            for path in paths:
                key = path.stem.lower()
                stem_counts[key] = stem_counts.get(key, 0) + 1
            duplicate_stems = {stem for stem, count in stem_counts.items() if count > 1}
            frames: list[pd.DataFrame] = []
            loaded_paths: list[Path] = []
            errors: list[str] = []
            first_time_column = "time_s"
            multiple = len(paths) > 1
            used_names: set[str] = set()
            source_paths: dict[str, Path] = {}
            for path in paths:
                try:
                    loaded = self._read_trace_file(path, (sheet_names or {}).get(path))
                    if loaded is None:
                        continue
                    source_data, source_time_column = loaded
                    if not frames:
                        first_time_column = source_time_column
                    time_values = pd.to_numeric(
                        source_data[source_time_column], errors="coerce"
                    )
                    valid_time = time_values.notna()
                    if valid_time.sum() < 2:
                        raise ValueError("The time column contains fewer than two numeric values.")
                    signal_columns = [
                        column for column in source_data.columns
                        if column != source_time_column
                        and pd.to_numeric(source_data[column], errors="coerce").notna().sum() >= 5
                    ]
                    if not signal_columns:
                        raise ValueError("No numeric trace with at least five samples was found.")
                    frame = pd.DataFrame(index=time_values[valid_time].to_numpy(float))
                    for column in signal_columns:
                        output_name = self.trace_display_name(
                            path,
                            column,
                            multiple=multiple,
                            duplicate_stems=duplicate_stems,
                        )
                        original_name = output_name
                        suffix = 2
                        while output_name.lower() in used_names:
                            output_name = f"{original_name} ({suffix})"
                            suffix += 1
                        used_names.add(output_name.lower())
                        source_paths[output_name] = path
                        frame[output_name] = pd.to_numeric(
                            source_data.loc[valid_time, column], errors="coerce"
                        ).to_numpy(float)
                    frames.append(frame.groupby(level=0, sort=True).mean(numeric_only=True))
                    loaded_paths.append(path)
                except Exception as error:
                    errors.append(f"{path.name}: {error}")
            if not frames:
                details = "\n".join(errors[:10])
                raise ValueError(f"No trace files could be loaded.\n{details}")
            self.time_column = "time_s" if multiple else first_time_column
            data = pd.concat(frames, axis=1, join="outer").sort_index()
            data.index.name = self.time_column
            data = data.reset_index()
            self.data, self.path, self.paths = data, loaded_paths[0], loaded_paths
            self.trace_source_paths = source_paths
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
            self.trace_list.blockSignals(True)
            self.trace_list.clear()
            self.trace_list.blockSignals(False)
            self.visible_columns = []
            self.plot_cache = {}
            self.summary = pd.DataFrame()
            self.beats = pd.DataFrame()
            self.baseline_columns.clear()
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.axes.clear()
            self.style_plot()
            self.populate_traces()
            available_columns = set(self.signal_columns())
            self.trace_processing = {
                column: processing
                for column, processing in self.trace_processing.items()
                if column in available_columns
            }
            preferred_names = ["force_un", "xy_combo_norm", "xy_combo"]
            preferred_row = next(
                (
                    row
                    for preferred_name in preferred_names
                    for row, column in enumerate(self.visible_columns)
                    if column.lower() == preferred_name
                    or (
                        preferred_name == "force_un"
                        and column.lower().endswith("_force_un")
                    )
                ),
                0,
            )
            self.trace_list.setCurrentRow(preferred_row)
            if len(loaded_paths) == 1:
                source_text = (
                    f"{loaded_paths[0].name}\n{len(data)} samples, "
                    f"time column: {self.time_column}"
                )
            else:
                source_text = (
                    f"{len(loaded_paths)} files loaded\n{len(data)} combined time points"
                )
            self.source_label.setText(source_text)
            self.source_path.setText(self.loaded_source_folder_text(loaded_paths))
            self.set_status(
                f"Loaded {len(self.signal_columns())} numeric traces from "
                f"{len(loaded_paths)} file(s)."
            )
            if errors:
                QMessageBox.warning(self, "Some files were skipped", "\n".join(errors[:10]))
        except Exception as error:
            LOGGER.exception("Could not load trace files")
            QMessageBox.critical(self, "Could not open traces", str(error))

    def signal_columns(self) -> list[str]:
        if self.data is None:
            return []
        return [
            column for column in self.data.columns
            if column != self.time_column
            and pd.to_numeric(self.data[column], errors="coerce").notna().sum() >= 5
        ]

    @staticmethod
    def loaded_source_folder_text(paths: list[Path]) -> str:
        """Return a useful source-folder label for one or more loaded trace files."""
        folders = {path.parent.resolve() for path in paths}
        if len(folders) == 1:
            return str(next(iter(folders)))
        if not folders:
            return ""
        return "Multiple source folders — select a trace and use Open file location"

    def populate_traces(self):
        if self.data is None:
            return
        query = self.search.text().strip().lower()
        selected = set(self.selected_columns())
        self.visible_columns = [
            column
            for column in self.signal_columns()
            if query in column.lower()
            and not (
                self.auxiliary_traces_hidden
                and is_auxiliary_trace_column(column)
            )
        ]
        self.trace_list.blockSignals(True)
        self.trace_list.clear()
        self.trace_row_controls.clear()
        self._trace_drag_anchor_item = None
        self._trace_drag_base_rows.clear()
        self._trace_drag_additive = False
        self._trace_drag_selecting = True
        trace_control_size = max(16, self.trace_list.fontMetrics().height() + 2)
        trace_icon_size = max(8, trace_control_size - 6)
        trace_row_height = trace_control_size + 2
        for column in self.visible_columns:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, column)
            item.setToolTip(column)
            self.trace_list.addItem(item)
            row_widget = TraceListRow()
            row_widget.setMinimumHeight(trace_row_height)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(2, 1, 2, 1)
            row_layout.setSpacing(3)
            invert_button = QPushButton()
            invert_button.setObjectName("traceInvertButton")
            invert_button.setCheckable(True)
            set_polarity_icon(invert_button)
            invert_button.setChecked(
                self.trace_processing.get(column, TraceProcessingSettings()).invert
            )
            invert_button.setProperty("role", "traceControl")
            invert_button.setFixedSize(trace_control_size, trace_control_size)
            invert_button.setIconSize(QSize(trace_icon_size, trace_icon_size))
            invert_button.setLayoutDirection(Qt.LeftToRight)
            invert_button.setStyleSheet(
                "QPushButton { padding: 0; margin: 0; font-size: 14px; font-weight: 600; }"
            )
            invert_button.setToolTip("Toggle this trace's signal polarity")
            earlier_button = QPushButton()
            earlier_button.setObjectName("traceShiftEarlierButton")
            earlier_button.setProperty("role", "traceControl")
            set_action_icon(earlier_button, "previous")
            earlier_button.setFixedSize(trace_control_size, trace_control_size)
            earlier_button.setStyleSheet("QPushButton { padding: 0; margin: 0; }")
            earlier_button.setIconSize(QSize(trace_icon_size, trace_icon_size))
            earlier_button.setToolTip("Move this trace one frame earlier")
            later_button = QPushButton()
            later_button.setObjectName("traceShiftLaterButton")
            later_button.setProperty("role", "traceControl")
            set_action_icon(later_button, "next")
            later_button.setFixedSize(trace_control_size, trace_control_size)
            later_button.setStyleSheet("QPushButton { padding: 0; margin: 0; }")
            later_button.setIconSize(QSize(trace_icon_size, trace_icon_size))
            later_button.setToolTip("Move this trace one frame later")
            trace_label = QLabel(
                f"[REF] {column}" if column in self.baseline_columns else column
            )
            trace_label.setObjectName("traceNameLabel")
            trace_label.setToolTip(column)
            trace_label.setAttribute(Qt.WA_TransparentForMouseEvents)
            shift_label = QLabel()
            shift_label.setObjectName("traceShiftLabel")
            shift_label.setAlignment(Qt.AlignCenter)
            shift_label.setFixedWidth(46)
            shift_label.setToolTip("Current time alignment in frames")
            row_layout.addWidget(invert_button, 0, Qt.AlignVCenter)
            row_layout.addWidget(earlier_button, 0, Qt.AlignVCenter)
            row_layout.addWidget(later_button, 0, Qt.AlignVCenter)
            row_layout.addWidget(trace_label, 1)
            row_layout.addWidget(shift_label)
            row_widget.row_clicked.connect(
                lambda modifiers, target=item: self.select_trace_row(target, modifiers)
            )
            row_widget.row_dragged.connect(self.extend_trace_drag)
            invert_button.toggled.connect(
                lambda inverted, target=column: self.set_trace_inverted(target, inverted)
            )
            earlier_button.clicked.connect(
                lambda _checked=False, target=column: self.shift_trace_by_frames(target, -1)
            )
            later_button.clicked.connect(
                lambda _checked=False, target=column: self.shift_trace_by_frames(target, 1)
            )
            item.setSizeHint(QSize(row_widget.sizeHint().width(), trace_row_height))
            self.trace_list.setItemWidget(item, row_widget)
            self.trace_row_controls[column] = (invert_button, shift_label)
            self.update_trace_row_processing(column)
            if column in selected:
                item.setSelected(True)
        self.update_trace_row_selection_styles()
        self.trace_list.blockSignals(False)

    def toggle_auxiliary_traces(self, hidden: bool) -> None:
        self.auxiliary_traces_hidden = bool(hidden)
        self.toggle_auxiliary_traces_button.setText(
            "Show baselines/originals" if hidden else "Hide baselines/originals"
        )
        self.populate_traces()
        self.schedule_plot(preserve_view=True)
        self.set_status(
            "Baseline, original, and uncorrected traces are hidden."
            if hidden
            else "Baseline, original, and uncorrected traces are visible."
        )

    def select_trace_row(self, item: QListWidgetItem, modifiers) -> None:
        if modifiers & Qt.ShiftModifier and self._trace_drag_anchor_item is not None:
            self.extend_trace_selection_to(item)
            return
        if modifiers & Qt.ControlModifier:
            selected = item.isSelected()
            self._trace_drag_base_rows = {
                row
                for row in range(self.trace_list.count())
                if self.trace_list.item(row).isSelected()
            }
            self._trace_drag_additive = True
            self._trace_drag_selecting = not selected
            self.trace_list.setCurrentItem(item, QItemSelectionModel.NoUpdate)
            item.setSelected(not selected)
            self._trace_drag_anchor_item = item
            return
        self._trace_drag_base_rows.clear()
        self._trace_drag_additive = False
        self._trace_drag_selecting = True
        self.trace_list.setCurrentItem(item, QItemSelectionModel.ClearAndSelect)
        self._trace_drag_anchor_item = item

    def extend_trace_drag(self, global_position, _modifiers=Qt.NoModifier) -> None:
        """Extend the row selection while a custom trace row is being dragged."""
        viewport_position = self.trace_list.viewport().mapFromGlobal(global_position)
        target = self.trace_list.itemAt(viewport_position)
        if target is None and self.trace_list.count():
            target = self.trace_list.item(
                0 if viewport_position.y() < 0 else self.trace_list.count() - 1
            )
        if target is not None:
            self.extend_trace_selection_to(target)

    def extend_trace_selection_to(self, target: QListWidgetItem) -> None:
        """Select, add, or remove the dragged range according to the press modifiers."""
        anchor = self._trace_drag_anchor_item
        if anchor is None:
            self._trace_drag_anchor_item = target
            return
        start = self.trace_list.row(anchor)
        end = self.trace_list.row(target)
        if start < 0 or end < 0:
            return
        lower, upper = sorted((start, end))
        dragged_rows = set(range(lower, upper + 1))
        if self._trace_drag_additive:
            selected_rows = (
                self._trace_drag_base_rows | dragged_rows
                if self._trace_drag_selecting
                else self._trace_drag_base_rows - dragged_rows
            )
        else:
            selected_rows = dragged_rows
        self.trace_list.blockSignals(True)
        try:
            self.trace_list.setCurrentItem(target, QItemSelectionModel.NoUpdate)
            for row in range(self.trace_list.count()):
                self.trace_list.item(row).setSelected(row in selected_rows)
        finally:
            self.trace_list.blockSignals(False)
        self.trace_selection_changed()

    def set_trace_inverted(self, column: str, invert: bool) -> None:
        current = self.trace_processing.get(column, TraceProcessingSettings())
        if current.invert == invert:
            return
        self.trace_processing[column] = replace(current, invert=invert)
        self.plot_cache.pop(column, None)
        self.update_trace_row_processing(column)
        self.schedule_plot(preserve_view=True)
        self.set_status(
            f"{'Inversion enabled' if invert else 'Inversion disabled'} for {column}."
        )

    def selected_columns(self) -> list[str]:
        return [
            self.visible_columns[index.row()]
            for index in self.trace_list.selectedIndexes()
            if index.row() < len(self.visible_columns)
        ]

    def trace_selection_changed(self) -> None:
        self.update_trace_row_selection_styles()
        self.update_trace_actions()
        self.load_selected_processing()
        self.schedule_plot()

    def update_trace_row_selection_styles(self) -> None:
        """Synchronize embedded row backgrounds with QListWidget selection state."""
        for row in range(self.trace_list.count()):
            item = self.trace_list.item(row)
            row_widget = self.trace_list.itemWidget(item)
            if isinstance(row_widget, TraceListRow):
                row_widget.set_selected(item.isSelected())

    def update_trace_actions(self) -> None:
        """Keep trace-specific actions synchronized with the current selection."""
        selected = self.selected_columns()
        actions_enabled = not self._operation_busy
        self.remove_traces_button.setEnabled(actions_enabled and bool(selected))
        self.open_trace_location_button.setEnabled(
            actions_enabled and bool(selected and self.trace_source_paths.get(selected[0]))
        )
        self._selected_recording_path = None
        if len(selected) == 1:
            source = self.trace_source_paths.get(selected[0])
            self._selected_recording_path = find_annotated_recording(selected[0], source)
        self.watch_recording_button.setEnabled(
            actions_enabled and self._selected_recording_path is not None
        )
        selected_trace_controls_enabled = actions_enabled and bool(selected)
        self.legend_polarity_button.setEnabled(selected_trace_controls_enabled)
        self.legend_shift_earlier_button.setEnabled(selected_trace_controls_enabled)
        self.legend_shift_later_button.setEnabled(selected_trace_controls_enabled)
        self.align_first_peak_button.setEnabled(actions_enabled and len(selected) >= 2)
        self.update_first_peak_alignment_action()
        if self._selected_recording_path is not None:
            self.watch_recording_button.setToolTip(
                f"Open the annotated recording in a new window:\n{self._selected_recording_path}"
            )
        elif len(selected) > 1:
            self.watch_recording_button.setToolTip(
                "Select exactly one trace to watch its annotated recording."
            )
        elif selected:
            self.watch_recording_button.setToolTip(
                "No annotated recording was found beside this trace's source file."
            )
        else:
            self.watch_recording_button.setToolTip(
                "Select one trace with an available annotated recording."
            )

    def watch_selected_recording(self) -> None:
        """Open the selected trace's generated overlay in a non-modal player window."""
        selected = self.selected_columns()
        if len(selected) != 1:
            QMessageBox.information(
                self,
                "Select one trace",
                "Select exactly one trace before opening its annotated recording.",
            )
            return
        source = self.trace_source_paths.get(selected[0])
        recording = find_annotated_recording(selected[0], source)
        if recording is None or not recording.is_file():
            self.update_trace_actions()
            QMessageBox.warning(
                self,
                "Annotated recording unavailable",
                "No annotated recording matching the selected trace was found in the expected "
                "results folders. It may not have been generated, or it may have been moved.",
            )
            return
        try:
            viewer = AnnotatedRecordingViewer(recording, self.window())
        except Exception as error:
            LOGGER.exception("Could not open annotated recording")
            QMessageBox.critical(self, "Could not open annotated recording", str(error))
            return
        self.recording_viewers.add(viewer)
        viewer.destroyed.connect(
            lambda _object=None, opened_viewer=viewer: self.recording_viewers.discard(
                opened_viewer
            )
        )
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()
        QTimer.singleShot(0, viewer.play)

    def open_selected_trace_location(self) -> None:
        """Open the folder containing the first selected trace's original source file."""
        selected = self.selected_columns()
        if not selected:
            return
        source = self.trace_source_paths.get(selected[0])
        if source is None or not source.exists():
            QMessageBox.warning(
                self,
                "Source file unavailable",
                "The selected trace's source file is no longer available at its saved location.",
            )
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(source.parent.resolve()))):
            QMessageBox.warning(self, "Could not open folder", f"Could not open:\n{source.parent}")

    def schedule_plot(self, preserve_view: bool = False) -> None:
        """Coalesce rapid list/control changes into one plot render."""
        if preserve_view and self.axes.has_data():
            x_limits = tuple(map(float, self.axes.get_xlim()))
            y_limits = tuple(map(float, self.axes.get_ylim()))
            limits = (*x_limits, *y_limits)
            self._pending_plot_view = (
                (x_limits[0], x_limits[1]),
                (y_limits[0], y_limits[1]),
            ) if np.isfinite(limits).all() else None
        else:
            self._pending_plot_view = None
        self.plot_timer.start()

    def remove_selected_traces(self) -> None:
        columns = self.selected_columns()
        if self.data is None or not columns:
            return
        selected_rows = [index.row() for index in self.trace_list.selectedIndexes()]
        next_row = min(selected_rows) if selected_rows else 0
        self.data.drop(columns=columns, inplace=True, errors="ignore")
        self.baseline_columns.difference_update(columns)
        for column in columns:
            self.trace_processing.pop(column, None)
            self.trace_source_paths.pop(column, None)
        self.plot_cache.clear()
        self.summary = pd.DataFrame()
        self.beats = pd.DataFrame()
        self.results_table.clear()
        self.results_table.setRowCount(0)
        self.results_table.setColumnCount(0)
        self.axes.clear()
        self.style_plot()
        self.canvas.draw_idle()
        remaining = len(self.signal_columns())
        if remaining == 0:
            self.data = None
            self.path = None
            self.paths = []
            self.trace_source_paths.clear()
            self.time_column = ""
            self.visible_columns = []
            self.trace_processing.clear()
            self.baseline_columns.clear()
            self._preloaded_trace_files.clear()
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
            self.trace_list.blockSignals(True)
            self.trace_list.clear()
            self.trace_list.blockSignals(False)
            self.source_label.setText(
                "No trace file loaded — drop CSV/Excel files anywhere on this page"
            )
            self.source_path.clear()
            self.processing_target.setText("Select one or more traces")
            self.remove_traces_button.setEnabled(False)
            self.open_trace_location_button.setEnabled(False)
            count = len(columns)
            self.set_status(
                f"Removed {count} trace{'s' if count != 1 else ''}; the analysis workspace is empty. "
                "Source files were not changed."
            )
            return

        # A source file should be retained only while it still contributes a visible trace.
        # Without this, removing all columns from one file left a stale file count in the
        # Add/replace dialog even though only the surviving traces were displayed.
        active_sources = {path.resolve() for path in self.trace_source_paths.values()}
        self.paths = [path for path in self.paths if path.resolve() in active_sources]
        self.path = self.paths[0] if self.paths else None
        self.populate_traces()
        if self.trace_list.count():
            self.trace_list.setCurrentRow(min(next_row, self.trace_list.count() - 1))
        else:
            self.remove_traces_button.setEnabled(False)
        count = len(columns)
        self.set_status(
            f"Removed {count} trace{'s' if count != 1 else ''} from this workspace; "
            f"{remaining} remain from {len(self.paths)} source file(s)."
        )

    def select_all(self):
        self.trace_list.selectAll()
        self.schedule_plot()

    def set_baselines(self):
        selected = self.selected_columns()
        if not selected:
            QMessageBox.warning(self, "No reference", "Select one or more reference traces.")
            return
        specimens: dict[str, list[str]] = {}
        for column in selected:
            specimens.setdefault(reference_match_key(column), []).append(column)
        duplicates = {
            specimen: columns
            for specimen, columns in specimens.items()
            if len(columns) > 1
        }
        if duplicates:
            details = "\n".join(
                f"{specimen}: {', '.join(columns)}"
                for specimen, columns in duplicates.items()
            )
            QMessageBox.warning(
                self,
                "Ambiguous references",
                "Select only one reference trace for each specimen.\n\n" + details,
            )
            return
        self.baseline_columns = set(selected)
        self.populate_traces()
        self.set_status(
            f"{len(selected)} reference trace(s) assigned. They will be matched by specimen "
            "name after removing the condition prefix."
        )

    def clear_baselines(self):
        self.baseline_columns.clear()
        self.populate_traces()
        self.set_status("Reference traces cleared. No reference normalization will be applied.")

    def reference_columns(self) -> list[str]:
        """Return selected references in the same stable order as the trace list."""
        return [
            column
            for column in self.signal_columns()
            if column in self.baseline_columns
        ]

    def reference_export_frame(self, analyzed_columns: list[str] | None = None) -> pd.DataFrame:
        analyzed = analyzed_columns or []
        rows = []
        for column in self.reference_columns():
            condition, specimen, timestamp = parse_trace_name(column)
            matched = [
                candidate
                for candidate in analyzed
                if reference_match_key(candidate) == reference_match_key(column)
            ]
            rows.append({
                "Reference trace": column,
                "Condition": condition,
                "Specimen": specimen,
                "Timestamp": timestamp,
                "Matched analyzed traces": "; ".join(matched),
                "Matching rule": "Specimen name after condition prefix",
                "Purpose": (
                    "Denominator for beat rate, peak force, amplitude, time-to-peak, and CTD90 "
                    "percentage comparisons"
                ),
            })
        if not rows:
            rows.append({
                "Reference trace": "None selected",
                "Condition": "",
                "Specimen": "",
                "Timestamp": "",
                "Matched analyzed traces": "",
                "Matching rule": "Specimen name after condition prefix",
                "Purpose": "No reference normalization applied",
            })
        return pd.DataFrame(rows)

    def export_settings_frame(self, settings: AnalysisSettings) -> pd.DataFrame:
        references = self.reference_columns()
        per_trace_fields = {
            "invert",
            "baseline_method",
            "baseline_percentile",
            "baseline_window_s",
            "baseline_asls_log10",
            "time_shift_frames",
        }
        rows = [
            (name, value)
            for name, value in asdict(settings).items()
            if name not in per_trace_fields
        ]
        rows.extend([
            ("invert_and_baseline_scope", "Saved separately for each trace"),
            ("reference_trace_count", len(references)),
            ("reference_matching_rule", "Specimen name after condition prefix"),
            ("reference_traces", "; ".join(references) if references else "None selected"),
            (
                "reference_percentage_metrics",
                "; ".join(REFERENCE_METRICS),
            ),
        ])
        return pd.DataFrame(rows, columns=["Setting", "Value"])

    def trace_processing_frame(self, columns: list[str]) -> pd.DataFrame:
        rows = []
        for column in columns:
            processing = self.trace_processing.get(column, TraceProcessingSettings())
            rows.append({"Trace": column, **asdict(processing)})
        return pd.DataFrame(rows)

    def draw_processed_trace(
        self,
        column: str,
        info: dict,
        show_baseline_labels: bool,
        show_auxiliary_curves: bool,
    ) -> None:
        has_correction = info["baseline_method"] != "none"
        shift = self.trace_processing.get(column, TraceProcessingSettings()).time_shift_frames
        shift_suffix = f" [{shift:+d} frames]" if shift else ""
        label = (f"{column} corrected" if has_correction else column) + shift_suffix
        point_count = len(info["t"])
        stride = max(1, int(np.ceil(point_count / 5000)))
        display = slice(None, None, stride)
        line, = self.axes.plot(
            info["t"][display],
            info["smooth"][display],
            linewidth=1.7,
            label=label,
            zorder=3,
        )
        color = line.get_color()
        self.axes.plot(
            info["t"][display],
            info["raw"][display],
            color=color,
            alpha=0.28,
            linewidth=0.9,
            label="_nolegend_",
            zorder=2,
        )
        if has_correction and show_auxiliary_curves:
            self.axes.plot(
                info["t"][display],
                info["original"][display],
                color=color,
                alpha=0.35,
                linewidth=1,
                label=f"{column} original" if show_baseline_labels else "_nolegend_",
                zorder=1,
            )
            self.axes.plot(
                info["t"][display],
                info["baseline"][display],
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"{column} baseline" if show_baseline_labels else "_nolegend_",
                zorder=2,
            )
        peaks = info.get("peaks")
        if peaks is not None:
            self.axes.scatter(
                info["t"][peaks],
                info["smooth"][peaks],
                color=color,
                marker="x",
                s=28,
                zorder=4,
            )

    def y_axis_label(self, columns: list[str]) -> str:
        if not columns:
            return "Signal"
        names = [column.lower() for column in columns]
        explicitly_micronewtons = all(
            "force" in name and ("_un" in name or "µn" in name or "μn" in name)
            for name in names
        )
        calibrated_force_names = (
            self.force_calibration_available
            and all("force" in name for name in names)
        )
        explicitly_micrometers = all(
            ("distance" in name or "displacement" in name) and "_um" in name
            for name in names
        )
        if explicitly_micronewtons or calibrated_force_names:
            return "Force (µN)"
        if explicitly_micrometers:
            return "Distance (µm)"
        return "Signal"

    def zoom_with_mouse_wheel(self, event) -> None:
        if event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            return
        step = float(getattr(event, "step", 0) or 0)
        if event.button == "up" or step > 0:
            scale = 0.8
        elif event.button == "down" or step < 0:
            scale = 1.25
        else:
            return

        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        self.axes.set_xlim(
            event.xdata - (event.xdata - x_min) * scale,
            event.xdata + (x_max - event.xdata) * scale,
        )
        self.axes.set_ylim(
            event.ydata - (event.ydata - y_min) * scale,
            event.ydata + (y_max - event.ydata) * scale,
        )
        self.canvas.draw_idle()

    def plot_selected(self):
        preserved_view = self._pending_plot_view
        self._pending_plot_view = None
        self.axes.clear()
        self.style_plot()
        if self.data is None:
            self.canvas.draw_idle()
            return
        columns = self.selected_columns()
        show_baseline_labels = len(columns) == 1
        time_values = pd.to_numeric(self.data[self.time_column], errors="coerce").to_numpy(float)
        try:
            shared_settings = self.settings()
        except ValueError:
            shared_settings = AnalysisSettings()
        for column in columns:
            if column in self.plot_cache:
                self.draw_processed_trace(
                    column,
                    self.plot_cache[column],
                    show_baseline_labels,
                    not self.auxiliary_traces_hidden,
                )
            else:
                values = pd.to_numeric(self.data[column], errors="coerce").to_numpy(float)
                try:
                    settings = self.settings_for_column(column, shared_settings)
                    prepared = prepare_signal(time_values, values, settings)
                    self.draw_processed_trace(
                        column,
                        {
                            "t": prepared.t,
                            "original": prepared.original,
                            "baseline": prepared.baseline,
                            "raw": prepared.corrected,
                            "smooth": prepared.smooth,
                            "baseline_method": settings.baseline_method,
                        },
                        show_baseline_labels,
                        not self.auxiliary_traces_hidden,
                    )
                except ValueError:
                    continue
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel(self.y_axis_label(columns))
        self.axes.grid(True, linestyle=":", alpha=0.4)
        if columns and len(columns) <= 12:
            legend = self.axes.legend(fontsize=8, facecolor="#20262c", edgecolor="#46515a")
            for text in legend.get_texts():
                text.set_color("#e3e8ec")
        self.figure.tight_layout()
        if preserved_view is not None and self.axes.has_data():
            self.axes.set_xlim(*preserved_view[0])
            self.axes.set_ylim(*preserved_view[1])
        self.canvas.draw_idle()

    def style_plot(self):
        self.figure.patch.set_facecolor("#171c20")
        self.axes.set_facecolor("#171c20")
        self.axes.tick_params(colors="#b8c2ca")
        self.axes.xaxis.label.set_color("#d8dfe4")
        self.axes.yaxis.label.set_color("#d8dfe4")
        self.axes.title.set_color("#eef2f5")
        for spine in self.axes.spines.values():
            spine.set_color("#4a555e")

    def analyze_selected(self):
        if self.data is None:
            QMessageBox.warning(self, "No data", "Open a trace file first.")
            return
        columns = self.selected_columns()
        if not columns:
            QMessageBox.warning(self, "No traces", "Select at least one trace.")
            return
        try:
            shared_settings = self.settings()
        except ValueError as error:
            QMessageBox.warning(self, "Invalid settings", str(error))
            return
        time_values = pd.to_numeric(
            self.data[self.time_column], errors="coerce"
        ).to_numpy(float, copy=True)
        reference_columns = self.reference_columns()
        analysis_columns = list(dict.fromkeys(columns + reference_columns))
        signals = {
            column: pd.to_numeric(self.data[column], errors="coerce").to_numpy(
                float, copy=True
            )
            for column in analysis_columns
        }
        settings_by_column = {
            column: self.settings_for_column(column, shared_settings)
            for column in analysis_columns
        }
        baseline_columns = set(self.baseline_columns)

        def analyze_snapshot() -> dict:
            return analyze_trace_batch(
                time_values,
                signals,
                settings_by_column,
                list(columns),
                list(reference_columns),
                baseline_columns,
            )

        self.start_background_operation(
            "analyzing-traces",
            analyze_snapshot,
            self.finish_trace_analysis,
            "Signal analysis failed",
        )

    def finish_trace_analysis(self, result: object) -> None:
        if not isinstance(result, dict):
            self.background_operation_failed(
                "Signal analysis failed", "The analysis returned an invalid result."
            )
            return
        self.summary = result.get("summary", pd.DataFrame())
        self.beats = result.get("beats", pd.DataFrame())
        self.plot_cache = result.get("plot_cache", {})
        errors = result.get("errors", [])
        selected_count = int(result.get("selected_count", 0))
        self.refresh_results()
        self.plot_selected()
        self.set_status(
            f"Listed {len(self.summary)}/{selected_count} selected traces; detected "
            f"{len(self.beats)} complete beats "
            "using the processing settings saved for each trace. "
            f"{int((self.summary.get('Reference_trace', pd.Series(dtype=str)) != '').sum())} "
            "trace(s) matched a reference."
        )
        if errors:
            QMessageBox.warning(
                self,
                "Some traces have no beat metrics",
                "These traces remain listed in Summary with N_beats = 0 and NaN metrics:\n\n"
                + "\n".join(errors[:10]),
            )

    def refresh_results(self):
        columns = summary_display_columns(self.summary)
        self.results_table.setSortingEnabled(False)
        self.results_table.setRowCount(len(self.summary))
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(
            [SUMMARY_COLUMN_LABELS.get(column, column) for column in columns]
        )
        for row_index, (_, row) in enumerate(self.summary.iterrows()):
            for column_index, column in enumerate(columns):
                value = row[column]
                text = f"{value:.5g}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)
                self.results_table.setItem(row_index, column_index, QTableWidgetItem(text))
        self.results_table.resizeColumnsToContents()
        self.results_table.refresh_frozen_column()
        self.results_table.setSortingEnabled(True)

    def export_metrics(self):
        if self.summary.empty:
            QMessageBox.warning(self, "No results", "Analyze selected traces first.")
            return
        selected_columns = self.selected_columns()
        if not selected_columns:
            QMessageBox.warning(
                self,
                "No traces selected",
                "Select the analyzed traces that should be exported.",
            )
            return
        selected_summary, selected_beats = selected_result_frames(
            self.summary,
            self.beats,
            selected_columns,
        )
        if selected_summary.empty:
            QMessageBox.warning(
                self,
                "Selected traces not analyzed",
                "Analyze the currently selected traces before exporting their metrics.",
            )
            return
        default = (
            "multi_trace_metrics.xlsx"
            if len(self.paths) > 1
            else f"{self.path.stem}_metrics.xlsx"
            if self.path
            else "trace_metrics.xlsx"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export metrics",
            self.export_default_path(default),
            "Excel workbook (*.xlsx)",
        )
        if not filename:
            return
        try:
            selected_names = selected_summary["Trace"].astype(str).tolist()
            references = self.reference_export_frame(selected_names)
            settings_frame = self.export_settings_frame(self.settings())
            processing = self.trace_processing_frame(selected_names)
        except Exception as error:
            LOGGER.exception("Metric export failed")
            QMessageBox.critical(self, "Export failed", str(error))
            return

        def export_snapshot() -> str:
            return write_metrics_workbook(
                filename,
                selected_summary.copy(deep=True),
                selected_beats.copy(deep=True),
                references.copy(deep=True),
                settings_frame.copy(deep=True),
                processing.copy(deep=True),
            )

        self.start_background_operation(
            "exporting-metrics",
            export_snapshot,
            lambda saved: self.set_status(
                f"Metrics for {len(selected_summary)} selected trace(s) saved to {saved}"
            ),
            "Metric export failed",
        )

    def export_processed(self, on_success=None) -> bool:
        if self.data is None or not self.selected_columns():
            QMessageBox.warning(self, "No traces", "Load data and select traces first.")
            return False
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export processed traces",
            self.export_default_path(
                "multi_processed_traces.xlsx"
                if len(self.paths) > 1
                else f"{self.path.stem}_processed.xlsx"
                if self.path
                else "processed_traces.xlsx"
            ),
            "Excel workbook (*.xlsx)",
        )
        if not filename:
            return False
        try:
            shared_settings = self.settings()
            selected_columns = self.selected_columns()
            time_values = pd.to_numeric(
                self.data[self.time_column], errors="coerce"
            ).to_numpy(float, copy=True)
            signals = {
                column: pd.to_numeric(self.data[column], errors="coerce").to_numpy(
                    float, copy=True
                )
                for column in selected_columns
            }
            settings_by_column = {
                column: self.settings_for_column(column, shared_settings)
                for column in selected_columns
            }
            references = self.reference_export_frame(selected_columns)
            settings_frame = self.export_settings_frame(shared_settings)
            processing = self.trace_processing_frame(selected_columns)
            time_column = self.time_column
        except Exception as error:
            LOGGER.exception("Processed-trace export failed")
            QMessageBox.critical(self, "Export failed", str(error))
            return False

        def export_snapshot() -> str:
            return write_processed_workbook(
                filename,
                time_column,
                time_values,
                signals,
                settings_by_column,
                references.copy(deep=True),
                settings_frame.copy(deep=True),
                processing.copy(deep=True),
            )

        def exported(saved: str) -> None:
            self.set_status(f"Processed traces saved to {saved}")
            if callable(on_success):
                on_success(saved)

        return self.start_background_operation(
            "exporting-processed-traces",
            export_snapshot,
            exported,
            "Processed-trace export failed",
        )

    def export_default_path(self, filename: str) -> str:
        """Start exports beside the first opened trace file when it is available."""
        source = self.path or (self.paths[0] if self.paths else None)
        return str(source.parent / filename) if source is not None else filename
