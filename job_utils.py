"""Pure helpers for StimTrace job metadata and result discovery."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


TERMINAL_JOB_STATES = frozenset({"complete", "failed", "cancelled"})
RUNNING_JOB_STATES = frozenset(
    {"downloading", "analyzing", "training", "uploading_results"}
)
ACTIVE_JOB_STATES = RUNNING_JOB_STATES | frozenset({"queued", "cancelling"})
RESULT_TRACE_PREFERENCE = (
    "kalman_benchmark_all_traces.csv",
    "kalman_benchmark_all_traces.xlsx",
    "stimtrace_force_traces.csv",
    "point_tracking_force_traces.csv",
    "combined_force_results.csv",
    "combined_force_results.xlsx",
)
VIDEO_FILE_SUFFIXES = frozenset({".avi", ".mp4", ".mov", ".mkv", ".m4v"})


def local_now_iso(*, timespec: str = "milliseconds") -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec=timespec)


def parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def elapsed_seconds(start: object, end: datetime | None = None) -> float | None:
    started = parse_timestamp(start)
    if started is None:
        return None
    finished = end or datetime.now(timezone.utc)
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    return max(
        0.0,
        (finished.astimezone(timezone.utc) - started.astimezone(timezone.utc)).total_seconds(),
    )


def format_elapsed(seconds: object) -> str:
    if seconds in (None, ""):
        return ""
    try:
        total = max(0, round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def job_history_identity(job: Mapping[str, Any]) -> str:
    """Return the stable identifier used to suppress cleared cloud history."""
    return str(job.get("folder_id") or job.get("job_id") or "")


def safe_path_component(value: object, *, fallback: str = "item") -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value).strip()
    ).strip("_")
    return cleaned or fallback


def duplicate_output_stems(paths: list[Path]) -> list[str]:
    """Return filenames whose case-insensitive output stems would collide."""
    seen: set[str] = set()
    duplicates: list[str] = []
    duplicate_keys: set[str] = set()
    for path in paths:
        name = Path(path).name
        key = Path(name).stem.casefold()
        if key in seen and key not in duplicate_keys:
            duplicates.append(name)
            duplicate_keys.add(key)
        seen.add(key)
    return duplicates


def video_files_in_folder(folder: Path) -> list[Path]:
    """Return supported video files directly inside one user-selected folder."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"The selected video folder is unavailable: {root}")
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in VIDEO_FILE_SUFFIXES
        ),
        key=lambda path: (path.name.casefold(), str(path).casefold()),
    )


def result_download_target(
    job: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path | None:
    saved_text = str(job.get("local_results_path", "") or "")
    saved_target = Path(saved_text) if saved_text else None
    if job.get("backend") == "local":
        return saved_target if saved_target and saved_target.exists() else None
    if saved_target and saved_target.exists() and not force:
        return saved_target
    if saved_target:
        return saved_target
    source_paths = [Path(value) for value in job.get("source_paths", []) if value]
    if not source_paths:
        return None
    source_folders = {path.parent for path in source_paths}
    recording_folder = (
        next(iter(source_folders)) if len(source_folders) == 1 else source_paths[0].parent
    )
    return recording_folder / "results" / str(job.get("job_id") or "StimTrace_results")


def analysis_trace_file(results_folder: Path) -> Path | None:
    """Find the preferred combined trace in one directory traversal."""
    root = Path(results_folder)
    if not root.exists():
        return None
    matches: dict[str, Path] = {}
    wanted = set(RESULT_TRACE_PREFERENCE)
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.name in wanted:
            matches.setdefault(candidate.name, candidate)
            if RESULT_TRACE_PREFERENCE[0] in matches or len(matches) == len(wanted):
                break
    return next(
        (matches[name] for name in RESULT_TRACE_PREFERENCE if name in matches),
        None,
    )
