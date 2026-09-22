"""Small, UI-independent operations for submitting and monitoring cloud jobs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol


class CloudDrive(Protocol):
    def submit(
        self,
        videos: list[Path],
        study: str,
        settings: Any,
        progress_callback: Callable[[int, float, str], None] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[str, str, str, str, str]: ...

    def job_progress(self, progress_file_id: str) -> dict[str, Any]: ...
    def job_terminal_status(self, folder_id: str) -> dict[str, Any]: ...
    def update_job_progress(self, progress_file_id: str, payload: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class CloudSubmissionRequest:
    videos: list[Path]
    study: str
    settings: Any
    parameters: dict[str, Any]


def upload_cloud_job(
    drive: CloudDrive,
    request: CloudSubmissionRequest,
    progress_callback: Callable[[int, float, str], None] | None = None,
) -> dict[str, str]:
    """Upload a complete job and return only the identifiers the UI needs."""
    job_id, folder_id, progress_file_id, preview_file_id, result_archive_file_id = drive.submit(
        request.videos,
        request.study,
        request.settings,
        progress_callback,
        request.parameters,
    )
    return {
        "job_id": job_id,
        "folder_id": folder_id,
        "progress_file_id": progress_file_id,
        "preview_file_id": preview_file_id,
        "result_archive_file_id": result_archive_file_id,
    }


def fetch_cloud_job_progress(
    drive: CloudDrive, progress_file_id: str, folder_id: str
) -> dict[str, Any]:
    """Fetch one job status and resolve a completed cancellation request."""
    progress = drive.job_progress(progress_file_id)
    if progress.get("state") == "cancelling" and folder_id:
        terminal = drive.job_terminal_status(folder_id)
        terminal_state = terminal.get("state", "")
        if terminal_state:
            terminal_messages = {
                "complete": "Analysis completed before cancellation was received.",
                "failed": "Analysis failed before cancellation was received.",
                "cancelled": "Analysis cancelled.",
            }
            progress.update(terminal)
            progress["state"] = terminal_state
            progress["message"] = terminal_messages[terminal_state]
            if terminal_state == "complete":
                progress.update({"job_progress_fraction": 1.0, "estimated_remaining_seconds": 0})
            drive.update_job_progress(progress_file_id, progress)
    return {"progress_file_id": progress_file_id, "progress": progress}


def make_submitted_segmentation_job(
    result: dict[str, str], context: dict[str, Any]
) -> dict[str, Any]:
    """Create the persisted desktop job record after a successful Drive upload."""
    submitted_at = context["submitted_at"]
    return {
        "job_id": result["job_id"],
        "study": context["study"],
        "type": "segmentation",
        "folder_id": result["folder_id"],
        "progress_file_id": result["progress_file_id"],
        "preview_file_id": result["preview_file_id"],
        "result_archive_file_id": result["result_archive_file_id"],
        "videos": context["video_names"],
        "source_paths": [str(video) for video in context["videos"]],
        "state": "queued",
        "message": "Waiting for Colab compute.",
        "created_at": submitted_at,
        "submitted_at": submitted_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": context["model_name"],
        "overlay_mode": context.get("overlay_mode", "cloud"),
        "kalman_benchmark": [item["name"] for item in context["kalman_benchmark"]],
    }
