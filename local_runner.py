"""Run StimTrace segmentation directly on this computer."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch


EVENT_PREFIX = "STIMTRACE_EVENT "
DEFAULT_PARAMETERS: dict[str, Any] = {
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
    "inference_batch_size": 1,
    "cpu_postprocess_workers": 1,
    "progress_interval_seconds": 2.0,
    "generate_overlays": True,
}


def emit_event(event: dict[str, Any]) -> None:
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=True), flush=True)


def hardware_summary() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "cuda_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "cpu_threads": os.cpu_count() or 1,
        "torch_version": torch.__version__,
    }


def choose_device(preference: str) -> torch.device:
    preference = preference.lower()
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "NVIDIA CUDA was selected, but this Python environment cannot access a CUDA GPU."
        )
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def render_downloaded_overlays(
    results_folder: Path,
    source_paths: list[str],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[str], list[str]]:
    """Render cloud-tracked overlays locally from downloaded CSV traces.

    This is deliberately independent from local segmentation: it reuses the
    recorded tracking centers and reads only the original local videos.
    """
    import cv2
    import numpy as np
    import pandas as pd
    from analysis_worker import create_overlay_video

    sources = {Path(path).stem.casefold(): Path(path) for path in source_paths}
    created: list[str] = []
    skipped: list[str] = []
    trace_files = sorted((results_folder / "traces").glob("*_stimtrace_tracking.csv"))
    work = [(trace_path, pd.read_csv(trace_path)) for trace_path in trace_files]
    total_frames = max(1, sum(len(trace) for _, trace in work))
    completed_frames = 0
    for file_index, (trace_path, trace) in enumerate(work, start=1):
        recording_stem = trace_path.name.removesuffix("_stimtrace_tracking.csv")
        video_path = sources.get(recording_stem.casefold())
        if video_path is None or not video_path.is_file():
            skipped.append(f"{recording_stem}: original video is unavailable")
            completed_frames += len(trace)
            if progress_callback:
                progress_callback({
                    "phase": "overlay_progress", "file_index": file_index,
                    "file_count": len(work), "filename": trace_path.stem,
                    "current_frame": len(trace), "current_file_frames": len(trace),
                    "completed_frames": completed_frames, "total_frames": total_frames,
                    "overall_fraction": completed_frames / total_frames, "skipped": True,
                })
            continue
        required = {"smooth_center_x", "smooth_center_y"}
        if not required.issubset(trace.columns):
            skipped.append(f"{trace_path.stem}: tracking center columns are missing")
            completed_frames += len(trace)
            continue
        centers = list(zip(trace["smooth_center_x"], trace["smooth_center_y"]))
        capture = cv2.VideoCapture(str(video_path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        finally:
            capture.release()
        if not np.isfinite(fps) or fps <= 0 or min(frame_size) <= 0:
            skipped.append(f"{video_path.name}: video metadata is unreadable")
            completed_frames += len(trace)
            continue
        axes = trace.reindex(columns=["ellipse_axis_a", "ellipse_axis_b"]).to_numpy(float)
        radius = int(round(np.nanmedian(axes) / 2)) if np.isfinite(axes).any() else 40
        report_interval = max(1, len(centers) // 20)

        def report_frame(_name, _stage, current, file_total):
            if not progress_callback:
                return
            if current != 1 and current != file_total and current % report_interval:
                return
            overall_current = completed_frames + min(current, len(centers))
            progress_callback({
                "phase": "overlay_progress", "file_index": file_index,
                "file_count": len(work), "filename": video_path.name,
                "current_frame": current, "current_file_frames": file_total,
                "completed_frames": overall_current, "total_frames": total_frames,
                "overall_fraction": min(1.0, overall_current / total_frames),
            })

        create_overlay_video(
            video_path, results_folder, centers, max(1, radius), fps, frame_size,
            stage_callback=report_frame,
        )
        created.append(video_path.name)
        completed_frames += len(trace)
        if progress_callback:
            progress_callback({
                "phase": "overlay_progress", "file_index": file_index,
                "file_count": len(work), "filename": video_path.name,
                "current_frame": len(trace), "current_file_frames": len(trace),
                "completed_frames": completed_frames, "total_frames": total_frames,
                "overall_fraction": min(1.0, completed_frames / total_frames),
            })
    return created, skipped


def default_model_path() -> Path:
    resource_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    candidates = [
        resource_dir / "model.pth",
        resource_dir / "unet_multitask_center_ellipse_512x512 - Backup.pth",
        Path(__file__).resolve().parent.parent
        / "unet_multitask_center_ellipse_512x512 - Backup.pth",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def create_test_clip(source: Path, target: Path, maximum_frames: int) -> int:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(
            f"{source.name} has no valid frame-rate metadata; a trial clip cannot "
            "preserve its scientific time axis."
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"Could not create temporary trial clip: {target}")
    written = 0
    while written < maximum_frames:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    capture.release()
    if written == 0:
        raise ValueError(f"No readable frames found in: {source}")
    return written


def video_frame_count(video: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video))
    frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    return frames


def run_local(
    videos: list[Path],
    model_path: Path,
    output: Path,
    maximum_frames: int = 0,
    *,
    parameters: dict[str, Any] | None = None,
    device_preference: str = "auto",
    cpu_threads: int = 0,
    batch_size: int = 0,
    postprocess_workers: int = 0,
    generate_overlays: bool = True,
    control_file: Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    # Keep the startup hardware probe lightweight. The scientific segmentation
    # stack is only needed once a local analysis actually starts.
    from analysis_worker import (
        DeferredOverlayResult,
        PillarCenterTrackingError,
        create_benchmark_master,
        create_master,
        create_overlay_video,
        load_model,
        process_video,
    )

    missing = [str(path) for path in [model_path, *videos] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s):\n" + "\n".join(missing))
    stems = [path.stem.lower() for path in videos]
    if len(stems) != len(set(stems)):
        raise ValueError("Selected videos must have unique file names.")

    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(device_preference)
    cfg = dict(DEFAULT_PARAMETERS)
    cfg.update(parameters or {})
    default_torch_threads = torch.get_num_threads()
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    if batch_size > 0:
        cfg["inference_batch_size"] = batch_size
    else:
        cfg["inference_batch_size"] = 8 if device.type == "cuda" else 1
    if postprocess_workers > 0:
        cfg["cpu_postprocess_workers"] = postprocess_workers
    else:
        cfg["cpu_postprocess_workers"] = max(1, min(4, (os.cpu_count() or 1) // 2))
    cfg["generate_overlays"] = generate_overlays
    cfg["progress_interval_seconds"] = 0.1
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    notify = event_callback or (lambda _event: None)
    automatic_batch_size = 8 if device.type == "cuda" else 1
    automatic_postprocess_workers = max(1, min(4, (os.cpu_count() or 1) // 2))
    control_modified_ns = -1
    runtime_values = {
        "inference_batch_size": int(cfg["inference_batch_size"]),
        "cpu_postprocess_workers": int(cfg["cpu_postprocess_workers"]),
        "generate_overlays": bool(cfg["generate_overlays"]),
    }

    def runtime_settings() -> dict[str, Any]:
        nonlocal control_modified_ns, runtime_values
        if not control_file or not control_file.is_file():
            return runtime_values
        try:
            modified_ns = control_file.stat().st_mtime_ns
            if modified_ns == control_modified_ns:
                return runtime_values
            requested = json.loads(control_file.read_text(encoding="utf-8"))
            requested_batch = max(0, int(requested.get("inference_batch_size", 0)))
            requested_workers = max(0, int(requested.get("postprocess_workers", 0)))
            runtime_values = {
                "inference_batch_size": requested_batch or automatic_batch_size,
                "cpu_postprocess_workers": (
                    requested_workers or automatic_postprocess_workers
                ),
                "generate_overlays": bool(
                    requested.get("generate_overlays", generate_overlays)
                ),
            }
            control_modified_ns = modified_ns
            notify({
                "event": "settings_applied",
                "cpu_threads": torch.get_num_threads(),
                **runtime_values,
            })
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # The GUI may be replacing the small JSON file as it is read.
            # Retain the last complete settings and try again next batch.
            pass
        return runtime_values

    runtime_settings()
    device_name = (
        torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else f"CPU ({torch.get_num_threads()} threads)"
    )
    notify({
        "event": "starting",
        "device": device.type,
        "device_name": device_name,
        "processing_started_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Loading {model_path.name} on {device_name}...", flush=True)
    model = load_model(model_path, device)
    traces: list[Path] = []
    benchmark_traces: dict[str, list[Path]] = {}
    skipped_videos: list[dict[str, str]] = []
    started = time.perf_counter()
    temporary_root = output / f".stimtrace_local_{uuid.uuid4().hex[:8]}"
    temporary_files: list[Path] = []
    source_frame_totals = [
        min(video_frame_count(video), maximum_frames) if maximum_frames else video_frame_count(video)
        for video in videos
    ]
    total_frames = max(1, sum(source_frame_totals))
    completed_frames = 0
    segmentation_weight = 0.98 if not generate_overlays else 0.90
    overlay_weight = 0.0 if not generate_overlays else 0.08
    if maximum_frames:
        temporary_root.mkdir()
    pipeline_overlays = (
        device.type == "cuda"
        and generate_overlays
        and not bool(cfg.get("kalman_benchmark"))
        and len(videos) > 1
    )
    overlay_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="stimtrace-overlay")
        if pipeline_overlays
        else None
    )
    overlay_futures = []
    try:
        for index, video in enumerate(videos, start=1):
            if cancel_check and cancel_check():
                raise InterruptedError("Processing cancelled.")
            processing_video = video
            if maximum_frames:
                processing_video = temporary_root / video.name
                temporary_files.append(processing_video)
                count = create_test_clip(video, processing_video, maximum_frames)
                source_frame_totals[index - 1] = count
                print(f"Created {count}-frame trial clip for {video.name}.", flush=True)
            file_total = source_frame_totals[index - 1]
            print(f"Video {index}/{len(videos)}: {video.name}", flush=True)

            def report_stage(
                _video_name: str,
                stage: str,
                current: int,
                stage_total: int,
            ) -> None:
                denominator = max(1, stage_total or file_total)
                stage_fraction = min(1.0, current / denominator)
                if stage == "segmenting":
                    file_fraction = segmentation_weight * stage_fraction
                else:
                    file_fraction = segmentation_weight + overlay_weight * stage_fraction
                job_fraction = min(
                    0.98,
                    (completed_frames + file_fraction * file_total) / total_frames,
                )
                elapsed = time.perf_counter() - started
                eta = elapsed * (1.0 - job_fraction) / job_fraction if job_fraction else None
                notify({
                    "event": "progress",
                    "stage": stage,
                    "file_index": index,
                    "file_count": len(videos),
                    "file_name": video.name,
                    "current_frame": current,
                    "current_file_frames": denominator,
                    "processed_frames": completed_frames + (
                        min(current, file_total) if stage == "segmenting" else file_total
                    ),
                    "total_job_frames": total_frames,
                    "file_progress_fraction": file_fraction,
                    "job_progress_fraction": job_fraction,
                    "estimated_remaining_seconds": round(eta) if eta is not None else None,
                })

            def report_segmentation(
                current_video_name: str,
                current: int,
                stage_total: int,
            ) -> None:
                """Adapt per-batch segmentation updates to the local job event format."""
                report_stage(current_video_name, "segmenting", current, stage_total)

            try:
                trace = process_video(
                    processing_video,
                    output,
                    model,
                    device,
                    cfg,
                    progress_callback=report_segmentation,
                    stage_callback=report_stage,
                    cancel_check=cancel_check,
                    runtime_settings_callback=runtime_settings,
                    defer_overlay=pipeline_overlays,
                )
            except PillarCenterTrackingError as error:
                reason = str(error)
                skipped_videos.append({"video": video.name, "reason": reason})
                print(f"Skipping {video.name}: {reason}", flush=True)
                completed_frames += file_total
                notify({
                    "event": "video_skipped", "file_index": index,
                    "file_count": len(videos), "file_name": video.name,
                    "reason": reason, "processed_frames": completed_frames,
                    "total_job_frames": total_frames,
                })
                continue
            if isinstance(trace, DeferredOverlayResult):
                traces.append(trace.trace_path)
                overlay_futures.append(overlay_executor.submit(
                    create_overlay_video,
                    trace.video_path,
                    trace.output,
                    trace.centers,
                    trace.radius,
                    trace.fps,
                    trace.frame_size,
                    cancel_check=cancel_check,
                ))
            elif isinstance(trace, dict):
                for name, variant_trace in trace.items():
                    benchmark_traces.setdefault(name, []).append(variant_trace)
            else:
                traces.append(trace)
            completed_frames += file_total
        for future in overlay_futures:
            future.result()
        notify({
            "event": "progress",
            "stage": "workbook",
            "file_index": len(videos),
            "file_count": len(videos),
            "file_name": videos[-1].name,
            "current_frame": source_frame_totals[-1],
            "current_file_frames": source_frame_totals[-1],
            "file_progress_fraction": 1.0,
            "job_progress_fraction": 0.99,
            "processed_frames": completed_frames,
            "total_job_frames": total_frames,
            "estimated_remaining_seconds": 0,
        })
        if skipped_videos:
            (output / "skipped_videos.json").write_text(
                json.dumps(skipped_videos, indent=2), encoding="utf-8"
            )
        if benchmark_traces:
            for name, variant_traces in benchmark_traces.items():
                create_master(
                    variant_traces,
                    output / "kalman_benchmark" / name,
                    cfg,
                )
            create_benchmark_master(
                benchmark_traces,
                output / "kalman_benchmark",
                cfg,
            )
        elif traces:
            create_master(traces, output, cfg)
        else:
            raise PillarCenterTrackingError(
                "No video had enough unambiguous measured pillar centers to create a trace. "
                "See skipped_videos.json in the results folder."
            )
    finally:
        if overlay_executor:
            overlay_executor.shutdown(wait=True, cancel_futures=True)
        for temporary_file in temporary_files:
            temporary_file.unlink(missing_ok=True)
        if maximum_frames and temporary_root.exists():
            temporary_root.rmdir()

    elapsed = time.perf_counter() - started
    notify({
        "event": "complete",
        "output": str(output.resolve()),
        "elapsed_seconds": elapsed,
        "processed_frames": completed_frames,
        "total_job_frames": total_frames,
        "device": device.type,
        "device_name": device_name,
    })
    print(f"Local analysis complete in {elapsed:.1f} seconds.", flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parameters-json", default="")
    parser.add_argument(
        "--kalman-benchmark-json",
        default="",
        help="JSON list of Kalman benchmark configurations.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--postprocess-workers", type=int, default=0)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--hardware-json", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N frames of each video for a speed test.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.hardware_json:
        print(json.dumps(hardware_summary(), ensure_ascii=True), flush=True)
        return
    if not arguments.videos:
        raise ValueError("Choose at least one input video.")
    if arguments.max_frames < 0:
        raise ValueError("--max-frames cannot be negative.")
    output = arguments.output
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = arguments.videos[0].parent / "results" / f"local_{stamp}"
    parameters = json.loads(arguments.parameters_json) if arguments.parameters_json else {}
    if arguments.kalman_benchmark_json:
        benchmark = json.loads(arguments.kalman_benchmark_json)
        if not isinstance(benchmark, list) or not benchmark:
            raise ValueError("--kalman-benchmark-json must contain a non-empty JSON list.")
        parameters["kalman_benchmark"] = benchmark
    cancelled = lambda: bool(
        arguments.cancel_file and arguments.cancel_file.exists()
    )
    try:
        run_local(
            arguments.videos,
            arguments.model,
            output,
            arguments.max_frames,
            parameters=parameters,
            device_preference=arguments.device,
            cpu_threads=arguments.cpu_threads,
            batch_size=arguments.batch_size,
            postprocess_workers=arguments.postprocess_workers,
            generate_overlays=not arguments.no_overlays,
            control_file=arguments.control_file,
            cancel_check=cancelled,
            event_callback=emit_event,
        )
    except InterruptedError:
        emit_event({"event": "cancelled", "job_id": arguments.job_id})
        raise SystemExit(2)
    except Exception as error:
        emit_event({
            "event": "failed",
            "job_id": arguments.job_id,
            "error": str(error),
        })
        raise


if __name__ == "__main__":
    main()
