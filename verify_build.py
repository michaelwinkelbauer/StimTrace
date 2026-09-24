"""Fast preflight checks for a reproducible StimTrace Windows build."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from app_metadata import APPLICATION_NAME, APPLICATION_VERSION


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    PROJECT_DIR / "model.pth",
    PROJECT_DIR.parent / "unet_multitask_center_ellipse_512x512 - Backup.pth",
)
REQUIRED_RESOURCES = (
    "assets",
    "assets/stimtrace.ico",
    "assets/stimtrace-logo.png",
    "analysis_worker.py",
    "stimtrace_colab_worker.ipynb",
    "LICENSE",
    "NOTICE.md",
    "AUTHORS.md",
    "CITATION.cff",
    "THIRD_PARTY_LICENSES",
    "QUICK_START.md",
    "StimTrace.spec",
)
REQUIRED_IMPORTS = (
    "PySide6",
    "googleapiclient",
    "cv2",
    "numpy",
    "pandas",
    "scipy",
    "torch",
)


def main() -> int:
    failures: list[str] = []
    for resource in REQUIRED_RESOURCES:
        if not (PROJECT_DIR / resource).exists():
            failures.append(f"Missing required build resource: {resource}")
    if not any(path.is_file() for path in MODEL_CANDIDATES):
        failures.append("Missing bundled segmentation model (model.pth or approved backup checkpoint).")
    for module in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module)
        except ImportError as error:
            failures.append(f"Missing build dependency {module}: {error}")
    if failures:
        print("StimTrace build preflight failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"{APPLICATION_NAME} {APPLICATION_VERSION} build preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
