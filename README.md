# StimTrace

For an end-user walkthrough, see [QUICK_START.md](QUICK_START.md).

Current desktop release metadata is stored in [`app_metadata.py`](app_metadata.py).
Version 1.0.5 is a research-use software release; it does not claim clinical use or
completed external scientific validation. Passing software tests does not by itself validate
segmentation, tracking, or force measurements.

StimTrace can process recordings either on the researcher's computer or through Google Colab:

1. `desktop_app.py` is the Windows desktop GUI and contains the persistent Cloud/This computer selector.
2. `local_runner.py` runs the model in a separate local process, automatically selecting NVIDIA CUDA when supported and otherwise using the CPU.
3. `stimtrace_colab_worker.ipynb` runs in the researcher's own Google Colab GPU runtime. The desktop app uploads the notebook, `colab_worker.py`, and approved model to the researcher's Drive on first sign-in.

Supporting modules keep infrastructure concerns out of the GUI and scientific
worker: `background_tasks.py` owns asynchronous Qt/process workers,
`job_utils.py` owns shared job-state, timestamp, path, and result-selection
logic, and `app_logging.py` configures bounded rotating application logs. The
public entry points remain `desktop_app.py`, `local_runner.py`, and
`colab_worker.py`.

## Source repository and release files

The Git repository contains source code, tests, documentation, and build metadata.
It intentionally excludes credentials, user state, recordings, generated results,
trained checkpoints, and compiled application folders.

End users should download the complete versioned Windows ZIP from the repository's
**Releases** page, extract it, and run `StimTrace.exe`. Developers running from
source must provide an approved segmentation checkpoint as `model.pth`. The release
does not contain Google OAuth credentials. Cloud workflows require each user to select
their own OAuth Desktop client JSON file; that file and every user's `token.json` remain
outside Git and outside release archives.

Run the focused regression tests from the project directory with:

```powershell
python -m unittest discover -s tests -v
```

The desktop app never sends Google credentials to another user. Each user signs in with their own Google account via the standard OAuth browser flow. Share the Drive root folder with allowed users, or give each user their own root folder.

## Optional Google Drive and Colab setup

1. In your own Google Cloud project, enable the Google Drive API and create an OAuth
   client with application type **Desktop app**. Use the `drive.file` scope (files
   created by this app only).
2. Download the client JSON file and keep it in a location only you control. Do not
   commit it, send it to collaborators, or place it in the StimTrace installation folder.
3. In StimTrace, select **Sign in** and choose that JSON file when prompted. StimTrace
   stores only the local path; it does not copy or upload the file.
4. Complete browser authorization. For cloud processing, open Colab, select a GPU
   runtime, and run both notebook cells.

Submitted job folders remain in the Drive root and receive a `status.json` plus `results` directory when complete.

## Desktop use

### Prebuilt Windows application

Researchers do not need Python or Visual Studio Code. Copy the complete
`dist\StimTrace` folder to the other Windows computer and start
`StimTrace.exe`. Do not copy only the EXE: its `_internal` folder contains the
scientific runtime, model, and application resources.

StimTrace detects the new computer's CPU and compatible NVIDIA GPU. Settings,
job history, tokens, and downloaded model profiles are stored separately for
each Windows user in `%LOCALAPPDATA%\StimTrace`. Google authorization must be
completed once on each computer when Cloud compute is used; local analysis
does not require a Google account.

The supplied executable is built with the PyTorch variant installed in the
build environment. A CUDA-enabled build also contains CPU support: Auto uses a
compatible NVIDIA GPU and falls back to CPU when CUDA is unavailable. The
CUDA-enabled package is substantially larger than the CPU-only package.

### Run from Visual Studio Code

Open the repository folder in VS Code, install the recommended Python
extension, and select **Run and Debug > Run StimTrace**. The included launch
configuration uses `%USERPROFILE%\emt-env\Scripts\python.exe`. Change the
`python` entry in `.vscode\launch.json` when the environment is elsewhere.

The equivalent terminal commands are:

```powershell
python -m pip install -r requirements-desktop.txt
python desktop_app.py
```

Build a new distributable Windows folder with:

```powershell
.\build_windows.ps1
```

Use `-SkipInstall` after PyInstaller is installed, and `-Clean` only when a
full cache-free rebuild is required. The complete distributable is written to
`dist\StimTrace`.

Build one Windows package that supports both NVIDIA CUDA and CPU fallback with:

```powershell
.\build_universal_windows.ps1
```

This creates an isolated `.venv-universal` environment, installs the official
CUDA-enabled PyTorch runtime, and writes `dist\StimTrace-Universal`. The build
computer does not need an NVIDIA GPU, but a destination computer needs a
compatible NVIDIA driver for GPU acceleration. Run the script again with
`-SkipInstall` to reuse the prepared build environment.

Choose **Cloud** or **This computer**, add videos, enter a study name, and submit the analysis. Cloud mode requires Google sign-in and a running Colab worker. Local mode requires no Google account and saves results directly beside the recordings.

After completion, StimTrace also downloads the result tree to `results/<job ID>` inside the local folder containing the source recordings. The Drive copy remains available for job history and recovery. If StimTrace was closed when processing finished, pending local downloads are resumed after the next sign-in.

During analysis, the desktop shows two graphical progress bars: current-video frames and total weighted job progress with an approximate remaining time. Colab prints occasional batch timing messages but does not render progress bars. Live preview transfer is intentionally disabled so image encoding and Drive synchronization do not slow segmentation.

## Local processing

Choose **This computer** on the Segment page. **Local compute settings** controls automatic/CPU/CUDA selection, CPU threads, inference batch size, postprocessing workers, and overlay generation. Automatic mode uses a compatible NVIDIA CUDA GPU when the installed PyTorch build supports it and otherwise falls back to any supported CPU.

Local processing runs outside the GUI process, so the window remains responsive and cancellation is available from the current-job panel. Frames are streamed in bounded batches. The overlay is generated in a second streamed pass instead of retaining the full video in RAM. Results are written to `results/<job ID>` beside the source recordings and appear in Job History.

The local settings dialog remains available while a job is running. Inference
batch size, postprocessing workers, and overlay generation are applied after
the current inference batch. Device and CPU thread count are fixed for the
running process and take effect on the next job. This avoids unsafe device or
thread-pool changes midway through model execution.

The command-line runner remains available for testing:

`local_runner.py` runs the same model and processing pipeline without Google Drive or Colab. Install the CPU runtime once:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-colab.txt
```

Process only the first eight frames to check compatibility and speed:

```powershell
python local_runner.py "C:\path\recording.avi" --max-frames 8
```

Remove `--max-frames 8` to process the complete recording. By default, results are saved under `results/local_<timestamp>` beside the first selected video. Use `--output`, `--model`, or multiple video paths when needed. To use an NVIDIA GPU, install the PyTorch build matching that computer's CUDA driver. A CPU-only PyTorch installation remains fully supported.

## Job history

Open **Jobs** in the top navigation to review both Cloud and Local jobs, resume monitoring, cancel an active job, retry a failed or cancelled job, or open its results. StimTrace reconnects to the latest active cloud job automatically after sign-in. A local job interrupted by closing StimTrace is marked failed and can be retried from Job History.

## Signal-trace processing

Open **Analyze signals** in the persistent top navigation. The central workspace changes pages without opening another window or discarding the current segmentation page. Load one or more trace CSV or Excel files, select one or more numeric traces, and adjust smoothing, peak prominence, minimum beat interval, analysis range, inversion, time alignment, or baseline correction. Multiple sources are aligned on their numeric time columns, and trace names are prefixed with the source filename. Available baseline methods are none, constant percentile, rolling percentile, and asymmetric least squares.

The analyzer provides:

- Raw and smoothed trace overlays with detected peaks
- Beat frequency, amplitude, contraction and relaxation kinetics, CTD50/CTD90, slopes, and signal-to-noise values
- Baseline assignments and selected normalized summary values
- Excel export with trace summary, beat-level measurements, and analysis settings
- Export of selected processed traces

Newly generated `stimtrace_force_traces.csv` files contain one calibrated `Force_uN`
signal per recording for direct analysis. When recordings use different frame rates,
combined normal and Kalman-benchmark CSVs preserve each recording's original timestamps
on a shared time axis; blank cells indicate times not sampled by that recording. Detailed per-recording
`*_stimtrace_tracking.csv` files retain pixel and micrometer displacement, tracked-center,
ellipse, force, and Kalman innovation-QC data. With Kalman smoothing, StimTrace
automatically applies the fixed 99% covariance-based innovation check with the
original notebook's 120 px minimum acceptance radius and labels larger statistically
implausible segmented centers as
`innovation_rejected`. Their raw centers remain available for inspection, while their
accepted center, displacement, and force values remain blank. Legacy `combined_force_results.csv` and
`*_pillar_displacement.csv` files remain readable. Inversion and baseline correction are
applied on demand in the signal-analysis workspace instead of being stored as duplicate
traces.

Whenever a Kalman benchmark is enabled, StimTrace also writes an `Unfiltered` reference
from the raw segmented centers. The combined benchmark CSV therefore lets every Kalman
variant be compared directly against the same unsmoothed signal.

### Kalman filtering and kinetic measurements

Kalman smoothing can improve visual continuity and reduce the influence of isolated
segmentation errors, but it changes the tracked displacement trajectory. It can therefore
alter peak amplitude, rise time, relaxation time, contraction or relaxation slopes, and
beat timing. For primary kinetic endpoints, use the unfiltered measured trajectory
(`Tracking filter: None`) when segmentation quality is adequate. If a Kalman-filtered
trajectory is used, report the filter settings and validate that they
do not materially change the endpoint of interest. Do not treat a filter-predicted
position as an independent measurement. This tracking choice is separate from any
signal-analysis smoothing or baseline correction, which can also alter kinetic metrics.

## Training a segmentation model

Open **Train models** in the persistent top navigation, or choose **Train new model** from **Models**. The training workspace replaces the central page and keeps the same application ribbon. Create an annotation project by extracting every Nth frame from one or more videos, then draw polygons around the target object. Polygon labels such as `Pillar`, `Tissue`, and `Other` are stored in per-frame JSON files; a binary PNG mask is generated alongside every annotation.

After the minimum saved-annotation and foreground-mask checks pass, the software permits submission to Colab, but those checks do not establish a scientifically adequate dataset. Training starts from the currently approved checkpoint, assigns complete source recordings to separate development and validation sets before frame selection, and uses image augmentation only for development frames. It reports validation loss, Dice, and IoU and saves the checkpoint with the best validation Dice. This remains internal model-development validation, not an independent external performance estimate.

`Default pillar` is preserved as the initial model profile. Selecting a trained checkpoint adds a separate custom profile and regenerates the Colab notebook. Use **Models** to view the active model, object type, source, creation date, dataset size, and parameter summary; activate, import/export, rename, or remove profiles; and edit each profile's independent settings.

All named polygons are currently combined into one foreground class. This supports training separate binary models for pillars or another tissue-related object type. A true multi-class model requires a different output head and is not implied by assigning multiple labels in one project.

Use representative videos from different experiments, reserve independent test videos, and inspect predictions before using a custom model for quantitative conclusions.

## Expert point tracking

Open **Track points** in the persistent navigation to show the integrated optical-flow
workspace inside the main StimTrace window. It supports manual points or automatic feature detection across one or
more videos, reads each recording's FPS metadata by default, accepts calibration in
pixels per micrometer, and calculates absolute displacement, relative point distance,
or cumulative path length. Absolute displacement can reference the initialization
frame, a selected real frame, or the legacy automatically detected relaxed position.

Tracking runs in the background with per-file and total progress. The video canvas
shows live frames, the currently valid tracked points, and the current frame number.
Each recording produces an analyzer-compatible `*_point_tracking_force.csv` containing
only `time_s` and force traces in micronewtons. A separate
`*_point_tracking_displacement.csv` retains the distance traces in micrometers. Force uses
the active model profile's calibrated slope:
`Force_uN = distance_um * force_slope_uN_per_um`. Point Tracking does not estimate a
bending axis, apply an intercept, or select an additional force-zero frame. The model
name and slope are recorded in the provenance JSON. Coordinate CSVs and trajectory
plots are stored separately. Select exported files and choose **Open selected in
Signal Analysis** to load them together in the main analysis workspace.

This is an expert exploratory workflow. It uses forward-backward optical-flow checks, but
its bounded recovery fallback may retain forward-only estimates when every strict check
fails. Users must inspect lost-point gaps and trajectories before using the measurements
quantitatively. Force conversion of relative or cumulative-distance modes is not
mechanically validated.

## Calibration defaults

Defaults mirror the supplied notebook for pixel calibration (`4.35 um/px`) and
uniaxial force slope (`6.14 uN/um`). The Advanced settings dialog changes these
per submitted job, preserving reproducibility in that job's `manifest.json`.

StimTrace projects the tracked pillar center onto one uniaxial bending axis; it
does not combine independent minimum x and y coordinates. The default **Active
force** mode estimates that axis from the recording, selects one actual frame
nearest the robust median diastolic position as the optical zero, and computes
`Force_uN = slope * displacement_um`. Absolute passive or preload force is not
reported because the acquisition does not provide an unloaded optical reference.
A known bending direction can be entered as a fixed image angle (0 degrees right,
90 degrees down). Segmented-area normalization is retained only
as an explicitly labelled legacy option and is disabled by default because it
requires separate experimental validation. The individual displacement CSVs retain
the selected diastolic reference frame, bending axis, calibration values, and
equation alongside every measured trace.

Performance controls are also available under Advanced settings. Cloud batch
sizes up to 4096 and up to 32 CPU postprocessing workers can be requested for
the next submitted job. The worker probes progressively larger batches up to
the requested maximum and backs down after a GPU-memory failure. The effective
worker count is capped at the runtime's logical CPU count minus one. Batch size
consumes GPU VRAM rather than Colab system RAM, so 50 GB of system RAM does not imply that a batch will
fit on the GPU. The connected-runtime row reports the detected GPU, GPU VRAM,
system RAM, logical CPU count, sampled GPU utilization, and power draw.

The Colab workflow step reports **Worker online** only while a fresh heartbeat is received from the running notebook. The input table lists duration, frame count, file size, upload state, and missing files before submission.

## Licence, attribution, and citation

StimTrace is licensed under the [MIT License](LICENSE).
© 2026 ETH Zurich. See
[NOTICE.md](NOTICE.md), [AUTHORS.md](AUTHORS.md), and [CITATION.cff](CITATION.cff).

If you use StimTrace in research, please cite the software using the repository's
GitHub **Cite this repository** entry. The acknowledgement/citation request
complements, but does not replace, the legal licence terms.

Portions of the point-distance implementation were adapted from Cell Motion Tracker;
see [NOTICE.md](NOTICE.md). Licences and notices supplied with the release
environment's Python dependencies are retained in `THIRD_PARTY_LICENSES`.
