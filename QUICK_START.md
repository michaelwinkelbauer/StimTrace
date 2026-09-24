# StimTrace quick-start guide

StimTrace segments contraction recordings, converts pillar movement into force traces,
analyzes contraction signals, tracks manually or automatically selected points, and can
train custom binary segmentation models.

## 1. Start StimTrace

1. Keep the complete StimTrace distribution folder together. Do not move only
   `StimTrace.exe`; the `_internal` folder contains the model and scientific runtime.
2. Start `StimTrace.exe`.
3. Use the navigation bar to switch between **Segment**, **Analyze signals**,
   **Track points**, **Train models**, **Models**, and **Jobs**.

Local analysis does not require a Google account. Cloud segmentation and model training
require Google sign-in and a running Google Colab worker.

### Optional Google Drive setup

StimTrace does not include Google OAuth credentials. Before using cloud features, create
an OAuth client of type **Desktop app** in your own Google Cloud project after enabling
the Google Drive API, then download its JSON configuration file. When you select
**Sign in** in StimTrace, choose that file when prompted. The file stays where you chose
to keep it; StimTrace stores only its local path. Do not commit, distribute, or place the
file in the StimTrace installation folder.

## 2. Calibrate the experiment

Open **Segment > Advanced settings**. Calibration and processing settings are stored
separately for each model profile.

### Pixel calibration

Pixel calibration converts image movement to physical distance in `um/px`.

1. Add the recordings for the analysis first when convenient.
2. In **Advanced settings**, select **Measure from video** beside **Pixel calibration**.
   If the analysis list contains at least three recordings, StimTrace automatically
   loads three randomly selected recordings. Otherwise, select exactly three videos.
3. Use videos recorded with the same camera, magnification, resolution, and zoom.
4. Choose what the known physical distance represents: ellipse width, height, mean
   diameter, or perimeter.
5. Enter the matching known distance in micrometers.
6. Left-drag an ellipse around the calibration object and adjust its handles. Use the
   wheel to zoom and right- or middle-button drag to pan.
7. Record three different frames from each of the three videos, for nine measurements
   total.
8. Review the mean and variability, then select **Use calibration**.

The selected pixel measurement and entered physical distance must describe the same
dimension. A high coefficient of variation suggests inconsistent ellipse placement or
recordings with different optical settings.

### Force calibration

Enter the model- and setup-specific **Force slope** in `uN/um`. Segmentation reports
active force relative to an optical diastolic reference:

`Force_uN = displacement_um x force_slope_uN_per_um`

## 3. Segment recordings

### Common setup

1. On **Segment**, confirm the active segmentation model.
2. Select **Cloud** or **This computer**.
3. Select **Add videos** and add the recordings.
4. Enter a descriptive, unique **Study name**.
5. Review calibration and model-specific settings under **Advanced settings**.
6. Submit the analysis and follow the current-file and total-job progress bars.

### Cloud processing with Colab

Use Cloud when the local computer has no suitable GPU or when Colab is faster.

1. Select **Cloud**, then sign in to the Google account used for Colab.
2. Select **View Colab**. In Colab, choose a GPU runtime (A100 > L4 > T4) and run both StimTrace cells.
3. Wait until the notebook says the worker is online and watching the queue.
4. Return to StimTrace, add videos, and submit the job.
5. Keep StimTrace and the Colab worker open while uploading and processing.

Cloud jobs are stored in the personal StimTrace Drive workspace. Completed results are
also downloaded to a local `results/<job ID>` folder beside the source recordings. Under
**Advanced settings**, cloud overlays default to local CPU rendering after the tracking
results download; Colab rendering remains available. Local rendering reduces Colab work
but requires the original videos to remain accessible.

If Colab stops or its runtime crashes, restart the runtime, run both cells again, and
retry the failed job from **Jobs**. Checkpointed per-recording results are reused, so a
retry continues with missing recordings rather than intentionally repeating completed
ones. Videos without a usable pair of pillar-center detections are reported as skipped.

### Local processing

Use **This computer** for offline processing and direct access to local files.

1. Select **This computer**.
2. Open **Local compute settings** if you need to change the device or performance
   settings. **Automatic** uses a compatible NVIDIA CUDA GPU when available and otherwise
   falls back to CPU.
3. Add videos, enter a study name, and select **Run locally**.

Local processing runs in a separate process so the interface remains responsive. Results
are written to `results/<job ID>` beside the source recordings and recorded in **Jobs**.
GPU inference is normally much faster than CPU inference; overlay encoding and some
postprocessing remain CPU tasks.

### Review outputs

Use the current-job buttons or **Jobs** to:

- Open the Drive job folder for cloud jobs.
- Open the local results folder.
- Resume monitoring or retry a failed job.
- Open completed force traces in **Analyze signals**.

The detailed tracking CSV files contain centers, ellipse measurements, displacement,
calibration, force, and Kalman innovation quality-control fields. The combined force-trace
CSV is intended for signal analysis.

## 4. Analyze signals

1. Open **Analyze signals**.
2. Open one or more StimTrace CSV or Excel files. Additional selections can be added to
   or replace the current workspace.
3. Select the traces to display and analyze.
4. Set the expected minimum beat interval, peak prominence, smoothing, and analysis time
   range. Invert a trace if contractions point downward.
5. If required, apply baseline correction:
   - **Constant percentile** removes one resting offset.
   - **Rolling percentile** follows slow baseline drift.
   - **Asymmetric least squares** fits a smooth baseline while discounting contraction
     peaks.
6. Select **Analyze selected signals** and inspect the detected peaks, beat boundaries,
   corrected/smoothed traces, and **Summary** tab.
7. Export metrics or processed traces. Export dialogs begin in the folder from which the
   source data was loaded.

### Reference traces and normalized metrics

Reference traces are controls for percentage comparisons; they are not subtracted from
the analyzed signal.

1. Select no more than one reference trace for each specimen.
2. Choose **Set as reference trace**.
3. StimTrace matches references by specimen identity, ignoring processing suffixes such
   as `smoothed` and `corrected`.
4. Run the analysis again to calculate normalized summary metrics for matched specimens.

Baseline correction and reference normalization solve different problems. Baseline
correction removes offset or drift within a trace. Reference normalization expresses a
sample metric relative to its matched control. Always inspect the match-status column
before interpreting normalized values.

Exported metrics include beat rate, amplitude, time to peak, rise time, relaxation time,
CTD50/CTD90, maximum contraction and relaxation slopes, and beat-level measurements when
the detected beats support those calculations.

## 5. Track points without segmentation

**Track points** is an optical-flow workflow for exploratory motion tracking.

1. Add one or more videos.
2. Confirm the frame rate. Use the video metadata unless it is missing or known to be
   wrong.
3. Enter calibration in `px/um`. This is the reciprocal of segmentation's `um/px` value:
   `px/um = 1 / (um/px)`.
4. Choose manual points or automatic features:
   - **Manual points:** select a video and left-click the initialization frame to add
     points; right-click removes the last point.
   - **Automatic features:** draw a rectangular region of interest around the object.
     Apply it to all recordings only when their framing is comparable.
5. Choose absolute displacement, relative distance, or cumulative path length. Relative
   distance uses the first two points.
6. Run tracking and inspect the overlay, trajectories, and any lost-point gaps.

The active model profile supplies the force slope. Outputs include detailed displacement
and coordinate CSV files, overlay videos, plots when enabled, and analyzer-compatible
force CSV files. Point tracking does not estimate a pillar bending axis or an additional
force-zero frame, so validate the measurement geometry before quantitative use.

## 6. Train a new segmentation model

Training currently uses the Colab worker.

### Create the dataset

1. Open **Train models**.
2. Sign in and ensure the current Colab notebook worker is online.
3. Choose a dataset size:
   - Minimum: about 60 frames from 15 recordings.
   - Recommended: about 125 frames from 25 recordings.
   - Robust: about 200 frames from 40 recordings.
4. Select **Create training dataset**, select representative videos, and choose a new
   annotation-project folder.

StimTrace chooses diverse recordings and frames covering relaxed, transition, maximum
deformation, and recovery appearances. Complete recordings are reserved for independent
quality checking. Keep genuinely independent test videos outside the annotation project
for final validation.

### Annotate frames

1. Select the object class, normally **Pillar** for a pillar model.
2. Left-click to add polygon points. Double-left-click or **Finish polygon** completes the
   polygon. Right-click undoes the latest point or polygon.
3. Use the wheel to zoom and middle-button drag to pan. Use **Reset zoom** to fit the
   complete frame.
4. Select **Save annotation and open next frame**. Moving to another frame also saves
   completed annotation work.
5. For a valid negative-control frame with no pillar, leave the mask empty and save the
   annotation. An explicitly saved empty mask counts as annotated; an untouched frame
   does not.
6. Green list entries are saved annotations, including empty negative controls. Red
   entries are not yet annotated.

**Auto-mask next frame** and copied masks are editable drafts. Always inspect and correct
their boundaries before accepting them. All current polygon labels are combined into one
foreground class; labels do not create a multi-class network.

### Submit and use the model

1. Enter a unique model name.
2. Start with the default **30 epochs** and **batch size 4** for an initial training test.
   Increase batch size only when GPU memory permits; more epochs are useful only while
   validation performance continues to improve.
3. Select **Submit training to Colab** and keep the worker running.
4. Review validation Dice, IoU, losses, phase coverage, and quality-control predictions.
5. After completion, StimTrace downloads the checkpoint and reports into
   `<annotation project>/system/training/<job ID>`. A cloud copy remains in the Drive
   training folder.
6. Select **Use trained model for segmentation**, or open **Models** later and activate
   the model profile.
7. Return to **Segment**, confirm the new model name, review its calibration/settings,
   and test it on independent recordings before quantitative use.

The built-in **Default pillar** model is retained. A newly trained model becomes a
separate profile; it does not overwrite the default and is not automatically included in
future application builds.

## 7. Recommended first-run workflow

1. Calibrate pixel size with three matched recordings.
2. Confirm the force slope and bending-axis assumptions.
3. Segment a short test recording locally or in Colab.
4. Inspect its overlay and detailed tracking CSV before processing a large dataset.
5. Analyze the force trace and tune peak-detection settings while viewing the plot.
6. Save the job results, analysis settings, calibration information, and active model
   name with the experiment.

With Kalman smoothing enabled, **Advanced settings** includes an innovation gate. It is
enabled at 99% by default with the original notebook's 120 px minimum acceptance radius.
Frames labelled `innovation_rejected` retain the raw segmented center for review but have
blank accepted-center, displacement, and force values. Do not treat the filter's internal
prediction as a measured center. Set the minimum radius to zero only when Q and R have been
calibrated for a pure covariance gate on representative recordings.

Kalman smoothing changes the tracked displacement trajectory and can alter peak amplitude,
rise/relaxation timing, and contraction or relaxation slopes. For primary kinetic
measurements, use **None (raw segmentation centers)** when segmentation quality is
adequate. If using Kalman smoothing, record the filter and gate settings and verify that
they do not materially change the endpoint of interest. This choice is separate from
signal-analysis smoothing and baseline correction, which can also change kinetic metrics.

Use the **Help** button on each page for control-specific instructions. Use **Jobs** as the
central place to find, retry, download, and reopen previous segmentation and training
jobs.
