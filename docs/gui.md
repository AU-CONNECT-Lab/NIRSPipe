---
title: GUI
---

# GUI

`nirspipe-gui` is a Dash-based desktop interface for BIDS conversion, data preparation, batch editing, dyad alignment, pipeline launching and QC reporting. It is a graphical front-end for the same CLI commands: anything you do in the GUI maps onto a `nirspipe-recon`, `nirspipe-prep`, `nirspipe`, `nirspipe-hyper` or `nirspipe-qc` invocation. The sidebar groups the pages by which of those they drive.

```bash
nirspipe-gui                  # port 8050, or the next free one above it
nirspipe-gui --port 8051      # exactly this port, or stop if it is taken
```

Open the address it prints (`http://localhost:8050` by default) in your browser. The sidebar is collapsible (click the ☰ button).

## Pages

| Page | Purpose | CLI analogue |
|------|---------|--------------|
| **Data Preparation** | Single-subject, single-run inspection, channel rating, marker editing, crop | `nirspipe-qc prep-raw` + manual edits |
| **Recon** | Batch-convert a folder of raw SNIRF files into a BIDS dataset | `nirspipe-recon`, once per file |
| **Batch Prep** | Apply marker edits / crops / keep-spans tables / dyad alignment to many subjects at once | `nirspipe-prep` |
| **Hyper Preparation** | Align dyad recordings by shared trigger, inspect the alignment, rate channels, export | `nirspipe-qc hyper-raw` interactively, plus `nirspipe-prep align` |
| **Analysis** | Configure and launch the full preprocessing + postprocessing pipeline | `nirspipe` |
| **Hyper Analysis** | Wavelet coherence and inter-subject correlation over dyads, run on what the pipeline wrote | `nirspipe-hyper` |
| **Cohort Reports** | Cohort aggregation over the quality records, provenance graphs | `nirspipe-qc` |

## When to use the GUI

- **Use the GUI for:** interactive data exploration, marker fixing, channel quality decisions, configuring a one-off run.
- **Use the CLI for:** unattended batch processing, reproducible scripted runs, HPC submission.

The GUI never runs preprocessing in its own process: the **Analysis** page builds a `nirspipe` command and either previews it or hands it off (see [Analysis](#analysis)). **Hyper Analysis** and **Cohort Reports** build a `nirspipe-hyper` or `nirspipe-qc` command and run it as a subprocess. **Batch Prep** builds a `nirspipe-prep` one and streams its output. **Recon** is the one page that does its work in process.

## Shared state across pages

The sidebar persists `bids_dir` and `output_dir` in browser session storage, so switching pages keeps your paths filled in. Per-page state (loaded subject, marker edits) lives in memory only: refreshing or restarting `nirspipe-gui` clears it. Channel decisions are the exception: the Data Preparation and Hyper Preparation pages write each one as it is made to the subject's `desc-rawdecision_qc.json` under the output directory, the same file `nirspipe-rate raw` reads.

:::{note}
The GUI runs in `debug=False`. Tracebacks are not shown in the browser; check the terminal where `nirspipe-gui` was launched for any errors.
:::

## Data Preparation

The **Data Preparation** page loads one subject × one run, lets you inspect signal quality, rate channels good/bad, edit annotations (markers), and crop the recording. It is the GUI counterpart to `nirspipe-qc prep-raw` plus manual editing.

URL: `http://localhost:8050/`

---

### Data Source

| Field | Required | Notes |
|-------|----------|-------|
| **BIDS Directory** | Yes | Path to the BIDS dataset root. |
| **Derivatives Directory** | Yes | Where channel decisions, marker-edited SNIRFs, cropped SNIRFs and the raw QC report are written. |

Both paths are remembered across pages.

### Subject Selection

Type a subject ID into **Enter Subject ID** (e.g. `11`) and click **Detect**. The page scans the BIDS directory and lists all SNIRF runs for that subject. Selecting a subject populates the **Run** dropdown.

### Run

| Field | Default | Notes |
|-------|---------|-------|
| **Run dropdown** | — | One entry per `(session, task, run)` triplet found. |
| **SCI Threshold** | required | The SCI line a window has to clear, e.g. `0.8`; starts empty, and a run does not load without it. It pairs with PSP inside each window, and what rejects is the share of windows a channel is coupled in. |
| **DPF** | `6.0` | Differential pathlength factor, used to convert to haemoglobin for the panels that need it. |
| **Cardiac Band (Hz)** | required | The band SCI, PSP and cardiac power are measured in. Population-dependent. |
| **SCI / PSP Window (s)** | `10` | The window grid those series, and the screening count, are measured on. |
| **Separations (mm)** | `10` / `15` / off | Short, long and the upper long bound, the same split the pipeline uses. |
| **Epoch Window (s)** | -5 / 25 | Trial window for the epoch panels. |

Selecting a run loads the SNIRF and computes the SQM and figures used by both tabs below.

---

### Viewer tab

Interactive panels for inspecting raw signal, optode geometry, and editing markers.

#### Signal Topo
Raw HbO / HbR per channel arranged by topographic position. Click a trace to select the channel; the selection is shared with **Channel Detail** and **PSD**.

#### Optode Layout 2D
2D source-detector map. Each channel is coloured by its windowed SCI, the `SCI (10 s)` of the QC Metrics table rather than the whole-run SCI, against the **SCI Threshold**: green at or above it, amber down to 0.625 of it, red below that. A rejected channel is grey whatever its SCI. Click a channel to select it.

#### 3D (fsaverage)
The same channel layout projected onto the fsaverage cortical surface, each line running from source to detector and each channel coloured long, short or unclassified by separation. View-only.

#### Channel Detail
HbO and HbR time series for the currently selected channel. The dropdown also lets you pick a channel pair manually.

#### PSD
Power spectral density of the selected channel, or the mean across channels when no channel is selected.

#### Crop
Two modes:

- **Single segment**: pick `tmin` and `tmax` in seconds. **Use zoom** snaps the values to the current zoom range of the trigger timeline.
- **Multi-segment**: add `(onset, duration, task)` rows. **Combine into one file** concatenates them back-to-back into a single cropped SNIRF; otherwise each segment is exported as a separate SNIRF named by its Task cell, which every row then needs, each one different and letters and digits only.

Click **Apply** to write the cropped SNIRF(s) to the `cropped/` tree inside the derivatives directory.

#### Markers

Edit the SNIRF annotation table inline.

| Action | Effect |
|--------|--------|
| **Step (s)** + **− All / + All** | Shift every marker onset by ± step seconds (clipped to 0). |
| **− Sel / + Sel** | Same shift, applied only to selected rows. |
| **Rename: from → to + Apply All** | Batch-rename one description to another. |
| **+ Add** | Insert a new marker row. |
| **↓ Export TSV** | Download the current marker table as `events.tsv`. |
| **Save** | Write the annotations into a copy of the SNIRF in the `marker_edited/` tree inside the derivatives directory. |

Under the table the page states the drift cutoff a GLM on this run could use, with the slowest repeat of every condition beside it.

#### Epoch Preview
Stimulus-locked epoch average for the selected channel.

---

### QC Metrics tab

The same content as `nirspipe-qc prep-raw`, embedded in the GUI.

#### Raw Signal
All-channel time series. Blue traces are short channels, orange are long. Click a trace to select; double-click resets.

#### Carpet + GVTD
The raw QC motion panel: the correction strip and the IMU rows where the recording has them, a global variance of temporal derivatives (GVTD) row per channel set, and the OD carpet, each row linearly detrended and then z-scored. See [QC Figures](qc.md#the-run-report).

#### SCI / PSP
Two-row heatmap: scalp coupling index (top) and peak spectral power (bottom), windowed across the recording.

#### Image Quality Metrics
Scalar SQM values from `compute_raw_sqm` (see [the quality record](qc.md#the-run-report)), labelled and coloured by the same metric registry the subject report reads, so a metric that passes here passes there.

On a montage with short channels the panel reads the **long channels only**, with an All / Long / Short comparison below it.

#### Channel Quality Summary
Per-channel pass/fail grid for Status, Coupled, and SCI, CV, PSP and SNR over 10 s windows, long block first with a divider before the short one.

#### Per-channel Metrics
One row per source-detector pair: status, `SCI (10 s)`, `SCI (whole run)`, PSP, SNR and CV, grouped by separation, with the rating decision as the last column. Click each chip to cycle the channel state: `—` (undecided) → `good` → `bad`. Auto-saved to the run's `desc-rawdecision_qc.json` under the subject's `nirs/` in the derivatives directory; persists across runs.

---

### What gets written

| Action | Output path |
|--------|-------------|
| Save markers | `<output_dir>/marker_edited/sub-<id>/[ses-<id>/]nirs/<run-name>_nirs.snirf`, with its sidecars |
| Export TSV | A browser download, `events.tsv`; nothing is written to `<output_dir>` |
| Crop Apply | `<output_dir>/cropped/sub-<id>/[ses-<id>/]nirs/<run-name>_nirs.snirf`, or one file per segment with its Task cell in place of the run's `task-` |
| Channel decisions (auto) | `<output_dir>/sub-<id>/[ses-<id>/]nirs/sub-<id>[_ses-<id>]_task-<task>_desc-rawdecision_qc.json` |
| Write QC report | `<output_dir>/sub-<id>/<run-name>_desc-raw_report.html`, its figures under `sub-<id>/figures/` and its quality record `sub-<id>/nirs/<run-name>_desc-sqmraw_qc.json` |

#### Write QC report

The Setup card's Report section runs `nirspipe-qc prep-raw` on the loaded run with the parameters above it, and embeds the report it produced. Everything the Viewer and QC Metrics tabs draw lives in the browser only; this is what files a copy.

It runs when you ask for it, never when the page opens, and needs **DPF** and both ends of the **Cardiac band**.

:::{note}
Raw SNIRFs in the BIDS dataset are never modified. All edits are written to the output directory as new files.
:::

## Hyper Preparation

The **Hyper Preparation** page is dyad-level quality control: it aligns the recordings of a group to a shared trigger, shows you the result, lets you rate channels, and writes what you decide. It is the dyad counterpart of [Data Preparation](#data-preparation), and the interactive form of `nirspipe-qc hyper-raw`.

The alignment itself is the same one `nirspipe-prep align` and `nirspipe-hyper` perform. Loading aligns every group in the CSV at once and keeps the result in memory; nothing reaches disk until you export.

URL: `http://localhost:8050/hyper-align`

---

### Data Source

| Field | Notes |
|-------|-------|
| **BIDS Directory** | Source SNIRFs. |
| **Derivatives Directory** | Where aligned SNIRFs, channel decisions and the QC report are written. |
| **Group CSV** | CSV with columns `group_id, subject_id, task`, and the optional `session` / `run` columns that pick one recording where a member has several. See [BIDS Layout](data-preparation.md#pairs-file). |

Click **Load & Align** to run alignment for every `(group_id, task)` listed in the CSV. The page calls the same `align_recordings` routine used by the CLI:

1. Find the set of annotation descriptions shared across all subjects in the group.
2. For each subject, crop from the first occurrence of any shared trigger.
3. Trim all subjects to the duration of the shortest one after cropping.

See [Hyperscanning Pipeline](hyperscanning.md#alignment) for the algorithm details and failure modes.

---

### Results

After **Load & Align** finishes, the results panel shows four cards.

#### Alignment Offsets

A flat table with one row per `(group, subject, task)`:

| Column | Notes |
|--------|-------|
| **Group** | `group_id` from the CSV. |
| **Subject** | `subject_id` from the CSV. |
| **Task** | `task` from the CSV. |
| **Offset (s)** | Time of the first shared trigger in the original recording: this much was cropped from the start. |
| **Duration (s)** | Recording length after trimming to the shortest. |

#### Trigger Timeline

Pick a group from the dropdown, one entry per group, task and, where the CSV has them, session. The figure shows the aligned (post-crop) time axis with one row per subject, with markers drawn at each annotation. Use this to sanity-check that the chosen shared trigger lines up.

#### Signal Overlay

HbO traces for all subjects in the selected group, plotted on the aligned time axis. First long channel by default; switch to inspect a specific channel.

#### Channel Decisions

Per-subject good / bad chip table. Click to cycle: `—` → `good` → `bad`. Auto-saves each member's decisions to `sub-<id>/nirs/sub-<id>[_ses-<ses>]_task-<task>_desc-rawdecision_qc.json` in the derivatives directory. The session is the one of the file that member was read from, so on a tree with several sessions the decisions land in the same file the raw QC page reads for that run.

The card also holds the **Cardiac band (Hz)** the per-channel quality panels are measured in (population-dependent, adult resting ~0.7 to 1.5), the **DPF**, and the **SCI threshold** (required, no default). The decisions table colours SCI against it, and leaves SCI uncoloured while it is empty.

---

### Export

Two buttons, writing different things.

| Button | Writes | CLI equivalent |
|--------|--------|----------------|
| **Export aligned SNIRFs** | `derivatives/aligned/`, one recording per member, its offset recorded in its own `_nirs.json` | `nirspipe-prep align` |
| **Write QC report** | The static dyad QC report (`group-<gid>/group-<gid>_task-<task>_desc-raw_report.html`) and its quality record, for the group picked in the Trigger Timeline, or every group when none is, into the **Dyad QC directory**: a raw QC tree of its own or the `nirspipe-hyper` tree | `nirspipe-qc hyper-raw` |

The report needs **DPF** and both ends of the **Cardiac band**, which sit in the Channel Decisions card. The report is refused until both are given. It is written when you ask for it, not when the page opens.

:::{important}
`nirspipe-hyper` does not read `derivatives/aligned/`. It reads the individual pipeline's output and aligns again in memory, so exporting here is for inspection, archiving and tools outside this package rather than a step the dyad analysis depends on.
:::


**Export aligned SNIRFs** writes one file per subject into `derivatives/aligned/`, with the alignment crop baked in, named after the file that member was read from and placed in its `ses-` folder. The member's `_nirs.json` gains `align_group`, `align_step`, `align_trigger`, `align_offset_s` and `aligned_duration_s`; no separate offsets table is written. They are the same files `nirspipe-prep align` writes, and they carry the same alignment `nirspipe-hyper` computes for itself.

The tree holds one copy per recording, so a member that two groups in the CSV share for the same task is refused for the second group and listed in the status line, rather than overwritten with that group's cut. Export such groups into separate derivatives directories. Exporting the same group again overwrites its own copies.

:::{warning}
`--no-align` does not mean "use the exported tree". It replaces alignment with trimming every member to the shortest, which is a different operation. There is no flag that makes the dyad analysis read `derivatives/aligned/`.
:::

### Failure modes

| Symptom | Cause |
|---------|-------|
| `AlignmentError: No non-BAD annotations found` | One or more subjects have no markers (excluding `BAD_*`). |
| `AlignmentError: No shared trigger descriptions` | The subjects in this group don't share any common annotation description. Use [Data Preparation → Markers](#markers) to rename triggers into a common scheme first. |
| `... is already aligned for group ..., and group ... would overwrite it` | On export: another group in the CSV already aligned this member's recording into the same tree. Export the groups that share a member into separate derivatives directories. |

## Recon

The **Recon** page converts a folder of raw `.snirf` files into a BIDS dataset, one row per file. It is the GUI equivalent of running [`nirspipe-recon`](usage.md#nirspipe-recon) once per recording.

URL: `http://localhost:8050/recon`

This is the only page that performs its work in its own process. **Run All** calls the same `write_bids_from_snirf` the CLI calls, rather than handing you a command to paste. The command preview is still there for when you would rather run the conversion in a terminal or keep it in a script.

---

### Data Source

| Field | Notes |
|-------|-------|
| **Input Folder (raw snirf)** | Searched recursively for `*.snirf`. |
| **Output BIDS Directory** | `BIDS_DIR` positional. Created if it does not exist. |

### Files

**Detect** lists every `.snirf` found under the input folder and fills one row per file:

| Column | CLI flag | Editable | Notes |
|--------|----------|----------|-------|
| `File` | (the positional input file) | no | Filename only. The full path is kept behind the row. |
| `subject` | `--participant-label` | yes | Required. Pre-filled from the filename with every non-alphanumeric character stripped, since BIDS labels allow nothing else. Check it before running. |
| `task` | `--task-label` | yes | Required. Blank by default, so this is the column you always have to fill in. |
| `session` | `--session-label` | yes | Optional. |
| `run` | `--run-label` | yes | Optional. |

**Overwrite existing** adds `--overwrite` to every row.

A row with no `subject` or no `task` is skipped: it produces no command in the preview and is counted as a failure when run.

### Command Preview

One `nirspipe-recon` invocation per row, in the line-continuation style of the shell you pick (`bash` / `zsh`, `cmd`, PowerShell; the default follows the platform). Paths are quoted, so a folder with spaces pastes correctly.

```bash
nirspipe-recon "/data/raw/sub01_run1.snirf" \
  "/data/bids" \
  --participant-label 01 \
  --task-label tapping
```

### Run

**Run All** converts every valid row in order and reports per file:

- `✓ <file> → sub-<label>` for each conversion that succeeded
- `✗ <file>: <error>` for each one that did not, with the exception message

One failing file does not stop the rest. The banner at the top gives the totals.

:::{tip}
Convert a couple of files first and open the resulting tree before running the whole folder. A wrong `subject` or `task` here propagates into every derivative and QC report downstream, and the fix is a re-conversion.
:::

---

## Batch Prep

The **Batch Prep** page applies the same data-preparation operation (marker editing, crop, a keep-spans table, or dyad alignment) to many subjects at once, without running preprocessing. Use it when you already used [Data Preparation](#data-preparation) on one subject to decide on the edit, and now want to roll it out across the cohort.

URL: `http://localhost:8050/batch-prep`

---

### Data Source

| Field | Notes |
|-------|-------|
| **BIDS Directory** | Source SNIRFs. |
| **Derivatives Directory** | Where edited SNIRFs are written. |

### Subjects

Click **Detect** to scan the BIDS directory. Use **Select All** / **Clear** to manage the multi-select chip list.

### Run Filter (optional)

| Field | Notes |
|-------|-------|
| **Session** | Restrict to one session label, e.g. `01`. |
| **Task** | Restrict to one task label, e.g. `tapping`. |
| **Run** | Restrict to one run label, e.g. `01`. |

Leave blank to apply the operation to every run of each selected subject.

---

### Operation

Four mutually exclusive operations:

#### Edit Markers

Three sub-operations:

| Sub-op | Inputs | Effect |
|--------|--------|--------|
| **Shift onsets** | Shift (s) | Adds the value to every marker onset; clipped to 0. |
| **Set duration** | Duration (s) | Overwrites the duration of every marker. |
| **Rename markers** | From / To rows | Batch-renames each `from_name` to `to_name`. |

#### Crop

Two modes:

- **Single segment**: `tmin` and `tmax` in seconds.
- **Multi-segment**: `(onset, duration, task)` rows and a **Combine into one file** toggle, with the same rules as the [Data Preparation crop card](#crop).

#### Keep-spans Table

Writes a table for `nirspipe --keep-spans` holding every occurrence of the named conditions in the selected recordings, one row each, onsets on each recording's own time axis. No SNIRF is edited or written.

| Field | Notes |
|-------|-------|
| **Conditions** | Annotation names, separated by commas, spelled as the markers spell them. A selected recording holding none of them stops the command. |
| **Duration (s)** | Optional. Gives every occurrence this length instead of its own; needed when a marker has no length, and worth giving when the markers carry a nominal length (1 s, say) rather than the block's. |

The table is written to `keep-spans.tsv` in the derivatives directory, and nothing is written unless every selected subject succeeds. See [`nirspipe-prep keep-spans`](usage.md#nirspipe-prep) for the columns and the checks, and [Excluded time](workflows.md#excluded-time) for what the pipeline does with the table.

#### Hyperscanning Align

Aligns dyad recordings by shared trigger, in batch.

| Field | Notes |
|-------|-------|
| **Group CSV** | CSV with columns `group_id, subject_id, task`, plus optional `session` and `run` columns naming which recording of a member to read. Each `(group_id, task)` pair is aligned independently. |

The aligned copies are written to an `aligned/` BIDS tree inside the derivatives directory, each under its source file's own name and `ses-` folder (`aligned/sub-<id>/nirs/sub-<id>_task-<task>_nirs.snirf` with its `_nirs.json`, `_events.tsv` and `_channels.tsv`). Each member's offset is recorded in its own `_nirs.json` (`align_group`, `align_step`, `align_trigger`, `align_offset_s`, `aligned_duration_s`); no separate offsets table is written. A member two groups share for the same task is refused rather than overwritten by the second group.

See [Hyperscanning BIDS Layout](data-preparation.md#pairs-file) for the CSV format.

---

### Run

| Field | Default | Notes |
|-------|---------|-------|
| **Parallel jobs** | 1 | Number of subjects processed concurrently. |
| **Generate command** button | — | Builds the `nirspipe-prep` command and prints it in the Command panel. |
| **Run batch** button | — | Runs the command that was generated, as a subprocess. |
| **Stop** button | — | Asks the running command to end. |

The output streams into the log panel while the command runs, so a long batch shows progress rather than freezing the page. Run and Generate are ordered: **Run** uses the argv from the last **Generate**, not the current form, so an edit made after generating is not in what runs.

### Command

Everything this page does, `nirspipe-prep` also does, so the form has an exact command-line equivalent:

| Operation | Builds |
|-----------|--------|
| Edit markers | `nirspipe-prep edit-markers apply BIDS DERIV --participant-label ... --shift` / `--set-duration` / `--rename old:new` |
| Crop, single segment | `nirspipe-prep crop BIDS DERIV --participant-label ... --tmin --tmax` |
| Crop, multi-segment | `nirspipe-prep crop BIDS DERIV --participant-label ... --segments-path ... [--combine]` |
| Keep-spans table | `nirspipe-prep keep-spans BIDS DERIV/keep-spans.tsv --participant-label ... --conditions ... [--duration]` |
| Hyperscanning align | `nirspipe-prep align BIDS DERIV --group-csv ...` |

The selected subjects become `--participant-label`, the run filter becomes `--session-label` / `--task-label` / `--run-label`, and parallel jobs becomes `--n-jobs`. Alignment takes none of those: its subjects come from the group CSV.

:::{note}
Multi-segment crop is the one case that needs a file. **Generate command** writes the segment table to `batch-crop-segments.tsv` in the derivatives directory and points `--segments-path` at it, so the command it prints can be run as it stands.
:::

:::{warning}
Batch Prep does not run preprocessing: it only edits, crops or aligns SNIRFs, or writes a keep-spans table. To run the full pipeline across subjects use the [Analysis page](#analysis) or the `nirspipe` CLI directly.
:::

## Analysis

The **Analysis** page configures and launches the full `nirspipe` pipeline (preprocessing + optional postprocessing) and visualizes the resulting DAG. It is the GUI equivalent of the `nirspipe` CLI.

URL: `http://localhost:8050/analysis`

:::{note}
**Run Pipeline** launches the command in the background and shows its output on the page as it arrives. Closing the browser tab does not stop the run, but nothing will be watching it: use **Stop** if you want it to end.
:::

---

### Data Source

| Field | CLI flag |
|-------|----------|
| **BIDS Directory** | `bids_dir` (positional) |
| **Output Directory** | `output_dir` (positional) |
| **Detect Subjects** | Scans BIDS, populates the subject list. |

### Subject Selection

The subject chip list comes from the BIDS scan. Filters:

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Session label(s)** | `--session-label` | Space-separated; optional. |
| **Task label(s)** | `--task-label` | Space-separated; optional. |

Selecting subjects translates to repeated `--participant-label` flags.

### Preprocessing

| Field | CLI flag | Default | Notes |
|-------|----------|---------|-------|
| **DPF** | `--dpf` | — | Differential pathlength factor. Required. |
| **SCI Threshold** | `--sci-threshold` | required | The SCI line a window has to clear to count as coupled, e.g. `0.8`. Starts empty. |
| **PSP Threshold** | `--psp-threshold` | `0.1` | The PSP line for the same window. A window counts as coupled only when it clears both lines. |
| **Coupled Windows** | `--min-good-frac` | `0.75` | Share of windows a channel has to be coupled in to be kept. This is the criterion that rejects. |
| **Screening Scope** | `--screen-scope` | `run` | Which windows count toward it: the whole run, or the annotated task blocks only. |
| **SCI / PSP Window (s)** | `--window-length` | `10` | The grid those series, and the screening count, are measured on. |
| **Separations (mm)** | `--short-max-dist` / `--long-min-dist` / `--long-max-dist` | `10` / `15` / off | The bands the montage is split by. |
| **Epoch tmin / tmax (s)** | `--epoch-tmin` / `--epoch-tmax` | -5 / 25 | Trial window for the report's epoch figures and per-trial scoring. Given together, or neither. |
| **Trial Chunk (s)** | `--epoch-chunk-duration` | off | Cut each task annotation into trials this long before epoching, so a block design gets one trial per piece rather than one per block. |
| **Censoring** (GVTD censoring) | `--gvtd-censor` | off | Annotate the frames GVTD flags as `BAD_gvtd`, on the long channels or on all of them. Nothing is cut. |
| **Threshold (SD)** | `--gvtd-censor-n-std` | `10` | Left-tail SDs above the GVTD mode. Shown only when censoring is on. |
| **Fill** | `--censor-fill` | `linear` | How the `BAD_gvtd` spans are filled before the bandpass: linear, cubic spline or Lomb-Scargle. Shown when censoring is on; sent only with a post mode. |
| **Min kept time (s)** | `--min-time` | off | A run with less time outside the spans gets no FC or ALFF. Shown when censoring is on and the run writes FC or ALFF (`rest`, or **Connectivity** ticked). |
| **Also write a mean-filled copy** | `--export-mean-filled` | off | `desc-meanfilled` beside the final residual, its `BAD_` frames at each channel's kept mean, for tools that ignore annotations; not for ALFF or a GLM. Shown with a post mode. |
| **A QC page per condition** | `--by-condition` | off | One report page per annotated condition, sliced out of the run's own windows. |
| **Motion Correction** | `--motion-correction` | `tddr` | `tddr` / `wavelet` / `tpca` / `spline` / `spline_sg` / `wavelet_tddr` / `spline_wavelet` / `none`. |
| **Detection threshold (SD)** | `--motion-std-thresh` | per method | Shown for `spline` and `spline_wavelet`; the placeholder names the method's default. |
| **Wavelet IQR** | `--wavelet-iqr` | per method | Shown for `wavelet`, `wavelet_tddr` and `spline_wavelet`; the placeholder names the method's default. |
| **Cardiac Band (Hz)** | `--cardiac-l-freq` / `--cardiac-h-freq` | required | Population-dependent, no default. Adult ~`0.7`/`1.5`; higher for paediatric data. |
| **Respiration Band (Hz)** | `--resp-l-freq` / `--resp-h-freq` | required | Population-dependent, no default. Adult ~`0.1`/`0.5`; higher for infants. |

`--bad-channels`, `--keep-spans`, `--keep-spans-from` and `--bad-spans` have no field: they take per-recording tables, so reach for the command line. [Batch Prep](#batch-prep) can write a `--keep-spans` table from condition names. See [Excluded time](workflows.md#excluded-time) for what the spans do.

The page has no switch for `--ignore`, so a run started here always marks bad the channels each recording's `channels.tsv` marks `bad`. Use the command line with `--ignore channel-status` to leave them to screening.

See [Preprocessing](workflows.md#preprocessing) for what each step does.

### Postprocessing

The filter, drift, HRF and noise-model fields start empty. An empty field passes nothing, so the chosen mode's default applies, and the field shows that default in grey once a mode is picked ([Mode defaults](usage.md#mode-defaults)). Type `none` into a cutoff field to switch off the cutoff the mode would fill in, as `--high-pass none` / `--low-pass none` do on the command line; the field accepts a frequency or `none` and nothing else, by the same rule the command line applies. The drift model can be set to `none` from its dropdown.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Mode** | `--mode` | `none` skips postprocessing; otherwise `denoise` / `glm` / `rest`. |
| **High-pass (Hz)** | `--high-pass` | Bandpass lower bound. Empty takes the mode's default, shown in grey in the field; `glm` fills none, since its drift basis detrends. |
| **Low-pass (Hz)** | `--low-pass` | Bandpass upper bound. Empty takes the mode's default. |
| **Filter** | `--filter-method` | Bandpass design: zero-phase Butterworth (`iir`) or hamming-windowed linear phase (`fir`). |
| **Filter order** | `--filter-order` | Butterworth order, ignored by `fir`. |
| **Resample (Hz)** | `--resample-sfreq` | Target sampling rate after filtering. |
| **n_jobs** | `--n-jobs` | How many subjects run at once. Takes effect only when more than one subject is selected. |

#### Confound regression

Its own collapsible section, shown for **every** mode: `denoise`, `glm` and `rest` all honour these. Only the task model below is GLM-only.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Drift Model** | `--drift-model` | `cosine` / `polynomial` / `none`. Empty takes the mode's default: cosine for `glm` and `rest`, none for `denoise`, where the bandpass already detrends. |
| **Drift High-pass (Hz)** | `--drift-high-pass` | Cosine drift cutoff. Shown only for a cosine model, and not pre-filled: the value depends on your design. See below. |
| **Drift Order** | `--drift-order` | Polynomial drift order. Shown only for a polynomial model. |
| **Short Channel** | `--short-channel` | `none` / `mean` / `pca`. Honoured by every mode. |
| **ROI Mapping** | `--roi-mapping` | JSON of ROI label to source-detector pairs ([format](data-preparation.md#roi-mapping-file)). Groups the stage carpet and enables the ROI and seed connectivity products, and ROI-level amplitude in rest mode. |
| **Connectivity** | `--fc` | Also write what rest mode writes, from whatever the mode produced. Rest writes them anyway. |

`--aux-regressors` and `--aux-channels` have no field on this page. They are experimental and withheld from the CLI's `--help`; reach for the command line if you want them.

#### GLM

Visible only when **Mode = glm**.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **HRF Model** | `--hrf-model` | `spm` / `spm + derivative` / `glover` / `glover + derivative` / `fir`. Empty takes `spm`, the `glm` default. |
| **Noise Model** | `--noise-model` | Free text with a suggestion list, `ols` / `auto` / `arN` / `ar_irls`, the same values the command line takes. Empty takes the mode's default: `auto` for `glm`, `ols` for `rest` and `denoise`. The browser refuses what the CLI would, since the field carries the CLI's own pattern. The field is GLM-only, so a `denoise` or `rest` run launched from the page fits the CLI default. |
| **Stim Duration (s)** | `--stim-dur` | Duration given to every annotation in the GLM design. A GLM needs this or an events file. |

See [Postprocessing](workflows.md#postprocessing) for the GLM details.

### Execution

| Field | CLI flag |
|-------|----------|
| **Dry run** | `--dry-run` |
| **Skip BIDS validation** | `--skip-bids-validation` |
| **No report** | `--no-report` |
| **Combine runs** | `--combine-runs` |

Three buttons:

- **Generate Command**: builds the full `nirspipe` command from the current form values and shows it in the **Command Preview** panel.
- **Run Pipeline**: runs that command. Output appears below as it is produced, and the button stays down until the run ends, so one run is in flight at a time. The last 2000 lines are kept; if more are produced, the panel says how many were dropped.
- **Stop**: asks the running command to end. It is enabled only while a run is in flight.

### Choosing the drift cutoff

The cosine drift basis spans every frequency below its cutoff, which puts the cutoff between two bounds:

```
data high-pass   <=   drift cutoff   <   1 / longest repeat interval
```

The lower bound is the field above it: a basis that does not cover what the filter removed leaves the task regressors unfiltered, and the betas come out too small. The upper bound is your design: a condition that repeats every `T` seconds is a rhythm at `1/T` Hz, and a basis reaching it fits the condition away as drift. The usual choice is `1 / (2T)`, half the task frequency. Both bounds are checked when the run starts, and the upper one names each condition it would absorb: see [Choosing the cosine drift cutoff](workflows.md#mode-glm).

[Data Preparation](#data-preparation) reads the number off a loaded run's markers, one line per condition. The page warns while you fill the form when the two bounds are violated, and the run warns again per subject, naming the conditions the basis would span.

:::{note}
0.01 Hz suits conditions repeating every half minute or so; a design whose blocks are minutes apart needs something nearer 0.001 Hz.
:::

### Command Preview

The exact `nirspipe` command that would run. Copy and paste into your terminal:

```bash
nirspipe /data/bids /data/output participant \
    --participant-label 11 \
    --dpf 6.0 6.0 \
    --sci-threshold 0.8 \
    --cardiac-l-freq 0.7 --cardiac-h-freq 1.5 \
    --resp-l-freq 0.1 --resp-h-freq 0.5 \
    --motion-correction tddr \
    --mode glm --stim-dur 15 --hrf-model spm --noise-model ar1 \
    --drift-high-pass 0.01 \
    --high-pass 0.01 --low-pass 0.5 --resample-sfreq 2.0
```

### Pipeline DAG

A live dependency graph that updates as you change Mode. Node colours:

| Colour | Stage |
|--------|-------|
| Orange | Preprocessing (OD → SCI → motion → Beer-Lambert) |
| Blue | Postprocessing (bandpass → resample) |
| Green | GLM (design matrix → fit) |
| Purple | QC report |

Use scroll-wheel to zoom and drag to pan.

## Hyper Analysis

The **Hyper Analysis** page drives `nirspipe-hyper`, the hyperscanning pipeline that runs on what the individual pipeline already wrote. It is a separate stage, not a report: the Analysis page produces per-subject derivatives, and this page reads them to measure coupling between the members of a dyad.

URL: `http://localhost:8050/hyper-analysis`

Like the Analysis page it builds an argv and does not reimplement anything. Unlike the Analysis page, **Run** actually executes it, as a subprocess, and shows the report that came out.

---

### Scope

| Field | Notes |
|-------|-------|
| **Command** | Which `nirspipe-hyper` command to build. Each is its own console script, and the form below changes with it: only the sections that command needs are shown. |
| **Derivatives Directory** | The dyad tree the command writes to, and for the four commands that only re-read tables, the only tree it reads. Nothing here reads BIDS. |

| Command | Builds | What it needs |
|---------|--------|---------------|
| Analyse dyads | `nirspipe-hyper` | Derivatives directory, source derivatives, pairs CSV, the dyad analysis section |
| Re-paired null | `nirspipe-hyper-pairnull` | Derivatives directory, source derivatives, pairs CSV, and a finished `nirspipe-hyper` |
| Read the draws above the cell | `nirspipe-hyper-groupnull` | Derivatives directory, and a finished `nirspipe-hyper-pairnull` |
| Re-average saved WTC maps | `nirspipe-hyper-band` | Derivatives directory and a band |
| Merge WTC tables across dyads | `nirspipe-hyper-merge` | Derivatives directory only |
| Rebuild the dyad landing pages | `nirspipe-hyper-index` | Derivatives directory only |

Every command gets `group` as its analysis level, after its directories.

The **Shell** picker in the Run panel sets the line-continuation style for the preview (`bash`, `cmd`, PowerShell). It follows the platform by default.

### Dyad analysis

Shown for `nirspipe-hyper`, and for `nirspipe-hyper-pairnull`, which reuses part of it. The full meaning of each option is in [Hyperscanning QC Report](qc.md#dyad-reports); this table is the field-to-flag map.

#### Inputs

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Source derivatives** | first positional | The tree `nirspipe` wrote, holding `sub-*/nirs/`. Required by `nirspipe-hyper` and `nirspipe-hyper-pairnull`, the two commands that open a member's recording, and must differ from the derivatives directory above. |
| **Pairs CSV** | `--pairs-csv` | Required. |
| **Group ID** | `--group-id` | Empty runs every group in the CSV. |
| **Task label** | `--task-label` | Space-separated for several; each name is passed as its own flag. |
| **Source stage** | `--desc` | Default `preproc`. The dropdown offers the haemoglobin stages only, because the earlier ones are refused. |
| **ROI mapping** | `--roi-mapping` | Required by the ROI options. |

#### Coherence

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Frequency range (Hz)** | `--wtc-fmin` / `--wtc-fmax` | The axis the coherence is computed on. |
| **Band averaged over (Hz)** | `--wtc-band-fmin` / `--wtc-band-fmax` | Empty averages over the whole axis. |
| **ISC band (Hz)** | `--isc-fmin` / `--isc-fmax` | Its own pair, independent of the one above; set it to the same band to compare the correlation with the coherence. Empty leaves the correlation on whatever the preprocessing passband held. |
| **Chromophore** | `--wtc-chroma` | HbO and HbR by default, which costs exactly twice and is a consistency check rather than two results. |
| **Min channels per ROI** | `--wtc-roi-min-channels` | Drops an ROI cell where either member keeps fewer channels than this. Empty means 1, so one surviving channel keeps a region. |
| **Equal windows (s)** | `--wtc-window-s` | Empty leaves each condition as the unit. Set, every condition is cut into non-overlapping windows of this length. Needs per-condition results. |
| **Prewhitening order (s)** | `--wtc-whiten` | Empty or 0 transforms the signals themselves. Set, every long channel is prewhitened with one autoregressive order of this many seconds before the coherence; the correlation keeps its own **Whitening order**. Passed to `nirspipe-hyper` only: `nirspipe-hyper-pairnull` reads it off the real tables. |

#### Coherence null

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Surrogates** | `--wtc-mc-count` | Runtime scales with it. |
| **Seed** | `--wtc-seed` | Makes the significance contours reproducible. |
| **Phase-null iterations** | `--wtc-phase-null` | Empty or 0 draws no null. Each iteration is a full WTC run, so this is the expensive half. |

#### Correlation

| Field | CLI flag | Notes |
|-------|----------|-------|
| **ISC threshold** | `--isc-threshold` | Empty picks the rule instead of a number: the surrogate's 95th percentile where a null was drawn, the strongest tenth where none was. |
| **Prewhitening order (s)** | `--isc-whiten-s` | 0 correlates the signals themselves. A non-zero value correlates autoregressive residuals, one order in seconds for every channel of both members, and shrinks r markedly. |
| **Max lag (s)** | `--isc-max-lag` | 0 correlates sample against sample. Larger searches that many seconds either way and keeps the strongest. |
| **Null iterations** | `--isc-phase-null` | Ranks each correlation against phase-scrambled surrogates. Pair it with a max lag, which needs a null. |

#### Options

| Switch | CLI flag | Notes |
|--------|----------|-------|
| **Significance testing (slow)** | `--wtc-significance` | What the surrogate count is spent on. |
| **Average the whole band (no COI mask)** | `--no-wtc-mask-coi` | Masking is the default, so this switch turns it off. |
| **Cross channels between brains (slow)** | `--channel-cross` | |
| **Cross channels for the null too (very slow)** | `--wtc-phase-null-cross` | Crossing squares the pair count, and the null pays that on every iteration. |
| **Whole recording only (no per-condition results)** | `--no-by-condition` | Per-condition results are on by default. |
| **Bad channels: union over runs** | `--bads-scope subject` | Off means `run`, this task's own rejections. |
| **Save WTC maps (large)** | `--wtc-save-maps` | What `nirspipe-hyper-band` later reads. |
| **Skip alignment** | `--no-align` | |
| **Normalize recordings** | `--normalize` | |
| **Check only (compute nothing)** | `--check-only` | Prints the channel budget each dyad would be measured on, then stops. |

The page has no switch for `--no-report`; use the command line to skip the report.

### Re-paired null

Shown for `nirspipe-hyper-pairnull`, together with the dyad analysis section, whose source derivatives, pairs CSV, group, task, ROI mapping and chromophore it reuses; the rest of that section is not passed. Pairs one member with people from the other groups who did the same task. Run it after `nirspipe-hyper`: the band, the mask, the frequency range, the window, the prewhitening, the source stage, the rejected channels, the ROI minimum and the crossing come off the tables `nirspipe-hyper` wrote, **not off this form**. A second `nirspipe-hyper` run afterwards draws each condition page's arrows and chords against this null.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Stand-in pool** | `--wtc-pair-pool` | *Same position in the group* replaces a member only with another group's member at the same index, which keeps a role where the two members are not interchangeable and is the only safe pool where one person appears in several groups. *Any member of another group* doubles the pool and is refused where the pairs table shows anybody repeated. |
| **Draw limit** | `--wtc-pair-max` | A ceiling rather than a count: the pool is finite, and leaving it empty uses every eligible stand-in, which is what gives the percentile its best resolution. |
| **Homologous pairings only, even over a crossed table** | `--no-wtc-pair-cross` | Off, the null is crossed exactly when the real table is. On, it draws the homologous pairings only, which is cheaper and ranks a crossed table's diagonal. |

The number of draws is the number of other groups, which is what limits how finely this can rank: G groups give G - 1 draws.

### Null above the cell

Shown for `nirspipe-hyper-groupnull`. Averages the channels before ranking, once per occasion and once over the cohort. It cannot say which channel, and in exchange it can say whether the pairing beats its null at all. It reads what `nirspipe-hyper-pairnull` wrote, runs no transform, and takes one chromophore at a time. It reads the coherence and the correlation, each wherever its draws exist.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Task** | `--task` | Required. Which task's tables to read. |
| **Chromophore** | `--chroma` | HbO or HbR, no *both*. |
| **Draws to read** | `--null` | *Re-paired* or *Phase-scrambled*. Both are read the same way above the cell; they differ only in what a draw is. |
| **ROI mapping** | `--roi-mapping` | Given it, each region is reported as its own level beside the whole-brain mean. |
| **Resamples** | `--n-resample` | Default 20000. Each picks one stand-in per occasion, so the finest *p* it can express is 1/(n+1). |
| **Seed** | `--seed` | Makes the cohort *p* reproducible. |
| **P correction** | `--p-correction` | *None* by default. The raw *p* is written either way; a method adds a corrected column beside it. |
| **ISC test** | `--isc-test` | *Signed, two-sided* by default, which counts a negative correlation; *Magnitude* averages \|z\| and tests it above the null. The coherence is always tested above its null. |

Both of these write tables rather than a report, so the preview panel stays empty after a successful run and says so.

### Re-average saved WTC maps

Shown for `nirspipe-hyper-band`. Reads the maps a previous run saved and averages them over a different band, with no second wavelet transform.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **New band (Hz)** | `--wtc-band-fmin` / `--wtc-band-fmax` | Both required. Must lie inside the axis the saved maps were computed on. |
| **Suffix** | `--wtc-suffix` | The value of the new tables' `band-` entity, letters and digits only, so they sit beside the originals. Defaults to the band, e.g. `0p05to0p2`. |

This one writes tables, not a report, so the preview panel stays empty after a successful run and says so.

### Analysis window

Shown for `nirspipe-hyper`: restricts the synchrony metrics to one window of the aligned recording, so a baseline or a wrap-up segment can be left out of the coherence.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Window (s)** | `--tstart` / `--tend` | Measured from the shared trigger, after alignment. Empty uses the whole recording. It selects rather than cuts: the transform runs over the whole recording and the window is read out of it. |

### Run

- **Generate command** builds the argv and prints it. Arguments the CLI would reject as missing are caught here first and shown as an error instead, which costs no subprocess.
- **Run** executes what was generated. It blocks until the command finishes, then shows the exit status and the last 30 lines of its output.

:::{important}
Click **Generate command** before **Run**. Run reads the stored argv from the last generate, not the current form values, so an edit made after generating is not in what runs.
:::

### Report preview

After a successful `nirspipe-hyper` or `nirspipe-hyper-index`, the dyad's landing page (`group-<gid>/group-<gid>_desc-index_report.html`, the newest one) is embedded in an iframe; a tree whose index is missing falls back to the newest task page. A command that should have written a page and did not says so instead. The path is printed above the frame.

---

## Cohort Reports

The **Cohort Reports** page drives `nirspipe-qc`, the cohort aggregates. Every command here reads quality records that earlier stages already wrote and rolls them up across a tree, so each one takes a derivatives directory and nothing else.

URL: `http://localhost:8050/qc`

Like the Analysis page it builds an argv and does not reimplement anything. Unlike the Analysis page, **Run** actually executes it, as a subprocess, and shows the report that came out.

---

### Scope

| Field | Notes |
|-------|-------|
| **Command** | Which `nirspipe-qc` subcommand to build. |
| **Derivatives Directory** | The only directory field on the page. Nothing here reads BIDS. |

| Command | Builds | Produces |
|---------|--------|----------|
| Cohort QC (individual subjects) | `nirspipe-qc cohort` | `desc-subjects_report.html` (and `desc-subjects_qc.tsv`) |
| Cohort QC (hyperscanning groups) | `nirspipe-qc cohort-hyper` | `desc-groups_report.html` (and `desc-groups_qc.tsv`) |
| Provenance graphs | `nirspipe-qc provenance` | Redrawn provenance graphs, no report page |

The **Shell** picker in the Run panel sets the line-continuation style for the preview (`bash`, `cmd`, PowerShell). It follows the platform by default.

### Run

- **Generate command** builds the argv and prints it. A missing directory is caught here rather than by the subprocess.
- **Run** executes what was generated. It blocks until the command finishes, then shows the exit status and the last 30 lines of its output.

:::{important}
Click **Generate command** before **Run**. Run reads the stored argv from the last generate, not the current form values.
:::

### Report preview

After a successful run, the newest HTML the command is known to write (`desc-subjects_report.html` or `desc-groups_report.html` at the tree root) is embedded in an iframe. The path is printed above the frame. `provenance` writes graphs rather than a page, so the panel says so instead.
