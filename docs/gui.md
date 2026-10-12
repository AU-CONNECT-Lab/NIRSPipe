---
title: GUI
---

# GUI

`nirspipe-gui` is a Dash interface that runs in your browser. Every page drives one of the command-line tools: it builds the command from the form, shows it, and runs it, so anything done in the GUI can be repeated from a terminal with the command it printed. Data Preparation is the one page that also works in its own process, to load a recording for viewing; its edits still go through `nirspipe-prep`.

```bash
nirspipe-gui                  # port 8050, or the next free one above it
nirspipe-gui --port 8051      # exactly this port, or stop if it is taken
```

Open the address it prints (`http://localhost:8050` by default) in your browser. The sidebar is collapsible (click the ☰ button).

## Pages

The sidebar has two groups. **Interactive** holds Data Preparation, which loads one recording to look at and edit. **Batch** holds the pages that each run one command-line tool over many subjects or dyads.

| Page | Group | Purpose | Runs |
|------|-------|---------|------|
| **Data Preparation** | Interactive | Look at one recording, edit its markers, crop it | `nirspipe-prep edit-markers apply`, `nirspipe-prep crop` |
| **Recon** | Batch | Convert a folder of raw SNIRF files into a BIDS dataset | `nirspipe-recon`, once per file |
| **Batch Prep** | Batch | Marker edits, crops, a keep-spans table or dyad alignment over many subjects | `nirspipe-prep` |
| **Raw QC** | Batch | Raw quality reports for subjects or dyads, then the rating and export commands | `nirspipe-qc prep-raw` / `hyper-raw`, `nirspipe-rate export` |
| **Analysis** | Batch | Preprocessing and postprocessing | `nirspipe` |
| **Hyper Analysis** | Batch | Dyad coherence and correlation, their nulls, re-banding, merged tables | `nirspipe-hyper` and its companions |
| **Cohort Reports** | Batch | Cohort pages and provenance graphs | `nirspipe-qc cohort` / `cohort-hyper` / `provenance` |

A study usually moves down the table: convert with Recon, fix markers on Data Preparation and roll the fix out with Batch Prep, write the raw reports on Raw QC, rate them in `nirspipe-rate`, then run Analysis, Hyper Analysis and Cohort Reports. Channel decisions are taken in `nirspipe-rate` only; no GUI page records them.

## When to use the GUI

- **Use the GUI for:** looking at a recording, fixing markers, configuring a run without typing every flag, launching the raw reports and the analyses.
- **Use the CLI for:** unattended batch processing, reproducible scripted runs, HPC submission.

## How a page runs its command

The batch pages that write derivatives (Batch Prep, Raw QC, Analysis, Hyper Analysis, Cohort Reports) share three buttons:

- **Generate command** builds the command from the form and prints it. Arguments the command line would reject as missing are caught here and shown as an error instead.
- **Run** starts what was generated. The output appears in the log panel as it is printed, and the button stays down until the command ends, so one command per page runs at a time. The last 2000 lines are kept; if more are printed, the panel says how many were dropped.
- **Stop** asks the running command to end. It is enabled only while a command runs.

:::{important}
Run uses the command from the last **Generate command**, not the current form, so an edit made after generating is not in what runs.
:::

Closing the browser tab does not stop a command, but nothing will be watching it: use **Stop** if you want it to end. The **Shell** picker sets the line-continuation style of the printed command (`bash`, `cmd`, PowerShell) and follows the platform by default.

Raw QC, Hyper Analysis and Cohort Reports embed the report the command wrote once it succeeds, with its path printed above the frame. A command that writes tables rather than a page says so, and a command that should have written a page and did not shows a warning instead.

Recon streams the same way, one command per file, with **Run all** and **Stop** but no **Generate command**: its preview follows the table as you edit it. The two edit buttons on Data Preparation run a short command per click and wait for it, then show the command and what it printed.

## Shared state across pages

The BIDS and output directories typed on Data Preparation or Batch Prep are kept in browser session storage and fill the same fields on Batch Prep and Analysis. Everything else a page holds (the loaded run, the marker table, a generated command) lives in memory only: refreshing the page or restarting `nirspipe-gui` clears it.

:::{note}
The GUI runs with `debug=False`. Tracebacks are not shown in the browser; check the terminal where `nirspipe-gui` was launched for any errors.
:::

## Data Preparation

The **Data Preparation** page loads one subject's run so you can look at it, edit its markers and crop it. The quality reports and the channel decisions are elsewhere: write the reports on [Raw QC](#raw-qc) and decide channels in `nirspipe-rate raw`.

URL: `http://localhost:8050/`

---

### Data source

| Field | Required | Notes |
|-------|----------|-------|
| **BIDS directory** | Yes | The BIDS dataset root. |
| **Output directory** | Yes | Where edited and cropped SNIRFs are written, and where loaded runs are cached. |

### Subject

Type a subject ID into **Subject ID** (e.g. `11`) and click **Detect**. The **Run** dropdown then lists every SNIRF under `sub-<id>/nirs/` and `sub-<id>/ses-*/nirs/`, one entry per session, task and run.

### Parameters

| Field | Default | Notes |
|-------|---------|-------|
| **SCI threshold** | required | The SCI line a window has to clear, e.g. `0.8`. |
| **SCI / PSP window (s)** | `10` | The window grid SCI and PSP are measured on. |
| **PSP threshold** | `0.1` | The PSP line for the same window. A window counts as coupled only when it clears both lines. |
| **Coupled windows** | `0.75` | Share of windows a channel has to be coupled in to be kept. This is what rejects. |
| **Screening scope** | Whole run | Which windows count: the whole run, or the annotated task blocks only. |
| **DPF** | required | Differential pathlength factor, used for the haemoglobin panels. |
| **Cardiac band (Hz)** | required | The band SCI, PSP and cardiac power are measured in. Population-dependent. |
| **Epoch window (s)** | -5 / 25 | Trial window for the epoch preview. |
| **Separations (mm)** | `10` / `15` / off | Short, long and the upper long bound, the same split the pipeline uses. |
| **Ignore channels.tsv status** | off | Off, the channels the recording's `channels.tsv` marks `bad` are rejected, as `nirspipe-qc prep-raw` and `nirspipe` reject them. |

The settings are read when a run is selected, and with the same settings the page rejects the same channels the raw report does. After changing a setting, select the run again to apply it.

---

### Panels

#### Signal Topo
Raw HbO / HbR per channel arranged by topographic position. Click a trace to select the channel; the selection is shared with the other panels.

#### Optode Layout (2D)
2D source-detector map. Each channel is coloured by its windowed SCI against the **SCI threshold**: green at or above it, amber down to 0.625 of it, red below that. A rejected channel, by screening or by `channels.tsv`, is grey whatever its SCI. Click a channel to select it.

#### 3D (fsaverage)
The same channel layout projected onto the fsaverage cortical surface, each line running from source to detector and each channel coloured long, short or unclassified by separation. View-only.

#### Channel Detail
HbO and HbR time series for the selected channel, with that channel's power spectrum below. The dropdown also lets you pick a channel pair by name.

#### PSD
Power spectral density averaged across channels, on optical density before Beer-Lambert.

#### Crop
Two modes:

- **Single segment**: pick `tmin` and `tmax` in seconds. **Use zoom** snaps the values to the current zoom range of the trigger timeline.
- **Multi-segment**: add `(onset, duration, task)` rows. **Combine into one file** concatenates them back-to-back into a single cropped SNIRF; otherwise each segment is exported as a separate SNIRF named by its Task cell, which every row then needs, each one different and letters and digits only.

**Apply** runs `nirspipe-prep crop` on the loaded run (see [What gets written](#what-gets-written)).

#### Markers

Edit the run's annotation table inline.

| Action | Effect |
|--------|--------|
| **Step (s)** + **− All / + All** | Shift every marker onset by ± step seconds (clipped to 0). |
| **− Sel / + Sel** | Same shift, applied only to selected rows. |
| **Rename: from → to + Apply All** | Batch-rename one description to another. |
| **+ Add** | Insert a new marker row. |
| **↓ Export TSV** | Download the current marker table as `events.tsv`. |
| **Save** | Run `nirspipe-prep edit-markers apply` with the table (see [What gets written](#what-gets-written)). |

Under the table the page states the drift cutoff a GLM on this run could use, with the slowest repeat of every condition beside it.

#### Epoch Preview
Stimulus-locked epoch average for the selected channel.

---

### What gets written

Both edits run `nirspipe-prep` on the loaded run alone: its subject, session, task and run labels become the selection flags, and BIDS validation is skipped since the run was just read from the dataset. The status line under the button shows the command and what it printed.

| Action | Command | Output path |
|--------|---------|-------------|
| Save markers | `nirspipe-prep edit-markers apply BIDS OUT --participant-label <sub> ... --tsv <table>`, the table written to a temporary file | `<output_dir>/marker_edited/sub-<id>/[ses-<id>/]nirs/<run-name>_nirs.snirf`, with its sidecars |
| Crop, single segment | `nirspipe-prep crop BIDS OUT --participant-label <sub> ... --tmin --tmax` | `<output_dir>/cropped/sub-<id>/[ses-<id>/]nirs/<run-name>_nirs.snirf` |
| Crop, multi-segment | `nirspipe-prep crop BIDS OUT --participant-label <sub> ... --segments-path <table> [--combine]` | The same folder, one file per segment with its Task cell in place of the run's `task-`, or one combined file |
| Export TSV | none | A browser download, `events.tsv`; nothing is written to `<output_dir>` |
| Loading a run | none | `<output_dir>/.fnirs_cache/`: the loaded run's figures and quality record, reused when the same run is loaded with the same settings. Safe to delete. |

:::{note}
Raw SNIRFs in the BIDS dataset are never modified. Both commands refuse an output directory inside the BIDS dataset.
:::

## Recon

The **Recon** page converts a folder of raw `.snirf` files into a BIDS dataset, one row per file. **Run all** runs [`nirspipe-recon`](usage.md#nirspipe-recon) once per row, one after another, so the page writes exactly what the command line writes.

URL: `http://localhost:8050/recon`

---

### Files

| Field | Notes |
|-------|-------|
| **Input folder (raw snirf)** | Searched recursively for `*.snirf`. |
| **Output BIDS directory** | `BIDS_DIR` positional. Created if it does not exist. |
| **Optode frame** | `--optode-frame` for every file: `Unknown` (the default, sends nothing), `Head` for positions digitised against the nasion and preauricular points, or `MRI`. SNIRF does not record it, and without it no `_optodes.tsv` or `_coordsystem.json` is written. |

**Detect** lists every `.snirf` found under the input folder and fills one row per file:

| Column | CLI flag | Editable | Notes |
|--------|----------|----------|-------|
| `File` | (the positional input file) | no | Filename only. The full path is kept behind the row. |
| `subject` | `--participant-label` | yes | Required. Pre-filled from the filename with every non-alphanumeric character stripped, since BIDS labels allow nothing else. Check it before running. |
| `task` | `--task-label` | yes | Required. Blank by default, so this is the column you always have to fill in. |
| `session` | `--session-label` | yes | Optional. |
| `run` | `--run-label` | yes | Optional. |

**Overwrite existing** adds `--overwrite` to every row. A row with no `subject` or no `task` is skipped: it produces no command in the preview, and **Run all** names it under the buttons.

### Command preview

One `nirspipe-recon` invocation per row, in the line-continuation style of the shell you pick. Paths are quoted, so a folder with spaces pastes correctly.

```bash
nirspipe-recon "/data/raw/sub01_run1.snirf" \
  "/data/bids" \
  --participant-label 01 \
  --task-label tapping \
  --optode-frame head
```

### Run

**Run all** converts every valid row in order and shows the output as it is printed, as the batch pages do. Each file starts with a line `[2/5] <file> -> sub-<label>`; a file whose command failed ends with `failed (exit N)` under the error it printed. One failing file does not stop the rest, and the last line gives the totals, for example `4 of 5 succeeded`.

**Stop** ends the file being converted and skips the rest; the last line counts the files not run. Files converted before the stop stay written.

:::{tip}
Convert a couple of files first and open the resulting tree before running the whole folder. A wrong `subject` or `task` here propagates into every derivative and QC report downstream, and the fix is a re-conversion.
:::

## Batch Prep

The **Batch Prep** page applies the same data-preparation operation (marker editing, crop, a keep-spans table, or dyad alignment) to many subjects at once, without running preprocessing. Use it when you already used [Data Preparation](#data-preparation) on one subject to decide on the edit, and now want to roll it out across the cohort.

URL: `http://localhost:8050/batch-prep`

---

### Data source

| Field | Notes |
|-------|-------|
| **BIDS directory** | Source SNIRFs. |
| **Derivatives directory** | Where edited SNIRFs are written. |

### Subjects

Click **Detect** to scan the BIDS directory. Use **Select All** / **Clear** to manage the multi-select chip list.

### Run filter (optional)

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

The table is written to `keep-spans.tsv` in the derivatives directory, and nothing is written unless every selected subject succeeds. See [`nirspipe-prep keep-spans`](usage.md#nirspipe-prep) for the columns and the checks, and [Excluded time](workflows.md#excluded-time) for what the pipeline does with the table. Give the file to the Analysis page's **Keep spans** field.

#### Hyperscanning Align

Aligns dyad recordings by shared trigger, in batch.

| Field | Notes |
|-------|-------|
| **Group CSV** | CSV with columns `group_id, subject_id, task`, plus optional `session` and `run` columns naming which recording of a member to read. Each `(group_id, task)` pair is aligned independently. See [the pairs file](data-preparation.md#pairs-file). |

For each group the alignment finds the annotation descriptions every member shares, crops each member from the first occurrence of a shared trigger, and trims all members to the shortest; see [Hyperscanning Pipeline](hyperscanning.md#alignment). The aligned copies are written to an `aligned/` BIDS tree inside the derivatives directory, each under its source file's own name and `ses-` folder (`aligned/sub-<id>/nirs/sub-<id>_task-<task>_nirs.snirf` with its `_nirs.json`, `_events.tsv` and `_channels.tsv`). Each member's offset is recorded in its own `_nirs.json` (`align_group`, `align_step`, `align_trigger`, `align_offset_s`, `aligned_duration_s`); no separate offsets table is written. A member two groups share for the same task is refused rather than overwritten by the second group.

:::{important}
`nirspipe-hyper` does not read `derivatives/aligned/`. It reads the individual pipeline's output and aligns again in memory, so this tree is for inspection, archiving and tools outside this package rather than a step the dyad analysis depends on. `--no-align` does not mean "use the aligned tree" either: it replaces alignment with trimming every member to the shortest.
:::

| Symptom | Cause |
|---------|-------|
| `AlignmentError: No non-BAD annotations found` | One or more members have no markers (excluding `BAD_*`). |
| `AlignmentError: No shared trigger descriptions` | The members of this group share no annotation description. Use [Data Preparation → Markers](#markers) or **Edit Markers** above to rename triggers into a common scheme first. |
| `... is already aligned for group ..., and group ... would overwrite it` | Another group in the CSV already aligned this member's recording into the same tree. Align the groups that share a member into separate derivatives directories. |

---

### Run

| Field | Default | Notes |
|-------|---------|-------|
| **Parallel jobs** | 1 | Number of subjects processed concurrently. |
| **Generate command** | | Builds the `nirspipe-prep` command and prints it in the Command panel. |
| **Run batch** / **Stop** | | Runs the generated command and streams its output; see [How a page runs its command](#how-a-page-runs-its-command). |

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

## Raw QC

The **Raw QC** page writes the quality reports on the recordings as delivered, before any analysis: one per subject with `nirspipe-qc prep-raw`, or one per dyad with `nirspipe-qc hyper-raw`. It then names the `nirspipe-rate` commands that open the reports for rating, and runs the export that turns the ratings into the Analysis page's bad-channel table. See [Raw QC reports](qc.md) for what the reports hold.

URL: `http://localhost:8050/raw-qc`

---

### Scope

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Command** | | *Subjects* builds `nirspipe-qc prep-raw`, *Dyads* builds `nirspipe-qc hyper-raw`. The sections below follow it. |
| **BIDS directory** | first positional | The recordings. |
| **Output directory** | second positional | Where the reports go: a raw QC tree of its own, or for dyads the `nirspipe-hyper` tree. |
| **Subjects** | `--participant-label` | Space-separated. Required for subjects; for dyads it keeps only the groups holding one of them. |
| **Session labels** / **Task labels** | `--session-label` / `--task-label` | Space-separated; empty takes every one. |
| **ROI mapping** | `--roi-mapping` | Groups the motion carpet's rows by region. |
| **Ignore channels.tsv status** | `--ignore channel-status` | Leaves the channels `channels.tsv` marks `bad` to screening. |
| **Skip BIDS validation** | `--skip-bids-validation` | |

### Screening

| Field | CLI flag | Default | Notes |
|-------|----------|---------|-------|
| **SCI threshold** | `--sci-threshold` | required | |
| **DPF** | `--dpf` | required | |
| **Cardiac band (Hz)** | `--cardiac-l-freq` / `--cardiac-h-freq` | required | Population-dependent. Adult resting ~0.7 to 1.5. |
| **PSP threshold** | `--psp-threshold` | `0.1` | |
| **Coupled windows** | `--min-good-frac` | `0.75` | Share of windows a channel has to be coupled in. This is what rejects. |
| **Screening scope** | `--screen-scope` | Whole run | |
| **Separations (mm)** | `--short-max-dist` / `--long-min-dist` / `--long-max-dist` | `10` / `15` / off | |

### Subject reports

Shown for *Subjects*.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **SCI / PSP window (s)** | `--window-length` | Default 10. |
| **Trial window (s)** | `--epoch-tmin` / `--epoch-tmax` | Both or neither; empty uses each event's own duration. |
| **Motion correction preview** | `--motion-correction` | Draws the chosen correction beside the recording; nothing is written back. |
| **Detection threshold (SD)** / **Wavelet IQR** | `--motion-std-thresh` / `--wavelet-iqr` | Shown for the methods that read them; the placeholder names the method's default. |
| **Per-trial QC** | `--epoch-qc` | |
| **One report per condition** | `--by-condition` | |

### Dyad reports

Shown for *Dyads*.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Pairs CSV** | `--pairs-csv` | Required. See [the pairs file](data-preparation.md#pairs-file). |
| **Group ID** | `--group-id` | Empty writes every group. |
| **nirspipe derivatives** | `--derivatives-dir` | Optional. The individual pipeline's tree: the report draws the after-correction side from it, and `nirspipe-rate hyper` keeps each member's decisions there. |
| **Window (s)** | `--tstart` / `--tend` | On the aligned clock; narrows the synchrony metrics only. |
| **Z-score each channel** | `--normalize` | |
| **No trigger alignment** | `--no-align` | |

### Rating and export

After **Generate command** the Run panel lists the `nirspipe-rate raw` command for each subject, or the `nirspipe-rate hyper` command for the dyads, that opens the reports for rating, and the `nirspipe-rate export` command that follows the rating. Run the rating commands in a terminal; each starts its own viewer.

**Export channel decisions** runs `nirspipe-rate export` on the output directory, and on the nirspipe derivatives too for dyads, and shows what it printed. That names the `--bad-channels` table it wrote (`desc-ratedchannels_qc.tsv` in the output directory), which goes into the Analysis page's **Bad channels** field. See [Rating](qc.md#rating) for the table.

## Analysis

The **Analysis** page configures and launches the full `nirspipe` pipeline (preprocessing and optional postprocessing) and draws its stages. It is the GUI equivalent of the `nirspipe` CLI.

URL: `http://localhost:8050/analysis`

---

### Scope

| Field | CLI flag | Notes |
|-------|----------|-------|
| **BIDS directory** | `bids_dir` (positional) | |
| **Output directory** | `output_dir` (positional) | |
| **Session label(s)** | `--session-label` | Space-separated; optional. |
| **Task label(s)** | `--task-label` | Space-separated; optional. |
| **Detect subjects** | `--participant-label` | Scans BIDS and lists the subjects; each selected one becomes a `--participant-label`. |
| **Run record** + **Load settings** | | Fills the form from a run record; see below. |

#### Loading a run record

Every `nirspipe` run writes a record of its settings to `<output_dir>/sub-<id>/logs/sub-<id>.toml`. Give its path to **Run record** and click **Load settings**: the page reads it the way `nirspipe --config` reads it and fills every field the record holds. A field the record leaves out is reset to the page's default, so the form shows that run and nothing of the one before. The directories and the subject list stay as they were, as do the execution switches other than **Combine runs**.

A setting the record holds that no field on this page can show (`--epoch-single-trial`, `--ignore channel-status` or a per-wavelength `--dpf`, for example) is listed under the button. The generated command leaves those out; run `nirspipe --config <record>` from a terminal to keep them. A record written by another nirspipe version loads with a warning, and one the command line would refuse (an older record layout, say) shows the command line's message instead.

### Preprocessing

| Field | CLI flag | Default | Notes |
|-------|----------|---------|-------|
| **DPF** | `--dpf` | | Differential pathlength factor. Required. |
| **SCI Threshold** | `--sci-threshold` | required | The SCI line a window has to clear to count as coupled, e.g. `0.8`. Starts empty. |
| **PSP Threshold** | `--psp-threshold` | `0.1` | The PSP line for the same window. A window counts as coupled only when it clears both lines. |
| **Coupled Windows** | `--min-good-frac` | `0.75` | Share of windows a channel has to be coupled in to be kept. This is the criterion that rejects. |
| **Screening Scope** | `--screen-scope` | `run` | Which windows count toward it: the whole run, or the annotated task blocks only. |
| **SCI / PSP Window (s)** | `--window-length` | `10` | The grid those series, and the screening count, are measured on. |
| **Separations (mm)** | `--short-max-dist` / `--long-min-dist` / `--long-max-dist` | `10` / `15` / off | The bands the montage is split by. |
| **Epoch tmin / tmax (s)** | `--epoch-tmin` / `--epoch-tmax` | -5 / 25 | Trial window for the report's epoch figures and per-trial scoring. Given together, or neither. |
| **Trial Chunk (s)** | `--epoch-chunk-duration` | off | Cut each task annotation into trials this long before epoching, so a block design gets one trial per piece rather than one per block. |
| **Censoring** (GVTD censoring) | `--gvtd-censor` | off | Annotate the frames GVTD flags as `BAD_gvtd`, measured on the long channels, the short channels or all of them. Nothing is cut. |
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

#### Channel and time tables

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Bad channels** | `--bad-channels` | Source-detector pairs separated by commas, or the path of a table. The export on [Raw QC](#rating-and-export) writes the table, including the channels a reviewer kept. |
| **Keep spans** | `--keep-spans` | The path of a table of stretches to use. [Batch Prep](#keep-spans-table) writes one from condition names. |
| **Keep spans from marker** | `--keep-spans-from` | Measures the keep-spans onsets from each recording's first marker of this name. Sent only with a keep-spans table. |
| **Bad spans** | `--bad-spans` | The path of a table of corrupted stretches. |

See [Excluded time](workflows.md#excluded-time) for what the spans do. The page has no switch for `--ignore`, so a run started here always marks bad the channels each recording's `channels.tsv` marks `bad`. Use the command line with `--ignore channel-status` to leave them to screening.

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
| **HRF Model** | `--hrf-model` | `spm` / `spm + derivative` / `glover` / `glover + derivative` / `fir`, and the dispersion variants. Empty takes `spm`, the `glm` default. |
| **Noise Model** | `--noise-model` | Free text with a suggestion list, `ols` / `auto` / `arN` / `ar_irls`, the same values the command line takes. Empty takes the mode's default: `auto` for `glm`, `ols` for `rest` and `denoise`. The browser refuses what the CLI would, since the field carries the CLI's own pattern. The field is GLM-only, so a `denoise` or `rest` run launched from the page fits the CLI default. |
| **Stim Duration (s)** | `--stim-dur` | Duration given to every annotation in the GLM design. A GLM needs this or an events table. |
| **Events table** | `--events-path` | An `*_events.tsv` to build the design from instead of the annotations. Give this or a stim duration, not both. |
| **Contrast file** | `--contrast-file` | A TOML of contrasts to compute after the fit. |

See [Postprocessing](workflows.md#postprocessing) for the GLM details.

### Execution

| Field | CLI flag |
|-------|----------|
| **Dry run** | `--dry-run` |
| **Skip BIDS validation** | `--skip-bids-validation` |
| **No report** | `--no-report` |
| **Combine runs** | `--combine-runs` |

**Generate Command**, **Run Pipeline** and **Stop** work as described in [How a page runs its command](#how-a-page-runs-its-command).

### Choosing the drift cutoff

The cosine drift basis spans every frequency below its cutoff, which puts the cutoff between two bounds:

```
data high-pass   <=   drift cutoff   <   1 / longest repeat interval
```

The lower bound is the field above it: a basis that does not cover what the filter removed leaves the task regressors unfiltered, and the betas come out too small. The upper bound is your design: a condition that repeats every `T` seconds is a rhythm at `1/T` Hz, and a basis reaching it fits the condition away as drift. The usual choice is `1 / (2T)`, half the task frequency. Both bounds are checked when the run starts, and the upper one names each condition it would absorb: see [Choosing the cosine drift cutoff](workflows.md#mode-glm).

[Data Preparation](#markers) reads the number off a loaded run's markers, one line per condition. The page warns while you fill the form when the two bounds are violated, and the run warns again per subject, naming the conditions the basis would span.

:::{note}
0.01 Hz suits conditions repeating every half minute or so; a design whose blocks are minutes apart needs something nearer 0.001 Hz.
:::

### Command Preview

The exact `nirspipe` command that would run. Copy and paste into your terminal:

```bash
nirspipe /data/bids /data/output participant \
    --participant-label 11 \
    --dpf 6.0 \
    --sci-threshold 0.8 \
    --cardiac-l-freq 0.7 --cardiac-h-freq 1.5 \
    --resp-l-freq 0.1 --resp-h-freq 0.5 \
    --motion-correction tddr \
    --mode glm --stim-dur 15 --hrf-model spm --noise-model ar1 \
    --drift-high-pass 0.01 \
    --high-pass 0.01 --low-pass 0.5 --resample-sfreq 2.0
```

### Pipeline DAG

Before a run, a schematic of the stages the chosen mode runs, in the order the pipeline runs them: optical density, channel screening, motion correction and Beer-Lambert, then the bandpass and resampling, then the GLM or the confound regression, ALFF in rest mode, and connectivity in rest mode or with **Connectivity** ticked. In denoise mode the confound regression is drawn only when a drift model or a short-channel regression gives it something to remove. After a run, the graph is the real one, read from the sidecars in the output directory.

| Colour | Stage |
|--------|-------|
| Orange | Preprocessing |
| Blue | Postprocessing |
| Green | GLM |

The view is fixed and fitted to the panel: it does not zoom or pan. **Download PNG** saves the graph as shown, on a white background at twice the screen resolution. Each run's full provenance graph, one box per file, is already in `figures/` ([Sidecars and provenance](outputs.md#sidecars-and-provenance)).

## Hyper Analysis

The **Hyper Analysis** page drives `nirspipe-hyper`, the hyperscanning pipeline that runs on what the individual pipeline already wrote. It is a separate stage, not a report: the Analysis page produces per-subject derivatives, and this page reads them to measure coupling between the members of a dyad.

URL: `http://localhost:8050/hyper-analysis`

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

Shown for `nirspipe-hyper-groupnull`. Averages the channels before ranking, once per occasion and once over the cohort. It cannot say which channel, and in exchange it can say whether the pairing beats its null at all. It reads what `nirspipe-hyper-pairnull` and the phase-scrambled null wrote, and runs no transform. It reads the coherence and the correlation, each wherever its draws exist.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Task** | `--task-label` | Required. Which task's tables to read. |
| **Chromophores** | `--chroma` | HbO, HbR or both, each into its own tables. None ticked reads every chromophore the draws hold. |
| **Draws to read** | `--null` | *Re-paired*, *Phase-scrambled* or both, each into its own tables. None ticked reads every null whose draws are on disk. Both are read the same way above the cell; they differ only in what a draw is. |
| **ROI mapping** | `--roi-mapping` | Given it, each region is reported as its own level beside the whole-brain mean. |
| **Resamples** | `--n-resample` | Default 20000. Each picks one stand-in per occasion, so the finest *p* it can express is 1/(n+1). |
| **Seed** | `--seed` | Makes the cohort *p* reproducible. |
| **P correction** | `--p-correction` | *None* by default. The raw *p* is written either way; a method adds a corrected column beside it. |
| **ISC test** | `--isc-test` | *Signed, two-sided* by default, which counts a negative correlation; *Magnitude* averages \|z\| and tests it above the null. The coherence is always tested above its null. |

### Re-average saved WTC maps

Shown for `nirspipe-hyper-band`. Reads the maps a previous run saved and averages them over a different band, with no second wavelet transform.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **New band (Hz)** | `--wtc-band-fmin` / `--wtc-band-fmax` | Both required. Must lie inside the axis the saved maps were computed on. |
| **Suffix** | `--wtc-suffix` | The value of the new tables' `band-` entity, letters and digits only, so they sit beside the originals. Defaults to the band, e.g. `0p05to0p2`. |

### Analysis window

Shown for `nirspipe-hyper`: restricts the synchrony metrics to one window of the aligned recording, so a baseline or a wrap-up segment can be left out of the coherence.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Window (s)** | `--tstart` / `--tend` | Measured from the shared trigger, after alignment. Empty uses the whole recording. It selects rather than cuts: the transform runs over the whole recording and the window is read out of it. |

### Run and report preview

**Generate command**, **Run** and **Stop** work as described in [How a page runs its command](#how-a-page-runs-its-command). After a successful `nirspipe-hyper` or `nirspipe-hyper-index`, the dyad's landing page (`group-<gid>/group-<gid>_desc-index_report.html`, the newest one) is embedded; a tree whose index is missing falls back to the newest task page. The re-paired null, the cohort test, re-banding and the merge write tables, so the preview says so instead.

## Cohort Reports

The **Cohort Reports** page drives `nirspipe-qc`, the cohort aggregates. Every command here reads quality records that earlier stages already wrote and rolls them up across a tree, so each one takes a derivatives directory and nothing else.

URL: `http://localhost:8050/qc`

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

### Run and report preview

**Generate command**, **Run** and **Stop** work as described in [How a page runs its command](#how-a-page-runs-its-command). After a successful run, the newest HTML the command is known to write (`desc-subjects_report.html` or `desc-groups_report.html` at the tree root) is embedded. `provenance` writes graphs rather than a page, so the panel says so instead.
