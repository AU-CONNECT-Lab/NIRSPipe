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

The screenshots on this page come from the [tutorial](tutorial.md) run through the GUI, its folder mapped to the drive `S:` so the paths stay short.

## Fields and flags

A field sets the flag named beside it in the tables below. What a flag does, its default and the values it takes are on the [Usage](usage.md) page, generated from each command's `--help`, and are not repeated here. The notes in the tables cover only what the page itself does: when a field is shown, what an empty field sends, and what the page starts with.

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

Closing the browser tab does not stop a command, but nothing will be watching it: use **Stop** if you want it to end. The **Shell** picker sets the line-continuation style of the printed command (`bash`, `cmd`, PowerShell) and follows the platform by default. The printed command puts each positional argument on its own line and each flag beside its values, and quotes every path for the chosen shell, so it pastes into that shell as it stands.

Raw QC, Hyper Analysis and Cohort Reports embed the report the command wrote once it succeeds, with its path and an **Open in a new tab** link above the frame. The frame is as tall as the browser window, so the report scrolls inside it and keeps its own tabs in view. A command that writes tables rather than a page says so, and a command that should have written a page and did not shows a warning instead.

Recon streams the same way, one command per file, with **Run all** and **Stop** but no **Generate command**: its preview follows the table as you edit it. The two edit buttons on Data Preparation run a short command per click and wait for it, then show the command and what it printed.

## Shared state across pages

The BIDS and output directories typed on Data Preparation or Batch Prep are kept in browser session storage and fill the same fields on Batch Prep and Analysis when those pages are opened. Everything else a page holds (the loaded run, the marker table, a generated command) lives in memory only: refreshing the page or restarting `nirspipe-gui` clears it.

:::{note}
The GUI runs with `debug=False`. Tracebacks are not shown in the browser; check the terminal where `nirspipe-gui` was launched for any errors.
:::

## Data Preparation

The **Data Preparation** page loads one subject's run so you can look at it, edit its markers and crop it. The quality reports and the channel decisions are elsewhere: write the reports on [Raw QC](#raw-qc) and decide channels in `nirspipe-rate raw`.

URL: `http://localhost:8050/`

```{figure} figures/gui_dataprep.png
:alt: The Data Preparation page with a run loaded, its Signal Topo, optode layout, PSD, crop and marker cards
:width: 100%

Data Preparation with sub-101's drawing run loaded: Signal Topo averages each channel per condition, the optode layout colours each channel by SCI, and the Crop and Markers cards show the four renamed blocks.
```

---

### Data source

| Field | Required | Notes |
|-------|----------|-------|
| **BIDS directory** | Yes | The BIDS dataset root. |
| **Output directory** | Yes | Where edited and cropped SNIRFs are written, and where loaded runs are cached. Created if it does not exist. |

### Subject

Type a subject ID into **Subject ID** (e.g. `11`) and click **Detect**. The **Run** dropdown then lists every SNIRF under `sub-<id>/nirs/` and `sub-<id>/ses-*/nirs/`, one entry per session, task and run.

### Parameters

The page screens a loaded run as [`nirspipe-qc prep-raw`](usage.md#nirspipe-qc) does, and each field is one of its flags.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **SCI threshold** | `--sci-threshold` | Starts empty. |
| **SCI / PSP window (s)** | `--window-length` | |
| **PSP threshold** | `--psp-threshold` | |
| **Coupled windows** | `--min-good-frac` | |
| **Screening scope** | `--screen-scope` | |
| **DPF** | `--dpf` | Used for the haemoglobin panels. |
| **Cardiac band (Hz)** | `--cardiac-l-freq` / `--cardiac-h-freq` | |
| **Epoch window (s)** | `--epoch-tmin` / `--epoch-tmax` | The epoch preview's window, -5 to 25 s until changed. |
| **Separations (mm)** | `--short-max-dist` / `--long-min-dist` / `--long-max-dist` | |
| **Ignore channels.tsv status** | `--ignore channel-status` | |

The settings are read when a run is selected, and with the same settings the page rejects the same channels the raw report does. After changing a setting, select the run again to apply it.

---

### Panels

#### Signal Topo
Raw HbO / HbR per channel, each in a cell at the channel's position on the head. A run with markers shows one average per condition over the **Epoch window**, named in the legend below the head; a run without them shows the continuous signal. Click a trace to select the channel; the selection is shared with the other panels.

#### Optode Layout (2D)
2D source-detector map. Each channel is coloured by its windowed SCI against the **SCI threshold**: green at or above it, amber down to 0.625 of it, red below that. A rejected channel, by screening or by `channels.tsv`, is grey whatever its SCI. Click a channel to select it.

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
nirspipe-recon \
  '/data/raw/sub01_run1.snirf' \
  '/data/bids' \
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
| **Skip BIDS validation** | `--skip-bids-validation`, sent with every operation. |

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

```{figure} figures/gui_rawqc.png
:alt: The Raw QC page after a dyad run, with the form, the command, its log and the dyad report embedded
:width: 100%

Raw QC after `nirspipe-qc hyper-raw` on the tutorial's three dyads: the command and its log on the right, the rating commands under them, and the newest dyad report below the form.
```

---

### Scope

The flags are [`nirspipe-qc`](usage.md#nirspipe-qc)'s.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Command** | | *Subjects* builds `nirspipe-qc prep-raw`, *Dyads* builds `nirspipe-qc hyper-raw`. The sections below follow it. |
| **BIDS directory** | first positional | |
| **Output directory** | second positional | |
| **Subjects** | `--participant-label` | Space-separated. |
| **Session labels** / **Task labels** | `--session-label` / `--task-label` | Space-separated; empty sends nothing. |
| **ROI mapping** | `--roi-mapping` | |
| **Ignore channels.tsv status** | `--ignore channel-status` | |
| **Skip BIDS validation** | `--skip-bids-validation` | |

### Screening

| Field | CLI flag | Notes |
|-------|----------|-------|
| **SCI threshold** | `--sci-threshold` | |
| **DPF** | `--dpf` | |
| **Cardiac band (Hz)** | `--cardiac-l-freq` / `--cardiac-h-freq` | |
| **PSP threshold** | `--psp-threshold` | |
| **Coupled windows** | `--min-good-frac` | |
| **Screening scope** | `--screen-scope` | |
| **Separations (mm)** | `--short-max-dist` / `--long-min-dist` / `--long-max-dist` | |

### Subject reports

Shown for *Subjects*.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **SCI / PSP window (s)** | `--window-length` | |
| **Trial window (s)** | `--epoch-tmin` / `--epoch-tmax` | |
| **Motion correction preview** | `--motion-correction` | |
| **Detection threshold (SD)** / **Wavelet IQR** | `--motion-std-thresh` / `--wavelet-iqr` | Shown for the methods that read them; the placeholder names the method's default. |
| **Per-trial QC** | `--epoch-qc` | |
| **One report per condition** | `--by-condition` | |

### Dyad reports

Shown for *Dyads*.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Pairs CSV** | `--pairs-csv` | See [the pairs file](data-preparation.md#pairs-file). |
| **Group ID** | `--group-id` | |
| **nirspipe derivatives** | `--derivatives-dir` | |
| **Window (s)** | `--tstart` / `--tend` | |
| **Z-score each channel** | `--normalize` | |
| **No trigger alignment** | `--no-align` | |

### Rating and export

After **Generate command** the Run panel lists the `nirspipe-rate raw` command for each subject, or the `nirspipe-rate hyper` command for the dyads, that opens the reports for rating, and the `nirspipe-rate export` command that follows the rating. Run the rating commands in a terminal; each starts its own viewer.

**Export channel decisions** runs `nirspipe-rate export` on the output directory, and on the nirspipe derivatives too for dyads, and shows what it printed. That names the `--bad-channels` table it wrote (`desc-ratedchannels_qc.tsv` in the output directory), which goes into the Analysis page's **Bad channels** field. See [Rating](qc.md#rating) for the table.

## Analysis

The **Analysis** page configures and launches the full `nirspipe` pipeline (preprocessing and optional postprocessing) and draws its stages. It is the GUI equivalent of the `nirspipe` CLI.

URL: `http://localhost:8050/analysis`

```{figure} figures/gui_analysis.png
:alt: The Analysis page filled in for a denoise run, with the generated command and the stage schematic
:width: 100%

The Analysis page set up as the tutorial's preprocessing step, after the run: the command it generated on the right, and the schematic of the denoise stages below the form.
```

---

### Scope

The flags are [`nirspipe`](usage.md#nirspipe)'s.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **BIDS directory** | `bids_dir` (positional) | |
| **Output directory** | `output_dir` (positional) | |
| **Session label(s)** | `--session-label` | Space-separated. |
| **Task label(s)** | `--task-label` | Space-separated. |
| **Detect subjects** | `--participant-label` | Scans BIDS and lists the subjects; the selected ones become its values. |
| **Run record** + **Load settings** | | Fills the form from a run record; see below. |

#### Loading a run record

Every `nirspipe` run writes a record of its settings to `<output_dir>/sub-<id>/logs/sub-<id>.toml`. Give its path to **Run record** and click **Load settings**: the page reads it the way `nirspipe --config` reads it and fills every field the record holds. A field the record leaves out is reset to the page's default, so the form shows that run and nothing of the one before. The directories and the subject list stay as they were, as do the execution switches other than **Combine runs**.

A setting the record holds that no field on this page can show (`--epoch-single-trial`, `--ignore channel-status` or a per-wavelength `--dpf`, for example) is listed under the button. The generated command leaves those out; run `nirspipe --config <record>` from a terminal to keep them. A record written by another nirspipe version loads with a warning, and one the command line would refuse (an older record layout, say) shows the command line's message instead.

### Preprocessing

| Field | CLI flag | Notes |
|-------|----------|-------|
| **DPF** | `--dpf` | |
| **SCI Threshold** | `--sci-threshold` | Starts empty. |
| **PSP Threshold** | `--psp-threshold` | |
| **Coupled Windows** | `--min-good-frac` | |
| **Screening Scope** | `--screen-scope` | |
| **SCI / PSP Window (s)** | `--window-length` | |
| **Separations (mm)** | `--short-max-dist` / `--long-min-dist` / `--long-max-dist` | |
| **Epoch tmin / tmax (s)** | `--epoch-tmin` / `--epoch-tmax` | |
| **Trial Chunk (s)** | `--epoch-chunk-duration` | |
| **Censoring** (GVTD censoring) | `--gvtd-censor` | |
| **Threshold (SD)** | `--gvtd-censor-n-std` | Shown only when censoring is on. |
| **Fill** | `--censor-fill` | Shown when censoring is on; sent only with a post mode. |
| **Min kept time (s)** | `--min-time` | Shown when censoring is on and the run writes FC or ALFF (`rest`, or **Connectivity** ticked). |
| **Also write a mean-filled copy** | `--export-mean-filled` | Shown with a post mode. |
| **A QC page per condition** | `--by-condition` | |
| **Motion Correction** | `--motion-correction` | |
| **Detection threshold (SD)** | `--motion-std-thresh` | Shown for `spline` and `spline_wavelet`; the placeholder names the method's default. |
| **Wavelet IQR** | `--wavelet-iqr` | Shown for `wavelet`, `wavelet_tddr` and `spline_wavelet`; the placeholder names the method's default. |
| **Cardiac Band (Hz)** | `--cardiac-l-freq` / `--cardiac-h-freq` | Starts empty; the placeholders give the adult band, 0.7 to 2.0. |
| **Respiration Band (Hz)** | `--resp-l-freq` / `--resp-h-freq` | Starts empty; the placeholders give the adult band, 0.1 to 0.5. |

#### Channel and time tables

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Bad channels** | `--bad-channels` | The export on [Raw QC](#rating-and-export) writes the table, including the channels a reviewer kept. |
| **Keep spans** | `--keep-spans` | [Batch Prep](#keep-spans-table) writes one from condition names. |
| **Keep spans from marker** | `--keep-spans-from` | Sent only with a keep-spans table. |
| **Bad spans** | `--bad-spans` | |

See [Excluded time](workflows.md#excluded-time) for what the spans do. The page has no switch for `--ignore`, so a run started here always marks bad the channels each recording's `channels.tsv` marks `bad`. Use the command line with `--ignore channel-status` to leave them to screening.

See [Preprocessing](workflows.md#preprocessing) for what each step does.

### Postprocessing

The filter, drift, HRF and noise-model fields start empty. An empty field passes nothing, so the chosen mode's default applies, and the field shows that default in grey once a mode is picked ([Mode defaults](usage.md#mode-defaults)). Type `none` into a cutoff field to switch off the cutoff the mode would fill in, as `--high-pass none` / `--low-pass none` do on the command line; the field accepts a frequency or `none` and nothing else, by the same rule the command line applies. The drift model can be set to `none` from its dropdown.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Mode** | `--mode` | |
| **High-pass (Hz)** | `--high-pass` | |
| **Low-pass (Hz)** | `--low-pass` | |
| **Filter** | `--filter-method` | |
| **Filter order** | `--filter-order` | |
| **Resample (Hz)** | `--resample-sfreq` | |
| **n_jobs** | `--n-jobs` | |

#### Confound regression

Its own collapsible section, shown for **every** mode: `denoise`, `glm` and `rest` all honour these. Only the task model below is GLM-only.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Drift Model** | `--drift-model` | |
| **Drift High-pass (Hz)** | `--drift-high-pass` | Shown only for a cosine model, and not pre-filled: the value depends on your design. See below. |
| **Drift Order** | `--drift-order` | Shown only for a polynomial model. |
| **Short Channel** | `--short-channel` | |
| **ROI Mapping** | `--roi-mapping` | See [the ROI mapping file](data-preparation.md#roi-mapping-file). |
| **Connectivity** | `--fc` | |

`--aux-regressors` and `--aux-channels` have no field on this page. They are experimental and withheld from the CLI's `--help`; reach for the command line if you want them.

#### GLM

Visible only when **Mode = glm**.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **HRF Model** | `--hrf-model` | |
| **Noise Model** | `--noise-model` | Free text with a suggestion list. The browser refuses what the CLI would, since the field carries the CLI's own pattern. The field is GLM-only, so a `denoise` or `rest` run launched from the page fits the mode's default. |
| **Stim Duration (s)** | `--stim-dur` | |
| **Events table** | `--events-path` | |
| **Contrast file** | `--contrast-file` | |

See [Postprocessing](workflows.md#postprocessing) for the GLM details.

### Execution

| Field | CLI flag |
|-------|----------|
| **Dry run** | `--dry-run` |
| **Skip BIDS validation** | `--skip-bids-validation` |
| **No report** | `--no-report` |
| **Combine runs** | `--combine-runs` |

**Dry run** starts switched on, so a run launched without touching it writes each subject's run record and nothing else. Switch it off to process the data.

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
nirspipe \
  '/data/bids' \
  '/data/output' \
  participant \
  --participant-label 11 \
  --dpf 6 \
  --sci-threshold 0.8 \
  --cardiac-l-freq 0.7 \
  --cardiac-h-freq 2 \
  --resp-l-freq 0.1 \
  --resp-h-freq 0.5 \
  --motion-correction tddr \
  --mode glm \
  --high-pass 0.01 \
  --low-pass 0.5 \
  --resample-sfreq 2 \
  --hrf-model spm \
  --noise-model ar1 \
  --drift-high-pass 0.01 \
  --stim-dur 15
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

```{figure} figures/gui_hyper.png
:alt: The Hyper Analysis page after a dyad run, with the form, the command, its log and a dyad landing page embedded
:width: 100%

Hyper Analysis after the tutorial's dyad analysis: the form as the tutorial fills it, the command and its log on the right, and the newest dyad's landing page below.
```

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

Shown for `nirspipe-hyper`, and for `nirspipe-hyper-pairnull`, which reuses part of it. The flags are [`nirspipe-hyper`](usage.md#nirspipe-hyper)'s; what the outputs mean is in [Dyad reports](qc.md#dyad-reports).

#### Inputs

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Source derivatives** | first positional | Sent to `nirspipe-hyper` and `nirspipe-hyper-pairnull`, the two commands that open a member's recording. |
| **Pairs CSV** | `--pairs-csv` | |
| **Group ID** | `--group-id` | |
| **Task label** | `--task-label` | Space-separated for several, all passed under the one flag. |
| **Source stage** | `--desc` | The dropdown offers the haemoglobin stages only, because the earlier ones are refused. |
| **ROI mapping** | `--roi-mapping` | |

#### Coherence

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Frequency range (Hz)** | `--wtc-fmin` / `--wtc-fmax` | |
| **Band (Hz)** | `--band-fmin` / `--band-fmax` | |
| **WTC band (Hz)** | `--wtc-band-fmin` / `--wtc-band-fmax` | |
| **ISC band (Hz)** | `--isc-fmin` / `--isc-fmax` | |
| **Chromophore** | `--wtc-chroma` | |
| **Min channels per ROI** | `--wtc-roi-min-channels` | |
| **Equal windows (s)** | `--wtc-window-s` | |
| **Prewhitening order (s)** | `--wtc-whiten` | Passed to `nirspipe-hyper` only: `nirspipe-hyper-pairnull` reads it off the real tables. |

#### Coherence null

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Surrogates** | `--wtc-mc-count` | |
| **Seed** | `--wtc-seed` | |
| **Phase-null iterations** | `--wtc-phase-null` | |

#### Correlation

| Field | CLI flag | Notes |
|-------|----------|-------|
| **ISC threshold** | `--isc-threshold` | |
| **Prewhitening order (s)** | `--isc-whiten-s` | |
| **Max lag (s)** | `--isc-max-lag` | |
| **Null iterations** | `--isc-phase-null` | |

#### Options

| Switch | CLI flag |
|--------|----------|
| **Significance testing (slow)** | `--wtc-significance` |
| **Average the whole band (no COI mask)** | `--no-wtc-mask-coi` |
| **Cross channels between brains (slow)** | `--channel-cross` |
| **Cross channels for the null too (very slow)** | `--wtc-phase-null-cross` |
| **Whole recording only (no per-condition results)** | `--no-by-condition` |
| **Bad channels: union over runs** | `--bads-scope subject` |
| **Save WTC maps (large)** | `--wtc-save-maps` |
| **Skip alignment** | `--no-align` |
| **Normalize recordings** | `--normalize` |
| **Check only (compute nothing)** | `--check-only` |

The page has no switch for `--no-report`; use the command line to skip the report.

### Re-paired null

Shown for `nirspipe-hyper-pairnull`, together with the dyad analysis section, whose source derivatives, pairs CSV, group, task, ROI mapping and chromophore it reuses; the rest of that section is not passed. Run it after `nirspipe-hyper`: the band, the mask, the frequency range, the window, the prewhitening, the source stage, the rejected channels, the ROI minimum and the crossing come off the tables `nirspipe-hyper` wrote, **not off this form**. A second `nirspipe-hyper` run afterwards draws each condition page's arrows and chords against this null. The flags are [`nirspipe-hyper-pairnull`](usage.md#nirspipe-hyper-pairnull)'s.

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Stand-in pool** | `--wtc-pair-pool` | *Same position in the group* is `position`, *Any member of another group* is `any`. |
| **Draw limit** | `--wtc-pair-max` | |
| **Homologous pairings only, even over a crossed table** | `--no-wtc-pair-cross` | |

### Null above the cell

Shown for `nirspipe-hyper-groupnull`. The flags are [its own](usage.md#nirspipe-hyper-groupnull).

| Field | CLI flag | Notes |
|-------|----------|-------|
| **Task** | `--task-label` | |
| **Chromophores** | `--chroma` | None ticked sends nothing, so every chromophore the draws hold is read. |
| **Draws to read** | `--null` | None ticked sends nothing, so every null whose draws are on disk is read. |
| **ROI mapping** | `--roi-mapping` | |
| **Resamples** | `--n-resample` | |
| **Seed** | `--seed` | |
| **P correction** | `--p-correction` | |
| **ISC test** | `--isc-test` | *Signed, two-sided* is `signed`, *Magnitude* is `magnitude`. |

### Re-average saved WTC maps

Shown for `nirspipe-hyper-band`. The flags are [its own](usage.md#nirspipe-hyper-band).

| Field | CLI flag |
|-------|----------|
| **New band (Hz)** | `--wtc-band-fmin` / `--wtc-band-fmax` |
| **Suffix** | `--wtc-suffix` |

### Analysis window

Shown for `nirspipe-hyper`.

| Field | CLI flag |
|-------|----------|
| **Window (s)** | `--tstart` / `--tend` |

### Run and report preview

**Generate command**, **Run** and **Stop** work as described in [How a page runs its command](#how-a-page-runs-its-command). After a successful `nirspipe-hyper` or `nirspipe-hyper-index`, the dyad's landing page (`group-<gid>/group-<gid>_desc-index_report.html`, the newest one) is embedded; a tree whose index is missing falls back to the newest task page. The re-paired null, the cohort test, re-banding and the merge write tables, so the preview says so instead.

## Cohort Reports

The **Cohort Reports** page drives `nirspipe-qc`, the cohort aggregates. Every command here reads quality records that earlier stages already wrote and rolls them up across a tree, so each one takes a derivatives directory and nothing else.

URL: `http://localhost:8050/qc`

```{figure} figures/gui_cohort.png
:alt: The Cohort Reports page after a cohort run, with the cohort report embedded
:width: 100%

Cohort Reports after `nirspipe-qc cohort` on the tutorial's six members.
```

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

**Generate command**, **Run** and **Stop** work as described in [How a page runs its command](#how-a-page-runs-its-command). After a successful run, the newest HTML the command is known to write (`desc-subjects_report.html` or `desc-groups_report.html` at the tree root) is embedded. `provenance` writes graphs rather than a page, so the panel says so instead. A tree that holds no quality records stops the cohort commands with an error rather than an empty page.
