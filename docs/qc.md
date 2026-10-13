---
title: Quality control
---

# Quality control

Every run gets a quality record, a set of scalar, per-channel and per-window metrics measured once after processing, and an HTML report built from it. Standalone commands report on raw recordings before any processing, on a whole cohort, and on dyads. A rating viewer turns any of the reports into a form for recording human verdicts.

| Report | Command | File |
|--------|---------|------|
| Run report | `nirspipe ... participant`, automatically | `sub-<label>/<run>_report.html` |
| Condition pages | `--by-condition` on `nirspipe` or `nirspipe-qc prep-raw` | `<run>_cond-<condition>_report.html`, `..._desc-raw_report.html` |
| Subject index | `nirspipe` or `nirspipe-qc prep-raw` | `sub-<label>/sub-<label>_desc-index_report.html` |
| Raw report | `nirspipe-qc prep-raw` | `sub-<label>/<run>_desc-raw_report.html` |
| Cohort of runs | `nirspipe-qc cohort` or `nirspipe ... group` | `desc-subjects_report.html` at the tree root |
| Dyad raw report | `nirspipe-qc hyper-raw` | `group-<id>/<group run>_desc-raw_report.html` |
| Dyad analysis report | `nirspipe-hyper` | `group-<id>/<group run>_report.html`, one page per condition, and the dyad index |
| Cohort of dyads | `nirspipe-qc cohort-hyper` or `nirspipe ... group` | `desc-groups_report.html` at the root of the tree `hyper-raw` wrote |

The metrics assume no task model: every one of them can be computed on any recording, whatever its design. CNR is the one that needs events, and it is measured from windows around the annotations rather than from a fitted model; a run without annotations has no CNR.

## The quality record

`nirspipe` writes one record per run, `<run>_desc-sqm_qc.json`, after preprocessing and postprocessing have both finished, by reading the files they left behind. `nirspipe-qc prep-raw` writes the raw sections of the same record to `<run>_desc-sqmraw_qc.json`, and `nirspipe-qc hyper-raw` a dyad record under `group-<id>/nirs/`. The cohort pages aggregate all of them.

The record is divided into sections, each named after the file it was measured on, so a section exists only when its file does:

| Section | Measured on |
|---------|-------------|
| `raw`, `raw_long`, `raw_short` | The original recording: every channel, the long channels, the short channels |
| `motion` | What the motion correction changed, from the optical density either side of it |
| `motion_post` | The corrected optical density, with the same keys as `raw`, so the pair says whether the correction helped and whether it cost cardiac signal |
| `windowed` | Every time-indexed series: per-window SCI, PSP and GVTD, and the spans the report draws |
| `preproc`, `filtered`, `resampled`, `errts` | Each haemoglobin stage the run wrote, each also `_long` and `_short` |
| `imu` | The motion sensor, where the recording carries one |
| `excluded`, `censor`, `by_condition` | Present when the run had `BAD_` spans, ran `--gvtd-censor`, or ran `--by-condition` |

Every family is written over all channels, the long ones and the short ones, so a metric is only ever compared with one measured on the same channel set. Metrics that aggregate across the montage (GVTD, the spike counts, the correction footprint) are not written for the short channels alone, and `pct_data_retained`, a share of the recording's duration, is one number for every set. Sections measured on intensity or optical density include rejected channels, the archival view of what was recorded; sections measured on haemoglobin exclude them, the view of what is analysed.

The JSON holds the scalars. The arrays sit in tab-separated tables beside it, listed under `data.tables`: the channel-by-window matrices (`stat-sci`, `stat-psp`, `stat-cv`), the per-window series (`stat-summary`) and the per-channel values (`desc-sqm_nirsmap.tsv`). A matrix has one row per window and one column per channel; row `i` is the window starting at `i * WindowLength` seconds, both given in its sidecar. `<run>_desc-channel_qc.tsv` beside it holds the per-channel metrics and verdicts, one row per channel.

## Metrics

Every metric the record keeps is listed below with the label the report prints, the keys it is stored under, the text the report shows when you hover that label and, where the code defines one, its formula. Each entry ends with how to read the value: higher is better, lower is better, best near a given value, or descriptive (no better end). The report colours five of them, and none of them rejects anything: rejection is per channel, by the screening ([Quality Metrics](#quality-metrics)).

### Coupling and signal level

Sections `raw`, `raw_long` and `raw_short`; SCI and PSP also in `motion_post`. For coupling, read the windowed SCI: it is the estimate screening counts on, and the whole-run value can disagree with it. `n_long_channels` and `n_short_channels` give the size of the split; both zero means the recording has no registered optode positions.

```{eval-rst}
.. metric-reference:: coupling
```

### Motion from the optical signal

Sections `raw`, `raw_long` and `motion`, measured on optical density. The band-passed GVTD is the trace the reports draw and the threshold applies to; the unfiltered and variance-normalised ones are recorded for comparison and drawn nowhere. The unfiltered and band-passed GVTD also have per-window series (`*_per_window`). The spike counts are experimental.

```{eval-rst}
.. metric-reference:: motion
```

### Motion-correction footprint

When a correction ran, the optical density before and after it is compared to report which samples the correction actually changed. These are kept in the record's `motion` section, and the report draws them as a strip at the top of the motion figures. Experimental.

```{eval-rst}
.. metric-reference:: correction
```

### Motion from the sensor

Section `imu`, only where the recording carries one. These belong to no channel set and are descriptive; each also has a per-window series on the GVTD grid, and the record keeps the unit the recording declared.

```{eval-rst}
.. metric-reference:: imu
```

### Processed signal

Sections `preproc`, `filtered`, `resampled` and `errts`; band power and drift in `preproc` only.

```{eval-rst}
.. metric-reference:: processed
```

```{footbibliography}
```

## The run report

`nirspipe` writes one self-contained HTML page per run, plus a subject index that lists every run and condition page on disk, from either command, read back from the quality records. Each page can be opened directly in a browser or served by the [rating viewer](#rating). Every report follows one layout: a summary, the processing steps in order, **Quality Metrics** (the tables), **Quality Figures** (the figures that read off them), and the run record last. Nothing is folded. A figure that fails to build leaves a placeholder and an entry under Errors & Warnings; the rest of the page still renders.

### Summary

Subject metadata, the run command, the parameters, and every post setting nobody typed, grouped by whether `--config` or the mode set it. Run notes follow: what the run had to decide for itself, such as a pair listed in two regions or channels whose separation falls between the short and long ranges. A recording whose optodes were never registered to the head says so: everything drawn from positions is then not anatomical, while separations, and so the short/long split and the screening, are unaffected.

A run with `BAD_` spans gets an **Excluded** section: the kept seconds, the spans with their kind and source, and per condition its kept seconds, counted screening windows and GLM trials ([Excluded time](workflows.md#excluded-time)).

### Preprocessing

- **Raw signal.** A per-channel selector with each channel's HbO/HbR trace, spectrum and epoch average. Every pair is listed, tagged `(rejected)`, `(short)` or `(short, rejected)`. Long traces are thinned for display by keeping each stretch's lowest and highest sample, so the cardiac pulse draws as a band rather than folding into a slow wave.
- **Signal quality.** Windowed SCI and PSP heatmaps (channels by time) with each channel's mean beside them, against the run's own lines, measured on `desc-sci`. Then the brain views and the optode flat map, coloured by windowed SCI.
- **Motion.** The motion-band GVTD trace above a carpet of the optical density (each row detrended and z-scored; each column the largest \|z\| of its stretch, so a short burst is not lost). Before and after correction share one scale and one threshold line. Spike segments are shaded on the trace; where the recording carries an IMU, a gyroscope row and an accelerometer row sit above it, the only motion record not taken from the optical data. With `--roi-mapping` the long rows are grouped by region. A per-channel selector shows one channel's derivative and its optical density before and after, with every `BAD_` span shaded by kind. A zoom compares the longest detected segments before and after correction.
- **HbO / HbR.** The HbO–HbR correlation matrices and each pair's r, on `desc-preproc` and again after denoising. Rejected pairs are kept but ranked apart: independent noise at the two wavelengths converts to anticorrelated HbO and HbR, so a dead channel would otherwise rank with the best. Then a carpet of the last haemoglobin stage, HbO above HbR, each channel detrended and z-scored.
- **Spectra by stage.** The mean spectrum over the long channels for `desc-preproc` and for the last stage the run wrote, with the cardiac and respiration bands, the Mayer wave and the filter cutoffs marked, and a per-channel selector. A run that high-passed notes that the first and last period of the cutoff are partly the filter settling.
- **Epoch preview** (task data). An event timeline, one row per condition; the grand-average HbO and HbR per condition, baseline-corrected, with short channels dotted; channel maps of the condition average along each source-detector path, stepped through the epoch by a slider, short channels in their own row with their share of the long row's peak; and single-trial images per region and per channel. Skipped, with the reason, when nothing can be averaged: no events, no window that fits, or no condition that repeats (`--epoch-single-trial` waives the last).

### Postprocessing

- **glm.** The design matrix as condition time series (nuisance columns left out) and as a whole image; an activation map per condition, HbO betas on the cortical surface on one shared colour scale; the beta table and, with `--contrast-file`, the contrasts.
- **denoise.** The filter settings and the carpet after denoising; with `--fc`, the connectivity step below.
- **rest.** An ALFF step, a flat map of mALFF and fALFF per channel (short channels not drawn, rejected ones grey). A connectivity step, the HbO and HbR correlation matrices as triangles on one scale with mALFF and fALFF as strips under them on the same channel order; with `--roi-mapping`, the region matrix and the seed maps.

### Quality Metrics

**Run summary.** The record's scalars as one table, a metric per row, with columns **All**, **Long** and **Short**. Long is the column the verdict is read off and the only one coloured; Short says whether the regressors those channels feed can be trusted; All is the archival number. A montage that cannot be split has one value column. Where a metric was measured before and after a step, the cell reads before → after. Hover a label for what it measures and which way is good. Seven metrics are marked as the ones that decide whether a run is usable: channel retention, data retained, GVTD share above threshold, band-passed GVTD p95, windowed SCI, coupled windows and HbO–HbR correlation.

```{figure} figures/qc_run_summary.png
:alt: The run summary table of a run report
:width: 75%

The run summary of one run. A cell with an arrow holds the value before and after a step.
```

| Indicator | Green | Yellow | Red |
|-----------|-------|--------|-----|
| Channel retention | ≥ 90 % | 70–90 % | < 70 % |
| Mean SCI (whole run) | ≥ 0.75 | 0.5–0.75 | < 0.5 |
| Coupled windows | ≥ 75 % | 50–75 % | < 50 % |
| Flat channels | 0 | 1 | ≥ 2 |
| % data retained | ≥ 80 % | 60–80 % | < 60 % |

No other metric is coloured: uncoloured means no cutoff was applied, not that the metric passed. None of the averages rejects anything; rejection is per channel, by the screening.

**Per-channel table.** One row per source-detector pair: status, `SCI (10 s)`, `SCI (whole run)`, `PSP (10 s)`, `SNR (10 s)`, `CV (10 s)`, spike share, saturated share and HbO–HbR correlation, grouped long, short, and neither. Wavelength-specific columns print both wavelengths in one cell. Status is `BAD` for a channel coupled in too few windows and `BAD (manual)` for one rejected by hand.

### Quality Figures

- **By denoising stage.** One small panel per metric across `desc-preproc`, `desc-filtered` and `desc-errts`, recomputed with every stage on the analysis passband so the panels show what each step did rather than what the filter did; a stage already filtered to that band is read as stored.
- **By channel.** A pass/fail grid over status, coupled windows, SCI, CV, PSP and SNR, long channels first. A channel can be BAD with a green SCI cell: the cell is the average over windows, and rejection counts the windows that clear both lines.
- **By condition.** With `--by-condition`, each condition's grid regrouped so one metric's conditions sit together, headed **In condition**: whether each channel passes on that stretch alone. A guide to choosing conditions; it rejects nothing.
- **By trial.** Every trial window scored on its own, as a trial-by-metric heatmap, on the intensity recording. Colour is relative within a row, not a threshold. A window of 5 to 10 s is scored on SCI and GVTD only, since CV, SNR and PSP need 10 s, and one shorter than 5 s is left unscored.

```{figure} figures/qc_channel_grid.png
:alt: A pass/fail grid of status, coupled windows, SCI, CV, PSP and SNR per channel
:width: 100%

The pass/fail grid by channel and wavelength, long channels first. S3_D3 is rejected with a green SCI cell: the cell averages SCI over windows, and rejection counts the windows that clear both lines.
```

### Run record

The appendix holds the errors and warnings, the run's provenance graph with a table of every output and the step that made it ([Sidecars and provenance](outputs.md#sidecars-and-provenance)), the Methods paragraph, and software versions. Every run also gets MNE's own report beside the page, `<run>_desc-mne_report.html`.

### Condition pages

A condition page is a view of the run, not a second measurement: its numbers are sliced out of the windowed pass already in the record, on the run's window grid and filter. Every rejection on it is the run's. The condition's own result is shown apart: the channels **Failing in this condition**, and the share **Passing in condition** in place of retention. On a run with `BAD_` spans, a condition's shares leave out every window touching one, and a condition with no counted window shows `n/a`. The GLM panel shows that condition's map from the one model fitted over the whole run. Two conditions whose names keep the same letters and digits would share a file, so neither gets a page; the run page names them as an error.

## Raw report

`nirspipe-qc prep-raw` reports on raw recordings without running the pipeline: it screens the channels with the same criteria and draws the recording, one page per session and task, for as many subjects as you name. It writes the raw sections of the quality record, a per-channel table and the subject index, and takes the same screening and separation flags as `nirspipe`; pass what the run was preprocessed with, or the two reports describe different montages. Its figures share the subject's `figures/` folder with a `raw` prefix on `desc-`.

Its run summary is the same one table, grouped Optical density, Motion and Haemoglobin, the haemoglobin rows from its own Beer-Lambert conversion of the unfiltered recording. `--motion-correction` runs a correction on a copy of the optical density and reports the recording either side of it, writing nothing back, so the screening still describes the recording as delivered; pass the pipeline's method to read the values the run report will show. `--epoch-qc` adds the per-trial section, and `--by-condition` one page per condition. It writes no Methods paragraph, since it describes a recording rather than an analysis.

## Cohort reports

### Cohort of runs

`nirspipe-qc cohort` (or `nirspipe ... group`) reads every quality record under `sub-*/` into `desc-subjects_qc.tsv`, one row per run, a column per `section_metric`, and renders `desc-subjects_report.html`. A run with both a pipeline and a raw record contributes the pipeline's. `desc-subjects_qc.json` describes every column.

| Panel | Shows |
|-------|-------|
| Overview | One row per metric, one dot per run at its robust z: its distance from the cohort median in units of 1.4826 × MAD. |
| Distributions | Metrics split into charts by scale, each point a run; clicking a point opens its report. |
| By condition | Where the records carry conditions: each metric per condition, one line per run, and a run-by-condition matrix of robust z. |
| Over time | Windowed SCI, CV and motion-band GVTD per channel set, every run pale under the cohort median and interquartile band. |
| Outliers | One test per run rather than per metric: a run is flagged when its mean \|z\| over every charted metric exceeds 2, with the metrics that put it there. The place to start triage. |
| Quality Metrics | The cohort summary and every scalar in a sortable table. |

```{figure} figures/qc_cohort_distributions.png
:alt: Box plots of coupling and retention metrics across a cohort, one point per run
:width: 100%

One Distributions chart of a six-run cohort, each point a run, split by channel set.
```

For an analysis of your own, read `desc-subjects_qc.tsv` directly: it is tab-separated with `n/a` for a missing value, and its first column, `bids_name`, is the run's file name without suffix, so the subject, session, task and run are read off its entities.

There is no windowed variant: for a cohort report over one window of time, crop first ([Cropping](data-preparation.md#cropping)).

### Cohort of dyads

`nirspipe-qc cohort-hyper` reads each dyad's raw quality record and usable-time table into `desc-groups_qc.tsv`, one row per dyad, task and session, and renders `desc-groups_report.html`. It reports what a dyad has and a subject cannot, and measures no synchrony.

| Panel | Shows |
|-------|-------|
| Shared usable time | One bar per dyad. A channel pair's window is usable only while the pair is coupled in **both** members at once; the rest splits into a loss that is one member's and a loss that is shared. |
| Where the time went | The usable share per channel pair, and per condition for the dyads that lost the most. A pair pale across every dyad points at the cap rather than a dyad. |
| Quality Metrics | The cohort summary and every dyad's numbers. |

`usable_stretch_s` in the table is the median over pairs of the longest unbroken usable run, since a coherence window needs contiguous data.

## Dyad reports

### Raw dyad report

`nirspipe-qc hyper-raw` reads each group's raw recordings, screens each member, converts to haemoglobin internally and aligns them, one page per group, task and session. It measures no synchrony; that is the analysis, on the preprocessed signal.

| Section | Content |
|---------|---------|
| Alignment | Each member's offset onto the shared clock, and each block's onset difference from the first member, red beyond two samples. |
| Per-channel signals | Both members' traces and spectra, one channel at a time. |
| Shared usable time | The usable share per channel pair and condition, screened again on the aligned grid. |
| Motion | A GVTD row per channel set with both members, over each member's carpet; simultaneous movement is the pointwise minimum of the two traces. IMU rows where present. With `--derivatives-dir`, each member after motion correction too. |
| Where on the head | Each member's coupled-window share on the montage, and each window's SCI by slider. |
| Member and dyad metrics | Each member's numbers side by side, its IMU summaries where the recording carries one, and the dyad's own: pairs good in every member, mixed and bad, simultaneous movement before and after correction, the largest offset. |
| Channel decisions | Each member's per-channel table side by side, with the rating decision, and the pairs each member's `channels.tsv` marked bad. |

```{figure} figures/qc_dyad_usable.png
:alt: Shared usable time of one dyad per channel pair over the recording
:width: 100%

Shared usable time of one dyad: each channel pair's windows coupled in both members, in one member only or in neither, below the members' windowed SCI, PSP and CV and the condition blocks, on the shared clock.
```

It writes the dyad's quality record and three tables: `desc-subject_qc.tsv`, `desc-channel_qc.tsv` and `desc-usable_qc.tsv`. They go into the dyad tree or into a raw QC tree of its own, as `prep-raw` writes the subjects' raw reports; `nirspipe-qc cohort-hyper` reads the quality records, so it runs on whichever tree `hyper-raw` wrote.

### Dyad analysis report

`nirspipe-hyper` writes one page for the whole recording, one per condition window and, with three or more members, one per member pairing.

| Section | Content |
|---------|---------|
| Summary | The members, the stage and band, the alignment, the region grouping, and the levels the arrows and chords were drawn against. |
| Coherence maps | Time-by-frequency coherence per channel pairing, with the cone of influence, the phase arrows and each condition's span. Excluded columns are washed out. Region maps, with `--roi-mapping`, are interactive. |
| Coherence matrices | HbO and HbR band means side by side, per channel and per region pairing. |
| ISC | The correlation matrix and a connectivity circle of the pairings that stand out. |
| All values | Every number behind the panels, one row per pairing, the whole run then each condition. |
| Member metrics | Each member's quality numbers and IMU summaries; on a condition page, that window's. |
| By condition | Each member's channel pairs in a grid: a **Run** row for the whole-run screening every coherence was computed on, and per condition an **In condition** row. |

The dyad index, `group-<id>_desc-index_report.html`, lists every analysed window with its coherence, the share outside the cone of influence, the count past the null and the ISC, over the homologous pairs and over every crossed pairing. A cell more than 3.5 median absolute deviations from the dyad's other windows is marked. `nirspipe-hyper-groupnull` writes a page of its own per task at the tree root ([Hyperscanning](hyperscanning.md#above-the-channel-nirspipe-hyper-groupnull)).

## Rating

`nirspipe-rate` serves an existing report with a rating bar: a verdict per report section and, on a raw run page, a decision per channel. Condition pages take no channel decisions, since a decision applies to the whole run. The report itself is never modified, and opened directly in a browser it shows the bar greyed out. Reopening the viewer reloads what was saved, so a session can be resumed.

| Command | Serves | Default port |
|---------|--------|--------------|
| `nirspipe-rate rate OUTPUT_DIR` | Every subject's index and the run reports it links | 8765 |
| `nirspipe-rate raw OUTPUT_DIR --participant-label <id>` | One subject's raw report and its condition pages | 5052 |
| `nirspipe-rate hyper OUTPUT_DIR --group-id <id> --task-label <t> --pairs-csv <file>` | One dyad's raw report, in the tree `nirspipe-qc hyper-raw` wrote | 5053 |

When the default port is busy the server takes the next free one; a port given with `--port` is used as given or the command stops. Clicking a section's cell cycles unrated, good, bad and other; ctrl-click adds a comment. A channel chip cycles undecided, good and bad; a red tint marks a pair the screening rejected but sets no decision.

| Viewer | Section ratings | Channel decisions |
|--------|-----------------|-------------------|
| `rate` | `sub-<id>/nirs/<page>_desc-rating_qc.json`, every save appended to `logs/group_ratings.jsonl` | none |
| `raw` | `sub-<id>/nirs/<page>_desc-rawrating_qc.json`, every save appended to `logs/group_raw_ratings.jsonl` | `sub-<id>/nirs/sub-<id>[_ses-<s>]_task-<t>_desc-rawdecision_qc.json` |
| `hyper` | `group-<id>/nirs/<group run>_desc-rawrating_qc.json` | each member's `desc-rawdecision_qc.json`, in the `nirspipe` tree given by `--derivatives-dir` |

`nirspipe-rate export TREE [TREE ...]` gathers the channel decisions under the trees into `desc-ratedchannels_qc.tsv`, written to the first tree unless `--out-dir` names another folder: one row per rated run, the pairs rated bad in `bad_channels` and the pairs rated good in `keep_channels`. A dyad page's row leaves `run` blank, so it holds for every run of that member's task and session. Give it to `nirspipe --bad-channels`, where a kept pair stays in even when screening rejected it ([Tables that name recordings](data-preparation.md#tables-that-name-recordings)). A pair rated bad and good for one recording is listed as a warning, and `nirspipe` refuses the table until the viewer settles it. A decision file holding a condition page's key, which older versions wrote, stops the export; delete that key and decide on the run page. The same command lists every page's FINAL verdict in `desc-ratedpages_qc.tsv`, for reading only. For dyads, pass the `nirspipe` tree given to `hyper` as `--derivatives-dir` as a second tree, since the members' decisions are kept there.

The GUI's Raw QC page prints these commands for the reports it wrote and runs `nirspipe-rate export`. `--sci-threshold` on `raw` and `hyper` outlines in orange every windowed SCI cell below a second line, as a reading aid; no verdict changes. Ratings are not aggregated across reports; read the JSONL logs or the sidecars. A log holds one JSON object per line, one line per save, carrying the rated page's `stem`, the save time `rated_at`, each section's verdict and the comments, so the current ratings are the line with the latest `rated_at`.

## Methods text

The run report and the dyad analysis report carry a Methods paragraph ready to paste, with its references, as rendered text, plain text, Markdown and LaTeX. It is built from the provenance sidecars, so it describes the steps that ran with the values they ran with, and a step that was skipped leaves no sentence. A dyad's paragraph follows each member file's `Sources` back to the raw recording, so `--desc` decides what it describes. Each sentence names the method and the parameters a reader needs to reproduce it. `nirspipe-hyper-groupnull` writes its own paragraph beside its tables, in `logs/`. Raw reports carry none.

> fNIRS data were processed using nirspipe v*X.Y.Z*. Raw intensity signals were converted to optical density (OD) using MNE-Python (Gramfort et al., 2013). Channels were screened in non-overlapping 10 s windows: a window counted as coupled when its scalp coupling index (SCI) was at least 0.8 and its peak spectral power (PSP) at least 0.1, both measured in the 0.7–1.5 Hz cardiac band (Pollonini et al., 2014). Channels coupled in fewer than 75% of their windows were marked as bad. Motion artifacts in the OD were corrected with temporal derivative distribution repair (TDDR) (Fishburn et al., 2019). Haemoglobin concentration changes (HbO, HbR) were computed from the OD with the modified Beer-Lambert law (Delpy et al., 1988; Cope & Delpy, 1988), using a differential pathlength factor (DPF) of 6. Data were bandpass filtered (0.01–0.5 Hz) using a zero-phase Butterworth filter (order 4, applied forward and backward). Finally, nuisance signals were removed by regressing an orthogonal basis spanning the retained short channels (HbO and HbR decomposed together) out of each channel in a model with no task regressors, fitted with ordinary least squares without prewhitening; all subsequent analysis used the residuals (Yücel et al., 2021).

The reference list comes from `nirspipe/qc/boilerplate/references.bib`. The hyperscanning sentences cite placeholder entries whose author reads `TODO-ADD-...`, so an unfinished citation is visible rather than missing.

## Figure files

Every figure is written to one flat `figures/` folder beside the pages that embed it: `sub-<label>/figures/`, `group-<id>/figures/`, or `figures/` at the root for the cohort pages. Names carry the run's entities, a `desc-` naming the panel and a suffix naming what is drawn:

```
<run>[_chan-<pair>][_seg-<map>_label-<roi>][_cond-<condition>]_desc-<panel>_<suffix>.<ext>
```

The suffix is `nirs` for signal panels, `qc` for quality grids, `nirsmap` for maps over the montage, `design` for the design matrix and `relmat` for a matrix of pairs; `.html` is an interactive figure and `.png` a static one. Raw reports put a `raw` prefix on `desc-` (`desc-rawcarpet`), and the dyad cohort page a `groups` prefix.
