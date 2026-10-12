---
title: nirspipe.qc
---

# `nirspipe.qc`

## `subject.report`: single-subject report

### `build_subject_report(...)`

Render a per-subject HTML QC report and save it to `out_path`. Accepts optional arguments for windowed SCI / PSP / GVTD matrices, motion segments, spatial registration outputs (`coords_head`, `good_mask`), and GLM outputs (`design_matrix`, `glm_est`).

Composed of independent section builders (`_section_sqm`, `_section_sci`, `_section_motion`, `_section_haemo`, `_section_glm`, `_section_rest`, …). Each section is wrapped in a `_guard` context manager: a failure leaves a warning banner but does not abort the rest of the render.

## `subject.prep_raw_report`: raw QC report (no preprocessing)

| Function | Description |
|----------|-------------|
| `build_prep_raw_report(runs, output_path, cardiac_l_freq, cardiac_h_freq, dpf, sci_threshold, ...)` | Render the report consumed by `nirspipe-qc prep-raw`, `<run label>_desc-raw_report.html`. Operates on raw BIDS SNIRFs. |

## `subject.group_writer`: cohort aggregation

| Function | Description |
|----------|-------------|
| `build_group_raw_report(output_dir)` | Aggregate `sub-*/nirs/*_desc-sqm{,raw}_qc.json` → `desc-subjects_qc.tsv` and `desc-subjects_report.html` at the tree root, figures under `figures/` |

## `hyper.group_hyper_writer`: cohort of dyads

| Function | Description |
|----------|-------------|
| `build_group_hyper_report(output_dir)` | Every dyad in a tree on one page → `desc-groups_qc.tsv` and `desc-groups_report.html`, figures `figures/desc-groups<panel>_nirs.html`. Returns `None` for a tree with no dyad record |
| `collect_rows(output_dir)` | One row per dyad-task, off the records and usable-time tables on disk |

## `hyper.hyper_usable`: shared usable time

| Function | Description |
|----------|-------------|
| `usable_scalars(grid, subject_ids)` | The dyad's three shares, its median usable stretch and the counts behind them |
| `usable_table(grid, subject_ids, conditions)` | One row per (pair, condition), what the cohort panels read |
| `write_usable_table(path, ...)` | The table plus its sidecar, beside the dyad's record |

## `hyper.hyper_report`: dyad reports

| Function | Description |
|----------|-------------|
| `build_hyper_report(group_id, task, group, sqm_data, aligned_raws, offsets, output_dir, raw_raws=None, …)` | Renders `nirspipe-qc hyper-raw` output, `group-<id>_task-<task>_desc-raw_report.html` |
| `build_hyper_post_report(group_id, task, group, aligned_raws, offsets, output_dir, roi_map=None, roi_map_name="custom", bad_channels=None, …)` | Renders the `nirspipe-hyper` report, `group-<id>_task-<task>_report.html`, with `cond-<label>` and `pair-<slug>` pages beside it |

---

## `metrics`: the measuring functions

Which file each is called on, and which section of the record its output becomes, is decided by `sqm_record` below.

One module per metric family. Imports run one way down this table, and every name is re-exported from the package, so `from nirspipe.qc.metrics import X` reaches any of them.

| Module | Holds |
|--------|-------|
| `_helpers` | Shared plumbing: `_safe_metrics` (declares a function's output schema), `_mask_to_segments`, `long_short_channels` and the separation constants |
| `gvtd` | The global motion index end to end: `gvtd_timetrace`, `gvtd_threshold`, `gvtd_channel_picks`, `gvtd_censor_spans`, the windowed variants |
| `coupling` | SCI, PSP, CV, SNR, mean amplitude, channel distance, CP |
| `motion` | Per-channel spikes and the motion-correction footprint. Experimental |
| `haemo` | HbO–HbR correlation, CNR, band powers, gcor, drift, retention |
| `windowed` | The same questions per window, on one shared grid |
| `aggregate` | `compute_raw_sqm`, `compute_haemo_sqm`, `compute_prep_haemo_sqm`, `comparable_stage_metrics` |

| Function | Input | Returns |
|----------|-------|---------|
| `compute_raw_sqm(raw_intensity, sci_scores, bad_channels, cardiac_l_freq, cardiac_h_freq) → dict` | Raw intensity `mne.io.Raw` | The `raw*` sections: SCI, channel retention, CV, SNR, channel distance, PSP, CP, GVTD (+ filtered / channel-standardized), spike metrics, … On already-OD input the intensity and motion metrics are set to `None`. |
| `compute_haemo_sqm(raw_haemo) → dict` | Haemo `mne.io.Raw` | The `filtered` / `resampled` / `errts` sections, all valid after filtering and resampling: HbO–HbR corr, CNR, gcor HbO / HbR, pct_data_retained |
| `compute_prep_haemo_sqm(raw_haemo, cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq) → dict` | Haemo `mne.io.Raw` at `desc-preproc` | The `preproc` section: the above plus cardiac / resp band power + fraction (per chromophore) and low-freq drift. Rejects filtered input: after a band-pass those describe the filter, not the recording |

| Helper | Description |
|--------|-------------|
| `gcor_metrics(raw) → dict` | Global correlation per chromophore. Called on both sides of the short-channel regression to produce the `_prereg` / `_postreg` pair. |
| `motion_correction_metrics(before, after) → dict`, `motion_corrected_segments(before, after)` | Motion-correction footprint: which timepoints the correction changed, as scalars and as segments for the figure bands. Experimental. |
| `spike_segments(raw_intensity, ch_frac=0.1)` | Timepoints where at least `ch_frac` of channels spike, as segments for the figure bands. |
| `attach_windowed_series(sqm, raw_od, cardiac_l_freq, cardiac_h_freq, window_s)` | Compute and attach the windowed SCI / PSP / GVTD series; return the matrices with their channel names (`sci_channels`, `psp_channels`, `cv_channels`). |

See [Figure Gallery](../qc/figures.md#sqm-scalar-metrics) for the full field-by-field definitions, and [SQM Database](../logging-and-database/sqm-database.md#schema) for the SQLite column set.

---

## `subject.sqm_record`: the per-run quality record

Assembles one `<label>_desc-sqm_qc.json` per run from what is on disk, after both passes have finished (`<label>_desc-sqmraw_qc.json` for `nirspipe-qc prep-raw`; the two suffixes are `RECORD_SUFFIXES`). The functions above measure; these decide what gets measured on which file and write the result.

| Function | Description |
|----------|-------------|
| `SECTIONS` | `('raw', 'raw_long', 'raw_short', 'motion', 'motion_long', 'motion_short', 'motion_post', 'motion_post_long', 'motion_post_short', 'windowed', 'preproc', 'preproc_long', 'preproc_short', 'filtered', 'filtered_long', 'filtered_short', 'resampled', 'resampled_long', 'resampled_short', 'errts', 'errts_long', 'errts_short')`, the order the record lists them in. Each names the file it was measured on, so a section exists only when that file does |
| `OPTIONAL_SECTIONS` | `('censor', 'imu', 'rawhaemo', 'rawhaemo_long', 'rawhaemo_short', 'rawhaemo_post', 'rawhaemo_post_long', 'rawhaemo_post_short')`: sections only some runs have, which the cohort table still reads. The `rawhaemo*` ones are `nirspipe-qc prep-raw`'s, measured on a conversion it holds in memory |
| `scan_runs(nirs_dir) → {label: {stage: path}}` | Group a `nirs/` directory's files into runs by BIDS label, and each run's files by processing stage |
| `entities_of(label) → {"ses": …, "task": …, "run": …}` | The BIDS entities of one run label |
| `record_path(nirs_dir, label) → Path` | Where that run's record goes |
| `compute_run_sections(stages, *, cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq, qc_window_s=10.0, bids_root=None) → dict` | Every section for one run, plus `per_channel`. Each section guards itself: a section that cannot be measured is reported missing rather than dropped |
| `write_run_sqm(nirs_dir, label, stages, sections, bids_root=None) → Path` | Write the record and its tables, provenance keys included |
| `write_channel_table(nirs_dir, label, stages, sections)` | Write `<label>_desc-channel_qc.tsv` from the record and the sci sidecar, so it exists whether or not a report runs |
| `build_sqm_records(nirs_dir, *, cardiac_l_freq=None, …, bids_root=None) → list[Path]` | One record per run found under `nirs_dir`. Band edges default to what the run's own sidecars recorded, so rebuilding from a past output tree needs no arguments |

`bids_root` is where to look for the original recording when the absolute path its sidecar recorded no longer resolves, which is what happens to any derivatives tree that has been copied or moved. The filename is still correct, so it is searched for there. Without it such a tree loses its three `raw*` sections, with a warning saying why. `nirspipe` passes its own input directory.

```python
from nirspipe.qc.subject.sqm_record import build_sqm_records

# rebuild the records of a tree that was processed elsewhere
build_sqm_records("/data/out/sub-01/nirs", bids_root="/data/bids")
```

## `subject.record_io`: the record on disk

A record is a JSON of scalars plus the TSV tables it lists under `data.tables` (see [BIDS Layout](../configuration/bids-layout.md)). Every reader in the package goes through `read_record`; reading the JSON alone gives a record without its matrices, series or per-channel values.

| Function | Description |
|----------|-------------|
| `write_record(path, record) → Path` | Write the scalars to `path` and the arrays to the tables beside it, each with a sidecar describing its columns |
| `read_record(path) → dict` | The record with its tables folded back in, in the shape `write_record` was handed. Raises `StageError` for a record written before the tables existed, and for a listed table missing from disk |

```python
from nirspipe.qc.subject.record_io import read_record

record = read_record("/data/out/sub-01/nirs/sub-01_task-rest_desc-sqm_qc.json")
record["windowed"]["sci_matrix"]    # channel x window, rows named by record["windowed"]["sci_channels"]
```

Outside Python the tables are plain TSV. pandas needs `float_precision="round_trip"` to read the values back bit for bit.

---

## `boilerplate`

| Function | Description |
|----------|-------------|
| `generate_methods_text(versions=None, nirs_dir=None, label=None, steps=None) → dict` | Paste-ready Methods paragraph from the sidecars under `nirs_dir` (scoped to the run `label`), or from an already filled `steps` list; returns `plain`, `markdown`, `latex` and `html`. Skipped steps are omitted. |
| `vocabulary.steps_from_sidecars(nirs_dir, label=None)` / `vocabulary.steps_from_lineage(path)` | The method steps a run's sidecars record, or those behind one file, read up its `Sources` to the raw recording; the second is what the dyad paragraph is built from. |
| `collect_software_versions() → dict` | Installed versions of MNE / MNE-NIRS / nilearn / nirspipe / dependencies |

Two more modules sit behind those and are read directly by the report builders. `boilerplate.vocabulary` holds every metric's label, print format, thresholds and one-line explanation in one table, so a report's hover text, its traffic-light colour and this documentation cannot drift apart; a metric with no established cutoff prints in the default colour rather than being called good. `boilerplate.notes` holds every other sentence a report prints to explain itself: the paragraph under each heading, and the caveats a run attaches when its data cannot support something, one module per report family. Numbers quoted in that prose are filled from the constants that set them, so the text cannot drift from the computation.

---

## `common`: what both report paths share

Nothing in `qc.common` may import `qc.subject` or `qc.hyper`: a helper that needs one of them belongs in that one.

| Module | What it owns |
|--------|--------------|
| `report_shell` | What the base template reads: the head, the nav bar, the rating bar, the closing sections and the tab script. A report template extends it and fills one block, so what a header shows and what a footer carries is decided once |
| `channel_table` | The per-channel rows every view prints, plus the column lists the split tables use. Every view reads a channel's scores from here, so no two can disagree about them |
| `record_views` | One condition's numbers read back out of the record that holds them: an entry in, a flat dict per channel set out |
| `screen_scope` | Which stretches of a recording the channel screening counts over |
| `windows` | Condition windows and crop provenance, read off a recording's annotations |
| `figure_io` | Getting a figure onto disk: Plotly figures become iframe-ready HTML, matplotlib ones are decoded to PNG. Both land in the subject's or group's one `figures/` folder beside the report; `figure_namer(label, condition=None, prefix="")` names each one with the run label and its entities (`sub-01_task-rest_cond-game1_desc-carpet_nirs.html`, and `desc-rawcarpet` for the raw viewer) |
| `provenance` | Rebuilds the file-level graph from the JSON sidecars, so any past run can be re-read from its own output |

---

## `figures`

The figures subpackage (`nirspipe.qc.figures`) exposes one function per figure type. All return either a Plotly figure object or a base64-encoded PNG string.

See [Figure Gallery](../qc/figures.md) for the visual descriptions.

Three modules under `figures/common/` are not figures but conventions, imported by the builders above: `head_map` (the one flat projection every per-channel map is drawn on, and the bar-versus-disc rule), `matrix_map` (which colour scale a quantity takes and what a blank cell looks like) and `circle_map` (the connectogram's geometry).

| Function | Returns | Description |
|----------|---------|-------------|
| `build_sci_psp_figure(...)` | Plotly | Windowed SCI + PSP heatmaps with per-channel mean lollipops |
| `channel_quality_heatmap(...)` | Plotly | Channel × metric pass/fail grid |
| `binary_heatmap_figure(...)`, `lollipop_scores_figure(...)` | Plotly | Building blocks of the quality panel |
| `build_ts_figure(...)`, `build_channel_figure(...)`, `build_layout_figure(...)`, `build_psd_mean_figure(...)` | Plotly | Raw traces, per-channel detail, probe layout, mean PSD |
| `carpet_gvtd_figure(...)` | Plotly figure | Raw + filtered GVTD over a per-channel OD carpet, with spike / motion-correction bands; a second trace and carpet when given the motion-corrected recording |
| `build_motion_detail_figure(...)` | Plotly | Per-channel before / after motion correction, 4 rows |
| `bad_segment_zoom_figure(...)` | base64 PNG | Before / after zoom on worst segments |
| `carpet_compare_figure(...)` | Plotly figure | One carpet block per stage under a shared GVTD row, for one chromophore |
| `hbo_hbr_correlation_figure(...)` | Plotly | Channel × channel r heatmap per stage, over a per-pair before / after dumbbell |
| `psd_figure(...)` | Plotly | PSD by pipeline stage, HbO and HbR rows, with annotated bands |
| `stage_metrics_figure(...)` | Plotly | One small panel per metric across a chain's stages |
| `denoise_stage_panels(...)` | — | Build that figure's panels for the haemoglobin chain |
| `stage_slope_figure(...)` | Plotly | One line per channel across the denoising stages |
| `build_trial_image_figure(...)`, `build_roi_trial_image_figure(...)` | Plotly | Single-trial HbO heatmap + trial average, per channel or per ROI |
| `build_epoch_preview_figure(...)`, `build_evoked_topo_figure(...)`, `build_trigger_timeline_single(...)` | Plotly | Epoch averages, evoked topography, event timeline |
| `quality_brain_views(...)` | base64 PNG | 3D cortical surface with channel quality |
| `optode_layout_static(...)` | base64 PNG | 2D probe flat map |
| `design_matrix_static_figure(...)`, `design_matrix_heatmap(...)`, `design_matrix_figure(...)` | base64 PNG / Plotly | GLM design matrix |
| `activation_condition_figures(...)`, `activation_panel(...)`, `activation_brain_figure(...)`, `glm_betas_figure(...)`, `per_channel_hrf_figure(...)` | base64 PNG / Plotly | GLM activation, betas, per-channel HRFs. The report uses `activation_condition_figures`, one render per condition on one shared scale; `activation_panel` stacks the same renders into one image |
| `rest_channel_panel(...)`, `fc_roi_matrix_figure(...)`, `fc_seed_topo_figure(...)`, `alff_topo_figure(...)` | Plotly | Resting-state FC matrices, seed topography, ALFF / fALFF |
| `hyper_post_figures.build_wtc_channel(...)`, `.compute_isc(...)`, `.build_isc_panel(...)` | Plotly / PNG | WTC heatmap (optional significance contour), ISC matrix + connectogram |
| `hyper_figures.build_alignment_timeline(...)`, `.build_usable_time(...)`, `.build_motion_panel(...)`, `.build_head_slider(...)`, `.build_signal_overlay(...)` | Plotly | The raw dyad page: the two clocks, shared usable time, motion either side of correction, the head maps, and both members' traces |
| `group_hyper_figures.build_usable_bars(...)`, `.build_pair_field(...)`, `.build_condition_dials(...)` | Plotly | Cohort of dyads: shared usable time, where it went |
| `groupnull_figures.build_occasion_panels(...)`, `.build_region_lift(...)` | Plotly | The cohort test page: each occasion against its null mean per condition, and each region's lift with the paired test's marks; `hyper.groupnull_report.write_groupnull_report(output_dir, task)` writes the page |
| `group_figures.build_deviation_strip(...)`, `.build_grouped_boxes(...)`, `.build_window_grid(...)`, `.build_condition_timeline(...)`, `.build_condition_matrix(...)`, `.detect_outliers(...)` | Plotly | The cohort page: subject × metric deviations, per-scale distributions, the window grid, and the per-condition panels |

---

## `rating`

Flask viewer apps used by `nirspipe-rate`:

| Class | Used by |
|-------|---------|
| `FNIRSRatingApp(output_dir, subjects)` | `nirspipe-rate rate`: multi-subject post-processing rater |
| `RawRatingApp(html_path, output_dir, sci_threshold=None)` | `nirspipe-rate raw`: single-subject raw QC viewer; `sci_threshold` is the optional second line its SCI cells are outlined against |
| `HyperRatingApp(html_path, output_dir, subject_ids, sci_threshold=None)` | `nirspipe-rate hyper`: dyad raw QC viewer; `sci_threshold` as for `RawRatingApp` |

All three expose `/load_*` and `/save_*` REST endpoints and serve the underlying QC HTML unchanged.
