---
title: nirspipe.io
---

# `nirspipe.io`

## `bids`

### `get_layout(bids_dir, validate=True) → BIDSLayout`

Return a PyBIDS `BIDSLayout` for the dataset.

### `get_nirs_files(layout, subject, session=None, task=None, filter_file=None) → list[Path]`

Return paths to all SNIRF files for a given subject/session/task. Accepts an optional JSON filter file with extra PyBIDS query kwargs.

### `get_participant_age(layout, subject, fallback=None) → float`

Resolve participant age from `participants.tsv`, matching `participant_id` with or without its `sub-` prefix, as the bad-channel table is matched. Raises `ValueError` if not found and no fallback is provided.

---

## `snirf`

### `write_snirf(raw, path)`

Write an `mne.io.Raw` object to a SNIRF file using MNE's HDF5 backend.

### `read_snirf(path, **kwargs) → mne.io.Raw`

Read a SNIRF and restore what the file format cannot hold: the pipeline stage from the `desc-` entity (a file with none is a BIDS input and stamps as `raw`), and the bad-channel marks and the bandpass from the sidecar.

---

## `auxiliary`

The SNIRF `aux` group: accelerometers, gyroscopes, a pulse trace, whatever the device wrote beside the optical channels. MNE's reader does not touch that group, so none of it survives `read_raw_snirf`. These functions read it with h5py instead and put it on whatever time axis a downstream stage ends up with. See [Postprocessing](../pipeline/postprocessing.md) for how `--aux-regressors` consumes the result; that flag is withheld from `--help` pending evaluation, but the table is written whatever the run was asked for.

### `read_aux_snirf(path) → (times, values, units)`

Three dicts keyed by channel name: recorded timestamps, samples, and the unit string. Empty when the file has no `aux` group.

### `write_aux_table(source_path, out_path) → (DataFrame, dict) | None`

Extract the aux channels into a gzipped TSV at the rate they were recorded, and return the table plus the facts a sidecar should carry (columns, units, nominal rate, start time, and how far sample counting would drift from the recorded stamps by the end). Returns `None` for a file with no aux group. Not resampled: postprocessing puts it on its own grid.

### `read_aux_table(path) → DataFrame`

The table `write_aux_table` wrote, `time` column included.

### `aux_table_path(reference) → Path` / `find_aux_table(reference) → Path | None`

Where a run's aux table sits, given any SNIRF of the same run: the `desc-` entity is dropped and `desc-aux_timeseries.tsv.gz` put in its place, so every stage can find it. `find_aux_table` returns `None` when it is not there.

### `resample_to_grid(t_src, x, t_dst) → ndarray`

Put a signal on another time grid, anti-aliased: going down in rate, a low-pass runs before the decimation; going up in rate, only the interpolation happens.

### `write_aux_window(source_path, dest_path, windows) → list[str]`

Copy a recording's aux channels into a SNIRF that `write_snirf` has just written, keeping only `windows` and rebasing each so the result starts at zero. This is what carries aux through `nirspipe-prep crop`, which goes through MNE and would otherwise drop the group. Returns the channel names written.

---

## `tables`

### `read_table(path, **kwargs) → DataFrame`

Read a table the user supplied, with the delimiter chosen by extension: `.tsv` tab-separated, `.csv` comma-separated, anything else sniffed from the header. Used for events, segments, the pairs file and `participants.tsv`, so a flag named `--pairs-csv` accepts a TSV. Tables the package writes back are always tab-separated, as BIDS requires.

### `read_tsv_or_none(path, lost) → DataFrame | None`

A tab-separated table the package wrote, or None when it is absent or unreadable. `lost` says what an unreadable one costs the caller and goes into the warning.

---

## `naming`

Every derivative name the package writes is built from entities by one config, `nirspipe/data/nirspipe_bids_config.json`: the entities this package adds to the ones pybids knows (`group`, `pair`, `chromo`, `chan`, `agg`, `band`, `cond`, `null`, `stat`) and the path pattern each kind of output follows.

### `derivative_path(output_dir, suffix, extension, **entities) → Path`

Where one output goes. Entities whose value is `None` are dropped. Raises `ValueError` when no pattern fits, rather than returning a name outside the scheme. Does not create directories.

```
derivative_path(out, "relmat", ".tsv", group="G1", task="rest", statistic="wtc")
  -> out/group-G1/nirs/group-G1_task-rest_stat-wtc_relmat.tsv
```

### `parse_path(path) → dict`

The inverse of `derivative_path`: the entities, `suffix` included, read back out of a name. Pass a path relative to the output directory.

### `report_name(label, *, desc=None, condition=None, pairing=None) → str`

One report page's filename from a run or dyad label: `sub-01_task-rest_report.html`, `..._desc-raw_report.html`, `..._cond-game1_report.html`, `..._pair-<slug>_report.html`.

### `figure_name(stem, desc, *, suffix="nirs", extension=".html", **entities) → str`

One figure's filename. `desc` names the panel, `suffix` what it draws: `sub-01_task-rest_desc-carpet_nirs.html`, `sub-01_task-rest_chan-S1D1_desc-detail_nirs.html`.

### `record_table_name(label, desc, suffix, statistic=None) → str`

One table a quality record keeps beside its JSON: `sub-01_task-rest_stat-sci_desc-sqm_timeseries.tsv`, `sub-01_task-rest_desc-sqm_nirsmap.tsv`.

### `rating_path(output_dir, report_stem) → Path`

Where the human ratings of one report page are kept, under the analysis unit's `nirs/`: `desc-rawrating_qc.json` for a raw viewer page, `desc-rating_qc.json` (with the page's `cond-` entity) otherwise.

### `channel_decisions_path(output_dir, subject, task=None, session=None) → Path`

Where the raw QC page keeps a run's keep/drop decisions: `sub-01/[ses-<id>/]nirs/sub-01_[ses-<id>_]task-rest_desc-rawdecision_qc.json`. Re-exported from `derivatives`.

### `roi_map_name(mapping) → str` / `bids_label(text, fallback="custom") → str`

The `seg-` entity ROI outputs carry: the `--roi-mapping` file's stem reduced to a BIDS label (`frontal_rois.json` → `frontalrois`), or `custom` without one.

### `layout_config() → list[str]`

What to hand `BIDSLayout(config=...)` so pybids can index a tree this package wrote.

---

## `derivatives`

### `build_output_path(output_dir, subject, entities, suffix, extension, session=None) → Path`

Construct a BIDS-Derivatives compliant output path, creating its directory. The name comes from `naming`; the `desc` entity in `entities` distinguishes processing steps (e.g. `{"desc": "preproc"}`).

**Example output paths:**

```
# No session
sub-01/nirs/sub-01_desc-preproc_nirs.snirf

# With session
sub-01/ses-wave1/nirs/sub-01_ses-wave1_desc-preproc_nirs.snirf
```

### `write_sidecar_json(out_path, provenance)`

Write a JSON sidecar next to `out_path`. The `provenance` dict should include `pipeline_version`, `step`, `Sources`, and `parameters`; `timestamp` is added automatically, and `Sources` paths are written as BIDS URIs relative to the tree `out_path` sits in (`to_bids_uri`). See [Provenance](../logging-and-database/provenance.md).

### `write_step_sidecar(path, step, source, bads, **params)`

The sidecar of one postprocessing output: `pipeline_version`, `step`, `Sources` (the one file it came from), `parameters` and `bad_channels`.

### `read_json(path) → dict`

A JSON object off disk, or `{}` when the file is missing, unreadable or not an object. Read as UTF-8 on every platform.

### `subject_labels(root) → list[str]`

The bare labels of a tree's `sub-*` folders, sorted.

### `dataset_root_of(path) → Path | None`

The nearest folder above `path` holding a `dataset_description.json`: the dataset a file sits in, found from the file alone.

### `group_output_path(output_dir, group_id, entities, suffix, extension, session=None) → Path`

The dyad mirror of `build_output_path`: `<out>/group-G1/nirs/group-G1_task-rest_stat-wtc_relmat.tsv` for `entities={"task": "rest", "statistic": "wtc"}`, `suffix="relmat"`.

### `write_dataset_description(output_dir, *, name="nirspipe output", generated_by="nirspipe", source=None)`

Write `dataset_description.json` for the BIDS derivatives dataset, with the package version under `GeneratedBy`. `source` is the tree this one was computed from, recorded under `SourceDatasets` and under `DatasetLinks` as `link` (`raw` or `preprocessed`, by default from the source's `DatasetType`), which is what the BIDS URIs in `Sources` resolve against. Links already in the file are kept and a conflicting one raises; a dyad tree passes the subject tree it read, and the crop, edit-markers and aligned trees the dataset their input sat in. An existing file is rewritten only when its content changes, through a temporary file, so the version stamp stays current.

### `write_bidsignore(output_dir)`

Write the tree's `.bidsignore`: `*.html`, `logs`, `figures` and the five JSON-only records (`*_desc-{sqm,sqmraw,rating,rawrating,rawdecision}_qc.json`). Every table stays checked.

### `find_preproc_snirf(output_dir, subject_id, task, desc="preproc", session=None, run=None) → Path`

Locate the `desc-<desc>` SNIRF for one subject × task, across the subject's session folders. Used by `pipeline.hyper.load_group_haemo`. Raises `MissingDerivativesError` if not found.
