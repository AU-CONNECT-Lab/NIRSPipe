---
title: nirspipe.pipeline
---

# `nirspipe.pipeline`

Signal-processing modules. All functions return MNE / NumPy / pandas objects; SNIRF and TSV outputs are written by the dispatchers (`run_prep`, `run_post`).

## `prep_pipeline`

### `PrepConfig`

Dataclass holding all preprocessing parameters for one subject.

| Field | Type | Description |
|-------|------|-------------|
| `subject` | `str` | Subject label |
| `dpf` | `list[float]` | Differential path-length factor(s), one value or one per wavelength |
| `sci_threshold`, `psp_threshold` | `float` | The two lines a window has to clear to count as coupled |
| `min_good_frac` | `float` | Share of counted windows a channel has to be coupled in. The criterion that rejects |
| `screen_scope` | `str` | Which windows count toward it: `"run"` or `"task"` |
| `cardiac_l_freq`, `cardiac_h_freq` | `float` | Cardiac band bounds (Hz) |
| `resp_l_freq`, `resp_h_freq` | `float` | Respiration band bounds (Hz) |
| `session` | `str \| None` | Session label |
| `qc_window_s` | `float` | Sliding-window length (s) for the windowed SCI / PSP / GVTD series |
| `epoch_tmin`, `epoch_tmax` | `float \| None` | Trial window for the report's epoch figures and per-trial scoring |
| `epoch_chunk_duration` | `float \| None` | Cut each task annotation into trials this long before epoching |
| `gvtd_censor` | `str \| None` | Channel set to censor on (`"long"`, `"all"`), or `None` for no censoring |
| `gvtd_censor_n_std`, `gvtd_min_epoch_s` | `float` | Censoring threshold and the shortest stretch it keeps |
| `motion_correction` | `str \| None` | `"tddr"`, `"wavelet"`, `"tpca"`, `"spline"`, `"spline_sg"`, `"wavelet_tddr"`, `"spline_wavelet"`, or `"none"`. |
| `motion_std_thresh`, `wavelet_iqr` | `float \| None` | Spline detection threshold and wavelet fence; `None` takes the method's default, and a method that reads neither refuses them (`motion_settings`) |
| `short_max_dist`, `long_min_dist`, `long_max_dist` | `float \| None` | The separation bands the montage is split by |
| `bad_channels` | `list[str]` | S-D labels to mark bad, unioned with the channels screening rejected |
| `channels_tsv_bads` | `list[str]` | The channels the recording's `channels.tsv` marks `bad`, unioned the same way |
| `keep_spans` | `list[tuple[float, float]]` | `(onset, duration)` stretches to keep; the rest is marked `BAD_unselected` |
| `keep_spans_from` | `str \| None` | Marker the `keep_spans` onsets are measured from; `run_prep` moves them onto the data axis |
| `bad_spans` | `list[tuple[float, float]]` | `(onset, duration)` stretches marked `BAD_manual` |
| `ignore` | `list[str]` | Aspects to ignore |

### `PrepResult`

Dataclass returned by `run_prep()`.

| Field | Type | Description |
|-------|------|-------------|
| `raw_haemo` | `mne.io.Raw` | Final HbO / HbR signal |
| `sci_scores` | `dict[str, float]` | Per-channel whole-run SCI values |
| `bad_channels` | `list[str]` | Channels marked as bad |
| `censor_spans` | `list[tuple[float, float]]` | The spans `gvtd_censor` annotated, empty when it was off |

`run_prep` carries no quality metrics. They are measured once from the files on disk after prep and post have both finished, by [`qc.sqm_record`](qc.md).

### `run_prep(raw, config, output_dir, source_entities=None, work_dir=None, source_path=None) → PrepResult`

Run the preprocessing pipeline in locked order and write an intermediate SNIRF at every step. `source_path` is the BIDS file `raw` was read from: it becomes the `Sources` entry of the first output, and the aux extraction reads the aux channels back out of it.

### Individual step functions

| Function | Description |
|----------|-------------|
| `intensity_to_od(raw)` | Step 1: intensity → optical density |
| `compute_sci(raw_od, cardiac_l_freq, cardiac_h_freq)` | Whole-run SCI per channel |
| `mark_bad_channels(raw_od, threshold, cardiac_l_freq, cardiac_h_freq, psp_threshold, min_good_frac, screen_scope)` | Step 2: window the recording, count the coupled windows and mark the bads |
| `correct_motion(raw_od, method)` (in `pipeline.motion`) | Step 3: TDDR, wavelet, tPCA, spline, Spline-SG, wavelet then TDDR, or pass-through |
| `od_to_haemo(raw_od, dpf)` | Step 4: Beer-Lambert |

The windowed series themselves are measured in [`qc.metrics.windowed`](qc.md), not here.

---

## `post_pipeline`

### `PostConfig`

See [Pipeline Config Reference](../configuration/pipeline-config.md) for the canonical field descriptions. The Python dataclass mirrors the TOML keys (no separate `[denoising]` / `[glm]` namespacing; all fields live flat on `PostConfig`).

### `run_post(raw_haemo, config, output_dir, mode, source_entities=None, source_path=None) → tuple`

Run postprocessing in `mode = "denoise" | "glm" | "rest"`. Returns:

```python
(result, glm_est, design_matrix, alff_df, fc_df, fc_hbr_df, fc_seed, fc_roi)
```

`glm_est` / `design_matrix` are `None` only when no regression ran: `denoise` and `rest` return them too, fitted on empty events, so the design holds confound columns and no condition. `alff_df` is `None` outside `rest`. The FC products come from `rest`, and from `glm` and `denoise` under `config.fc`; `fc_df` holds the HbO matrix and `fc_hbr_df` the HbR one. `fc_seed` and `fc_roi` are `{chromophore: frame}`, both empty without `--roi-mapping`.

---

## `glm`

| Function | Description |
|----------|-------------|
| `build_design_matrix(raw, ...)` | Build a nilearn first-level design matrix from SNIRF annotations or an external events TSV. |
| `fit_glm(haemo, design_matrix, noise_model)` | Fit the GLM using `mne_nirs.statistics.run_glm`. |
| `compute_contrasts(glm_est, contrast_def, design_matrix)` | Compute linear contrasts from a fitted GLM estimate, one weight vector over the design's columns per contrast. |
| `run_glm_pipeline(haemo, ...)` | One-shot helper: design matrix + fit + contrasts, written as `design.tsv`, `desc-glm_nirsmap.tsv` and `desc-contrast_nirsmap.tsv` (tab separated). Used by both `glm` and `rest` modes. |

---

## `ar_irls`

The solver behind `--noise-model ar_irls`: AR whitening alternated with an iteratively reweighted refit, so residual motion is down-weighted instead of fitted. `fit_glm` dispatches to it on any spec starting `ar_irls` and hands back an ordinary nilearn regression result, so contrasts and the residual work unchanged.

| Function | Description |
|----------|-------------|
| `bic_ar_order(resid, pmax)` | AR coefficients of a residual, order chosen by BIC over `0..pmax` in one Levinson-Durbin pass |
| `fit_channel(y, design, pmax, ...)` | One channel's robust whitened fit: theta, robust scale and effective degrees of freedom |
| `satterthwaite_df(design, weights)` | The effective degrees of freedom a weighted fit actually spent |
| `whiten(X, rho)` | Apply an AR filter to both sides of the regression |
| `resolve_pmax(spec, sfreq)` | Largest order a spec may choose. Bare `ar_irls` follows `auto`'s 4x rule; `ar_irlsN` pins it |

---

## `denoise`

| Function | Description |
|----------|-------------|
| `bandpass_filter(haemo, l_freq, h_freq, method, order)` | Bandpass. `iir` (default) is a zero-phase Butterworth applied with `filtfilt`; `fir` is a hamming-windowed linear-phase filter |
| `filter_kwargs(...)` / `filter_description(...)` | The MNE arguments the choice resolves to, and the sentence the report and the Methods paragraph print |
| `filter_response(...)` / `filter_array(...)` | The designed response, and the same filter applied to a bare array |
| `resample(haemo, sfreq)` | Downsample to target frequency |

---

## `restingstate`

| Function | Description |
|----------|-------------|
| `compute_alff(raw, low_pass, high_pass) → pd.DataFrame` | Per-channel amplitude, columns `channel`, `alff`, `falff`, `malff`, `zalff` |
| `compute_alff_roi(alff_df, raw, roi_map) → pd.DataFrame` | The same four measures averaged within each ROI, one row per ROI and chromophore, with the surviving channel count. Averages after the measure, unlike `compute_fc_roi` |
| `compute_fc(raw, chromophore) → pd.DataFrame` | N × N Pearson correlation matrix indexed by channel name, one chromophore at a time (via `nilearn.connectome.ConnectivityMeasure`) |
| `fisher_z(df)` | Fisher z of a correlation matrix |
| `compute_fc_roi(...)` / `compute_fc_seed(...)` | ROI × ROI and ROI × channel connectivity, with `--roi-mapping`. Both average the signals first |

---

## Hyperscanning

The dyad path lives in `pipeline.hyper` and is several modules, not one. `group_io` finds and loads the members, `alignment` puts them on one clock, `group_quality` decides which channels the metrics may use, and `wtc`, `surrogate`, `coherence`, `roi` and `isc` compute them. `hyper_post` is the dispatcher `nirspipe-hyper` calls; `pair_null` and `pair_null_group` are behind `nirspipe-hyper-pairnull` and `nirspipe-hyper-groupnull`.

### `group_io`

| Function / class | Description |
|------------------|-------------|
| `parse_group_csv(csv_path) → dict[(gid, task), [GroupEntry]]` | Validate the pairs table |
| `member_snirfs(bids_dir, group, validate=False)` | The one raw SNIRF each member's CSV row names (its `session` and `run` columns included), `{subject_id: path}`. Loading, the aligned copy and the channel-decisions session all follow it |
| `load_group_haemo(output_dir, group, desc="preproc")` | Load one haemo stage per member by its `desc-` entity, `desc-preproc` by default |
| `load_group_stage(output_dir, group, desc)` | Load any haemo stage by its `desc-` entity |
| `load_group_raw_bids(bids_dir, group)` | Load raw CW SNIRFs straight from BIDS |
| `warn_outside_passband(...)` / `unfiltered_stage_note(...)` | Say when the band asked for sits outside the band the stage still holds. A cosine drift basis counts as a low edge alongside the high-pass, whichever sits higher, so a residual detrended but never filtered is not reported as unfiltered |

### `alignment`

| Function | Description |
|----------|-------------|
| `align_recordings(raws, task)` | Trigger-based alignment, then trim to the shortest |
| `write_aligned_member(raw_aligned, snirf_path, out_dir, group_id)` | Write one member's aligned SNIRF and events under its source file's name, recording the offset in its own `_nirs.json` (`align_group`, `align_step`, `align_trigger`, `align_offset_s`, `aligned_duration_s`). Refuses a member another group already aligned into the same tree |
| `align_like(...)` | Put one recording on an alignment already resolved |
| `trim_to_shortest(raws)` | Trim without alignment (for resting state) |
| `alignment_params(...)` / `resolve_analysis_window(...)` | The offsets that pair the two clocks, and the `--tstart` / `--tend` window on the aligned one |
| `crop_aligned_window(...)` | Cut to that window, where a caller needs the cut rather than a read-out |
| `normalize_raws(raws)` | Z-score channels per subject |

### `group_quality`

| Function | Description |
|----------|-------------|
| `compute_group_sqm_raw(group, raws, sci_threshold, output_dir, ...)` | Per-subject SQM, and the `group-<id>_task-<task>_desc-subject_qc.tsv` / `desc-channel_qc.tsv` under `group-<id>/nirs/` |
| `write_group_bads(...)` / `apply_group_bads(...)` | The rejected set under `--bads-scope`, written once and applied to every member |
| `resolve_group_bands(...)` | Read the separation bands back off each member's record, and refuse a group whose members disagree |

### `wtc`

| Function / class | Description |
|------------------|-------------|
| `compute_wtc(...) → WTCResult` | Pairwise wavelet coherence via `pycwt` |
| `WTCResult` | The coherence, phase, frequency axis, times and cone of influence |
| `wtc_grid_params(...)` / `cone_margin_s(...)` | The scale grid a band implies, and the margin the cone costs |
| `wtc_band_mean(...)` / `window_result(...)` | Average a map over a band; read one window out of a whole-run transform |
| `wtc_phase_by_scale(...)` | The same collapse kept per scale, for the phase |

### `surrogate`

| Function / class | Description |
|------------------|-------------|
| `phase_scramble(...)` / `compute_wtc_phase_null(...)` / `NullDraws` | The phase-scrambled null |
| `compute_wtc_pair_null(...)` | The same accounting for the re-paired null, whose draws arrive from outside |

### `roi`

| Function | Description |
|----------|-------------|
| `roi_mean_of_channels(...)` / `roi_mean_of_homologous(...)` / `roi_maps_from_channels(...)` | ROI averages of the channel results |

### `isc`

| Function | Description |
|----------|-------------|
| `compute_isc(...)` / `compute_isc_pairs(...)` / `roi_mean_of_isc(...)` | Inter-subject correlation, as a matrix and one row per pairing |

### `hyper_post`, `pair_null`, `pair_null_group`, `wtc_store`, `whiten`, `wtc_null`, `wtc_aggregate`

| Module | Does |
|--------|------|
| `hyper_post` | `run_hyper_post(group_id, task, aligned_raws, output_dir, config, ...)`, the dispatcher behind `nirspipe-hyper`, plus `HyperPostConfig` / `HyperPostResult` |
| `pair_null` | `run_pair_null(...)`, the re-paired null behind `nirspipe-hyper-pairnull`: each member against partners from the other groups of the same task, written as `null-pair` tables beside the real ones |
| `pair_null_group` | `write_group_null(output_dir, task, chroma, null, ...)`, behind `nirspipe-hyper-groupnull`: the channel-averaged null per occasion and over the cohort, written at the tree root as `chromo-<c>_task-<t>_cond-all_null-<pair\|phase>_stat-<wtc\|isc>_desc-{bycell,byoccasion,cohort}_relmat.tsv`, one set per statistic whose draws exist; `p_correction` and `isc_test` set the correction and the correlation's read |
| `wtc_store` | Save and load the time-frequency maps (`--wtc-save-maps`), and `reband` / `reband_tree` behind `nirspipe-hyper-band`; `level_params` / `level_mismatch` decide whether a null's saved level still fits a run |
| `whiten` | `whiten_raws(raws, order_s, sep_bands=None)`, the AR prewhitening behind `--wtc-whiten`: one order in seconds for every long channel of both members, before the coherence only |
| `wtc_null` | Draw the phase-scrambled null and rank a real table inside it |
| `wtc_aggregate` | Concatenate the per-dyad long tables into one per kind at the tree root, named as the inputs minus `group-` and `task-`, behind `nirspipe-hyper-merge`; matrices stay with their dyad |

See [Hyperscanning Pipeline](../hyperscanning/pipeline.md) for parameter detail and failure modes.

---

## `crop`, `edit_markers`

Standalone utilities consumed by `nirspipe-prep`:

| Module | Key function |
|--------|--------------|
| `crop` | `crop_snirf(bids_dir, derivatives_dir, sub, ses, task, run, *, tmin, tmax, segments_path, combine, align, trigger_name, validate, input_desc, margin_s)`, and `crop_snirf_from_path` for a file that is not in a BIDS tree |
| `edit_markers` | `export_markers(...)` / `apply_markers(..., tsv, shift, set_duration, rename)`, and `apply_markers_from_df` |
