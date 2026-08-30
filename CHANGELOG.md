# Changelog

All notable changes to this project will be documented in this file.

<!-- Format: Keep a Changelog (https://keepachangelog.com/en/1.0.0/) -->

## [Unreleased]

### Added
- The quality record gains a `motion_post` section: GVTD, spikes, SCI and PSP measured again on the motion-corrected file. Set against the same keys in `raw`, it says whether the correction reduced motion and whether it cost any cardiac signal
- The quality record gains a `windowed` section: the per-window SCI, PSP and GVTD series, with the window length they were binned on. They used to exist only in what `fnirs-qc prep-raw` wrote
- Metric tooltips in the subject report now say which processing stage the number was measured on, which matters for the keys that appear in two sections
- `fnirs-prep crop` accepts a `task` column in the segments table: each segment is written under that task entity instead of `_seg-NN`, so a recording holding several conditions becomes a BIDS dataset the pipeline can read back one condition at a time
- `fnirs-qc hyper-post --wtc-roi-cross` crosses the two brains' ROIs instead of pairing each with its counterpart, so one person's PFC can be tested against the other's TPJ. `hyper-wtc-roi.tsv` gains a `label2` column, the report an ROI × ROI matrix. Needs `--roi-mapping`
- `fnirs-qc hyper-post --wtc-mc-count` sets how many surrogate series stand behind each significance contour (default 300, unchanged). It is what the runtime is spent on and it scales with the value, so a run can be previewed cheaply and settled expensively

### Changed
- `snr_pass_rate` counts every channel, not only the ones with a finite SNR. A flat or saturated channel has no SNR at all and used to drop out of the fraction entirely, so a recording whose channels were dying read as one whose channels were passing. The new `n_flat_channels` says how many those are. **`snr_pass_rate` falls for any run that has them**
- The subject report reads the spike and motion-correction spans from the quality record instead of detecting them a second time, and reads the optical density either side of the correction back from disk. Prep no longer keeps a copy of it in memory
- The subject report's per-window SCI and PSP panel reads the quality record instead of values passed from prep, and the series are measured on the motion-corrected file. Prep no longer computes them
- Tables you hand the package read by extension: `.tsv` tab-separated, `.csv` comma-separated, anything else sniffed. Applies to events, segments, the pairs file and `participants.tsv`. What the package writes stays tab-separated
- zALFF standardizes by the population SD (n) instead of the sample SD (n-1). Values grow by `sqrt(n/(n-1))`.

## [0.21.0] - 2026-08-24

### Added
- Topographic maps of the condition-averaged response in the subject report, at time points after onset, one row per condition and chromophore. The first spatial view of activity that does not wait for the GLM
- `fnirs-qc hyper-post --desc` picks which per-subject stage the inter-brain metrics read (default `preproc`; `errts` reads the confound-regression residual, so short-channel regression can precede a coherence analysis). Optical-density stages are refused
- `fnirs-qc hyper-post` writes `group-<id>_task-<task>_hyper-wtc.tsv` (and `hyper-wtc-roi.tsv` with `--roi-mapping`): one coherence value per pair and channel, averaged over the requested band inside the cone of influence. The figures and a group analysis now read the same numbers
- Rest mode writes `desc-<chromo>_fcseed.tsv` and `_fcseedz.tsv` when `--roi-mapping` is given: each ROI's mean signal against every channel. Cells for a seed's own channels are blank
- The subject report draws the seed maps as flat maps, one per seed ROI and chromophore, each channel coloured by its correlation with that seed. Channels inside the seed are grey rather than zero-coloured, rejected channels are faded, and short channels are left out

### Fixed
- ISC, band coherence and windowed coherence pair channels by S-D label instead of by position, so a dyad whose members had different channels rejected is no longer compared off-by-one. **These values change for any such dyad**
- The `raw_long` and `raw_short` quality sections keep rejected channels, which used to make their mean SCI and channel retention read better than the run was
- Provenance is no longer empty when the pipeline is handed a recording it did not read from disk
- A run where no channel passes the SCI threshold now stops there and says so, instead of failing later inside Beer-Lambert
- `--motion-correction wavelet` sets its outlier threshold per wavelet scale over the whole recording, not per time window
- Rest mode no longer writes ALFF/fALFF when the drift model leaves linear drift in the data (`--drift-model none`, or `polynomial` with `--drift-order 0`); that drift's leakage falls inside the ALFF band
- The windowed SCI and PSP heatmaps plotted each window at roughly half its true time, so the report's time axis covered only the first half of the recording
- GVTD per-window series share the time axis of the windowed SCI and PSP instead of drifting apart from it over a run
- A cardiac band the filter cannot use no longer takes the GVTD per-window series down with the SCI and PSP ones
- A metrics database written by an older version gains any column it is missing instead of failing every insert
- Fisher z no longer zeroes the leading diagonal of a non-square matrix, where the cells are ordinary values rather than self-correlations
- The seed-map sidecar lists the channels each seed was built from, not the ones the ROI mapping asked for

### Changed
- The subject report's trial images are split by condition, with the pooled image kept first. Stacking every condition's trials together hides a response only one condition drives. They were called erpimages; the figure files are now `trialimage_*.html`
- The per-channel layout figure shows the condition-averaged response, not the continuous signal. Runs without events keep the continuous view
- The hyperscanning WTC, coherence and ISC read long channels only. Short channels carry scalp physiology that two people in one room share whatever their brains are doing. **Any montage with short channels loses those rows from its hyper figures**
- Inter-brain metrics stop when the members of a group were sampled at different rates, instead of applying the first member's rate to everyone
- Functional connectivity is plain Pearson, not the shrinkage estimate inherited from a library default, which shrank hardest for subjects with shorter runs or more rejected channels. **Every FC and FCZ value changes; weak connections change most**
- Quality records store `qc_window_s`, so a group report whose subjects were run at different `--window-length` values says so instead of stacking incompatible rows in one heatmap
- `Mean PSP` is labelled with its 10 s window in the report. The window is pinned there and does not follow `--window-length`, which still sets the windowed PSP heatmap
- zALFF standardizes by the sample standard deviation. Values shrink by sqrt((n-1)/n), where n counts the channels of one chromophore: 5.1% at 10 per chromophore, 2.6% at 20
- `--motion-correction wavelet` reaches artifacts up to about 25 s long; it used to stop at about 1.6 s

## [0.20.0] - 2026-08-11

### Added
- `<sub>_<task>_desc-sqm_nirs.json`: one quality record per run, grouped into sections by what each metric was measured on (`raw`, `raw_long`, `raw_short`, `motion`, `preproc`, `final`). A subject with several tasks gets one record per task. The `raw*` sections include rejected channels, the rest exclude them
- Sidecars record per-channel SCI and the respiration band, so a record can be rebuilt from an output directory alone
- The QC report's metrics panel explains itself: hover any metric for what it is and which way is good, and the five that decide whether a run is usable are marked. The explanations used to live only in a separate guide

### Changed
- Quality metrics are measured once and written once. They used to be computed three to four times per subject, and the number shown in the report did not match the number on disk
- `sub-XX_sqm.toml` and `sub-XX_sqm_raw.toml` are gone, replaced by the per-run JSON. Group aggregation now sees pipeline runs, not only `fnirs-qc prep-raw` output
- The metrics database stores one row per section and fills in the task
- `fnirs-qc prep-raw` writes `<sub>_<task>_desc-sqmraw_nirs.json`. It used to write `desc-sqm`, the same name the pipeline uses, so running both into one output directory left whichever finished last. The group report reads both and prefers the pipeline's record for a run that has both

### Fixed
- The resting-state FC heatmap and connectogram lost their HbR half, or failed to render at all
- A single failing metric no longer discards a run's whole quality record; only the section it belongs to is lost
- The QC report says so when a run's quality record is unreadable, instead of showing an empty metrics panel
- The provenance table lists the metric names again for the quality record
- `<sub>_channel_metrics.csv` gains the run's task and session, so a subject with several tasks keeps one file per task instead of only the last
- Short and long channels are decided the same way everywhere. The QC report, the `prep-raw` figures and the quality record each used a different separation cutoff, so the same channel could be short in one and long in another
- Cardiac Power was silently unavailable on any recording with a rejected channel: `cp_mean`, `cp_per_channel` and `cp_pass_rate` all came back empty
- Peak spectral power is measured on optical density, matching the windowed PSP series shown beside it in the report. The two were measured on different signals; values shift in the fourth decimal
- Rebuilding a quality record from an existing derivatives tree lost every SCI metric when the sidecar carried no stored per-channel scores
- The group report's boxplots grouped every metric under "Other". Grouping matched bare metric names and the columns had gained their section prefix
- Per-wavelength CV (`cv_mean_760` and the like) vanished from a record whenever the CV/SNR step failed, instead of being reported as missing like every other metric
- A derivatives tree that has been moved, or is read on another machine, keeps its raw-signal quality metrics. They were silently skipped because the sidecars name the original recording by an absolute path that no longer resolved

## [0.19.0] - 2026-07-31

### Added
- `fnirs-qc hyper-post --wtc-seed` makes the wavelet-coherence significance contour reproducible; without it the Monte Carlo surrogates, and pycwt's on-disk cache of them, made two runs on the same data disagree
- Sidecars record the shape of the data each step left behind: channel count, how many were marked bad, sampling rate and duration
- The provenance diagram shows the settings each step used (filter band, dpf, motion-correction method, GLM models) and the data shape at every node, instead of the step name alone
- `*_alff.tsv` and `*_glm_results.csv` gain a `bad` column, and connectivity, ALFF and GLM sidecars list the rejected channels: results for rejected channels are kept and flagged rather than dropped, so the table shape stays predictable for group analysis
- The SQM checkpoints appear in the provenance diagram; the raw one points back at the original recording it measured, not at the file it happens to sit beside

### Changed
- The provenance diagram moves from `sub-XX/logs/` to `sub-XX/figures/provenance.png` and is shown in the QC report; `fnirs-qc provenance` writes to the same place, so re-rendering refreshes the diagram an existing report displays
- The two SQM checkpoints name the metric families they computed, so the diagram distinguishes the raw checkpoint from the thinner one post-processing leaves behind, instead of showing both as `sqm`
- The QC report's Provenance section lists every output with the step that made it and a line saying what that step did; the SQM rows expand to the full list of metric names each checkpoint recorded
- The Methods paragraph is built from the sidecars the run wrote rather than from the configuration, so it describes what actually ran
- The provenance diagram renders at 300 dpi, matching the other report figures
- Per-subject log, run record, script and provenance diagram lose the timestamp in their names (`sub-01.log`, `sub-01.toml`); a re-run overwrites them
- The 3-view brain figure colours the source-detector links by SCI, like the flat map beside it, instead of spheres at channel midpoints; the surface is opaque so the far side of the head no longer shows through, and optodes are red (source) / blue (detector)
- Writing two different files under one pipeline stage now fails instead of silently crediting an output to the wrong source
- Postprocessing now rejects a `desc-` that disagrees with the stage stamped on the data, matching the check preprocessing already had
- GLM outputs carry the subject and task of their input: `sub-01_task-tapping_design_matrix.csv`, and likewise for `glm_results.csv` and `contrasts.csv`

### Fixed
- A subject whose short channels were all rejected for one chromophore crashed the GLM instead of continuing without short-channel regressors
- `--drift-model cosine` without `--drift-high-pass` now says so before the run starts, instead of failing minutes later inside nilearn with a `NoneType` multiplication error
- The Methods paragraph stopped after the Beer-Lambert sentence: filtering, resampling and the GLM were never described, because the report built the text without the post-processing configuration
- Channels marked bad during preprocessing were unmarked again as soon as postprocessing reloaded the data, so filtering, resampling and the GLM all treated SCI-rejected channels as good; the marks now travel with the file
- A short channel rejected by SCI was still averaged into the short-channel regressor, and that regressor sits in the design matrix, so one bad channel shifted the fit of every channel
- mALFF and zALFF were standardized against a mean and standard deviation taken over all channels, rejected ones included, distorting the value of every good channel
- ROI connectivity averaged rejected channels into their ROI signal, carrying them into every correlation that ROI took part in
- Brain figure drew each channel from its midpoint to its source instead of source to detector
- Design matrix labels overlapped when regressors were numerous
- Postprocessing ran once per matching subject found anywhere under the output directory, so a nested output tree from an earlier run was silently reprocessed and its results overwrote the real ones
- With more than one task per subject, every task wrote the same `design_matrix.csv` and `glm_results.csv`, leaving only the last

## [0.18.0] - 2026-07-28

### Added
- Sidecars now record the file each output came from (`Sources`) and the step that produced it
- ALFF, connectivity, GLM and hyperscanning outputs now have sidecars
- Rest mode writes `desc-filtered` and `desc-errtsbroad`, previously discarded
- Each run writes a provenance flow diagram (PNG + mermaid) to `sub-XX/logs/`, showing which file every output came from and via which step
- `fnirs-qc provenance <output_dir>` renders that diagram for any past run, reading only the sidecars already on disk

### Changed
- Cardiac/respiration band power and low-frequency drift are now written only to the preprocessing checkpoint; on filtered data they describe the filter, not the recording

### Removed
- Durbin–Watson on GLM residuals: ~2 by construction once prewhitening runs

### Fixed
- Cardiac/respiration band power was silently missing on recordings with bad channels
- Rest mode overwrote its SQM file, losing the metrics computed earlier in the run

## [0.17.0] - 2026-07-28

### Added
- Standardized ALFF outputs `malff` (channel value over the chromophore mean) and `zalff` (per-chromophore z-score) in `*_alff.tsv`, for group-level comparison

### Changed
- The per-subject reproducible script (`sub-XX/logs/*_script.py`) is now a step-by-step transcript of the run: one block per processing stage (OD, SCI, motion, Beer-Lambert, filter, GLM/rest) with its parameters and outputs, instead of an opaque call into the pipeline
- ALFF/fALFF are now computed on a broadband residual (drift removed, not band-pass filtered) so fALFF spans the full spectrum as defined; functional connectivity keeps the band-pass filtered residual
- Functional connectivity (channel `*_fc.tsv`, ROI `*_fcroi.tsv`, and their Fisher-z variants) is now written per chromophore as separate HbO/HbR matrices (`desc-hbo` / `desc-hbr`) instead of one matrix mixing both; HbO and HbR anti-correlate, so a mixed matrix was not meaningful
- Cardiac and respiration band power/fraction are now reported per chromophore (`*_hbo` / `*_hbr`)

### Fixed
- fALFF was close to 1 on nearly every channel because it was computed on band-pass filtered data, leaving no out-of-band power in its denominator; it is now valid
- Subject report showed a blank low-frequency drift (HbO) value due to a stale metric key

## [0.16.0] - 2026-07-08

### Added
- Motion-correction footprint (experimental): which timepoints a motion correction actually repaired, summarised like framewise motion (`motion_corrected_pct` / `_num` / `_n_segments`) and drawn as a coloured band beneath the carpet and per-channel motion figures
- Spike frame metrics: fraction of all channel-samples flagged (`spike_pct`) and fraction of timepoints where many channels spike together (`spike_pct_frames`), alongside the existing spike count; spikes are also drawn as a band on the motion figures
- erpimage in the subject report for task data: single-trial HbO heatmaps per channel and per ROI (each stimulus repetition is a row, with a trial-average trace below), computed on the denoised signal and smoothed across trials
- Fisher z-transformed functional connectivity output (`*_fcz.tsv`, ROI `*_fcroiz.tsv`) for group-level statistics, alongside the raw correlation matrices

### Changed
- Windowed SCI/PSP/GVTD now share one sliding-window length (default 10 s, previously a mix of 30 s / 10 s), exposed as `--window-length` on `fnirs-prep`, `fnirs-qc prep-raw`, and `fnirs-qc window-raw`
- Spike detection now runs on the motion-band-filtered optical density (cardiac removed first), so the count reflects motion rather than cardiac pulsation
- Global correlation is now reported before → after the short-channel regression (the step that removes global/systemic signal) instead of before/after the bandpass, which inflated it; a single value is shown when no short-channel regression ran
- The GVTD carpet figure splits the raw and filtered traces into separate panels, with the plotted GVTD downsampled by max-pooling so it stays readable (metric values unchanged)
- Global correlation and spike metrics are now marked experimental
- QC figures render at ≥300 dpi, and the before/after denoising carpet no longer has heavy panel borders

### Removed
- tSNR (haemoglobin mean/std): ΔHbO/ΔHbR has no stable baseline, so its mean/std is not a meaningful signal-to-noise ratio; raw-intensity SNR already covers channel signal stability

## [0.15.0] - 2026-07-06

### Added
- Wavelet motion correction (`--motion-correction wavelet`): zeroes outlier wavelet detail coefficients per channel in OD space
- Filtered GVTD (0.01–0.5 Hz motion band) reported alongside the raw GVTD, plus per-run motion summaries above an adaptive (histogram-mode) threshold: number of motion frames, percent of run, and the threshold value
- Motion figures (carpet + per-channel detail) overlay the raw and filtered GVTD traces and draw the motion threshold; group reports gain a filtered-GVTD time × subject heatmap
- Cardiac and respiration band power are also reported as a fraction of total spectral power (`cardiac_band_frac` / `resp_band_frac`), comparable across subjects regardless of overall amplitude
- `fnirs-qc epoch`: per-trial QC — one window per task event (read from the SNIRF or a `--events-csv`), recomputes SQM per trial and renders a trial × metric TSV + HTML. `--mode epoch|duration` is required (epoch mode needs `--tmin`/`--tmax`); SQM values are reported per trial without a good/bad label
- Channel-standardized GVTD (`gvtd_vstd_*`): each channel's temporal derivative is divided by its own SD before the cross-channel RMS, so high-dynamic-range channels no longer dominate the motion index
- Global correlation per chromophore (`gcor_hbo` / `gcor_hbr`): mean pairwise channel correlation; the subject report shows it before → after denoising (a drop means systemic/global signal was removed)
- Durbin–Watson on GLM residuals (`durbin_watson_mean`): residual autocorrelation check (~2 = white residuals); shown in the GLM report section and saved with the post-processing SQM
- Per-window GVTD now reports p95 (worst-moment) alongside the mean, so a brief motion burst is not averaged away; group reports gain p95 time × subject heatmaps and the subject report shows the GVTD p95 scalar

### Changed
- The cardiac band (`--cardiac-l-freq` / `--cardiac-h-freq`) is now required, with no default: it is population-dependent (adult vs infant heart rate) and now consistently sets the band for channel SCI, PSP, Cardiac Power, and cardiac band power, so a non-adult band is no longer silently ignored by some metrics
- The respiration band (`--resp-l-freq` / `--resp-h-freq`) is now a required option too (population-dependent, same rationale as the cardiac band)
- Spectral confound metrics renamed `residual_cardiac_power` / `residual_resp_power` → `cardiac_band_power` / `resp_band_power` (they measure band power present, not a post-filter residual)
- Spike count now uses a robust (MAD-based) threshold, so a few large spikes no longer inflate the threshold and hide themselves
- Cardiac Power is marked experimental (overlaps PSP; may be removed); SCI and PSP remain the primary channel-quality metrics
- Group report distributions are now split into per-scale charts (coupling, GVTD amplitude, counts, drift, etc.) showing real values with one colour per metric; clicking a point opens that subject's report

### Fixed
- fALFF is now the fraction of total spectral amplitude in the low band (sum/sum, in [0, 1]), matching its definition; was previously a ratio of mean amplitudes. ALFF unchanged (validated against a reference implementation).
- Cardiac Power now sums band power (was averaging) and is computed on optical density, so it lies in [0, 1] and matches the CP ≥ 0.5 quality gate
- `pct_data_retained` no longer double-counts overlapping bad segments
- Low-frequency drift amplitude is now estimated from a polynomial trend fit, avoiding the filter edge artifacts of the previous 0.01 Hz low-pass

---

## [0.14.0] - 2026-07-04

### Added
- ROI-level WTC in the hyperscanning post report: coherence on HbO averaged within each ROI from `--roi-mapping`, excluding the union of all subjects' bad channels so both use the same channel set
- Hyper post report marks bad channels: ISC matrix blanks bad rows/columns and drops their connectogram arcs; per-channel WTC selector tags bad channels
- Denoising before/after carpet in the subject QC report: HbO and HbR shown separately, both scaled by the pre-denoising SD so reduced fluctuation renders paler
- `--roi-mapping` on the main pipeline: a JSON file mapping ROI labels to channel lists, used to group the denoising carpet by ROI and to compute ROI-level connectivity
- ROI-level functional connectivity (`_fcroi.tsv`) in rest mode when `--roi-mapping` is given: each ROI's HbO channels are averaged, then correlated between ROIs; channel-level FC is unchanged

### Changed
- `--exclude-channels` renamed to `--bad-channels`: channels are now marked bad (kept in the data, unioned with SCI-detected bads) instead of being dropped
- Quality metrics renamed from IQM to SQM (signal quality metrics): sidecars are now `sub-<id>_sqm.toml` and `sub-<id>_sqm_raw.toml`
- Motion QC carpet now shows all channels (both wavelengths) instead of a single wavelength

### Fixed
- PSD before/after panel and the run-record database now reflect bandpass cutoffs set in a TOML config file; previously only command-line `--high-pass` / `--low-pass` were read, so a TOML-configured filter showed as no filter
- GVTD trace in the motion carpet is now computed at full resolution instead of on the down-sampled display data, so its spikes and p95 line match the reported GVTD metric

### Known Issues
- `compute_alff` numerical validation pending

---

## [0.13.0] - 2026-07-03

### Changed
- All CLIs reworked to follow the BIDS App convention: `--participant-label` and other list options now accept space-separated values (e.g. `--participant-label 01 02 03`) as well as repeated flags
- `--help` output is grouped into labelled sections

---

## [0.12.0] - 2026-07-03

### Added
- New GUI "Recon" page: detect raw snirf files in a folder, assign BIDS metadata per file in a table, preview the commands, and batch-convert to BIDS
- Command preview (Analysis and Recon pages) has a shell selector so line-continuation matches bash / cmd / PowerShell
- Pipeline auto-detects already-OD input, skips the OD-conversion step, and marks intensity-only QC metrics as unavailable

### Changed
- `fnirs-recon --subject` help clarifies that labels are alphanumeric and can encode group, e.g. `patient01`

### Fixed
- GUI "generate command" emitted invalid multi-subject commands; participant/session/task labels now repeat the flag per value
- QC no longer crashes on low sampling-rate data — PSD and cardiac-power frequency limits clamp to the Nyquist frequency
- QC brain views no longer fail when the static-image export backend was missing (now a hard dependency)

---

## [0.11.0] - 2026-05-25

### Added
- Self-contained rating viewers: rating bar inlined into the QC HTML template; static open works read-only, `fnirs-rate` open enables writes
- Section ratings + channel decisions persist via `/save_*` REST endpoints only — HTML never modified

### Changed
- `HyperRatingApp` / `RawRatingApp` / `FNIRSRatingApp` refactored: no more regex reverse-parsing or string-concat injection

### Fixed
- `HyperRatingApp` channel-decisions sidecar is now session-aware

---

## [0.10.0] - 2026-05-30

### Added
- `fnirs-qc group-raw` and `group-hyper-raw` aggregate per-subject / per-dyad SQM into cohort HTML reports (heatmap, boxplots, outliers, sortable table)
- `fnirs-qc window-raw` crops each subject's raw to a time window and re-runs the group-raw layout — for "is this minute dropping for everyone?"
- prep-raw SQM JSON now stores `sci_per_window` / `psp_per_window` / `gvtd_per_window`; group reports render time × subject heatmaps for these

### Changed
- QC output directory follows a BIDS-derivatives layout: per-subject / per-group `figures/` and `[ses-XX/]nirs/` subdirs
- QC filenames standardised to `_nirs.*` suffix with BIDS `desc` entity
- Hyper raw report converted to an iframe shell with auto-resize; Plotly loaded from CDN

### Fixed
- `fnirs-qc` CLI now initialises logging

---

## [0.8.0] - 2026-05-23

### Added
- `fnirs-log merge` CLI + `utils/job_db.py`: JSONL run events consolidated into a SQLite database (`pipeline_executions`, `runs`, `sqm`, `command_outputs`)
- `PrepResult` exposes `sqm_raw` and `sqm_final` for downstream use
- Paragraph-style Methods boilerplate with a "Rendered" tab in the HTML report

### Removed
- `--mode connectivity` (was never implemented)
- `--ica` postprocessing flag (ICA for fNIRS is experimental; removed to simplify the post pipeline)
- `--segments-path` / `--crop-tmin` / `--crop-tmax` on `fnirs-pipe` (superseded by `fnirs-prep crop`)

---

## [0.7.0] - 2026-05-14

### Added
- `fnirs-gui` CLI entry point launches a Dash multi-page application
- Data Preparation page: SNIRF file loader, editable stimulus marker table, per-subject SQM display
- Analysis page skeleton wired into the sidebar router

---

## [0.6.0] - 2026-05-13

### Added
- Resting-state analysis mode added to the hyperscanning pipeline (`--mode rest`)
- ALFF and fALFF computation (`pipeline/restingstate.py`)
- Resting-state functional connectivity (Pearson correlation matrix across channels)
- Resting-state QC figures and report section
- `--trim` option for hyperscanning pipeline to discard leading/trailing seconds before analysis

---

## [0.5.0] - 2026-05-13

### Added
- ISC (inter-subject correlation) computation and heatmap visualization in hyperscanning post report
- WTC (wavelet transform coherence) computation and figure builders
- Connectogram visualization for hyperscanning functional connectivity (`qc/figures/connectogram.py`)
- Hyperscanning post-processing QC report (`qc/hyper_report.py`) with WTC, ISC, and connectogram sections
- Hyperscanning group-level raw QC report: Plotly figure builders for multi-dyad channel summaries
- Normalization option for hyperscanning raw data; `z_score` utility added

### Changed
- Hyperscanning functionality consolidated from a standalone CLI into `pipeline/hyperscanning.py`; `fnirs-hyper` CLI removed and subsumed under `fnirs-qc`

---

## [0.4.0] - 2026-05-12

### Added
- `fnirs-qc` CLI (`cli/qc.py`) with `prep-raw` and hyperscanning subcommands
- Interactive raw QC viewer: standalone HTML page with Plotly channel traces (`qc/app.py`)
- SQM expansion: cardiac power band metrics and tSNR per channel (`qc/quantitative_metrics.py`)
- Channel quality summary figure integrated into prep-raw report and templates

### Changed
- SCI computation corrected; figure axis labels clarified across multiple panels

### Fixed
- Empty epoch subsets no longer crash `build_channel_figure`
- Unused imports and variables cleaned up across figure modules

---

## [0.3.0] - 2026-05-05

### Added
- Task GLM postprocessing via nilearn (`pipeline/glm.py`): design matrix construction, contrast estimation
- Denoise mode: bandpass filter + optional resampling (`pipeline/denoise.py`)
- GLM QC section in per-subject report: design matrix timeseries figure + nilearn activation heatmap
- Raw residuals written as a separate SNIRF file after GLM fitting
- Motion correction figures improved: bad-segment zoom panel now runs independently of the motion carpet

### Fixed
- `_section_motion` bad-segment zoom was incorrectly nested inside the motion panel's `except` block; now executes unconditionally after the main panel

---

## [0.2.0] - 2026-05-05

### Added
- Per-subject HTML QC report generated via Jinja2 + Plotly (`qc/prep_raw_report.py`)
- Report sections: OD traces, SCI/PSP heatmaps, GVTD timeseries, motion carpet, HbO/HbR correlation panel, PSD, brain views, short-channel PSD, SQM table
- SQM sidecar files: `sub-{id}_sqm.toml` + `sub-{id}_channel_metrics.csv`
- `_guard` context manager for uniform per-section error handling; failed sections render a warning card rather than crashing the report

---

## [0.1.0] - 2026-05-04

### Added
- End-to-end single-subject preprocessing: OD conversion, SCI/PSP channel QC, motion detection and correction (TDDR), Beer-Lambert law
- Pipeline reorganized into `pipeline/` module: `prep_pipeline.py`, `post_pipeline.py`, `glm.py`, `denoise.py`
- BIDS Derivatives sidecar writing; intermediate SNIRF written at each prep step
- `fnirs-pipe` CLI via Typer with all planned preprocessing flags wired
- `fnirs-recon` CLI for raw → BIDS conversion
- Run record and script utilities (`utils/run_record.py`, `utils/run_script.py`)
- Logging setup (`utils/logging.py`)
