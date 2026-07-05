# Changelog

All notable changes to this project will be documented in this file.

<!-- Format: Keep a Changelog (https://keepachangelog.com/en/1.0.0/) -->

## [Unreleased]

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
