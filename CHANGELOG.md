# Changelog

## [Unreleased]

### Added

**Pipeline (v0.1–v0.3 scope, needs review)**
- End-to-end preprocessing: OD conversion, SCI/PSP channel QC, motion detection & correction (TDDR), Beer-Lambert
- Pipeline reorganized into `pipeline/` module (`prep_pipeline.py`, `post_pipeline.py`, `glm.py`, `denoise.py`, `channel_registration.py`)
- Task GLM postprocessing via nilearn (`pipeline/glm.py`)
- Denoise mode: bandpass filter + optional resample (`pipeline/denoise.py`)
- BIDS Derivatives sidecar writing; intermediate snirf written at each prep step
- Full CLI (`fnirs-pipe`) via Typer; all planned flags wired
- Separate `fnirs-recon` CLI for raw → BIDS conversion

**QC report (demo quality, needs review)**
- Per-subject HTML report via Jinja2 + Plotly
- Sections: OD traces, SCI/PSP heatmaps, GVTD timeseries, motion carpet, HbO/HbR correlation, PSD, brain views, short-channel PSD, IQM table
- IQM sidecar: `sub-{id}_iqm.toml` + `sub-{id}_channel_metrics.csv`
- GLM section: Design matrix timeseries figure + nilearn heatmap
- `_guard` context manager — uniform error handling across all section builders

### Fixed
- `_section_motion`: bad-segment zoom was incorrectly nested inside motion panel's except block; now runs independently

### Known Issues
- `_write_denoised_snirf` references `config.ica` which does not exist on `PostConfig` → `AttributeError` at runtime in denoise/glm mode
- Functional connectivity mode raises `NotImplementedError`
- Group-level report raises `NotImplementedError`

---

<!-- Instructions for future entries:

## [0.1.0] - YYYY-MM-DD
### Added      <- new features
### Changed    <- changes to existing behaviour
### Deprecated <- soon-to-be-removed features
### Removed    <- removed features
### Fixed      <- bug fixes
### Security   <- vulnerability fixes

-->
