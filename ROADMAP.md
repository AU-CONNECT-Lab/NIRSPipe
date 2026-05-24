# Roadmap

Status: `[ ]` not started · `[~]` in progress · `[x]` done · `[?]` needs review

---

## v0.1 — Preprocessing Core `[x]`

End-to-end single-subject preprocessing: OD conversion, SCI marking, motion correction (TDDR), Beer-Lambert.
Pipeline reorganized into `pipeline/prep_pipeline.py`; intermediate SNIRF written at each step.

## v0.2 — QC Preprocessing Report `[x]`

Per-subject HTML report: SCI/PSP heatmaps, GVTD timeseries, motion carpet, HbO/HbR correlation panel, PSD, brain views, IQM sidecar.

## v0.3 — Postprocessing & GLM `[x]`

Task GLM (`pipeline/glm.py`) and denoise mode implemented.
GLM QC section added to report; raw residuals written as a separate SNIRF.

## v0.4 — Raw QC CLI `[x]`

`fnirs-qc` CLI implemented with `prep-raw` subcommand and interactive HTML viewer.
IQM expanded with cardiac power and tSNR metrics; channel quality summary figure added.

## v0.5 — Hyperscanning Pipeline `[x]`

Hyperscanning pipeline consolidated into `pipeline/hyperscanning.py`.
WTC computation, connectogram visualization, ISC computation.
Group-level raw QC report and per-dyad post-processing QC report implemented.

## v0.6 — Resting-State Analysis `[x]`

Resting-state mode added to hyperscanning pipeline: ALFF/fALFF computation, functional connectivity matrix, resting-state QC report.

## v0.7 — GUI Interface `[~]`

Dash-based `fnirs-gui` application with multi-page routing.
Data Preparation page (SNIRF loader, marker editor, IQM display) implemented.
Analysis page is a stub; callbacks and data flow not yet wired.

## v0.8 — Run Logging & IQM Database `[x]`

JSONL-based event logging wired into `fnirs-pipe` run flow.
`fnirs-log merge` consolidates JSONL files into a SQLite database (`pipeline_executions`, `runs`, `iqm`, `command_outputs`).
`PrepResult` now returns `iqm_raw` and `iqm_final` for downstream use.

## v0.9 — GUI Analysis Page & Pipeline Integration `[ ]`

Complete the Analysis page in `fnirs-gui`:

- Connect pipeline execution (prep, GLM, denoise) to the GUI via background callbacks
- Display live progress and log output in the interface
- Show report preview or figure output after run completes

## v0.10 — Group-Level QC Report `[ ]`

Aggregate IQM scalars from per-subject TOML sidecars into a group-level summary.

- Distribution plots per metric, outlier flagging across subjects
- Output: group-level HTML report + CSV/TSV table of all subject IQMs

---

## Pending decisions

- Short-channel regression: `--short-channel {none,mean,pca}` flag wired in CLI; confirm integration into prep pipeline
- Group-level second-level GLM: not started

## Backlog

- **Per-step before/after comparison** — interactive channel-level signal comparison across preprocessing steps (OD → SCI → motion-corrected → haemo); needs design work before implementation
- **Epoch/HRF preview** — stimulus-locked epoch average per channel (HbO/HbR mean ± std across trials), task data only
- **Multi-run QC comparison** — aggregate and compare IQM scalars across runs within the same subject/session; BIDS `run-` entity already supported
- Functional connectivity mode for task data (currently raises `NotImplementedError`)
