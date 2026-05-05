# Roadmap

Status: `[ ]` not started · `[~]` in progress · `[x]` done · `[?]` needs review

---

## v0.1 — Preprocessing Core `[x]`

End-to-end single-subject preprocessing implemented: OD conversion, SCI marking, motion correction (TDDR), Beer-Lambert.  
Pipeline reorganized into `pipeline/prep_pipeline.py`; intermediate snirf written at each step.

## v0.2 — QC Preprocessing Report `[x]`

Full QC report system implemented: SCI/PSP heatmaps, GVTD timeseries, carpet plot, HbO-HbR correlation panel, PSD, brain views, IQM sidecar.  
Output correctness has not been formally validated; treat as demo quality pending review.

## v0.3 — Postprocessing `[~]`

Task GLM (`pipeline/glm.py`) and denoise mode implemented.  
Connectivity mode removed from CLI; deferred to a later milestone.  
Known bug: `_write_denoised_snirf` references `config.ica` which does not exist on `PostConfig` → `AttributeError` in denoise/glm mode.

## v0.4 — QC Postprocessing Visualization `[~]`

GLM design matrix figure and activation panel in place.  
Per-channel HRF panel and FC matrix heatmap are stubs.

## v0.5 — Raw QC CLI (`fnirs-qc`) `[ ]`

Separate CLI command for lightweight raw data quality inspection, independent of the prep pipeline.

- BIDS dir + subject label as input (consistent with `fnirs-pipe` conventions)
- Runs `compute_raw_iqm` only (no OD conversion, no Beer-Lambert)
- Outputs IQM TOML sidecar + lightweight HTML report (SCI/PSP channel summary, intensity figure)
- Use case: same-day acquisition QA before committing to full preprocessing

## v0.6 — Group-Level QC Report `[ ]`

Group-level QC aggregation across subjects.

- Aggregate IQM scalars from per-subject TOML sidecars into a group-level summary
- Visualizations: distribution plots per metric, outlier flagging across subjects
- Output: group-level HTML report + CSV/TSV table of all subject IQMs

---

## Pending decisions

- Short-channel regression: `--short-channel {none,mean,pca}` flag wired in CLI; confirm integration into prep pipeline
- ICA: module stub exists; implementation deferred
- Group-level CLI (second-level GLM): not started
- Fix `PostConfig` missing `ica` field (`_write_denoised_snirf` bug)

## Backlog

- Functional connectivity mode (deferred; not in CLI)
- Group-level report (`run_group_level`)
- **Per-step before/after comparison** — interactive channel-level signal comparison across preprocessing steps (OD → SCI → motcorrected → haemo); needs design work before implementation
- **Epoch/HRF preview** — stimulus-locked epoch average per channel (HbO/HbR mean ± std across trials), task data only; quick sanity check without full GLM
- **Multi-run QC comparison** — aggregate and compare IQM scalars across runs within the same subject/session; BIDS `run-` entity already supported