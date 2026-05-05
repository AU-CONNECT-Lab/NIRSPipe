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

---

## Pending decisions

- Short-channel regression: `--short-channel {none,mean,pca}` flag wired in CLI; confirm integration into prep pipeline
- ICA: module stub exists; implementation deferred
- Group-level CLI (second-level GLM): not started
- Fix `PostConfig` missing `ica` field (`_write_denoised_snirf` bug)

## Backlog

- Functional connectivity mode (deferred; not in CLI)
- Group-level report (`run_group_level`)