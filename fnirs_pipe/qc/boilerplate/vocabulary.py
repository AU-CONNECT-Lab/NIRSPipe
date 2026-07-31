"""The bridge between the step names the pipeline records and the prose describing them.

Two vocabularies exist and they sit at different granularities on purpose, so this maps
rather than renames:

- the pipeline records one ``motion_correction`` step with the method as a parameter,
  while ``steps.toml`` needs one paragraph per method because the citations differ
- the pipeline records one ``bandpass`` step with two cutoffs, while the prose splits
  into bandpass / highpass / lowpass depending on which cutoff was given
- ``design_matrix``, ``glm_fit``, ``glm_residuals`` and ``contrasts`` are four files from
  one method described in a single paragraph

Steps with no paragraph are not omissions: reading a file or recording quality metrics is
bookkeeping, not method, and putting it in ``steps.toml`` would leak it into the Methods
section. Those get a plain one-liner here instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---- pipeline step -> steps.toml section ----

_DIRECT = ("od_conversion", "beer_lambert", "resample")


def boilerplate_key(step: str | None, params: dict[str, Any], mode: str | None = None) -> str | None:
    """Section of steps.toml describing this step, or None when it has no method prose."""
    if not step:
        return None
    if step in _DIRECT:
        return step
    if step == "sci_pruning":
        return "sci_marking"
    if step == "motion_correction":
        method = params.get("motion_correction")
        return f"motion_{method}" if method and method != "none" else None
    if step == "bandpass":
        low_edge, high_edge = params.get("high_pass"), params.get("low_pass")
        if low_edge and high_edge:
            return "bandpass"
        if low_edge:
            return "highpass"
        return "lowpass" if high_edge else None
    if step == "glm_fit":
        # rest mode runs this same code to regress confounds out; only a task run is a
        # first-level GLM, and nothing in the sidecar separates the two
        return "glm" if mode == "glm" else None
    return None


def template_slots(key: str, params: dict[str, Any]) -> dict[str, str]:
    """Fill a section's {slots} from a sidecar's parameters.

    Note the naming: a sidecar's ``high_pass`` is the *lower* edge of the band, which the
    filter templates call ``l_freq``.
    """
    if key == "sci_marking":
        return {
            "threshold": str(params.get("sci_threshold", "")),
            "action": "marked as bad and excluded from further analysis",
        }
    if key == "beer_lambert":
        dpf = params.get("dpf")
        return {"dpf": ", ".join(str(d) for d in dpf) if isinstance(dpf, (list, tuple)) else str(dpf)}
    if key == "bandpass":
        return {"l_freq": str(params.get("high_pass")), "h_freq": str(params.get("low_pass"))}
    if key == "highpass":
        return {"l_freq": str(params.get("high_pass"))}
    if key == "lowpass":
        return {"h_freq": str(params.get("low_pass"))}
    if key == "resample":
        return {"sfreq": str(params.get("sfreq") or params.get("resample_sfreq", ""))}
    if key == "glm":
        return {
            "hrf_model": str(params.get("hrf_model", "")),
            "noise_model": str(params.get("noise_model", "")),
            "drift_model": str(params.get("drift_model", "")),
            "drift_high_pass": str(params.get("drift_high_pass", "")),
        }
    return {}


# ---- steps with no method prose ----

STEP_SUMMARY = {
    "load": "Read from disk; the pipeline stage comes from the filename.",
    "od_passthrough": "Input was already optical density, so the conversion was skipped.",
    "motion_correction": "Motion correction, using the method named in the settings.",
    "sqm_raw": "Quality metrics measured on the original intensity recording.",
    "sqm": "Quality metrics measured on the haemoglobin signal at this point.",
    "design_matrix": "Regressors assembled for the fit: conditions, drift and confounds.",
    "glm_fit": "Per-channel model fit; rejected channels are flagged, not dropped.",
    "contrasts": "Contrast estimates derived from the fitted model.",
    "glm_residuals": "What the model left behind, once the fitted signal was removed.",
    "glm_residuals_broadband": "The same regression without the low-pass, so fALFF keeps a full spectrum.",
    "alff": "Amplitude of low-frequency fluctuation, per channel.",
    "fc": "Channel-by-channel correlation within one chromophore.",
    "fisher_z": "Fisher r-to-z of a correlation matrix, for group-level statistics.",
    "fc_roi": "Connectivity between ROI-averaged signals.",
    "group_sqm_raw": "Quality metrics pooled across the members of a dyad.",
    "group_sqm_raw_channels": "The same pooling, kept per channel.",
}


def step_summary(step: str | None) -> str:
    return STEP_SUMMARY.get(step or "", "")


# ---- what a run actually did ----

def steps_from_sidecars(nirs_dir: Path, mode: str | None = None) -> list[tuple[str, dict[str, str]]]:
    """(steps.toml key, filled slots) for every method step a run recorded, in run order.

    Ordered by depth in the provenance graph, so the sentences follow the data rather than
    the filenames. A step that ran more than once contributes one entry, with the later
    parameters filling anything the first was missing (the GLM's four outputs each carry
    part of the picture).
    """
    from fnirs_pipe.qc.provenance import scan

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for node in sorted(scan(nirs_dir).values(), key=lambda n: (n.depth, n.label)):
        key = boilerplate_key(node.step, node.params, mode)
        if key is None:
            continue
        if key not in merged:
            merged[key] = {}
            order.append(key)
        merged[key] = {**node.params, **merged[key]}

    return [(key, template_slots(key, merged[key])) for key in order]
