"""The stages a ``nirspipe`` run goes through for the chosen mode, in the order it runs them."""

from __future__ import annotations


def pipeline_stages(form) -> "list[list[str]]":
    """Stages as levels left to right; a level of two runs both from the one before.

    ``form`` is the Analysis form (``get``/``getlist``, as Flask's ``request.form``)::

        --mode rest  ->  [..., ["Bandpass / Resample"], ["Confound Regression"], ["ALFF", "Connectivity"]]
    """
    # motion correction works on optical density, so it comes before Beer-Lambert
    levels = [["Load SNIRF"], ["Optical Density"], ["Channel Screening"], ["Motion Correction"],
              ["Beer-Lambert Law"]]
    mode = form.get("--mode") or None
    if mode is None:
        return levels
    # denoise regresses only when there is something to regress, as the post stage decides
    confounds = (form.get("--drift-model") in ("cosine", "polynomial")
                 or form.get("--short-channel") in ("mean", "pca"))
    levels.append(["Bandpass / Resample"])
    if mode == "glm":
        levels.append(["GLM"])
    elif mode == "rest" or confounds:
        levels.append(["Confound Regression"])
    # rest writes connectivity whatever the switch says
    last = (["ALFF"] if mode == "rest" else []) + (["Connectivity"] if mode == "rest" or form.get("--fc") else [])
    if last:
        levels.append(last)
    return levels
