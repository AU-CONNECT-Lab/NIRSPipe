"""Per command, what a generated form shows as it is filled: placeholders, inactive flags, suggestions, notes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from nirspipe.cli.run import _NOISE_CHOICES, mode_defaults
from nirspipe.pipeline.motion import IQR_FACTOR_DEFAULTS, STD_THRESH_DEFAULTS


@dataclass
class Hints:
    placeholders: "dict[str, str]" = field(default_factory=dict)
    # flags that do nothing with the form as it stands: hidden, and dropped before the argv is built
    inactive: "list[str]" = field(default_factory=list)
    suggestions: "dict[str, list[str]]" = field(default_factory=dict)
    # (text, tone), tone as the status line colours it: "warn" or ""
    notes: "list[tuple[str, str]]" = field(default_factory=list)

    def as_json(self) -> dict:
        return asdict(self)


_GLM_ONLY = ["--hrf-model", "--noise-model", "--stim-dur", "--events-path", "--contrast-file",
             "--fir-delays"]
_CONFOUNDS = ["--drift-model", "--drift-high-pass", "--drift-order", "--short-channel", "--fc"]
_CENSOR = ["--gvtd-censor-n-std", "--gvtd-min-epoch-s"]


def _float(text: "str | None") -> "float | None":
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


def band_note(mode: "str | None", high_pass: "str | None", drift_model: "str | None",
              drift_high_pass: "str | None") -> "str | None":
    """Whether the high-pass and the drift basis agree, with the mode's defaults filling empty fields."""
    if mode in (None, "none"):
        return None
    defaults = mode_defaults(mode)
    typed = (high_pass or "").strip().lower()
    if typed == "none":
        cutoff = None
    elif typed:
        cutoff = _float(typed)
        if cutoff is None:
            return None  # the command line's own check names a value that is not a number
    else:
        cutoff = defaults.get("high_pass")
    drift = drift_model or defaults.get("drift_model")
    drift_cutoff = _float(drift_high_pass)
    if drift_cutoff is None:
        drift_cutoff = defaults.get("drift_high_pass")

    # cosine without a cutoff stops the run, so say it while the form is filled
    if drift == "cosine" and drift_cutoff is None:
        return ("A cosine drift model needs a cutoff. Open the run's raw report in Data Preparation: "
                "its events section gives the slowest repeat of each condition and the number to put here.")
    if mode != "glm" or cutoff is None:
        return None
    if drift != "cosine":
        return (f"The data is high-passed at {cutoff} Hz but the task regressors are not, and a "
                f"{drift or 'none'} drift model does not span that band. Task betas will come out too "
                f"small. Use a cosine drift model at {cutoff} Hz, or leave the high-pass empty and let "
                "the drift basis do it.")
    if drift_cutoff < cutoff:
        return (f"Drift high-pass {drift_cutoff} Hz is below the {cutoff} Hz the data is filtered at, "
                f"so the drift basis does not cover what the filter removed. Set it to {cutoff} Hz or higher.")
    return (f"The high-pass and the cosine drift basis are both at {cutoff} Hz and doing the same job. "
            "The high-pass can be left empty.")


def analysis(form) -> Hints:
    """``nirspipe``: the mode's defaults as placeholders, and the flags the mode or method does not read."""
    mode = form.get("--mode") or None
    defaults = mode_defaults(mode)
    hints = Hints(suggestions={"--noise-model": list(_NOISE_CHOICES)})

    def default(dest: str, empty: str) -> str:
        return f"{defaults[dest]} ({mode} default)" if dest in defaults else empty

    for flag, dest in (("--high-pass", "high_pass"), ("--low-pass", "low_pass")):
        hints.placeholders[flag] = (f"{defaults[dest]} ({mode} default; none switches it off)"
                                    if dest in defaults else "off")
    hints.placeholders["--drift-high-pass"] = default("drift_high_pass", "from your design")
    hints.placeholders["--drift-model"] = default("drift_model", "none")
    hints.placeholders["--hrf-model"] = default("hrf_model", "none")
    hints.placeholders["--noise-model"] = default("noise_model", "auto")
    hints.placeholders["--short-channel"] = "auto (pca where the montage has short channels)"

    method = form.get("--motion-correction") or None
    std, iqr = STD_THRESH_DEFAULTS.get(method), IQR_FACTOR_DEFAULTS.get(method)
    if std is None:
        hints.inactive.append("--motion-std-thresh")
    else:
        hints.placeholders["--motion-std-thresh"] = f"{std:g} ({method} default)"
    if iqr is None:
        hints.inactive.append("--wavelet-iqr")
    else:
        hints.placeholders["--wavelet-iqr"] = f"{iqr:g} ({method} default)"

    censored = bool(form.get("--gvtd-censor"))
    if not censored:
        hints.inactive += _CENSOR
    # the fill reads every BAD_ span, the ones a table hands in as well as GVTD's
    if not censored and not form.get("--bad-spans"):
        hints.inactive.append("--censor-fill")
    if mode in (None, "none"):
        hints.inactive += _CONFOUNDS + ["--export-mean-filled"]
    if mode != "glm":
        hints.inactive += _GLM_ONLY
    elif form.get("--hrf-model") != "fir":
        hints.inactive.append("--fir-delays")
    # FC and ALFF are what it gates
    if mode != "rest" and not form.get("--fc"):
        hints.inactive.append("--min-time")
    # the drift model that will run, typed or the mode's, decides which companion it reads
    drift = form.get("--drift-model") or defaults.get("drift_model")
    if drift != "cosine":
        hints.inactive.append("--drift-high-pass")
    if drift != "polynomial":
        hints.inactive.append("--drift-order")

    note = band_note(mode, form.get("--high-pass"), form.get("--drift-model"),
                     form.get("--drift-high-pass"))
    if note:
        hints.notes.append((note, "warn"))
    hints.inactive = sorted(set(hints.inactive))
    return hints


RULES = {("nirspipe", None): analysis}


def apply(prog: str, sub: "str | None", form) -> "tuple[object, dict | None]":
    """The form less its inactive flags, which is what the command gets, and the hints as JSON."""
    rule = RULES.get((prog, sub))
    if rule is None:
        return form, None
    hints = rule(form)
    active = form.copy()
    for flag in hints.inactive:
        active.poplist(flag)
    return active, hints.as_json()
