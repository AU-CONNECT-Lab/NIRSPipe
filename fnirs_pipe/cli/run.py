"""fnirs-pipe CLI entry point (argparse, BIDS App convention)."""

import argparse
import sys
from pathlib import Path

from fnirs_pipe import __version__

_MOTION_CHOICES        = ["tddr", "wavelet", "spline", "none"]
_MODE_CHOICES          = ["denoise", "glm", "rest"]
_HRF_CHOICES           = [
    "spm", "spm + derivative", "spm + derivative + dispersion",
    "glover", "glover + derivative", "glover + derivative + dispersion", "fir",
]
_NOISE_CHOICES         = ["ols", "ar1", "ar2", "ar3", "ar4", "ar5"]
_DRIFT_CHOICES         = ["cosine", "polynomial", "none"]
_SHORT_CHANNEL_CHOICES = ["none", "mean", "pca"]
_IGNORE_CHOICES        = ["events", "bids-validation"]
_LEVEL_CHOICES         = ["participant", "group", "hyper", "wtc-band"]
_BADS_SCOPE_CHOICES    = ["run", "subject"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-pipe",
        description="BIDS-compatible fNIRS preprocessing, postprocessing and dyad analysis. "
                    "At participant level prep always runs; add --mode to run postprocessing. "
                    "The hyper level computes the inter-brain metrics over the derivatives.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-pipe {__version__}")

    p.add_argument("bids_dir",   type=Path, help="BIDS dataset directory.")
    p.add_argument("output_dir", type=Path, help="Output directory (BIDS Derivatives).")
    p.add_argument("analysis_level", choices=_LEVEL_CHOICES,
                   help="Processing level. participant and group take BIDS_DIR; hyper and "
                        "wtc-band read the derivatives tree only and accept BIDS_DIR "
                        "without using it.")

    prep = p.add_argument_group("preprocessing (required for participant level)")
    # extend: accepts space-separated (--dpf 6 6) and repeated (--dpf 6 --dpf 6) forms.
    prep.add_argument("--dpf", nargs="+", type=float, action="extend",
                      help="Differential pathlength factor. One value or one per wavelength.")
    prep.add_argument("--sci-threshold", type=float,
                      help="SCI threshold for bad channel detection, e.g. 0.8. At the hyper "
                           "level it detects nothing and only colours the per-subject quality "
                           "table, where it is optional and defaults to 0.8; pass what the run "
                           "was prepped with.")

    sel = p.add_argument_group("subject / session / task selection")
    sel.add_argument("--participant-label", nargs="+", action="extend", help="Subject ID(s) to process.")
    sel.add_argument("--session-label",     nargs="+", action="extend", help="Session label(s) to process.")
    sel.add_argument("--task-label",        nargs="+", action="extend", help="Task label(s) to process.")
    sel.add_argument("--bids-filter-file",  type=Path, help="JSON file with extra pybids query filters.")

    prep_opt = p.add_argument_group("preprocessing (optional)")
    prep_opt.add_argument("--motion-correction", choices=_MOTION_CHOICES, default="tddr",
                          help="Motion correction method.")
    prep_opt.add_argument("--bad-channels",
                          help="Comma-separated source-detector labels to mark as bad, e.g. 'S1_D1,S2_D3'. "
                               "Kept in the data, unioned with SCI-detected bad channels.")
    prep_opt.add_argument("--cardiac-l-freq", type=float,
                          help="Lower bound of cardiac band in Hz (required at participant level; "
                               "population-dependent). Adult resting ~0.7; children/infants higher "
                               "(e.g. 1.0-1.7).")
    prep_opt.add_argument("--cardiac-h-freq", type=float,
                          help="Upper bound of cardiac band in Hz (required at participant level; "
                               "population-dependent). Adult resting ~1.5; children/infants higher "
                               "(e.g. 2.5-3.0).")
    prep_opt.add_argument("--resp-l-freq", type=float,
                          help="Lower bound of respiration band in Hz (required at participant level; "
                               "population-dependent). Adult ~0.1; infants higher.")
    prep_opt.add_argument("--resp-h-freq", type=float,
                          help="Upper bound of respiration band in Hz (required at participant level; "
                               "population-dependent). Adult ~0.5; infants higher.")
    prep_opt.add_argument("--window-length", type=float, default=10.0,
                          help="Sliding-window length (s) for windowed SCI/PSP/GVTD QC series.")

    post = p.add_argument_group("postprocessing (requires --mode)")
    post.add_argument("--mode", choices=_MODE_CHOICES,
                      help="Postprocessing mode: denoise (bandpass, plus confound regression "
                           "when --short-channel or --drift-model is given), glm or rest.")
    post.add_argument("--config", type=Path,
                      help="TOML file providing post parameter values. CLI flags override TOML.")
    post.add_argument("--high-pass", type=float, help="High-pass filter cutoff in Hz, e.g. 0.01.")
    post.add_argument("--low-pass",  type=float, help="Low-pass filter cutoff in Hz, e.g. 0.5.")
    post.add_argument("--resample-sfreq", type=float,
                      help="Target sampling rate in Hz after filtering, e.g. 2.0.")
    post.add_argument("--combine-runs", action=argparse.BooleanOptionalAction, default=False,
                      help="Concatenate multiple runs before postprocessing.")

    glm = p.add_argument_group("postprocessing: GLM and confound regression")
    glm.add_argument("--stim-dur", type=float,
                     help="Stimulus duration (s) for annotation-based events. Mutually exclusive with --events-path.")
    glm.add_argument("--hrf-model",   choices=_HRF_CHOICES,
                     help="HRF basis. 'spm + derivative' adds temporal derivative column.")
    glm.add_argument("--noise-model", choices=_NOISE_CHOICES, help="Residual autocorrelation model.")
    glm.add_argument("--drift-model", choices=_DRIFT_CHOICES,
                     help="Low-frequency drift regressors in design matrix. Required by "
                          "--mode glm and rest; optional in denoise, where the bandpass detrends.")
    glm.add_argument("--drift-high-pass", type=float, help="High-pass cutoff for cosine drift in Hz.")
    glm.add_argument("--drift-order", type=int, default=1,
                     help="Polynomial drift order (polynomial drift model only).")
    glm.add_argument("--fir-delays",
                     help="FIR delay bins in scans, comma-separated, e.g. '0,1,2,3,4,5' (only used when --hrf-model fir).")
    glm.add_argument("--short-channel", choices=_SHORT_CHANNEL_CHOICES,
                     help="Short-channel confound regressor strategy. Honoured by every mode: "
                          "glm fits it alongside the task, denoise and rest on its own.")
    glm.add_argument("--aux-regressors", action="store_true", default=None,
                     help="Add the recording's auxiliary channels (accelerometers, "
                          "gyroscopes, pulse and whatever else the device wrote to the "
                          "snirf aux group) to the confound regression, in every mode that "
                          "regresses. Preprocessing extracts them to desc-aux_timeseries.tsv.gz; "
                          "this reads that table, resamples it onto the data's time axis "
                          "with an anti-alias filter, and band-limits it to --high-pass / "
                          "--low-pass so regressors and data sit in one frequency band.")
    glm.add_argument("--aux-channels", nargs="+", action="extend",
                     help="Which aux channels to use, by the name the recording gives them. "
                          "Default is all of them. Name them when the aux group holds "
                          "channels that are not confounds, such as an event or trigger line.")
    glm.add_argument("--fc", action="store_true", default=None,
                     help="Also write the connectivity products rest mode writes, from "
                          "whatever the mode produced: glm correlates the task residual, so "
                          "what correlates is what the model did not explain; denoise "
                          "correlates its confound residual, or the bandpassed data itself "
                          "when no regression was asked for. Ignored by --mode rest, which "
                          "writes them anyway.")
    glm.add_argument("--events-path", type=Path,
                     help="Path to *_events.tsv. If omitted, extracted from snirf annotations.")
    glm.add_argument("--contrast-file", type=Path, help="TOML file defining GLM contrasts.")

    hyp = p.add_argument_group("hyperscanning analysis (hyper level)")
    hyp.add_argument("--pairs-csv", type=Path,
                     help="CSV with columns: group_id, subject_id, task. Each unique "
                          "(group_id, task) pair is processed as one session. Required by "
                          "the hyper level.")
    hyp.add_argument("--group-id", default=None,
                     help="Process only this group_id. Omit to process all groups.")
    hyp.add_argument("--desc", default="preproc",
                     help="desc entity of the per-subject stage the inter-brain metrics read, "
                          "e.g. 'preproc' (Beer-Lambert output) or 'errts' (confound-regression "
                          "residual, which is what short-channel regression leaves behind). "
                          "Must be a haemoglobin stage, not an optical-density one.")
    hyp.add_argument("--wtc-fmin", type=float, default=0.004, help="Lower bound (Hz) for WTC frequency axis.")
    hyp.add_argument("--wtc-fmax", type=float, default=0.20,  help="Upper bound (Hz) for WTC frequency axis.")
    hyp.add_argument("--wtc-band-fmin", type=float, default=None,
                     help="Lower bound (Hz) of the band the per-channel WTC TSV averages over. "
                          "Defaults to --wtc-fmin, i.e. the whole computed axis.")
    hyp.add_argument("--wtc-band-fmax", type=float, default=None,
                     help="Upper bound (Hz) of that band. Defaults to --wtc-fmax.")
    hyp.add_argument("--wtc-significance", action="store_true",
                     help="Overlay a Monte Carlo significance contour on WTC "
                          "(slow: see --wtc-mc-count for how slow). This asks whether a "
                          "time-frequency cell beats red noise, which is a different question "
                          "from the pseudo-dyad null of --wtc-pseudo.")
    hyp.add_argument("--wtc-mc-count", type=int, default=300,
                     help="Surrogate series behind each --wtc-significance contour "
                          "(default 300). This is what the runtime is spent on and it "
                          "scales linearly; lower it to preview a run, raise it to settle "
                          "a contour. Ignored without --wtc-significance.")
    hyp.add_argument("--wtc-seed", type=int, default=None,
                     help="Seed the Monte Carlo surrogates behind --wtc-significance and the "
                          "phase randomisation behind --wtc-pseudo. Also bypasses pycwt's "
                          "on-disk cache, which is not keyed on the seed.")
    hyp.add_argument("--wtc-mask-coi", action="store_true",
                     help="Average each band mean only over cells inside the cone of "
                          "influence. Off by default, which is what the field does; the share "
                          "inside the cone is reported as n_valid_frac either way. Masking "
                          "discards more of a short segment than of a long one, so it moves "
                          "conditions of different length by different amounts.")
    hyp.add_argument("--wtc-roi-min-channels", type=int, default=2, metavar="N",
                     help="Drop an ROI cell resting on fewer than N channel pairs, so one "
                          "surviving optode does not stand in for a region (default 2).")
    hyp.add_argument("--wtc-channel-cross", action="store_true",
                     help="Cross every long channel with every other across the two brains "
                          "instead of pairing each channel with its counterpart, so n "
                          "channels give n^2 coherence values rather than n. The extra "
                          "pairs reach the channel TSV with a label2 column; the "
                          "time-frequency heatmaps stay on the homologous pairs. Single "
                          "channels are noisier than ROI averages, so treat the off-diagonal "
                          "as exploratory and correct for the number of tests. Does not "
                          "affect the null: see --wtc-pseudo-cross.")
    hyp.add_argument("--bads-scope", choices=_BADS_SCOPE_CHOICES, default="run",
                     help="Which rejected channels are excluded from the inter-brain "
                          "metrics. 'run' (default) uses this task's own rejections. "
                          "'subject' unions them over every run of the subject, so all "
                          "conditions rest on the same channel set, which is what comparing "
                          "conditions needs; the cost is losing a channel everywhere because "
                          "one segment was bad.")
    hyp.add_argument("--wtc-limit-scales", action=argparse.BooleanOptionalAction, default=True,
                     help="Compute only the wavelet scales inside --wtc-fmin/--wtc-fmax "
                          "plus margin, instead of every scale the record length allows "
                          "(default on). A narrow band over a long record leaves most of the "
                          "default scale range unused, and the saving is proportional. The "
                          "kept scales land on pycwt's own grid and the margin is wider than "
                          "its scale-smoothing window, so the coherences match the "
                          "unrestricted ones bit for bit. --no-wtc-limit-scales restores the "
                          "old behaviour.")
    hyp.add_argument("--wtc-save-maps", action="store_true",
                     help="Save the full time-frequency coherence maps beside each TSV as "
                          "npz, so a different band can be averaged later with the wtc-band "
                          "level instead of a second wavelet transform. Large: one array "
                          "per pair per dyad per task.")
    hyp.add_argument("--wtc-pseudo", type=int, default=None, metavar="N",
                     help="Also write the pseudo-dyad null: the same band means against a "
                          "phase-scrambled partner, averaged over N iterations (100 is what "
                          "published work uses). Coherence between two unrelated recordings "
                          "is not zero, so this is what a real value is read against. Omit "
                          "it and no null is computed: each iteration costs a full WTC run, "
                          "so this is the expensive half of a hyper run.")
    hyp.add_argument("--wtc-pseudo-cross", action="store_true",
                     help="Cross the channels for the null too. Deliberately separate from "
                          "--wtc-channel-cross: crossing squares the pair count, and the null "
                          "pays that on every iteration. The homologous null is still the "
                          "null for the homologous cells of a crossed real table, which are "
                          "the rows where label and label2 agree.")
    hyp.add_argument("--isc-threshold", type=float, default=0.3,
                     help="Minimum mean ISC to draw an arc in the connectivity circle.")
    hyp.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                     help="Z-score each channel per subject after alignment.")
    hyp.add_argument("--no-align", action="store_true",
                     help="Skip trigger-based alignment; trim all recordings to the shortest duration.")
    hyp.add_argument("--tstart", type=float, default=None,
                     help="Keep only from this time (s) on the aligned clock, where 0 is the "
                          "shared trigger. Omit to start at the alignment point.")
    hyp.add_argument("--tend", type=float, default=None,
                     help="Keep only up to this time (s) on the aligned clock. Omit to run to "
                          "the end; a value past the end is clipped. The window narrows the "
                          "synchrony metrics only: the per-subject quality record describes "
                          "the whole recording either way.")

    band = p.add_argument_group("wtc-band level")
    band.add_argument("--band-fmin", type=float, default=None,
                      help="Lower bound (Hz) of the new band. Required by the wtc-band level.")
    band.add_argument("--band-fmax", type=float, default=None,
                      help="Upper bound (Hz) of the new band. Required by the wtc-band level.")
    band.add_argument("--mask-coi", action="store_true",
                      help="Average only over cells inside the cone of influence, matching "
                           "--wtc-mask-coi. Off by default.")
    band.add_argument("--suffix", default=None,
                      help="Name added to each output TSV. Defaults to the band, e.g. "
                           "'band0p05-0p2', so the new tables sit beside the originals "
                           "rather than replacing them.")

    out = p.add_argument_group("output")
    out.add_argument("--no-report", action="store_true", help="Skip the QC HTML report.")
    out.add_argument("--roi-mapping", type=Path, default=None,
                     help="JSON file mapping ROI labels to lists of channel names, for ROI grouping in the report denoising carpet. Optional.")
    out.add_argument("--n-jobs", type=int, default=1, help="Parallel subject jobs.")
    out.add_argument("--work-dir", type=Path, help="Hash cache directory.")

    esc = p.add_argument_group("escape hatches")
    esc.add_argument("--ignore", nargs="+", action="extend", choices=_IGNORE_CHOICES,
                     help="Processing aspects to skip.")
    esc.add_argument("--skip-bids-validation", action="store_true", help="Skip BIDS validation.")
    esc.add_argument("--dry-run", action="store_true", help="Write pipeline script without executing.")
    esc.add_argument("--verbose", action="store_true")
    return p


def _require(args: argparse.Namespace, *flags: str) -> None:
    """Exit non-zero naming the first flag the chosen level needs and did not get."""
    for flag in flags:
        if getattr(args, flag.lstrip("-").replace("-", "_")) is None:
            print(f"Error: Missing option '{flag}'.", file=sys.stderr)
            raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from fnirs_pipe.cli.workflows import (
        run_group_level,
        run_hyper_level,
        run_participant_level,
        run_wtc_band,
    )

    opts = vars(args)
    if args.analysis_level == "participant":
        _require(args, "--dpf", "--sci-threshold",
                 "--cardiac-l-freq", "--cardiac-h-freq", "--resp-l-freq", "--resp-h-freq")
        run_participant_level(opts)
    elif args.analysis_level == "hyper":
        _require(args, "--pairs-csv")
        run_hyper_level(opts)
    elif args.analysis_level == "wtc-band":
        _require(args, "--band-fmin", "--band-fmax")
        run_wtc_band(opts)
    else:
        run_group_level(opts)
