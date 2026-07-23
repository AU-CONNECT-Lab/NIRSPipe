"""Write a per-subject step-by-step reproducible script to output_dir/logs/.

The script is an expanded transcript of the pipeline run: each block is one
processing stage that calls the same functions run_prep/run_post use, so reading
it means reading the whole flow. Auxiliary QC/SQM metrics the CLI computes are
noted in comments rather than re-run here.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fnirs_pipe.utils import unwrap_enum as _unwrap


def _lit(v: Any) -> str:
    """Python literal for a value, with forward-slashed strings for paths."""
    if isinstance(v, str):
        return repr(v.replace("\\", "/"))
    return repr(v)


def _build_script_text(
    subject: str,
    timestamp: str,
    bids_dir: str,
    out_dir: str,
    work_dir: str | None,
    session_label: list[str] | None,
    task_label: list[str] | None,
    dpf: list[float],
    sci_threshold: float,
    motion_correction: str,
    ignore: list[str],
    mode: str | None,
    cardiac_l_freq: float | None = None,
    cardiac_h_freq: float | None = None,
    resp_l_freq: float | None = None,
    resp_h_freq: float | None = None,
    qc_window_s: float = 10.0,
    bad_channels: list[str] | None = None,
    high_pass: float | None = None,
    low_pass: float | None = None,
    resample_sfreq: float | None = None,
    stim_dur: float | None = None,
    hrf_model: str = "spm",
    noise_model: str = "ar1",
    drift_model: str = "cosine",
    drift_high_pass: float = 0.01,
    drift_order: int = 1,
    fir_delays: tuple[int, ...] = (0,),
    short_channel: bool | str = False,
    events_path: str | None = None,
    contrast_file: str | None = None,
    combine_runs: bool = False,
) -> str:
    dt_str = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    sessions_repr = repr(session_label if session_label else [None])
    tasks_repr    = repr(task_label    if task_label    else [None])
    bad_channels  = bad_channels or []

    L: list[str] = []

    def w(*lines: str) -> None:
        L.extend(lines)

    # ---- header / imports ----
    w(
        '#!/usr/bin/env python3',
        f'"""fnirs-pipe processing script for sub-{subject}, generated {dt_str}.',
        '',
        'Step-by-step transcript of the pipeline: each block is one processing stage,',
        'calling the same functions the CLI uses. QC/SQM metrics the CLI computes are',
        'noted in comments, not re-run here.',
        '',
        'Reproduce:',
        f'    python sub-{subject}_{timestamp}_script.py',
        '"""',
        'from pathlib import Path',
        '',
        'import mne',
        '',
        'from fnirs_pipe import __version__',
        'from fnirs_pipe.io.bids import get_layout, get_nirs_files',
        'from fnirs_pipe.io.derivatives import build_output_path, carry_entities, write_sidecar_json',
        'from fnirs_pipe.io.snirf import write_snirf',
        'from fnirs_pipe.utils import is_optical_density',
        'from fnirs_pipe.pipeline.prep_pipeline import (',
        '    intensity_to_od, mark_bad_channels, correct_motion, od_to_haemo,',
        ')',
    )
    if mode:
        w('from fnirs_pipe.pipeline.denoise import bandpass_filter, resample')
    if mode in ("glm", "rest"):
        w('from fnirs_pipe.pipeline.glm import run_glm_pipeline')
    if mode == "rest":
        w('import pandas as pd')
    if mode and contrast_file:
        w(
            'try:',
            '    import tomllib',
            'except ImportError:',
            '    import tomli as tomllib  # type: ignore[no-redef]',
        )

    # ---- paths ----
    w('', '', f'SUBJECT    = "{subject}"', f'BIDS_DIR   = Path({_lit(bids_dir)})',
      f'OUTPUT_DIR = Path({_lit(out_dir)})')
    if work_dir:
        w(f'WORK_DIR   = Path({_lit(work_dir)})')

    # ---- prep parameters ----
    w(
        '',
        '# ---- prep parameters ----',
        f'DPF            = {dpf!r}',
        f'SCI_THRESHOLD  = {sci_threshold!r}',
        f'MOTION_METHOD  = {motion_correction!r}',
        f'CARDIAC_L_FREQ = {cardiac_l_freq!r}',
        f'CARDIAC_H_FREQ = {cardiac_h_freq!r}',
        f'RESP_L_FREQ    = {resp_l_freq!r}',
        f'RESP_H_FREQ    = {resp_h_freq!r}',
        f'BAD_CHANNELS   = {bad_channels!r}',
        f'IGNORE         = {ignore!r}',
    )

    # ---- post parameters ----
    if mode:
        w('', '# ---- post parameters ----', f'HIGH_PASS      = {high_pass!r}',
          f'LOW_PASS       = {low_pass!r}', f'RESAMPLE_SFREQ = {resample_sfreq!r}')
    if mode == "glm":
        w(
            f'STIM_DUR       = {stim_dur!r}',
            f'HRF_MODEL      = {hrf_model!r}',
            f'NOISE_MODEL    = {noise_model!r}',
            f'DRIFT_MODEL    = {drift_model!r}',
            f'DRIFT_HIGH_PASS= {drift_high_pass!r}',
            f'DRIFT_ORDER    = {drift_order!r}',
            f'FIR_DELAYS     = {fir_delays!r}',
            f'SHORT_CHANNEL  = {short_channel!r}',
            f'EVENTS_PATH    = {events_path!r}',
        )
    elif mode == "rest":
        w(
            f'DRIFT_MODEL    = {drift_model!r}',
            f'DRIFT_HIGH_PASS= {drift_high_pass!r}',
            f'DRIFT_ORDER    = {drift_order!r}',
            f'SHORT_CHANNEL  = {short_channel!r}',
        )

    # ---- sidecar params + save helper ----
    w(
        '',
        'PREP_PARAMS = {',
        '    "subject": SUBJECT, "session": None, "dpf": DPF,',
        '    "motion_correction": MOTION_METHOD, "sci_threshold": SCI_THRESHOLD,',
        '    "cardiac_l_freq": CARDIAC_L_FREQ, "cardiac_h_freq": CARDIAC_H_FREQ,',
        '    "bad_channels": BAD_CHANNELS, "ignore": IGNORE,',
        '}',
        '',
        '',
        'def save_step(raw_step, desc, step, session=None, extra=None):',
        '    path = build_output_path(',
        '        output_dir=OUTPUT_DIR, subject=SUBJECT,',
        '        entities={**carry_entities(None), "desc": desc},',
        '        suffix="nirs", extension=".snirf", session=session)',
        '    write_snirf(raw_step, path)',
        '    write_sidecar_json(path, {"pipeline_version": __version__, "step": step,',
        '        "parameters": {**PREP_PARAMS, "session": session}, **(extra or {})})',
        '    return path',
    )

    if mode and contrast_file:
        w(
            '',
            f'with open({_lit(contrast_file)}, "rb") as _fh:',
            '    _contrast_def = tomllib.load(_fh)',
        )

    # ---- main loop ----
    w('', '', 'layout = get_layout(BIDS_DIR)', '')
    i1, i2, i3 = '    ', '        ', '            '
    w(
        f'for session in {sessions_repr}:',
        f'{i1}for task in {tasks_repr}:',
        f'{i2}files = get_nirs_files(layout, subject=SUBJECT, session=session, task=task)',
        f'{i2}for snirf_path in files:',
        f'{i3}raw = mne.io.read_raw_snirf(str(snirf_path), preload=True)',
    )

    # ---- per-file processing body (built unindented, then indented into the loop) ----
    body: list[str] = []

    def b(*lines: str) -> None:
        body.extend(lines)

    b(
        '',
        '# ===== block: od | raw intensity -> optical density =====',
        'raw_od = raw.copy() if is_optical_density(raw) else intensity_to_od(raw)',
        'save_step(raw_od, "od", "od_conversion", session=session)',
        '',
        '# ===== block: sci | scalp coupling index channel pruning =====',
        '#   mark channels with SCI < SCI_THRESHOLD as bad (SCI in cardiac band)',
        'raw_od, bad_chs, sci_scores = mark_bad_channels(',
        '    raw_od, threshold=SCI_THRESHOLD,',
        '    cardiac_l_freq=CARDIAC_L_FREQ, cardiac_h_freq=CARDIAC_H_FREQ)',
        'if BAD_CHANNELS:  # merge manual --bad-channels',
        '    bad_chs = sorted(set(bad_chs) | set(BAD_CHANNELS))',
        '    raw_od.info["bads"] = bad_chs',
        'save_step(raw_od, "sci", "sci_pruning", session=session,',
        '          extra={"bad_channels": bad_chs})',
        '# QC (not run here): run_prep also computes raw SQM (compute_raw_sqm) and',
        '#   windowed SCI/PSP (compute_windowed_sci / compute_windowed_psp) here.',
        '',
        '# ===== block: motion | MOTION_METHOD artifact correction =====',
        'raw_od = correct_motion(raw_od, method=MOTION_METHOD)',
        'save_step(raw_od, "motcorrected", "motion_correction", session=session)',
        '',
        '# ===== block: beer_lambert | OD -> HbO/HbR (Beer-Lambert, dpf=DPF) =====',
        'raw_haemo = od_to_haemo(raw_od, dpf=DPF)',
        'save_step(raw_haemo, "preproc", "beer_lambert", session=session)',
        '# QC (not run here): compute_haemo_sqm baseline + motion-correction footprint.',
    )

    if mode:
        b('', '# ================= post =================', 'result = raw_haemo.copy()')

        if high_pass is not None or low_pass is not None:
            b(
                '',
                '# ----- block: bandpass | FIR bandpass filter (l=HIGH_PASS, h=LOW_PASS) -----',
                'result = bandpass_filter(result, l_freq=HIGH_PASS, h_freq=LOW_PASS)',
            )
            if mode in ("denoise", "glm"):
                b('save_step(result, "filtered", "bandpass", session=session)')

        if resample_sfreq is not None:
            b(
                '',
                '# ----- block: resample | downsample to RESAMPLE_SFREQ Hz -----',
                'result = resample(result, RESAMPLE_SFREQ)',
            )
            if mode in ("denoise", "glm"):
                b('save_step(result, "resampled", "resample", session=session)')

        if mode == "denoise":
            b('# QC (not run here): compute_haemo_sqm on the filtered/resampled signal.')

        if mode == "glm":
            b(
                '',
                '# ----- block: glm | GLM estimation (hrf=HRF_MODEL, noise=NOISE_MODEL) -----',
                '_, glm_est, design_matrix, raw_resid = run_glm_pipeline(',
                '    result,',
                '    stim_dur=STIM_DUR, hrf_model=HRF_MODEL, noise_model=NOISE_MODEL,',
                '    drift_model=DRIFT_MODEL, high_pass=DRIFT_HIGH_PASS, drift_order=DRIFT_ORDER,',
                '    fir_delays=FIR_DELAYS, short_channel=SHORT_CHANNEL, events_path=EVENTS_PATH,',
                ('    contrast_def=_contrast_def,' if contrast_file else '    contrast_def=None,'),
                '    output_dir=str(OUTPUT_DIR / f"sub-{SUBJECT}" / "nirs"),',
                ')',
                'save_step(raw_resid, "errts", "glm_residual", session=session)',
                '# QC (not run here): compute_glm_sqm (Durbin-Watson) on the residuals.',
            )

        if mode == "rest":
            b(
                '',
                '# ----- block: rest | confound regression + ALFF/FC derivatives -----',
                '_, glm_est, design_matrix, raw_resid = run_glm_pipeline(',
                '    result,',
                '    stim_dur=None, hrf_model="spm", noise_model="ols",',
                '    drift_model=DRIFT_MODEL, high_pass=DRIFT_HIGH_PASS, drift_order=DRIFT_ORDER,',
                '    fir_delays=None, short_channel=SHORT_CHANNEL,',
                '    events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),',
                '    output_dir=str(OUTPUT_DIR / f"sub-{SUBJECT}" / "nirs"),',
                ')',
                'save_step(raw_resid, "errts", "rest_residual", session=session)',
                '# Derivatives/QC (not run here): ALFF/fALFF, FC, ROI-FC and regression',
                '#   GCOR are written by run_post (_write_rest_derivatives / gcor_metrics).',
            )

    for line in body:
        w(f'{i3}{line}' if line else '')

    # ---- footer: generating command ----
    w(
        '', '',
        '# ' + '=' * 74,
        '# script generated by the command:',
        '#',
        '#   ' + " ".join(sys.argv).replace("\\", "/"),
    )

    return "\n".join(L) + "\n"


def write_run_script(
    args: dict[str, Any],
    subject: str,
    timestamp: str,
    output_dir: Path,
    sub_dir: Path | None = None,
) -> None:
    """Write a compact reproducible Python script for one subject to sub_dir/logs/."""
    sc   = _unwrap(args.get("short_channel"))
    mode = _unwrap(args.get("mode"))

    raw_fir = args.get("fir_delays")
    fir_delays: tuple[int, ...] = (
        tuple(int(x) for x in raw_fir.split(",")) if raw_fir else (0,)
    )

    raw_bad = args.get("bad_channels")
    bad_channels = [c.strip() for c in raw_bad.split(",")] if raw_bad else []

    def _fwd(p: Any) -> str | None:
        return str(p).replace("\\", "/") if p else None

    def _pick(key: str, default: Any) -> Any:
        v = args.get(key)
        return v if v is not None else default

    text = _build_script_text(
        subject=subject,
        timestamp=timestamp,
        bids_dir=str(args["bids_dir"]).replace("\\", "/"),
        out_dir=str(output_dir).replace("\\", "/"),
        work_dir=_fwd(args.get("work_dir")),
        session_label=args.get("session_label"),
        task_label=args.get("task_label"),
        dpf=args["dpf"],
        sci_threshold=args["sci_threshold"],
        motion_correction=_unwrap(args.get("motion_correction"), "tddr"),
        ignore=[_unwrap(ig) for ig in (args.get("ignore") or [])],
        mode=mode,
        cardiac_l_freq=args.get("cardiac_l_freq"),
        cardiac_h_freq=args.get("cardiac_h_freq"),
        resp_l_freq=args.get("resp_l_freq"),
        resp_h_freq=args.get("resp_h_freq"),
        qc_window_s=args.get("window_length", 10.0),
        bad_channels=bad_channels or None,
        high_pass=args.get("high_pass"),
        low_pass=args.get("low_pass"),
        resample_sfreq=args.get("resample_sfreq"),
        stim_dur=args.get("stim_dur"),
        hrf_model=_unwrap(args.get("hrf_model"), "spm"),
        noise_model=_unwrap(args.get("noise_model"), "ar1"),
        drift_model=_unwrap(args.get("drift_model"), "cosine"),
        drift_high_pass=_pick("drift_high_pass", 0.01),
        drift_order=_pick("drift_order", 1),
        fir_delays=fir_delays,
        short_channel=False if (sc is None or sc == "none") else sc,
        events_path=_fwd(args.get("events_path")),
        contrast_file=_fwd(args.get("contrast_file")),
        combine_runs=args.get("combine_runs", False),
    )

    base = sub_dir if sub_dir is not None else output_dir
    out  = base / "logs" / f"sub-{subject}_{timestamp}_script.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
