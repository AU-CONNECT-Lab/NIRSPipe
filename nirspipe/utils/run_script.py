"""Write a per-subject step-by-step reproducible script to output_dir/logs/.

The script is an expanded transcript of the pipeline run: each block is one
processing stage that calls the same functions run_prep/run_post use, so reading
it means reading the whole flow. Auxiliary QC/SQM metrics the CLI computes are
noted in comments rather than re-run here.
"""

from datetime import datetime

from nirspipe.pipeline.denoise import DEFAULT_FILTER_METHOD, DEFAULT_FILTER_ORDER
from pathlib import Path
from typing import Any

from nirspipe.utils import unwrap_enum as _unwrap
from nirspipe.utils.run_record import RUN_TIMESTAMP_FORMAT, _command_line, _fwd


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
    filter_method: str = DEFAULT_FILTER_METHOD,
    filter_order: int = DEFAULT_FILTER_ORDER,
    resample_sfreq: float | None = None,
    stim_dur: float | None = None,
    hrf_model: str = "spm",
    noise_model: str = "auto",
    drift_model: str = "none",
    drift_high_pass: float = 0.01,
    drift_order: int = 1,
    fir_delays: tuple[int, ...] = (0,),
    short_channel: bool | str = False,
    aux: bool = False,
    aux_channels: list[str] | None = None,
    fc: bool = False,
    events_path: str | None = None,
    contrast_file: str | None = None,
    combine_runs: bool = False,
    gvtd_censor: str | None = None,
    gvtd_censor_n_std: float = 10.0,
    gvtd_min_epoch_s: float = 30.0,
    sep_bands: tuple | None = None,
    bad_channels_table: str | None = None,
    psp_threshold: float | None = None,
    min_good_frac: float | None = None,
    screen_scope: str = "run",
    keep_spans_table: str | None = None,
) -> str:
    dt_str = datetime.strptime(timestamp, RUN_TIMESTAMP_FORMAT).strftime("%Y-%m-%d %H:%M:%S")
    # mirrors post_pipeline._has_confounds: denoise regresses only when asked to
    denoise_regress = mode == "denoise" and (
        bool(short_channel) or bool(aux) or drift_model not in (None, "none"))
    # the aux table sits beside the stage on disk; the script resolves it the way run_post does
    _aux_lines = ([
        '    aux_path=find_aux_table(preproc_path), aux_channels=AUX_CHANNELS,',
        '    data_band=(HIGH_PASS, LOW_PASS),',
        '    data_filter_method=FILTER_METHOD, data_filter_order=FILTER_ORDER,',
    ] if aux else [])
    # before desc-sci is written, as in run_prep, so every later stage inherits the spans
    _gvtd_lines = ([
        '',
        '# ===== block: gvtd | censor GVTD spikes as BAD_gvtd annotations =====',
        'censor_spans, censor_metrics = gvtd_censor_spans(',
        '    raw_od, n_std=GVTD_N_STD, min_epoch_s=GVTD_MIN_EPOCH,',
        '    sep_bands=SEP_BANDS, channel_set=GVTD_CENSOR)',
        'add_bad_spans(raw_od, censor_spans, "BAD_gvtd")',
    ] if gvtd_censor else [])
    sessions_repr = repr(session_label if session_label else [None])
    tasks_repr    = repr(task_label    if task_label    else [None])
    bad_channels  = bad_channels or []

    L: list[str] = []

    def w(*lines: str) -> None:
        L.extend(lines)

    # ---- header / imports ----
    w(
        '#!/usr/bin/env python3',
        f'"""nirspipe processing script for sub-{subject}, generated {dt_str}.',
        '',
        'Step-by-step transcript of the pipeline: each block is one processing stage,',
        'calling the same functions the CLI uses. QC/SQM metrics the CLI computes are',
        'noted in comments, not re-run here.',
        '',
        'Reproduce:',
        f'    python sub-{subject}_script.py',
        '"""',
        'from pathlib import Path',
        '',
        'import mne',
        '',
        'from nirspipe import __version__',
        'from nirspipe.io.auxiliary import aux_table_path, find_aux_table, write_aux_table',
        'from nirspipe.io.bids import get_layout, get_nirs_files',
        'from nirspipe.io.derivatives import build_output_path, carry_entities, data_state, write_sidecar_json',
        'from nirspipe.io.snirf import write_snirf',
        'from nirspipe.utils import is_optical_density',
        'from nirspipe.utils.spans import add_bad_spans, bad_spans, excluded_spans, excluded_time',
        'from nirspipe.pipeline.prep_pipeline import (',
        '    intensity_to_od, mark_bad_channels, correct_motion, od_to_haemo,',
        '    _expand_bad_pairs,',
        ')',
    )
    if gvtd_censor:
        w('from nirspipe.qc.metrics import gvtd_censor_spans')
    if bad_channels_table:
        w('from nirspipe.cli.workflows import _bad_channels_for')
    if keep_spans_table:
        w('from nirspipe.cli.workflows import _keep_spans_for',
          'from nirspipe.utils.spans import mark_unselected')
    if mode:
        w('from nirspipe.pipeline.denoise import bandpass_filter, resample')
    if mode in ("glm", "rest") or denoise_regress:
        w('from nirspipe.pipeline.glm import run_glm_pipeline')
    if mode == "rest" or denoise_regress:
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
        f'PSP_THRESHOLD  = {psp_threshold!r}',
        f'MIN_GOOD_FRAC  = {min_good_frac!r}',
        f'SCREEN_SCOPE   = {screen_scope!r}',
        f'MOTION_METHOD  = {motion_correction!r}',
        f'CARDIAC_L_FREQ = {cardiac_l_freq!r}',
        f'CARDIAC_H_FREQ = {cardiac_h_freq!r}',
        f'RESP_L_FREQ    = {resp_l_freq!r}',
        f'RESP_H_FREQ    = {resp_h_freq!r}',
        f'BAD_CHANNELS   = {bad_channels!r}',
        f'IGNORE         = {ignore!r}',
    )
    if bad_channels_table:
        w(f'BAD_CHANNELS_TABLE = Path({_lit(bad_channels_table)})')
    if keep_spans_table:
        w(f'KEEP_SPANS_TABLE = Path({_lit(keep_spans_table)})')
    if gvtd_censor:
        w(
            f'GVTD_CENSOR    = {gvtd_censor!r}',
            f'GVTD_N_STD     = {gvtd_censor_n_std!r}',
            f'GVTD_MIN_EPOCH = {gvtd_min_epoch_s!r}',
            f'SEP_BANDS      = {tuple(sep_bands) if sep_bands else None!r}',
        )

    # ---- post parameters ----
    if mode:
        w('', '# ---- post parameters ----', f'HIGH_PASS      = {high_pass!r}',
          f'LOW_PASS       = {low_pass!r}', f'FILTER_METHOD  = {filter_method!r}',
          f'FILTER_ORDER   = {filter_order!r}', f'RESAMPLE_SFREQ = {resample_sfreq!r}')
    # every mode that fits a regression honours --noise-model, so the constant cannot live
    # in the glm branch: the script would report a model the run did not use
    if mode in ("glm", "rest") or denoise_regress:
        w(f'NOISE_MODEL    = {noise_model!r}')
    if mode == "glm":
        w(
            f'STIM_DUR       = {stim_dur!r}',
            f'HRF_MODEL      = {hrf_model!r}',
            f'DRIFT_MODEL    = {drift_model!r}',
            f'DRIFT_HIGH_PASS= {drift_high_pass!r}',
            f'DRIFT_ORDER    = {drift_order!r}',
            f'FIR_DELAYS     = {fir_delays!r}',
            f'SHORT_CHANNEL  = {short_channel!r}',
            f'AUX_CHANNELS   = {aux_channels!r}',
            f'EVENTS_PATH    = {events_path!r}',
        )
    elif mode == "rest" or denoise_regress:
        w(
            f'DRIFT_MODEL    = {drift_model!r}',
            f'DRIFT_HIGH_PASS= {drift_high_pass!r}',
            f'DRIFT_ORDER    = {drift_order!r}',
            f'SHORT_CHANNEL  = {short_channel!r}',
            f'AUX_CHANNELS   = {aux_channels!r}',
        )

    # ---- sidecar params + save helper ----
    w(
        '',
        'PREP_PARAMS = {',
        '    "subject": SUBJECT, "session": None, "dpf": DPF,',
        '    "motion_correction": MOTION_METHOD, "sci_threshold": SCI_THRESHOLD,',
        '    "psp_threshold": PSP_THRESHOLD, "min_good_frac": MIN_GOOD_FRAC,',
        '    "screen_scope": SCREEN_SCOPE,',
        '    "cardiac_l_freq": CARDIAC_L_FREQ, "cardiac_h_freq": CARDIAC_H_FREQ,',
        '    "bad_channels": BAD_CHANNELS, "ignore": IGNORE, "keep_spans": [],',
        '}',
        '',
        '',
        'def save_step(raw_step, desc, step, src, extra=None):',
        '    # src: the entities of the recording, so each run keeps its task, run and session',
        '    path = build_output_path(',
        '        output_dir=OUTPUT_DIR, subject=SUBJECT,',
        '        entities={**carry_entities(src), "desc": desc},',
        '        suffix="nirs", extension=".snirf", session=src.get("session"))',
        '    write_snirf(raw_step, path)',
        '    write_sidecar_json(path, {"pipeline_version": __version__, "step": step,',
        '        "parameters": {**PREP_PARAMS, "session": src.get("session")},',
        '        "data": data_state(raw_step),',
        '        # read back by read_snirf: SNIRF itself cannot carry the marks',
        '        "bad_channels": list(raw_step.info["bads"]),',
        '        "excluded_spans": excluded_spans(raw_step, INPUT_SPANS), **(extra or {})})',
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
        # the file's own session: the loop variable is only the --session-label filter
        f'{i3}src = layout.parse_file_entities(str(snirf_path))',
        f'{i3}ses = src.get("session")',
        f'{i3}raw = mne.io.read_raw_snirf(str(snirf_path), preload=True)',
        f'{i3}INPUT_SPANS = bad_spans(raw)  # BAD_ spans the recording already carried',
    )
    if keep_spans_table:
        w(
            f'{i3}# the --keep-spans rows that match this recording; the rest is BAD_unselected',
            f'{i3}KEEP_SPANS = _keep_spans_for(str(KEEP_SPANS_TABLE), SUBJECT, src)',
            f'{i3}PREP_PARAMS["keep_spans"] = [list(span) for span in KEEP_SPANS]',
            f'{i3}raw = mark_unselected(raw, KEEP_SPANS)',
        )
    if bad_channels_table:
        w(
            f'{i3}# the --bad-channels rows that match this recording',
            f'{i3}BAD_CHANNELS = _bad_channels_for(str(BAD_CHANNELS_TABLE), SUBJECT, src)',
            f'{i3}PREP_PARAMS["bad_channels"] = BAD_CHANNELS',
        )

    # ---- per-file processing body (built unindented, then indented into the loop) ----
    body: list[str] = []

    def b(*lines: str) -> None:
        body.extend(lines)

    b(
        '',
        '# ===== block: od | raw intensity -> optical density =====',
        'raw_od = raw.copy() if is_optical_density(raw) else intensity_to_od(raw)',
        'save_step(raw_od, "od", "od_conversion", src)',
        '',
        '# ===== block: sci | channel screening =====',
        '#   mark a channel bad when too few windows are coupled: a window counts when its',
        '#   SCI clears SCI_THRESHOLD and its PSP clears its own line, both cardiac-band',
        '#   coupling measures, and the channel is kept when enough windows do.',
        'raw_od, bad_chs, sci_scores, good_frac_scores = mark_bad_channels(',
        '    raw_od, threshold=SCI_THRESHOLD, psp_threshold=PSP_THRESHOLD,',
        '    min_good_frac=MIN_GOOD_FRAC, screen_scope=SCREEN_SCOPE,',
        '    cardiac_l_freq=CARDIAC_L_FREQ, cardiac_h_freq=CARDIAC_H_FREQ)',
        'if BAD_CHANNELS:  # merge manual --bad-channels, both wavelengths of each pair',
        '    bad_chs = sorted(set(bad_chs) | set(_expand_bad_pairs(raw_od, BAD_CHANNELS)))',
        '    raw_od.info["bads"] = bad_chs',
        *_gvtd_lines,
        'save_step(raw_od, "sci", "sci_pruning", src,',
        ('          extra={"bad_channels": bad_chs, "gvtd_censor": censor_metrics,'
         if gvtd_censor else '          extra={"bad_channels": bad_chs,'),
        '                 "excluded_time": excluded_time(raw_od)})',
        '# QC (not run here): none of these steps measures anything. The quality record is',
        '#   assembled from the files they leave on disk, once both passes have finished',
        '#   (qc.sqm_record.build_sqm_records).',
        '',
        '# ===== block: motion | MOTION_METHOD artifact correction =====',
        'raw_od = correct_motion(raw_od, method=MOTION_METHOD)',
        'save_step(raw_od, "motcorrected", "motion_correction", src)',
        '',
        '# ===== block: beer_lambert | OD -> HbO/HbR (Beer-Lambert, dpf=DPF) =====',
        'raw_haemo = od_to_haemo(raw_od, dpf=DPF)',
        'preproc_path = save_step(raw_haemo, "preproc", "beer_lambert", src)',
        '# QC (not run here): the `preproc`, `motion` and `motion_post` sections are read',
        '#   back from desc-preproc, desc-sci and desc-motcorrected.',
        '',
        '# ===== block: aux | snirf aux group -> desc-aux table (only if the file has one) =====',
        '#   MNE never reads aux, so this is the only point at which it can be carried forward',
        'aux_path = aux_table_path(preproc_path)',
        '_aux = write_aux_table(Path(snirf_path), aux_path)',
        'if _aux is not None:',
        '    write_sidecar_json(aux_path, {',
        '        "pipeline_version": __version__, "step": "aux_extract",',
        '        "Sources": [Path(snirf_path).as_posix()],',
        '        "parameters": {**PREP_PARAMS, "session": ses}, **_aux[1]})',
    )

    if mode:
        b('', '# ================= post =================', 'result = raw_haemo.copy()')

        if high_pass is not None or low_pass is not None:
            b(
                '',
                '# ----- block: bandpass | bandpass filter (l=HIGH_PASS, h=LOW_PASS) -----',
                'result = bandpass_filter(result, l_freq=HIGH_PASS, h_freq=LOW_PASS,',
                '                         method=FILTER_METHOD, order=FILTER_ORDER)',
            )
            if mode in ("denoise", "glm"):
                b('save_step(result, "filtered", "bandpass", src)')

        if resample_sfreq is not None:
            b(
                '',
                '# ----- block: resample | downsample to RESAMPLE_SFREQ Hz -----',
                'result = resample(result, RESAMPLE_SFREQ)',
            )
            if mode in ("denoise", "glm"):
                b('save_step(result, "resampled", "resample", src)')

        if denoise_regress:
            b(
                '',
                '# ----- block: denoise | confound regression (no task model) -----',
                '_, glm_est, design_matrix, raw_resid = run_glm_pipeline(',
                '    result,',
                '    stim_dur=None, hrf_model="spm", noise_model=NOISE_MODEL,',
                '    drift_model=DRIFT_MODEL, high_pass=DRIFT_HIGH_PASS, drift_order=DRIFT_ORDER,',
                '    fir_delays=None, short_channel=SHORT_CHANNEL,',
                *_aux_lines,
                '    events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),',
                '    output_dir=str(OUTPUT_DIR / f"sub-{SUBJECT}" / (f"ses-{ses}" if ses else "") / "nirs"),',
                '    source_path=str(preproc_path),',
                ')',
                'save_step(raw_resid, "errts", "glm_residuals", src)',
            )

        if mode == "denoise":
            if fc:
                b('# Derivatives (not run here): FC, ROI-FC and the seed maps are written by'
                  ' run_post',
                  '#   (_write_fc_derivatives) from the residual above, or from `result` when no'
                  ' regression ran.')
            b('# QC (not run here): the record gets one section per haemo file this wrote,'
              ' named after it:',
              '#   `filtered`, `resampled`, `errts`. Each pairs with `preproc`, so the'
              ' bandpass and the',
              '#   regression are separable rather than collapsed into one "after".')

        if mode == "glm":
            b(
                '',
                '# ----- block: glm | GLM estimation (hrf=HRF_MODEL, noise=NOISE_MODEL) -----',
                '_, glm_est, design_matrix, raw_resid = run_glm_pipeline(',
                '    result,',
                '    stim_dur=STIM_DUR, hrf_model=HRF_MODEL, noise_model=NOISE_MODEL,',
                '    drift_model=DRIFT_MODEL, high_pass=DRIFT_HIGH_PASS, drift_order=DRIFT_ORDER,',
                '    fir_delays=FIR_DELAYS, short_channel=SHORT_CHANNEL, events_path=EVENTS_PATH,',
                *_aux_lines,
                ('    contrast_def=_contrast_def,' if contrast_file else '    contrast_def=None,'),
                '    output_dir=str(OUTPUT_DIR / f"sub-{SUBJECT}" / (f"ses-{ses}" if ses else "") / "nirs"),',
                '    source_path=str(preproc_path),',
                ')',
                'save_step(raw_resid, "errts", "glm_residuals", src)',
            )
            if fc:
                b('# Derivatives (not run here): FC, ROI-FC and the seed maps are written by'
                  ' run_post',
                  '#   (_write_fc_derivatives) from the task residual above.')

        if mode == "rest":
            b(
                '',
                '# ----- block: rest | confound regression + ALFF/FC derivatives -----',
                '_, glm_est, design_matrix, raw_resid = run_glm_pipeline(',
                '    result,',
                '    stim_dur=None, hrf_model="spm", noise_model=NOISE_MODEL,',
                '    drift_model=DRIFT_MODEL, high_pass=DRIFT_HIGH_PASS, drift_order=DRIFT_ORDER,',
                '    fir_delays=None, short_channel=SHORT_CHANNEL,',
                *_aux_lines,
                '    events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),',
                '    output_dir=str(OUTPUT_DIR / f"sub-{SUBJECT}" / (f"ses-{ses}" if ses else "") / "nirs"),',
                '    source_path=str(preproc_path),',
                ')',
                'save_step(raw_resid, "errts", "glm_residuals", src)',
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
        '#   ' + _command_line(),
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

    from nirspipe.cli._shared import resolved_separation_bands
    from nirspipe.cli.workflows import _bad_channels_for, _make_prep_config
    # resolved as the run resolves them, so a flag left off gets the same default
    screening = _make_prep_config(subject, None, args)
    bad_spec = args.get("bad_channels")
    bad_channels = _bad_channels_for(bad_spec, subject)
    bad_channels_table = (_fwd(Path(str(bad_spec)).resolve())
                          if bad_spec and Path(str(bad_spec)).exists() else None)
    keep_spec = args.get("keep_spans")
    keep_spans_table = _fwd(Path(str(keep_spec)).resolve()) if keep_spec else None

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
        filter_method=_pick("filter_method", DEFAULT_FILTER_METHOD),
        filter_order=_pick("filter_order", DEFAULT_FILTER_ORDER),
        resample_sfreq=args.get("resample_sfreq"),
        stim_dur=args.get("stim_dur"),
        hrf_model=_unwrap(args.get("hrf_model"), "spm"),
        noise_model=_unwrap(args.get("noise_model"), "auto"),
        drift_model=_unwrap(args.get("drift_model"), "none"),
        drift_high_pass=_pick("drift_high_pass", 0.01),
        drift_order=_pick("drift_order", 1),
        fir_delays=fir_delays,
        short_channel=False if (sc is None or sc == "none") else sc,
        aux=bool(args.get("aux_regressors")),
        aux_channels=args.get("aux_channels"),
        fc=bool(args.get("fc")),
        events_path=_fwd(args.get("events_path")),
        contrast_file=_fwd(args.get("contrast_file")),
        combine_runs=args.get("combine_runs", False),
        gvtd_censor=_unwrap(args.get("gvtd_censor")),
        gvtd_censor_n_std=_pick("gvtd_censor_n_std", 10.0),
        gvtd_min_epoch_s=_pick("gvtd_min_epoch_s", 30.0),
        sep_bands=resolved_separation_bands(args) if args.get("gvtd_censor") else None,
        bad_channels_table=bad_channels_table,
        keep_spans_table=keep_spans_table,
        psp_threshold=screening.psp_threshold,
        min_good_frac=screening.min_good_frac,
        screen_scope=screening.screen_scope,
    )

    base = sub_dir if sub_dir is not None else output_dir
    out  = base / "logs" / f"sub-{subject}_script.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
