"""Write a compact per-subject reproducible script to output_dir/logs/."""

from datetime import datetime
from pathlib import Path
from typing import Any

from fnirs_pipe.utils import unwrap_enum as _unwrap


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

    L: list[str] = []

    def w(*lines: str) -> None:
        L.extend(lines)

    def _q(v: Any) -> str:
        return f'"{v}"' if isinstance(v, str) else repr(v)

    # --- header / imports ---
    w(
        '#!/usr/bin/env python3',
        f'"""fnirs-pipe script for sub-{subject} — generated {dt_str}.',
        '',
        'Reproduce:',
        f'    python sub-{subject}_{timestamp}_script.py',
        '"""',
        'from pathlib import Path',
        '',
        'import mne',
        'from fnirs_pipe.io.bids import get_layout, get_nirs_files',
        'from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep',
    )
    if mode:
        w('from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post')
    if contrast_file:
        w(
            'try:',
            '    import tomllib',
            'except ImportError:',
            '    import tomli as tomllib  # type: ignore[no-redef]',
        )

    # --- paths ---
    w('', f'BIDS_DIR   = Path("{bids_dir}")', f'OUTPUT_DIR = Path("{out_dir}")')
    if work_dir:
        w(f'WORK_DIR   = Path("{work_dir}")')
    w('')

    # --- PrepConfig ---
    w('prep_config = PrepConfig(')
    w(f'    subject="{subject}",')
    w(f'    dpf={dpf!r},')
    w(f'    sci_threshold={sci_threshold},')
    w(f'    motion_correction="{motion_correction}",')
    if cardiac_l_freq is not None:
        w(f'    cardiac_l_freq={cardiac_l_freq},')
    if cardiac_h_freq is not None:
        w(f'    cardiac_h_freq={cardiac_h_freq},')
    if resp_l_freq is not None:
        w(f'    resp_l_freq={resp_l_freq},')
    if resp_h_freq is not None:
        w(f'    resp_h_freq={resp_h_freq},')
    w(f'    qc_window_s={qc_window_s},')
    if bad_channels:
        w(f'    bad_channels={bad_channels!r},')
    w(f'    ignore={ignore!r},')
    w(')', '')

    # --- contrast loading (before PostConfig so _contrast_def is defined) ---
    if mode and contrast_file:
        w(
            f'with open("{contrast_file}", "rb") as _fh:',
            '    _contrast_def = tomllib.load(_fh)',
            '',
        )

    # --- PostConfig ---
    if mode:
        w('post_config = PostConfig(')
        w(f'    subject="{subject}",')
        w(f'    cardiac_l_freq={cardiac_l_freq},')
        w(f'    cardiac_h_freq={cardiac_h_freq},')
        w(f'    resp_l_freq={resp_l_freq},')
        w(f'    resp_h_freq={resp_h_freq},')
        w(f'    high_pass={_q(high_pass)},')
        w(f'    low_pass={_q(low_pass)},')
        w(f'    resample_sfreq={_q(resample_sfreq)},')
        if stim_dur is not None:
            w(f'    stim_dur={stim_dur},')
        w(f'    hrf_model="{hrf_model}",')
        w(f'    noise_model="{noise_model}",')
        w(f'    drift_model="{drift_model}",')
        w(f'    drift_high_pass={drift_high_pass},')
        w(f'    drift_order={drift_order},')
        w(f'    fir_delays={fir_delays!r},')
        w(f'    short_channel={short_channel!r},')
        if events_path is not None:
            w(f'    events_path="{events_path}",')
        if contrast_file:
            w('    contrast_def=_contrast_def,')
        w(f'    combine_runs={combine_runs!r},')
        w(')', '')

    # --- main loop ---
    w('layout = get_layout(BIDS_DIR)', '')

    i1, i2, i3 = '    ', '        ', '            '
    w(
        f'for session in {sessions_repr}:',
        f'{i1}for task in {tasks_repr}:',
        f'{i2}prep_config.session = session',
        f'{i2}files = get_nirs_files(layout, subject="{subject}", session=session, task=task)',
        f'{i2}for snirf_path in files:',
        f'{i3}raw = mne.io.read_raw_snirf(str(snirf_path), preload=True)',
    )
    run_prep_call = (
        f'run_prep(raw, prep_config, output_dir=OUTPUT_DIR, work_dir=WORK_DIR)'
        if work_dir else
        f'run_prep(raw, prep_config, output_dir=OUTPUT_DIR)'
    )
    w(f'{i3}{run_prep_call}')

    if mode:
        w(
            '',
            f'{i2}from fnirs_pipe.io.bids import get_layout as _get_layout',
            f'{i2}post_layout = _get_layout(OUTPUT_DIR, validate=False)',
            f'{i2}preproc_files = [',
            f'{i2}    p for p in get_nirs_files(',
            f'{i2}        post_layout, subject="{subject}", session=session, task=task,',
            f'{i2}    )',
            f'{i2}    if "desc-preproc" in p.name',
            f'{i2}]',
            f'{i2}post_config.session = session',
            f'{i2}for snirf_path in preproc_files:',
            f'{i3}raw_haemo = mne.io.read_raw_snirf(str(snirf_path), preload=True)',
            f'{i3}run_post(raw_haemo, post_config, output_dir=OUTPUT_DIR, mode="{mode}")',
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
