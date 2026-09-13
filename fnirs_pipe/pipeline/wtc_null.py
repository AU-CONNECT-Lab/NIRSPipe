"""The pseudo-dyad null, computed and written on its own.

Called from ``fnirs-hyper run`` when ``--wtc-pseudo`` is given, so it inherits that run's
stage, band and window by construction: a null averaged over a different band is not the
null for the table it sits beside. Crossing is the one thing it does not inherit. The null
used to take ``--wtc-channel-cross`` from the real run, so asking for the exploratory
196-pair channel table also multiplied the surrogate cost by 14; it is now the null's own
decision, ``--wtc-pseudo-cross``, and defaults to off.

The chromophores are inherited, unlike crossing: a null missing one leaves that half of the
real table with nothing to be tested against, which is not a saving worth offering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def write_wtc_null(
    group_id: str,
    task: str,
    aligned_raws: dict,
    output_dir: Path,
    n_iter: int = 100,
    wtc_fmin: float = 0.004,
    wtc_fmax: float = 0.20,
    band_fmin: float | None = None,
    band_fmax: float | None = None,
    seed: int | None = None,
    cross: bool = False,
    limit_scales: bool = True,
    mask_coi: bool = True,
    chroma: "tuple[str, ...] | list[str]" = ("hbo", "hbr"),
    sep_bands=None,
    windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
) -> Path:
    """Run the phase-scrambled null for one dyad and write its band means beside the real ones.

    Writes ``group-<id>_task-<task>_hyper-wtc-pseudo.tsv``, one table with a
    ``chromophore`` column, matching the real band-mean tables. The sidecar additionally
    records ``n_iter``, ``cross`` and ``chroma``, without which a 5-iteration probe and a
    100-iteration null are indistinguishable on disk.

    ``chroma`` has to cover the real run's chromophores: a null computed on HbO says nothing
    about an HbR coupling, so a table missing one chromophore leaves that half of the real
    table with nothing to be tested against.

    ``analysis_window`` is ``--tstart``/``--tend``. The whole-run table has to be windowed
    the same way the real whole-run table was, or the two describe different spans of the
    recording and the comparison between them is not one.

    ``windows`` adds a second table, ``...hyper-wtcbycond-pseudo.tsv``, with a ``condition``
    column: the null for what ``--wtc-by-condition`` wrote. It mirrors the real side, where
    the whole-run and per-condition tables are also two files merged separately. Both come
    out of one pass of ``n_iter`` transforms. Returns the whole-run path either way; the
    per-condition one sits beside it.
    """
    # imported in the call, not at module load: the wiring tests patch these on the module
    # that defines them, which only a lookup made at call time can see
    from fnirs_pipe.io.derivatives import group_data_dir
    from fnirs_pipe.pipeline.hyperscanning import (
        _hyper_sidecar, alignment_params, compute_wtc_pseudo,
    )
    from fnirs_pipe.pipeline.synchrony import wtc_grid_params
    from fnirs_pipe.utils.lineage import path_from

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax

    chroma = tuple(dict.fromkeys(chroma))
    if not chroma or set(chroma) - {"hbo", "hbr"}:
        raise ValueError(f"chroma must be some of ('hbo', 'hbr'), got {chroma!r}")

    logger.info(
        "Pseudo-dyad WTC: %d phase-scrambled iterations, %s pairs, one full WTC run each, "
        "per chromophore (%s).",
        n_iter, "crossed" if cross else "homologous", "+".join(chroma))
    frames, cond_frames = [], []
    for ch_type in chroma:
        part, cond_part = compute_wtc_pseudo(
            aligned_raws, band_fmin, band_fmax, n_iter=n_iter,
            fmin=wtc_fmin, fmax=wtc_fmax, seed=seed, cross=cross,
            limit_scales=limit_scales, mask_coi=mask_coi, ch_type=ch_type,
            sep_bands=sep_bands, windows=windows, analysis_window=analysis_window)
        # tagged after the averaging, which groups on the columns it knows and drops the
        # rest, and on a copy, since the frame is not ours to mutate
        part = part.copy()
        part.insert(0, "chromophore", ch_type)
        frames.append(part)
        if cond_part is not None:
            cond_part = cond_part.copy()
            cond_part.insert(0, "chromophore", ch_type)
            cond_frames.append(cond_part)

    sources = [p for p in (path_from(r) for r in aligned_raws.values()) if p]
    params = dict(
        band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=mask_coi,
        **({"analysis_window_s": [round(t, 3) for t in analysis_window]}
           if analysis_window is not None else {}),
        wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, n_iter=n_iter, cross=cross, seed=seed,
        chroma=list(chroma), **wtc_grid_params(aligned_raws),
        # the null is subtracted from the real table row by row, so the two have to say
        # they were built on the same clock for that subtraction to mean anything
        **alignment_params(aligned_raws),
    )
    data_dir = group_data_dir(output_dir, group_id)
    stem = f"group-{group_id}_task-{task}_hyper"

    out_path = data_dir / f"{stem}-wtc-pseudo.tsv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, sep="\t", index=False)
    _hyper_sidecar(out_path, "hyper_wtc_pseudo", sources, **params)
    logger.info("Pseudo-dyad WTC band means saved: %s", out_path)

    if cond_frames:
        cond_path = data_dir / f"{stem}-wtcbycond-pseudo.tsv"
        pd.concat(cond_frames, ignore_index=True).to_csv(cond_path, sep="\t", index=False)
        _hyper_sidecar(cond_path, "hyper_wtc_bycondition_pseudo", sources,
                       conditions=[w[0] for w in (windows or [])], **params)
        logger.info("Pseudo-dyad WTC per condition saved: %s", cond_path)

    return out_path
