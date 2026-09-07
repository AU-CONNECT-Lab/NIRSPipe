"""The pseudo-dyad null, computed and written on its own.

Called from ``fnirs-hyper run`` when ``--wtc-pseudo`` is given, so it inherits that run's
stage, band and window by construction: a null averaged over a different band is not the
null for the table it sits beside. Crossing is the one thing it does not inherit. The null
used to take ``--wtc-channel-cross`` from the real run, so asking for the exploratory
196-pair channel table also multiplied the surrogate cost by 14; it is now the null's own
decision, ``--wtc-pseudo-cross``, and defaults to off.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    mask_coi: bool = False,
) -> Path:
    """Run the phase-scrambled null for one dyad and write its band means beside the real ones.

    Writes ``group-<id>_task-<task>_hyper-wtc-pseudo.tsv``, the name and columns this table
    has always had, so anything already reading it is unaffected. The sidecar additionally records ``n_iter`` and ``cross``, without which a
    5-iteration probe and a 100-iteration null are indistinguishable on disk.
    """
    from fnirs_pipe.io.derivatives import group_data_dir
    from fnirs_pipe.pipeline.hyperscanning import _hyper_sidecar, compute_wtc_pseudo
    from fnirs_pipe.pipeline.synchrony import wtc_grid_params
    from fnirs_pipe.utils.lineage import path_from

    band_fmin = band_fmin if band_fmin is not None else wtc_fmin
    band_fmax = band_fmax if band_fmax is not None else wtc_fmax

    logger.info(
        "Pseudo-dyad WTC: %d phase-scrambled iterations, %s pairs, one full WTC run each.",
        n_iter, "crossed" if cross else "homologous")
    df = compute_wtc_pseudo(
        aligned_raws, band_fmin, band_fmax, n_iter=n_iter,
        fmin=wtc_fmin, fmax=wtc_fmax, seed=seed, cross=cross,
        limit_scales=limit_scales, mask_coi=mask_coi)

    out_path = (group_data_dir(output_dir, group_id)
                / f"group-{group_id}_task-{task}_hyper-wtc-pseudo.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    _hyper_sidecar(
        out_path, "hyper_wtc_pseudo",
        [p for p in (path_from(r) for r in aligned_raws.values()) if p],
        band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=mask_coi,
        wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, n_iter=n_iter, cross=cross, seed=seed,
        **wtc_grid_params(aligned_raws),
    )
    logger.info("Pseudo-dyad WTC band means saved: %s", out_path)
    return out_path
