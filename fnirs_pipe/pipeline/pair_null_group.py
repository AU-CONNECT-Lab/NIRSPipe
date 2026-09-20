"""The re-paired null read above the cell: one verdict per occasion, one per cohort.

``pair-null`` ranks each channel of each dyad inside its own draws, which asks where a
channel stands and answers it with whatever resolution the pool allows: a cell drawn against
22 stand-ins cannot reach a p below 1/23, so a per-cell test corrected over a thousand cells
is close to unable to reject whatever the data does. Averaging first spends that resolution
differently. It cannot say which channel, and in exchange it can say whether the pairing
beats its null at all.

Two levels, because they need different data and only one of them fits inside the per-dyad
stage:

``occasion``  the channels averaged, ranked in that occasion's own draws. One dyad's draws
              are enough, so ``pair-null`` could carry this; it is here instead so that both
              levels come off one table and one set of choices.
``cohort``    the channels averaged and then the occasions averaged, against a null that
              draws one stand-in per occasion. This needs every occasion's draws at once,
              which is why it cannot live in a stage that runs one dyad at a time.

Homologous pairings only when the draws carry a ``label2``: those are the ones the re-paired
null draws, and a crossed real table's other cells have no null behind them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.pair_null_group")

DRAWS_SUFFIX = "_hyper-wtcbycond-pairnull-draws.tsv"
REAL_SUFFIX = "_hyper-wtcbycond.tsv"


def _homologous(frame: pd.DataFrame, chroma: str) -> pd.DataFrame:
    out = frame[frame["chromophore"] == chroma] if "chromophore" in frame else frame
    if "label2" in out.columns:
        out = out[out["label"] == out["label2"]]
    return out.dropna(subset=["coherence"])


def _read_tree(output_dir: Path, suffix: str, task: str, chroma: str) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(output_dir).rglob(f"group-*_task-{task}{suffix}")):
        found = re.search(r"group-([^_]+)_task-", path.name)
        frame = _homologous(pd.read_csv(path, sep="\t"), chroma)
        frames.append(frame.assign(occasion=found.group(1), source=str(path)))
    if not frames:
        raise FileNotFoundError(
            f"no group-*_task-{task}{suffix} under {output_dir}. The cohort levels are "
            f"built from the draws `fnirs-hyper pair-null` writes, so that has to have run.")
    return pd.concat(frames, ignore_index=True)


def _exact_p(beaten: int, n: int) -> float:
    """The permutation p of a real value that beat ``beaten`` of ``n`` draws.

    ::

      beat all 22 of 22  ->  1/23 = 0.043, the smallest a pool of 22 can express

    The real value is counted into its own null, which is what keeps the test exact rather
    than letting a pool of 22 report a p of 0.
    """
    return (n - beaten + 1) / (n + 1)


def by_occasion(draws: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Each occasion's channel mean, ranked in that occasion's own draws."""
    rows = []
    real_mean = real.groupby(["condition", "occasion"]).coherence.mean()
    draw_mean = draws.groupby(["condition", "occasion", "stand_in"]).coherence.mean()
    for (cond, occ), value in real_mean.items():
        try:
            pool = draw_mean.loc[(cond, occ)].to_numpy(dtype=float)
        except KeyError:
            continue
        pool = pool[~np.isnan(pool)]
        if pool.size == 0:
            continue
        beaten = int((value > pool).sum())
        rows.append({
            "condition": cond, "occasion": occ, "coherence": value,
            "null_mean": float(pool.mean()), "null_sd": float(pool.std(ddof=1)),
            "null_p95": float(np.percentile(pool, 95)),
            "lift": value - float(pool.mean()),
            "percentile": beaten / pool.size * 100,
            "p": _exact_p(beaten, pool.size), "n_iter": int(pool.size),
        })
    return pd.DataFrame(rows).sort_values(["condition", "occasion"], ignore_index=True)


def by_cohort(draws: pd.DataFrame, real: pd.DataFrame,
              n_resample: int = 20000, seed: int | None = None) -> pd.DataFrame:
    """The cohort mean against a null that redraws one stand-in per occasion.

    The statistic is the mean over occasions of each occasion's channel mean, and a resample
    is the same statistic with every occasion's real partner replaced by one of its own
    stand-ins. Resampling rather than pairing the pools by index, because a draw's position
    in one occasion's pool means nothing in another's.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cond, real_part in real.groupby("condition"):
        wide = (draws[draws.condition == cond]
                .groupby(["occasion", "stand_in"]).coherence.mean()
                .unstack("stand_in"))
        occasions = [o for o in real_part.occasion.unique() if o in wide.index]
        if not occasions:
            continue
        observed = real_part[real_part.occasion.isin(occasions)].groupby("occasion").coherence.mean()
        pools = [wide.loc[o].dropna().to_numpy(dtype=float) for o in occasions]
        pools = [p for p in pools if p.size]
        null = np.array([np.mean([p[rng.integers(p.size)] for p in pools])
                         for _ in range(n_resample)])
        value = float(observed.mean())
        # the real statistic counted into its own null, as at occasion level
        rows.append({
            "condition": cond, "coherence": value,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)),
            "lift": value - float(null.mean()),
            "p": (int((null >= value).sum()) + 1) / (n_resample + 1),
            "n_occasions": len(pools),
            "n_iter_min": int(min(p.size for p in pools)),
            "n_resample": n_resample,
        })
    return pd.DataFrame(rows).sort_values("condition", ignore_index=True)


def write_group_null(output_dir: Path, task: str = "full", chroma: str = "hbo",
                     n_resample: int = 20000, seed: int | None = None) -> list[Path]:
    """Both levels, written beside the merged tables."""
    from fnirs_pipe.pipeline.group_io import _hyper_sidecar

    output_dir = Path(output_dir)
    draws = _read_tree(output_dir, DRAWS_SUFFIX, task, chroma)
    real = _read_tree(output_dir, REAL_SUFFIX, task, chroma)
    sources = sorted(set(draws.source) | set(real.source))
    logger.info("group-level re-paired null: %d occasions, %d channels, chroma %s",
                draws.occasion.nunique(), draws.label.nunique(), chroma)

    params = dict(null_kind="repaired", chroma=chroma, task=task,
                  n_resample=n_resample, seed=seed,
                  channels="homologous", statistic="mean over channels, then over occasions")
    written = []
    for frame, stem, step in (
            (by_occasion(draws, real),
             "group_hyper_wtc_bycondition_pairnull_byoccasion",
             "hyper_pair_null_by_occasion"),
            (by_cohort(draws, real, n_resample=n_resample, seed=seed),
             "group_hyper_wtc_bycondition_pairnull_cohort",
             "hyper_pair_null_cohort")):
        path = output_dir / f"{stem}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        _hyper_sidecar(path, step, sources, **params)
        logger.info("%s: %d rows", path.name, len(frame))
        written.append(path)
    return written
