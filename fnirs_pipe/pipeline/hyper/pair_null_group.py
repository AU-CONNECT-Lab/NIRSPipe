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

_TAG = {"repaired": "pairnull", "phase": "phasenull"}

REAL_SUFFIX = "_hyper-wtcbycond.tsv"
CELL_SUFFIX = {
    "repaired": ("_hyper-wtcbycond-pairnull.tsv", "_hyper-wtcbycond-roihom-pairnull.tsv"),
    "phase": ("_hyper-wtcbycond-phasenull.tsv", "_hyper-wtcbycond-roihom-phasenull.tsv"),
}
DRAWS_SUFFIX = {
    "repaired": "_hyper-wtcbycond-pairnull-draws.tsv",
    "phase": "_hyper-wtcbycond-phasenull-draws.tsv",
}


def _of_chroma(frame: pd.DataFrame, chroma: str) -> pd.DataFrame:
    """One chromophore's rows, blanks dropped. The value column is named per table kind."""
    out = frame[frame["chromophore"] == chroma] if "chromophore" in frame else frame
    value = next((c for c in ("coherence", "percentile", "null_mean") if c in out.columns), None)
    return out.dropna(subset=[value]) if value else out


def _homologous(frame: pd.DataFrame) -> pd.DataFrame:
    """The pairings a homologous null draws: both members' own channel.

    A table with no label2 is homologous by construction, and a tree part way through a
    crossed rerun has both kinds, so a missing label2 reads as the channel's own rather than
    dropping the row.
    """
    if "label2" not in frame.columns:
        return frame
    return frame[frame["label"] == frame["label2"].fillna(frame["label"])]


def _variants(draws: pd.DataFrame, real: pd.DataFrame, roi_map: "dict | None",
              min_channels: int = 2):
    """Every aggregate the draws support, as (granularity, level, pairings, draws, real).

    Emitted rather than selected, because which ones exist is a property of the draws and not
    a choice: all 196 pairings only if the null was drawn crossed, regions only if a mapping
    was given. A caller that has to remember a flag to get the level its reference
    implementation used will one day forget it.
    """
    hom_d, hom_r = _homologous(draws), _homologous(real)
    yield "whole", "whole", "homologous", hom_d, hom_r
    # a NaN label2 is a homologous table concatenated beside a crossed one, not a crossed
    # pairing: comparing against NaN is always unequal and would count it as crossed
    crossed = {o for o, part in draws.groupby("occasion")
               if "label2" in part.columns
               and (part["label"] != part["label2"].fillna(part["label"])).any()}
    # and the real table, which is the half that was missed: a mean over 14 homologous cells
    # ranked inside a null built from 196 pairings is not the same statistic on both sides.
    # The centre survives it, the spread does not, and the null of 196 comes out about a
    # fifth narrower, which moves a p by a factor of two to four in the permissive direction
    real_crossed = ("label2" in real.columns
                    and (real["label"] != real["label2"].fillna(real["label"])).any())
    if crossed and not real_crossed:
        logger.warning("no all-pairings level: the draws are crossed and the real table is "
                       "not, so that level would rank a mean over the diagonal inside a null "
                       "over every pairing. Rerun the real tables with --wtc-channel-cross")
    if crossed and real_crossed:
        # Mousley's and Zexin's whole-brain mean is over every pairing, not the diagonal.
        # All or none: 196 pairings for one occasion and 14 for the next is not one
        # statistic, and a part-crossed tree is what a rerun looks like half way through
        short = sorted(set(draws["occasion"].unique()) - crossed)
        if short:
            logger.warning("no all-pairings level: %d of %d occasions are homologous only "
                           "(%s). Rerun those with --wtc-pair-cross, or read the whole "
                           "homologous level, which every occasion supports",
                           len(short), draws["occasion"].nunique(), ", ".join(short))
        else:
            yield "whole", "whole", "all", draws, real
    for name, channels in (roi_map or {}).items():
        d = hom_d[hom_d["label"].isin(channels)]
        r = hom_r[hom_r["label"].isin(channels)]
        # a thinly covered region is not a region; Nguyen and Miller average the pairings
        if d["label"].nunique() >= min_channels and r["label"].nunique() >= min_channels:
            yield "roi", name, "homologous", d, r
    # one test per channel pairing, pooled over occasions, which is the granularity the
    # field reports at: 27 of 30 WTC studies take the channel pair as the unit and test it
    # across their sample. Ours is a permutation rather than a t test, the sample being
    # occasions rather than independent dyads, but the cell is the same cell
    for label in sorted(hom_d["label"].unique()):
        d = hom_d[hom_d["label"] == label]
        r = hom_r[hom_r["label"] == label]
        if not d.empty and not r.empty:
            yield "channel", label, "homologous", d, r


def _read_tree(output_dir: Path, suffix: str, task: str, chroma: str,
               needs: "tuple[str, ...]" = ()) -> pd.DataFrame:
    """Every occasion's table concatenated, refusing rather than dropping a malformed one.

    A table missing a column the others have concatenates to NaN and then leaves the groupby
    without a word, taking that occasion's contribution to every mean with it. One table
    written by an older version of the pipeline is enough, and nothing downstream shows it.
    """
    frames = []
    for path in sorted(Path(output_dir).rglob(f"group-*_task-{task}{suffix}")):
        found = re.search(r"group-([^_]+)_task-", path.name)
        frame = _of_chroma(pd.read_csv(path, sep="\t"), chroma)
        missing = [c for c in needs if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path.name} has no {', '.join(missing)} column, so its occasion would drop "
                f"out of every mean silently. It predates the column: rerun that dyad's null, "
                f"or rename the column if the values are current.")
        frames.append(frame.assign(occasion=found.group(1), source=str(path)))
    if not frames:
        raise FileNotFoundError(
            f"no group-*_task-{task}{suffix} under {output_dir}. The cohort levels are "
            f"built from the draws a null writes, so `fnirs-hyper pair-null` or a `run "
            f"--wtc-phase-null` has to have produced them.")
    return pd.concat(frames, ignore_index=True)


def _band_of(paths: "set[str]") -> set:
    """The band each table's sidecar records, as a set so a mismatch is visible."""
    import json

    bands = set()
    for tsv in paths:
        side = Path(tsv).with_suffix(".json")
        if not side.exists():
            continue
        params = json.loads(side.read_text()).get("parameters", {})
        if params.get("band_fmin") is not None:
            bands.add((params["band_fmin"], params["band_fmax"]))
    return bands


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
    draw_mean = draws.groupby(["condition", "occasion", "draw"]).coherence.mean()
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
    """The cohort mean against its null, by both reads, one row of each per condition.

    ``resample`` redraws one stand-in per occasion and ranks the real statistic inside that,
    so the null's width is the spread between an occasion's own draws. Resampling rather than
    pairing the pools by index, because a draw's position in one occasion's pool means
    nothing in another's.

    ``paired`` averages each occasion's draws into one baseline and tests real against it
    paired over occasions, so the width is the spread between occasions. This is the read the
    released implementations use, and on the phase null it is the only valid one: phase
    randomisation is applied one channel at a time, which flattens the surrogate's
    inter-channel covariance, and a mean over channels then has a null narrower than it
    should be. A re-paired stand-in is a real recording whose channels covary naturally, so
    there the two reads answer the same question and should agree.

    ``lift`` is the same number in both rows; what differs is what it is divided by.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cond, real_part in real.groupby("condition"):
        wide = (draws[draws.condition == cond]
                .groupby(["occasion", "draw"]).coherence.mean()
                .unstack("draw"))
        occasions = [o for o in real_part.occasion.unique() if o in wide.index]
        if not occasions:
            continue
        observed = real_part[real_part.occasion.isin(occasions)].groupby("occasion").coherence.mean()
        pools = {o: wide.loc[o].dropna().to_numpy(dtype=float) for o in occasions}
        pools = {o: p for o, p in pools.items() if p.size}
        if not pools:
            continue
        null = np.mean([p[rng.integers(0, p.size, n_resample)] for p in pools.values()], axis=0)
        value = float(observed.mean())
        # the real statistic counted into its own null, as at occasion level
        rows.append({
            "condition": cond, "test": "resample", "coherence": value,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)),
            "lift": value - float(null.mean()),
            "p": (int((null >= value).sum()) + 1) / (n_resample + 1),
            "n_occasions": len(pools),
            "n_iter_min": int(min(p.size for p in pools.values())),
            "n_resample": n_resample,
        })
        rows.append(_paired_row(cond, observed, pools))
    return (pd.DataFrame(rows)
            .sort_values(["condition", "test"], ignore_index=True))


def _paired_row(cond, observed: pd.Series, pools: "dict[str, np.ndarray]") -> dict:
    """Real against each occasion's own averaged draws, paired over occasions, one-tailed.

    ::

      3 occasions, real .35 .34 .36 against baselines .34 .34 .35
        -> lift +0.0067, t over the three differences

    Fewer than three occasions leaves a t with no spread to estimate, so the row carries the
    lift and no test rather than a number that would be read as one.
    """
    from scipy import stats

    baseline = pd.Series({o: float(p.mean()) for o, p in pools.items()})
    diff = (observed.reindex(baseline.index) - baseline).to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]
    row = {
        "condition": cond, "test": "paired",
        "coherence": float(observed.reindex(baseline.index).mean()),
        "null_mean": float(baseline.mean()),
        "lift": float(diff.mean()) if diff.size else np.nan,
        "lift_sd": float(diff.std(ddof=1)) if diff.size > 1 else np.nan,
        "n_occasions": int(diff.size),
        "n_positive": int((diff > 0).sum()),
    }
    if diff.size >= 3:
        t, p_two = stats.ttest_1samp(diff, 0.0)
        row["t"] = float(t)
        row["df"] = int(diff.size - 1)
        # one-tailed, the direction the released implementations test: real above its baseline
        row["p"] = float(p_two / 2 if t > 0 else 1.0 - p_two / 2)
    return row


def by_cell(output_dir: Path, task: str, chroma: str, null: str) -> "pd.DataFrame | None":
    """The per-cell percentiles a null already wrote, turned into corrected p values.

    ::

      percentile 95.45 off 22 draws  ->  p 0.087, because 21 of 22 beaten is rank 2 of 23

    The stage that writes a cell's percentile runs one dyad at a time and so cannot correct
    across cells; nothing else in the package does either, which left every per-cell table
    uncorrected and the correction living in whatever script last read them. One family per
    condition and level, so the family is a number a reader can state.
    """
    from statsmodels.stats.multitest import multipletests

    parts = []
    for suffix, level in zip(CELL_SUFFIX[null], ("channel", "roi")):
        try:
            frame = _read_tree(output_dir, suffix, task, chroma, needs=("percentile",))
        except FileNotFoundError:
            continue
        frame = frame.dropna(subset=["percentile"]).copy()
        if frame.empty:
            continue
        # the real value counted into its own null, as everywhere else here
        beaten = frame["percentile"] / 100 * frame["n_iter"]
        frame["p"] = (frame["n_iter"] - beaten + 1) / (frame["n_iter"] + 1)
        frame["q"] = np.nan
        for cond, part in frame.groupby("condition"):
            frame.loc[part.index, "q"] = multipletests(part["p"], method="fdr_bh")[1]
            frame.loc[part.index, "family"] = len(part)
        parts.append(frame.assign(level=level))
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    front = ["level", "condition", "occasion", "label"]
    keep = front + [c for c in ("label2", "coherence", "null_mean", "null_sd", "null_p95",
                                "percentile", "n_iter", "p", "q", "family")
                    if c in out.columns]
    return out[keep].sort_values(["level", "condition", "q"], ignore_index=True)


def write_group_null(output_dir: Path, task: str = "full", chroma: str = "hbo",
                     null: str = "repaired", roi_map: "dict | None" = None,
                     n_resample: int = 20000, seed: int | None = None) -> list[Path]:
    """Every level the draws support, written beside the merged tables.

    Two files, not one per level: the levels differ in two columns and are read against each
    other, so they belong in one table. `level` is ``whole`` or a region name, `pairings` is
    which channel pairings entered the mean.
    """
    from fnirs_pipe.pipeline.hyper.group_io import _hyper_sidecar

    output_dir = Path(output_dir)
    draws = _read_tree(output_dir, DRAWS_SUFFIX[null], task, chroma, needs=("draw", "label"))
    real = _read_tree(output_dir, REAL_SUFFIX, task, chroma, needs=("label",))
    sources = sorted(set(draws.source) | set(real.source))
    # A null averaged over one band and a real value over another measure different things,
    # and a tree part way through a re-band has both. The stages read the band off each
    # other's sidecars for this reason; nothing checked it once the tables were on disk
    bands = _band_of(set(draws.source)) | _band_of(set(real.source))
    if len(bands) > 1:
        raise ValueError(
            f"the draws and the real tables are not on one band: {sorted(bands)}. A null "
            f"averaged over one band cannot be subtracted from a value averaged over "
            f"another. Finish whichever rerun is in progress before reading this.")

    occ_parts, coh_parts = [], []
    for gran, level, pairings, d, r in _variants(draws, real, roi_map):
        if gran != "channel":
            logger.info("%s null, level %s over %s pairings: %d occasions, %d channels",
                        null, level, pairings, d.occasion.nunique(), d.label.nunique())
        tag = dict(granularity=gran, level=level, pairings=pairings)
        occ_parts.append(by_occasion(d, r).assign(**tag))
        coh_parts.append(by_cohort(d, r, n_resample=n_resample, seed=seed).assign(**tag))
    n_ch = sum(1 for p in coh_parts if (p["granularity"] == "channel").all())
    logger.info("%s null, and one level per channel pairing: %d of them", null, n_ch)

    def tidy(parts):
        out = pd.concat(parts, ignore_index=True)
        front = ["granularity", "level", "pairings", "condition"]
        return out[front + [c for c in out.columns if c not in front]]

    params = dict(null_kind=null, chroma=chroma, task=task,
                  n_resample=n_resample, seed=seed,
                  granularities=sorted(pd.concat(coh_parts).granularity.unique()),
                  statistic="mean over channel pairings, then over occasions",
                  cell_fdr_family="one condition and level, over occasions and pairings")
    cells = by_cell(output_dir, task, chroma, null)
    if cells is not None:
        passing = int((cells["q"] < 0.05).sum())
        logger.info("%s null, per cell: %d cells, %d at q<0.05", null, len(cells), passing)

    written = []
    for frame, stem, step in (
            ([] if cells is None else cells,
             f"group_hyper_wtc_bycondition_{_TAG[null]}_bycell",
             f"hyper_{null}_null_by_cell"),
            (tidy(occ_parts),
             f"group_hyper_wtc_bycondition_{_TAG[null]}_byoccasion",
             f"hyper_{null}_null_by_occasion"),
            (tidy(coh_parts),
             f"group_hyper_wtc_bycondition_{_TAG[null]}_cohort",
             f"hyper_{null}_null_cohort")):
        if len(frame) == 0:
            continue
        path = output_dir / f"{stem}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        _hyper_sidecar(path, step, sources, **params)
        logger.info("%s: %d rows", path.name, len(frame))
        written.append(path)
    return written
