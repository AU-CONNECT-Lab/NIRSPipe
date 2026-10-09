"""The re-paired null read above the cell: one verdict per occasion, one per cohort.

``pair-null`` ranks each channel of each dyad inside its own draws, and a cell drawn against
n stand-ins cannot reach a p below 1/(n+1). Averaging first cannot say which channel, but it
can say whether the pairing beats its null at all.

Two levels, because they need different data and only one of them fits inside the per-dyad
stage:

``occasion``  the channels averaged, ranked in that occasion's own draws. One dyad's draws
              are enough; it is here so that both levels come off one table.
``cohort``    the channels averaged and then the occasions averaged, against a null that
              draws one stand-in per occasion. This needs every occasion's draws at once,
              which is why it cannot live in a stage that runs one dyad at a time.

A homologous null tests the homologous pairings only, since a crossed real table's other
cells have no null behind it. A crossed null also tests every pairing and, given a region
map, every ordered region pair.
"""

from __future__ import annotations

import re
from pathlib import Path
import json

import numpy as np
import pandas as pd

from fnirs_pipe.utils import ROI_MIN_CHANNELS, bare_roi_map
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.io.naming import derivative_path
from fnirs_pipe.io.tables import write_tsv

logger = get_logger("pipeline.pair_null_group")

_NULL = {"repaired": "pair", "phase": "phase"}


def _tail(**entities) -> str:
    """The part of a dyad table's name after its task, as a glob to find it by.

    ::

      _tail(condition="all", statistic="wtc") -> "_cond-all_stat-wtc_relmat.tsv"

    Built through the namer rather than spelled out, so the tables this module reads and
    the ones the null writers produce cannot be renamed apart. The ROI map's own name is a
    wildcard: this reads whichever definition the tree was written with.
    """
    name = derivative_path("", "relmat", ".tsv", group="X", task="T", **entities).name
    return name.split("_task-T", 1)[1]


_BY_COND = {"condition": "all", "statistic": "wtc"}
_ROI = {"segmentation": "*", "aggregation": "homologous"}
_ROI_CROSSED = {"segmentation": "*", "aggregation": "roi"}

REAL_SUFFIX = _tail(**_BY_COND)
# each cell table a null writes, under the level its cells are corrected as one family in
CELL_SUFFIX = {
    kind: {"channel": _tail(**_BY_COND, nulldist=null),
           "roi": _tail(**_ROI, **_BY_COND, nulldist=null),
           "roi_crossed": _tail(**_ROI_CROSSED, **_BY_COND, nulldist=null)}
    for kind, null in _NULL.items()
}
DRAWS_SUFFIX = {kind: _tail(**_BY_COND, nulldist=null, desc="draws")
                for kind, null in _NULL.items()}


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


def _thick(level: str, d: pd.DataFrame, r: pd.DataFrame, sides: "tuple[str, ...]",
           min_channels: int) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Both frames cut to the occasion-conditions whose real rows keep the region.

    ::

      right_tpj, one valid channel in G03's game  ->  G03/game dropped, every other kept

    The rule each dyad's own ROI table blanks a cell by, taken one occasion and condition at
    a time, so a region one dyad left blank is not ranked for that dyad here.
    """
    keys = ["occasion", "condition"]
    counts = r.groupby(keys)[list(sides)].nunique()
    kept = counts[(counts >= min_channels).all(axis=1)].reset_index()[keys]
    if len(kept) < len(counts):
        logger.info("level %s: %d of %d occasion-condition(s) under %d channel(s) per "
                    "member left out", level, len(counts) - len(kept), len(counts),
                    min_channels)
    return d.merge(kept, on=keys), r.merge(kept, on=keys)


def _variants(draws: pd.DataFrame, real: pd.DataFrame, roi_map: "dict | None",
              min_channels: int = ROI_MIN_CHANNELS):
    """Every aggregate the draws support, as (granularity, level, pairings, draws, real).

    Emitted rather than selected, because which ones exist is a property of the draws and not
    a choice: all n^2 pairings only if the null was drawn crossed, regions only if a mapping
    was given.
    """
    hom_d, hom_r = _homologous(draws), _homologous(real)
    yield "whole", "whole", "homologous", hom_d, hom_r
    # a NaN label2 is a homologous table concatenated beside a crossed one, not a crossed
    # pairing: comparing against NaN is always unequal and would count it as crossed
    crossed = {o for o, part in draws.groupby("occasion")
               if "label2" in part.columns
               and (part["label"] != part["label2"].fillna(part["label"])).any()}
    # and the real table: n homologous cells against an n^2 null is not one statistic
    real_crossed = ("label2" in real.columns
                    and (real["label"] != real["label2"].fillna(real["label"])).any())
    if crossed and not real_crossed:
        logger.warning("no all-pairings level: the draws are crossed and the real table is "
                       "not, so that level would rank a mean over the diagonal inside a null "
                       "over every pairing. Rerun the real tables with --channel-cross")
    all_pairings = False
    if crossed and real_crossed:
        # A whole-brain mean is over every pairing, not the diagonal.
        # All or none: n^2 pairings for one occasion and n for the next is not one statistic
        short = sorted(set(draws["occasion"].unique()) - crossed)
        if short:
            logger.warning("no all-pairings level: %d of %d occasions are homologous only "
                           "(%s). Rerun those with --wtc-pair-cross, or read the whole "
                           "homologous level, which every occasion supports",
                           len(short), draws["occasion"].nunique(), ", ".join(short))
        else:
            all_pairings = True
            yield "whole", "whole", "all", draws, real
    roi_map = bare_roi_map(roi_map or {})
    for name, channels in roi_map.items():
        d, r = _thick(name, hom_d[hom_d["label"].isin(channels)],
                      hom_r[hom_r["label"].isin(channels)], ("label",), min_channels)
        if not d.empty and not r.empty:
            yield "roi", name, "homologous", d, r
    # with the null drawn crossed, every ordered region pair as the crossed ROI matrix groups it
    if all_pairings:
        for a, chans_a in roi_map.items():
            for b, chans_b in roi_map.items():
                # each member's side needs its own channels, not just enough pairings
                d, r = _thick(f"{a}>{b}",
                              draws[draws["label"].isin(chans_a) & draws["label2"].isin(chans_b)],
                              real[real["label"].isin(chans_a) & real["label2"].isin(chans_b)],
                              ("label", "label2"), min_channels)
                if not d.empty and not r.empty:
                    yield "roi", f"{a}>{b}", "all", d, r
    # one test per channel pairing, pooled over occasions
    for label in sorted(hom_d["label"].unique()):
        d = hom_d[hom_d["label"] == label]
        r = hom_r[hom_r["label"] == label]
        if not d.empty and not r.empty:
            yield "channel", label, "homologous", d, r
    # with the null drawn crossed, every pairing is also tested on its own: a family of n^2
    if all_pairings:
        real_by_pair = dict(tuple(real.groupby(["label", "label2"])))
        for (a, b), d in draws.groupby(["label", "label2"]):
            r = real_by_pair.get((a, b))
            if r is not None and not r.empty:
                yield "channel", f"{a}>{b}", "all", d, r


def _read_tree(output_dir: Path, suffix: str, task: str, chroma: str,
               needs: "tuple[str, ...]" = ()) -> pd.DataFrame:
    """Every occasion's table concatenated, refusing rather than dropping a malformed one.

    A table missing a column the others have concatenates to NaN and then leaves the groupby
    without a word, taking that occasion's contribution to every mean with it.
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


def _params_of(tsv: str) -> dict:
    """One table's sidecar parameters, empty when it has no sidecar."""
    side = Path(tsv).with_suffix(".json")
    return json.loads(side.read_text()).get("parameters", {}) if side.exists() else {}


def _band_of(paths: "set[str]") -> set:
    """The band each table's sidecar records, as a set so a mismatch is visible."""
    bands = set()
    for tsv in paths:
        params = _params_of(tsv)
        if params.get("band_fmin") is not None:
            bands.add((params["band_fmin"], params["band_fmax"]))
    return bands


def _roi_min_of(paths: "set[str]") -> int:
    """The ROI minimum the real tables were grouped under, so a region here is one there too.

    ::

      three tables recording 2  ->  2
      one recording 2, one 3    ->  ValueError
      one recording nothing     ->  ValueError
    """
    recorded = {tsv: _params_of(tsv).get("roi_min_channels") for tsv in paths}
    unrecorded = sorted(Path(tsv).name for tsv, v in recorded.items() if v is None)
    if unrecorded:
        raise ValueError(
            f"{len(unrecorded)} real table(s) do not record the ROI minimum they were grouped "
            f"under, e.g. {unrecorded[0]}. Rerun `fnirs-hyper` for those dyads on current code.")
    found = {int(v) for v in recorded.values()}
    if len(found) > 1:
        raise ValueError(
            f"the real tables were grouped into regions under different minimums: "
            f"{sorted(found)}. A region thinned out in one dyad and kept in another is not "
            f"one level. Rerun the dyads on one --wtc-roi-min-channels before reading this.")
    return found.pop()


def _exact_p(beaten: int, n: int) -> float:
    """The permutation p of a real value that beat ``beaten`` of ``n`` draws.

    ::

      beat all 19 of 19  ->  1/20 = 0.05, the smallest a pool of 19 can express

    The real value is counted into its own null, which is what keeps the test exact rather
    than letting a pool of 19 report a p of 0.
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
    # a channel no occasion has both a real value and draws for
    if not rows:
        return pd.DataFrame(columns=["condition", "occasion"])
    return pd.DataFrame(rows).sort_values(["condition", "occasion"], ignore_index=True)


def by_cohort(draws: pd.DataFrame, real: pd.DataFrame,
              n_resample: int = 20000, seed: int | None = None) -> pd.DataFrame:
    """The cohort mean against its null, by both reads, one row of each per condition.

    ``resample`` redraws one stand-in per occasion and ranks the real statistic inside that,
    so the null's width is the spread between an occasion's own draws. Resampling rather than
    pairing the pools by index, because a draw's position in one occasion's pool means
    nothing in another's.

    ``paired`` averages each occasion's draws into one baseline and tests real against it
    paired over occasions, so the width is the spread between occasions.

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
    if not rows:
        return pd.DataFrame(columns=["condition", "test"])
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
        # one-tailed: real above its baseline
        row["p"] = float(p_two / 2 if t > 0 else 1.0 - p_two / 2)
    return row


# the multipletests method names; none leaves the raw p as the only one written
P_CORRECTIONS = ("none", "fdr_bh", "fdr_by", "holm", "bonferroni")


def _correct(frame: pd.DataFrame, keys: list[str], method: str) -> pd.DataFrame:
    """Add ``family`` and, unless ``method`` is none, a ``p_<method>`` column beside ``p``.

    ::

      4 channels in one condition, holm  ->  family 4, p kept, p_holm added

    Rows whose ``p`` is absent take no part in any family and keep a blank corrected p.
    """
    from statsmodels.stats.multitest import multipletests

    if method not in P_CORRECTIONS:
        raise ValueError(f"unknown p correction {method!r}; one of {P_CORRECTIONS}")
    frame = frame.copy()
    col = f"p_{method}"
    if method != "none":
        frame[col] = np.nan
    frame["family"] = 0
    if "p" not in frame.columns or not keys:
        return frame
    for _, part in frame.groupby(keys, dropna=False):
        usable = part[part["p"].notna()]
        if usable.empty:
            continue
        if method != "none":
            frame.loc[usable.index, col] = multipletests(usable["p"], method=method)[1]
        frame.loc[part.index, "family"] = len(usable)
    return frame


def correct_cohort(frame: pd.DataFrame, method: str = "none") -> pd.DataFrame:
    """Correct the cohort p within one condition at one level, when a method is asked for.

    ::

      channel granularity, N channels in one condition  ->  family N

    ``family`` is the number of tests a correction runs over, written whether or not one ran.
    At whole-brain granularity the family is one cell, so a corrected p equals its p.

    Rows whose ``p`` is absent, which is what a paired test over fewer than three occasions
    leaves, take no part in any family.
    """
    keys = [k for k in ("granularity", "pairings", "test", "condition") if k in frame.columns]
    return _correct(frame, keys, method)


def by_cell(output_dir: Path, task: str, chroma: str, null: str,
            method: str = "none") -> "pd.DataFrame | None":
    """The per-cell percentiles a null already wrote, turned into p values.

    ::

      percentile 94.74 off 19 draws  ->  p 0.1, because 18 of 19 beaten is rank 2 of 20

    The stage that writes a cell's percentile runs one dyad at a time and so cannot correct
    across cells. One family per condition and level, so the crossed ROI matrix is corrected
    apart from the homologous regions: its diagonal is a different quantity from theirs.
    """
    parts = []
    for level, suffix in CELL_SUFFIX[null].items():
        try:
            frame = _read_tree(output_dir, suffix, task, chroma, needs=("percentile",))
        except FileNotFoundError:
            continue
        frame = frame.dropna(subset=["percentile"]).copy()
        if frame.empty:
            continue
        # the real value counted into its own null, as everywhere else here
        beaten = frame["percentile"] / 100 * frame["n_iter"]
        frame["p"] = _exact_p(beaten, frame["n_iter"])
        parts.append(_correct(frame, ["condition"], method).assign(level=level))
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    front = ["level", "condition", "occasion", "label"]
    keep = front + [c for c in ("label2", "coherence", "null_mean", "null_sd", "null_p95",
                                "percentile", "n_iter", "p", f"p_{method}", "family")
                    if c in out.columns]
    return out[keep].sort_values(["level", "condition", "p"], ignore_index=True)


def write_group_null(output_dir: Path, task: str, chroma: str = "hbo",
                     null: str = "repaired", roi_map: "dict | None" = None,
                     n_resample: int = 20000, seed: int | None = None,
                     p_correction: str = "none") -> list[Path]:
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
    # a null averaged over one band and a real value over another measure different things
    bands = _band_of(set(draws.source)) | _band_of(set(real.source))
    if len(bands) > 1:
        raise ValueError(
            f"the draws and the real tables are not on one band: {sorted(bands)}. A null "
            f"averaged over one band cannot be subtracted from a value averaged over "
            f"another. Finish whichever rerun is in progress before reading this.")

    min_channels = _roi_min_of(set(real.source)) if roi_map else ROI_MIN_CHANNELS

    occ_parts, coh_parts = [], []
    for gran, level, pairings, d, r in _variants(draws, real, roi_map,
                                                 min_channels=min_channels):
        if gran != "channel":
            logger.info("%s null, level %s over %s pairings: %d occasions, %d channels",
                        null, level, pairings, d.occasion.nunique(), d.label.nunique())
        tag = dict(granularity=gran, level=level, pairings=pairings)
        occ = by_occasion(d, r)
        coh = by_cohort(d, r, n_resample=n_resample, seed=seed)
        if len(occ):
            occ_parts.append(occ.assign(**tag))
        if len(coh):
            coh_parts.append(coh.assign(**tag))
    n_ch = sum(1 for p in coh_parts if (p["granularity"] == "channel").all())
    logger.info("%s null, and one level per channel pairing: %d of them", null, n_ch)

    def tidy(parts):
        out = pd.concat(parts, ignore_index=True)
        front = ["granularity", "level", "pairings", "condition"]
        return out[front + [c for c in out.columns if c not in front]]

    params = dict(null_kind=null, chroma=chroma, task=task,
                  n_resample=n_resample, seed=seed,
                  **({"roi_min_channels": min_channels} if roi_map else {}),
                  granularities=sorted(pd.concat(coh_parts).granularity.unique()),
                  statistic="mean over channel pairings, then over occasions",
                  p_correction=p_correction,
                  cell_correction_family="one condition and level, over occasions and "
                                         "pairings",
                  cohort_correction_family="one condition, one granularity, one pairing set "
                                           "and one test, over that level's cells; the "
                                           "`family` column carries its size")
    cells = by_cell(output_dir, task, chroma, null, method=p_correction)
    if cells is not None:
        logger.info("%s null, per cell: %d cells, %d at uncorrected p<0.05", null,
                    len(cells), int((cells["p"] < 0.05).sum()))
        if p_correction != "none":
            logger.info("%s null, per cell: %d at %s-corrected p<0.05", null,
                        int((cells[f"p_{p_correction}"] < 0.05).sum()), p_correction)

    # at the root, so no group- and no sub-: what marks a table as cross-dyad is having no
    # analysis unit in its name. The task and the chromophore stay, both being a filter this
    # command was given rather than something it merged over, so a second run for another
    # task or chromophore does not overwrite the first.
    common = {"chromophore": chroma, "task": task, "condition": "all",
              "nulldist": _NULL[null], "statistic": "wtc"}
    written = []
    for frame, desc, step in (
            ([] if cells is None else cells, "bycell", f"hyper_{null}_null_by_cell"),
            (tidy(occ_parts), "byoccasion", f"hyper_{null}_null_by_occasion"),
            (correct_cohort(tidy(coh_parts), p_correction), "cohort",
             f"hyper_{null}_null_cohort")):
        if len(frame) == 0:
            continue
        path = derivative_path(output_dir, "relmat", ".tsv", **common, desc=desc)
        write_tsv(frame, path)
        _hyper_sidecar(path, step, sources, **params)
        logger.info("%s: %d rows", path.name, len(frame))
        written.append(path)
    return written
