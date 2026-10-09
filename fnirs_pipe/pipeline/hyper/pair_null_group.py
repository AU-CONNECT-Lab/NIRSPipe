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
# the correlation's real table holds the whole run and every condition in one file
ISC_REAL_SUFFIX = _tail(statistic="isc")
ISC_DRAWS_SUFFIX = {kind: _tail(condition="all", nulldist=null, statistic="isc", desc="draws")
                    for kind, null in _NULL.items()}
# signed: Fisher z, either direction; magnitude: |Fisher z|, larger than the null only
ISC_TESTS = ("signed", "magnitude")


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


def _tailed_p(value: float, pool: np.ndarray, two_sided: bool) -> float:
    """The permutation p of ``value`` in ``pool``, upper tail or both.

    ::

      value above all 19 draws  ->  0.05 one-tailed, 0.1 two-sided

    Two-sided doubles the smaller tail rather than comparing magnitudes, because a null need
    not be centred on zero: two people on one task share a positive floor.
    """
    upper = _exact_p(int((value > pool).sum()), pool.size)
    if not two_sided:
        return upper
    lower = _exact_p(int((value < pool).sum()), pool.size)
    return min(1.0, 2 * min(upper, lower))


def by_occasion(draws: pd.DataFrame, real: pd.DataFrame,
                two_sided: bool = False) -> pd.DataFrame:
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
            "p": _tailed_p(value, pool, two_sided), "n_iter": int(pool.size),
        })
    # a channel no occasion has both a real value and draws for
    if not rows:
        return pd.DataFrame(columns=["condition", "occasion"])
    return pd.DataFrame(rows).sort_values(["condition", "occasion"], ignore_index=True)


def by_cohort(draws: pd.DataFrame, real: pd.DataFrame,
              n_resample: int = 20000, seed: int | None = None,
              two_sided: bool = False) -> pd.DataFrame:
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
            "p": _tailed_p(value, null, two_sided),
            "n_occasions": len(pools),
            "n_iter_min": int(min(p.size for p in pools.values())),
            "n_resample": n_resample,
        })
        rows.append(_paired_row(cond, observed, pools, two_sided))
    if not rows:
        return pd.DataFrame(columns=["condition", "test"])
    return (pd.DataFrame(rows)
            .sort_values(["condition", "test"], ignore_index=True))


def _paired_row(cond, observed: pd.Series, pools: "dict[str, np.ndarray]",
                two_sided: bool = False) -> dict:
    """Real against each occasion's own averaged draws, paired over occasions.

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
        # one-tailed unless asked: real above its baseline
        row["p"] = float(p_two if two_sided else p_two / 2 if t > 0 else 1.0 - p_two / 2)
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
    return _cell_table(pd.concat(parts, ignore_index=True), method)


def by_cell_from_draws(draws: pd.DataFrame, real: pd.DataFrame, two_sided: bool = False,
                       method: str = "none", roi_map: "dict | None" = None,
                       min_channels: int = ROI_MIN_CHANNELS) -> "pd.DataFrame | None":
    """Each channel pairing, and each region, ranked in its own draws.

    ::

      one occasion's S1_D1 > S2_D2 in game, real 0.31 against 9 draws  ->  one row with its p

    For tables that store no percentile: the correlation's cells are ranked here rather than
    read back, so that one switch sets their tail and the cohort's. The levels and families
    are the ones `by_cell` reads for the coherence: channel, homologous region, and the
    crossed region matrix, each corrected on its own.
    """
    keys = [k for k in ("occasion", "condition", "sub1", "sub2", "label", "label2")
            if k in draws.columns and k in real.columns]
    truth = real.groupby(keys)["coherence"].first()
    rows = []
    for key, pool in draws.groupby(keys)["coherence"]:
        if key not in truth.index:
            continue
        value = float(truth[key])
        pool = pool.dropna().to_numpy(dtype=float)
        if not pool.size:
            continue
        rows.append({**dict(zip(keys, key)), "coherence": value,
                     "null_mean": float(pool.mean()),
                     "null_sd": float(pool.std(ddof=1)) if pool.size > 1 else np.nan,
                     "null_p95": float(np.percentile(pool, 95)),
                     "percentile": float((value > pool).mean() * 100),
                     "n_iter": int(pool.size), "p": _tailed_p(value, pool, two_sided)})
    parts = []
    if rows:
        parts.append(_correct(pd.DataFrame(rows), ["condition"], method).assign(level="channel"))
    parts += _roi_cells(draws, real, two_sided, method, roi_map, min_channels)
    if not parts:
        return None
    return _cell_table(pd.concat(parts, ignore_index=True), method)


def _roi_cells(draws: pd.DataFrame, real: pd.DataFrame, two_sided: bool, method: str,
               roi_map: "dict | None", min_channels: int) -> list[pd.DataFrame]:
    """Each region's mean ranked in its occasion's own draws, which is `by_occasion` per region.

    ::

      region front, homologous   ->  level roi, label front
      region front > back, all   ->  level roi_crossed, label front, label2 back
    """
    if not roi_map:
        return []
    by_level: dict[str, list] = {"roi": [], "roi_crossed": []}
    for gran, name, pairings, d, r in _variants(draws, real, roi_map,
                                                min_channels=min_channels):
        if gran != "roi":
            continue
        occ = by_occasion(d, r, two_sided=two_sided)
        if occ.empty:
            continue
        if pairings == "all":
            a, b = name.split(">", 1)
            by_level["roi_crossed"].append(occ.assign(label=a, label2=b))
        else:
            by_level["roi"].append(occ.assign(label=name))
    return [_correct(pd.concat(parts, ignore_index=True), ["condition"], method)
            .assign(level=level)
            for level, parts in by_level.items() if parts]


def _cell_table(out: pd.DataFrame, method: str) -> pd.DataFrame:
    front = ["level", "condition", "occasion", "label"]
    keep = front + [c for c in ("label2", "coherence", "null_mean", "null_sd", "null_p95",
                                "percentile", "n_iter", "p", f"p_{method}", "family")
                    if c in out.columns]
    return out[keep].sort_values(["level", "condition", "p"], ignore_index=True)


def write_group_null(output_dir: Path, task: str, chroma: str = "hbo",
                     null: str = "repaired", roi_map: "dict | None" = None,
                     n_resample: int = 20000, seed: int | None = None,
                     p_correction: str = "none", isc_test: str = "signed") -> list[Path]:
    """Every level the draws support, for each statistic whose draws are on disk.

    Two files per statistic, not one per level: the levels differ in two columns and are read
    against each other, so they belong in one table. `level` is ``whole`` or a region name,
    `pairings` is which channel pairings entered the mean.

    Coherence is tested one-tailed, being unsigned. The correlation follows ``isc_test``.
    """
    if isc_test not in ISC_TESTS:
        raise ValueError(f"unknown ISC test {isc_test!r}; one of {ISC_TESTS}")
    output_dir = Path(output_dir)
    have = {stat: any(output_dir.rglob(f"group-*_task-{task}{suffix}"))
            for stat, suffix in (("wtc", DRAWS_SUFFIX[null]), ("isc", ISC_DRAWS_SUFFIX[null]))}
    if not any(have.values()):
        raise FileNotFoundError(
            f"no {null} draws for task {task} under {output_dir}, for the coherence or the "
            f"correlation. The cohort levels are built from the draws a null writes, so "
            f"`fnirs-hyper pair-null` or a `run --wtc-phase-null` / `--isc-phase-null` has "
            f"to have produced them.")
    shared = dict(output_dir=output_dir, task=task, chroma=chroma, null=null, roi_map=roi_map,
                  n_resample=n_resample, seed=seed, p_correction=p_correction)
    written = []
    if have["wtc"]:
        written += _write_wtc(**shared)
    else:
        logger.info("no coherence draws from the %s null; reading the correlation only", null)
    if have["isc"]:
        written += _write_isc(**shared, isc_test=isc_test)
    else:
        logger.info("no correlation draws from the %s null; reading the coherence only", null)
    return written


def _write_wtc(output_dir: Path, task: str, chroma: str, null: str, roi_map, n_resample: int,
               seed, p_correction: str) -> list[Path]:
    draws = _read_tree(output_dir, DRAWS_SUFFIX[null], task, chroma, needs=("draw", "label"))
    real = _read_tree(output_dir, REAL_SUFFIX, task, chroma, needs=("label",))
    # a null averaged over one band and a real value over another measure different things
    bands = _band_of(set(draws.source)) | _band_of(set(real.source))
    if len(bands) > 1:
        raise ValueError(
            f"the draws and the real tables are not on one band: {sorted(bands)}. A null "
            f"averaged over one band cannot be subtracted from a value averaged over "
            f"another. Finish whichever rerun is in progress before reading this.")
    cells = by_cell(output_dir, task, chroma, null, method=p_correction)
    return _write_levels(output_dir, task, chroma, null, "wtc", draws, real, cells, roi_map,
                         n_resample, seed, p_correction, two_sided=False,
                         value_name="coherence",
                         statistic="mean over channel pairings, then over occasions")


def _write_isc(output_dir: Path, task: str, chroma: str, null: str, roi_map, n_resample: int,
               seed, p_correction: str, isc_test: str) -> list[Path]:
    draws = _read_isc(output_dir, ISC_DRAWS_SUFFIX[null], task, chroma, isc_test,
                      needs=("draw", "label", "label2"))
    real = _read_isc(output_dir, ISC_REAL_SUFFIX, task, chroma, isc_test,
                     needs=("label", "label2"))
    settings = _isc_settings_of(set(draws.source) | set(real.source))
    if len(settings) > 1:
        raise ValueError(
            f"the correlation's draws and real tables were not computed alike (band, "
            f"whitening order, lag search): {sorted(settings, key=str)}. Finish whichever "
            f"rerun is in progress before reading this.")
    two_sided = isc_test == "signed"
    min_channels = _roi_min_of(set(real.source)) if roi_map else ROI_MIN_CHANNELS
    cells = by_cell_from_draws(draws, real, two_sided=two_sided, method=p_correction,
                               roi_map=roi_map, min_channels=min_channels)
    magnitude = "|Fisher z|" if isc_test == "magnitude" else "Fisher z"
    return _write_levels(output_dir, task, chroma, null, "isc", draws, real, cells, roi_map,
                         n_resample, seed, p_correction, two_sided=two_sided,
                         value_name="abs_r_z" if isc_test == "magnitude" else "r_z",
                         statistic=f"mean {magnitude} over channel pairings, then over "
                                   f"occasions",
                         extra={"isc_test": isc_test})


def _read_isc(output_dir: Path, suffix: str, task: str, chroma: str, isc_test: str,
              needs: "tuple[str, ...]" = ()) -> pd.DataFrame:
    """The correlation's rows with the value its test ranks in ``coherence``.

    ::

      r -0.4, signed  ->  coherence -0.42;  magnitude  ->  +0.42

    Renamed into the column the coherence code reads rather than that code duplicated, as the
    re-paired null does. The whole-run rows carry no condition and are dropped, the draws
    being per condition only.
    """
    frame = _read_tree(output_dir, suffix, task, chroma, needs=("r_z", *needs))
    if "condition" in frame.columns:
        frame = frame[frame["condition"].notna()]
    z = frame["r_z"]
    frame = frame.assign(coherence=z.abs() if isc_test == "magnitude" else z)
    return frame.dropna(subset=["coherence"])


def _isc_settings_of(paths: "set[str]") -> set:
    """What each correlation table was computed with, refusing a table that does not say.

    ::

      every sidecar band 0.01-0.1, AR 32, lag 2 s  ->  {((0.01, 0.1), 32, 2.0)}
    """
    found = set()
    for tsv in paths:
        params = _params_of(tsv)
        if "isc_whiten_max_order" not in params:
            raise ValueError(
                f"{Path(tsv).name} does not record how its correlation was computed. Rerun "
                f"`fnirs-hyper` (and its null) for that dyad on current code.")
        band = params.get("isc_band_hz")
        found.add((tuple(band) if band else None, params["isc_whiten_max_order"],
                   params.get("isc_max_lag_s")))
    return found


def _write_levels(output_dir: Path, task: str, chroma: str, null: str, measure: str,
                  draws: pd.DataFrame, real: pd.DataFrame, cells: "pd.DataFrame | None",
                  roi_map, n_resample: int, seed, p_correction: str, two_sided: bool,
                  value_name: str, statistic: str, extra: "dict | None" = None) -> list[Path]:
    """One statistic's three tables and their sidecars, the cells already ranked."""
    from fnirs_pipe.pipeline.hyper.group_io import _hyper_sidecar

    sources = sorted(set(draws.source) | set(real.source))
    min_channels = _roi_min_of(set(real.source)) if roi_map else ROI_MIN_CHANNELS

    occ_parts, coh_parts = [], []
    for gran, level, pairings, d, r in _variants(draws, real, roi_map,
                                                 min_channels=min_channels):
        if gran != "channel":
            logger.info("%s %s null, level %s over %s pairings: %d occasions, %d channels",
                        measure, null, level, pairings, d.occasion.nunique(),
                        d.label.nunique())
        tag = dict(granularity=gran, level=level, pairings=pairings)
        occ = by_occasion(d, r, two_sided=two_sided)
        coh = by_cohort(d, r, n_resample=n_resample, seed=seed, two_sided=two_sided)
        if len(occ):
            occ_parts.append(occ.assign(**tag))
        if len(coh):
            coh_parts.append(coh.assign(**tag))
    n_ch = sum(1 for p in coh_parts if (p["granularity"] == "channel").all())
    logger.info("%s %s null, and one level per channel pairing: %d of them", measure, null,
                n_ch)

    def tidy(parts):
        if not parts:
            return []
        out = pd.concat(parts, ignore_index=True)
        front = ["granularity", "level", "pairings", "condition"]
        return out[front + [c for c in out.columns if c not in front]]

    params = dict(measure=measure, null_kind=null, chroma=chroma, task=task,
                  n_resample=n_resample, seed=seed,
                  **({"roi_min_channels": min_channels} if roi_map else {}),
                  granularities=sorted({p.granularity.iloc[0] for p in coh_parts}),
                  statistic=statistic, two_sided=two_sided, **(extra or {}),
                  p_correction=p_correction,
                  cell_correction_family="one condition and level, over occasions and "
                                         "pairings",
                  cohort_correction_family="one condition, one granularity, one pairing set "
                                           "and one test, over that level's cells; the "
                                           "`family` column carries its size")
    if cells is not None:
        logger.info("%s %s null, per cell: %d cells, %d at uncorrected p<0.05", measure,
                    null, len(cells), int((cells["p"] < 0.05).sum()))
        if p_correction != "none":
            logger.info("%s %s null, per cell: %d at %s-corrected p<0.05", measure, null,
                        int((cells[f"p_{p_correction}"] < 0.05).sum()), p_correction)

    # at the root, so no group- and no sub-: what marks a table as cross-dyad is having no
    # analysis unit in its name. The task and the chromophore stay, both being a filter this
    # command was given rather than something it merged over, so a second run for another
    # task or chromophore does not overwrite the first.
    common = {"chromophore": chroma, "task": task, "condition": "all",
              "nulldist": _NULL[null], "statistic": measure}
    cohort = tidy(coh_parts)
    written = []
    for frame, desc, step in (
            ([] if cells is None else cells, "bycell", f"hyper_{null}_null_by_cell"),
            (tidy(occ_parts), "byoccasion", f"hyper_{null}_null_by_occasion"),
            ([] if len(cohort) == 0 else correct_cohort(cohort, p_correction), "cohort",
             f"hyper_{null}_null_cohort")):
        if len(frame) == 0:
            continue
        path = derivative_path(output_dir, "relmat", ".tsv", **common, desc=desc)
        write_tsv(frame.rename(columns={"coherence": value_name}), path)
        _hyper_sidecar(path, step, sources, **params)
        logger.info("%s: %d rows", path.name, len(frame))
        written.append(path)
    return written
