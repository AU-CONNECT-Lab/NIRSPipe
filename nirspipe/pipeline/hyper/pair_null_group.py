"""The re-paired null read above the cell: one verdict per occasion, one per cohort.

``pair-null`` ranks each channel of each dyad inside its own draws, and a cell drawn against
n stand-ins cannot reach a p below 1/(n+1). Averaging first cannot say which channel, but it
can say whether the pairing beats its null at all.

Two levels, because they need different data and only one of them fits inside the per-dyad
stage:

``occasion``  the channels averaged, ranked in that occasion's own draws, each draw put on
              the real pairing's channels (`matched_pools`). One dyad's draws are enough;
              it is here so that both levels come off one table.
``cohort``    the channels averaged and then the occasions averaged, against a null that
              draws one stand-in per occasion. This needs every occasion's draws at once,
              which is why it cannot live in a stage that runs one dyad at a time.

A group of more than two members is one value per occasion as well, the mean over its member
pairings, and where the pairs table named roles each unordered role pair is also read on its
own (granularity ``rolepair``).

A homologous null tests the homologous pairings only, since a crossed real table's other
cells have no null behind it. A crossed null also tests every pairing and, given a region
map, every ordered region pair.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from nirspipe.utils import ROI_MIN_CHANNELS, bare_roi_map, correct_p
from nirspipe.utils.logging import get_logger
from nirspipe.io.derivatives import occasion_label
from nirspipe.io.naming import derivative_path, parse_path
from nirspipe.io.tables import LABEL_DTYPES, WHOLE_RUN, write_tsv

logger = get_logger("pipeline.pair_null_group")

_NULL = {"repaired": "pair", "phase": "phase"}


def _tail(**entities) -> str:
    """The part of a dyad table's name after its task, as a glob to find it by.

    ::

      _tail(nulldist="phase", statistic="wtc") -> "_null-phase_stat-wtc_relmat.tsv"

    Built through the namer rather than spelled out, so the tables this module reads and
    the ones the null writers produce cannot be renamed apart. The ROI map's own name is a
    wildcard: this reads whichever definition the tree was written with.
    """
    name = derivative_path("", "relmat", ".tsv", group="X", task="T", **entities).name
    return name.split("_task-T", 1)[1]


_REAL = {"statistic": "wtc"}
_BY_COND = {"condition": "all", "statistic": "wtc"}
_ROI = {"segmentation": "*", "aggregation": "homologous"}
_ROI_CROSSED = {"segmentation": "*", "aggregation": "roi"}

# the real table holds the whole run and every condition in one file
REAL_SUFFIX = _tail(**_REAL)
# the phase null's cell tables are shaped like the real one; the re-paired null has no
# whole-run half, so its tables hold the conditions only
_CELLS = {"repaired": _BY_COND, "phase": _REAL}
# each cell table a null writes, under the level its cells are corrected as one family in
CELL_SUFFIX = {
    kind: {"channel": _tail(**_CELLS[kind], nulldist=null),
           "roi": _tail(**_ROI, **_CELLS[kind], nulldist=null),
           "roi_crossed": _tail(**_ROI_CROSSED, **_CELLS[kind], nulldist=null)}
    for kind, null in _NULL.items()
}
DRAWS_SUFFIX = {kind: _tail(**_BY_COND, nulldist=null, desc="draws")
                for kind, null in _NULL.items()}
ISC_REAL_SUFFIX = _tail(statistic="isc")
ISC_DRAWS_SUFFIX = {kind: _tail(condition="all", nulldist=null, statistic="isc", desc="draws")
                    for kind, null in _NULL.items()}
# signed: Fisher z, either direction; magnitude: |Fisher z|, larger than the null only
ISC_TESTS = ("signed", "magnitude")


def magnitude_against_unwhitened_phase(isc_test, null_kind, isc_whiten_s) -> bool:
    """|r| ranked against the phase null on correlations that were not whitened, whose p runs small."""
    return isc_test == "magnitude" and null_kind == "phase" and not isc_whiten_s


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
    # every member pairing of a group has to keep the region, its mean being over all of them
    members = [k for k in ("sub1", "sub2") if k in r.columns]
    counts = r.groupby(keys + members)[list(sides)].nunique()
    thick = (counts >= min_channels).all(axis=1).groupby(level=keys).all()
    kept = thick[thick].reset_index()[keys]
    if len(kept) < len(thick):
        logger.info("level %s: %d of %d occasion-condition(s) under %d channel(s) per "
                    "member left out", level, len(thick) - len(kept), len(thick),
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
    # the whole brain once per unordered role pair, where the cohort holds more than one
    if "rolepair" in real.columns and real["rolepair"].nunique() > 1:
        for pair in sorted(real["rolepair"].dropna().unique()):
            yield ("rolepair", pair, "homologous", hom_d[hom_d["rolepair"] == pair],
                   hom_r[hom_r["rolepair"] == pair])
            if all_pairings:
                yield ("rolepair", pair, "all", draws[draws["rolepair"] == pair],
                       real[real["rolepair"] == pair])
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

    Every reading here is per condition, so a table's whole-run rows (condition whole_run)
    are dropped, and a table with no ``condition`` column is refused.
    """
    frames = []
    for path in sorted(Path(output_dir).rglob(f"group-*_task-{task}{suffix}")):
        entities = parse_path(path.name)
        frame = _of_chroma(pd.read_csv(path, sep="\t", dtype=LABEL_DTYPES), chroma)
        if "condition" not in frame.columns:
            raise ValueError(
                f"{path.name} has no condition column: its run had no conditions, or it "
                f"predates the table holding the whole run and the conditions as one. Rerun "
                f"`nirspipe-hyper` for that dyad with --wtc-by-condition.")
        frame = frame[frame["condition"] != WHOLE_RUN]
        missing = [c for c in needs if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path.name} has no {', '.join(missing)} column, so its occasion would drop "
                f"out of every mean silently. It predates the column: rerun that dyad's null, "
                f"or rename the column if the values are current.")
        frames.append(frame.assign(
            occasion=occasion_label(entities["group"], entities.get("session")),
            source=str(path)))
    if not frames:
        raise FileNotFoundError(
            f"no group-*_task-{task}{suffix} under {output_dir}. The cohort levels are "
            f"built from the draws a null writes, so `nirspipe-hyper pair-null` or a `run "
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
            f"under, e.g. {unrecorded[0]}. Rerun `nirspipe-hyper` for those dyads on current code.")
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


def _cell(frame: pd.DataFrame) -> pd.Series:
    """A pairing's key: its label, and its partner's where that is another channel.

    A homologous pairing keys alike whether its table carries ``label2`` or not.
    """
    label = frame["label"].astype(str)
    if "label2" not in frame.columns:
        return label
    other = frame["label2"].fillna(frame["label"]).astype(str)
    return label.where(other == label, label + ">" + other)


def matched_pools(draws: pd.DataFrame, real: pd.DataFrame) -> "dict[tuple, tuple[float, np.ndarray]]":
    """Each occasion-condition's real channel mean, and its draws moved onto the real pairing's channels.

    ::

      real kept a b c at .40 .30 .20 (mean .30), a draw kept a b at .35 .25
        -> over a and b the draw is .05 under the real; the draw counts as .30 - .05 = .25

    Each draw is compared on the pairings it shares with the real one and the difference is
    taken off the real mean, so the pool ranks, averages and spreads as the draws would on
    the real pairing's channels. With nothing rejected it is the draws' own means. A draw
    sharing no pairing is left out.

    A group of more than two members takes the mean over its member pairings of each one's
    channel mean, and so does each draw: matched pairing by pairing, then averaged under
    the draw id its pairings share. A draw missing one of the pairings is left out.
    """
    members = [k for k in ("sub1", "sub2") if k in real.columns and k in draws.columns]
    real = real.assign(cell=_cell(real))
    draws = draws.assign(cell=_cell(draws))
    by_occ = dict(tuple(draws.groupby(["condition", "occasion"])))
    out = {}
    for (cond, occ), part in real.groupby(["condition", "occasion"]):
        d = by_occ.get((cond, occ))
        if d is None:
            continue
        pairings = dict(tuple(part.groupby(members))) if members else {(): part}
        drawn = dict(tuple(d.groupby(members))) if members else {(): d}
        missing = [key for key in pairings if key not in drawn]
        if missing:
            logger.debug("%s %s: no draws for %s, so the mean over its pairings has no "
                         "null", occ, cond, ", ".join("-".join(k) for k in missing))
            continue
        got = [_matched(pairings[key], drawn[key]) for key in pairings]
        if any(m is None for m in got):
            continue
        value = float(np.mean([v for v, _ in got]))
        pool = pd.concat([p for _, p in got], axis=1).dropna().mean(axis=1)
        if len(got) > 1 and pool.empty:
            logger.debug("%s %s: no draw covers all %d of its pairings", occ, cond, len(got))
        out[(cond, occ)] = (value, pool.to_numpy(dtype=float))
    return out


def _matched(real: pd.DataFrame, draws: pd.DataFrame) -> "tuple[float, pd.Series] | None":
    """One pairing's real channel mean, and its draws on its channels by draw id; None with no real value."""
    x = real.groupby("cell").coherence.mean().dropna()
    if x.empty:
        return None
    by_draw = draws.groupby(["draw", "cell"]).coherence.first().unstack("cell")
    y = by_draw.reindex(columns=x.index).to_numpy(dtype=float)
    shared = np.isfinite(y)
    n = shared.sum(axis=1)
    keep = n > 0
    x_shared = (shared * x.to_numpy())[keep].sum(axis=1) / n[keep]
    y_shared = np.where(shared, y, 0.0)[keep].sum(axis=1) / n[keep]
    value = float(x.mean())
    return value, pd.Series(value - (x_shared - y_shared), index=by_draw.index[keep])


def _with_role_pairs(draws: pd.DataFrame, real: pd.DataFrame
                     ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Both frames with each pairing's unordered role pair, off the roles the real tables record.

    ::

      G01 roles {a: teacher, b: student}, row a-b  ->  rolepair "student-teacher"

    Left as they are where no real table records ``member_roles``.
    """
    roles = {}
    for source, occ in real[["source", "occasion"]].drop_duplicates().itertuples(index=False):
        for sub, role in (_params_of(source).get("member_roles") or {}).items():
            roles[(occ, sub)] = role
    if not roles or not all({"sub1", "sub2"} <= set(f.columns) for f in (draws, real)):
        return draws, real

    def pair_of(frame: pd.DataFrame) -> list:
        sides = [[roles.get(k) for k in zip(frame["occasion"], frame[side])]
                 for side in ("sub1", "sub2")]
        return ["-".join(sorted((a, b))) if a and b else None for a, b in zip(*sides)]
    return draws.assign(rolepair=pair_of(draws)), real.assign(rolepair=pair_of(real))


def by_occasion(draws: pd.DataFrame, real: pd.DataFrame,
                two_sided: bool = False) -> pd.DataFrame:
    """Each occasion's channel mean, ranked in that occasion's own draws on the real pairing's channels."""
    rows = []
    for (cond, occ), (value, pool) in matched_pools(draws, real).items():
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

    ``lift`` is the same number in both rows; what differs is what it is divided by. Each
    occasion's draws are on the real pairing's channels, as in `matched_pools`.
    """
    rng = np.random.default_rng(seed)
    matched = matched_pools(draws, real)
    rows = []
    for cond, real_part in real.groupby("condition"):
        # in the real table's order, which sets the order the resample draws in
        pools = {o: matched[(cond, o)][1] for o in real_part.occasion.unique()
                 if (cond, o) in matched and matched[(cond, o)][1].size}
        if not pools:
            continue
        observed = pd.Series({o: matched[(cond, o)][0] for o in pools})
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
    return correct_p(frame, keys, method)


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
        # the real value counted into its own null, as everywhere else here; rounded, as the
        # percentile is a count over n_iter written as a share
        beaten = (frame["percentile"] / 100 * frame["n_iter"]).round()
        frame["p"] = _exact_p(beaten, frame["n_iter"])
        parts.append(correct_p(frame, ["condition"], method).assign(level=level))
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
        parts.append(correct_p(pd.DataFrame(rows), ["condition"], method).assign(level="channel"))
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
    return [correct_p(pd.concat(parts, ignore_index=True), ["condition"], method)
            .assign(level=level)
            for level, parts in by_level.items() if parts]


def _cell_table(out: pd.DataFrame, method: str) -> pd.DataFrame:
    front = ["level", "condition", "occasion", "label"]
    keep = front + [c for c in ("label2", "coherence", "null_mean", "null_sd", "null_p95",
                                "percentile", "n_iter", "p", f"p_{method}", "family")
                    if c in out.columns]
    return out[keep].sort_values(["level", "condition", "p"], ignore_index=True)


def nulls_on_disk(output_dir: Path, task: str) -> list[str]:
    """The nulls whose draws, coherence or correlation, a dyad tree holds for ``task``."""
    output_dir = Path(output_dir)
    return [null for null in DRAWS_SUFFIX
            if any(any(output_dir.rglob(f"group-*_task-{task}{suffix}"))
                   for suffix in (DRAWS_SUFFIX[null], ISC_DRAWS_SUFFIX[null]))]


def chromas_on_disk(output_dir: Path, task: str, null: str) -> list[str]:
    """The chromophores one null's draws, coherence or correlation, hold for ``task``."""
    found = set()
    for suffix in (DRAWS_SUFFIX[null], ISC_DRAWS_SUFFIX[null]):
        for path in Path(output_dir).rglob(f"group-*_task-{task}{suffix}"):
            found |= set(pd.read_csv(path, sep="\t", usecols=["chromophore"]).chromophore.dropna())
    return sorted(found)


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
            f"`nirspipe-hyper pair-null` or a `run --wtc-phase-null` / `--isc-phase-null` has "
            f"to have produced them.")
    # otherwise every level comes out empty and nothing is written, without an error
    held = chromas_on_disk(output_dir, task, null)
    if chroma not in held:
        raise ValueError(
            f"no {chroma} draws from the {null} null for task {task} under {output_dir}; "
            f"they hold {', '.join(held) or 'no chromophore'}. Rerun the null with {chroma}, "
            f"or ask only for what the draws hold.")
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
    shared = _phase_shared_of(set(draws.source)) if null == "phase" else set()
    if len(shared) > 1:
        raise ValueError(
            "the coherence's phase draws were scrambled both ways, some sharing their phases "
            "across a member's channels and some not. Rerun `nirspipe-hyper` for every dyad "
            "with the same --phase-null-shared before reading this.")
    cells = by_cell(output_dir, task, chroma, null, method=p_correction)
    return _write_levels(output_dir, task, chroma, null, "wtc", draws, real, cells, roi_map,
                         n_resample, seed, p_correction, two_sided=False,
                         value_name="coherence",
                         statistic="mean over channel pairings, then over occasions",
                         extra={"phase_null_shared": next(iter(shared))} if shared else None)


def _phase_shared_of(paths: "set[str]") -> set:
    """Whether each phase-null table's draws shared their phases, refusing one that does not say."""
    found = set()
    for tsv in paths:
        params = _params_of(tsv)
        if "phase_null_shared" not in params:
            raise ValueError(
                f"{Path(tsv).name} does not record whether its phase null shared its phases. "
                f"Rerun `nirspipe-hyper` for that dyad on current code.")
        found.add(bool(params["phase_null_shared"]))
    return found


def _write_isc(output_dir: Path, task: str, chroma: str, null: str, roi_map, n_resample: int,
               seed, p_correction: str, isc_test: str) -> list[Path]:
    draws = _read_isc(output_dir, ISC_DRAWS_SUFFIX[null], task, chroma, isc_test,
                      needs=("draw", "label", "label2"))
    real = _read_isc(output_dir, ISC_REAL_SUFFIX, task, chroma, isc_test,
                     needs=("label", "label2"))
    settings = _isc_settings_of(set(draws.source) | set(real.source), null)
    if len(settings) > 1:
        raise ValueError(
            f"the correlation's draws and real tables were not computed alike (band, "
            f"whitening order, lag search{', shared phases' if null == 'phase' else ''}): "
            f"{sorted(settings, key=str)}. Finish whichever rerun is in progress before "
            f"reading this.")
    whiten_s = next(iter(settings))[1] if settings else None
    if magnitude_against_unwhitened_phase(isc_test, null, whiten_s):
        logger.warning("ISC %s, phase null: |r| on unwhitened correlations tends to sit above "
                       "its phase-scrambled null, so these p run small; whiten in "
                       "nirspipe-hyper (--isc-whiten-s) or use --isc-test signed", chroma)
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
                         extra={"isc_test": isc_test, "isc_whiten_s": whiten_s,
                                **({"phase_null_shared": next(iter(settings))[3]}
                                   if null == "phase" and settings else {})})


def _read_isc(output_dir: Path, suffix: str, task: str, chroma: str, isc_test: str,
              needs: "tuple[str, ...]" = ()) -> pd.DataFrame:
    """The correlation's rows with the value its test ranks in ``coherence``.

    ::

      r -0.4, signed  ->  coherence -0.42;  magnitude  ->  +0.42

    Renamed into the column the coherence code reads rather than that code duplicated, as the
    re-paired null does.
    """
    frame = _read_tree(output_dir, suffix, task, chroma, needs=("r_z", *needs))
    z = frame["r_z"]
    frame = frame.assign(coherence=z.abs() if isc_test == "magnitude" else z)
    return frame.dropna(subset=["coherence"])


def _isc_settings_of(paths: "set[str]", null: str = "repaired") -> set:
    """What each correlation table was computed with, refusing a table that does not say.

    ::

      every sidecar band 0.01-0.1, AR 10 s, lag 2 s  ->  {((0.01, 0.1), 10.0, 2.0)}

    The phase null also says whether its draws shared their phases across a member's channels.
    """
    found = set()
    for tsv in paths:
        params = _params_of(tsv)
        if "isc_whiten_s" not in params:
            raise ValueError(
                f"{Path(tsv).name} does not record how its correlation was computed. Rerun "
                f"`nirspipe-hyper` (and its null) for that dyad on current code.")
        if null == "phase" and "phase_null_shared" not in params:
            raise ValueError(
                f"{Path(tsv).name} does not record whether its phase null shared its phases. "
                f"Rerun `nirspipe-hyper` for that dyad on current code.")
        band = params.get("isc_band_hz")
        found.add((tuple(band) if band else None, params["isc_whiten_s"],
                   params.get("isc_max_lag_s"),
                   *([params["phase_null_shared"]] if null == "phase" else [])))
    return found


def _write_levels(output_dir: Path, task: str, chroma: str, null: str, measure: str,
                  draws: pd.DataFrame, real: pd.DataFrame, cells: "pd.DataFrame | None",
                  roi_map, n_resample: int, seed, p_correction: str, two_sided: bool,
                  value_name: str, statistic: str, extra: "dict | None" = None) -> list[Path]:
    """One statistic's three tables and their sidecars, the cells already ranked."""
    from nirspipe.pipeline.hyper.group_io import _hyper_sidecar

    sources = sorted(set(draws.source) | set(real.source))
    min_channels = _roi_min_of(set(real.source)) if roi_map else ROI_MIN_CHANNELS
    draws, real = _with_role_pairs(draws, real)

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
        # once per level and by name: a channel only some pairings kept loses occasions as a
        # matter of course, a whole-brain level losing one is worth a warning
        unranked = sorted(set(r[["condition", "occasion"]].itertuples(index=False, name=None))
                          - set(occ[["condition", "occasion"]].itertuples(index=False,
                                                                          name=None)))
        if unranked:
            (logger.info if gran == "channel" else logger.warning)(
                "%s %s null, level %s over %s pairings: %d occasion-condition(s) with no draw "
                "to rank against, left out: %s", measure, null, level, pairings, len(unranked),
                ", ".join(f"{o} {c}" for c, o in unranked))
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
