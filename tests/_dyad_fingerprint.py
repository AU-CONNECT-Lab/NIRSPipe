"""Synthetic groups whose every member, channel, chromophore, clock and coupling is identifiable."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import zlib

import mne
import numpy as np

from tests._fingerprint import (CARDIAC_AMP, CARDIAC_FREQ, DEAD_OD, DPF, LONG_DISTANCE,
                                SHORT_DISTANCE,
                                _haemo_to_od, _info, _layout)
from tests._synth import _write_dataset_root, _write_subject

SFREQ = 10.0
DURATION = 400.0                     # the aligned span every member covers
N_LONG = 5
N_SHORT = 1
LONG_AMP = 0.6e-6                    # each long pair's own HbO oscillation
NOISE_AMP = 0.3e-6                   # independent in-band noise, every channel
NOISE_BAND = (0.01, 0.12)
COUPLING_BAND = (0.03, 0.10)         # the coherence band the dyad runs are given
SHARED_PAD = 120.0                   # the shared draw covers every member's own span
HBR_MARK_AMP = 0.3e-6
OD_NOISE = 0.002
TASK = "main"

# ---- The shared clock: the trigger every member carries, and the two blocks ----
SYNC = "sync"                        # zero duration, at 0 on the aligned clock
BLOCKS = (("ca", 30.0, 160.0), ("cb", 210.0, 160.0))

# ---- Members (group, subject, own-clock offset of the trigger, tail past the span, rejected
# long pair, OD noise scale). Every offset differs, so a figure on a member's own clock is off
# by exactly that member's offset; the noise scales rank the cohort.
MEMBERS = (
    ("G01", "01", 12.0, 0.0, "S5_D5", 0.5),
    ("G01", "02", 37.0, 7.0, "S4_D4", 0.75),
    ("G02", "03", 12.0, 3.0, "S4_D4", 1.0),
    ("G02", "04", 37.0, 0.0, "S3_D3", 1.25),
    ("G02", "05", 61.0, 5.0, "S5_D5", 1.5),
)
# in no group, for the cohort page alone: one run without short channels whose rejected pair
# is dead flat, sorted before an outlier that moves and is noisy throughout
COHORT_ONLY = (
    ("", "00", 20.0, 0.0, "S2_D2", 0.6),
    ("", "03b", 20.0, 0.0, "S1_D1", 1.4),
)
NO_SHORT = {"00"}
DEAD = {"00"}                        # the rejected pair is constant rather than uncoupled
# per member, a kept and uncoupled pair whose wavelengths decouple over part of one block only
# (aligned onset, span): the run keeps it (0.85 coupled) and that block alone fails it (0.625).
# Each starts on a multiple of 60 s of its member's own clock, so the 10 s and 12 s window
# grids both cut it whole and every member's windowed SCI drops by the same amount
CONDITION_FAILING = {"01": ("S3_D3", 228.0, 60.0), "02": ("S5_D5", 83.0, 60.0),
                     "03": ("S3_D3", 228.0, 60.0), "04": ("S1_D1", 83.0, 60.0),
                     "05": ("S2_D2", 239.0, 60.0)}


@dataclass(frozen=True)
class Coupling:
    """Band-limited noise shared by two members' channels; ``b`` lags ``a`` by ``lag_s``.
    A negative lag means ``b`` leads."""
    a: str                           # subject
    a_pair: str
    b: str
    b_pair: str
    chroma: str
    lag_s: float
    block: str | None = None         # only inside this block, or the whole span
    amp: float = 0.8e-6

    def phase_deg(self, freq: float) -> float:
        """The WTC phase of ``a`` against ``b`` at ``freq``: positive when ``a`` leads."""
        return 360.0 * freq * self.lag_s

    def shared(self, t_al: np.ndarray) -> np.ndarray:
        """The common signal on the aligned clock, the same draw for both members."""
        rng = np.random.default_rng(zlib.crc32(f"{self.a}{self.a_pair}{self.b}{self.chroma}".encode()))
        grid = np.arange(-SHARED_PAD * SFREQ, (DURATION + SHARED_PAD) * SFREQ) / SFREQ
        wave = self.amp * _band_noise(rng, len(grid), COUPLING_BAND)
        return np.interp(t_al, grid, wave)


COUPLINGS = (
    # the dyad: homologous in one block, off the diagonal over the whole span, and one in HbR
    Coupling("01", "S2_D2", "02", "S2_D2", "hbo", 2.5, block="ca"),
    Coupling("01", "S1_D1", "02", "S3_D3", "hbo", -2.0),
    Coupling("01", "S4_D4", "02", "S1_D1", "hbr", 3.0),
    # the triad: each pairing coupled somewhere else, so a pairing label shifted shows
    Coupling("03", "S2_D2", "04", "S2_D2", "hbo", 2.5, block="ca"),
    Coupling("03", "S1_D1", "05", "S3_D3", "hbo", -2.0),
    Coupling("04", "S4_D4", "05", "S1_D1", "hbr", 3.0),
)

# motion: one bump of 0.5 OD on every channel, at these aligned times; the last is shared
MOTION_OD = 0.5
OWN_SPIKE = {"01": 100.0, "02": 250.0, "03": 100.0, "04": 250.0, "05": 160.0, "00": 100.0,
             "03b": 100.0}
EXTRA_SPIKES = {"03b": tuple(range(20, 400, 25))}
SHARED_SPIKE = 320.0

# ---- Fingerprint frequencies (Hz): own HbO per subject and pair, HbR mark per subject ----
# all distinct across the whole cohort, all above the coherence band
_STEP = 1 / 360.0


def own_freq(subject: str, pair_index: int) -> float:
    s = [m[1] for m in MEMBERS + COHORT_ONLY].index(subject)
    return (40 + 5 * (N_LONG * s + pair_index)) * _STEP


def hbr_mark_freq(subject: str) -> float:
    s = [m[1] for m in MEMBERS + COHORT_ONLY].index(subject)
    return (167 + 5 * s) * _STEP


@dataclass
class Member:
    group: str
    subject: str
    offset: float                    # own-clock onset of the shared trigger
    tail: float
    rejected: str
    noise: float

    @property
    def sid(self) -> str:
        return f"sub-{self.subject}"


@dataclass
class GroupTruth:
    members: list[Member]
    couplings: tuple[Coupling, ...]
    long_pairs: list[str]
    short_pairs: list[str]
    blocks: tuple = BLOCKS
    # haemoglobin on the aligned clock per subject and channel name ("S1_D1 hbo"), in M
    haemo: dict[str, dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    def member(self, subject: str) -> Member:
        subject = subject.removeprefix("sub-")
        return next(m for m in self.members if m.subject == subject)

    def group(self, group: str) -> list[Member]:
        return [m for m in self.members if m.group == group]

    def couplings_between(self, a: str, b: str, chroma: str | None = None) -> list[Coupling]:
        """The couplings of one pairing, oriented so ``a`` is the first member."""
        a, b = a.removeprefix("sub-"), b.removeprefix("sub-")
        out = []
        for c in self.couplings:
            if chroma and c.chroma != chroma:
                continue
            if (c.a, c.b) == (a, b):
                out.append(c)
            elif (c.a, c.b) == (b, a):
                out.append(Coupling(c.b, c.b_pair, c.a, c.a_pair, c.chroma, -c.lag_s,
                                    c.block, c.amp))
        return out

    def block(self, name: str) -> tuple[float, float]:
        onset, span = next((on, dur) for n, on, dur in self.blocks if n == name)
        return onset, onset + span


def _band_noise(rng, n: int, band) -> np.ndarray:
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n, 1 / SFREQ)
    spectrum[(freqs < band[0]) | (freqs > band[1])] = 0
    out = np.fft.irfft(spectrum, n)
    return out / out.std()


def _gate(t_al: np.ndarray, block: str | None) -> np.ndarray:
    if block is None:
        return np.ones_like(t_al)
    onset, span = next((on, dur) for n, on, dur in BLOCKS if n == block)
    boxcar = ((t_al >= onset) & (t_al < onset + span)).astype(float)
    # edges smoothed over 10 s, so switching the coupling on adds no broadband step
    kernel = np.hanning(int(10 * SFREQ))
    return np.convolve(boxcar, kernel / kernel.sum(), mode="same")


def member_raw(member: Member) -> tuple[mne.io.Raw, dict[str, np.ndarray]]:
    rng = np.random.default_rng(zlib.crc32(f"dyad/{member.subject}".encode()))
    n = int(round(SFREQ * (member.offset + DURATION + member.tail)))
    t = np.arange(n) / SFREQ
    t_al = t - member.offset
    layout = _layout(N_LONG, 0 if member.subject in NO_SHORT else N_SHORT)
    info = _info(layout)

    cardiac = CARDIAC_AMP * np.sin(2 * np.pi * CARDIAC_FREQ * t)
    mark = HBR_MARK_AMP * np.sin(2 * np.pi * hbr_mark_freq(member.subject) * t_al)
    haemo: dict[str, np.ndarray] = {}
    for k, (name, short, _, _) in enumerate(layout):
        if short:
            haemo[f"{name} hbo"] = (LONG_DISTANCE / SHORT_DISTANCE - 1) * cardiac
            haemo[f"{name} hbr"] = np.zeros(n)
            continue
        own = LONG_AMP * np.sin(2 * np.pi * own_freq(member.subject, k) * t_al + k)
        haemo[f"{name} hbo"] = (own + cardiac
                                + NOISE_AMP * _band_noise(rng, n, NOISE_BAND))
        haemo[f"{name} hbr"] = (-0.3 * own + mark
                                + NOISE_AMP * _band_noise(rng, n, NOISE_BAND))

    for c in COUPLINGS:
        for subject, pair, delay in ((c.a, c.a_pair, 0.0), (c.b, c.b_pair, c.lag_s)):
            if subject == member.subject:
                wave = c.shared(t_al - delay)
                haemo[f"{pair} {c.chroma}"] = haemo[f"{pair} {c.chroma}"] + wave * _gate(t_al, c.block)

    to_od = _haemo_to_od(info, DPF)
    od = np.empty((len(info["ch_names"]), n))
    for k, (name, _, _, _) in enumerate(layout):
        od[2 * k: 2 * k + 2] = to_od[k] @ np.vstack([haemo[f"{name} hbo"], haemo[f"{name} hbr"]])
        if name == member.rejected:
            od[2 * k: 2 * k + 2] = 0.01 * rng.normal(size=(2, n))
    od += member.noise * OD_NOISE * rng.normal(size=od.shape)
    if member.subject in CONDITION_FAILING:
        name, onset, span = CONDITION_FAILING[member.subject]
        k = [l[0] for l in layout].index(name)
        off = (t_al >= onset) & (t_al < onset + span)
        anti = DEAD_OD * np.sin(2 * np.pi * CARDIAC_FREQ * t[off])
        od[2 * k, off] += anti
        od[2 * k + 1, off] -= anti
    if member.subject in DEAD:
        k = [l[0] for l in layout].index(member.rejected)
        od[2 * k: 2 * k + 2] = 0.0

    bump = MOTION_OD * np.hanning(int(2 * SFREQ))
    for at in (OWN_SPIKE[member.subject], SHARED_SPIKE, *EXTRA_SPIKES.get(member.subject, ())):
        i = int(round((at + member.offset) * SFREQ))
        od[:, i: i + len(bump)] += bump

    intensity0 = 0.05 + 0.01 * np.arange(len(info["ch_names"]))[:, None]
    raw = mne.io.RawArray(intensity0 * np.exp(-od), info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.info["subject_info"] = {"first_name": "sub", "last_name": member.subject}
    events = [(member.offset, 0.0, SYNC)] + [(member.offset + on, dur, name)
                                            for name, on, dur in BLOCKS]
    raw.set_annotations(mne.Annotations([e[0] for e in events], [e[1] for e in events],
                                        [e[2] for e in events]))
    aligned = {ch: v[(t_al >= 0) & (t_al < DURATION)] for ch, v in haemo.items()}
    return raw, aligned


def make_group_dataset(root: Path, name: str = "bids_dyad") -> tuple[Path, Path, GroupTruth]:
    """Two groups (a dyad and a triad) as one BIDS dataset, plus the pairs CSV."""
    members = [Member(*m) for m in MEMBERS + COHORT_ONLY]
    bids_dir = Path(root) / name
    _write_dataset_root(bids_dir, [m.subject for m in members])
    layout = _layout(N_LONG, N_SHORT)
    truth = GroupTruth(members=members, couplings=COUPLINGS,
                       long_pairs=[l[0] for l in layout if not l[1]],
                       short_pairs=[l[0] for l in layout if l[1]])
    for m in members:
        raw, haemo = member_raw(m)
        _write_subject(bids_dir, m.subject, TASK, raw)
        truth.haemo[m.sid] = haemo
    pairs_csv = Path(root) / f"{name}_pairs.csv"
    rows = ["group_id,subject_id,task"] + [f"{m.group},{m.sid},{TASK}" for m in members
                                           if m.group]
    pairs_csv.write_text("\n".join(rows) + "\n")
    return bids_dir, pairs_csv, truth


ROI_MAP = {"L": ["S1_D1", "S2_D2"], "R": ["S3_D3", "S4_D4"], "M": ["S5_D5"]}


def write_roi_map(root: Path) -> Path:
    path = Path(root) / "roi_dyad.json"
    path.write_text(json.dumps(ROI_MAP))
    return path
