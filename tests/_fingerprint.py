"""A synthetic recording whose every channel, chromophore, stage and moment is identifiable."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import zlib

import mne
import numpy as np

from tests._synth import WAVELENGTHS, _write_dataset_root, _write_subject

SFREQ = 10.0
DURATION = 400.0
HEAD_RADIUS = 0.09
LONG_DISTANCE = 0.03
SHORT_DISTANCE = 0.008
DPF = (5.8, 6.4)                     # two values, so a reversed list changes the haemoglobin

EVENT_ONSETS = (40.0, 100.0, 160.0, 220.0, 280.0, 340.0)
EVENT_DURATION = 10.0
# three trials 60 s apart in each block of the two-level design, six in all
_CYCLE = 1 / (3 * 60.0)

# ---- Fingerprint frequencies (Hz) ----
# multiples of _CYCLE but not of 1/60 Hz, so each averages to zero over three trials or six
LONG_HBO_FREQS = tuple(m * _CYCLE for m in (7, 23, 34, 38, 55, 65))
LONG_HBO_AMPS = (0.4e-6, 0.6e-6, 0.8e-6, 1.0e-6, 1.2e-6, 1.4e-6)
# not -0.9 on a kept pair: its HbR would cancel its HbO in the 760 nm optical density
HBO_HBR_CORR = (-0.5, -0.7, -0.9, -0.3, -0.1, 0.3)
HBR_FREQ = 28 * _CYCLE               # every long HbR carries it, no HbO does
SYSTEMIC_FREQ = 17 * _CYCLE          # short and long HbO alike
SYSTEMIC_AMP = 0.8e-6
CARDIAC_FREQ = 244 * _CYCLE
CARDIAC_AMP = 0.3e-6

# ---- Resting state: everything inside the 0.01-0.08 Hz band, and a known network ----
REST_HBO_FREQS = (0.020, 0.026, 0.032, 0.038, 0.044, 0.050)
REST_HBR_FREQ = 0.058
REST_NET_FREQ = 0.066
REST_NET_AMP = 0.8e-6
NETWORK = (1.0, 1.0, 0.0, -1.0, 0.0, 0.0)   # per long pair: in phase, out of phase, or absent

N_SHORT_PAIRS = 2
BAD_PAIR = 2                         # 0-based long pair with uncoupled wavelengths
RESPONDERS = (0, 3)
RESPONSE_AMP = 1.5e-6
SPIKE = (4, 80.0)                    # (long pair, time in s), outside every -5..25 s epoch
STEP = (5, 385.0)                    # past the last epoch: a high-passed step rings for tens of seconds
MOTION_OD = 0.5
DEAD_TRIAL = 3                       # index into EVENT_ONSETS: the wavelengths decouple for its duration
DEAD_OD = 0.06

# ---- Two-level design: condition blocks with the trials inside them ----
# (name, onset, duration, response gain of its trials)
BLOCKS = (("ca", 30.0, 160.0, 1.0), ("cb", 210.0, 160.0, 0.4))
# a long pair whose wavelengths decouple over the second half of block cb only, so it stays
# in the run (0.8 of windows coupled) and in ca, and fails in cb (0.5)
BLOCK_BAD = (5, 290.0, 80.0)

OD_NOISE = 0.002

# Non-default values, so a label written as a literal cannot match by accident.
CLI_ARGS = (
    "--dpf", *(str(v) for v in DPF),
    "--sci-threshold", "0.75",
    "--psp-threshold", "0.12",
    "--min-good-frac", "0.7",
    "--cardiac-l-freq", "0.8", "--cardiac-h-freq", "1.9",
    "--resp-l-freq", "0.15", "--resp-h-freq", "0.45",
    "--window-length", "12",
    "--motion-correction", "tddr",
)


@dataclass(frozen=True)
class Pair:
    name: str                        # "S1_D1"
    short: bool
    position: np.ndarray             # source-detector midpoint, head coordinates (m)
    hbo_freq: float | None = None
    hbo_amp: float = 0.0
    hbo_hbr_corr: float | None = None
    responds: bool = False
    bad: bool = False


@dataclass
class Truth:
    pairs: list[Pair]
    sfreq: float
    duration: float
    events: list[tuple[float, float, str]]
    spike: tuple[str, float] | None
    step: tuple[str, float] | None
    dead_trial: float | None         # onset of the trial whose wavelengths do not couple
    # haemoglobin per channel name ("S1_D1 hbo"), in M, before and after the systemic part
    haemo: dict[str, np.ndarray] = field(repr=False)
    haemo_no_systemic: dict[str, np.ndarray] = field(repr=False)
    blocks: tuple = ()               # BLOCKS, on the two-level design only
    block_bad: tuple | None = None   # (pair, onset, duration) decoupled inside one block

    @property
    def long_pairs(self) -> list[Pair]:
        return [p for p in self.pairs if not p.short]

    @property
    def moved(self) -> set[str]:
        return {art[0] for art in (self.spike, self.step) if art}

    def pair(self, name: str) -> Pair:
        return next(p for p in self.pairs if p.name == name)


def _sphere(point: np.ndarray) -> np.ndarray:
    return HEAD_RADIUS * point / np.linalg.norm(point)


def _layout(n_long: int, n_short: int) -> list[tuple[str, bool, np.ndarray, np.ndarray]]:
    """(name, short, source, detector) per pair: long pairs on a 2 x 3 grid, short beside them."""
    out = []
    for k in range(n_long + n_short):
        short = k >= n_long
        col, row = (k % 3, k // 3) if not short else (k - n_long, 2)
        src = _sphere(np.array([(col - 1) * 0.04, (row - 1) * 0.035, HEAD_RADIUS]))
        tangent = np.array([1.0, 0.0, 0.0]) - src[0] * src / HEAD_RADIUS**2
        tangent /= np.linalg.norm(tangent)
        distance = SHORT_DISTANCE if short else LONG_DISTANCE
        det = _sphere(src + distance * tangent)
        out.append((f"S{k + 1}_D{k + 1}", short, src, det))
    return out


def _locs(src: np.ndarray, det: np.ndarray, wavelength: float) -> np.ndarray:
    loc = np.zeros(12)
    loc[0:3] = (src + det) / 2
    loc[3:6] = src
    loc[6:9] = det
    loc[9] = wavelength
    return loc


def _info(layout) -> mne.Info:
    names, locs = [], []
    for name, _, src, det in layout:
        for wavelength in WAVELENGTHS:
            names.append(f"{name} {wavelength:.0f}")
            locs.append(_locs(src, det, wavelength))
    info = mne.create_info(names, SFREQ, ["fnirs_cw_amplitude"] * len(names))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc
    return info


def _haemo_to_od(info: mne.Info, dpf) -> np.ndarray:
    """Per pair, the 2 x 2 matrix taking (HbO, HbR) in M to (OD 760, OD 850)."""
    from mne.preprocessing.nirs import beer_lambert_law

    od_info = info.copy()
    for ch in od_info["chs"]:
        ch["kind"] = mne.io.constants.FIFF.FIFFV_FNIRS_CH
        ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_FNIRS_OD
    unit = np.zeros((len(od_info["ch_names"]), 2))
    unit[0::2, 0] = 1.0
    unit[1::2, 1] = 1.0
    od = mne.io.RawArray(unit, od_info, verbose="error")
    haemo = beer_lambert_law(od, ppf=list(dpf)).get_data()
    # rows hbo, hbr per pair; columns the response to unit OD at 760 and at 850
    forward = haemo.reshape(-1, 2, 2)
    return np.linalg.inv(forward)


def _response(t: np.ndarray, gains=None) -> np.ndarray:
    boxcar = np.zeros_like(t)
    for onset, gain in zip(EVENT_ONSETS, gains or [1.0] * len(EVENT_ONSETS)):
        boxcar[(t >= onset) & (t < onset + EVENT_DURATION)] = gain
    kt = np.arange(0, 20, 1 / SFREQ)
    kernel = kt**5 * np.exp(-kt)                 # gamma, peak at 5 s
    kernel /= kernel.sum()
    smooth = np.convolve(boxcar, kernel)[: len(t)]
    return smooth / smooth.max()


def fingerprint_raw(subject: str, task: str, seed: int | None = None,
                    rest: bool = False, blocks: bool = False) -> tuple[mne.io.Raw, Truth]:
    if seed is None:
        seed = zlib.crc32(f"fingerprint/{subject}/{task}".encode())
    rng = np.random.default_rng(seed)
    n = int(SFREQ * DURATION)
    t = np.arange(n) / SFREQ
    freqs, hbr_freq = (REST_HBO_FREQS, REST_HBR_FREQ) if rest else (LONG_HBO_FREQS, HBR_FREQ)
    n_long = len(freqs)
    layout = _layout(n_long, N_SHORT_PAIRS)
    info = _info(layout)

    systemic = SYSTEMIC_AMP * np.sin(2 * np.pi * SYSTEMIC_FREQ * t)
    cardiac = CARDIAC_AMP * np.sin(2 * np.pi * CARDIAC_FREQ * t)
    hbr_mark = np.sin(2 * np.pi * hbr_freq * t)
    gains = [next(g for _, on, dur, g in BLOCKS if on <= o < on + dur) for o in EVENT_ONSETS]         if blocks else None
    response = np.zeros(n) if rest else _response(t, gains)
    network = REST_NET_AMP * np.sin(2 * np.pi * REST_NET_FREQ * t) if rest else np.zeros(n)
    responders = () if rest else RESPONDERS

    pairs: list[Pair] = []
    haemo: dict[str, np.ndarray] = {}
    clean: dict[str, np.ndarray] = {}
    for k, (name, short, src, det) in enumerate(layout):
        if short:
            pairs.append(Pair(name, True, (src + det) / 2))
            # optical density scales with separation, so a short pair needs more pulse to couple
            hbo_own = (LONG_DISTANCE / SHORT_DISTANCE - 1) * cardiac
            # no HbR mark here, or short-channel regression would strip it from the long pairs
            hbr_own = np.zeros(n)
        else:
            f, a, r = freqs[k], LONG_HBO_AMPS[k], HBO_HBR_CORR[k]
            pairs.append(Pair(name, False, (src + det) / 2, f, a, r,
                              responds=k in responders, bad=k == BAD_PAIR))
            own = np.sin(2 * np.pi * f * t + k)
            hbo_own = a * own
            # own and hbr_mark are orthogonal, so corr(HbO, HbR) = r once systemic is regressed out
            hbr_own = 0.4 * a * (r * own + np.sqrt(1 - r**2) * hbr_mark)
            hbo_own = hbo_own + NETWORK[k] * network
            if k in responders:
                hbo_own = hbo_own + RESPONSE_AMP * response
                hbr_own = hbr_own - RESPONSE_AMP / 3 * response
        clean[f"{name} hbo"] = hbo_own
        clean[f"{name} hbr"] = hbr_own
        haemo[f"{name} hbo"] = hbo_own + systemic + cardiac
        haemo[f"{name} hbr"] = hbr_own

    to_od = _haemo_to_od(info, DPF)
    od = np.empty((len(info["ch_names"]), n))
    for k, (name, _, _, _) in enumerate(layout):
        hb = np.vstack([haemo[f"{name} hbo"], haemo[f"{name} hbr"]])
        od[2 * k: 2 * k + 2] = to_od[k] @ hb
        if k == BAD_PAIR:
            # no shared cardiac oscillation between the two wavelengths
            od[2 * k: 2 * k + 2] = 0.01 * rng.normal(size=(2, n))
    od += OD_NOISE * rng.normal(size=od.shape)

    if not rest:
        # a 2 s bump rather than a few samples: the motion figures band-limit to 0.5 Hz first
        spike_pair, spike_t = SPIKE
        bump = MOTION_OD * np.hanning(int(2 * SFREQ))
        i = int(spike_t * SFREQ)
        od[2 * spike_pair: 2 * spike_pair + 2, i: i + len(bump)] += bump
        step_pair, step_t = STEP
        od[2 * step_pair: 2 * step_pair + 2, int(step_t * SFREQ):] += MOTION_OD

        # an anti-phase pulse at the two wavelengths, above every passband in use
        dead = (t >= EVENT_ONSETS[DEAD_TRIAL]) & (t < EVENT_ONSETS[DEAD_TRIAL] + EVENT_DURATION)
        anti = DEAD_OD * np.sin(2 * np.pi * CARDIAC_FREQ * t[dead])
        for k in range(len(layout)):
            if k != BAD_PAIR:
                od[2 * k, dead] += anti
                od[2 * k + 1, dead] -= anti

    if blocks:
        pair, onset, span = BLOCK_BAD
        off = (t >= onset) & (t < onset + span)
        anti = DEAD_OD * np.sin(2 * np.pi * CARDIAC_FREQ * t[off])
        od[2 * pair, off] += anti
        od[2 * pair + 1, off] -= anti

    intensity0 = 0.05 + 0.01 * np.arange(len(info["ch_names"]))[:, None]
    data = intensity0 * np.exp(-od)

    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.info["subject_info"] = {"first_name": "sub", "last_name": subject}

    if rest:
        events = []
    elif blocks:
        # the blocks name the conditions; the trials inside them share one name
        events = sorted([(on, dur, name) for name, on, dur, _ in BLOCKS]
                        + [(o, EVENT_DURATION, "trial") for o in EVENT_ONSETS])
    else:
        events = [(o, EVENT_DURATION, task) for o in EVENT_ONSETS]
    raw.set_annotations(mne.Annotations([e[0] for e in events], [e[1] for e in events],
                                        [e[2] for e in events]))

    names = [p.name for p in pairs]
    truth = Truth(
        pairs=pairs, sfreq=SFREQ, duration=DURATION, events=events,
        spike=None if rest else (names[SPIKE[0]], SPIKE[1]),
        step=None if rest else (names[STEP[0]], STEP[1]),
        dead_trial=None if rest else EVENT_ONSETS[DEAD_TRIAL],
        blocks=BLOCKS if blocks else (),
        block_bad=(names[BLOCK_BAD[0]], *BLOCK_BAD[1:]) if blocks else None,
        haemo=haemo, haemo_no_systemic=clean,
    )
    return raw, truth


def make_fingerprint_dataset(
    root: Path, subject: str = "01", task: str = "tapping", name: str = "bids_fingerprint",
    rest: bool = False, blocks: bool = False, aux: bool = False,
) -> tuple[Path, Truth]:
    bids_dir = Path(root) / name
    _write_dataset_root(bids_dir, [subject])
    raw, truth = fingerprint_raw(subject, task, rest=rest, blocks=blocks)
    path = _write_subject(bids_dir, subject, task, raw)
    if aux:
        _add_aux(path)
    return bids_dir, truth


AUX_NAME = "ACC_X"
AUX_FS = 50.0


def _add_aux(path: Path) -> None:
    """One accelerometer axis in the file's aux group, noise unrelated to anything planted."""
    import h5py

    values = np.random.default_rng(7).standard_normal(int(DURATION * AUX_FS))
    with h5py.File(path, "a") as handle:
        group = handle["nirs"].create_group("aux1")
        group.create_dataset("name", data=AUX_NAME.encode())
        group.create_dataset("dataTimeSeries", data=values)
        group.create_dataset("time", data=np.arange(len(values)) / AUX_FS)
        group.create_dataset("dataUnit", data=b"m/s^2")
