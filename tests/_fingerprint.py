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
_CYCLE = 1 / (len(EVENT_ONSETS) * 60.0)

# ---- Fingerprint frequencies (Hz) ----
# multiples of _CYCLE but not of 1/60 Hz, so each averages to zero over the six trials
LONG_HBO_FREQS = tuple(m * _CYCLE for m in (13, 45, 67, 77, 110, 131))
LONG_HBO_AMPS = (0.4e-6, 0.6e-6, 0.8e-6, 1.0e-6, 1.2e-6, 1.4e-6)
HBO_HBR_CORR = (-0.9, -0.7, -0.5, -0.3, -0.1, 0.3)
HBR_FREQ = 56 * _CYCLE               # every long HbR carries it, no HbO does
SYSTEMIC_FREQ = 34 * _CYCLE          # short and long HbO alike
SYSTEMIC_AMP = 0.8e-6
CARDIAC_FREQ = 487 * _CYCLE
CARDIAC_AMP = 0.3e-6

N_SHORT_PAIRS = 2
BAD_PAIR = 2                         # 0-based long pair with uncoupled wavelengths
RESPONDERS = (0, 3)
RESPONSE_AMP = 1.5e-6
SPIKE = (4, 80.0)                    # (long pair, time in s), outside every -5..25 s epoch
STEP = (5, 385.0)                    # past the last epoch: a high-passed step rings for tens of seconds
MOTION_OD = 0.5

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
    spike: tuple[str, float]
    step: tuple[str, float]
    # haemoglobin per channel name ("S1_D1 hbo"), in M, before and after the systemic part
    haemo: dict[str, np.ndarray] = field(repr=False)
    haemo_no_systemic: dict[str, np.ndarray] = field(repr=False)

    @property
    def long_pairs(self) -> list[Pair]:
        return [p for p in self.pairs if not p.short]

    @property
    def moved(self) -> set[str]:
        return {self.spike[0], self.step[0]}

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


def _response(t: np.ndarray) -> np.ndarray:
    boxcar = np.zeros_like(t)
    for onset in EVENT_ONSETS:
        boxcar[(t >= onset) & (t < onset + EVENT_DURATION)] = 1.0
    kt = np.arange(0, 20, 1 / SFREQ)
    kernel = kt**5 * np.exp(-kt)                 # gamma, peak at 5 s
    kernel /= kernel.sum()
    smooth = np.convolve(boxcar, kernel)[: len(t)]
    return smooth / smooth.max()


def fingerprint_raw(subject: str, task: str, seed: int | None = None) -> tuple[mne.io.Raw, Truth]:
    if seed is None:
        seed = zlib.crc32(f"fingerprint/{subject}/{task}".encode())
    rng = np.random.default_rng(seed)
    n = int(SFREQ * DURATION)
    t = np.arange(n) / SFREQ
    n_long = len(LONG_HBO_FREQS)
    layout = _layout(n_long, N_SHORT_PAIRS)
    info = _info(layout)

    systemic = SYSTEMIC_AMP * np.sin(2 * np.pi * SYSTEMIC_FREQ * t)
    cardiac = CARDIAC_AMP * np.sin(2 * np.pi * CARDIAC_FREQ * t)
    hbr_mark = np.sin(2 * np.pi * HBR_FREQ * t)
    response = _response(t)

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
            f, a, r = LONG_HBO_FREQS[k], LONG_HBO_AMPS[k], HBO_HBR_CORR[k]
            pairs.append(Pair(name, False, (src + det) / 2, f, a, r,
                              responds=k in RESPONDERS, bad=k == BAD_PAIR))
            own = np.sin(2 * np.pi * f * t + k)
            hbo_own = a * own
            # own and hbr_mark are orthogonal, so corr(HbO, HbR) = r once systemic is regressed out
            hbr_own = 0.4 * a * (r * own + np.sqrt(1 - r**2) * hbr_mark)
            if k in RESPONDERS:
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

    # a 2 s bump rather than a few samples: the motion figures band-limit to 0.5 Hz first
    spike_pair, spike_t = SPIKE
    bump = MOTION_OD * np.hanning(int(2 * SFREQ))
    i = int(spike_t * SFREQ)
    od[2 * spike_pair: 2 * spike_pair + 2, i: i + len(bump)] += bump
    step_pair, step_t = STEP
    od[2 * step_pair: 2 * step_pair + 2, int(step_t * SFREQ):] += MOTION_OD

    intensity0 = 0.05 + 0.01 * np.arange(len(info["ch_names"]))[:, None]
    data = intensity0 * np.exp(-od)

    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.info["subject_info"] = {"first_name": "sub", "last_name": subject}
    raw.set_annotations(mne.Annotations(
        list(EVENT_ONSETS), [EVENT_DURATION] * len(EVENT_ONSETS), [task] * len(EVENT_ONSETS)))

    names = [p.name for p in pairs]
    truth = Truth(
        pairs=pairs, sfreq=SFREQ, duration=DURATION,
        events=[(o, EVENT_DURATION, task) for o in EVENT_ONSETS],
        spike=(names[spike_pair], spike_t), step=(names[step_pair], step_t),
        haemo=haemo, haemo_no_systemic=clean,
    )
    return raw, truth


def make_fingerprint_dataset(
    root: Path, subject: str = "01", task: str = "tapping", name: str = "bids_fingerprint",
) -> tuple[Path, Truth]:
    bids_dir = Path(root) / name
    _write_dataset_root(bids_dir, [subject])
    raw, truth = fingerprint_raw(subject, task)
    _write_subject(bids_dir, subject, task, raw)
    return bids_dir, truth
