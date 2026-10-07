"""One finished fingerprint run: where its figures and stage files are, and the truth it was built from."""

from dataclasses import dataclass, field
from pathlib import Path

import mne

from tests._fingerprint import DPF, Truth


@dataclass
class Run:
    out: Path
    truth: Truth
    subject: str = "01"
    task: str = "tapping"
    captured: dict = field(default_factory=dict, repr=False)
    _stages: dict = field(default_factory=dict, repr=False)

    @property
    def figures(self) -> Path:
        return self.out / f"sub-{self.subject}" / "figures"

    @property
    def nirs(self) -> Path:
        return self.out / f"sub-{self.subject}" / "nirs"

    def figure(self, desc: str, chan: str | None = None, suffix: str = "nirs",
               entities: str = "") -> Path:
        """``entities`` are the ones between the run and ``desc``, e.g. ``stat-alff``."""
        middle = "".join(f"_{part}" for part in (f"chan-{chan}" if chan else "", entities) if part)
        return self.figures / f"sub-{self.subject}_task-{self.task}{middle}_desc-{desc}_{suffix}.html"

    @property
    def report(self) -> Path:
        return self.out / f"sub-{self.subject}" / f"sub-{self.subject}_task-{self.task}_report.html"

    def table(self, suffix: str = "_desc-channel_qc.tsv") -> Path:
        return self.nirs / f"sub-{self.subject}_task-{self.task}{suffix}"

    def record(self) -> dict:
        from fnirs_pipe.qc.subject.record_io import read_record
        return read_record(self.nirs / f"sub-{self.subject}_task-{self.task}_desc-sqm_qc.json")

    def stage(self, desc: str) -> Path:
        return self.nirs / f"sub-{self.subject}_task-{self.task}_desc-{desc}_nirs.snirf"

    def read(self, desc: str) -> mne.io.Raw:
        """A stage file, or ``uncorrected``: Beer-Lambert on desc-sci, which no file holds."""
        if desc not in self._stages:
            if desc == "uncorrected":
                from mne.preprocessing.nirs import beer_lambert_law
                raw = beer_lambert_law(self.read("sci").copy(), ppf=list(DPF))
            else:
                raw = mne.io.read_raw_snirf(self.stage(desc), verbose="error").load_data()
            self._stages[desc] = raw
        return self._stages[desc]

    def channel(self, desc: str, name: str):
        raw = self.read(desc)
        return raw.times, raw.get_data(picks=[name])[0]
