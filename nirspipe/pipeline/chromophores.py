"""The chromophores an analysis writes, and HbT, made from the HbO-HbR pair where an analysis reads it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from nirspipe.utils import pair_of

if TYPE_CHECKING:
    import mne

CHROMOPHORES = ("hbo", "hbr", "hbt")
DEFAULT_CHROMOPHORES = ("hbo", "hbr")


def in_order(chosen: "Iterable[str]") -> tuple[str, ...]:
    """The chosen chromophores once each, in the order hbo, hbr, hbt."""
    chosen = set(chosen)
    unknown = sorted(chosen - set(CHROMOPHORES))
    if unknown or not chosen:
        raise ValueError(f"chromophores must be some of {CHROMOPHORES}, got {sorted(chosen)}")
    return tuple(c for c in CHROMOPHORES if c in chosen)


def chroma_of(ch_name: str) -> str:
    """The chromophore a channel's name ends with: "S1_D1 hbt" -> "hbt"."""
    return str(ch_name).rsplit(" ", 1)[-1]


def hbt_names(names: "Iterable[str]") -> list[str]:
    """The HbT channel of every pair both of whose halves are listed: "S1_D1 hbo", "S1_D1 hbr" -> "S1_D1 hbt"."""
    names = list(names)
    listed = set(names)
    return [f"{pair_of(n)} hbt" for n in names
            if chroma_of(n) == "hbo" and f"{pair_of(n)} hbr" in listed]


def hbt_of(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """HbO + HbR of each pair as one channel "<pair> hbt", typed hbo as MNE has no HbT type, bad where either half is."""
    names = set(raw.ch_names)
    pairs = [pair_of(n) for n in raw.ch_names
             if chroma_of(n) == "hbo" and f"{pair_of(n)} hbr" in names]
    out = raw.copy().pick([f"{p} hbo" for p in pairs]).load_data()
    out._data += raw.get_data(picks=[f"{p} hbr" for p in pairs])
    bads = set(raw.info["bads"])
    out.rename_channels({f"{p} hbo": f"{p} hbt" for p in pairs})
    out.info["bads"] = [f"{p} hbt" for p in pairs if {f"{p} hbo", f"{p} hbr"} & bads]
    return out


def chroma_channels(raw: mne.io.BaseRaw, chroma: str) -> mne.io.BaseRaw:
    """A copy of ``raw`` holding one chromophore's channels, HbT made from the pair."""
    if chroma == "hbt":
        return hbt_of(raw)
    return raw.copy().pick([ch for ch in raw.ch_names if chroma_of(ch) == chroma])


def with_hbt(raw: mne.io.BaseRaw, chromophores: "Iterable[str]") -> mne.io.BaseRaw:
    """A copy of ``raw`` with :func:`hbt_of` added when ``chromophores`` holds hbt, else ``raw`` itself; never written to a file."""
    if "hbt" not in chromophores:
        return raw
    return raw.copy().add_channels([hbt_of(raw)])
