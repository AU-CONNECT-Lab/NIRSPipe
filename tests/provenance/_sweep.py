"""What a run wrote, reduced to digests a flag sweep can compare between two runs."""

import gzip
import hashlib
import json
import tomllib
from pathlib import Path

import h5py
import numpy as np

from tests.figure_accuracy._payload import plotly_figures


def argv(settings: dict, paths: dict) -> list[str]:
    """``{"--flag": ["{name}"]}`` -> ``["--flag", paths["name"]]``; a None value drops the flag."""
    out = []
    for flag, values in settings.items():
        if values is not None:
            out += [flag, *(v.format(**paths) for v in values)]
    return out


def stage(name: str) -> str:
    """``..._desc-errts_nirs.json`` -> ``errts``; ``..._stat-wtc_relmat.json`` -> ``wtc``; else the suffix."""
    stem = name.split(".")[0]
    for entity in ("desc", "stat"):
        if f"_{entity}-" in stem:
            return stem.split(f"_{entity}-")[1].split("_")[0]
    return stem.rsplit("_", 1)[-1]


def _snirf(path: Path) -> str:
    h = hashlib.sha1()
    with h5py.File(path, "r") as handle:
        def visit(name, item):
            if isinstance(item, h5py.Dataset):
                value = np.asarray(item[()])
                h.update(name.encode())
                h.update(repr(value.tolist()).encode() if value.dtype.kind in "OSU" else value.tobytes())
        handle.visititems(visit)
    return h.hexdigest()


def _npz(path: Path) -> str:
    h = hashlib.sha1()
    with np.load(path, allow_pickle=False) as arrays:
        for key in sorted(arrays.files):
            h.update(key.encode())
            h.update(np.ascontiguousarray(arrays[key]).tobytes())
    return h.hexdigest()


def snapshot(root: Path, record: "Path | None" = None, sections: "dict | None" = None) -> dict:
    """Digests of every output under ``root``, each sidecar's parameters by stage, and the record.

    ``sections`` picks the record's sections, ``{name: None}`` for all of one's keys or
    ``{name: (key, ...)}`` for some. Logs are left out; figures count by their Plotly data,
    since the div ids in the HTML change from run to run.
    """
    outputs, params = {}, {}
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root).as_posix()
        if "logs" in f.relative_to(root).parts or f.suffix == ".log":
            continue
        # the provenance graph restates the sidecars; a report page restates its figures
        in_figures = "figures" in f.relative_to(root).parts
        if f.suffix == ".html" or in_figures:
            if in_figures and f.suffix == ".html":
                outputs[rel] = hashlib.sha1(json.dumps(
                    [fig.get("data") for fig in plotly_figures(f)], sort_keys=True, default=str
                ).encode()).hexdigest()
            elif in_figures and f.suffix == ".png" and "_desc-provenance" not in f.name:
                outputs[rel] = hashlib.sha1(f.read_bytes()).hexdigest()
            continue
        if f.suffix == ".snirf":
            outputs[rel] = _snirf(f)
        elif f.suffix == ".npz":
            outputs[rel] = _npz(f)
        elif f.suffix == ".json":
            data = json.loads(f.read_text(encoding="utf-8"))
            data.pop("timestamp", None)
            if "parameters" in data:
                params.setdefault(stage(f.name), {})[rel] = data.pop("parameters")
            outputs[rel] = json.dumps(data, sort_keys=True, default=str)
        elif f.suffix == ".gz":
            outputs[rel] = hashlib.sha1(gzip.decompress(f.read_bytes())).hexdigest()
        else:
            outputs[rel] = hashlib.sha1(f.read_bytes()).hexdigest()
    flat = {}
    if record is not None:
        loaded = tomllib.loads(record.read_text(encoding="utf-8"))
        for section, keys in (sections or {}).items():
            for key, value in loaded.get(section, {}).items():
                if keys is None or key in keys:
                    flat[f"{section}.{key}"] = value
    return {"outputs": outputs, "params": params, "record": flat}


def moved(base: dict, changed: dict) -> list[str]:
    return sorted(f for f in set(base["outputs"]) | set(changed["outputs"])
                  if base["outputs"].get(f) != changed["outputs"].get(f))
