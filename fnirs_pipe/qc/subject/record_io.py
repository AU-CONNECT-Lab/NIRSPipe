"""A quality record on disk: the JSON keeps its scalars, its arrays sit in TSV tables beside it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.io.naming import record_table_name
from fnirs_pipe.io.tables import NA, write_tsv
from fnirs_pipe.qc.boilerplate.vocabulary import (
    RECORD_CHANNEL_COLUMNS, RECORD_CHANNEL_METRICS, RECORD_SECTIONS, RECORD_WINDOW_COLUMNS,
)
from fnirs_pipe.qc.metrics.imu import IMU_QUANTITIES

MATRIX_STATS = ("sci", "psp", "cv")
SUMMARY_STAT = "summary"
CHANNEL_COLUMNS = ("name", "section", "metric", "value")


def _label_and_desc(path: Path) -> tuple[str, str]:
    # "sub-01_task-rest_desc-sqm_qc.json" -> ("sub-01_task-rest", "sqm")
    label, _, desc = Path(path).name.removesuffix("_qc.json").rpartition("_desc-")
    return label, desc


def _dumps(obj: Any, depth: int = 0) -> str:
    """JSON with objects indented and every array on one line."""
    if isinstance(obj, dict) and obj:
        pad = "  " * (depth + 1)
        body = ",\n".join(f"{pad}{json.dumps(str(key))}: {_dumps(value, depth + 1)}"
                          for key, value in obj.items())
        return "{\n" + body + "\n" + "  " * depth + "}"
    return json.dumps(obj, default=str)


def _is_window_series(key: str) -> bool:
    return "_per_window" in key


def _window_length(windowed: dict) -> "float | None":
    """Seconds per window, off the stored window edges, else off the centres."""
    for key in ("sci_times", "psp_times", "cv_times"):
        if windowed.get(key):
            start, stop = windowed[key][0]
            return float(stop) - float(start)
    for key, centres in windowed.items():
        # every grid starts at the recording's first sample, so the first centre is half a window
        if key.endswith("_window_times_s") and centres:
            return 2.0 * float(centres[0])
    return None


def _described(descriptions: dict, keys, units: "dict | None" = None) -> dict:
    out = {}
    for key in keys:
        entry = {"Description": descriptions[key]} if key in descriptions else {}
        if units and units.get(key):
            entry["Units"] = units[key]
        if entry:
            out[key] = entry
    return out


def _matrix_tables(windowed: dict, label: str, desc: str, header: dict) -> dict:
    tables = {}
    for stat in MATRIX_STATS:
        matrix = windowed.pop(f"{stat}_matrix", None)
        names = windowed.pop(f"{stat}_channels", None)
        if matrix is None:
            continue
        if not names:
            raise ValueError(f"{stat}_matrix has no channel names to head its columns")
        frame = pd.DataFrame(np.asarray(matrix, dtype=float).T, columns=names)
        tables[record_table_name(label, desc, "timeseries", statistic=stat)] = (frame, header)
    return tables


def _summary_table(windowed: dict, label: str, desc: str, header: dict, imu: dict) -> dict:
    series = {key: windowed.pop(key) for key in [k for k in windowed if _is_window_series(k)]}
    if not series:
        return {}
    lengths = {len(values) for values in series.values()}
    if len(lengths) > 1:
        raise ValueError(f"per-window series of unequal length {sorted(lengths)}; "
                         "they cannot share one table")
    frame = pd.DataFrame({key: np.asarray(values, dtype=float) for key, values in series.items()})
    units = {f"{name}{part}_per_window": imu[f"{name}_unit"]
             for name in IMU_QUANTITIES if imu.get(f"{name}_unit") for part in ("", "_p95")}
    sidecar = {**header, **_described(RECORD_WINDOW_COLUMNS, frame.columns, units)}
    return {record_table_name(label, desc, "timeseries", statistic=SUMMARY_STAT): (frame, sidecar)}


def _channel_table(per_channel: dict, label: str, desc: str, header: dict) -> dict:
    rows = [(name, section, metric, value)
            for section, metrics in per_channel.items()
            for metric, values in metrics.items()
            for name, value in values.items()]
    if not rows:
        return {}
    frame = pd.DataFrame(rows, columns=list(CHANNEL_COLUMNS))
    columns = _described(RECORD_CHANNEL_COLUMNS, CHANNEL_COLUMNS)
    columns["section"]["Levels"] = _described_levels(RECORD_SECTIONS, frame["section"])
    columns["metric"]["Levels"] = _described_levels(RECORD_CHANNEL_METRICS, frame["metric"])
    return {record_table_name(label, desc, "nirsmap"): (frame, {**header, **columns})}


def _described_levels(descriptions: dict, values: pd.Series) -> dict:
    return {v: descriptions[v] for v in dict.fromkeys(values) if v in descriptions}


def write_record(path: Path, record: dict[str, Any]) -> Path:
    """Write the record's scalars to ``path`` and its arrays to the tables beside it."""
    path = Path(path)
    label, desc = _label_and_desc(path)
    record = dict(record)
    windowed = dict(record.get("windowed") or {})
    per_channel = record.pop("per_channel", None) or {}

    header: dict[str, Any] = {"Sources": record.get("Sources") or []}
    step = _window_length(windowed)
    if step:
        header.update(SamplingFrequency=1.0 / step, WindowLength=step)

    tables = {**_matrix_tables(windowed, label, desc, header),
              **_summary_table(windowed, label, desc, header, record.get("imu") or {}),
              **_channel_table(per_channel, label, desc, {"Sources": header["Sources"]})}

    if "windowed" in record:
        record["windowed"] = windowed
    # the reader's list: a table missing from disk is an error, one never written is not
    record["data"] = {**(record.get("data") or {}), "tables": sorted(tables)}

    path.parent.mkdir(parents=True, exist_ok=True)
    for name, (frame, sidecar) in tables.items():
        write_tsv(frame, path.parent / name)
        (path.parent / name).with_suffix(".json").write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    path.write_text(_dumps(record) + "\n", encoding="utf-8")
    return path


def _read_table(path: Path, text_columns=()) -> pd.DataFrame:
    # round_trip: the default parser drops the last digit pandas wrote
    return pd.read_csv(path, sep="\t", na_values=[NA], keep_default_na=False,
                       float_precision="round_trip",
                       dtype={column: str for column in text_columns})


def read_record(path: Path) -> dict[str, Any]:
    """The record with its tables folded back in, in the shape the writer was handed."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    tables = (record.get("data") or {}).get("tables")
    if tables is None:
        raise StageError(f"{path.name} keeps its arrays inline, which this version no longer "
                         "reads; rerun the command that wrote it.")

    label, desc = _label_and_desc(path)
    kinds = {record_table_name(label, desc, "timeseries", statistic=stat): stat
             for stat in (*MATRIX_STATS, SUMMARY_STAT)}
    kinds[record_table_name(label, desc, "nirsmap")] = "channels"

    for name in tables:
        kind = kinds.get(name)
        if kind is None:
            raise StageError(f"{path.name} lists {name}, which is no table of its run")
        table = path.parent / name
        if not table.exists():
            raise StageError(f"{path.name} lists {name}, which is missing beside it")
        if kind == "channels":
            frame = _read_table(table, text_columns=("name", "section", "metric"))
            per_channel: dict = {}
            for ch, section, metric, value in frame.itertuples(index=False):
                per_channel.setdefault(section, {}).setdefault(metric, {})[ch] = float(value)
            record["per_channel"] = per_channel
            continue
        frame = _read_table(table)
        windowed = record.setdefault("windowed", {})
        if kind == SUMMARY_STAT:
            windowed.update({key: frame[key].astype(float).tolist() for key in frame.columns})
        else:
            windowed[f"{kind}_matrix"] = frame.to_numpy(dtype=float).T.tolist()
            windowed[f"{kind}_channels"] = list(frame.columns)
    return record
