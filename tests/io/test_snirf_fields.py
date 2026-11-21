"""Every SNIRF the package writes keeps its source's probe and metadata fields, but for the few it must rewrite."""
import h5py
import mne
import numpy as np
import pandas as pd

from nirspipe.io.derivatives import write_dataset_description
from nirspipe.io.snirf import read_raw_snirf
from nirspipe.pipeline.edit_markers import apply_markers_from_df
from nirspipe.pipeline.post_pipeline import PostConfig, run_post
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from tests._synth import make_bids_dataset

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)
_SOURCE_LABELS = [b"F7", b"F5", b"F3", b"FC5", b"FT7"]
_DETECTOR_LABELS = [b"AFF5h", b"FFT7h", b"FFC5h", b"CP5", b"TTP7h"]
_TAGS = {"MeasurementDate": b"2023-11-22", "MeasurementTime": b"14:19:51+01",
         "ManufacturerName": b"Some Maker", "Model": b"Some Cap"}
# the data beside them is written from the Raw, so these say what it is, not what the source was
_REWRITTEN = {"metaDataTags/LengthUnit", "metaDataTags/TimeUnit", "metaDataTags/FrequencyUnit",
              "metaDataTags/SubjectID", "probe/wavelengths"}


def _source(tmp_path):
    """A recording carrying 10-10 optode labels, a date with an hours-only offset, two maker tags, a per-sample tag and its stim group stored twice."""
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",), name="raw")
    path = bids / "sub-01" / "nirs" / "sub-01_task-tapping_nirs.snirf"
    with h5py.File(path, "r+") as f:
        probe, tags = f["nirs/probe"], f["nirs/metaDataTags"]
        for key, labels in (("sourceLabels", _SOURCE_LABELS), ("detectorLabels", _DETECTOR_LABELS)):
            assert len(probe[key]) == len(labels)
            del probe[key]
            probe.create_dataset(key, data=labels)
        for key, value in _TAGS.items():
            if key in tags:
                del tags[key]
            tags.create_dataset(key, data=[value])
        tags.create_dataset("missing_sample", data=np.zeros(len(f["nirs/data1/time"])))
        f.copy(f["nirs/stim1"], f["nirs"], name="stim01")
    return path


def _fields(path) -> dict:
    with h5py.File(path, "r") as f:
        out = {}
        for group in ("probe", "metaDataTags"):
            for key in f[f"nirs/{group}"]:
                out[f"{group}/{key}"] = f[f"nirs/{group}/{key}"][()]
        return out


def _stims(path) -> list[tuple[str, tuple]]:
    with h5py.File(path, "r") as f:
        return [(np.ravel(f[f"nirs/{k}/name"][()])[0], tuple(map(tuple, np.atleast_2d(f[f"nirs/{k}/data"][()])[:, :2])))
                for k in f["nirs"] if k.startswith("stim")]


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "SOU" or b.dtype.kind in "SOU":
        return [bytes(x) if isinstance(x, (bytes, np.bytes_)) else str(x).encode() for x in a.ravel()] == \
               [bytes(x) if isinstance(x, (bytes, np.bytes_)) else str(x).encode() for x in b.ravel()]
    return a.shape == b.shape and np.allclose(a, b, equal_nan=True)


def _written(tmp_path, source) -> list:
    out = tmp_path / "out"
    write_dataset_description(out, source=source.parents[2])
    raw = read_raw_snirf(source, preload=True)
    result = run_prep(raw, PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.0,
                                      motion_correction="none", **_BANDS),
                      out, source_entities={"task": "tapping"}, source_path=source)
    nirs_dir = out / "sub-01" / "nirs"
    post = PostConfig(subject="01", high_pass=0.01, drift_model="polynomial", drift_order=1,
                      short_channel="mean", **_BANDS)
    run_post(result.raw_haemo.copy(), post, out, mode="denoise", source_entities={"task": "tapping"},
             source_path=nirs_dir / "sub-01_task-tapping_desc-preproc_nirs.snirf")
    events = pd.DataFrame({"onset": raw.annotations.onset, "duration": raw.annotations.duration,
                           "trial_type": raw.annotations.description})
    edited = apply_markers_from_df(source, tmp_path / "edited", "01", None, events)
    stages = sorted(nirs_dir.glob("*_desc-*_nirs.snirf"))
    assert {"od", "preproc", "filtered", "errts"} <= {p.name.split("desc-")[1][:-11] for p in stages}
    return [*stages, edited]


def test_every_written_snirf_keeps_the_sources_fields_but_those_it_must_rewrite(tmp_path):
    source = _source(tmp_path)
    expected = _fields(source)
    for path in _written(tmp_path, source):
        written = _fields(path)
        lost = [k for k in expected if k not in written and k != "metaDataTags/missing_sample"]
        changed = [k for k in expected if k in written and k not in _REWRITTEN
                   and not _same(expected[k], written[k])]
        assert not lost and not changed, (path.name, lost, changed)
        assert "metaDataTags/missing_sample" not in written, path.name


def test_a_stim_group_stored_twice_is_written_once(tmp_path):
    source = _source(tmp_path)
    groups = _stims(source)
    assert len(groups) == 2 and groups[0] == groups[1]
    for path in _written(tmp_path, source):
        assert _stims(path) == groups[:1], path.name


def test_an_event_listed_twice_is_read_once(tmp_path):
    source = _source(tmp_path)
    doubled = mne.io.read_raw_snirf(str(source), verbose="error").annotations
    assert len(read_raw_snirf(source).annotations) == len(doubled) // 2

