# fnirs-pipe

A BIDS-compatible fNIRS preprocessing and postprocessing pipeline.

## Overview

`fnirs-pipe` is split into two stages:

| Stage | What it does |
|-------|--------------|
| `prep` | Fixed-order preprocessing: OD conversion → SCI pruning → motion detection/correction → Beer-Lambert |
| `post` | Flexible postprocessing via modes: GLM, denoise, or custom YAML |

Outputs follow the [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) spec. Each subject gets an HTML QC report with figures and an auto-generated Methods paragraph.

## Requirements

- Python ≥ 3.10
- Core dependencies: `mne`, `mne-nirs`, `nilearn`, `scipy`, `pybids`, `h5py`, `pandas`, `jinja2`, `plotly`, `rich`

## Installation

```bash
git clone <repo>
cd fNIRS_pipe
pip install -e ".[dev]"
```

## Quick Start

```bash
# Preprocessing only
fnirs-pipe /data/bids /data/derivatives participant \
  --participant-label 01 02 \
  --dpf 6.0 \
  --sci-threshold 0.8

# Preprocessing + GLM
fnirs-pipe /data/bids /data/derivatives participant \
  --participant-label 01 \
  --dpf 6.0 --sci-threshold 0.8 \
  --mode glm \
  --hrf-model spm --noise-model ar1 \
  --high-pass 0.01 --low-pass 0.5 \
  --short-channel mean \
  --events-path /data/bids/sub-01/func/sub-01_task-tapping_events.tsv

# FIR GLM
fnirs-pipe /data/bids /data/derivatives participant \
  --participant-label 01 \
  --dpf 6.0 --sci-threshold 0.8 \
  --mode glm --hrf-model fir --fir-delays "0,1,2,3,4,5,6,7,8,9"

# Children dataset (higher cardiac band)
fnirs-pipe /data/bids /data/derivatives participant \
  --participant-label 01 \
  --dpf 5.5 --sci-threshold 0.8 \
  --cardiac-l-freq 1.0 --cardiac-h-freq 2.5

# Parallel subjects, custom TOML config for post
fnirs-pipe /data/bids /data/derivatives participant \
  --participant-label 01 02 03 \
  --dpf 6.0 --sci-threshold 0.8 \
  --mode glm --config glm_params.toml \
  --n-jobs 4
```

After preprocessing, launch the interactive QC rating interface:

```bash
fnirs-rate /data/derivatives --participant-label 01 02
```

Opens a local browser UI for rating each subject's QC report section by section. Ratings are saved to `sub-<id>/figures/sub-<id>_ratings.toml` and appended to `group_ratings.jsonl`.

To merge run logs and IQM metrics into a queryable SQLite database:

```bash
fnirs-log merge /data/derivatives
```

Each `fnirs-pipe` run writes JSONL event files to `logs/json/`. `fnirs-log merge` consolidates them into `logs/fnirs_pipe.db` (tables: `pipeline_executions`, `runs`, `iqm`, `command_outputs`).

To convert raw scanner files to BIDS format, use the separate conversion tool:

```bash
fnirs-recon /raw/sub-01.snirf /data/bids --participant-label 01 --task tapping
```


## CLI Reference

```
fnirs-pipe bids_dir output_dir {participant,group} [OPTIONS]

Required:
  --dpf FLOAT [FLOAT ...]      Differential pathlength factor. One value or one per wavelength.
  --sci-threshold FLOAT        SCI threshold for bad channel detection (e.g. 0.8).

Subject / session / task selection:
  --participant-label LABEL [LABEL ...]
  --session-label     LABEL [LABEL ...]
  --task-label        LABEL [LABEL ...]
  --bids-filter-file  FILE     JSON file with extra pybids query filters.

Preprocessing:
  --motion-correction          {tddr,wavelet,spline,none}   [default: tddr]
  --exclude-channels           Comma-separated channel names to manually exclude,
                               e.g. "S1_D1 hbo,S1_D1 hbr"
  --cardiac-l-freq FLOAT       Lower cardiac band bound in Hz. Increase for children.  [default: 0.7]
  --cardiac-h-freq FLOAT       Upper cardiac band bound in Hz. Increase for children.  [default: 1.5]

Postprocessing mode:
  --mode                       {denoise,glm,connectivity}
  --config FILE                TOML file for post parameters. CLI flags override TOML.

Filtering / resampling:
  --high-pass FLOAT            High-pass filter cutoff in Hz (e.g. 0.01).
  --low-pass  FLOAT            Low-pass filter cutoff in Hz (e.g. 0.5).
  --resample-sfreq FLOAT       Target sampling rate in Hz after filtering (e.g. 2.0).

Cropping (applied after filtering, before GLM):
  --segments-path FILE         TSV file (onset/duration columns) defining multiple segments to keep.
  --crop-tmin FLOAT            Start time in seconds (single segment; ignored if --segments-path set).
  --crop-tmax FLOAT            End time in seconds  (single segment; ignored if --segments-path set).

GLM options (--mode glm):
  --hrf-model                  {spm,spm + derivative,spm + derivative + dispersion,
                                glover,glover + derivative,glover + derivative + dispersion,fir}
                               [default: spm]
  --noise-model                {ols,ar1,ar2,ar3,ar4,ar5,auto}  [default: ar1]
  --drift-model                {cosine,polynomial,none}         [default: cosine]
  --drift-high-pass FLOAT      High-pass cutoff for cosine drift in Hz.  [default: 0.01]
  --drift-order INT            Polynomial drift order (polynomial only).  [default: 1]
  --fir-delays STR             FIR delay bins in scans, comma-separated.
                               e.g. "0,1,2,3,4,5"  (only used when --hrf-model fir)
  --short-channel              {none,mean,pca}  Short-channel confound regressor.  [default: none]
  --events-path FILE           Path to *_events.tsv. Falls back to snirf annotations if omitted.
  --contrast-file FILE         TOML file defining GLM contrasts.

Output:
  --no-report                  Skip HTML QC report.
  --output-space               {native,MNI152}  [default: native]
  --n-jobs INT                 Parallel subject jobs.  [default: 1]
  --work-dir DIR               Cache directory.

Escape hatches:
  --ignore ASPECT [ASPECT ...]  Skip: events, short-channels, physio, bids-validation
  --skip-bids-validation
  --dry-run

Other:
  --verbose
  --version
```

## Output Structure

```
output/
  dataset_description.json
  logs/
    fnirs_pipe.db                       # SQLite DB (after fnirs-log merge)
    json/                               # JSONL event files written per run
  sub-01/
    sub-01_qc.html                      # per-subject QC report
    logs/
      sub-01_TIMESTAMP.toml             # run record (parameters)
      sub-01_TIMESTAMP_script.py        # reproducible run script
    nirs/
      sub-01_desc-od_nirs.snirf
      sub-01_desc-sci_nirs.snirf
      sub-01_desc-motcorrected_nirs.snirf
      sub-01_desc-preproc_nirs.snirf
      sub-01_iqm_raw.toml               # IQM at raw checkpoint
      sub-01_iqm.toml                   # IQM at Beer-Lambert checkpoint
      sub-01_channel_metrics.csv
      sub-01_desc-denoised_nirs.snirf   # present when --mode denoise or glm
```

## QC Report Contents

- SCI topography (good/bad channel map)
- Carpet plot before/after motion correction
- PSD before/after bandpass (with heartbeat peak annotation)
- HbO/HbR negative correlation check
- Bad channel rate indicator (green < 10% / yellow 10–30% / red > 30%)
- Auto-generated Methods section + software versions + references

## QC Report Output Layout

`fnirs-qc prep-raw` and `fnirs-qc hyper-raw` follow a BIDS-derivatives layout
(mirroring mriqc / fmriprep): main HTMLs at the root, per-entity subdirs hold
figures and IQM data.

```
output/qc/
├── sub-01_task-tapping_desc-raw_nirs.html               # main viewer
├── sub-01_task-tapping_raw_channel_decisions.json       # rating sidecar
├── sub-01_task-tapping_raw_ratings.json
├── sub-01/
│   ├── figures/                                         # standalone Plotly HTMLs (iframe-loaded)
│   │   ├── sub-01_task-tapping_desc-scipsp_nirs.html
│   │   ├── sub-01_task-tapping_desc-chS1D1_nirs.html    # per-channel detail
│   │   └── ...
│   └── [ses-XX/]nirs/
│       └── sub-01[_ses-XX]_task-tapping_desc-iqm_nirs.json
├── group-G1003_task-nohold_desc-hyperraw_nirs.html
└── group-G1003/
    ├── figures/
    └── [ses-XX/]nirs/
        └── group-G1003[_ses-XX]_task-nohold_desc-iqm_nirs.json
```

## Package Structure

```
fnirs_pipe/
  cli/          fnirs-pipe, fnirs-recon, fnirs-qc, fnirs-rate, fnirs-gui, fnirs-log
  pipeline/     prep_pipeline, post_pipeline, glm, denoise, hyperscanning, restingstate
  io/           BIDS layout, snirf read/write, derivatives output
  qc/           HTML report, Plotly figures, quantitative metrics, boilerplate text
  utils/        logging, run_record, job_db
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

See [ROADMAP.md](ROADMAP.md) for planned features and [CHANGELOG.md](CHANGELOG.md) for release history.

## Citation

> *fnirs-pipe is in active development. Citation instructions will be added at first stable release.*
