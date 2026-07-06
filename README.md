# fnirs-pipe

A BIDS-compatible fNIRS preprocessing, postprocessing, hyperscanning, and QC pipeline.

## Overview

`fnirs-pipe` is split into two stages:

| Stage | What it does |
|-------|--------------|
| `prep` | Fixed-order preprocessing: OD conversion → SCI channel marking → motion correction (TDDR) → Beer-Lambert |
| `post` | Mode-driven postprocessing: `denoise`, `glm`, or `rest` (bandpass + resample; GLM residuals or ALFF/FC) |

Outputs follow the [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) spec. Each subject gets an HTML QC report with figures and an auto-generated Methods paragraph. Group-level QC, hyperscanning (dyad WTC/ISC), an interactive rating viewer, a Dash desktop GUI, and a JSONL→SQLite run-log database are all first-class features.

## Requirements

- Python ≥ 3.10
- Dependencies (all required, resolved via `pip install .`): `mne`, `mne-nirs` (≥ 0.7), `nilearn`, `scipy`, `numpy`, `pybids`, `mne-bids`, `h5py`, `tables`, `pandas`, `jinja2`, `plotly`, `kaleido`, `joblib`, `flask`, `dash`, `dash-bootstrap-components`, `dash-cytoscape`, `pycwt`, `PyWavelets`, `bibtexparser`, `tomli` (Python < 3.11 only)

## Installation

```bash
git clone <repo>
cd fNIRS_pipe
pip install -e ".[dev]"
```

## Quick Start

Preprocessing only:

```bash
fnirs-pipe /data/bids /data/derivatives participant \
  --participant-label 01 02 \
  --dpf 6.0 --sci-threshold 0.8 \
  --cardiac-l-freq 0.7 --cardiac-h-freq 1.5 \
  --resp-l-freq 0.1 --resp-h-freq 0.5
```

Preprocessing + first-level GLM (events from SNIRF annotations by default):

```bash
fnirs-pipe /data/bids /data/derivatives participant \
  --participant-label 01 \
  --dpf 6.0 --sci-threshold 0.8 \
  --cardiac-l-freq 0.7 --cardiac-h-freq 1.5 \
  --resp-l-freq 0.1 --resp-h-freq 0.5 \
  --mode glm \
  --hrf-model spm --noise-model ar1 \
  --drift-model cosine --drift-high-pass 0.01 \
  --high-pass 0.01 --low-pass 0.5 \
  --short-channel mean
```

See [`docs/pipeline/common-scenarios.md`](docs/pipeline/common-scenarios.md) for resting-state, FIR GLM, paediatric, parallel, dry-run, and config-file examples. Per-command references live under [`docs/cli/`](docs/cli/) — one page per `fnirs-*` entry point.

## CLI Reference

### `fnirs-pipe`

```
fnirs-pipe BIDS_DIR OUTPUT_DIR {participant,group} [OPTIONS]

Required:
  --dpf FLOAT [FLOAT ...]      Differential pathlength factor. One value or one per wavelength.
  --sci-threshold FLOAT        SCI threshold for bad channel detection (e.g. 0.8).
  --cardiac-l-freq FLOAT       Lower cardiac band bound in Hz (population-dependent, no default).
  --cardiac-h-freq FLOAT       Upper cardiac band bound in Hz.
  --resp-l-freq FLOAT          Lower respiration band bound in Hz (population-dependent, no default).
  --resp-h-freq FLOAT          Upper respiration band bound in Hz.

Subject / session / task selection:
  --participant-label LABEL [LABEL ...]   Space-separated or repeated.
  --session-label LABEL [LABEL ...]       Space-separated or repeated.
  --task-label LABEL [LABEL ...]          Space-separated or repeated.
  --bids-filter-file FILE      JSON file with extra pybids query filters.

Preprocessing:
  --motion-correction          {tddr,wavelet,spline,none}   [default: tddr]
                               tddr + wavelet implemented; spline raises NotImplementedError.
  --bad-channels               Comma-separated S-D labels to mark bad,
                               e.g. "S1_D1,S2_D3" (unioned with SCI bads)

Postprocessing mode:
  --mode                       {denoise,glm,rest}
  --config FILE                TOML file for post parameters. CLI flags override TOML.

Filtering / resampling (all modes):
  --high-pass FLOAT            High-pass filter cutoff in Hz (e.g. 0.01).
  --low-pass  FLOAT            Low-pass filter cutoff in Hz (e.g. 0.5).
  --resample-sfreq FLOAT       Target sampling rate in Hz after filtering (e.g. 2.0).
  --combine-runs               Concatenate runs before postprocessing.

GLM (--mode glm):
  --hrf-model                  {spm,spm + derivative,spm + derivative + dispersion,
                                glover,glover + derivative,glover + derivative + dispersion,fir}
  --noise-model                {ols,ar1,ar2,ar3,ar4,ar5}
  --drift-model                {cosine,polynomial,none}
  --drift-high-pass FLOAT      Cosine drift high-pass cutoff in Hz.
  --drift-order INT            Polynomial drift order.                                [default: 1]
  --fir-delays STR             FIR delay bins in scans, e.g. "0,1,2,3,4,5"
  --short-channel              {none,mean,pca}                                        [default: none]
  --events-path FILE           Optional. BIDS *_events.tsv overriding SNIRF annotations.
                               Mutually exclusive with --stim-dur.
  --stim-dur FLOAT             Optional. Fixed duration for SNIRF annotations without one.
                               Mutually exclusive with --events-path.
  --contrast-file FILE         TOML file defining GLM contrasts.

Rest (--mode rest):
  Reuses GLM flags for confound regression (drift model, short-channel).
  --high-pass + --low-pass required for ALFF (FC computed regardless).

Output:
  --no-report                  Skip HTML QC report.
  --roi-mapping FILE           JSON mapping ROI labels → channel lists. Groups the denoising
                               carpet by ROI and enables ROI-level FC (rest mode).
  --n-jobs INT                 Parallel subject jobs.                                 [default: 1]
  --work-dir DIR               Hash cache directory (not yet implemented).

Escape hatches:
  --ignore ASPECT              Repeatable. Options: events, bids-validation.
  --skip-bids-validation
  --dry-run

Other:
  --verbose
  --version
```

### `fnirs-recon` — raw SNIRF → BIDS

```
fnirs-recon INPUT_FILE BIDS_DIR --subject LABEL --task LABEL
                                [--session LABEL] [--run INDEX]
                                [--overwrite]
```

### `fnirs-prep` — headless data-preparation utilities

```
fnirs-prep crop BIDS_DIR DERIVATIVES_DIR --participant-label SUB ...
                ( --tmin FLOAT [--tmax FLOAT] | --segments-path PATH [--combine] )
                [--ses TEXT] [--task TEXT] [--run TEXT]
                [--n-jobs INT] [--skip-bids-validation]

fnirs-prep align BIDS_DIR DERIVATIVES_DIR --group-csv PATH
                 [--skip-bids-validation]

fnirs-prep edit-markers export BIDS_DIR OUT_DIR --participant-label SUB ...
                               [--ses/--task/--run] [--n-jobs INT]

fnirs-prep edit-markers apply BIDS_DIR DERIVATIVES_DIR --participant-label SUB ...
                              ( --tsv PATH | --shift FLOAT | --set-duration FLOAT
                                | --rename OLD:NEW ... )
                              [--ses/--task/--run] [--n-jobs INT]
```

### `fnirs-qc` — QC reports

```
# prep-raw / hyper-raw / window-raw / epoch require --cardiac-l-freq/--cardiac-h-freq (no default)

fnirs-qc prep-raw BIDS_DIR OUTPUT_DIR PARTICIPANT_LABEL
                  --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
                  [--session-label / --task-label]
                  [--sci-threshold FLOAT] [--skip-bids-validation]

fnirs-qc hyper-raw BIDS_DIR OUTPUT_DIR --pairs-csv PATH
                   --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
                   [--group-id / --task-label / --session-label]
                   [--sci-threshold FLOAT] [--fmin/--fmax FLOAT]
                   [--normalize] [--no-align]

fnirs-qc hyper-post BIDS_DIR OUTPUT_DIR --pairs-csv PATH
                    [--group-id / --task-label / --session-label]
                    [--roi-mapping PATH]
                    [--wtc-fmin/--wtc-fmax FLOAT] [--isc-threshold FLOAT]
                    [--normalize] [--no-align]

fnirs-qc group-raw       OUTPUT_DIR
fnirs-qc group-hyper-raw OUTPUT_DIR

fnirs-qc window-raw BIDS_DIR OUTPUT_DIR --task-label TEXT
                    --tstart FLOAT --tend FLOAT
                    --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
                    [--participant-label ...] [--session-label ...]
                    [--align none|trigger] [--trigger-name TEXT]
                    [--name TEXT] [--sci-threshold FLOAT]

fnirs-qc epoch BIDS_DIR OUTPUT_DIR --task-label TEXT
               --mode epoch|duration
               --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
               [--tmin/--tmax FLOAT] [--events-csv PATH]
               [--participant-label ...] [--session-label ...]
               [--sci-threshold FLOAT]
```

### `fnirs-rate` — Flask rating viewers

```
fnirs-rate rate  OUTPUT_DIR [--participant-label SUB ...] [--port INT]   # default 8765
fnirs-rate raw   OUTPUT_DIR PARTICIPANT_LABEL
                 [--session-label / --task-label] [--sci-threshold FLOAT] [--port INT]   # default 5052
fnirs-rate hyper OUTPUT_DIR GROUP_ID TASK_LABEL --pairs-csv PATH
                 [--session-label TEXT] [--sci-threshold FLOAT] [--port INT]             # default 5053
```

### `fnirs-gui` — Dash desktop interface

```
fnirs-gui [--port INT]        # default 8050
```

### `fnirs-log` — JSONL → SQLite merge

```
fnirs-log merge OUTPUT_DIR [--db-path PATH]
```

Full per-command references (parameter tables, examples, sidecar formats) live in [`docs/cli/`](docs/cli/).

## Output Structure

```
output/
├── dataset_description.json
├── logs/
│   ├── fnirs-pipe_{ts}.log             # text log
│   ├── json/                            # JSONL event stream per run
│   │   ├── _pipeline/execution_*.jsonl
│   │   ├── _runs/sub-*_*.jsonl
│   │   ├── _sqm/sub-*_*.jsonl
│   │   └── _outputs/sub-*_*.jsonl
│   └── fnirs_pipe.db                    # SQLite (after fnirs-log merge)
├── sub-01/
│   ├── sub-01_qc.html                   # per-subject QC report
│   ├── logs/
│   │   └── sub-01_{ts}.toml             # run record (env + params)
│   └── nirs/
│       ├── sub-01_desc-od_nirs.snirf              # prep step 2
│       ├── sub-01_desc-sci_nirs.snirf             # prep step 3
│       ├── sub-01_desc-motcorrected_nirs.snirf    # prep step 4
│       ├── sub-01_desc-preproc_nirs.snirf         # prep step 5 (terminus)
│       ├── sub-01_desc-filtered_nirs.snirf        # post: bandpass applied
│       ├── sub-01_desc-resampled_nirs.snirf       # post: resample applied
│       ├── sub-01_desc-errts_nirs.snirf           # post glm/rest: GLM residuals
│       ├── sub-01_desc-sqm_nirs.json              # SQM (raw + final checkpoints)
│       ├── design_matrix.csv                       # glm mode
│       ├── glm_results.csv                         # glm mode
│       ├── contrasts.csv                           # glm mode + --contrast-file
│       ├── sub-01_alff.tsv                         # rest mode
│       ├── sub-01_fc.tsv                           # rest mode (channel × channel)
│       └── sub-01_fcroi.tsv                        # rest mode + --roi-mapping (ROI × ROI)
├── group_nirs.{tsv,html}                # fnirs-qc group-raw
├── group_hyper_nirs.{tsv,html}          # fnirs-qc group-hyper-raw
└── group_nirs_window-<a>-<b>.{tsv,html} # fnirs-qc window-raw (per invocation)
```

Standalone QC HTMLs from `fnirs-qc prep-raw` / `hyper-raw` / `hyper-post` sit at the derivatives root:

```
output/
├── sub-01_task-tapping_desc-raw_nirs.html               # fnirs-qc prep-raw
├── sub-01_task-tapping_raw_channel_decisions.json       # fnirs-rate raw sidecar
├── sub-01_task-tapping_raw_ratings.json
├── group-G1003_task-tapping_desc-hyperraw_nirs.html     # fnirs-qc hyper-raw
├── group-G1003_task-tapping_desc-hyperpost_nirs.html    # fnirs-qc hyper-post
└── group-G1003/
    └── figures/
```

## QC Reports

`fnirs-pipe` writes a per-subject HTML report automatically; `fnirs-qc` adds standalone, group, and hyperscanning reports:

| Report | Command | Level / stage |
|--------|---------|---------------|
| Per-subject | (pipeline, automatic) | individual — raw + post |
| Raw pre-flight viewer | `fnirs-qc prep-raw` | individual — raw only |
| Group | `fnirs-qc group-raw` / `group-hyper-raw` | group — raw |
| Time-window group | `fnirs-qc window-raw` | group — raw, cropped window |
| Per-trial | `fnirs-qc epoch` | individual — SQM per task event |
| Dyad raw | `fnirs-qc hyper-raw` | hyperscanning — raw coherence |
| Dyad post | `fnirs-qc hyper-post` | hyperscanning — post: WTC + ISC (ROI-level with `--roi-mapping`) |

### Per-subject report contents

- Executive summary with traffic-light badges (bad channel rate, mean SCI, HbO–HbR corr, GVTD p95)
- SCI / PSP probe layout + windowed heatmap
- Carpet plot before / after motion correction (GVTD trace over all channels)
- Per-channel motion panel with SCI-coloured traces
- PSD before / after bandpass (cardiac + Mayer wave peaks annotated)
- HbO–HbR correlation panel
- Denoising carpet — before / after, HbO and HbR separately, both scaled by the pre-denoising SD (grouped by ROI with `--roi-mapping`), when postprocessing runs
- GLM section (design matrix + activation panel) when `--mode glm`
- Rest section (ALFF table + FC heatmap; ROI-level FC with `--roi-mapping`) when `--mode rest`
- Auto-generated Methods paragraph + software versions + references

Group / window / dyad reports add subject × metric heatmaps, per-metric boxplots (Tukey 1.5 × IQR outliers), and sortable tables.

## Package Structure

```
fnirs_pipe/
  cli/          fnirs-pipe, fnirs-recon, fnirs-prep, fnirs-qc, fnirs-rate, fnirs-gui, fnirs-log
  pipeline/     prep_pipeline, post_pipeline, glm, denoise, restingstate, hyperscanning,
                crop, edit_markers, channel_registration
  io/           BIDS layout, snirf read/write, derivatives output
  qc/           HTML report, Plotly figures, quantitative metrics, boilerplate text,
                group / hyper / window writers, rating apps
  interface/    Dash GUI (4 pages + sidebar)
  utils/        logging, run_record, job_db
  exceptions.py AlignmentError, GroupCSVError, MissingDerivativesError, ...
```

## Documentation

See [`docs/`](docs/) for the full handbook — installation, quickstart, configuration, per-command references, QC anatomy, hyperscanning, GUI, and the logging + database subsystem.

## Development

```bash
pip install -e ".[dev]"
pytest
```

See [ROADMAP.md](ROADMAP.md) for planned features and [CHANGELOG.md](CHANGELOG.md) for release history.

## Citation

> *fnirs-pipe is in active development. Citation instructions will be added at first stable release.*
