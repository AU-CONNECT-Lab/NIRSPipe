# fnirs-pipe

A BIDS-compatible fNIRS preprocessing, postprocessing, hyperscanning, and QC pipeline.

## Overview

`fnirs-pipe` runs three kinds of work, selected by the analysis level:

| Level | What it does |
|-------|--------------|
| `participant` | `prep`: fixed-order preprocessing, OD conversion → SCI channel marking → motion correction (TDDR or wavelet) → Beer-Lambert. Then `post` when `--mode` is given: `denoise`, `glm`, or `rest` (bandpass + resample; confound regression; GLM residuals or ALFF/FC) |
| `hyper` | Dyad analysis over the derivatives: wavelet coherence, inter-subject correlation, and the pseudo-dyad null |
| `group` | Cohort aggregation of the quality records and the per-dyad coherence tables |

Outputs follow the [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) spec. Each run gets an HTML QC report with figures, a provenance graph and an auto-generated Methods paragraph, and each subject an index page over their runs. Group-level QC, hyperscanning (dyad WTC/ISC), an interactive rating viewer, a Dash desktop GUI, and a JSONL→SQLite run-log database are all first-class features.

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

Wherever a command takes a table from you (events, segments, the pairs file, `participants.tsv`), the extension decides the delimiter: `.tsv` is tab-separated, `.csv` comma-separated, and any other extension has its delimiter sniffed from the header. Option names such as `--pairs-csv` and `--events-path` say nothing about which format you must supply. Files the package writes back are always tab-separated, as BIDS requires.

### `fnirs-pipe`

```
fnirs-pipe BIDS_DIR OUTPUT_DIR {participant,group,hyper,wtc-band} [OPTIONS]

Required at participant level:
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
  --window-length FLOAT        Window (s) for the windowed SCI / PSP / GVTD series. [default: 10.0]

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
  --aux-regressors             Add the recording's auxiliary channels to the confound
                               regression (accelerometers, gyroscopes, pulse trace).
                               Preprocessing extracts them to desc-aux_timeseries.tsv.gz.
  --aux-channels NAME [NAME ...]
                               Which aux channels to use. Default: all of them.
  --fc                         Also write the connectivity products rest mode writes,
                               from the residual. Works in glm and denoise modes.
  --events-path FILE           Optional. BIDS *_events.tsv overriding SNIRF annotations.
                               Mutually exclusive with --stim-dur.
  --stim-dur FLOAT             Optional. Fixed duration for SNIRF annotations without one.
                               Mutually exclusive with --events-path.
  --contrast-file FILE         TOML file defining GLM contrasts.

Denoise (--mode denoise):
  Reuses GLM flags for confound regression (drift model, short-channel, aux).
  Any one of them writes desc-errts; no task model, no resting-state derivatives.
  --fc adds the connectivity products, from the residual or from the bandpassed
  data itself when no regression was asked for.

Rest (--mode rest):
  Reuses GLM flags for confound regression (drift model, short-channel, aux).
  --high-pass + --low-pass required for ALFF (FC computed regardless).

Hyperscanning analysis (hyper level):
  --pairs-csv FILE             Required. Columns: group_id, subject_id, task.
  --group-id TEXT              Process one group only.
  --desc TEXT                  Per-subject stage to read.                     [default: preproc]
  --wtc-fmin/--wtc-fmax FLOAT  WTC frequency axis.                       [default: 0.004 / 0.20]
  --wtc-band-fmin FLOAT        Band the per-channel TSV averages over. Defaults to the axis.
  --wtc-band-fmax FLOAT
  --wtc-significance           Monte Carlo significance contour (slow).
  --wtc-mc-count INT           Surrogates behind each contour.                      [default: 300]
  --wtc-seed INT               Seeds the surrogates and the phase randomisation.
  --wtc-mask-coi               Restrict each band mean to the cone of influence.
  --wtc-roi-min-channels N     Drop an ROI cell resting on fewer pairs.               [default: 2]
  --wtc-channel-cross          Cross every long channel with every other.
  --wtc-save-maps              Keep the time-frequency maps for the wtc-band level.
  --wtc-pseudo N               Also write the pseudo-dyad null, averaged over N iterations.
  --wtc-pseudo-cross           Cross the channels for the null too. Independent of the above.
  --bads-scope {run,subject}   Which rejections the metrics exclude.                [default: run]
  --isc-threshold FLOAT        Minimum mean ISC to draw an arc.                     [default: 0.3]
  --normalize                  Z-score each channel per subject after alignment.
  --no-align                   Trim to the shortest recording instead of aligning on a trigger.
  --tstart/--tend FLOAT        Window on the aligned clock, where 0 is the shared trigger.

wtc-band level:
  --band-fmin/--band-fmax      Required. The new band to average the saved maps over.
  --mask-coi                   Cone-of-influence masking, matching --wtc-mask-coi.
  --suffix TEXT                Names the new tables. Defaults to the band.

Output:
  --no-report                  Skip HTML QC report.
  --roi-mapping FILE           JSON mapping ROI labels → channel lists. Groups the denoising
                               carpet by ROI and enables ROI-level FC (rest mode); at the
                               hyper level it drives the ROI x ROI coherence matrix.
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

`participant` and `group` take a BIDS dataset. `hyper` and `wtc-band` read the derivatives tree only, and accept `BIDS_DIR` without using it, so the positional shape is the same at every level.

The `hyper` level computes wavelet coherence and inter-subject correlation for each dyad named in the pairs file, and writes one report per group. `--wtc-significance` is slow: it draws `--wtc-mc-count` surrogate series per channel pair, 300 by default, and the runtime scales with that count. `--wtc-channel-cross` pairs every long channel with every other across the two brains, 196 values instead of 14, and is what builds the ROI x ROI matrix when `--roi-mapping` is given; the heatmaps stay on the homologous pairs.

`--wtc-pseudo N` adds the pseudo-dyad null: the same band means taken against a phase-scrambled partner, averaged over N iterations. Coherence between two unrelated recordings is not zero, so this is what a real value is read against. Each iteration is a full WTC run, which makes it the expensive half of a hyper run, so nothing is computed unless you ask. It shares this run's stage, band and window by construction, which is what makes it the null for the table it sits beside; its crossing is the one thing it does not share, since crossing squares the pair count and the null would pay that on every iteration. Ask for it separately with `--wtc-pseudo-cross`.

The `group` level aggregates the per-subject and per-dyad quality records into cohort reports, and merges every per-dyad WTC band-mean table into one long table per kind.

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
                [--align none|trigger] [--trigger-name TEXT]
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

`crop` takes its window either from `--tmin` / `--tmax` or from a segments table passed to `--segments-path`. That table holds `onset` and `duration`, plus an optional `task` column naming each segment's output task entity instead of the default `_seg-NN`. Under `--align trigger`, times are measured from the first annotation of the name given to `--trigger-name` rather than from the start of the recording, so a single window selects the same stretch of task in every subject.

### `fnirs-qc` — QC reports

`prep-raw` and `hyper-raw` read raw recordings, so both require `--cardiac-l-freq` / `--cardiac-h-freq`, which are population-dependent and have no default, and `--dpf`, which they use to convert to haemoglobin internally.

```
fnirs-qc prep-raw BIDS_DIR OUTPUT_DIR PARTICIPANT_LABEL
                  --dpf FLOAT [FLOAT ...]
                  --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
                  [--session-label / --task-label]
                  [--sci-threshold FLOAT] [--skip-bids-validation]
                  [--epoch-qc] [--epoch-tmin/--epoch-tmax FLOAT]

fnirs-qc hyper-raw BIDS_DIR OUTPUT_DIR --pairs-csv PATH
                   --dpf FLOAT [FLOAT ...]
                   --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
                   [--group-id / --task-label / --session-label]
                   [--sci-threshold FLOAT] [--fmin/--fmax FLOAT]
                   [--normalize] [--no-align] [--tstart/--tend FLOAT]

fnirs-qc group-raw       OUTPUT_DIR
fnirs-qc group-hyper-raw OUTPUT_DIR
fnirs-qc provenance      OUTPUT_DIR
```

`provenance` redraws the graphs from the sidecars already on disk.

The dyad analysis used to live here as `hyper-post`, `hyper-null`, `wtc-band` and `group-hyper-wtc`. It is `fnirs-pipe` now: those four are the `hyper` level, its `--wtc-pseudo` flag, the `wtc-band` level, and the `group` level.

For a cohort report over one time window, crop first and then run the usual pair of commands:

```
fnirs-prep crop BIDS_DIR DERIV_DIR --participant-label ... --tmin FLOAT --tmax FLOAT
                [--align none|trigger] [--trigger-name TEXT]
fnirs-qc   prep-raw DERIV_DIR/cropped OUTPUT_DIR PARTICIPANT_LABEL --dpf ...
                    --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
fnirs-qc   group-raw OUTPUT_DIR
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
│   ├── sub-01_qc.html                   # index over the subject's runs
│   ├── sub-01_task-<t>_qc.html          # QC report, one per run
│   ├── figures/
│   │   └── sub-01_task-<t>/             # figures, one directory per run
│   │       ├── provenance.png           # provenance graph (embedded in that run's report)
│   │       └── provenance.mmd           # same graph, mermaid source
│   ├── logs/
│   │   ├── sub-01_{ts}.toml             # run record (env + params)
│   │   └── sub-01_script.py             # reproduction script
│   └── nirs/
│       ├── sub-01_desc-od_nirs.snirf              # prep step 2
│       ├── sub-01_desc-sci_nirs.snirf             # prep step 3
│       ├── sub-01_desc-motcorrected_nirs.snirf    # prep step 4
│       ├── sub-01_desc-preproc_nirs.snirf         # prep step 5 (terminus)
│       ├── sub-01_desc-filtered_nirs.snirf        # post: bandpass applied
│       ├── sub-01_desc-resampled_nirs.snirf       # post: resample applied
│       ├── sub-01_desc-errts_nirs.snirf           # post glm/rest/denoise: GLM residuals
│       ├── sub-01_desc-errtsbroad_nirs.snirf      # rest: un-bandpassed residual, ALFF input
│       ├── sub-01_task-<t>_desc-aux_timeseries.tsv.gz  # aux channels, if the file had any
│       ├── sub-01_task-<t>_desc-sqm_nirs.json     # SQM record, one per run
│       ├── sub-01_task-<t>_channel_metrics.csv    # per-channel metrics, one per run
│       ├── sub-01_design_matrix.csv                # glm mode
│       ├── sub-01_glm_results.csv                  # glm mode
│       ├── sub-01_contrasts.csv                    # glm mode + --contrast-file
│       ├── sub-01_alff.tsv                         # rest mode
│       ├── sub-01_desc-hbo_fc.tsv                  # channel × channel, per chromophore
│       ├── sub-01_desc-hbo_fcz.tsv                 # Fisher z of the above
│       ├── sub-01_desc-hbo_fcroi.tsv               # + --roi-mapping (ROI × ROI)
│       ├── sub-01_desc-hbo_fcroiz.tsv              # + --roi-mapping
│       ├── sub-01_desc-hbo_fcseed.tsv              # + --roi-mapping (ROI × channel)
│       └── sub-01_desc-hbo_fcseedz.tsv             # + --roi-mapping
├── group_nirs.{tsv,html}                # fnirs-qc group-raw
└── group_hyper_nirs.{tsv,html}          # fnirs-qc group-hyper-raw
```

The `fc*` files are written in rest mode, or in any mode run with `--fc`.

Standalone HTMLs from `fnirs-qc prep-raw` / `hyper-raw` and from `fnirs-pipe ... hyper` sit at the derivatives root:

```
output/
├── sub-01_task-tapping_desc-raw_nirs.html               # fnirs-qc prep-raw
├── sub-01_task-tapping_raw_channel_decisions.json       # fnirs-rate raw sidecar
├── sub-01_task-tapping_raw_ratings.json
├── group-G1003_task-tapping_desc-hyperraw_nirs.html     # fnirs-qc hyper-raw
├── group-G1003_task-tapping_desc-hyperpost_nirs.html    # fnirs-pipe ... hyper
└── group-G1003/
    └── figures/
```

## QC Reports

`fnirs-pipe` writes an HTML report per run automatically, plus an index page per subject; `fnirs-qc` adds standalone, group, and hyperscanning reports:

| Report | Command | Level / stage |
|--------|---------|---------------|
| Per-run | (pipeline, automatic) | individual: raw + post, one report per run plus a subject index |
| Raw pre-flight viewer | `fnirs-qc prep-raw` | individual — raw only |
| Group | `fnirs-qc group-raw` / `group-hyper-raw` | group — raw |
| Time-window group | `fnirs-prep crop` then `prep-raw` + `group-raw` | group — raw, cropped window |
| Per-trial | `fnirs-qc prep-raw --epoch-qc` | individual — SQM per task event, in the raw report |
| Dyad raw | `fnirs-qc hyper-raw` | hyperscanning — raw coherence |
| Dyad post | `fnirs-pipe ... hyper` | hyperscanning — post: WTC + ISC (ROI-level with `--roi-mapping`) |

### Per-run report contents

- Executive summary with traffic-light badges (bad channel rate, mean SCI, HbO–HbR corr, GVTD p95), and a metrics panel whose tooltips say which stage each number was measured on
- SCI / PSP probe layout + windowed heatmap
- Carpet plot before / after motion correction (GVTD trace over all channels), with the spike spans and the correction footprint drawn beneath
- Per-channel motion panel with SCI-coloured traces
- PSD before / after bandpass (cardiac + Mayer wave peaks annotated)
- HbO–HbR correlation panel
- Topographic maps of the condition-averaged response, at time points after onset, for task data
- Trial images, split by condition, ROI-averaged and per channel, for task data
- Denoising carpet, before / after, HbO and HbR separately, both scaled by the pre-denoising SD (grouped by ROI with `--roi-mapping`), when postprocessing runs
- Global correlation (`gcor`) before → after denoising when postprocessing runs
- GLM section (design matrix + activation panel) when `--mode glm`
- Rest section (ALFF and fALFF as bars and on the optode layout, FC heatmap, connectogram, ROI-to-ROI matrix and seed topography with `--roi-mapping`) when `--mode rest`, or in any mode run with `--fc`
- The run's provenance graph and a table of every output with the step that made it
- Auto-generated Methods paragraph + software versions + references

Group / window / dyad reports add subject × metric heatmaps, per-scale grouped boxplots with clickable strip points (Tukey 1.5 × IQR outliers), and sortable tables.

## Package Structure

```
fnirs_pipe/
  cli/          fnirs-pipe, fnirs-recon, fnirs-prep, fnirs-qc, fnirs-rate, fnirs-gui, fnirs-log
  pipeline/     prep_pipeline, post_pipeline, glm, denoise, restingstate, motion,
                hyperscanning, synchrony, crop, edit_markers
  io/           BIDS layout, snirf read/write, derivatives output, snirf aux group,
                delimiter-sniffing table reader
  qc/           HTML report, subject index, Plotly figures, quantitative metrics, quality
                record, provenance graph, boilerplate text, group / hyper / window / epoch
                writers, WTC store and aggregation, rating apps
  interface/    Dash GUI (6 pages + sidebar)
  utils/        logging, run_record, job_db, lineage, run_script
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
