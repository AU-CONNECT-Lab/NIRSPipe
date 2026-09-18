# fnirs-pipe

A BIDS-compatible fNIRS preprocessing, postprocessing, hyperscanning, and QC pipeline.

## Overview

`fnirs-pipe` runs two levels; dyad analysis is a separate tool, `fnirs-hyper`:

| Level | What it does |
|-------|--------------|
| `participant` | `prep`: fixed-order preprocessing, OD conversion → windowed channel screening → motion correction (TDDR or wavelet) → Beer-Lambert. Then `post` when `--mode` is given: `denoise`, `glm`, or `rest` (bandpass + resample; confound regression; GLM residuals or ALFF/FC) |
| `group` | Cohort aggregation of the per-subject and per-dyad quality records |

Outputs follow the [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) spec. Each run gets an HTML QC report with figures, a provenance graph and an auto-generated Methods paragraph, and each subject an index page over their runs. Cohort-level QC, hyperscanning (dyad WTC/ISC), an interactive rating viewer, a Dash desktop GUI, and a JSONL→SQLite run-log database are all first-class features.

## Requirements

- Python ≥ 3.10
- Every dependency is required and resolved by `pip install .`; what each one is for is in [`docs/getting-started/installation.md`](docs/getting-started/installation.md)

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

Entity flags are spelled the same on every command: `--participant-label`, `--session-label`, `--task-label`, `--run-label`, `--group-id`. A label may be given with its `sub-` / `ses-` / `task-` prefix or without. Every command takes `--version`.

Wherever a command takes a table from you (events, segments, the pairs file, `participants.tsv`), the extension decides the delimiter: `.tsv` is tab-separated, `.csv` comma-separated, and any other extension has its delimiter sniffed from the header. Option names such as `--pairs-csv` and `--events-path` say nothing about which format you must supply. Files the package writes back are always tab-separated, as BIDS requires.

### Channel screening

Four flags decide which channels survive, and they are the same on `fnirs-pipe`, `fnirs-qc prep-raw` and `fnirs-qc hyper-raw`:

- `--sci-threshold` and `--psp-threshold` define a *coupled window*: it has to clear both. PSP catches the movement that fakes a high SCI.
- `--min-good-frac` (default 0.75) is the criterion that actually rejects: the share of windows a channel has to be coupled in. Counting windows rather than averaging them is what catches a channel that was fine for the first half of a recording and dead for the second.
- `--screen-scope` picks which windows count: `run` (default) the whole recording, `task` only the annotated blocks, so the lead-in and the gaps stop being held against a channel that is coupled throughout every block. `task` falls back to `run` when no annotation holds two windows.

`--window-length` (default 10 s) sets the window grid these, and the GVTD series, are measured on.

### `fnirs-pipe`

```
fnirs-pipe BIDS_DIR OUTPUT_DIR {participant,group} [OPTIONS]

Required at participant level:
  --dpf FLOAT [FLOAT ...]      Differential pathlength factor. One value or one per wavelength.
  --sci-threshold FLOAT        Scalp coupling index a window must reach (e.g. 0.8).
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
                               tddr and wavelet are implemented; spline is withheld and is
                               refused at the start of the run.
  --bad-channels               S-D labels ("S1_D1,S2_D3") or a table with participant_id +
                               bad_channels columns. Either wavelength marks the pair.
                               Unioned with the screening's bads.
  --psp-threshold FLOAT        Peak spectral power a window must reach.           [default: 0.1]
  --min-good-frac FLOAT        Share of windows a channel must be coupled in.    [default: 0.75]
  --screen-scope               {run,task}  Which windows count toward it.         [default: run]
  --window-length FLOAT        Window (s) for the windowed SCI / PSP / GVTD series. [default: 10.0]
  --short-max-dist MM          Separation at or below which a channel is short. [default: 10]
                               Short channels see scalp only and are measured, and regressed,
                               separately from the long ones.
  --long-min-dist MM           Separation at or above which a channel is long. [default: 15]
                               The gap above --short-max-dist is deliberate: channels inside it
                               reach neither scalp nor cortex cleanly, join no section, and are
                               named in a run note.
  --long-max-dist MM           Separation above which a channel is too far to be long. Off by
                               default; set it on a montage carrying pairs too far apart to
                               trust, since SCI and PSP catch most but not all of them.
  --epoch-tmin FLOAT           Trial window for the epoch figures and per-trial scoring,
  --epoch-tmax FLOAT           relative to each event onset. Both or neither. Omitted, the
                               figures use -5 to 25 s and the scoring uses each event's own
                               duration, which is what a block design records.
  --epoch-chunk-duration SEC   Cut each task annotation into trials this long before epoching,
                               so a block design gets one trial per piece. A 240 s block at
                               25 s gives 9 trials, remainder dropped.
  --epoch-single-trial         Draw the epoch section even when no condition repeats. Skipped
                               by default there: nothing is averaged, and a 30 s window off a
                               block running for minutes reads as a response.
  --by-condition               One QC report page per annotated condition, as desc-<condition>.
                               Sliced out of the windowed pass already in the quality record, so
                               nothing is cut and nothing is measured again.
  --gvtd-censor [SET]          Mark the frames GVTD flags as BAD_gvtd annotations. Off by
                               default; takes the channel set to flag on, default long, `all`
                               being the conservative choice. Nothing is cut: epoching drops the
                               trials the spans overlap, continuous analyses take what survives.
  --gvtd-censor-n-std FLOAT    Censoring threshold, in left-tail SDs above the GVTD mode.
                               [default: 10.0, the lenient value; the reports score at 3]
  --gvtd-min-epoch-s FLOAT     Shortest surviving stretch censoring keeps (s). Anything
                               shorter is censored with the artifacts around it. [default: 30.0]

Postprocessing mode:
  --mode                       {denoise,glm,rest}
  --config FILE                TOML file for post parameters, plus the cardiac, respiration
                               and separation bands. CLI flags override TOML.

Filtering / resampling (all modes):
  --high-pass FLOAT            High-pass filter cutoff in Hz (e.g. 0.01).
  --low-pass  FLOAT            Low-pass filter cutoff in Hz (e.g. 0.5).
  --filter-method              {iir,fir}   [default: iir]
                               iir is a zero-phase Butterworth, what the fNIRS toolboxes use;
                               fir is hamming-windowed linear-phase and is refused when its
                               3.3 * sfreq / transition samples exceed the recording.
  --filter-order INT           Butterworth order, ignored by --filter-method fir. [default: 4]
                               Applied with filtfilt, so the rolloff is twice this and the
                               cutoff sits at -6 dB.
  --resample-sfreq FLOAT       Target sampling rate in Hz after filtering (e.g. 2.0).
  --combine-runs               Concatenate runs before postprocessing.

GLM (--mode glm):
  --hrf-model                  {spm,spm + derivative,spm + derivative + dispersion,
                                glover,glover + derivative,glover + derivative + dispersion,fir}
  --noise-model                ols | auto | arN                            [default: auto]
                               auto is an AR order of 4x the sampling rate, which is what
                               the fNIRS implementations use; the low orders come from fMRI,
                               where a slower sampling rate makes one lag enough. An order
                               too low leaves a contrast's t values several times too large.
                               Honoured by every mode, not only glm.
  --drift-model                {cosine,polynomial,none}
  --drift-high-pass FLOAT      Cosine drift high-pass cutoff in Hz. No default: it has to sit
                               at or above --high-pass and below the rate a condition repeats
                               at, and the run warns when it does not.
  --drift-order INT            Polynomial drift order.                                [default: 1]
  --fir-delays STR             FIR delay bins in scans, e.g. "0,1,2,3,4,5"
  --short-channel              {none,mean}  Refused on a montage with no short channel.
                                                                                   [default: none]
  --fc                         Also write the connectivity products rest mode writes,
                               from the residual. Works in glm and denoise modes.
  --events-path FILE           Optional. BIDS *_events.tsv overriding SNIRF annotations.
                               Mutually exclusive with --stim-dur.
  --stim-dur FLOAT             Optional. Fixed duration for SNIRF annotations without one.
                               Mutually exclusive with --events-path.
  --contrast-file FILE         TOML file defining GLM contrasts.

Denoise (--mode denoise):
  Reuses GLM flags for confound regression (drift model, short-channel).
  Any one of them writes desc-errts; no task model, no resting-state derivatives.
  --fc adds the connectivity products, from the residual or from the bandpassed
  data itself when no regression was asked for.

Rest (--mode rest):
  Reuses GLM flags for confound regression (drift model, short-channel).
  --high-pass + --low-pass required for ALFF (FC computed regardless).

Output:
  --no-report                  Skip HTML QC report.
  --roi-mapping FILE           JSON mapping ROI labels to channel lists. Groups the denoising
                               carpet by ROI and enables ROI-level FC (rest mode).
  --n-jobs INT                 Parallel subject jobs.                                 [default: 1]
  --work-dir DIR               Hash cache directory (recorded; caching not yet implemented).

Escape hatches:
  --ignore ASPECT              Repeatable. Options: events, bids-validation.
  --skip-bids-validation
  --allow-cropped-input        Run on a `fnirs-prep crop` tree, otherwise refused. Motion
                               correction and the bandpass then each see one segment. Prefer
                               preprocessing the uncut recording and cropping the result.
  --dry-run                    Write each subject's run record and script, then stop.

Other:
  --verbose
  --version
```

`participant` preprocesses each subject, and postprocesses when `--mode` is given. `group` aggregates the per-subject quality records into cohort reports. Dyad analysis is `fnirs-hyper`, below.

### `fnirs-hyper` — dyad analysis

```
fnirs-hyper run OUTPUT_DIR --pairs-csv PATH
                [--group-id TEXT] [--task-label LABEL ...] [--desc TEXT]
                [--roi-mapping PATH]
                [--wtc-fmin/--wtc-fmax FLOAT]        [default: 0.004 / 0.20]
                [--wtc-band-fmin/--wtc-band-fmax FLOAT]
                [--wtc-significance] [--wtc-mc-count INT] [--wtc-seed INT]
                [--wtc-chroma {hbo,hbr,both}]        [default: both]
                [--wtc-mask-coi | --no-wtc-mask-coi] [--wtc-roi-min-channels N]
                [--wtc-arrow-min R]                  [default: 0.5]
                [--wtc-channel-cross] [--by-condition | --no-by-condition]
                [--wtc-cond-transform] [--wtc-cond-pad-s SEC|auto]
                [--wtc-limit-scales | --no-wtc-limit-scales] [--wtc-save-maps]
                [--wtc-pseudo N] [--wtc-pseudo-cross]
                [--isc-threshold FLOAT] [--isc-whiten ORDER]
                [--isc-max-lag SECONDS] [--isc-pseudo N]
                [--bads-scope {run,subject}] [--check-only]
                [--sci-threshold FLOAT]
                [--short-max-dist/--long-min-dist/--long-max-dist MM]
                [--normalize] [--no-align] [--tstart/--tend FLOAT]

fnirs-hyper band  OUTPUT_DIR --wtc-band-fmin FLOAT --wtc-band-fmax FLOAT
                             [--wtc-mask-coi | --no-wtc-mask-coi] [--wtc-suffix TEXT]

fnirs-hyper index OUTPUT_DIR [--group-id TEXT]

fnirs-hyper merge OUTPUT_DIR
```

Every subcommand reads the derivatives tree `fnirs-pipe` wrote and takes no BIDS input, which is why there is one positional and not two.

`run` computes wavelet coherence and inter-subject correlation for each dyad in the pairs file, one report per group. A group of more than two gets one report, one set of figures and one ISC table per pairing, suffixed `_<sub1>x<sub2>`, the transform still running once over the whole group. `--check-only` aligns each dyad, prints what the metrics would be computed on, and stops, which is how to check a cohort's channel budget before a run that with a null takes hours.

`--wtc-significance` draws `--wtc-mc-count` surrogate series per channel pair, 300 by default, and the runtime scales with that count. `--wtc-channel-cross` pairs every long channel with every other across the two brains, 196 values instead of 14, and adds the channel x channel matrix of band means, the ROI x ROI matrix under `--roi-mapping`, and a per-brain selector on each map panel. Every map carries relative phase as arrows, so a pair moving together is distinguishable from one moving together a few seconds apart; they are drawn against the null's per-frequency level where one exists, and above the flat `--wtc-arrow-min` where none does.

Per-condition results are on by default: the coherence is read out of each task annotation's own window, one result per block, and `--no-by-condition` turns that off. Each window is read off the whole-run transform rather than transformed on its own, so it costs almost nothing and a short condition is not inflated by its own edges. `--wtc-cond-transform` transforms each condition separately instead, keeping `--wtc-cond-pad-s` seconds either side and windowing them back off; with the `auto` margin the numbers match the default route, so it is a form a methods section can describe rather than a different result. `--wtc-limit-scales`, on by default, computes only the scales inside the band plus margin, bit for bit identical to the unrestricted transform.

`--wtc-chroma` picks the chromophore(s), both by default: two parallel passes, HbO pairing only with HbO, so nothing is mixed and the cost is exactly twice. The point is a consistency check rather than two results: HbO has the better SNR, HbR is less contaminated by the respiration and heart rate two people in one room share, and coupling in HbO with nothing in HbR is a caution flag, though not a quantitative one, coherence being unsigned and bounded. One switch moves every panel between them, and every band-mean table carries a `chromophore` column.

`--wtc-pseudo N` adds the pseudo-dyad null: the same band means against a phase-scrambled partner, averaged over N iterations. Coherence between two unrelated recordings is not zero, so this is what a real value is read against, and each iteration being a full WTC run makes it the expensive half. It shares this run's stage, band and window by construction, and follows `--by-condition`'s windows at no extra transform, a short condition tested against a whole-record null looking further above chance than it is. Crossing is the one thing it does not share, being paid on every iteration; ask for it with `--wtc-pseudo-cross`. Each null table carries `null_sd`, `null_p95`, `n_iter` and each cell's `percentile`.

The correlation side has three flags of its own, all off by default. `--isc-whiten ORDER` correlates autoregressive residuals rather than the series, putting r back on the scale its sample count implies; it shrinks r by roughly a factor of six, so a whitened matrix does not compare with an unwhitened one. `--isc-max-lag SECONDS` keeps the strongest correlation over every shift within that many seconds either way and reports the winning shift, since two people's responses do not peak at the same instant. `--isc-pseudo N` ranks each correlation against N phase-scrambled surrogates, the null a maximum over many shifts needs. All three write into `hyper-iscpairs.tsv`, one row per channel pair. `--isc-threshold` forces an absolute cut on the connectogram; left alone, a chord is drawn where the pairing beats its own null, or for the strongest tenth when none was drawn.

`band` re-averages the maps `run --wtc-save-maps` saved over a different band, with no second wavelet transform, taking the same `--wtc-band-fmin` / `--wtc-band-fmax` / `--wtc-mask-coi` as `run` so the two cannot drift apart under two spellings.

`index` rebuilds `group-<id>_index.html`, one row per analysed window. `run` writes it too; this is for a tree produced earlier, or after the pages were regenerated by hand.

`merge` concatenates every per-dyad band-mean table into one per kind at the root of the tree, adding `group_id` and `task` columns, so a cohort analysis reads one file. It refuses to merge tables that disagree on the band, on the cone-of-influence masking, or on the null's iteration count.

### `fnirs-recon` — raw SNIRF → BIDS

```
fnirs-recon INPUT_FILE BIDS_DIR --participant-label LABEL --task-label LABEL
                                [--session-label LABEL] [--run-label INDEX]
                                [--optode-frame {unknown,head,mri}]
                                [--overwrite]
```

`--optode-frame` names the space the SNIRF's optode coordinates were measured in. SNIRF does not record it, and without it no `_optodes.tsv` or `_coordsystem.json` is written, both of which BIDS requires. Use `head` for positions digitised against the nasion and preauricular points.

### `fnirs-prep` — headless data-preparation utilities

```
fnirs-prep crop BIDS_DIR DERIVATIVES_DIR --participant-label SUB ...
                ( --tmin FLOAT [--tmax FLOAT] | --segments-path PATH [--combine] )
                [--align none|trigger] [--trigger-name TEXT]
                [--margin SEC|auto] [--band-fmin HZ] [--input-desc DESC]
                [--session-label / --task-label / --run-label]
                [--n-jobs INT] [--skip-bids-validation]

fnirs-prep align BIDS_DIR DERIVATIVES_DIR --group-csv PATH
                 [--skip-bids-validation]

fnirs-prep edit-markers export BIDS_DIR OUT_DIR --participant-label SUB ...
                               [--session-label / --task-label / --run-label] [--n-jobs INT]

fnirs-prep edit-markers apply BIDS_DIR DERIVATIVES_DIR --participant-label SUB ...
                              ( --tsv PATH | --shift FLOAT | --set-duration FLOAT
                                | --rename OLD:NEW ... )
                              [--session-label / --task-label / --run-label] [--n-jobs INT]
```

`crop` takes its window from `--tmin` / `--tmax` or from a segments table passed to `--segments-path`, which holds `onset` and `duration` plus an optional `task` column naming each segment's output task entity instead of the default `_seg-NN`. Under `--align trigger`, times are measured from the first annotation named by `--trigger-name`, so one window selects the same stretch of task in every subject.

`--margin` keeps extra seconds either side of every segment and records the span asked for in the sidecar: a segment cut to its own boundaries cannot be analysed at those boundaries by anything that convolves. `auto` takes the width from `--band-fmin`, the lowest frequency the later analysis will average over.

`--input-desc` cuts a processed stage instead of a recording (`errts`, `filtered`, and so on), `BIDS_DIR` then being a derivatives tree. This is the order to prefer: cutting first makes motion correction and the bandpass each see one condition.

### `fnirs-qc` — QC reports

`prep-raw` and `hyper-raw` read raw recordings, so both require `--cardiac-l-freq` / `--cardiac-h-freq`, population-dependent and without a default, and `--dpf`, used to convert to haemoglobin internally. Both screen channels and split the montage, so both take the same screening and separation flags as `fnirs-pipe`; pass what the run was prepped with.

```
fnirs-qc prep-raw BIDS_DIR OUTPUT_DIR --participant-label LABEL [LABEL ...]
                  --dpf FLOAT [FLOAT ...]
                  --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
                  [--session-label / --task-label]
                  [--sci-threshold FLOAT] [--psp-threshold FLOAT]
                  [--min-good-frac FLOAT]                        [default: 0.75]
                  [--screen-scope {run,task}]                    [default: run]
                  [--window-length FLOAT]                        [default: 10.0]
                  [--epoch-qc] [--epoch-tmin/--epoch-tmax FLOAT]
                  [--by-condition]
                  [--motion-correction {tddr,wavelet,spline,none}]  [default: none]
                  [--short-max-dist/--long-min-dist/--long-max-dist MM]
                  [--skip-bids-validation]

fnirs-qc hyper-raw BIDS_DIR OUTPUT_DIR --pairs-csv PATH
                   --dpf FLOAT [FLOAT ...]
                   --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
                   [--group-id / --task-label / --session-label]
                   [--sci-threshold FLOAT] [--psp-threshold FLOAT]
                   [--min-good-frac FLOAT] [--screen-scope {run,task}]
                   [--coh-fmin/--coh-fmax FLOAT]                 [default: 0.01 / 0.10]
                   [--short-max-dist/--long-min-dist/--long-max-dist MM]
                   [--normalize] [--no-align] [--tstart/--tend FLOAT]

fnirs-qc cohort       OUTPUT_DIR
fnirs-qc cohort-hyper OUTPUT_DIR
fnirs-qc provenance   OUTPUT_DIR
```

`prep-raw` takes more than one subject, one subject's failure does not stop the rest, and it writes the subject index too. Its `--by-condition` writes one report per annotated condition beside the run's own, the condition becoming the `task-` entity. Its `--motion-correction` runs that correction on a copy of the optical density and reports the recording either side of it, writing nothing back, so the screening verdict still describes the recording as delivered.

`cohort` puts every subject in a tree on one page. `cohort-hyper` is about dyads: how much of each recording both members could use at the same moment, split into one member's loss and the shared loss; where that time went, per channel pair and per condition; and each window's coherence as its rank inside its own null. `provenance` redraws the graphs from the sidecars already on disk.

`hyper-raw`'s per-subject quality table is the long-channel view the individual reports print, so a subject's SCI, CV, SNR and GVTD can be read against their own `sub-*_desc-raw` page.

Renamed since earlier versions: `--fmin` / `--fmax` on `hyper-raw` are now `--coh-fmin` / `--coh-fmax`, the old names still accepted as aliases; `hyper-post`, `hyper-null`, `wtc-band` and `group-hyper-wtc` are now `fnirs-hyper run`, its `--wtc-pseudo` flag, `fnirs-hyper band` and `fnirs-hyper merge`; `group-raw` and `group-hyper-raw` are now `cohort` and `cohort-hyper`, writing `cohort_nirs` and `cohort_hyper_nirs`.

For a cohort report over one time window, crop first and then run the usual pair of commands:

```
fnirs-prep crop BIDS_DIR DERIV_DIR --participant-label ... --tmin FLOAT --tmax FLOAT
                [--align none|trigger] [--trigger-name TEXT]
fnirs-qc   prep-raw DERIV_DIR/cropped OUTPUT_DIR --participant-label LABEL ... --dpf ...
                    --cardiac-l-freq FLOAT --cardiac-h-freq FLOAT
fnirs-qc   cohort OUTPUT_DIR
```

### `fnirs-rate` — Flask rating viewers

```
fnirs-rate rate  OUTPUT_DIR [--participant-label SUB ...] [--port INT]   # default 8765
fnirs-rate raw   OUTPUT_DIR --participant-label SUB
                 [--session-label / --task-label] [--sci-threshold FLOAT] [--port INT]   # default 5052
fnirs-rate hyper OUTPUT_DIR --group-id GROUP_ID --task-label TASK_LABEL --pairs-csv PATH
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
│   ├── sub-01_task-<t>_desc-<cond>_qc.html   # + --by-condition, one per condition
│   ├── figures/
│   │   └── sub-01_task-<t>/             # figures, one directory per run
│   │       ├── provenance.png           # provenance graph (embedded in that run's report)
│   │       └── provenance.mmd           # same graph, mermaid source
│   ├── logs/
│   │   ├── sub-01_{ts}.toml             # run record (env + params)
│   │   └── sub-01_script.py             # reproduction script
│   └── nirs/
│       ├── sub-01_desc-od_nirs.snirf              # prep step 1
│       ├── sub-01_desc-sci_nirs.snirf             # prep step 2 (channel screening)
│       ├── sub-01_desc-motcorrected_nirs.snirf    # prep step 3
│       ├── sub-01_desc-preproc_nirs.snirf         # prep step 4 (terminus)
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
├── cohort_nirs.{tsv,html}                # fnirs-qc cohort
└── cohort_hyper_nirs.{tsv,html}          # fnirs-qc cohort-hyper
```

The `fc*` files are written in rest mode, or in any mode run with `--fc`.

Standalone HTMLs from `fnirs-qc prep-raw` / `hyper-raw` and everything `fnirs-hyper run` writes sit at the derivatives root:

```
output/
├── sub-01_task-tapping_desc-raw_nirs.html               # fnirs-qc prep-raw
├── sub-01_task-tapping_raw_channel_decisions.json       # fnirs-rate raw sidecar
├── sub-01_task-tapping_raw_ratings.json
├── group-G1003_task-tapping_desc-hyperraw_nirs.html     # fnirs-qc hyper-raw
└── group-G1003/
    ├── group-G1003_index.html                           # one row per analysed window
    ├── group-G1003_task-tapping_desc-hyperpost_nirs.html
    ├── group-G1003_task-tapping_hyper-wtc.tsv           # band means, one row per channel pair
    ├── group-G1003_task-tapping_hyper-wtcbycond.tsv     # the same, one row per condition
    ├── group-G1003_task-tapping_hyper-wtc-*.npz         # + --wtc-save-maps
    ├── group-G1003_task-tapping_hyper-isc-hbo.tsv       # ISC matrix, per chromophore
    ├── group-G1003_task-tapping_hyper-isc-roichan-*.tsv # ROI-level ISC, per condition
    ├── group-G1003_task-tapping_hyper-iscpairs.tsv      # one row per channel pair
    ├── group-G1003_task-tapping_hyper-usable.tsv        # shared usable time
    └── figures/
```

A group of more than two members writes one report and one ISC table per pairing, suffixed `_<sub1>x<sub2>`. `fnirs-hyper merge` concatenates the band-mean tables into one file per kind at the root of the tree.

## QC Reports

`fnirs-pipe` writes an HTML report per run automatically, plus an index page per subject; `fnirs-qc` adds standalone, cohort, and hyperscanning reports:

| Report | Command | Level / stage |
|--------|---------|---------------|
| Per-run | (pipeline, automatic) | individual: raw + post, one report per run plus a subject index |
| Per-condition | `--by-condition`, pipeline or `prep-raw` | individual: one page per annotated condition |
| Raw pre-flight viewer | `fnirs-qc prep-raw` | individual — raw only |
| Cohort | `fnirs-qc cohort` | cohort — every subject in a tree |
| Dyad cohort | `fnirs-qc cohort-hyper` | cohort — every dyad in a tree |
| Time-window cohort | `fnirs-prep crop` then `prep-raw` + `cohort` | cohort — raw, cropped window |
| Per-trial | `fnirs-qc prep-raw --epoch-qc` | individual — SQM per task event, in the raw report |
| Dyad raw | `fnirs-qc hyper-raw` | hyperscanning — raw coherence |
| Dyad post | `fnirs-hyper run` | hyperscanning — post: WTC + ISC (ROI-level with `--roi-mapping`) |
| Dyad index | `fnirs-hyper run` / `index` | hyperscanning — one row per analysed window |

### Per-run report contents

- Executive summary with traffic-light badges (bad channel rate, mean SCI, HbO–HbR corr, GVTD p95), and a metrics panel whose tooltips say which stage each number was measured on
- SCI / PSP probe layout + windowed heatmap, coloured by the screening's verdict
- Carpet plot before / after motion correction, with the spike spans and the correction footprint drawn beneath. The GVTD trace covers the long channels, the set `--long-min-dist` / `--long-max-dist` define, and names its channel set and count on the figure; censored spans, if any, are drawn over it
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

Cohort and window reports add subject × metric heatmaps, per-scale grouped boxplots with clickable strip points (Tukey 1.5 × IQR outliers), and sortable tables. Dyad reports carry the coherence maps and matrices, the ISC matrix and its connectogram, the motion figures either side of correction, the usable-time breakdown, and one table holding every number behind the panels.

## Package Structure

```
fnirs_pipe/
  cli/          fnirs-pipe, fnirs-recon, fnirs-prep, fnirs-qc, fnirs-hyper, fnirs-rate,
                fnirs-gui, fnirs-log
  pipeline/     prep_pipeline, post_pipeline, glm, denoise, restingstate, motion,
                hyperscanning, hyper_post, synchrony, alignment, crop, edit_markers,
                wtc_store, wtc_null, wtc_aggregate, group_io, group_quality
  io/           BIDS layout, snirf read/write, derivatives output, snirf aux group,
                delimiter-sniffing table reader
  qc/
    common/     report shell, channel table, provenance graph, window grid, figure IO
    metrics/    coupling, screening, motion, GVTD, haemoglobin, windowed, hyper, aggregate
    figures/    Plotly and matplotlib panels, split common / subject / hyper
    subject/    per-run report, condition pages, subject index, SQM record, trial QC,
                cohort writer
    hyper/      dyad raw and post reports, dyad index, usable time, cohort-hyper writer
    rating/     Flask rating apps
    boilerplate/  Methods text and references
  interface/    Dash GUI (analysis, batch prep, data prep, hyper align, qc, recon)
  utils/        logging, run_record, job_db, lineage, run_script, snirf_prep
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
