---
title: Usage
---

# Usage

NIRSPipe is a set of command-line entry points. What each step does is described in [Processing workflow](workflows.md) and [Hyperscanning](hyperscanning.md); this page lists how to call them.

| Command | Purpose |
|---------|---------|
| [`nirspipe`](#nirspipe) | Preprocessing and postprocessing of a BIDS dataset (`participant`), or the cohort QC pages over its output (`group`) |
| [`nirspipe-recon`](#nirspipe-recon) | Convert one raw SNIRF file into BIDS |
| [`nirspipe-prep`](#nirspipe-prep) | Data preparation: crop, align dyads, edit markers, write span tables |
| [`nirspipe-qc`](#nirspipe-qc) | Raw, dyad and cohort QC reports, and the provenance graphs |
| [`nirspipe-hyper`](#nirspipe-hyper) | Dyad analysis: coherence, correlation, phase-scrambled null |
| [`nirspipe-hyper-pairnull`](#nirspipe-hyper-pairnull) | The re-paired null |
| [`nirspipe-hyper-groupnull`](#nirspipe-hyper-groupnull) | The nulls read with the channels averaged, per occasion and cohort |
| [`nirspipe-hyper-band`](#nirspipe-hyper-band) | Re-average saved coherence maps over another band |
| [`nirspipe-hyper-index`](#nirspipe-hyper-index) | Rebuild each dyad's landing page |
| [`nirspipe-hyper-merge`](#nirspipe-hyper-merge) | Concatenate the per-dyad long tables into one table per kind |
| [`nirspipe-rate`](#nirspipe-rate) | Rating viewers for the QC reports |
| [`nirspipe-gui`](#nirspipe-gui) | The desktop interface |
| [`nirspipe-log`](#nirspipe-log) | Merge the run logs into a SQLite database |

## Examples

The cardiac and respiration bands depend on the population and have no default, so every `nirspipe` run names them. The first example spells them out; the rest shorten them to `--cardiac-* --resp-*`.

Preprocessing only, with a report per run:

```bash
nirspipe /data/bids /data/derivatives participant \
    --dpf 6.0 --sci-threshold 0.8 \
    --cardiac-l-freq 0.7 --cardiac-h-freq 2.0 \
    --resp-l-freq 0.1 --resp-h-freq 0.5
```

A first-level GLM, the events taken from the SNIRF annotations with a fixed duration:

```bash
nirspipe /data/bids /data/derivatives participant \
    --dpf 6.0 --sci-threshold 0.8 --cardiac-* --resp-* \
    --mode glm --stim-dur 15 --drift-high-pass 0.01 \
    --short-channel mean --contrast-file contrasts.toml
```

Resting state, ALFF and connectivity, with regions:

```bash
nirspipe /data/bids /data/derivatives participant \
    --dpf 6.0 --sci-threshold 0.8 --cardiac-* --resp-* \
    --mode rest --short-channel mean --roi-mapping rois.json
```

A denoised residual for analyses done elsewhere, such as dyad coherence on task data:

```bash
nirspipe /data/bids /data/derivatives participant \
    --dpf 6.0 --sci-threshold 0.8 --cardiac-* --resp-* \
    --mode denoise --short-channel mean --high-pass 0.02 --low-pass 0.2
```

Children's faster heart rate needs a higher cardiac band, or good channels are rejected:

```bash
nirspipe /data/bids /data/derivatives participant \
    --dpf 6.0 --sci-threshold 0.8 \
    --cardiac-l-freq 1.0 --cardiac-h-freq 2.5 --resp-l-freq 0.2 --resp-h-freq 0.8
```

Then the cohort pages, which need no preprocessing flag:

```bash
nirspipe /data/bids /data/derivatives group
```

`--dry-run` resolves the settings, writes each subject's run record and stops. `--n-jobs 4` runs four subjects at once. One subject failing does not end a batch: the failed subjects are named at the end and the command exits non-zero.

## Conventions

- **Entity flags** are spelled the same on every command: `--participant-label`, `--session-label`, `--task-label`, `--run-label`, `--group-id`. A label may be given with its `sub-` / `ses-` / `task-` prefix or without. List options take space-separated values or the flag repeated. The BIDS App options also answer to their underscore spellings (`--participant_label`, `--work_dir`, ...).
- **Every command** takes `--version` and `--verbose`.
- **Tables** you supply are read by extension: `.tsv` tab-separated, `.csv` comma-separated, anything else sniffed ([Table files](data-preparation.md#table-files)).
- **BIDS validation.** Commands that read a BIDS dataset (`nirspipe`, `nirspipe-prep`, `nirspipe-qc prep-raw` and `hyper-raw`) check it with bids-validator when it is installed; `--skip-bids-validation` skips the check. Commands that read only a derivatives tree never validate.
- **Output directories.** A command that writes a tree stops when its output directory is its input directory; `nirspipe-prep crop --input-desc` is the exception.
- **Parallelism.** `--n-jobs` runs subjects on threads, each held to one BLAS thread. `nirspipe ... participant` and the `nirspipe-prep` subcommands `crop`, `edit-markers` and `keep-spans` take it.

## Configuration file

`--config` takes a TOML file of settings, so a run can be kept under version control and run again. Every run writes one: its run record, `sub-<label>/logs/sub-<label>.toml`, is in this format, so `--config` on a record reruns the run it records.

```toml
[prep]
dpf = [6.0]
sci_threshold = 0.8
cardiac_l_freq = 0.7
cardiac_h_freq = 2.0
resp_l_freq = 0.1
resp_h_freq = 0.5

[post]
mode = "glm"
high_pass = 0.01
low_pass = 0.5
noise_model = "auto"
drift_model = "cosine"
drift_high_pass = 0.01
events_path = "path/to/events.tsv"
```

```bash
nirspipe /data/bids /data/out participant --config study_config.toml

# rerun a subject exactly, or with one setting changed
nirspipe /data/bids /data/out2 participant --config /data/out/sub-01/logs/sub-01.toml --noise-model ar1
```

Keys are flag names without the leading `--` and with hyphens turned into underscores, in the flag's own units. `[prep]` holds the preprocessing flags, the cardiac, respiration and separation bands among them; `[post]` holds `mode`, the postprocessing flags and `roi_mapping`; `[execution]` holds the selection (`participant_label`, `session_label`, `task_label`, `bids_filter_file`, `ignore`, `allow_cropped_input`). A key outside a section, an unknown section or key, or a value the flag would refuse stops the run before anything is read, naming the key. The input and output directories and the analysis level always come from the command line.

### Mode defaults

Each post setting is taken from the first layer that has it: the command line, then `--config`, then the mode's defaults below, then the code default (mostly off). A preprocessing setting has no mode layer.

| Setting | `denoise` | `glm` | `rest` |
|---|---|---|---|
| `high_pass` | 0.01 | none | 0.01 |
| `low_pass` | 0.5 | none | 0.08 |
| `drift_model` | none | `cosine` | `cosine` |
| `drift_high_pass` | none | **required** | 0.01 |
| `noise_model` | `ols` | `auto` | `ols` |
| `hrf_model` | | `spm` | |

No mode fills a value that depends on the design, the montage or the population: the cosine cutoff of a task design, `short_channel`, and the cardiac and respiration bands. To switch a mode's value off, say so: `--high-pass none`, `--drift-model none`, `--short-channel none`, or `high_pass = "none"` in the file. The run record's `[post_sources]` names the layer each value came from, and the run report lists every setting nobody typed.

### Contrast file

`--contrast-file` names a TOML file of linear contrasts over the condition regressors, one top-level key per contrast, mapping `trial_type` values to weights:

```toml
task_vs_rest = { task = 1.0, rest = -1.0 }
left_minus_right = { tap_left = 1.0, tap_right = -1.0 }
```

Written to `<run>_desc-contrast_nirsmap.tsv`, one row per contrast and channel.

## `nirspipe`

`participant` preprocesses each selected subject, and postprocesses it when `--mode` is given. `group` reads the quality records under `OUTPUT_DIR` and writes the cohort pages; it reads nothing from `BIDS_DIR` and needs no preprocessing flag. The run stops before writing anything when the output directory is the input, when `--work-dir` lies inside the input, or when `--participant-label` names a subject the dataset does not have. What each step does is in [Processing workflow](workflows.md).

```{argparse}
:module: cli_parsers
:func: nirspipe
:prog: nirspipe
```

## `nirspipe-recon`

Converts one raw SNIRF file per call ([Converting raw SNIRF](data-preparation.md#converting-raw-snirf)).

```{argparse}
:module: cli_parsers
:func: nirspipe_recon
:prog: nirspipe-recon
```

## `nirspipe-prep`

`crop`, `align` and `edit-markers apply` each write a new tree under `OUTPUT_DIR` (`cropped/`, `aligned/`, `marker_edited/`); `edit-markers export` writes event tables for editing and `keep-spans` one table. What each does is in [Data preparation](data-preparation.md).

```{argparse}
:module: cli_parsers
:func: nirspipe_prep
:prog: nirspipe-prep
```

## `nirspipe-qc`

What each report shows is in [QC](qc.md). `hyper-raw` writes into `OUTPUT_DIR`: the dyad tree, or a raw QC tree of its own, which it stamps `GeneratedBy: nirspipe-qc hyper-raw`. A `nirspipe` tree is refused, so one tree never holds both `sub-*` and `group-*`.

```{argparse}
:module: cli_parsers
:func: nirspipe_qc
:prog: nirspipe-qc
```

## `nirspipe-hyper`

`DERIVATIVES_DIR` is the `nirspipe` tree and `OUTPUT_DIR` the dyad tree; they must differ. What each step does is in [Hyperscanning](hyperscanning.md).

```{argparse}
:module: cli_parsers
:func: nirspipe_hyper
:prog: nirspipe-hyper
```

## `nirspipe-hyper-pairnull`

Runs after `nirspipe-hyper`, whose tables it reads ([Re-paired null](hyperscanning.md#re-paired-null)).

```{argparse}
:module: cli_parsers
:func: nirspipe_hyper_pairnull
:prog: nirspipe-hyper-pairnull
```

## `nirspipe-hyper-groupnull`

Reads the null draws already written and runs no transform ([Above the channel](hyperscanning.md#above-the-channel-nirspipe-hyper-groupnull)).

```{argparse}
:module: cli_parsers
:func: nirspipe_hyper_groupnull
:prog: nirspipe-hyper-groupnull
```

## `nirspipe-hyper-band`

Re-averages the maps `nirspipe-hyper --wtc-save-maps` saved over another band.

```{argparse}
:module: cli_parsers
:func: nirspipe_hyper_band
:prog: nirspipe-hyper-band
```

## `nirspipe-hyper-index`

```{argparse}
:module: cli_parsers
:func: nirspipe_hyper_index
:prog: nirspipe-hyper-index
```

## `nirspipe-hyper-merge`

```{argparse}
:module: cli_parsers
:func: nirspipe_hyper_merge
:prog: nirspipe-hyper-merge
```

## `nirspipe-rate`

`hyper` takes the tree `nirspipe-qc hyper-raw` wrote as `OUTPUT_DIR` (the dyad tree or a raw QC tree of its own), and `--derivatives-dir` names the `nirspipe` tree its channel decisions are written to. See [Rating](qc.md#rating).

```{argparse}
:module: cli_parsers
:func: nirspipe_rate
:prog: nirspipe-rate
```

## `nirspipe-gui`

Starts the desktop interface and prints the address to open ([GUI](gui.md)).

```{argparse}
:module: cli_parsers
:func: nirspipe_gui
:prog: nirspipe-gui
```

## `nirspipe-log`

`merge` adds every finished run's logs to `OUTPUT_DIR/logs/nirspipe.db` and archives them; `rebuild` writes a new, timestamped database from every log and leaves the existing one alone. See [Event stream and database](outputs.md#event-stream-and-database).

```{argparse}
:module: cli_parsers
:func: nirspipe_log
:prog: nirspipe-log
```

## Troubleshooting

`--verbose` prints debug output, including each step's timing and paths; the same stream goes to `sub-<label>/logs/sub-<label>.log`. Rerunning a subject overwrites its outputs and run record; to keep both, write to a second output directory. A rerun that changes the passband, the resampling rate or the drift basis warns before overwriting, since those files' names do not show the change.

| Message | Cause and fix |
|---------|---------------|
| `... is not valid BIDS` | bids-validator found errors; the message lists them. Check `dataset_description.json`, the `sub-<label>/[ses-<label>/]nirs/` layout and the file names, or pass `--skip-bids-validation`. |
| `fsaverage is needed to place the optodes on the brain` | MNE's fsaverage is downloaded the first time a brain figure is drawn, and this machine could not fetch it, usually for lack of internet access. The rest of the report is unaffected. |
| `Spline correction needs at least 3.333 Hz` | Spline correction does not support recordings sampled below 10/3 Hz; choose another `--motion-correction`. |
| Every channel marked bad | The `--sci-threshold` is too high for the data, or the cardiac band does not match the population's heart rate. |
| `GroupCSVError` | The pairs file is missing `group_id`, `subject_id` or `task`, is empty, or has a group of one. |
| `AlignmentError: No shared trigger descriptions` | The members carry no annotation in common. Rename the markers into one scheme, or use `--no-align` for recordings without triggers. |
| `MissingDerivativesError` | A member lacks the `--desc` stage in the `nirspipe` tree; run `nirspipe` for every member first. |
| A report with missing figures | A figure that fails leaves a placeholder; its error is under Errors & Warnings, and the traceback in the `--verbose` log. |
| A GLM with no results | Check that exactly one of `--stim-dur` and `--events-path` is set and that the conditions exist in the events. |
| A rating port in use | Pass `--port`. |
