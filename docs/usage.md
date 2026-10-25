---
title: Usage
---

# Usage

NIRSPipe is a set of command-line entry points. What each step does is described in [Processing workflow](workflows.md) and [Hyperscanning](hyperscanning.md); this page lists how to call them.

| Command | Purpose |
|---------|---------|
| [`nirspipe`](#nirspipe) | Preprocessing and postprocessing of a BIDS dataset (`participant`), or the cohort QC pages over its output (`group`) |
| [`nirspipe-snirf2bids`](#nirspipe-snirf2bids) | Convert one raw SNIRF file into BIDS |
| [`nirspipe-prep`](#nirspipe-prep) | Data preparation: crop, align dyads, edit markers, write span tables |
| [`nirspipe-qc`](#nirspipe-qc) | Raw QC as a BIDS App: each recording's report (`participant`), then the cohort and dyad pages (`group`) |
| [`nirspipe-provenance`](#nirspipe-provenance) | Redraw the provenance graphs of a tree from its sidecars |
| [`nirspipe-hyper`](#nirspipe-hyper) | Dyad analysis: coherence, correlation, phase-scrambled null |
| [`nirspipe-hyper-pairnull`](#nirspipe-hyper-pairnull) | The re-paired null |
| [`nirspipe-hyper-groupnull`](#nirspipe-hyper-groupnull) | The nulls read with the channels averaged, per occasion and cohort |
| [`nirspipe-hyper-band`](#nirspipe-hyper-band) | Re-average saved coherence maps over another band |
| [`nirspipe-hyper-index`](#nirspipe-hyper-index) | Rebuild each dyad's landing page |
| [`nirspipe-hyper-merge`](#nirspipe-hyper-merge) | Concatenate the per-dyad long tables into one table per kind |
| [`nirspipe-rate`](#nirspipe-rate) | Rating viewers for the QC reports |
| [`nirspipe-gui`](#nirspipe-gui) | The desktop interface |

## Examples

The cardiac and respiration bands, the DPF and the head template depend on the participants' age. `--population` fills them for an age group, and a flag typed for one of them overrides it ([Population](#population)). `--sci-threshold` is never filled: every run names its own line.

Preprocessing only, with a report per run:

```bash
nirspipe /data/bids /data/derivatives participant --population adult --sci-threshold 0.8
```

A first-level GLM, the events taken from the SNIRF annotations with a fixed duration. Short-channel regression runs by default on a montage with short channels:

```bash
nirspipe /data/bids /data/derivatives participant --population adult --sci-threshold 0.8 \
    --mode glm --stim-dur 15 --drift-high-pass 0.01 --contrast-file contrasts.tsv
```

Resting state, ALFF and connectivity, with regions:

```bash
nirspipe /data/bids /data/derivatives participant --population adult --sci-threshold 0.8 \
    --mode rest --roi-mapping rois.json
```

A denoised residual for analyses done elsewhere, such as dyad coherence on task data:

```bash
nirspipe /data/bids /data/derivatives participant --population adult --sci-threshold 0.8 \
    --mode denoise --high-pass 0.02 --low-pass 0.2
```

Children, whose faster heart rate needs a higher cardiac band or good channels are rejected:

```bash
nirspipe /data/bids /data/derivatives participant --population child --sci-threshold 0.8
```

An ageing study from 20 to 80 years, one DPF per participant from their age in `participants.tsv`:

```bash
nirspipe /data/bids /data/derivatives participant --population adult --dpf age --sci-threshold 0.8
```

An exercise task, which raises the heart rate past the resting band:

```bash
nirspipe /data/bids /data/derivatives participant --population adult --cardiac-h-freq 3.0 \
    --sci-threshold 0.8
```

Every value typed, without `--population`:

```bash
nirspipe /data/bids /data/derivatives participant \
    --dpf 6.0 --sci-threshold 0.8 \
    --cardiac-l-freq 0.7 --cardiac-h-freq 1.8 --resp-l-freq 0.1 --resp-h-freq 0.5
```

Then the cohort pages, which need no preprocessing flag:

```bash
nirspipe /data/bids /data/derivatives group
```

`--dry-run` resolves the settings, writes each subject's run record and stops. `--nprocs 4` lets the run use four cores: four subjects at once, or fewer subjects with the spare cores going to each one's numerics. One subject failing does not end a batch: the failed subjects are named at the end and the command exits non-zero.

## Conventions

- **Entity flags** are spelled the same on every command: `--participant-label`, `--session-label`, `--task-label`, `--run-label`, `--group-id`. A label may be given with its `sub-` / `ses-` / `task-` prefix or without. List options take space-separated values or the flag repeated. The BIDS App options also answer to their underscore spellings (`--participant_label`, `--bids_filter_file`, ...).
- **Every command** takes `--version` and `--verbose`.
- **Tables** you supply are read by extension: `.tsv` tab-separated, `.csv` comma-separated, anything else sniffed ([Table files](data-preparation.md#table-files)).
- **BIDS validation.** Commands that read a BIDS dataset (`nirspipe`, `nirspipe-prep` and `nirspipe-qc` at both levels) check it with bids-validator when it is installed; `--skip-bids-validation` skips the check. Commands that read only a derivatives tree never validate.
- **Output directories.** A command that writes a tree stops when its output directory is its input directory; `nirspipe-prep crop --input-desc` is the exception.
- **Parallelism.** `--nprocs` (also `--n_cpus`) is the number of cores a command may use in all. Subjects run on threads, as many at once as there are cores or subjects, and the cores left over are shared out as BLAS threads. Without it, subjects run one at a time and BLAS uses every core. `nirspipe ... participant`, `nirspipe-qc ... participant` and the `nirspipe-prep` subcommands `crop`, `edit-markers` and `keep-spans` take it. The dyad commands `nirspipe-hyper`, `nirspipe-hyper-pairnull` and `nirspipe-qc ... group` take it too and run each group in a process of its own, so memory grows with the number of groups at once.

## Configuration file

`--config` takes a TOML file of settings, so a run can be kept under version control and run again. Every run writes one: its run record, `sub-<label>/logs/sub-<label>.toml` (with `_ses-` and `_task-` added when `--session-label` or `--task-label` chose part of the subject), is in this format, so `--config` on a record reruns the run it records.

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

`nirspipe-hyper --config` reads a dyad's run record, `group-<id>/logs/group-<id>_task-<task>.toml`, the same way: its `[hyper]` section, keyed by flag name, fills every setting the command line leaves out, and the directories and `--pairs-csv` come from the command line. A band or crossing the run took from `--band-fmin`/`--band-fmax` or `--channel-cross` follows them when one of those is typed over the record.

```bash
nirspipe-hyper /data/out /data/hyper2 group --pairs-csv pairs.csv \
  --config /data/hyper/group-G01/logs/group-G01_task-hold.toml --band-fmin 0.05
```

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

No mode fills a value that depends on the design, the montage or the population: the cosine cutoff of a task design; short-channel regression, which defaults to `auto` in every mode (pca where the run's montage has a channel at or under `--short-max-dist`, none where it has none); and the cardiac and respiration bands, which `--population` fills. To switch a mode's value off, say so: `--high-pass none`, `--drift-model none`, `--short-channel none`, or `high_pass = "none"` in the file. The run record's `[post_sources]` names the layer each value came from, and the run report lists every setting nobody typed.

### Population

`--population` fills the values that follow the participants' age, where nothing else set them: the command line wins, then `--config`, then the population. It is ours, after QSIPrep's `--infant`; no BIDS App has adult, child and older groups.

| `--population` | Ages | Cardiac band (Hz) | Respiration band (Hz) | DPF | Head template |
|---|---|---|---|---|---|
| `infant` | under 1 year | 1.5 to 3.1 | 0.3 to 1.1 | the age equation at 0.5 years | each participant's nearest infant template, by their age in `participants.tsv` |
| `child` | 1 to 12 years | 0.8 to 2.6 | 0.2 to 0.9 | at 8 years | fsaverage, as MNE ships no template for 2 to 18 years |
| `adult` | 12 to 65 years | 0.7 to 1.8 | 0.1 to 0.5 | at 30 years | fsaverage |
| `older` | 65 years and over | 0.7 to 1.8 | 0.1 to 0.5 | at 70 years | fsaverage |

Each band is the union of the rows of the [Channel screening](workflows.md#channel-screening) table the ages span. The DPF is the age equation of the [Beer-Lambert law](workflows.md#beer-lambert-law) at one age, one value per wavelength for the whole batch; `--dpf age` takes each participant's own age instead, for a cohort wider than one group. `older` differs from `adult` in its DPF only: resting heart and breathing rates are flat across adult ages, while the DPF at 850 nm is 5.23 at 30 years and 6.30 at 70, so an older cohort run as adult reads amplitudes about 20% high. Under `infant`, a participant without an age in `participants.tsv` stops the run before its first subject: add the age, or pass `--template`. `nirspipe-qc ... participant` takes it the same way for its cardiac band, DPF and template. The run record lists what the population filled under `[prep_sources]` and leaves its DPF out of `[prep]`, so a rerun's `--population` sets it again; the run report lists those values with the other settings nobody typed, and the Methods say which values were the population's.

### Contrast file

`--contrast-file` names a table of linear contrasts over the condition regressors: a TSV or CSV with one row per contrast, its name in a `contrast` column and one column per `trial_type` value holding its weight. An empty cell leaves that condition out of the contrast (the columns are aligned here for reading; the file separates them with tabs or commas):

```
contrast          task  rest  tap_left  tap_right
task_vs_rest      1     -1
left_minus_right              1         -1
```

A TOML file of the same contrasts, one entry per contrast mapping `trial_type` values to weights, is read as well:

```toml
task_vs_rest = { task = 1.0, rest = -1.0 }
left_minus_right = { tap_left = 1.0, tap_right = -1.0 }
```

A cell that is not a number, a contrast named twice or one that weights nothing stops the run before its first subject, naming the line; a condition the design does not carry stops the subject. Written to `<run>_desc-contrast_nirsmap.tsv`, one row per contrast and channel.

## `nirspipe`

`participant` preprocesses each selected subject, and postprocesses it when `--mode` is given. `group` reads the quality records under `OUTPUT_DIR` and writes the cohort pages; it reads nothing from `BIDS_DIR` and needs no preprocessing flag. The run stops before writing anything when the output directory is the input, or when `--participant-label` names a subject the dataset does not have. What each step does is in [Processing workflow](workflows.md).

```{argparse}
:module: cli_parsers
:func: nirspipe
:prog: nirspipe
```

## `nirspipe-snirf2bids`

Converts one raw SNIRF file per call ([Converting raw SNIRF](data-preparation.md#converting-raw-snirf)).

```{argparse}
:module: cli_parsers
:func: nirspipe_snirf2bids
:prog: nirspipe-snirf2bids
```

## `nirspipe-prep`

`crop`, `align` and `edit-markers apply` each write a new tree under `OUTPUT_DIR` (`cropped/`, `aligned/`, `marker_edited/`); `edit-markers export` writes event tables for editing and `keep-spans` one table. What each does is in [Data preparation](data-preparation.md).

```{argparse}
:module: cli_parsers
:func: nirspipe_prep
:prog: nirspipe-prep
```

## `nirspipe-qc`

What each report shows is in [QC](qc.md). Run the `participant` level on every subject first, then `group` on the same `OUTPUT_DIR`: the cohort pages and, with `--pairs-csv`, each dyad's page, read from the members' records rather than screened again ([The raw QC tree](outputs.md#the-raw-qc-tree)). A flag typed at the level it does not belong to is refused. The participant level writes a run record per subject, which `--config` reruns; the group level takes no screening flag.

```bash
nirspipe-qc /data/bids /data/qc participant --population adult --sci-threshold 0.8
nirspipe-qc /data/bids /data/qc group --pairs-csv pairs.csv
```

```{argparse}
:module: cli_parsers
:func: nirspipe_qc
:prog: nirspipe-qc
```

## `nirspipe-provenance`

Redraws the provenance graph of every subject and dyad in a tree from the sidecars already on disk, so it works on any past run and refreshes the image a written report points at ([Outputs](outputs.md#logs-and-run-records)).

```{argparse}
:module: cli_parsers
:func: nirspipe_provenance
:prog: nirspipe-provenance
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

`hyper` takes the raw QC tree `nirspipe-qc ... group` wrote the dyad's page into as `OUTPUT_DIR`, and `--derivatives-dir` names the `nirspipe` tree its channel decisions are written to. See [Rating](qc.md#rating).

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

## Troubleshooting

`--verbose` prints debug output, including each step's timing and paths; the same stream goes to the subject's live log in `sub-<label>/logs/`, named like its run record. Rerunning a subject overwrites its outputs, and rerunning the same selection its run record; to keep both, write to a second output directory. A rerun that changes the passband, the resampling rate or the drift basis warns before overwriting, since those files' names do not show the change.

| Message | Cause and fix |
|---------|---------------|
| `... is not valid BIDS` | bids-validator found errors; the message lists them. Check `dataset_description.json`, the `sub-<label>/[ses-<label>/]nirs/` layout and the file names, or pass `--skip-bids-validation`. |
| `fsaverage is needed to place the optodes on the brain` | MNE's fsaverage is downloaded the first time it is needed, to place the optodes or to draw a brain figure, and this machine could not fetch it, usually for lack of internet access. Preprocessing goes on and writes `n/a` channel positions; the rest of the report is unaffected. |
| `no head template fits age ...` | Under `--dpf age` a participant is over 2 and under 18 years old, where no head template fits. Pass `--template` with the head to use for that study. |
| `the ... infant template could not be loaded` | The infant template a participant needs could not be downloaded. Fetch it on a machine with internet access with `mne.datasets.fetch_infant_template("<age>", subjects_dir)` and copy its folder next to fsaverage's. |
| `Spline correction needs at least 3.333 Hz` | Spline correction does not support recordings sampled below 10/3 Hz; choose another `--motion-correction`. |
| Every channel marked bad | The `--sci-threshold` is too high for the data, or the cardiac band does not match the population's heart rate. |
| `GroupCSVError` | The pairs file is missing `group_id`, `subject_id` or `task`, is empty, or has a group of one. |
| `AlignmentError: No shared trigger descriptions` | The members carry no annotation in common. Rename the markers into one scheme, or use `--no-align` for recordings without triggers. |
| `MissingDerivativesError` | A member lacks the `--desc` stage in the `nirspipe` tree; run `nirspipe` for every member first. |
| A report with missing figures | A figure that fails leaves a placeholder; its error is under Errors & Warnings, and the traceback in the `--verbose` log. |
| A GLM with no results | Check that exactly one of `--stim-dur` and `--events-path` is set and that the conditions exist in the events. |
| A rating port in use | Pass `--port`. |
