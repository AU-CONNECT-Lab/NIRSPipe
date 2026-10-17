---
title: Data preparation
---

# Data preparation

`nirspipe` reads a BIDS fNIRS dataset. This page covers getting a dataset into that shape and preparing the files the commands take beside it: event markers, cropped or aligned copies, tables of stretches to keep or drop, the pairs file for dyads and the ROI mapping. None of the commands here runs preprocessing. Their flags are listed in [Usage](usage.md).

## The BIDS input

```
bids_dir/
├── dataset_description.json
├── participants.tsv
└── sub-<label>/
    └── [ses-<label>/]
        └── nirs/
            ├── sub-<label>[_ses-<label>][_task-<label>][_run-<index>]_nirs.snirf
            └── sub-<label>[_ses-<label>][_task-<label>][_run-<index>]_nirs.json
```

The `_nirs.json` sidecar needs at least `SamplingFrequency`, `NIRSChannelCount`, `NIRSSourceOptodeCount` and `NIRSDetectorOptodeCount`. The dataset is checked with bids-validator at the start of a run ([Installation](installation.md#bids-validator)).

| Optional file | Used for |
|------|---------|
| `participants.tsv` | Subject metadata; an `age` column is shown in the QC report. |
| `*_events.tsv` | Task events. Without one, the pipeline reads the annotations embedded in the SNIRF. |
| `*_channels.tsv` | A channel whose `status` is `bad` is marked bad in preprocessing ([Channel screening](workflows.md#channel-screening)). |
| `*_optodes.tsv`, `*_coordsystem.json` | Optode positions, needed for the topographic maps and the 3D brain figures. |

## Converting raw SNIRF

`nirspipe-snirf2bids` converts one raw SNIRF into the BIDS layout above, one file per call:

```bash
nirspipe-snirf2bids sub01_run1.snirf /data/bids --participant-label 01 --task-label tapping

# with session and run
nirspipe-snirf2bids sub01_ses1_run1.snirf /data/bids \
    --participant-label 01 --session-label wave1 --run-label 01 --task-label tapping

# a folder of files
for f in raw/*.snirf; do
    subj=$(basename "$f" .snirf | grep -oP 'sub\K\d+')
    nirspipe-snirf2bids "$f" /data/bids --participant-label "$subj" --task-label tapping
done
```

Channel types and optode positions are read from the SNIRF by MNE-BIDS. SNIRF does not record which space its optode coordinates were measured in, so pass `--optode-frame`: without it no `_optodes.tsv` or `_coordsystem.json` is written, and BIDS requires both. Use `head` for positions digitised against the nasion and preauricular points. BIDS has no group folders, so encode patient and control in the subject label if their IDs overlap. `nirspipe-snirf2bids` does not validate what it writes; `nirspipe` does at the start of a run. The GUI's SNIRF to BIDS page converts a whole folder from one table ([GUI](gui.md)).

## Trees written by `nirspipe-prep`

`nirspipe-prep crop`, `align` and `edit-markers apply` each write a new tree under `OUTPUT_DIR` (`cropped/`, `aligned/`, `marker_edited/`) with its own `dataset_description.json`, which links the input dataset under `DatasetLinks` and `SourceDatasets`. Point `OUTPUT_DIR` at a derivatives root rather than at a dataset; the command refuses an output directory that is its input. Each tree reads as a BIDS dataset of its own and can be the input of the next command.

## Editing markers

`nirspipe-prep edit-markers export` writes each recording's events as a BIDS `_events.tsv` for editing by hand. `edit-markers apply` applies one operation to every selected recording and writes the edited copies to `marker_edited/`:

| Operation | Effect |
|-----------|--------|
| `--tsv PATH` | Replace the events with this table, the same file for every recording. |
| `--shift SEC` | Add this many seconds to every onset; negative values are allowed, and results are clipped at 0. |
| `--set-duration SEC` | Give every marker this duration. |
| `--rename OLD:NEW ...` | Rename marker descriptions, several pairs at once or the flag repeated. |

One operation per call; run it again on its own output to chain them. Dyad alignment needs every member to carry the trigger under the same name, so renaming is often the step before aligning:

```bash
nirspipe-prep edit-markers apply /data/bids /data/out --participant-label 01 02 \
    --rename "Stim1:tap_left" --rename "Stim2:tap_right"
nirspipe-prep align /data/bids /data/out/marker_edited --group-csv pairs.csv
```

## Cropping

`nirspipe-prep crop` cuts each selected recording to a window and writes it to `cropped/`. The window is `--tmin` / `--tmax` in seconds, or a segments table (`--segments-path`) with `onset` and `duration` columns, one row per segment. A `task` column names each segment, and its output carries that task; more than one segment needs it, unless `--combine` joins them into one file. The table is checked before any subject runs. With `--align trigger`, times are measured from the first annotation named `--trigger-name` rather than from the start of the recording, so one window selects the same stretch of task in recordings that started at different moments. A recording missing the trigger falls back to its own start, with a warning. The recording's aux group travels with each segment.

**Cut a processed stage rather than the raw recording.** With `--input-desc` (`preproc`, `errts`, `filtered`, ...) `crop` reads a derivatives tree and cuts that stage, keeping its `desc-` entity. This is the order to prefer: preprocess the whole recording, then cut the result, so motion correction and the bandpass see the whole recording rather than one segment. `nirspipe` refuses a cropped tree as input unless `--allow-cropped-input` is given.

**Keep a margin for analyses that convolve.** A segment cut to its own boundaries cannot be analysed at those boundaries by a wavelet transform or a filter. `--margin SEC` keeps that many extra seconds either side of every segment and records the span that was asked for in the sidecar; `--margin auto` takes the width from `--band-fmin`, the lowest frequency the later analysis will average over, as `2*sqrt(2)/f` (47 s at 0.06 Hz, 141 s at 0.02 Hz).

A cohort QC report over one window of time is a crop followed by the usual QC commands:

```bash
nirspipe-prep crop /data/bids /data/deriv --participant-label 01 02 \
    --tmin 0 --tmax 30 --align trigger --trigger-name task_block_1
nirspipe-qc prep-raw /data/deriv/cropped /data/out --participant-label 01 02 \
    --dpf 6.0 --cardiac-l-freq 0.7 --cardiac-h-freq 2.0 --sci-threshold 0.8
nirspipe-qc cohort /data/out
```

## Aligning dyads

`nirspipe-prep align` puts every member of each `(group_id, task)` in a group file on one clock: each member is cut at its first occurrence of a trigger all members carry, and all are trimmed to the shortest. It writes one aligned SNIRF per member, with its events and sidecars, to `aligned/`, and records the member's offset in its own `_nirs.json` (`align_group`, `align_step`, `align_trigger`, `align_offset_s`, `aligned_duration_s`). The group file has the format of the [pairs file](#pairs-file).

The tree holds one copy per recording, so a member two groups share for the same task is refused for the second group: align such groups into separate trees. A group with no shared trigger is skipped with a warning, the others continue, and the command exits non-zero when any group failed. Alignment is optional for the dyad commands, which align on the shared trigger themselves ([Hyperscanning](hyperscanning.md#alignment)).

## Tables that name recordings

Several flags take a table whose rows may apply to some recordings only: `--bad-channels` (when given a file), `--keep-spans` and `--bad-spans`. Optional `participant_id`, `session`, `task` and `run` columns limit a row:

- A blank cell or an absent column means every recording.
- Labels may keep or drop their `sub-`, `ses-`, `task-` and `run-` prefixes, and run numbers compare as numbers, so `1` and `01` are one run.
- A recording gets the union of every row it matches.
- A row that matches no recording, or a column the table does not define, stops the run before any subject starts, naming the line.

For `--bad-channels`, the other column is `bad_channels`, holding source-detector labels such as `S1_D1,S2_D3`; naming either wavelength of a pair marks both. An optional `keep_channels` column names pairs to keep although screening rejected them; `nirspipe-rate export` writes it from the pairs a reviewer rated good ([Rating](qc.md#rating)). A keep overrules screening only: `channels.tsv` and non-finite samples still mark the pair bad, with a warning, and a pair both marked bad and kept for one recording stops the run.

## Choosing stretches of a recording

`--keep-spans` keeps only the listed stretches of each recording, and `--bad-spans` marks listed stretches as damaged ([Excluded time](workflows.md#excluded-time)). Stretches can also be marked by eye on the raw report, and `nirspipe-rate export` writes them as a `--bad-spans` table ([Rating](qc.md#rating)). Both tables have `onset` and `duration` columns in seconds on the recording's own time axis, the axis of its `events.tsv` (not that of a cropped copy), plus the label columns above:

```
participant_id	task	run	onset	duration
01	rest	1	60	300
01	rest	1	600	240
02	rest		0	480
```

Here sub-01's first rest run keeps 60 to 360 s and 600 to 840 s, every rest run of sub-02 keeps its first 8 minutes, and every other recording is used whole. Overlapping rows merge, a stretch running past the end of a recording is clipped, and an onset past the end stops the run. A recording no row matches is used whole under `--keep-spans` and has no span under `--bad-spans`.

**Measuring from a marker.** One onset means a different moment in recordings that did not start together, such as the two members of a dyad. `--keep-spans-from MARKER` measures every `--keep-spans` onset from the first annotation named `MARKER` in that recording, so one row serves them all:

```
task	onset	duration
talk	-30	330
```

With `--keep-spans-from start`, every talk recording keeps from 30 s before its first `start` marker to 300 s after it. A negative onset is allowed as long as the stretch starts inside the recording. `--bad-spans` is always read on each recording's own axis.

**Writing the table from condition names.** `nirspipe-prep keep-spans` writes a `--keep-spans` table holding every occurrence of the conditions you name, read from each recording's markers the way the pipeline reads them, with the recording's own onsets:

```bash
nirspipe-prep keep-spans /data/bids keep.tsv --participant-label 01 02 --conditions rest talk --duration 120
nirspipe /data/bids /data/out participant --keep-spans keep.tsv ...
```

Its columns are `participant_id`, whichever of `session`, `task` and `run` the recordings have, then `onset` and `duration`. Each occurrence keeps its marker's own length unless `--duration` gives them all one. Some recordings store every marker with a nominal length, often 1 s, rather than the length of its block, and the table then holds 1 s stretches without any error: check the `duration` column. The onsets are already each recording's own, so the table is used without `--keep-spans-from`.

## Pairs file

The dyad commands take a table assigning subjects to groups, passed as `--pairs-csv` (`--group-csv` on `nirspipe-prep align`). Each row puts one subject's recording of one task in one group:

```text
group_id,subject_id,task
G01,sub-01,tap
G01,sub-02,tap
G02,sub-03,tap
G02,sub-04,tap
G01,sub-01,rest
G01,sub-02,rest
```

| Column | Notes |
|--------|-------|
| `group_id` | Any group label, e.g. `G01`. |
| `subject_id` | BIDS subject label, with or without `sub-`. |
| `task` | BIDS task label, used both to find the recording and as the alignment scope. |
| `session` | Optional. Picks the recording where a subject has several, and **keys the group**: a group recorded in two sessions is two dyads, each named `ses-<s>` throughout. |
| `run` | Optional. Picks the recording where a subject has several runs. |
| `occasion` | Optional. The sitting, where the members of one group carry different session labels for it. Keys the group in place of `session` and names its outputs `ses-<occasion>`, while each member's `session` still picks that member's file. Letters and digits only; a blank cell falls back to the row's `session`. |

- Each group needs at least two members; a group of one raises `GroupCSVError`. A group of three or more is analysed one member pairing at a time.
- A row with an empty `group_id`, `subject_id` or `task` is dropped.
- A subject can be in several groups, tasks and sessions; each `(group, task, session)` is processed on its own.
- **One row must name exactly one recording.** When a row still matches several files, the run stops and lists them; add `session` or `run` to that row.
- Without an `occasion` column, the members of one sitting must share its session label. With one, rows like `G01,sub-01,chat,3,1` and `G01,sub-02,chat,1,1` (columns `group_id,subject_id,task,session,occasion`) form one group, read from `sub-01/ses-3` and `sub-02/ses-1`.

## ROI mapping file

`--roi-mapping` takes a JSON object mapping each region to a list of source-detector pairs. `nirspipe` uses it for the region products of rest mode and to group the report's carpet; the dyad commands use it for region-level coherence, correlation and their nulls. Every command reads the file the same way.

```json
{
  "L-PFC": ["S1_D1", "S1_D2", "S2_D1"],
  "R-PFC": ["S3_D3", "S3_D4", "S4_D3"]
}
```

- **Entries are pairs, not channels.** A trailing ` hbo` or ` hbr` is dropped, and one map serves both chromophores. A wavelength suffix such as `"S1_D1 760"` matches nothing.
- **The file's stem names the outputs.** Reduced to letters and digits, it becomes the `seg-` entity of every region table and figure, so one output tree can hold results from two mappings.
- A pair no region lists takes no part in any region output.
- A pair may be listed in more than one region, for a border channel or nested regions. It then counts in each, the overlap is logged, and the reports name it. Regions that share pairs are not independent.
- A missing or unreadable file stops the run before the first subject.

## Table files

Wherever a command takes a table (events, segments, the pairs file, the span and bad-channel tables, `participants.tsv`), the extension decides the delimiter: `.tsv` is tab-separated, `.csv` comma-separated, and any other extension has its delimiter sniffed from the header. Option names such as `--pairs-csv` say nothing about the format you must supply. Files the package writes are always tab-separated, as BIDS requires.
