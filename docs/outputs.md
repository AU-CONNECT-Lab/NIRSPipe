---
title: Outputs
---

# Outputs

`nirspipe` writes a BIDS derivatives tree, and the dyad commands write a second tree of their own beside it. This page lists what each holds, how the files are named, what their sidecars record, and the logs and run records every run leaves.

## Naming

Every name is built from BIDS entities. `<run>` below stands for the input recording's own, `sub-<label>[_ses-<label>]_task-<label>[_run-<index>]`, and `<task>` for the same without the run.

| Entity | Means |
|--------|-------|
| `desc-` | The processing stage, or which record a QC file is |
| `stat-` | The measure in a table: `alff`, `pearson`, `fisherz`, `wtc`, `isc` |
| `chromo-` | The chromophore, where a file holds one |
| `seg-`, `agg-` | The ROI mapping a table was grouped by (the file's stem reduced to letters and digits) and how channels were aggregated: `roi`, `seed`, `homologous` |
| `cond-` | A condition window |
| `group-`, `pair-` | A dyad, and a member pairing inside a group of three or more |
| `null-` | Which null a table holds: `phase` or `pair` |
| `band-` | A re-averaged band from `nirspipe-hyper-band` |
| `chan-`, `label-` | A channel or a region, in figure names |

The suffix says what a file holds: `nirs` a signal, `relmat` a relation between channels or regions, `nirsmap` one value per channel or region, `qc` quality numbers, `report` a page. An entity appears only where it tells two otherwise identical files apart. `group-`, `pair-`, `chromo-`, `chan-`, `agg-`, `band-`, `cond-`, `null-` and `stat-`, and the `relmat` and `nirsmap` suffixes, are this package's extensions to BIDS, declared in its own naming configuration. Each tree's `.bidsignore` exempts the reports, `logs/`, `figures/`, the Plotly.js copy and the JSON-only records, so every table stays visible to a validator.

## The `nirspipe` tree

```
derivatives/nirspipe/
├── dataset_description.json             ← links the input as raw under DatasetLinks and SourceDatasets
├── .bidsignore
├── plotly-<version>.min.js             ← the reports' Plotly.js when the CDN does not load
├── desc-subjects_qc.tsv                 ← cohort table (nirspipe group / nirspipe-qc cohort)
├── desc-subjects_qc.json                ← what each of its columns holds
├── desc-subjects_report.html            ← cohort page
├── desc-ratedchannels_qc.tsv            ← nirspipe-rate export: the --bad-channels table
├── desc-ratedpages_qc.tsv               ← nirspipe-rate export: every FINAL verdict, for reading
├── figures/                             ← the cohort page's figures
├── logs/                                ← the rating viewers' saved verdicts
└── sub-<label>/
    ├── sub-<label>_desc-index_report.html          ← index over the subject's runs
    ├── <run>_report.html                           ← run report
    ├── <run>_cond-<condition>_report.html          ← + --by-condition
    ├── <run>_desc-raw_report.html                  ← nirspipe-qc prep-raw
    ├── <run>_desc-mne_report.html                  ← MNE's own report of the run
    ├── figures/                                    ← every figure of every run, one flat folder
    ├── logs/
    │   ├── sub-<label>.log                         ← text log
    │   └── sub-<label>.toml                        ← run record
    └── [ses-<label>/]nirs/                         ← every data file has a .json sidecar
        ├── <run>_desc-od_nirs.snirf                ← prep: optical density
        ├── <run>_desc-sci_nirs.snirf               ← prep: channel screening
        ├── <run>_desc-motcorrected_nirs.snirf      ← prep: motion correction
        ├── <run>_desc-preproc_nirs.snirf           ← prep: Beer-Lambert, end of preprocessing
        ├── <run>_desc-aux_timeseries.tsv.gz        ← aux channels, when the recording has any
        ├── <run>_space-fsaverage_channels.tsv      ← channel positions on fsaverage, in mm
        ├── <run>_desc-filtered_nirs.snirf          ← post: bandpass
        ├── <run>_desc-resampled_nirs.snirf         ← post: + --resample-sfreq
        ├── <run>_desc-errts_nirs.snirf             ← post: regression residual
        ├── <run>_desc-errtsbroad_nirs.snirf        ← rest: broadband residual, the ALFF input
        ├── <run>_desc-meanfilled_nirs.snirf        ← + --export-mean-filled
        ├── <run>_desc-sqm_qc.json                  ← quality record, its scalars
        ├── <run>_stat-{sci,psp,cv}_desc-sqm_timeseries.tsv   ← its channel × window matrices
        ├── <run>_stat-summary_desc-sqm_timeseries.tsv        ← its per-window series
        ├── <run>_desc-sqm_nirsmap.tsv              ← its per-channel values
        ├── <run>_desc-channel_qc.tsv               ← per-channel metrics and verdicts
        ├── <run>_desc-sqmraw_qc.json, <run>_desc-rawchannel_qc.tsv   ← the same from prep-raw
        ├── <run>_design.tsv                        ← design matrix, whenever a regression ran
        ├── <run>_desc-glm_nirsmap.tsv              ← fit results, one row per channel and regressor
        ├── <run>_desc-contrast_nirsmap.tsv         ← glm + --contrast-file
        ├── <run>_stat-alff_nirsmap.tsv             ← rest: alff, falff, malff, zalff per channel
        ├── <run>_seg-<map>_agg-roi_stat-alff_nirsmap.tsv                ← + --roi-mapping
        ├── <run>_chromo-{hbo,hbr}_stat-{pearson,fisherz}_relmat.tsv     ← rest or --fc
        ├── <run>_chromo-{hbo,hbr}_seg-<map>_agg-{roi,seed}_stat-{pearson,fisherz}_relmat.tsv  ← + --roi-mapping
        ├── <task>_desc-<stage>_nirs.snirf          ← --combine-runs: the task's runs joined, named without run-
        ├── <task>_chromo-{hbo,hbr}_..._relmat.tsv  ← --combine-runs: FC of the joined series, as above
        ├── <page>_desc-rating_qc.json, <page>_desc-rawrating_qc.json    ← section ratings
        └── sub-<label>[_ses-<label>]_task-<label>_desc-rawdecision_qc.json  ← channel decisions
```

A step that did not run writes nothing: `desc-motcorrected` is absent under `--motion-correction none`, `desc-filtered` when both cutoffs are off, and the quality record has no section for a stage that does not exist.

**Tables.** Every table is tab-separated and writes a missing or non-applicable value as `n/a`, as BIDS requires. pandas reads that as missing by default; R's `read.delim` and `readr::read_tsv` only when told (`na = c("", "NA", "n/a")`). pandas needs `float_precision="round_trip"` to read values back bit for bit. A connectivity matrix keeps every channel of the montage, a rejected channel as a blank row and column, so every subject's matrices have one shape. The quality record's layout is in [QC](qc.md#the-quality-record).

## The `nirspipe-hyper` tree

The dyad commands write a tree apart from the one `nirspipe` wrote. Its `dataset_description.json` links that tree as `preprocessed` (and, after `nirspipe-qc hyper-raw`, the raw dataset as `raw`), so the dyad tables name their member files as `bids:preprocessed:...`. A group the pairs file records per session puts `ses-<s>` after `group-<id>` in every name and its tables in `group-<id>/ses-<s>/nirs/`.

```
derivatives/nirspipe-hyper/
├── dataset_description.json
├── .bidsignore
├── plotly-<version>.min.js
├── desc-groups_qc.tsv, desc-groups_report.html    ← nirspipe-qc cohort-hyper
├── stat-wtc_relmat.tsv, stat-isc_relmat.tsv, ...  ← nirspipe-hyper-merge, one per long-table kind
├── chromo-<c>_task-<t>_cond-all_null-<n>_stat-<s>_desc-{bycell,byoccasion,cohort}_relmat.tsv  ← nirspipe-hyper-groupnull
├── task-<t>_desc-groupnull_report.html            ← nirspipe-hyper-groupnull
├── logs/                                          ← the cohort tests' Methods paragraphs
└── group-<id>/
    ├── group-<id>_desc-index_report.html          ← one row per analysed window
    ├── group-<id>_task-<t>_report.html            ← dyad analysis report
    ├── group-<id>_task-<t>_cond-<c>_report.html   ← one per condition window
    ├── group-<id>_task-<t>_desc-raw_report.html   ← nirspipe-qc hyper-raw
    ├── figures/
    ├── logs/group-<id>_task-<t>.{log,toml}        ← nirspipe-hyper's live log and run record
    ├── logs/group-<id>_task-<t>_desc-raw.log      ← nirspipe-qc hyper-raw's live log
    ├── logs/group-<id>_task-<t>_null-pair.log     ← nirspipe-hyper-pairnull's live log
    ├── logs/group-<id>_task-<t>_redraw.log        ← its redraw of the dyad's pages
    └── nirs/                                      ← every table has a .json sidecar
```

`nirspipe-qc hyper-raw` writes into the dyad tree or into a raw QC tree of its own, as `nirspipe-qc prep-raw` does for subjects. A tree it creates is stamped `GeneratedBy: nirspipe-qc hyper-raw`; a `nirspipe` tree is refused. `nirspipe-qc cohort-hyper` reads its quality records, so it runs on whichever tree `hyper-raw` wrote. Per group and task it writes:

| File | Holds |
|------|-------|
| `desc-sqm_qc.json` | The dyad's quality record, which `nirspipe-qc cohort-hyper` reads |
| `desc-subject_qc.tsv` | One row per member: every scalar of its raw quality view |
| `desc-channel_qc.tsv` | One row per member and channel: `sci_win`, `sci_whole`, `is_bad` |
| `desc-usable_qc.tsv` | One row per channel pair and condition: `usable_frac`, `one_member_frac`, `neither_frac`, `n_windows` |

`nirspipe-hyper` writes, per group and task (the names below drop the leading `group-<id>_task-<t>_`):

| File | Holds |
|------|-------|
| `stat-wtc_relmat.tsv` | Band-mean coherence, one row per channel pairing and chromophore, the whole recording's rows first and then each condition window's: `chromophore`, `condition` (`n/a` on the whole recording's rows; absent when no window was read), `sub1`, `sub2`, `label` (and `label2` when crossed), `coherence`, `coherence_z`, `n_valid_frac`, `phase_angle`, `phase_sd`, `phase_n`, `kept_s` |
| `seg-<map>_agg-roi_stat-wtc_relmat.tsv`, `seg-<map>_agg-homologous_stat-wtc_relmat.tsv` | The region tables, laid out the same way |
| `stat-wtcphase_relmat.tsv` | Relative phase per frequency, with `lag_s`, laid out the same way |
| `null-phase_stat-wtc_relmat.tsv` and its region twins | The phase-scrambled null, laid out the same way: the same columns plus `null_sd`, `null_p95`, `n_iter`, `percentile` (with `--wtc-phase-null`) |
| `cond-all_null-phase_stat-wtc_desc-draws_relmat.tsv` | Every iteration of that null |
| `chromo-<c>_null-phase_stat-wtc_desc-level_relmat.npz` | The null's level per frequency, which the phase arrows are drawn against, with a JSON sidecar |
| `chromo-<c>_stat-wtc_relmat.npz` | The full time-frequency maps (with `--wtc-save-maps`) |
| `stat-isc_relmat.tsv` | The correlations, one row per pairing and window: `chromophore`, `sub1`, `sub2`, `label`, `label2`, `condition`, `r`, `r_z`, and `lag_s` and the null columns where asked for |
| `seg-<map>_agg-roi_stat-isc_relmat.tsv` | The region correlations as a long table, laid out as `stat-isc_relmat.tsv`: one row per region pairing (the same region in both members unless crossed), chromophore and window, with `r` and `r_z` |
| `chromo-<c>[_cond-<c>]_stat-isc_relmat.tsv`, `..._seg-<map>_agg-roi_...` | The correlation matrices, per chromophore and window; they stay in the dyad's folder, `nirspipe-hyper-merge` merging the long tables only |
| `cond-all_null-phase_stat-isc_desc-draws_relmat.tsv` | Every surrogate of the correlation (with `--isc-phase-null`) |
| `desc-bad_qc.tsv` | Each member's excluded channels and the runs that rejected them |
| `space-fsaverage_channels.tsv` | Each member's channel positions on fsaverage, their own tables stacked with a `subject_id` column |

`nirspipe-hyper-pairnull` writes beside them, all per condition window: `cond-all_null-pair_stat-wtc_relmat.tsv` and its region twins, `cond-all_null-pair_stat-wtc_desc-draws_relmat.tsv`, the correlation's `cond-all_null-pair_stat-isc_relmat.tsv` and its draws, and the level `chromo-<c>_cond-all_null-pair_stat-wtc_desc-level_relmat.npz`. `nirspipe-hyper-band` writes `..._chromo-<c>_band-<band>_stat-wtc_relmat.tsv` beside each saved map.

Every band-mean table's sidecar records the settings a later command reads back rather than takes again: the band, `mask_coi`, `channel_cross`, `roi_min_channels`, `desc`, `bads_scope`, the whitening, the conditions and each window's span (`conditions`, `condition_windows_s`), and the level the phase was gated by (`phase_level_source` for the whole recording, `phase_level_source_by_condition` per window). The correlation's sidecars record `isc_band_hz`, its whitening and `isc_max_lag_s`, and the same `desc`, `bads_scope` and analysis window (`analysis_window_s`, from `--tstart`/`--tend`). Both record the separation bands the long channels were chosen by (`sep_short_max_mm`, `sep_long_min_mm`, `sep_long_max_mm`).

## Sidecars and provenance

Every file has a JSON sidecar naming the step that produced it and the files it was produced from, so the lineage of every derivative can be recovered from the tree alone.

| Field | Meaning |
|-------|---------|
| `TaskName`, `SamplingFrequency`, `NIRSChannelCount`, `NIRSSourceOptodeCount`, `NIRSDetectorOptodeCount` | On every derivative `_nirs.json`: the required fields of the raw sidecar. The rate and channel count describe the file itself |
| `pipeline_version` | The package version that wrote the file |
| `step` | The step that produced the file |
| `Sources` | The files it was derived from, as BIDS URIs. A GLM output or residual also lists the events table (`--events-path`) and the aux table it read |
| `external_sources` | An input table that sits in no linked dataset, by absolute path, since it has no BIDS URI |
| `parameters` | The settings that step used, as resolved; a regressor that was asked for but could not be built is absent |
| `data` | What the step left: `n_channels`, `n_bad`, `sfreq`, `duration_s` |
| `bad_channels` | The channels marked bad in this file; SNIRF has no field for them |
| `excluded_spans` | Every `BAD_` span the file carries, with its kind and source |

Each stage records its own step's settings. The postprocessed stages also carry the filter state of their file (`high_pass`, `low_pass`, `filter_method`, `filter_order`, `censor_fill`, `resample_sfreq`) and the drift settings, so a later analysis can tell what band a residual still holds. The contrast table's sidecar carries the weights it was computed from (`contrast_weights`). The whole configuration is in the run record.

| Stage (`desc-`) | Step |
|-----------------|------|
| `od` | `od_conversion`, or `od_passthrough` when the input is already optical density |
| `sci` | `sci_pruning` |
| `motcorrected` | `motion_correction` |
| `preproc` | `beer_lambert` |
| `filtered` | `bandpass` |
| `resampled` | `resample` |
| `errts` | `glm_residuals` |
| `errtsbroad` | `glm_residuals_broadband` |

The stage is carried by the file name, so renaming a `desc-` entity by hand breaks the lineage of everything downstream of that file.

**BIDS URIs.** Each `Sources` entry is a BIDS URI, never a local path, so a tree can be moved or shared without its lineage breaking: `bids::sub-01/nirs/...` names a file in the same tree, `bids:raw:...` one in the raw dataset, `bids:preprocessed:...` one in the subject-level tree a dyad tree was computed from. The names resolve through `DatasetLinks` in the tree's `dataset_description.json`, written with a relative path where both trees sit on one drive. A tree whose `Sources` are not BIDS URIs is refused where they are read.

**The provenance graph.** At the end of each subject, `nirspipe` draws one graph per run from the sidecars, `figures/<run>_desc-provenance_nirs.png` with its Mermaid source `.mmd`, and the run report embeds it. Nodes are files, labelled by their entities; edges are labelled with the step; three or more outputs one step made from the same inputs are drawn as one box. A node whose file is missing keeps its place with a dashed red outline. `nirspipe-qc provenance OUTPUT_DIR` redraws the graphs from the sidecars on disk, for subjects, dyads and the tree root alike, so it works on any past run and refreshes the image a written report points at.

```{figure} figures/provenance.png
:alt: A provenance graph of one run, one box per file
:width: 100%

The provenance graph of a GLM run: one box per file, labelled with the step that made it and, for the signal files, the channels kept, the sampling rate and the duration.
```

## Logs and run records

| Layer | Format | Path | For |
|-------|--------|------|-----|
| Live log | Text | `sub-<label>/logs/sub-<label>.log` | Following a run; `--verbose` adds debug output |
| Dyad live log | Text | `group-<id>/logs/group-<id>_task-<t>.log`, `_desc-raw.log` for `hyper-raw`, `_null-pair.log` for the re-paired null, `_redraw.log` for the pages it redraws | Following one group of a dyad command |
| Run record | TOML | `sub-<label>/logs/sub-<label>.toml` | What a subject was run with, and rerunning it |
| Provenance | JSON sidecars and a graph | `nirs/*.json`, `figures/*_desc-provenance_nirs.png` | How one file was made |

### Run record

Each subject's run record is written at the start of its run, with four sections: `[environment]` (package, Python and dependency versions, platform), `[execution]` (the verbatim command, paths, labels and switches), `[prep]` and `[post]` (every setting as resolved, keyed by flag name in the flag's units), and `[post_sources]`, naming for every post setting the layer it came from: `cli`, `config`, `mode` or `default`. Tables are written as typed, with absolute paths, so a record reads from anywhere. A post setting the run switched off is written `"none"`.

When the subject ends, an `[outcome]` section is appended: `status` (`success` or `failed`), `finished`, `duration_s` and, for a failure, the first line of the `error`; the full traceback is in the live log. A record without `[outcome]` belongs to a run that was stopped before it ended. The cohort page lists every subject whose last run did not succeed.

```toml
[execution]
run_command = "nirspipe /data/bids /data/out participant --dpf 6 --sci-threshold 0.8 --mode glm ..."
participant_label = [ "01" ]
task_label = [ "tapping" ]

[prep]
dpf = [ 6.0, 6.0 ]
sci_threshold = 0.8
motion_correction = "tddr"
window_length = 10.0

[post]
mode = "glm"
hrf_model = "spm"
noise_model = "auto"
drift_model = "cosine"
drift_high_pass = 0.01

[post_sources]
mode = "cli"
hrf_model = "mode"
noise_model = "mode"
drift_model = "mode"
drift_high_pass = "cli"
```

A run record is a config file: `--config` on it reruns the run it records ([Configuration file](usage.md#configuration-file)).
