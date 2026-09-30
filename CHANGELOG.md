# Changelog

All notable changes to this project will be documented in this file.

<!-- Format: Keep a Changelog (https://keepachangelog.com/en/1.0.0/) -->

## [Unreleased]

### Added
- The carpet and per-channel motion figures draw the gyroscope's angular speed above the GVTD rows when the recording carries one
- The same figures draw the accelerometer's jerk as a second IMU row when the recording carries one
- The dyad motion panel draws each member's gyroscope and accelerometer on the shared clock, each divided by its own median
- IMU rows print the unit the recording stores, on the axis or beside the dyad medians
- `fnirs-qc hyper-raw --derivatives-dir` names the fnirs-pipe tree its motion panel reads corrected recordings from

### Changed
- Carpets detrend each row linearly before z-scoring, so the grey shows fluctuation rather than drift
- `fnirs-qc hyper-raw` refuses an output directory that is its `--derivatives-dir`

### Fixed
- `fnirs-qc hyper-raw` looked for corrected recordings in the fnirs-hyper tree, so its motion panel never had a corrected side
- The Hyper Preparation page wrote its dyad report into the derivatives tree and restamped it; it now takes a fnirs-hyper directory

## [0.53.0] - 2026-09-26

### Added
- `fnirs-pipe` output names its input dataset under `SourceDatasets` in `dataset_description.json`
- `fnirs-hyper-groupnull` tests every crossed channel pairing across occasions, as its own family, when the null was drawn crossed

### Changed
- **Breaking**: `fnirs-qc hyper-raw` takes the analysis level `group` after its two directories, as `fnirs-hyper` does
- The brain figures stop with a message when fsaverage cannot be loaded, instead of drawing optodes off the brain
- **Breaking**: commands reading BIDS check their input with bids-validator when installed and stop on errors; `--skip-bids-validation` skips that check
- Files BIDS does not recognise are left out of the input even with `--skip-bids-validation`
- **Breaking**: `fnirs-hyper` crosses channels by default (`--no-wtc-channel-cross`); the phase-scrambled null follows that unless set
- **Breaking**: `fnirs-hyper-pairnull` draws its null over every channel pairing by default (`--no-wtc-pair-cross`)
- The Hyper Analysis page's crossing switches start on and pass their off state explicitly
- The 3D brain views render off-screen with pyvista instead of plotly and kaleido, so no browser is needed; a failed render is listed on the run page
- The GLM activation figures render off-screen with pyvista instead of a Qt window, matching the previous images
- `nibabel` is a declared dependency; the activation renders read FreeSurfer curvature with it
- The superior brain view puts the nose up and the subject's left on the left, as the optode flat map does

### Fixed
- `fnirs-prep` and `fnirs-qc` commands reading BIDS stop when the output directory is the input, except `crop --input-desc`
- A GLM activation render that fails is listed on the run page instead of being dropped silently
- `fnirs-pipe` stops when the output directory is the input directory or the work directory lies inside it
- `fnirs-pipe --participant-label` naming no participant in the dataset stops the run instead of exiting 0
- The brain view labelled Frontal showed the back of the brain
- The 3D brain views drew rejected channels in their SCI colour rather than as rejected
- `--mode glm` with `--n-jobs` above 1 hung or crashed while drawing the activation figures

## [0.52.0] - 2026-09-25

### Fixed
- Non-finite samples are zeroed at the OD step and their channels marked bad; they used to crash post inside the GLM
- A cardiac band at or above the recording's Nyquist frequency is refused, naming `--cardiac-h-freq`; QC used to score every channel's SCI as 1.0
- **Breaking**: `--mode glm` refuses a run with no events and points to `--mode denoise`; it used to fit drift only
- Duplicated events (same trial_type, onset and duration) are dropped with a warning instead of summed into one doubled regressor
- `intensity_to_od` refuses input that is not raw intensity with a `StageError` naming the channel types it got
- `run_glm_pipeline` asked to write outputs without `source_path` refuses before fitting; it used to fail at output naming
- `fnirs-hyper-pairnull` merged one person's stand-ins from different sessions or runs into one draw; a draw is now named by subject, session and run
- `fnirs-hyper-groupnull` failed on a channel with no occasion holding both a real value and draws; that channel is now left out

## [0.51.0] - 2026-09-25

### Changed
- **Breaking**: `fnirs-hyper-groupnull --task` is required; it had a study-specific default
- Comments, docstrings, help and report text no longer carry study-specific numbers, change history or documentation links
- Code comments and docstrings say what the code does, without design arguments, measured values or rejected alternatives
- The Hyper Analysis page names each command by its console script, not the retired subcommand names
- Help, GUI and report text say what each option or panel does, without design rationale or references to other tools

### Fixed
- The raw viewer never showed its before/after motion note or its read-only channel-decisions hint; both had lost their element

## [0.50.0] - 2026-09-23

### Added
- `fnirs-hyper --wtc-whiten SECONDS` prewhitens long channels before the coherence, one fixed AR order for both members; off by default
- `fnirs-hyper-pairnull` writes per-condition arrow levels and an ISC `null_abs_p95` column, which the next `fnirs-hyper` run uses on condition pages
- `fnirs-log rebuild` builds a new database from every JSONL log, archived ones included, leaving the old one untouched

### Changed
- **Breaking**: `fnirs-log merge` merges only finished executions, moves their logs to `archived/`, and backs up the database first
- A null level on disk is used only when its sidecar matches the run; each page names the null its arrows and chords used
- WTC table sidecars record `phase_level_source`, and saved maps get a sidecar that `fnirs-hyper band` carries onto its tables
- Conditions whose file names would collide get no subject report pages, with an error naming them; a dyad run refuses them, and a condition named `all`

### Fixed
- `fnirs-qc hyper-raw` failed on recordings whose aligned window ended a float round-off past the last sample
- `fnirs-hyper` and its companions' help and merge hint named the old `run` / `band` / `merge` / `pair-null` subcommands
- Condition pages captioned arrows drawn against the phase-scrambled null as the Monte Carlo level
- `fnirs-log merge` inserted every row again each time it ran
- `fnirs-hyper band` dropped every WTC parameter except the band and mask from its sidecars
- The dyad index warned that each optional table the run did not write was unreadable
- A dyad report's provenance diagram left out the phase-scrambled null table until the next run

## [0.49.0] - 2026-09-23

### Changed
- **Breaking**: the append-only rating logs are `logs/group_ratings.jsonl` and `logs/group_raw_ratings.jsonl`, not at the tree root
- **Breaking**: `fnirs-prep crop` refuses several segments without a `task` column instead of naming them `_seg-<NN>`; the GUI segment tables gain a Task column
- Removed the unused whole-run re-pairing helpers `_draw_pairs` and `condition_coverage`
- `fnirs-prep crop` checks the segments table once before any subject, so a bad table is one error and writes nothing

### Fixed
- Merging the per-pairing tables of groups of three or more wrote every pairing to one file; the merged name keeps `pair-`
- `fnirs-prep crop` wrote two segments sharing a `task` label to one file; it now refuses them
- The re-paired null cut every repeat of a condition from the stand-in's first block of it, not the matching one
- A segment cropped from a recording kept the source's `TaskName` and `RecordingDuration` and recorded no source or window
- `fnirs-hyper` reported a `StageError`, such as an optical-density `--desc`, as an unexpected error with a traceback
- `fnirs-hyper-band --wtc-suffix` is reduced to a valid `band-` value, and its help shows the real default
- The `fnirs-hyper-merge` help lists what refuses a merge and what only warns
- The raw viewer's condition pages name their channel-summary and trial tables `_qc`, as its run page does
- The `--epoch-tmin` help and the report docs say per-trial scoring uses each event's duration when no window is set
- Without `--session-label`, `fnirs-pipe participant` wrote a session tree's outputs to `sub-<id>/nirs` with no `ses-`, one session overwriting another
- The run report, the subject index and the rating server read and link each session's own `nirs/`
- The generated reproduction script keeps each run's task, run and session in its file names, and its GLM step runs again

## [0.48.0] - 2026-09-23

### Changed
- **Breaking**: the raw viewer's condition pages are `cond-<label>_desc-raw_report.html` under the run's own label; they swapped the condition into `task-`
- **Breaking**: the dyad cohort page's figures are `desc-groups<panel>_nirs.html`, not `cohort_hyper_<panel>.html`
- `.bidsignore` names the five JSON-only records (quality, ratings, channel decisions); every table stays checked
- **Breaking**: `fnirs-prep align` records each member's offset in its own `_nirs.json` and no longer writes `align-offsets.tsv`
- 128 function-level imports of modules their file already loads move to module level; startup is unchanged
- `mne_nirs` is imported only when a SNIRF is written; `fnirs-hyper --help` takes 0.9 s instead of 2.3 s
- A further 41 cheap package imports move to module level; no command's startup loads anything more
- Provenance node labels read every entity, so a table's label names its measure and slices instead of repeating the filename

### Fixed
- The subject index linked no raw condition page, looking for the pre-rename `desc-raw_nirs` spelling
- Two runs in one raw report wrote their condition pages to one file
- Raw condition pages dropped every panel redrawn for the condition as a leak
- The dyad rating server read its task as `unknown`, so channel decisions went to a file the raw page never reads
- A collapsed provenance box counted no conditions, the `cond-` entity being unknown to its parser
- The GLM tables go to the session folder their run's snirfs use, not `sub-<id>/nirs`
- `fnirs-qc provenance` refreshes the graph a dyad report links, and finds subjects with sessions
- The dyad quality loader read SCI from the per-channel table as comma separated and silently got none
- The `--by-condition` help of `fnirs-pipe` and `fnirs-qc prep-raw`, and messages naming `fnirs-hyper run`, describe the current commands and names
- On a session tree the Hyper Preparation page read and wrote channel decisions under a path the raw QC page never uses
- `fnirs-prep align` wrote a member of a two-session tree under the wrong session's name and folder
- A member two groups share is refused by `fnirs-prep align` instead of being silently re-cut by the second group
- On a tree both commands wrote, no page linked the raw condition pages; the index's Conditions rows now list them beside the pipeline's
- `fnirs-pipe group` warned of a split tree whenever a `qc/` or `derivatives/` folder existed, and looked for pre-rename record names

## [0.47.0] - 2026-09-23

BIDS App alignment and the output naming rework: every derivative name now comes from one config.

### Added
- Every long CLI flag also answers to its underscore spelling; `--n-jobs` also to `--nprocs` and `--n_cpus`, `--skip-bids-validation` to `--skip_bids_validator`
- `dataset_description.json` records the package version under `GeneratedBy`
- `fnirs_pipe/io/naming.py` builds and parses every derivative name from one pybids config, `fnirs_pipe/data/fnirs_pipe_bids_config.json`
- Each output tree carries a `.bidsignore` waving through reports, logs and figures, and nothing else
- A dyad tree's `dataset_description.json` names the tree it was computed from under `SourceDatasets`

### Changed
- **Breaking**: every report page is `_report.html` with entities: the four spellings (`_qc.html`, `_desc-raw_nirs.html`, `_qc_mne.html`, a subject index shaped like a run report) are one
- **Breaking**: the cohort pages are `desc-subjects_report.html` and `desc-groups_report.html`, their tables `desc-subjects_qc.tsv` and `desc-groups_qc.tsv`, their figures under `figures/`
- **Breaking**: the quality record is `desc-sqm_qc.json`, not `desc-sqm_nirs.json`; a .json whose suffix is `nirs` is by BIDS definition a snirf's sidecar
- **Breaking**: the GLM writes `design.tsv`, `desc-glm_nirsmap.tsv` and `desc-contrast_nirsmap.tsv`, tab separated; they were comma-separated `.csv`
- **Breaking**: per-channel quality is `desc-channel_qc.tsv`, not `channel_metrics.csv`
- **Breaking**: connectivity and amplitude outputs are named by entities rather than invented suffixes: `desc-hbo_fc.tsv` is `chromo-hbo_stat-pearson_relmat.tsv`, `alffroi.tsv` is `seg-<map>_agg-roi_stat-alff_nirsmap.tsv`, and the ROI map's filename names the `seg-` entity
- **Breaking**: each `fnirs-hyper` subcommand is its own console script now, taking `[<source tree>] <output tree> group`; `run` and `pair-null` read the source tree, the other four only re-read what this package wrote
- `fnirs-prep`'s output positional is named `output_dir`, not `derivatives_dir`; that name now means the tree `fnirs-hyper` reads
- A GUI page whose command should have written a report and did not now says so, instead of the line it shares with commands that write only tables
- **Breaking**: every figure is named by entities and every subject's runs share one `figures/`; the per-run and per-scope subdirectories are gone
- **Breaking**: the raw viewer's figures take a `raw` prefix on their `desc-`, sharing that folder with the report's own
- A section builder is handed a namer instead of a filename suffix, and the per-condition leak check reads the `cond-` entity instead of a name suffix and a prefix whitelist
- **Breaking**: every dyad table is named by entities: `hyper-wtcbycond-roihom-pairnull` is `seg-<map>_agg-homologous_cond-all_null-pair_stat-wtc_relmat`, and the dyad's cross-subject quality tables give `_channels.tsv` back to BIDS
- **Breaking**: the merged cross-dyad tables are the inputs' own name minus `group-` and `task-`; `fnirs-hyper merge` discovers kinds by entity instead of a 14-row table
- **Breaking**: human ratings and channel decisions are `desc-rawrating_qc.json` and `desc-rawdecision_qc.json` under the analysis unit; they sat loose in the derivatives root
- The band stays in the sidecar rather than the filename, as the reference BIDS Apps keep theirs; only `fnirs-hyper band`'s re-averaged tables carry `band-`, being the one output that exists to sit beside another
- `hyper_stem` is `group_output_path`, the dyad mirror of `build_output_path`, so a caller asks for a path instead of appending to a prefix
- Entity parsing, the channel-decisions path, the quality-record name, the dyad table name and a condition page's name each have one definition; they had three, four, five, nine and three

### Fixed
- Per-condition report pages were written under a name neither the nav strip inside them nor the subject index looked for, so every link to one was dead
- `fnirs-hyper group-null` wrote every task and chromophore to one filename, so a second run overwrote the first
- The `.bidsignore` waved nothing through: bids-validator matches nothing against `figures/` or `logs/`, so every figure stayed on its books
- The merge refuses two wide matrices that do not cover the same channels; discovering kinds by entity had started concatenating them into a grid whose blanks read as measured zeros

## [0.46.0] - 2026-09-21

### Added
- The Hyper Analysis page sends `--wtc-window-s`; the flag shipped with no control
- The Hyper Analysis page reaches `pair-null` and `group-null`; the command dropdown offered neither
- `fnirs-hyper group-null`'s cohort tables carry four corrections, `q` (BH), `q_by`, `q_holm`, `q_bonferroni`, and a `family` column; only the per-cell tables had one
- `fnirs-hyper group-null` writes both reads of the cohort test, `test` column `resample` and `paired`; the phase null is only valid read paired above the cell

### Fixed
- The Hyper Analysis page would not load: its ISC band field was declared twice, and Dash refuses a duplicate id
- `fnirs-hyper run` and `pair-null` no longer fail on the closing merge hint

### Changed
- The hyperscanning modules moved into `fnirs_pipe.pipeline.hyper`; `pipeline.hyperscanning` is now `pipeline.hyper`
- `pipeline.hyper.synchrony` split into `wtc`, `surrogate`, `coherence`, `roi`, `isc` and `_helpers`
- The logger `pipeline.hyperscanning` is gone; those modules log under their own names, `pipeline.wtc`, `pipeline.surrogate`, `pipeline.coherence`, `pipeline.roi`, `pipeline.isc`
- The ROI correlation tables record which channels each region was averaged from, as the seed map and the ROI amplitude table already did

## [0.45.0] - 2026-09-19

### Added
- `fnirs-hyper group-null` tests a null above the cell, per occasion and over the cohort, at every granularity the draws support: channel, region, whole brain, and whole brain over all pairings where the null was drawn crossed
- The per-cell tests carry a Benjamini-Hochberg `q`, one family per condition and granularity. Nothing corrected for multiple comparisons before
- `fnirs-hyper pair-null` writes the individual draws as `hyper-wtcbycond-pairnull-draws.tsv`, beside the summary
- `fnirs-hyper pair-null` also writes a re-paired null for the correlation, `hyper-isc-pairnull.tsv` and `hyper-iscbycond-pairnull.tsv`
- `--isc-fmin` / `--isc-fmax` set the band the inter-subject correlation reads; a run says so when the coherence is on another band
- `--n-jobs` runs subjects in parallel in `fnirs-pipe participant`, where it was accepted and ignored

### Changed
- The HbO-HbR panel no longer draws a pass line at r = -0.3, and the metric table no longer grades that number
- Every stage's sidecar records the separation bands the run used

### Fixed
- `fnirs-hyper group-null` reported an all-pairings level when only the draws were crossed, ranking a mean over the diagonal inside a null built from every pairing. It also never checked that the draws and the real tables were on one band
- `--wtc-cond-transform` put the real per-condition tables and their phase-scrambled null on different routes, silently. The combination is now refused
- mALFF and zALFF were biased on any run where one short channel was the whole short-channel regressor. The ROI amplitude table inherited it
- The re-paired null was read off a shorter stretch of its draws than the real table is read off its own. Both now measure the same window
- `fnirs-hyper pair-null` died on a dyad rather than returning a null

## [0.44.0] - 2026-09-18

### Added
- `--noise-model ar_irls` alternates autoregressive whitening with a robust refit, so residual motion carries less weight. Slower; `auto` stays the default
- The dyad coherence tables carry the relative phase: `phase_angle`, `phase_sd` and `phase_n`, positive meaning the first member leads
- The per-frequency phase is written to `hyper-wtc-phasescale.tsv` and `hyper-wtcbycond-phasescale.tsv`, each with a `lag_s` column
- A rest run given `--roi-mapping` writes ROI-level amplitude too, `alffroi.tsv`
- A run whose optodes were never registered to the head says so, and names the figures it invalidates
- `--short-channel pca` puts every short channel in the design matrix instead of their mean
- `fnirs-hyper pair-null`, a second null for the dyad coherence: each member is paired with people from other groups who did the same task
- A GLM run that low-passes its data is warned that prewhitening has little left to do. `--resample-sfreq` raises the same warning
- The subject report says how much louder the two ends of a filtered recording are than its middle
- The separation bands can be set in a `--config` TOML instead of on every command line

### Changed
- The rest report's channel matrix and its ALFF bar chart are one panel, on one channel order
- A correlation matrix is drawn as a triangle, with its tick labels against the matrix
- One correlation colour scale serves the whole report, and the ROI matrix is interactive
- The two resting-state flat maps are interactive and drawn on the report's own head
- The ALFF layout draws one disc per channel rather than a source-to-detector bar, and reports mALFF rather than molar amplitude
- The surrogate null is `--wtc-phase-null`, not `--wtc-pseudo`, and its tables are `...-wtc-phasenull.tsv`. `--wtc-pseudo-cross` and `--isc-pseudo` move the same way, with no aliases kept
- A channel falling between the two separation bands is reported with its own separation and the bound that would take it in
- A condition page's denoising carpet no longer quotes an SD ratio measured over the whole recording

### Fixed
- One bad recording no longer strands every subject queued behind it; the batch finishes and exits non-zero naming what failed
- `--roi-mapping` help named the denoising carpet only, not the ROI correlation matrix or the seed topographies
- A short channel could be regressed out of itself, and its ALFF and correlations reached the tables. They are blank now, and the run says which channel
- The activation figures were blank brains: the colour scale could not reach the range haemoglobin betas sit in
- The activation figures carried three overlapping copies of their colour scale. There is now one, in micromolar
- The channel maps' scalp-share note sat over the wrong condition
- A rejected channel that fell between the two separation bands gave no reason and showed no scores
- The generated processing script named the wrong noise model in `denoise` and `rest`
- The quality record was split on the default separation bands whatever the run was told
- `--contrast-file` could not run at all; every GLM given one died after the fit
- `--dry-run` ran the pipeline
- The warning about a drift basis absorbing the task read the recording's own annotations rather than `--events-path`
- The Methods paragraph named a filter, a regression and a drift term the run had not used
- `generate_methods_text` raised `KeyError` when given a config rather than a tree
- `denoise` and `rest` fit an autoregressive model on low-passed data without saying so
- `--noise-model` takes any autoregressive order again, defaults to `auto`, and every mode honours it
- A run with non-default separation bands was described with the default ones in the preprocessing report
- The confound-regression residual was not a residual when `--mode glm` was given an AR noise model. Betas, t values and contrasts were never affected, nor were `denoise` and `rest`
- A prep-only run's spectrum panel no longer presents its simulated bandpass as a stage the run wrote

### Removed
- The resting-state connectogram. The inter-brain one stays
- The `tables` (PyTables) dependency. Calling `save()` on a returned GLM object yourself now needs it installed

## [0.43.0] - 2026-09-17

### Changed
- The QC report's denoising carpet shows the denoised stage alone, HbO above HbR, under one GVTD row per channel set on a shared time axis
- The spectrum panel carries the confound regression's residual
- On a GLM run the HbO-HbR panel tests the before stage against the -0.3 rule, not the residual

### Fixed
- The confound-regression residual now records the short-channel and aux regressors it was built with

## [0.42.0] - 2026-09-16

### Added
- Data Preparation reads the drift cutoff a GLM on the loaded run could use, with the slowest repeat of every condition beside it
- The GLM warns when the drift basis reaches the frequency at which a condition repeats
- The two QC pages can write the static report of what they are showing, by running `fnirs-qc prep-raw` or `hyper-raw`
- Batch Prep writes out the `fnirs-prep` command it would run, for copying to a shell or a job script

### Changed
- A run records the drift cutoff it used, so a hyperscanning analysis can tell one residual's detrending from another's
- A run records the confound regressors it actually built, not the ones it was asked for
- Asking for short-channel regression on a montage that has no short channel now stops the run
- The Analysis page no longer pre-fills a drift cutoff, and says where the number should come from
- The Analysis page says on screen when the high-pass and the drift model disagree, and no longer suggests a high-pass
- The grid tables draw a cross again in the column that deletes a row
- The sidebar is two families: Quality control holds the pages that load one recording, Batch the pages that assemble a command. `Hyper Align` is now `Hyper Preparation` and the cohort page is `Cohort Reports`
- Batch Prep runs the `fnirs-prep` command it shows you, streaming its output with a Stop button
- The hyperscanning analysis has its own page, separate from the cohort aggregates
- Every GUI page keeps Run and the generated command in view while the form scrolls
- Parameters are laid out the same way on every GUI page, grouped under headings
- The GUI's tables are drawn by a grid widget Dash still supports

## [0.41.0] - 2026-09-14

### Added
- The dyad index says how each window stood against its own null, as the count of channel pairs that beat their surrogate
- The GUI's Analysis page can set what rejects a channel, plus trial chunking, GVTD censoring and the per-condition QC pages
- The dyad report draws the ROI by ROI correlations, HbO beside HbR on one colour scale
- The dyad report's numbers table gains a homologous ROI block

### Changed
- Spline motion correction is withheld while it is decided whether to build it. The two methods that work are unchanged
- The GUI shows a run's output while it runs, with a Stop button. One run at a time
- Aux channel regression is withheld pending evaluation. Preprocessing still writes the aux channels
- Every QC page carries the same top bar
- The dyad report's numbers are one table per kind of pairing, each condition beside the whole run
- The cone-of-influence share is stated once per condition rather than repeated down every row
- Every transform is about a fifth faster. Band-mean coherences do not move

### Fixed
- `fnirs-gui` and `fnirs-rate` refused to start when their default port was taken. A default port now moves to the first free one
- The `fnirs-rate` viewers reported they were serving when their server had failed to start
- Several `fnirs-pipe` runs against one output directory could corrupt each other's bookkeeping. Dyad runs were never affected

## [0.40.0] - 2026-09-13

### Added
- The ROI coherence is written over an ROI's homologous channel pairs too, as `hyper-wtc-roihom.tsv`. This is the number to report
- That ROI mean gets a null of its own, `hyper-wtc-roihom-pseudo.tsv`
- `--isc-whiten` removes each channel's autocorrelation before the correlation. Off by default, and whitened values do not compare with unwhitened ones
- `--isc-max-lag` searches a few seconds either way and keeps the strongest correlation, reporting the winning shift. Off by default
- `--isc-pseudo N` ranks each correlation against phase-scrambled surrogates. Off by default
- Correlations are also written one row per channel pair, in `hyper-iscpairs.tsv`
- The pseudo-dyad null reports its spread, and where the real value sits inside it

### Changed
- The coherence's scale smoothing is the width its definition fixes, where the backend used twice that. Coherences and significance from earlier runs do not carry over
- The connectogram draws a chord where a pairing beats its own surrogate null, or the strongest tenth when no null was drawn. `--isc-threshold` still forces an absolute cut
- Phase arrows are drawn against the pseudo-dyad null when one was computed, at a level per frequency
- Crossed wavelet coherence is about three times faster, every value identical
- The pseudo-dyad null is about a third faster, every value identical, at the cost of about a gigabyte of memory
- Entity flags are spelled the same on every command: `--participant-label`, `--session-label`, `--task-label`, `--run-label`, `--group-id`. Existing command lines have to be updated
- `fnirs-qc prep-raw` takes more than one subject, and one subject's failure no longer stops the rest
- `--version` works on every command
- The null tables call their centre `null_mean`, and `null_abs_*` for the correlation, rather than `coherence`

### Fixed
- The homologous ROI table listed crossed regions too
- The ROI coherence on the diagonal meant two different things depending on `--wtc-channel-cross`
- On a crossed run the pseudo-dyad null's `percentile` ranked the wrong cell
- The dyad report claimed the correlations were run on an unfiltered stage even when they were not
- Arrows and other non-ASCII characters in the log came out as mojibake on Windows

## [0.39.0] - 2026-09-13

### Added
- The dyad numbers table reports every condition, not just the page it is on
- ROI pairs carry an inter-subject correlation, written to `hyper-isc-roichan-*.tsv`, one file per condition
- A group of more than two members gets one report per pairing, under a `_<sub1>x<sub2>` suffix. A dyad is spelled exactly as before
- `fnirs-qc prep-raw` writes the subject index too, so its per-condition pages can be found
- `fnirs-qc cohort-hyper` reports every dyad in a tree on one page: usable time, where it went, and each window's rank inside its own null
- Each dyad writes a usable-time table, one row per channel pair and condition, beside its quality record
- The coherence matrices put HbO and HbR side by side on one colour scale, per channel pair and per ROI pair
- The ISC panel carries a connectogram beside its matrix, both live
- Every number behind a dyad page's panels is on the page, one row per pairing
- The ROI coherence maps are live figures; the per-channel maps stay stills

### Changed
- The dyad numbers table calls its coherence column WTC
- The dyad's motion section is two figures, before and after motion correction, each with a GVTD row per channel set and each member's carpet
- Simultaneous movement is the pointwise minimum of the two members' traces, reported as its mean, rather than a count of windows past a cutoff
- Spikes are marked only where every member was spiking at once
- The cohort reports are called `cohort`, not `group`: `fnirs-qc cohort` and `cohort-hyper`, writing `cohort_nirs.*` and `cohort_hyper_nirs.*`. The old names are gone rather than deprecated
- The cohort hyper page no longer repeats the per-subject cohort report
- A condition's ROI coherence map is a view of the run's, in the same file. `--wtc-cond-transform` still writes its own
- The live coherence map is drawn at the still's own line weights
- No figure in the dyad report is boxed in black
- There is one report look, and it is the default
- The two dyad reports read as documents, each opening with a summary and folding by section
- A dyad quality table is the subject report's table, with members where it has channel sets
- The page is named after the run, as a subject page is
- The orange caveat panels are gone from the dyad pages; the one coloured warning left is a window too short for the band it averages
- A coherence map names its conditions in a legend rather than above every block
- A dyad page reads what the run was, then the coherence, then the synchrony, then each member's own quality
- The panels of a dyad page are sized to each other
- The provenance diagram collapses repeats, so a step run per chromophore and condition is one box

### Fixed
- A rejected channel pair printed `nan` in the dyad numbers table
- A group of three drew one member pair's coherence under another pair's name. A dyad was never affected
- The subject index lists runs measured by `prep-raw` alone
- A cohort page no longer stops at a `qc/` subdirectory, which dropped every stage after Beer-Lambert
- The inter-brain connectogram drew connections that do not exist, including within-brain pairs and sub-threshold ones
- The dyad's optode maps and channel grid colour by the screening's verdict rather than by SCI alone
- Every per-channel SCI on the raw dyad page is the windowed estimate
- A subject's per-condition pages had quietly stopped being written
- The coherence maps were written at 200 dpi where every other figure is 300

## [0.38.0] - 2026-09-12

### Added
- The dyad post report can be rated. The run and each condition are filed separately
- Every page in a set links to the others from its bar

### Changed
- The raw dyad report's bar is one cell per section with anchors. Verdicts already filed are kept

### Fixed
- The dyad quality table was missing the coupled-window share, the one metric that screens
- The dyad table coloured the whole-run SCI and left the windowed one plain. Both are now judged against the run's own threshold
- A condition's GCOR before/after pair spanned the bandpass as well as the regression
- The channel-quality grid's SCI row is the windowed estimate on every page
- A condition page inherited three per-channel numbers from the whole run: mean amplitude, the motion-correction footprint and the windowed SCI
- A dyad page dropped a member's column when the two members' copies of one trigger sat a sample apart

## [0.37.0] - 2026-09-12

### Added
- Every hyperscanning table records what put the members on one clock: a shared trigger or a trim, which trigger, and where each was cut
- The subject index carries the run's conditions, with three figures comparing them against the run they were cut from
- `fnirs-qc prep-raw --motion-correction` runs that one step on a copy of the optical density and reports the recording either side of it. Nothing is written back; default `none`
- The raw report carries the grand mean and the single-channel trial images
- `fnirs-qc prep-raw` writes a `by_condition` section too, whenever the recording has conditions
- Its quality record carries the windowed matrices and the flagged spans
- The raw viewer's metrics panel has a motion table, beside the optical-density one
- The raw viewer closes with Provenance, Methods and Software Versions
- The motion table carries every motion and spike number over all three channel sets, each against its own cutoff
- Short channels carry their own spike numbers and their own correction footprint
- The spike count is printed per channel set, beside the channel count it is a sum over
- A channel-set column no set measured is dropped rather than printing dashes
- Every span the record stores is written per channel set, so a condition's rows are the measurements the run's rows are
- Raw ratings are saved per page and appended to `group_raw_ratings.jsonl`
- The raw viewer carries the per-channel motion figure, one file per channel carrying every condition's window

### Changed
- The event timeline draws a block for its length, not a tick at its onset, with the length printed on it
- A condition page shows the run's whole event timeline
- A dyad page's quality table prints all three channel sets, one table each
- A dyad page's per-condition quality table is read out of each member's quality record instead of being measured again
- The HbO-HbR correlation panel is interactive, and groups long channels before short ones inside each chromophore block
- The correlation matrix fills the width of the page it is opened on
- A condition page's carpet scales its GVTD rows to that condition
- A report page is titled by the run rather than by the words "QC Report"
- The raw viewer's per-condition pages read their numbers out of the record
- The optode layout opens the raw report
- The per-channel HbO/HbR and spectrum panel sits directly under the raw trace, with a channel picker
- The SCI/PSP panel carries its CV row
- A condition page's spectrum is measured on that condition
- A condition page's figures stay on their condition when reset
- A condition's raw-signal panel holds only that condition's samples. Pages are a third of their old size
- A per-condition page's event timeline shows that condition's row over the run's summary row
- A per-condition page prints each metric once
- A panel a page has nothing for is hidden, heading and all
- The rating bar lists every panel on the page and links to it
- The group report reads as a document, opening with a Summary
- Its distributions put a metric's channel sets side by side, and every metric lands in a named same-scale chart
- The group overview is a deviation strip, one row per metric and one dot per run at its robust z
- It reports per condition, one panel per metric plus a run by condition matrix sorted worst first
- Its time panel is a metric by channel-set grid, with the cohort's band and median over every run
- `sci_win_mean`, coupling measured inside 10 s windows, sits beside the whole-run `sci_mean`. Reports read the windowed one; `sci_mean` keeps the published cutoffs
- The lollipop beside each windowed strip is that strip's own row mean
- A condition's coupling number is named `sci_win_mean`
- The subject, hyper and prep-raw tables report both SCI estimators side by side
- Every coupling scalar says which grid it was measured on. All four are pinned to 10 s whatever `--qc-window` is
- A run is called an outlier on one number, its mean |z| over every metric
- A cohort too small to have a middle says so under every panel that measures distance from one

### Removed
- The Evoked response panel, from the raw report. It stays in the GUI
- The Epoch preview panel, which was hidden on load and never drawn into

### Fixed
- A dyad's per-condition quality table read the wrong stretch of each member's recording
- The subject index read its motion-correction column a hundred times low
- The raw viewer's GVTD carpet was blank, on every page
- A figure addressed at one condition's window goes back to that window on double-click
- The channel-detail epoch panel drops a condition with one trial
- A condition page in the raw viewer showed the whole run's figures
- Event onsets were drawn a cropped recording's start-time late
- The GUI's Data Prep page failed to load a run, reporting it as an unreadable file
- The GUI's channel-detail markers come off the recording being drawn
- A subject whose label begins with `s`, `u`, `b` or `-` no longer overwrites another subject's ratings
- Rating a section on a condition page no longer overwrites the run's own verdict
- The channel-quality grid shows the row that decides the verdict
- A condition page's channel-set table no longer carries an empty "Mean SCI (whole run)" column
- A condition's rejected channels go through the same screening as everything else

## [0.36.0] - 2026-09-11

### Added
- `--wtc-arrow-min` sets the coherence a cell has to reach before its phase arrow is drawn, when no significance was computed. Display only
- The quality record carries a `by_condition` section, one entry per annotated condition
- CNR is measured per condition, per channel set, with the epoch count beside it
- The QC page offers `fnirs-hyper index`, which rebuilds the dyad landing pages from the tables on disk
- A condition page says how many cycles of the slowest analysed frequency its window holds, with a caveat under four
- A per-channel spike rate, in the quality record and the per-channel table. Reported only
- The motion-correction footprint is split by separation, `motion_long` and `motion_short`
- A condition page carries the trial image and the per-trial quality heatmap

### Changed
- A condition page shares the run's carpet and per-channel motion figures instead of copying them
- The motion and PSD figures are a third and a fifth of their old size, with no pixel changed
- A condition page's per-channel motion figure is scaled to that condition, with the run's maximum stated beside it
- A condition page's bad-segment zoom shows that condition's flagged segments
- The HbO-HbR panel is one figure for both stages, on one channel order, with polarity on a diverging colour pair
- The subject report reads as a flat document instead of a stack of cards
- The report says less above each table, and each metric says more on hover
- Every panel that carries a legend puts it above the plot on the right
- The per-channel metrics CSV lists its columns in the order the tables print them
- A coherence map marks both ends of every condition
- The low-frequency drift trend follows the recording length, one degree per 150 s, instead of always being a cubic
- The cardiac band fraction is shown as a description, not as a quality reading
- The per-condition report pages read the record's numbers rather than computing their own

### Removed
- The Cardiac Power pass rate, and the CP >= 0.5 line behind it. `cp_mean` stays, described as what it measures
- The dotted reference cone on a condition's coherence map

### Fixed
- The group table's long-channel columns keep the runs whose montage is entirely long
- `--tstart` / `--tend` reaches the pseudo-dyad null
- A frequency axis narrower than two decades labels all its ticks the same way
- A dyad analysis runs on a multi-session tree
- A condition page's HbO-HbR correlation column is measured on that condition
- A condition too short to transform leaves its spectra out
- The per-trial panel scores the windows it names on a cropped recording
- A condition page no longer says the epoch and topography panels above it are deliberately empty when they are not

## [0.35.0] - 2026-09-11

### Added
- `fnirs-prep crop --margin` keeps extra seconds on each side of every segment and records the span asked for. `--margin auto` takes the width from `--band-fmin`
- `--wtc-cond-transform` transforms each condition on its own, over a cut padded by `--wtc-cond-pad-s`
- A dyad run says when its input is already a cut, naming the source window
- The coherence map panels take a selector per brain, so a crossed run can read any pairing at full size
- `--epoch-single-trial` draws the epoch section on a design where no condition repeats
- A dyad landing page, `group-<id>_index.html`, one row per analysed window. `fnirs-hyper index` rebuilds it over an earlier tree
- Each condition of a dyad gets a report page of its own
- The inter-brain correlation is computed per condition, matrix and connectogram
- Each condition page reports its own channel quality, sliced out of the quality record. It reports rather than re-decides

### Changed
- Both chromophores are on the page at once, HbO above HbR, and the chromophore switch is gone
- Per-condition results are the default. `--by-condition` replaces `--wtc-by-condition`, and `--no-by-condition` turns the pass off
- Every heatmap prints its values, through one drawing, with text ink taken from each cell's own colour
- Phase arrows are drawn only where the coherence stands out, and inside the cone either way
- A condition's map shows the cone it would have had if that block had been cut out on its own, as a dotted line
- The cone of influence is masked by default. `--no-wtc-mask-coi` averages the whole band, and `n_valid_frac` reports the share either way
- A rejected channel keeps its row in the coherence tables, blank, so every dyad's table has the shape of the montage
- `--tstart` / `--tend` select a window instead of cutting the recording. Numbers produced with these flags before are not reproducible
- `--bads-scope subject` says when it did nothing
- The whole-run dyad page says it is a summary
- The dyad report links its figures instead of carrying them. One page went from 174 MB to a few hundred kB
- The coherence maps carry the relative-phase arrows
- The cells outside the cone of influence are washed out rather than only outlined
- A design of one long block per condition skips the epoch section unless `--epoch-single-trial` says otherwise
- The grand mean is drawn on the denoised signal
- The evoked response is drawn per channel along its own source-detector path, one head per condition, with a slider
- Short channels get their own row in that map, captioned with the share of the long peak they reach
- The grand mean draws its short channels dotted
- The grand mean shades the task block, draws a zero line, and gives each condition a taller panel
- Per-condition pages carry the same All / Long / Short table as the run's own page
- The motion panel reports GVTD over the three channel sets, each keeping its before to after pair
- The stage-metric panel says which of its columns the metrics table will not match
- The per-trial heatmap says it reports whole-montage numbers
- `--gvtd-censor` takes the channel set to censor on: `long` (default), `short` or `all`
- Per-condition pages carry the event timeline
- Per-condition pages say why the trial image and the per-trial quality heatmap are not on them
- The pooled trial image names the condition each row came from
- The channel map is centred
- The per-trial quality panel is drawn the way the channel quality grid above it is
- Quantitative Metrics reads as one section, its lists now tables at their content width
- A metric name carries its tooltip cue wherever it is written
- The Raw Signal section drops its epoch panel, which drew cardiac ripple on a deliberately unfiltered stage

### Fixed
- The condition boundaries on a dyad coherence map are drawn where the conditions are, not late by the alignment offset
- The dyad analysis finds each member's quality record on a multi-session tree
- An ROI's reported share of cells inside the cone no longer falls with the number of channels rejected in it
- The condition blocks show on the dyad coherence maps
- A condition of one or two trials no longer sets the scale for the real ones
- The channel selector marks short-separation pairs
- The trial image caption no longer claims trials were smoothed on runs too short for it
- The global correlations either side of the confound regression are measured on the channel set their panel reports
- The per-condition haemoglobin panel reports the long channels, matching its own note
- The PSD panel's caption names the configured cardiac and respiration bands
- The Methods paragraph describes the screening rule the pipeline actually applies
- The grand mean no longer squeezes its curves into a sliver at the left of each panel
- The trial image no longer blends neighbouring trials into each other
- The per-trial quality heatmap keeps square cells on a run of few trials
- A per-condition page's channel map is the size of the run page's

## [0.34.0] - 2026-09-10

### Added
- The raw signal quality panel draws the coefficient of variation per channel and per window. SNR appears in that row's hover

### Changed
- CV and SNR are measured over short windows and averaged, so stored CV, SNR and `snr_pass_rate` differ from earlier runs
- The quality record carries CV and SNR per window
- A per-condition page screens on its own stretch. The recording is still processed under the run's verdict
- Per-condition pages report the GVTD above-threshold share and the spike share, counted from the run's own flags
- The windowed GVTD series is stored per separation set. The unsuffixed keys are the long channels now, so stored values differ from earlier runs
- Per-condition pages carry the global correlation, the HbO-HbR correlation and the cardiac and respiration bands, on that condition's own cut

### Fixed
- The before to after GVTD motion share is counted against one cutoff
- The channel quality maps show the screening verdict rather than colouring by SCI alone
- The epoch and grand-mean panels draw one condition per row, on a shared scale
- Per-condition pages show the channel quality maps, the per-channel motion figures and the denoising carpet
- Per-condition pages report the GVTD percentiles, the frame counts, the correction footprint and the mean CV
- Per-condition pages name the stage each motion row is measured on
- `gvtd_filt_p95` is marked as a metric that decides whether a run is usable

## [0.33.0] - 2026-09-09

### Added
- `fnirs-pipe --by-condition` writes one QC report page per annotated condition, sliced out of the run's own windowed pass
- `fnirs-qc prep-raw --by-condition` does the same for the raw QC report

### Changed
- The GLM activation panel switches between conditions instead of stacking them, on a shared colour scale

### Fixed
- The optode, flat-map and brain figures colour channels by the run's own SCI threshold instead of a fixed 0.75
- The per-channel timeseries on a per-condition page no longer draws its data and its condition shading in different time frames
- The per-channel epoch panel is no longer empty on a per-condition page
- The GVTD panel scales to the bulk of its trace rather than its largest spike
- The epoch and HRF preview draws one panel per condition, with HbO red and HbR blue
- A run no longer warns about the columns of its own events file
- The group quality table names the metric pairs that straddle the bandpass
- `fnirs-pipe` records the coupled-window share it screened on. Records written before this leave the row out
- The coupled-window share appears in the reports' scalar panels
- `--min-good-frac` now reaches the per-trial quality panel
- The channel quality summary no longer describes a rule it stopped using

## [0.32.0] - 2026-09-09

### Added
- `--min-good-frac` sets how much of a recording a channel has to be coupled for, default 0.75
- `--screen-scope task` counts coupled windows only inside the annotated task blocks. Default stays `run`
- The raw QC record carries the coupled-window share per condition. Reported only
- The pseudo-dyad null follows `--wtc-by-condition`, in its own table and at no extra iterations
- `fnirs-prep crop --input-desc` cuts a processed stage instead of a recording, keeping that stage's entity, bandpass and bad-channel marks

### Changed
- Channel screening counts coupled windows instead of comparing two whole-run averages. Rejected-channel sets differ from earlier runs, in both directions
- `--wtc-by-condition` reads each window off the whole-run transform instead of transforming it on its own, which changes its numbers. The channel pattern is unaffected
- A run whose input was written by `fnirs-prep crop` now stops and names the order to use. `--allow-cropped-input` runs it anyway
- A run that replaces an earlier one made with a different passband now says so

### Removed
- `--short-channel pca`, leaving `none` and `mean`. A run still asking for it stops and says so

### Fixed
- Censored spans no longer enter the GLM as a task condition
- Editing markers no longer drops the recording's auxiliary channels
- Editing markers no longer fails on a derivatives directory that does not exist yet
- `--wtc-by-condition` no longer takes every window late on a dyad aligned to its first shared trigger

## [0.31.0] - 2026-09-08

### Added
- `--short-max-dist` / `--long-min-dist` / `--long-max-dist` set what counts as a short and a long channel, in mm. They were fixed in the source
- Wavelet coherence runs on both chromophores, `--wtc-chroma {hbo,hbr,both}`, default both. Both costs twice the time; pass `hbo` for the old behaviour
- `fnirs-hyper run` reads the separation bands off the members' quality records. Members preprocessed with different bands are refused

### Changed
- The WTC band-mean tables gained a `chromophore` column, and the saved maps split one archive per chromophore, so tables written either side of this release will not concatenate

### Removed
- `--gvtd-channels`. GVTD always covers the long channels now

## [0.30.0] - 2026-09-07

### Added
- The dyad report says so when ISC runs on a stage with no bandpass on record
- The GVTD panel draws the short channels on their own row, with the carpet split into a long and a short block. The verdict still comes from the long row
- `--psp-threshold` sets the second screening line, which was fixed at 0.1
- `fnirs-qc hyper-raw` exposes the three windows its figures use
- `--epoch-tmin` / `--epoch-tmax` set the trial window the subject report works in, which was pinned to -5 to 25 s
- `--epoch-chunk-duration` cuts a long task annotation into trials the epoch figures can average
- The WTC sidecars record the wavelet grid
- The raw QC report and the interface judge long channels separately, with an All / Long / Short comparison
- The raw QC report and the interface show every per-channel metric, and write the per-channel metrics CSV
- The subject report gained the event timeline and the per-trial quality panel
- The analysis page offers the bandpass design, `--filter-method` and `--filter-order`
- `--gvtd-censor` marks the frames GVTD flags as `BAD_gvtd`, with `--gvtd-censor-n-std` and `--gvtd-min-epoch-s`. Off by default
- The hyperscanning and group reports end with the same closing sections the subject report does
- The hyperscanning reports carry a Methods paragraph, continuing from a member's preprocessing into the alignment and the cross-brain measures. Four of its citations are placeholders until `references.bib` is filled in
- The hyperscanning reports carry their provenance, as a diagram and as a table
- `fnirs-hyper run` writes a run record to `group-<id>/logs/`
- The raw QC report ends with an errors and warnings panel
- The hyperscanning raw report writes its coherence tables, whole-record and windowed

### Changed
- A channel over 45 mm is a long channel now. On a montage with channels past 45 mm every long-channel metric moves, GVTD included
- `fnirs-qc hyper-raw`'s coherence band is `--coh-fmin` / `--coh-fmax`. The old names still work
- The subject report's per-trial panel scores each trial over the event's own duration when no epoch window is given
- The analysis page offers the PSP threshold and the epoch window
- `fnirs-hyper run`'s report is named `desc-hyperpost`, the way every other report is named
- The hyperscanning reports print their per-subject metrics the way every other report does. Some labels change
- The all-channel PSD panel draws a mean per separation group instead of pooling every channel into one curve
- The channel quality heatmap prints its channel names in the same colour in the report and in the interface
- Every PSD panel says which stage it is measured on
- Channel screening rejects a channel that fails SCI or PSP, where it tested SCI alone. Runs reject at least as many channels as before
- The CV pass line in the channel quality grid is 5% (Lloyd-Fox 2009) rather than 50%, and the SNR line is derived from it
- The per-trial quality panel prints its numbers the way the rest of the report does
- Every QC report is rendered into one shared page shell
- The reports share one stylesheet for prose, tables and figures

### Fixed
- `--short-channel` built its regressors from every channel on a recording with no registered optode positions. Such a run now skips the regression
- The per-channel motion figure measured GVTD over every channel while the panels around it measured the long ones
- `fnirs-pipe` wrote no quality record for any run, and the failure was logged rather than raised
- "Long channel" meant a bounded band in one half of the package and anything over 10 mm in the other. One rule now
- Crossing the channels drew both axes from one member of the dyad, dropping every pairing that used a channel only the other member kept
- The dyad matrices are indexed by the montage, so every dyad's matrix has one shape
- ISC returned a number for two members recorded at different sampling rates. It refuses now
- The crossed WTC matrix drew its axis from one member of the dyad
- `fnirs-qc hyper-raw` scored its per-subject quality table over every channel where the individual reports score the long ones
- `--gvtd-censor` failed on any recording that has event markers
- The interface showed no SCI at all in its channel table
- The per-channel PSD shaded the cardiac and respiration bands at fixed frequencies instead of the run's own
- The interface's all-channel PSD panel never appeared. It and a channel's own spectrum are separate panels now
- The interface's Per-channel Metrics table stayed empty on every run
- The same metric could read as passing in one view and failing in another
- A figure that failed took the whole group report with it

## [0.29.0] - 2026-09-06

### Added
- The interface exposes the SCI/PSP window and the epoch window, and can run the per-trial QC panel
- `--gvtd-channels` chooses which channels GVTD covers, defaulting to the long ones

### Changed
- "What each denoising step did" sits directly under the metrics table
- The quantitative metrics table reads across instead of down: one column per metric, one row per channel set
- The provenance diagram says less; the quality record box no longer lists every stage it covers
- The correlation matrix draws its lower triangle only
- The interface was restyled, with the run picker beside the subject it belongs to
- `--epoch-tmin` and `--epoch-tmax` also set the window the epoch figures are drawn over
- The motion figures were redrawn: spike segments shade the traces themselves and the correction footprint sits directly above
- The GVTD and per-channel derivative traces are drawn at their own resolution
- The report's section links and the rating row are one bar, and every figure section can be rated

### Removed
- The "What the motion correction did, and what it cost" table. The GVTD panel answers the same question

### Fixed
- Short channels lost their HbO-HbR correlation in the per-channel table and the metrics CSV
- "What each denoising step did" averaged long and short channels together. It is measured on the long channels now
- A spike on the last sample of a recording was never drawn
- The 3D layout never marked the selected channel
- Peaks in the GVTD and per-channel derivative traces were drawn slightly late
- Clicking a cell in the Signal Topo selected a different channel
- The interface drew its per-channel figures at a fixed DPF of 6
- The "Raw Signal" section was not raw. Every figure in it is on the uncorrected stage now, and the section says which
- The interface measured GVTD over every channel where the subject report measures it over the long ones
- A subject's runs shared one set of ratings. Ratings saved before this release are not read back

## [0.28.0] - 2026-09-06

### Added
- Run notes for a montage the metrics cannot split: no registered positions, channels outside both separation bands, or short-channel regression with no usable short channel
- Run notes in the subject report, listing the sections a run left out because its data does not carry what they need
- `--filter-method` and `--filter-order`. The bandpass is now a zero-phase Butterworth of order 4; `--filter-method fir` keeps the linear-phase option
- The filter is recorded on every stage's sidecar, not only on `desc-filtered`
- A warning when a coherence range reaches past the bandpass
- `--wtc-by-condition` runs the coherence inside each task annotation's window as well as over the whole recording

### Changed
- The haemoglobin metrics are split by source-detector separation. These numbers change: rerun anything measured before this
- `fnirs-qc prep-raw` writes the same record shape the pipeline does. Existing flat records are still read
- The quantitative metrics panel reports SCI, PSP, SNR, CV, amplitude and retention in three columns, All / Long / Short
- The per-channel metrics table is grouped by source-detector separation, and short channels carry their SCI, PSP, SNR and CV instead of a dash
- The HbO-HbR correlation panel separates short channels from long ones, drawing them in grey and leaving them out of the pass/fail colouring
- The provenance diagram matches the interface's pipeline DAG
- The subject report's five before/after strips are now one small panel per metric across every stage on disk
- Metrics compared across the bandpass are measured in a common band. These panels will not match the per-stage values in the metrics table
- The motion figures drop the unfiltered GVTD trace. `gvtd_mean` and `gvtd_p95` are unchanged
- The per-channel motion row is labelled `|dOD/dt|` rather than TVD. The stored `temporal_derivative_variance` is unchanged
- The PSD panel draws the filter's own response over the filtered signal, and names the filter it drew

### Fixed
- A recording whose input was already optical density could lose its whole report
- The per-channel derivative trace showed pulse, not motion. It is now taken at full resolution over the 0.01-0.5 Hz band
- The low-pass filter barely filtered. Every filtered file, and every number measured on one, changes: rerun anything written before this
- A run whose markers cannot be epoched filled the console with warnings. It is now recognised up front and reported as a note

## [0.27.0] - 2026-09-05

### Added
- The subject report shows HbO-HbR correlation before and after denoising, with a per-channel strip
- Contrast-to-noise ratio, per channel and across denoising. Task runs only
- SCI and PSP before and after motion correction, as a per-channel strip
- Relative phase, drawn as arrows on the ROI coherence maps. The maps saved with `--wtc-save-maps` carry it too
- The crossed coherence reaches the report as figures: a channel-by-channel matrix of the band means and every ROI-by-ROI map on one grid
- `--bad-channels` also takes a table, one row per subject. A plain list still applies to everyone
- One quality-record section per haemoglobin file the run wrote: `preproc`, `filtered`, `resampled`, `errts`

### Changed
- The coherence maps label their frequency axis at the decades, and run frequency downward
- The `final` quality-record section is gone, replaced by the sections above. Group tables written before this hold `final_*` columns that will not be written again
- The PSD panel plots the files the run actually wrote, one line per stage. A run with no post-processing still gets a simulated curve, labelled as one
- The SCI/PSP panel is one stage throughout, the uncorrected optical density

### Fixed
- A hand-picked bad channel could survive into one chromophore. Either wavelength now marks both
- Connectivity and ALFF counted rejected channels. Every FC and ALFF table on a subject with a rejected channel changes

## [0.26.0] - 2026-09-04

### Added
- `fnirs-hyper`, a tool for dyad analysis: `run` writes the WTC and ISC report per dyad, `band` re-averages saved maps over another frequency band, and `merge` concatenates the per-dyad tables
- `fnirs-hyper run` says what the metrics will run on before it starts; `--check-only` stops after that and writes nothing
- `fnirs-hyper run` ends by saying whether the merged tables are behind the per-dyad ones, and with what command to catch them up
- The subject report shows GVTD before and after motion correction, both sides measured on the same channel set
- The carpet and GVTD figure carries the corrected recording too, on the uncorrected scale and threshold

### Changed
- The carpet and GVTD figure is interactive, so the before and after traces can be toggled from the legend
- The dyad analysis left `fnirs-qc`, which keeps `hyper-raw`. No output file changes name
- The pseudo-dyad null is a flag on the dyad run again, `--wtc-pseudo N`, sharing every parameter with it. Its crossing stays `--wtc-pseudo-cross`
- One name per parameter for the frequency band: `--band-fmin`, `--band-fmax` and `--mask-coi` are gone in favour of their `--wtc-` spellings
- `fnirs-pipe ... group` no longer demands the preprocessing flags, and a bad command line is rejected before the pipeline imports
- The QC Reports page drives both tools and asks for one directory

### Removed
- `fnirs-qc hyper-post`, `hyper-null`, `wtc-band` and `group-hyper-wtc`, with no forwarding. Use `fnirs-hyper run`, `--wtc-pseudo`, `fnirs-hyper band` and `fnirs-hyper merge`
- `--session-label` and `--skip-bids-validation` on the dyad commands, which used neither

### Fixed
- A `--no-report` run left the dyad analysis with nothing excluded. Every coherence value on a dyad with a rejected channel changes

## [0.25.0] - 2026-09-04

### Added
- `fnirs-qc hyper-post` / `hyper-null` / `hyper-raw` take `--tstart` / `--tend`, restricting the synchrony metrics to one window
- `fnirs-qc prep-raw --epoch-qc` adds a per-trial section to the raw report, scored per event window
- `fnirs-prep crop --align trigger --trigger-name TEXT` measures the window and every segment onset from a named annotation

### Changed
- The 2D optode layout is projected onto a head outline, with sources and detectors drawn and named. Read source-detector distance from the channel table rather than off the figure
- The hyperscanning reports and `hyper-raw_sqm.tsv` carried SCI and nothing else. Both now carry every scalar in the record, and gain a per-subject quality table
- The raw QC report moves into `sub-<id>/` with the subject's other reports
- The pseudo-dyad null is its own command, `fnirs-qc hyper-null`, and no longer inherits `--wtc-channel-cross`. `hyper-post --wtc-pseudo` is gone
- The null records how many iterations it ran, and `group-hyper-wtc` refuses to merge nulls of different lengths

### Removed
- `fnirs-qc window-raw`. Crop with `fnirs-prep crop`, then run `prep-raw` and `group-raw` over the cropped tree
- `fnirs-qc epoch`, replaced by `prep-raw --epoch-qc`

### Fixed
- The raw signal panel named its traces in the wrong order
- The 2D optode layout drew every channel link at half length, and drew no optodes
- Optodes rendered off the brain when the montage was already in MNI. Datasets in head space are unaffected
- The correlation matrix labelled only every other channel
- `fnirs-qc group-hyper-wtc` merged only the channel table
- `--bads-scope` never reached the coherence. Every WTC value on a dyad with a rejected long channel changes

## [0.24.0] - 2026-09-02

### Added
- `--wtc-pseudo N` writes a pseudo-dyad table, the same band means against a phase-scrambled partner
- WTC band-mean tables gain `coherence_z`, the Fisher r-to-z group statistics should average
- `--wtc-roi-min-channels` drops an ROI cell resting on too few channel pairs
- The subject page says which run is the odd one out, and which channels each run rejected
- `hyper-post` writes `hyper-wtc-roichan.tsv` whenever `--roi-mapping` is given, with `n_ch` behind each mean
- `--wtc-channel-cross` crosses every long channel with every other across the two brains. The off-diagonal is exploratory
- `--wtc-save-maps` keeps the full time-frequency maps, and `fnirs-qc wtc-band` re-averages them over another band
- `--bads-scope subject` unions each subject's rejected channels over their runs
- `hyper-post` writes `hyper-bads.tsv`: which channels the inter-brain metrics excluded, and which run rejected each
- The group CSV accepts optional `session` and `run` columns
- The QC Reports page offers `wtc-band`, the saved WTC maps and the two new `hyper-post` options
- `--aux-regressors` puts the recording's auxiliary channels into the confound regression, extracted to `desc-aux_timeseries.tsv.gz` and band-limited to the data's own passband
- `--aux-channels` picks individual aux channels by name

### Changed
- The cone of influence is no longer masked by default. `--wtc-mask-coi` restores it. Every coherence value moves, short conditions most
- `hyper-wtc-roi.tsv` and `--wtc-roi-cross` are gone. `hyper-wtc-roichan.tsv` is the ROI table and `--wtc-channel-cross` builds the cross-ROI matrix
- The subject QC report is one report per run, `sub-<id>_task-<task>_qc.html`, with `sub-<id>_qc.html` as an index over them
- WTC computes only the wavelet scales its frequency range keeps, 1.8x faster. Coherences are unchanged bit for bit
- A group's outputs live in `group-<id>/`, the way a subject's live in `sub-<id>/`. Filenames are unchanged and an existing tree is still found
- The quality record is drawn as a node rather than a step in the provenance diagram

### Fixed
- The provenance diagram drew steps that had not been run. Those nodes are dashed and labelled `file missing` now
- `fnirs-prep crop` dropped the aux group, so `--aux-regressors` had nothing to regress
- `hyper-post` excluded the wrong run's bad channels
- A group member with two sessions, or two runs of one task, had one silently analysed and the other dropped
- A missing task could be analysed as a different one
- Per-channel metrics CSVs are written for every run, not only the last one processed
- A `--task-label` run no longer rewrites the quality records of the tasks it did not process
- `fnirs-qc group-hyper-wtc` refuses to merge crossed and homologous channel tables
- The provenance diagram is drawn per run
- `--mode glm` warns when the data is high-passed but the drift model does not span that band

## [0.23.0] - 2026-08-30

### Added
- `--mode denoise --fc` writes the connectivity products rest mode writes, taken from the confound residual
- `fnirs-qc group-hyper-wtc` merges every dyad's coherence tables into one long table carrying `group_id` and `task`. It refuses to merge tables averaged over different bands
- A QC Reports page in the GUI, covering the `fnirs-qc` layer that was command line only. Each run's report is shown inline
- The Analysis page's pipeline diagram switches to the real provenance graph once a run has written sidecars
- The Analysis page offers the drift model, its cutoff and order, the ROI mapping and the connectivity flag

### Fixed
- The GUI could never produce a working `--mode glm` or `--mode rest` command, both requiring a drift model the page had no control for
- The GUI offers short-channel regression in every mode rather than GLM alone
- The GUI's Stim Duration box was drawn but never read
- The subject report's navigation bar links to the FC and GLM panels whatever mode drew them
- The reproduction script named the residual step differently from the pipeline it mirrors

## [0.22.0] - 2026-08-30

### Added
- `--mode glm --fc` writes the same connectivity products rest mode writes, from the GLM residual
- The subject report draws the ROI-to-ROI connectivity matrix
- The subject report draws ALFF and fALFF on the optode layout
- `fnirs-qc hyper-post` writes the inter-brain correlation matrix behind the ISC panel
- `--mode denoise` honours `--short-channel` and `--drift-model`, writing the residual as `desc-errts`. No task model, no ALFF, no FC
- The quality record gains a `motion_post` section, measured again on the motion-corrected file
- The quality record gains a `windowed` section: per-window SCI, PSP and GVTD with the window length they were binned on
- Metric tooltips say which processing stage the number was measured on
- `fnirs-recon --optode-frame` names the space the SNIRF's optode coordinates were measured in, which BIDS requires
- `fnirs-prep crop` accepts a `task` column in the segments table
- `--wtc-roi-cross` crosses the two brains' ROIs instead of pairing each with its counterpart. Needs `--roi-mapping`
- The Methods paragraph describes the confound regression, naming the short-channel strategy and the drift basis
- `--wtc-mc-count` sets how many surrogate series stand behind each significance contour (default 300, unchanged)

### Changed
- `snr_pass_rate` counts every channel, not only the ones with a finite SNR, and `n_flat_channels` says how many have none. `snr_pass_rate` falls for any run that has them
- The subject report reads the spike and motion-correction spans from the quality record instead of detecting them again
- The subject report's per-window SCI and PSP panel reads the quality record, and the series are measured on the motion-corrected file
- Tables you hand the package read by extension: `.tsv` tab-separated, `.csv` comma-separated, anything else sniffed
- zALFF standardizes by the population SD (n) instead of the sample SD (n-1). Values grow by `sqrt(n/(n-1))`

### Fixed
- Accented author names reach the Methods text as letters rather than LaTeX
- `--config` takes flat top-level keys, and the reference and the presets now show that. A sectioned file was silently ignored
- `fnirs-prep crop`, `edit-markers apply`, `align` and their two GUI equivalents never produced a file. They now use the same writer as the pipeline
- Writing a cropped recording put its markers at their position in the original recording
- `crop` names each segment's sidecars after the segment, and finds `_optodes.tsv` / `_coordsystem.json`
- `crop`, `edit-markers` and `align` carry `participants.tsv` and `README` into their output

## [0.21.0] - 2026-08-24

### Added
- Topographic maps of the condition-averaged response in the subject report, one row per condition and chromophore
- `fnirs-qc hyper-post --desc` picks which per-subject stage the inter-brain metrics read (default `preproc`). Optical-density stages are refused
- `fnirs-qc hyper-post` writes `hyper-wtc.tsv`, and `hyper-wtc-roi.tsv` with `--roi-mapping`: one coherence value per pair and channel
- Rest mode writes `desc-<chromo>_fcseed.tsv` and `_fcseedz.tsv` when `--roi-mapping` is given
- The subject report draws the seed maps as flat maps, one per seed ROI and chromophore

### Changed
- The subject report's trial images are split by condition, with the pooled image kept first. The figure files are now `trialimage_*.html`
- The per-channel layout figure shows the condition-averaged response. Runs without events keep the continuous view
- The hyperscanning WTC, coherence and ISC read long channels only. Any montage with short channels loses those rows from its hyper figures
- Inter-brain metrics stop when the members of a group were sampled at different rates
- Functional connectivity is plain Pearson, not a shrinkage estimate. Every FC and FCZ value changes; weak connections change most
- Quality records store `qc_window_s`, so a group report whose subjects were run at different window lengths says so
- `Mean PSP` is labelled with its 10 s window, which does not follow `--window-length`
- zALFF standardizes by the sample standard deviation. Values shrink by `sqrt((n-1)/n)`
- `--motion-correction wavelet` reaches artifacts up to about 25 s long, where it used to stop at about 1.6 s

### Fixed
- ISC, band coherence and windowed coherence pair channels by S-D label instead of by position. These values change for a dyad whose members had different channels rejected
- The `raw_long` and `raw_short` quality sections keep rejected channels
- Provenance is no longer empty when the pipeline is handed a recording it did not read from disk
- A run where no channel passes the SCI threshold stops there and says so
- Rest mode no longer writes ALFF and fALFF when the drift model leaves linear drift in the data
- `--motion-correction wavelet` sets its outlier threshold per wavelet scale over the whole recording
- The windowed SCI and PSP heatmaps plotted each window at roughly half its true time
- GVTD per-window series share the time axis of the windowed SCI and PSP
- A cardiac band the filter cannot use no longer takes the GVTD per-window series down with it
- A metrics database written by an older version gains any column it is missing
- Fisher z no longer zeroes the leading diagonal of a non-square matrix
- The seed-map sidecar lists the channels each seed was built from

## [0.20.0] - 2026-08-11

### Added
- `<sub>_<task>_desc-sqm_nirs.json`, one quality record per run, grouped into sections by what each metric was measured on
- Sidecars record per-channel SCI and the respiration band, so a record can be rebuilt from an output directory alone
- The QC report's metrics panel explains itself: hover any metric for what it is and which way is good

### Changed
- Quality metrics are measured once and written once, so the report and the file agree
- `sub-XX_sqm.toml` and `sub-XX_sqm_raw.toml` are gone, replaced by the per-run JSON
- The metrics database stores one row per section and fills in the task
- `fnirs-qc prep-raw` writes `desc-sqmraw` rather than the `desc-sqm` the pipeline uses. The group report reads both

### Fixed
- The resting-state FC heatmap and connectogram lost their HbR half, or failed to render at all
- A single failing metric no longer discards a run's whole quality record
- The QC report says so when a run's quality record is unreadable
- The provenance table lists the metric names again for the quality record
- `<sub>_channel_metrics.csv` gains the run's task and session
- Short and long channels are decided the same way everywhere
- Cardiac Power was silently unavailable on any recording with a rejected channel
- Peak spectral power is measured on optical density, matching the windowed series beside it
- Rebuilding a quality record from an existing derivatives tree lost every SCI metric where the sidecar carried no stored scores
- The group report's boxplots grouped every metric under "Other"
- Per-wavelength CV vanished from a record whenever the CV/SNR step failed
- A derivatives tree that has been moved, or is read on another machine, keeps its raw-signal quality metrics

## [0.19.0] - 2026-07-31

### Added
- `fnirs-qc hyper-post --wtc-seed` makes the wavelet-coherence significance contour reproducible
- Sidecars record the shape of the data each step left behind: channel count, bad channels, sampling rate and duration
- The provenance diagram shows the settings each step used and the data shape at every node
- `*_alff.tsv` and `*_glm_results.csv` gain a `bad` column, and the sidecars list the rejected channels, so the table shape stays predictable
- The quality-record checkpoints appear in the provenance diagram

### Changed
- The provenance diagram moves to `sub-XX/figures/provenance.png` and is shown in the QC report
- The two quality checkpoints name the metric families they computed
- The QC report's Provenance section lists every output with the step that made it
- The Methods paragraph is built from the sidecars the run wrote rather than from the configuration
- The provenance diagram renders at 300 dpi
- Per-subject log, run record, script and provenance diagram lose the timestamp in their names; a re-run overwrites them
- The 3-view brain figure colours the source-detector links by SCI, on an opaque surface
- Writing two different files under one pipeline stage now fails instead of crediting an output to the wrong source
- Postprocessing rejects a `desc-` that disagrees with the stage stamped on the data
- GLM outputs carry the subject and task of their input

### Fixed
- A subject whose short channels were all rejected for one chromophore crashed the GLM
- `--drift-model cosine` without `--drift-high-pass` says so before the run starts
- The Methods paragraph stopped after the Beer-Lambert sentence
- Channels marked bad during preprocessing were unmarked again as soon as postprocessing reloaded the data
- A short channel rejected by SCI was still averaged into the short-channel regressor
- mALFF and zALFF were standardized against a mean and SD taken over all channels, rejected ones included
- ROI connectivity averaged rejected channels into their ROI signal
- The brain figure drew each channel from its midpoint to its source instead of source to detector
- Design matrix labels overlapped when regressors were numerous
- Postprocessing ran once per matching subject found anywhere under the output directory, so a nested tree was reprocessed
- With more than one task per subject, every task wrote the same design matrix and results file

## [0.18.0] - 2026-07-28

### Added
- Sidecars record the file each output came from (`Sources`) and the step that produced it
- ALFF, connectivity, GLM and hyperscanning outputs now have sidecars
- Rest mode writes `desc-filtered` and `desc-errtsbroad`, previously discarded
- Each run writes a provenance flow diagram, PNG and mermaid
- `fnirs-qc provenance <output_dir>` renders that diagram for any past run, reading only the sidecars on disk

### Changed
- Cardiac and respiration band power and low-frequency drift are written only to the preprocessing checkpoint

### Removed
- Durbin-Watson on GLM residuals, which is ~2 by construction once prewhitening runs

### Fixed
- Cardiac and respiration band power was silently missing on recordings with bad channels
- Rest mode overwrote its quality record, losing the metrics computed earlier in the run

## [0.17.0] - 2026-07-28

### Added
- Standardized ALFF outputs `malff` and `zalff` in `*_alff.tsv`, for group-level comparison

### Changed
- The per-subject reproducible script is a step-by-step transcript of the run, one block per processing stage
- ALFF and fALFF are computed on a broadband residual so fALFF spans the full spectrum. Functional connectivity keeps the band-pass filtered residual
- Functional connectivity is written per chromophore as separate HbO and HbR matrices instead of one matrix mixing both
- Cardiac and respiration band power are reported per chromophore

### Fixed
- fALFF was close to 1 on nearly every channel, having been computed on band-pass filtered data
- The subject report showed a blank low-frequency drift value for HbO

## [0.16.0] - 2026-07-08

### Added
- Motion-correction footprint (experimental): which timepoints a correction repaired, drawn as a band beneath the motion figures
- Spike frame metrics, `spike_pct` and `spike_pct_frames`, alongside the existing spike count
- Single-trial heatmaps per channel and per ROI in the subject report, for task data
- Fisher z-transformed functional connectivity output, `*_fcz.tsv` and `*_fcroiz.tsv`

### Changed
- Windowed SCI, PSP and GVTD share one sliding-window length, exposed as `--window-length` (default 10 s)
- Spike detection runs on the motion-band-filtered optical density, so the count reflects motion rather than cardiac pulsation
- Global correlation is reported before and after the short-channel regression rather than across the bandpass
- The GVTD carpet figure splits the raw and filtered traces into separate panels. Metric values are unchanged
- Global correlation and spike metrics are marked experimental
- QC figures render at 300 dpi or more

### Removed
- tSNR on haemoglobin, which has no stable baseline. Raw-intensity SNR already covers channel signal stability

## [0.15.0] - 2026-07-06

### Added
- Wavelet motion correction, `--motion-correction wavelet`
- Filtered GVTD (0.01-0.5 Hz motion band) alongside the raw GVTD, with per-run motion summaries above an adaptive threshold
- Motion figures overlay the raw and filtered GVTD traces and draw the motion threshold
- Cardiac and respiration band power are also reported as a fraction of total spectral power
- `fnirs-qc epoch`: per-trial QC, recomputing the metrics per event window and rendering a trial by metric table
- Channel-standardized GVTD, `gvtd_vstd_*`, so high-dynamic-range channels no longer dominate the motion index
- Global correlation per chromophore, shown before and after denoising
- Durbin-Watson on GLM residuals
- Per-window GVTD reports p95 alongside the mean

### Changed
- The cardiac band, `--cardiac-l-freq` / `--cardiac-h-freq`, is required with no default, and now sets the band for SCI, PSP, Cardiac Power and cardiac band power alike
- The respiration band, `--resp-l-freq` / `--resp-h-freq`, is a required option too
- Spectral confound metrics renamed to `cardiac_band_power` and `resp_band_power`
- Spike count uses a robust MAD-based threshold
- Cardiac Power is marked experimental. SCI and PSP remain the primary channel-quality metrics
- Group report distributions are split into per-scale charts, and clicking a point opens that subject's report

### Fixed
- fALFF is the fraction of total spectral amplitude in the low band, matching its definition. ALFF is unchanged
- Cardiac Power sums band power and is computed on optical density, so it lies in [0, 1]
- `pct_data_retained` no longer double-counts overlapping bad segments
- Low-frequency drift amplitude is estimated from a polynomial trend fit, avoiding filter edge artifacts

## [0.14.0] - 2026-07-04

### Added
- ROI-level WTC in the hyperscanning post report, excluding the union of all subjects' bad channels
- The hyper post report marks bad channels: the ISC matrix blanks their rows and columns and drops their connectogram arcs
- Denoising before/after carpet in the subject QC report, HbO and HbR shown separately
- `--roi-mapping` on the main pipeline: a JSON file mapping ROI labels to channel lists
- ROI-level functional connectivity, `_fcroi.tsv`, in rest mode when `--roi-mapping` is given

### Changed
- `--exclude-channels` renamed to `--bad-channels`: channels are marked bad rather than dropped
- Quality metrics renamed from IQM to SQM (signal quality metrics)
- The motion QC carpet shows all channels instead of a single wavelength

### Fixed
- The PSD before/after panel and the run-record database reflect bandpass cutoffs set in a TOML config file
- The GVTD trace in the motion carpet is computed at full resolution, so it matches the reported metric

### Known Issues
- `compute_alff` numerical validation pending

## [0.13.0] - 2026-07-03

### Changed
- All CLIs follow the BIDS App convention: list options accept space-separated values as well as repeated flags
- `--help` output is grouped into labelled sections

## [0.12.0] - 2026-07-03

### Added
- A GUI Recon page: detect raw snirf files, assign BIDS metadata per file, preview the commands and batch-convert
- The command preview has a shell selector so line continuation matches bash, cmd or PowerShell
- The pipeline auto-detects already-OD input and marks intensity-only metrics as unavailable

### Changed
- `fnirs-recon --subject` help clarifies that labels are alphanumeric and can encode group

### Fixed
- The GUI emitted invalid multi-subject commands
- QC no longer crashes on low sampling-rate data: frequency limits clamp to the Nyquist frequency
- QC brain views no longer fail when the static-image export backend was missing

## [0.11.0] - 2026-05-25

### Added
- Self-contained rating viewers: static open works read-only, `fnirs-rate` open enables writes
- Section ratings and channel decisions persist through REST endpoints only. The HTML is never modified

### Changed
- The three rating apps were rewritten without regex reverse-parsing or string-concat injection

### Fixed
- The hyper rating app's channel-decisions sidecar is session-aware

## [0.10.0] - 2026-05-30

### Added
- `fnirs-qc group-raw` and `group-hyper-raw` aggregate per-subject and per-dyad metrics into cohort HTML reports
- `fnirs-qc window-raw` crops each subject's raw to a time window and re-runs the group layout
- The prep-raw record stores per-window SCI, PSP and GVTD, and group reports render them as time by subject heatmaps

### Changed
- The QC output directory follows a BIDS-derivatives layout
- QC filenames are standardised to the `_nirs.*` suffix with a BIDS `desc` entity
- The hyper raw report is an iframe shell with auto-resize

### Fixed
- `fnirs-qc` initialises logging

## [0.8.0] - 2026-05-23

### Added
- `fnirs-log merge`: JSONL run events consolidated into a SQLite database
- `PrepResult` exposes `sqm_raw` and `sqm_final` for downstream use
- Paragraph-style Methods boilerplate with a rendered tab in the HTML report

### Removed
- `--mode connectivity`, which was never implemented
- `--ica` postprocessing, ICA for fNIRS being experimental
- `--segments-path` / `--crop-tmin` / `--crop-tmax` on `fnirs-pipe`, superseded by `fnirs-prep crop`

## [0.7.0] - 2026-05-14

### Added
- `fnirs-gui` launches a Dash multi-page application
- Data Preparation page: SNIRF loader, editable stimulus marker table, per-subject metrics
- Analysis page skeleton wired into the sidebar router

## [0.6.0] - 2026-05-13

### Added
- Resting-state analysis mode, `--mode rest`
- ALFF and fALFF computation
- Resting-state functional connectivity across channels
- Resting-state QC figures and report section
- `--trim` discards leading and trailing seconds before analysis

## [0.5.0] - 2026-05-13

### Added
- ISC computation and heatmap in the hyperscanning post report
- WTC computation and figure builders
- Connectogram visualization for hyperscanning functional connectivity
- Hyperscanning post-processing QC report, with WTC, ISC and connectogram sections
- Hyperscanning group-level raw QC report, for multi-dyad channel summaries
- A normalization option for hyperscanning raw data

### Changed
- Hyperscanning moved from a standalone CLI into the pipeline package, subsumed under `fnirs-qc`

## [0.4.0] - 2026-05-12

### Added
- `fnirs-qc` CLI with `prep-raw` and hyperscanning subcommands
- An interactive raw QC viewer: a standalone HTML page with Plotly channel traces
- Cardiac power band metrics and tSNR per channel
- A channel quality summary figure in the prep-raw report

### Changed
- SCI computation corrected, and figure axis labels clarified across several panels

### Fixed
- Empty epoch subsets no longer crash the channel figure

## [0.3.0] - 2026-05-05

### Added
- Task GLM postprocessing: design matrix construction and contrast estimation
- Denoise mode: bandpass filter and optional resampling
- A GLM QC section in the per-subject report
- Raw residuals written as a separate SNIRF file after GLM fitting
- A bad-segment zoom panel that runs independently of the motion carpet

### Fixed
- The bad-segment zoom no longer depends on the motion panel having failed

## [0.2.0] - 2026-05-05

### Added
- A per-subject HTML QC report
- Report sections: OD traces, SCI/PSP heatmaps, GVTD timeseries, motion carpet, HbO/HbR correlation panel, PSD, brain views, short-channel PSD, metrics table
- Quality sidecar files, `sub-{id}_sqm.toml` and `sub-{id}_channel_metrics.csv`
- Uniform per-section error handling: a failed section renders a warning card rather than crashing the report

## [0.1.0] - 2026-05-04

### Added
- End-to-end single-subject preprocessing: OD conversion, SCI/PSP channel QC, motion detection and correction, Beer-Lambert law
- BIDS Derivatives sidecar writing, with an intermediate SNIRF at each prep step
- `fnirs-pipe` CLI with all planned preprocessing flags wired
- `fnirs-recon` CLI for raw to BIDS conversion
- Run record and script utilities
- Logging setup
