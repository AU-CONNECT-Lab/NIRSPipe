# Changelog

All notable changes to this project will be documented in this file.

<!-- Format: Keep a Changelog (https://keepachangelog.com/en/1.0.0/) -->

## [Unreleased]

### Fixed
- The cohort page's box charts widen their left margin for a long first label, which fell off the chart's left edge
- First-level GLM p values were far too large, their `df` being the number of regressors; `--p-correction` inherited them

## [0.79.0] - 2026-10-28

### Added
- The channel quality grids carry a PSP colour bar when PSP has no line
- Runs that compute FC after a regression also write the channel matrix of the series before it (`stat-pearson_desc-filtered_relmat.tsv`)
- The cohort page draws, per task, every run's connectivity values before and after the regression
- The cohort page draws each connection's r between head movement and FC across a task's runs, beside the spread with no relation and against distance, and writes them to `desc-fcmovement_qc.tsv`
- The run report's summary gives the recording's duration and the passing-window share screening requires, as the raw report does; condition pages give their window
- The run report's Run summary lists Mean CP and Flat channels per channel set, as the raw report does
- The run report shows the raw report's mean spectrum of the optical density, long and short channels and the heart rate, under the raw signal

### Changed
- The cohort page's per-task tables pool every acquisition of a task into one column
- The heart rate no longer reads the pulse's second harmonic, which it often did below about 75 bpm
- The Raw signal, the channel quality grids, the per-channel tables and `desc-channel_qc.tsv` list channels by source then detector number, a pair's two wavelengths side by side
- With `--events-path`, the run report's epoch figures and per-trial QC read the recording's markers again; the events table feeds only the GLM design
- Both reports print the DPF as `age 30 (6.30, 5.23)` or the numbers given, where the run report printed `age` and the raw report unrounded factors
- The optode flat map's legend sits top right outside the head, which it covered
- The provenance graph feeds outputs that share parents through one bar, so no arrow crosses a box; inputs name their dataset, labels repeat no word

### Fixed
- `--by-condition` runs no longer warn that the short-channel share, the wavelength check and the nuisance figure lack a per-condition form; the pages are unchanged
- The cohort page's connectivity caption names the matrix before the regression by its stage, `desc-resampled` under `--resample-sfreq`, where it always said `desc-filtered`
- The dyad raw page's captions call the strip where every member spiked at once green, as it is drawn, not amber

## [0.78.0] - 2026-10-28

### Added
- The quality record carries the heart rate, the median of the channels' spectral peaks, and the log warns when it lies outside the cardiac band
- The raw report shows the heart rate in its summary, metric table and spectrum, and notes when it lies outside the cardiac band
- The run report's summary names the cardiac and respiration bands and the heart rate, its spectra by stage mark the heart rate, and its notes say when it lies outside the band
- The dyad pages list each member's heart rate, and the dyad raw page notes a member whose heart rate lies outside its cardiac band
- The optode map and the 3-D brain views fill each source and detector by the share of its long channels rejected, so an optode that lost every channel stands out
- The cohort page lists each optode's share of its long channels rejected, pooled over each task's runs
- With `--roi-mapping`, the run report lists each region's long channels kept and the rejected ones, naming listed pairs that are not long channels
- With `--roi-mapping`, the raw report lists each region's long channels kept, as the run report does
- With `--roi-mapping`, the dyad raw page lists each member's long channels kept per region, side by side
- With `--roi-mapping` at group level, both commands' cohort page counts per task the runs in which each region kept no long channel
- The quality record, channel table and run report give each long channel's variance share carried by the fitted short-channel regressors, on a head map and as set means
- The run report draws the correlations among the nuisance series (short-channel, aux, drift, GVTD, head motion) and each one's correlation with the global signals
- `nirspipe-rate rate|raw|hyper --rater NAME` keeps each rater's ratings, channel decisions and marked stretches in files of their own, and the viewer shows who is rating
- The subject cohort page links the dyad cohort page when the tree holds dyads
- The rest FC panel and the stage correlation panel order channels by source then detector number, so S2_D9 precedes S2_D10
- With `--roi-mapping`, the subject page's FC panel over runs joined by `--combine-runs` is ordered and marked by ROI, as each run report's is
- With `--contrast-file`, the GLM page draws each t contrast's effect on the cortex, or on the flat map, as it draws the betas
- With `--events-path`, the run report's epoch figures read the design's events and name its conditions, instead of the recording's marker codes
- The quality record carries the pulse's HbO/HbR power ratio with the wavelength labels as given and swapped; the cohort page warns when most runs of a task read swapped
- `nirspipe-rate export` writes a rater column and takes `--rater NAME` for one rater's files; `--bad-channels` and `--bad-spans` refuse a table holding several raters

### Fixed
- The run report's per-channel table was drawn below the Quality Figures sections, which its unclosed table had swallowed
- The dyad motion panel's left margin is sized to its longest row name, which a fixed margin cut off
- A channel digitised a hair past a separation band's edge (a 10 mm short channel at 10.000025 mm) is classed in that band, within 0.01 mm, instead of in neither
- The GLM whitening figure's shaded bands drew a dot at each corner, which read as data points

### Changed
- Shared usable time names a pair a member rejected and counts it as lost, rows running worst to best; the dyad's usable numbers and table no longer count rejected channels
- `nirspipe-qc` participant requires the respiration band (`--resp-l-freq`, `--resp-h-freq`) as `nirspipe` does, unless `--population` fills it, and records it
- The raw report names the respiration band in its summary and shades it on its spectra, as the run report does
- The dyad raw page shades the respiration band its members share; a member record without one is refused, naming the rerun
- Spectra reach past 2 Hz when the cardiac band or the heart rate does, instead of cutting an infant's band off

## [0.77.0] - 2026-10-26

### Added
- A run whose optodes could not be placed shows its mALFF and GLM betas on the optode flat map in place of the cortex
- The run report's per-channel table names each pair's nearest cortical region (`aparc`), read from the run's channel table
- Raw QC takes `--bad-channels`, `--keep-spans`, `--keep-spans-from`, `--bad-spans` and `--bids-filter-file` as the pipeline does, and its verdict is the pipeline's for them
- The raw QC record carries the settings it screened with: lines, cardiac band, DPF, template, motion correction and the final bad channels
- `--population infant|child|adult|older` on `nirspipe` and `nirspipe-qc` participant fills the cardiac and respiration bands, one DPF for the batch and the head template; typed flags override it
- `--contrast-file` also reads a TSV or CSV table, one row per contrast and a weight column per condition; TOML still works
- `--drift-high-pass auto` in glm mode sets each run's cosine cutoff to 1/(2 x its slowest condition repeat), capped at 0.01 Hz, never below `--high-pass`

### Changed
- The channel FC matrices are drawn whole, ordered by ROI with rules and colour strips, on a range symmetric at the largest |r|; ALFF strips follow the order
- ROI colour strips are more saturated; the dyad page's "both spiking" strip is green, not the second member's amber
- Every dyad page draws the members in one palette, purple then orange; the motion panel and the circle plots used blue and orange, and purple and green
- The dyad raw page's after-correction GVTD rows take their own y range, so corrected peaks stay readable; values still use the uncorrected median
- A contrast weighting a condition that censoring dropped is skipped and named in the report notes and Methods, instead of failing the subject
- Raw QC's `--motion-correction` defaults to `tddr`, as the pipeline's, so its before and after panels show by default
- **Breaking**: `nirspipe-qc bids_dir output_dir participant|group` replaces `prep-raw`, `hyper-raw`, `cohort` and `cohort-hyper`; dyad pages sit beside the subjects in one tree
- **Breaking**: the dyad raw page reads each member's raw QC record (lines, cardiac band, DPF, verdict, motion correction) instead of screening again; it takes no screening flags
- Raw QC writes a run record, `sub-<label>/logs/*_desc-raw.toml`, that `nirspipe-qc --config` reruns
- **Breaking**: `nirspipe-provenance OUTPUT_DIR` replaces `nirspipe-qc provenance`
- **Breaking**: `--short-channel` defaults to `auto`, pca where the montage has short channels and none otherwise; it used to default to none
- **Breaking**: the quality record's `ch_dist_*` separations are in mm, as the flags and band stamps are; rerun QC for older records
- **Breaking**: raw QC writes one report per recording, named like its run report, so the subject index and cohort page link it; `nirspipe-rate raw` opens every matching one, `--run-label` added

### Removed
- **Breaking**: fallbacks for trees older versions wrote: flat QC records, sidecars without the filter or GLM conditions, dyad tables without member columns, the old rate report path

### Fixed
- The dyad motion panel's carpet titles covered the GVTD row above; each carpet is now named in the left margin
- A run whose short channels were all rejected listed them as outside every separation range and never noted that short-channel regression did not run
- The GUI's Edit events drift hint failed when the only repeated condition was a doubled marker
- A joined `--combine-runs` series without a regression did not record its separation bands, so the subject index split its FC by the defaults
- The raw QC cohort page listed every subject as unfinished, since raw QC run records never got an outcome; they now do
- A cohort of a few runs drew an empty condition matrix: the chart was too short for its colour bar
- Trials shorter than one 10 s screening window logged a windowed-SCI warning each
- A montage without short channels showed Separation n/a for every channel; channels in the long range now read long
- The raw signal overview left rows of blank space above and below a tall montage; its y range now fits the traces
- The raw signal overview draws its two wavelengths in a brighter orange and blue
- The raw signal overview spaces its rows wider and puts both wavelengths of a pair on adjacent rows, whatever order the file lists them in
- The raw signal overview reads top-down and puts short channels in their own labelled block below the long ones, in lighter shades, instead of dashed lines
- The mALFF and GLM cortex figures were saved at 100 dpi and the provenance graph at 220; all three are now 300 dpi, rendered to match
- With no PSP line, PSP takes the same red-to-green scales as SCI and CV (strip, row means, channel and condition grids, head slider), from 0 to 0.1, higher greener
- Optodes that could not be placed on the template head were still drawn beside a 3-D brain and projected onto the cortex (mALFF, GLM betas); such runs now show the flat optode map only
- The flat views (optode map, topographies, head slider, seed and flat value maps) of a montage no reading placed pushed optodes past a head outline; they now draw it as laid out, with no head

## [0.76.0] - 2026-10-25

### Added
- `--chromophores` chooses what the GLM tables, block average, FC and ALFF carry; `hbt`, HbO plus HbR, is off by default
- The cohort page counts, per task, how many subjects kept each source-detector pair
- `space-<template>_channels.tsv` names each channel's nearest cortical region: `aparc` on every template, `brodmann` on fsaverage; the dyad group table carries both

### Changed
- The run report's GLM activation and ALFF brains draw every chromophore of the run, HbR included, and its connectivity figures follow `--chromophores`
- `--wtc-chroma` and pairnull's `--chroma` take a list of `hbo`, `hbr`, `hbt`, default `hbo hbr`, replacing `both`; groupnull's `--chroma` takes `hbt`
- Methods define HbT where a run uses it, name the chromophores of ALFF and FC, and say FC is reported as r and Fisher z
- **Breaking**: GLM and contrast tables drop the uncorrected `Significant`; `--p-correction` (default none) adds `p_<method>` beside `p_value`, `family` counting each test's long good channels

### Fixed
- An event annotated twice, or two conditions starting on one sample, failed the subject's glm post in the block average; the evoked channel map shared it
- The hyper warning about a band past the recorded filter named each member `sub-sub-<label>`
- CNR on a recording that annotates an event twice failed with a warning and was left empty; it is computed, the repeat counted once
- A WTC band edge on a wavelet grid frequency could drop that frequency's row, depending on round-off; the band is closed as documented

## [0.75.0] - 2026-10-25

### Added
- `--dpf age` takes one DPF per wavelength from each participant's age in participants.tsv, by the Scholkmann and Wolf (2013) equation, and warns outside its recommended range
- An optional `role` column in the pairs table: pairnull's position pool matches stand-ins by role, and the dyad tables record each member's role
- `--wtc-phase-null` runs on groups of three or more, each pairing's later member scrambled once per draw and shared across its pairings
- pairnull draws groups of three or more: each draw takes one stand-in per later member from one other group, shared across its pairings, which `--wtc-pair-max` caps together
- groupnull reads a group of three or more as the mean over its pairings, and each role pair on its own where the pairs table has roles
- `--template` places and draws each participant on fsaverage or an MNE infant template; under `--dpf age` the template nearest their age is the default
- glm mode writes `desc-blockavg_timeseries.tsv`: each condition's trials averaged per channel over the epoch window, with standard error and trial count

### Changed
- Channel screening counts a window on SCI alone; PSP is shown without a line and screens only when `--psp-threshold` is given
- Reports, tables and the channel TSV's reason call the screened windows "passing windows" in place of "coupled windows"
- `--phase-null-shared` rotates the second member's channels together in the WTC phase null as well as the ISC one, replacing `--isc-phase-null-shared`
- glm mode refuses a zero-duration event, from `--stim-dur 0` or an events table, since its beta would scale with the sampling rate

### Fixed
- Channel screening that could not be measured (a cardiac band near Nyquist, a recording under two windows) kept every channel; preprocessing now stops, raw QC reports say why
- PSP was on a different scale from the 0.1 line it was compared with, so that line passed uncoupled windows; every reported PSP value changes
- pairnull drew a group of three or more for its first two members only, leaving the other pairings without a null
- groupnull's occasion and cohort levels ranked a group of three or more's mean over every pairing against one pairing's draws
- In a group of three or more, `--isc-phase-null` gave different members identical phase draws when seeded, and one member different surrogates across its pairings when not
- Every AR `--noise-model` failed on a recording with a constant channel; such channels are now fitted by OLS and named in the record
- Edge frames without light in most channels are held at the nearest lit frame and marked `BAD_edge`; one NaN first frame had cost whole channels
- Raw QC reports and the dyad raw pages hold those edge frames as prep does, so they no longer lose those channels either
- `--noise-model` help named `auto` as every mode's default; rest and denoise default to `ols`
- The raw signal figure drew both wavelengths in one colour; each pair's shorter wavelength is now orange, its longer blue, short channels dashed
- The stage carpet quoted a stage's SD against itself, and the head slider read "between blocks" on a run without events
- A run without `--mode glm` showed a developer hint in its GLM section; it now says no GLM was run
- The channel map stopped at 20 s whatever the epoch window; a run without a low-pass now gets epoch figures low-passed for display
- Methods never said how the optodes were placed; the PSD note credited short and aux regressors to runs without them

## [0.74.0] - 2026-10-23

### Added
- A dyad table's sidecar names the null level files that gated its phase columns, with the version and time of the run that drew each

### Changed
- The run report shows every condition's GLM activation figure, one under another, in place of a menu
- The rest report draws mALFF of HbO on the cortex in four views, in place of the flat ALFF and fALFF map
- groupnull's occasion and cohort levels compare each re-paired draw with the real pairing over the channels both kept, so differently rejected channels no longer bias them
- pairnull's region cells follow the same rule, and with them groupnull's per-cell coherence p for regions

### Fixed
- A rerun no longer refuses an output directory named through a Windows junction when an earlier run named its real path
- The optode flat map and the head maps draw the run's placed optode positions, so a file recorded in another frame lands on the head as in the 3-D views
- The subject index's Channel rejections columns name each run by task plus whichever of session, acq and run tell the runs apart
- The re-paired null's `percentile` counted a stand-in lacking a pairing as a draw the real value lost to; groupnull's per-cell coherence p inherited it
- A real partner's `BAD_` span lying before a condition excluded most of that condition from every re-paired draw, thinning or emptying its null
- A stand-in sharing no usable channel pairing with the fixed member stopped the dyad's whole re-paired null; it is now a blank draw
- So did a stand-in with no usable long channel at all; a dyad member with none is now reported as a refusal, not an unexpected error
- A dyad whose members kept no channel pairing in common is refused saying so, where it failed on an empty array
- Under `--wtc-window-s`, round-off could drop a condition's last window or move the column on a window edge into the next window
- On a loaded Windows machine, runs writing into one tree could still fail reading or replacing `dataset_description.json` while another run held it
- A fresh install lacked statsmodels, so `--noise-model ar_irls` and `nirspipe-hyper-groupnull` failed where they first needed it
- A SNIRF whose clock does not start at 0 had every event and aux sample placed that late, and events past the shifted end dropped
- Runs of one subject on different sessions or tasks shared one run record and log, which parallel runs left unreadable; each selection now keeps its own

## [0.73.0] - 2026-10-21

### Added
- `--combine-runs` now joins a task's runs after postprocessing, each z-scored over its kept frames, and computes FC on the joined series; ALFF and GLM tables stay per run
- The subject index draws each joined series' FC, channel and ROI, under Runs combined
- Runs combined also tables each joined run's length, kept time and rejected pairs, and draws the joined carpet with run starts and `BAD_` spans
- `--noise-model worsley`: auto's AR order with its coefficients bias-corrected for the fit; channels where the correction fails are fitted as auto and listed in the sidecar
- `nirspipe-hyper --isc-phase-null-shared` shifts all of a member's channels by one shared set of random phases in each ISC phase-null draw; off by default
- `nirspipe-hyper-groupnull` warns, in its log and on the page, when `--isc-test magnitude` meets the phase null on unwhitened correlations
- `nirspipe-hyper` and `nirspipe-hyper-pairnull` take `--device cuda`: wavelet transforms run on an NVIDIA card, checked against the CPU on each group's first pairing

### Changed
- Every derivative, record, report and figure name carries the input's `acq`; `--bad-channels`, `--bad-spans` and `--keep-spans` tables take an optional `acq` column
- `--combine-runs` joins a channel only from the runs where it is good (NaN in the others) and takes FC pairwise; ROIs are averaged per run
- `nirspipe-hyper-pairnull` redraws each re-paired dyad's pages from its run record, its phase null read from disk, so condition arrows read the re-paired null; `--no-report` skips it
- The run note on optode positions follows the run's placement on fsaverage, as the brain views do; it names why a run could not be placed

### Fixed
- Two recordings told apart only by `acq` no longer write one set of outputs, the second overwriting the first
- ISC sidecars record the condition windows: each condition's matrix its own, the long tables and draws all of them
- A reviewer's kept channels now rescue a run whose every channel failed screening, instead of the run stopping

## [0.72.0] - 2026-10-20

### Added
- `nirspipe-qc prep-raw` takes `--nprocs` and reports several subjects at once
- `nirspipe-hyper`, `nirspipe-hyper-pairnull` and `nirspipe-qc hyper-raw` take `--nprocs` and run several groups at once, each in its own process
- Each dyad command writes a log per group under `group-<id>/logs/`

### Changed
- **Breaking**: `--n-jobs` becomes `--nprocs`, the cores a command may use in all; subjects share them and run records store `nprocs`
- The dyad raw report's corrected motion figure no longer repeats the IMU rows, which motion correction leaves unchanged
- The dyad raw report's head slider gives each member a row: the window's coupling verdict, then its SCI and PSP, each scaled on the run's own lines
- The windowed SCI and PSP heatmaps draw one row per source-detector pair and name every row; CV keeps one per wavelength
- The raw report follows the subject report's section order, with its brain views under Signal Quality
- The raw report's clickable optode map is gone, leaving the flat map beside the brain views
- The subject report's Raw Signal opens with the raw report's trace overview and event table; a click picks a channel or outlines an event
- SNR is labelled μ/σ wherever it is shown, a linear ratio rather than dB
- Signal by position moves from the raw report to the subject report's epoch section, drawn on the processed epochs
- The raw and subject reports carry the dyad page's head slider: coupling, SCI and PSP on the head, window by window
- **Breaking**: `--epoch-single-trial` is gone; the epoch section is drawn when no condition repeats, and the run notes name conditions occurring once
- Epoch figures list conditions in order of first appearance, as the event timeline does
- A grand-mean panel with fewer than three trials gets its own y scale instead of being clipped to the shared one
- The raw report's grand mean and trial images are low-passed for display below `--cardiac-l-freq`
- Both subject reports set Signal by position beside the single-channel trial image; clicking a channel picks its trial image
- Signal by position shows one condition at a time, chosen from a menu, opening on the first with three or more trials
- The brain views drop the blank bands above and below the brains, so the picture is shorter
- A channel's trial image shows all trials pooled, then the condition chosen on Signal by position, in one figure on one colour bar
- Trial images draw the heatmap and its average 2:1 under one title, without the overlapping subtitle
- Signal by position names channels on hover only, as the cell labels overlapped their neighbours
- The raw report's channel detail drops its epoch average, which the trial images now carry
- Trial images draw the HbR average beside the HbO one
- The raw report's trial images split by event name, pooled first, as the subject report's do; condition pages keep their window's trials
- Without `--wtc-seed` or `--seed`, a dyad command that draws at random picks a seed, prints it and records it, so the draws can be repeated

### Removed
- **Breaking**: `--work-dir`, which cached nothing; every stage is already written to the output directory

### Fixed
- A rerun into an existing output tree no longer fails when its input is now reached through a junction or symlink; new dataset links stay relative as the paths were given
- The cohort report's strip jitter no longer moves between runs of the same cohort
- With `--n-jobs` above 1, a subject's GLM report could show another subject's activation, regressor or beta figure
- With `--n-jobs` above 1, subjects could fail at random on Windows, or a report could lose a panel
- Runs writing into one output tree at once could drop each other's `DatasetLinks`, and dyad QC reading that tree then failed
- On Windows, a run could fail to name its sources while another run sharing its output tree rewrote `dataset_description.json`
- A subject report whose epoch preview is skipped says why, instead of claiming the run has no events

## [0.71.0] - 2026-10-19

### Added
- Task GLM runs write a whitening diagnostic, and the report shows the residual spectrum against white noise and each channel's lag-1 autocorrelation
- Prep writes each run's channel positions on fsaverage, `space-fsaverage_channels.tsv`, with how the optodes were placed in its sidecar
- `nirspipe-hyper` writes each group's member channel positions on fsaverage, one table with a `subject_id` column
- Each subject's run record ends with an `[outcome]` section: status, finish time, duration and the error's first line
- The cohort page lists the subjects whose last `nirspipe` run failed or was stopped

### Changed
- `--motion-correction spline` detects motion at 20 SDs, 0.5 OD and a 2 s margin by default, replacing 50 SDs, 5 OD and 1 s
- Optodes are placed on fsaverage by the file's landmarks, else by the reading that fits the scalp; brain views follow the run's recorded placement

### Removed
- **Breaking**: `nirspipe-log`, the SQLite run database and the `logs/json/` event stream; run records and logs carry what they held

### Fixed
- Cropped and marker-edited recordings keep their source's probe landmarks
- `nirspipe-snirf2bids --optode-frame mri` is documented as a template brain, which is how it is read, not the subject's own MRI

## [0.70.0] - 2026-10-17

### Added
- `nirspipe-rate raw` marks bad stretches on a run's raw signal; `nirspipe-rate export` writes them as a `--bad-spans` table

### Changed
- `nirspipe-gui` is a new GUI: forms generated from each command, Data Preparation edits raw reports, rating built in; Dash and its four dependencies are gone
- The raw report's Events column is as tall as the raw signal beside it and scrolls when longer
- Reports load Plotly.js from the CDN and fall back to a copy at the output tree's root when offline
- `nirspipe-recon` is now `nirspipe-snirf2bids`, and the GUI's Recon page is SNIRF to BIDS

### Fixed
- `desc-glm` and `desc-contrast` tables leave every statistic n/a for a short channel fitted against itself, as ALFF and FC already did

## [0.69.0] - 2026-10-16

### Added
- GUI Batch Prep page takes `--skip-bids-validation`
- GUI Hyper Analysis page takes `--band-fmin/--band-fmax`; its WTC and ISC band fields override them
- `nirspipe-qc hyper-raw` names each member's pairs that `channels.tsv` marked bad
- Both dyad reports list each member's IMU summaries; `hyper-raw`'s `desc-subject_qc.tsv` carries them with their units
- `nirspipe-hyper --config` reruns a dyad from its run record
- The cohort step writes `desc-subjects_nirsmap.tsv`, every run's per-channel quality values in one long table
- `nirspipe-qc prep-raw --run-label` reports the chosen runs, each on a page of its own
- The raw report maps each channel's signal and coupling on the head; clicking a channel opens it below

### Changed
- `desc-sci` sidecars under `--gvtd-censor` record the separation bands that chose the censored channel set
- GLM and residual sidecars list the events table and aux table they read in `Sources`; a table outside every linked dataset goes to `external_sources`
- `desc-contrast` sidecars record each contrast's weights as `contrast_weights`
- `nirspipe-hyper` correlation sidecars record `desc`, `bads_scope` and the analysis window; coherence and correlation sidecars record the separation bands
- The group-test page's region matrices print every cell's lift, bold with a star where significant instead of a dot, in square cells
- The per-trial panel scores 5 to 10 s trials on SCI and GVTD, rather than leaving them blank
- GUI cardiac band fields suggest 2.0 Hz as the adult upper bound, matching the docs
- GUI command previews put each flag beside its values and quote paths for the chosen shell
- GUI Data Preparation creates an output directory that does not exist yet
- GUI report previews are a window tall and open in a new tab
- GUI commands give `--participant-label`, `--session-label` and `--task-label` once, followed by all their values
- GUI Data Preparation keeps the Signal Topo legend below the head and shows whole marker descriptions
- Report metric hover text is one or two short sentences: what the number is and which end is better
- Shorter captions on the dyad, cohort, group-test and raw QC pages, and shorter GUI hints
- Shorter captions on the run report, and shorter caveats on every report
- Shorter `--help` texts on 34 options
- `nirspipe-rate export` refuses a channel decision saved from a condition page, naming the file and key
- Python API: hyperscanning names are imported from the module that defines them; `nirspipe.io.groups` reads the pairs CSV, `nirspipe.cli.run_record` writes run records
- `nirspipe-rate` viewers run in one local web app; a report written before this version opens read-only there until it is regenerated

### Removed
- GUI Data Preparation's 3D fsaverage layout card; the 2D optode layout stays
- Raw viewer condition pages lose their channel decision column; decisions apply to the whole run and are made on the run page

### Fixed
- GUI number fields left values off their step grid out of the command, such as a 12 mm separation, 5 phase-null draws or a 0.01 Hz drift cutoff
- `nirspipe-qc cohort`, `cohort-hyper` and the `group` level exited 0 over a tree without quality records, `cohort` writing an empty page
- GUI parameter summaries showed `?` for a separation left at its default
- `nirspipe-prep crop`, `align` and `edit-markers apply` failed file by file, rather than refusing, when the input was the folder they write
- Run logs carried numpy divide and empty-slice warnings for a channel with no windowed SCI
- GUI Batch Prep could write an older directory back over one typed straight after another
- The dyad Methods' ISC sentence claimed every pairing and both chromophores under `--no-channel-cross` or `--wtc-chroma hbo`

## [0.68.0] - 2026-10-12

### Added
- GUI Analysis page downloads its pipeline graph as a PNG
- GUI Recon page takes `--optode-frame`
- GUI Raw QC page runs `nirspipe-rate export` from a button and shows the `--bad-channels` table it wrote
- GUI Analysis page fills its form from a run record, read as `--config` reads it, and names the settings it cannot hold
- GUI Analysis page offers `--gvtd-censor short`
- GUI Analysis page takes `--bad-channels`, `--keep-spans` with `--keep-spans-from`, `--bad-spans`, `--events-path` and `--contrast-file`
- `nirspipe-rate export` writes the viewers' channel decisions as a `--bad-channels` table, and lists every FINAL verdict
- A `--bad-channels` table takes a `keep_channels` column: pairs screening rejected that a reviewer kept; the Methods name them
- GUI Raw QC page runs `nirspipe-qc prep-raw` and `hyper-raw` over several subjects or dyads and names the `nirspipe-rate` command that opens each report
- `nirspipe-hyper --band-fmin/--band-fmax` set the coherence and correlation bands together; the per-metric band flags override them
- Documentation published to GitHub Pages, rebuilt from `docs/` on every push to `main`; each command's flag list is generated from its parser
- A tutorial page runs three public dyads from download to the cohort null tests, one command per step

### Changed
- GUI Recon page streams each file's `nirspipe-recon` output as it runs and can stop the batch
- GUI Raw QC, Hyper Analysis and Cohort Reports pages stream the command's output and can stop it, as Analysis does
- GUI Hyper Analysis sends several group-null `--null` and `--chroma` values; none ticked reads every one on disk
- GUI Data Preparation drops its QC Metrics tab; Raw QC and `nirspipe-rate raw` show those panels
- GUI Data Preparation screens channels with the PSP threshold, coupled-window share, scope and `channels.tsv` status, as `prep-raw` does
- GUI Data Preparation saves markers and crops by running `nirspipe-prep`
- GUI Recon runs `nirspipe-recon` once per file instead of converting in its own process
- `nirspipe-rate hyper` reads and writes each member's dyad decisions under `_hyper` only, apart from the run pages' own
- GUI Data Preparation no longer records channel decisions or writes the raw report; `nirspipe-rate raw` and the Raw QC page do
- The README is a short overview that links the documentation
- **Breaking**: `nirspipe-rate rate` keeps one JSON rating file per report page, as `raw` does; `logs/group_ratings.jsonl` lines name the page `stem`
- `nirspipe-hyper-groupnull` takes several `--null` and `--chroma` values in one run, each defaulting to every null and chromophore whose draws are on disk; a chromophore the draws lack is refused

### Removed
- GUI Hyper Preparation page; alignment, the dyad raw report and channel decisions are on Batch Prep, Raw QC and `nirspipe-rate hyper`

### Fixed
- GUI Data Preparation lists runs inside `ses-` folders
- GUI Data Preparation loads a run into an output directory that holds no `dataset_description.json`
- GUI Analysis page's schematic drew motion correction after Beer-Lambert; it follows the pipeline's order and each mode's stages
- `nirspipe-rate raw` serves the subject index's figures and tables when the index is opened from the raw report
- `nirspipe-rate hyper` refuses channel decisions for anyone outside the dyad
- The run report's stage comparison no longer filters an already band-passed stage a second time
- The cohort report's over-time grid aligns runs sampled at different rates onto one time axis, so its band and run lines draw again
- `--keep-spans` help said unselected time was left out of the GLM and confound fits only; it is left out of every estimate
- GUI pages report a stopped command as stopped on Windows, not as failed (exit 1)

## [0.67.0] - 2026-10-11

### Added
- Per-channel tables print each wavelength's share of samples held at the detector's ceiling; run summaries count saturated channels
- `--motion-correction spline_sg`, spline with Savitzky-Golay filtering (Jahani 2018) as its authors implemented it; also low-passes every channel near 0.1 Hz
- `--motion-correction wavelet_tddr`, the wavelet filtering followed by TDDR; also on the Analysis page
- `--motion-correction spline_wavelet`, spline at its paper's detection thresholds then wavelet filtering (Di Lorenzo 2019); also on the Analysis page
- `--motion-std-thresh` and `--wavelet-iqr` set the spline detection threshold and the wavelet fence, defaulting per method; methods that do not read them refuse them
- Spline motion correction warns in the log and the report when it marks over 10% of the samples; its threshold tightens as the sampling rate rises
- Each dyad writes its ROI ISC as a long table, `seg-<map>_agg-roi_stat-isc_relmat.tsv`, one row per pairing, chromophore and condition

### Changed
- Motion-corrected sidecars record the spline threshold and wavelet fence used; the Methods paragraph quotes them and refuses older trees
- The dyad analysis report folds its channel value table and links the full-precision WTC and ISC tables above it
- Both dyad reports print member metrics with each channel set over its members' columns; the dyad raw report counts pairs and motion per set
- The subject cohort report charts saturated share and saturated channels in their own panels after mean amplitude
- `--config` reads a run record, or a file of its shape with `[prep]` and `[post]` keyed by flag name, and reruns that run; flat files are refused
- Run records key settings by flag name in the flag's units, name tables by absolute path, and write a switched-off mode default as `"none"`
- `nirspipe-hyper-merge` merges long tables only; ISC matrices stay in each dyad's folder
- `nirspipe-qc hyper-raw` into a tree of its own stamps it `nirspipe-qc hyper-raw`; the nirspipe-hyper tree keeps its stamp. The GUI field follows
- Each WTC table kind holds the whole run and the conditions in one file, `condition` n/a on whole-run rows, as ISC does; readers refuse older trees

### Fixed
- Under `--short-channel pca`, short channels are blank in FC and seed FC and left out of ALFF; rest-mode mALFF/zALFF values change
- A `--config` cutoff set to `"none"` no longer stops the run with a KeyError
- `nirspipe-qc prep-raw` writes `.bidsignore` to its tree, as the other trees do
- Phase-scrambled and re-paired null percentiles are no longer blank for conditions with numeric labels
- `nirspipe-hyper` warns when members' markers disagree beyond two samples after alignment; dyad condition pages say why a member's column is blank

### Removed
- The per-subject `sub-<label>_script.py` transcript; the run record and package version reproduce a run, and `--dry-run` writes the record only

## [0.66.0] - 2026-10-11

### Added
- `nirspipe-prep keep-spans` writes a `--keep-spans` table from named conditions, one row per occurrence; also on the Batch Prep page
- `--keep-spans-from MARKER` measures `--keep-spans` onsets from each recording's first marker of that name
- `--motion-correction tpca`, targeted PCA (Yücel 2014), also on the Analysis page; the motion-corrected sidecar records each pass's removed components
- `--motion-correction spline` (Scholkmann 2010) with per-pair motion detection, also on the Analysis page; refuses recordings below 10/3 Hz

### Changed
- Wavelet motion correction follows the published procedure exactly: one fence per phase, coarsest level left alone, padding clipped
- Channels a run's BIDS `channels.tsv` marks `bad` are rejected with `--bad-channels`; `--ignore channel-status` turns this off
- `nirspipe-qc prep-raw` and `hyper-raw` reject the channels `channels.tsv` marks `bad`, as the pipeline does; `--ignore channel-status` turns this off
- Cohort pages list the median and range table under Quality Metrics, not Summary; the subject cohort's Overview drops the repeated small-cohort caveat
- Every report's navigation reads as the headings it jumps to, and no page links to a section it lacks
- The optode flat map beside the brain views uses MNE's topomap projection and head outline, as the head maps do
- Dyad ISC runs only the chromophores `--wtc-chroma` names, as the coherence does; an HbO-only run no longer writes HbR correlations
- The provenance graph wraps a tall column into a grid and shows at a fixed text size without the height cap
- Matrix figures draw only the sites that have a value, with no grey ground; the tables on disk keep every site
- The dyad raw report drops its Provenance section; the dyad analysis report keeps the graph
- The denoised carpet marks each ROI with a coloured strip named in the legend, replacing the ROI tick labels
- The motion carpet groups its long rows by ROI under the same coloured strip when an ROI map is given
- `hyper-raw --roi-mapping` groups each member's long motion carpet rows by ROI under the coloured strip
- `prep-raw --roi-mapping` groups the raw viewer's long motion carpet rows by ROI under the coloured strip
- Navigation labels abbreviate long headings, the full heading on hover; the raw viewer's bar names the subject, not the file
- Each stage's sidecar records only the settings that stage used and the filter state its file carries; the run record keeps the full configuration

### Fixed
- The dyad analysis page's chromophore note no longer describes stacked maps and side-by-side matrices when only one chromophore was run
- Run records leave out the span tables, `channels.tsv` bad channels and a `--bad-channels` table, which they misstated; the `desc-sci` sidecar holds them
- The denoised carpet's long and short GVTD row titles no longer overlap; each row is named inside its panel, as on the motion carpet
- Derivative `_nirs.json` sidecars carry the BIDS REQUIRED fields (`TaskName`, `SamplingFrequency`, channel and optode counts); post and crop refuse stages written without them
- The exported run script regresses on the run's separation bands, writes `dataset_description.json`, and records the same sidecar parameters as the run

## [0.65.0] - 2026-10-10

### Added
- `--bad-spans` takes a table of corrupted stretches per recording, marked `BAD_manual`: left out of screening, filled and excluded like GVTD spans
- `--export-mean-filled` writes `desc-meanfilled`, the final residual with `BAD_` frames at each channel's kept mean; also on the analysis page
- The subject report gains an Excluded time section on runs with `BAD_` spans; condition pages show their kept seconds

### Changed
- Dyad coherence and correlation leave out the union of both members' `BAD_` spans; every dyad table records each pairing's kept seconds
- Dyad prewhitening fits each member's AR coefficients on its time outside `BAD_` spans
- Both dyad nulls leave out the real pairing's excluded time; the re-paired null also leaves out its stand-in's corrupted spans
- Channel and optical-density spectra in the subject report use only frames outside `BAD_` spans
- The dyad raw report's per-channel spectrum uses only frames outside each member's `BAD_` spans
- The bad-segment zoom draws corrupted spans only, not `BAD_unselected`
- The motion panel and design matrix draw `BAD_unselected` grey and every other `BAD_` span red, with a legend by kind
- Methods describe unselected time, GVTD censoring, the fill, and what each regression, FC, ALFF and dyad measure leaves out
- desc-sci records the GVTD censoring settings; FC and ALFF sidecars list their `excluded_spans`
- Dyad WTC maps wash out excluded columns and draw no arrows there; scope headers lead with the kept seconds
- The subject report's processing steps are flat headings and no longer fold
- The subject report's results sit under one Quality Metrics heading: run summary, denoising stage, channel, condition, per-channel table, trial
- The Channel Quality Summary prints its figure's file path like every other figure
- The Haemoglobin group label says Long shows before → after denoising
- A metric no channel set owns prints in the All column, leaving Long and Short blank
- The subject report's metric, channel and condition notes are shorter
- The subject report's metrics table, split or single-column, prints each share with its count, `12.9% (1098)`, dropping the count rows
- QC report footers open with an "Appendix: run record" label above Errors
- The provenance table merges outputs with the same step description and settings into one row
- Software versions print as a compact three-column table
- The subject report's per-channel table has one row per channel pair, SNR, CV and Spike printing both wavelengths; the TSV stays per wavelength
- Seed and ALFF flat maps put HbO and HbR in rows; seeds run across, four to a row
- The PSD figures draw only the first stage and the last, on long channels; a short channel's own page leaves out its residual
- A channel's own PSD page overlays its stages, HbO and HbR side by side, the last stage dark
- The denoising-stage panels draw HbO only and drop cardiac power: six panels in two rows
- Quality Metrics lists its two tables first, then its figures; the per-channel table says which stage its scores come from
- The per-channel table's HbO–HbR corr prints before → after denoising where the record has the residual's value
- The subject report's quality figures sit under their own Quality Figures heading; its nav and the raw viewer's name them By channel, By condition, By trial
- The raw viewer's steps are flat headings and no longer fold; its results read Quality Metrics (tables) then Quality Figures
- The raw viewer's set tables print each share with its count, and its per-channel table shows both wavelengths for SNR, CV and Spike
- The dyad raw report's steps no longer fold; its channel table comes before the quality grid and shows both wavelengths for SNR, CV and Spike
- The dyad analysis report's coherence and synchrony panels no longer fold
- The cohort pages no longer fold or scroll inside a box; a long table grows the page, and each heading's hint is a sentence under it
- The dyad raw report shows per-channel signals as section 2 and its channel table as a row per member, a pair's members one above the other
- Dyad motion panels draw the members in the usable-time panel's blue and orange; dyad head maps draw a channel with no value white
- The raw viewer and its condition pages print their metrics as one table, a metric per row and a channel set per column; with --by-condition the run page draws every condition's grid
- The raw viewer's notes are shorter; its layout and signal hints are sentences under their headings
- The dyad raw report's notes are shorter, and its channel-table note describes the row-per-member layout
- The dyad analysis report's notes are shorter
- The subject report's processing-step and analysis notes are shorter
- The subject and dyad index pages' notes are shorter
- The cohort pages' notes are shorter
- The raw viewer's condition pages print both wavelengths for SNR, CV and Spike, as the run page does
- The cohort pages' big tables keep their header under the top bar while scrolling down
- Per-channel Status prints BAD alone for the coupled-window rule, which each table's note states once; BAD (manual) for a hand rejection
- A one-stage HbO–HbR panel draws its matrix at a two-stage column's width, centred, instead of page-wide
- Dyad head maps draw a channel with no value in the subject maps' light grey
- The raw viewer no longer shows a Provenance section
- The raw viewer draws the subject report's 3D and optode-map quality figure in place of its 3D channel plot, condition pages their own
- The dyad raw report's channel table sets the two members side by side, long and short channels in separate tables, without whole-run SCI
- The dyad raw report's simultaneous-movement numbers become a table under the motion figures instead of their own section
- The dyad channel table names wavelengths beside each header label, so each header row is one line
- The dyad raw report reads like the subject report: unlettered Recording steps, then Quality Metrics tables, then the channel grid
- The dyad analysis report drops step letters, ends with Quality Metrics and Quality Figures, and moves clocks and ROI grouping to the appendix
- The subject cohort summary names metrics by label in their report format; the dyad cohort prints its small-cohort caveat once
- The raw viewer drops its interactive 2D optode map and step letters, keeping the brain figure under a Recording heading
- The subject report's preprocessing steps are no longer lettered

### Fixed
- The subject report's motion panel, bad-segment zoom and design-matrix shading show the spans `--keep-spans` and `--bad-spans` add
- Accelerometer channels spelled `acc-x` are recognised, so the motion panels and IMU summaries include them
- A dyad member's SCI and PSP lines no longer vanish from the usable-time panel when a channel saturates; that channel leaves only its own windows' mean
- The mean PSD of the residual no longer sinks about 20 dB from retained short channels, whose residual is zero
- A condition page of a montage without short channels prints its metrics in one column, as the run page does
- A condition wholly inside `BAD_` spans keeps its denoising-stage panels, without the spectral ones, and shows n/a failing
- The channel epoch panel and the evoked channel map leave out trials under a `BAD_` span
- Dyad coherence is NaN where a member has no power, as inside a long filled span, instead of inf or above 1
- The re-paired ISC null's `n_valid_frac` is the share of each window it correlated, no longer always 1.0
- The raw viewer's condition pages state their kept seconds when the input has `BAD_` spans, and show n/a for an unassessed condition

## [0.64.0] - 2026-10-09

### Added
- `--keep-spans` takes a table of the stretches of each recording to use; the rest is marked `BAD_unselected`
- Each derivative sidecar lists its `excluded_spans`, with kind and source
- `--censor-fill linear|spline|lomb` fills corrupted `BAD_` spans before the bandpass; `desc-preproc` keeps the measured data
- The analysis page offers the fill method beside GVTD censoring
- `--min-time` skips FC and ALFF for a run with less kept time; default 0, also on the analysis page

### Changed
- **Breaking**: the `--bad-channels` table's `participant_id` is optional (blank means every subject), and any unknown column stops the run
- Channel screening leaves out every window touching a `BAD_` span, and stops a run left with fewer than two windows
- The GLM and confound regressions fit only frames outside `BAD_` spans, on the full-length design with gap-aware AR whitening; the residual stays full length
- A condition with no signal in kept frames is dropped and named; a fit with fewer than three kept frames per regressor is refused
- The saved design matrix gains a `censored` column; GLM and residual sidecars record kept frames, excluded shares and events left per condition
- Per-condition coupled-window shares and assessments leave out windows touching a `BAD_` span; each condition records its kept seconds
- FC uses only frames outside `BAD_` spans, and ALFF and fALFF a Lomb-Scargle spectrum of them; FC and ALFF sidecars record the kept seconds
- **Breaking**: the quality record gains an `excluded` section; derivative trees written before it must be re-run
- QC reports get a lighter look: new palette, hairline section rules, Methods tabs with a Copy button, and a one-row top bar without ratings
- The subject report's metrics print as one table, a metric per row and a channel set per column; dyad pages list members as columns
- Every QC report table drops its cell grid for hairline rows, numeric columns and their headers right-aligned
- QC reports print no exponents: drift in µM, band power in µM²/Hz, GVTD in mOD/sample, unit in the label

### Fixed
- `--screen-scope task` no longer counts a long `BAD_` span as a task block
- GVTD censoring is measured on the channels the screening kept, so a rejected channel no longer censors the run
- Seed and ALFF head maps keep their size however many rows they draw
- Provenance settings no longer name an HRF for a regression with no conditions
- `BAD_` rows in an `--events-path` table are no longer turned into conditions
- A channel with no finite sample gets NaN fALFF instead of 0
- A condition's CNR keeps a `BAD_` span starting before it, so the epochs it covers still drop
- A recording whose only annotations are `BAD_` spans no longer stops the subject report
- Per-condition SCI, PSP and CV averages no longer read `nan` when a channel has a window without a value; existing trees need a re-run
- Condition pages no longer print metric rows that have no value there, and the raw viewer heading carries the new package name

## [0.63.0] - 2026-10-09

### Changed
- **Breaking**: the package is renamed NIRSPipe: import and distribution `nirspipe`, commands `nirspipe` and `nirspipe-<x>`; old derivative trees must be re-run

### Fixed
- `nirspipe-prep edit-markers apply` and `crop` keep a session-labelled input's `ses-` folder when `--session-label` is not given

## [0.62.0] - 2026-10-09

### Added
- `fnirs-hyper --isc-phase-null` keeps every surrogate per condition in a `desc-draws` table, as the coherence's phase null does
- `fnirs-hyper-groupnull` tests the correlation too, wherever its draws exist; `--isc-test` picks signed two-sided (default) or magnitude
- `fnirs-hyper`, `fnirs-hyper-pairnull` and `fnirs-qc hyper-raw` take `--participant-label`, keeping the groups a listed subject belongs to
- A pairs CSV `occasion` column joins members whose session labels differ for one sitting, naming the group's outputs `ses-<occasion>`
- The dyad raw page's alignment timeline and a table show each block's onset difference between the members after alignment, flagged past two samples
- `fnirs-hyper-groupnull` writes `task-<task>_desc-groupnull_report.html`: each occasion against its null, and each region's lift marked by the paired test
- The `--bad-channels` table takes optional `session`, `task` and `run` columns; a row matching no recording stops the run

### Changed
- A failed dyad alignment now names `--no-align` and its assumption that the members started recording together
- **Breaking**: `fnirs-hyper --isc-whiten ORDER` is `--isc-whiten-s SECONDS`, one AR order for every channel of both members fitted on the whole record; `ar_order` columns are gone
- `fnirs-hyper-groupnull` takes `--task-label` like the other commands; `--task` still works
- **Breaking**: `fnirs-hyper-pairnull` and `fnirs-hyper-groupnull` take `--chroma` for `--wtc-chroma`, since both cover the correlation too
- `fnirs-hyper-pairnull` ranks each ISC cell's `percentile` as |r| among |draws|, as the phase null always did
- **Breaking**: `fnirs-hyper --wtc-channel-cross` is `--channel-cross`, since it crosses the ISC as well as the coherence
- **Breaking**: `fnirs-hyper-groupnull` corrects p only under `--p-correction` (default none), writing `p_<method>` beside the raw `p`; the `q` columns are gone

### Removed
- **Breaking**: `fnirs-qc hyper-raw` no longer measures coherence or its null; `--coh-fmin`, `--coh-fmax` and the cohort page's null panel are gone

### Fixed
- A pairs table may name subjects without the `sub-` prefix; `101` and `sub-101` now find the same recordings
- A run whose every short channel was rejected leaves its short sections out instead of logging tracebacks, and keeps its per-condition record
- `fnirs-qc hyper-raw` usable-time and head panels no longer come out empty when the members recorded at different sampling rates
- The `EDGE boundary` written where recordings are joined is no longer a condition, trigger, dyad onset row or event count
- The exported run script screens with the run's `--psp-threshold`, `--min-good-frac` and `--screen-scope`
- The exported run script applies `--gvtd-censor` as the run did
- `fnirs-hyper --wtc-cond-transform` drew each condition map on its cut's own clock instead of the aligned one
- A crossed dyad run never used an uncrossed phase-scrambled null, not even on its same-channel maps
- `fnirs-qc hyper-raw --tstart` left the record's alignment offsets and duration at their values before the cut
- `fnirs-qc hyper-raw --normalize` labelled z-scored traces in µmol/L
- The cohort dyad page drew a block a dyad lacks, and a missing share, as zero
- A pairs CSV with a `session` column merged one group's sessions into one group; each session is now its own, every dyad output named `ses-`
- `fnirs-qc hyper-raw --session-label` only renamed some outputs; it now picks each listed session to read, every output under `ses-`
- A dyad's provenance diagram drew every task of its group; it now holds its own run's files, and `fnirs-qc provenance` reaches session folders
- `--noise-model ar1` snapped each channel's AR coefficient down onto a 1/n_channels grid; it is now the channel's own estimate
- `fnirs-hyper` ROI maps drew arrows at `--wtc-arrow-min` beside a null; `--wtc-phase-null` now gives them their own level, averaged surrogate maps

## [0.61.0] - 2026-10-08

### Added
- Dyad pages show each member's run verdict and per-condition pass or fail by channel pair: one condition on its page, every condition on the run page

### Changed
- `fnirs-qc cohort-hyper` refuses a dyad record whose screening is not kept per pairing
- The dyad page's ROI homologous table prints the ISC over same-channel pairs, as its coherence
- The ISC follows `--wtc-channel-cross`: uncrossed, it keeps the same-channel pairs only, in its tables, ROI matrix and both nulls
- The dyad index reports WTC, past-null counts and ISC over same-channel and crossed pairings apart; its ISC is a Fisher z mean
- `fnirs-qc prep-raw` and `hyper-raw` require `--sci-threshold`, as `fnirs-pipe` does; the GUI's SCI threshold fields start empty
- Report builders and figure functions take the SCI line from their caller, with no default
- Dyad pages colour each member's SCI against the line that member was screened at, read from its sidecar
- The dyad decision table tints each member's cells by that member's own rejection
- `fnirs-rate raw` and `hyper` `--sci-threshold` now outline SCI (10 s) cells below it; off unless given
- The GUI's Data Preparation and Hyper Align callbacks log through the package logger instead of printing `[DEBUG]` lines

### Fixed
- The GUI alignment table coloured SCI against 0.8 whatever the page's SCI threshold
- The provenance diagram's screening arrow named only the SCI line, and its denoising arrow left out the short-channel regression
- The dyad raw page dropped its usable-time and head panels whenever the members' clock offsets differed; they are screened on the shared clock now
- The dyad raw page never drew its channel summary, and its channel table and record left the windowed SCI empty
- A screening window too short for a coherence bin was ranked as the 0th percentile; it now draws no point
- Groups of three ranked screening coherence by their first pairing only; every pairing now has its own percentile
- The dyad motion panel flagged the shared clock's start as a joint spike; GVTD and spikes are now measured before the cut
- Each dyad channel selector marked every member's rejections instead of its own member's
- A triad's pairing page listed every member in its quality and alignment tables
- Condition panels named a 10 s window for values measured on the run's `--window-length` grid
- The cohort's windowed grid drew a highlighted run with the next run's values once an earlier run lacked that channel set
- The dyad raw page's comparability table counted every pair, short ones included, as a long pair

### Removed
- `fnirs-hyper --sci-threshold`; members' own screening lines are read instead

## [0.60.0] - 2026-10-07

### Added
- `fnirs-qc prep-raw` reports HbO-HbR correlation: a channel-table column, a per-set table and a correlation panel, before and after `--motion-correction`, condition pages included

### Changed
- Per-channel tables print `SCI (10 s)` and `SCI (whole run)`, and `_desc-channel_qc.tsv` writes them as `sci_win` and `sci_whole`
- Per-channel grids, tables and the per-trial heatmap name each metric's window; the SCI/PSP panel names its `--window-length`
- Condition pages and the subject index no longer show the windowed SCI under the whole-run SCI's name
- Rejected pairs get channel detail and per-pair PSD pages, marked rejected, and rank in their own group on the HbO-HbR panel
- The brain views and the raw viewer's optode layout grade channels by the windowed SCI
- Dyad pages, the `fnirs-hyper` console summary and the GUI alignment table show the windowed SCI and name its window
- The dyad channel table writes `sci_win` and `sci_whole` instead of `sci`
- The subject index and cohort condition panels name each metric's window
- The raw viewer's HbO-HbR set means and channel table leave rejected pairs out, as the pipeline's do
- The raw viewer's channel picker marks short and rejected pairs
- Condition pages mark channels rejected by the run's verdict; their header counts channels failing on that condition's stretch alone
- The run page's per-condition grid shows In condition, pass or fail on each stretch, instead of Status
- The subject index Conditions table heads its channel column Channels passing
- Condition pages and the dyad's per-condition quality table call a condition's channel share Passing in condition, not Channel retention
- Per-trial quality leaves trial windows shorter than 10 s unscored, and a run note counts them

### Fixed
- `fnirs-qc prep-raw --by-condition` wrote no condition pages, its record holding a nested per-channel value that could not be read back
- The SCI/PSP panel ignored `--psp-threshold`, and the run page's per-condition grid ignored it and `--min-good-frac`
- The bad-segment zoom drew raw intensity as its After row instead of the motion-corrected optical density
- The raw viewer's optode layout ignored `--sci-threshold`
- `fnirs-hyper` graded dyad channels by the whole-run SCI, the record's windowed scores never reaching its pages
- The raw viewer's grand mean averaged rejected channels into its long-channel trace
- The raw viewer's mean spectrum grouped channels by the default separation bands, not the run's
- Channel tables titled their out-of-range block with the default separation gap, not the run's
- GLM design figures and activation maps drew aux regressors as conditions
- A short channel's motion figure shaded the long channels' corrected spans
- The motion figure labelled its GVTD and derivative rows 0.01-0.5 Hz instead of the band it used
- A channel rejected by hand or for non-finite samples passed on condition pages
- Line figures of long recordings drew the cardiac pulse folded into a slow wave; they now keep each bin's lowest and highest sample
- The motion carpet dropped bursts shorter than one column; each column now shows its bin's largest |z|, in at most 2000 columns
- The aux table rounded timestamps past 1000 s to 0.01 s, distorting the subject report's accelerometer jerk
- The raw viewer's channel map folded the pulse the same way on runs with no events, where it draws the continuous signal

### Removed
- `condition_window_fractions`, whose shares `by_condition` already records
- `design_matrix_heatmap`'s `conditions` and `title` parameters, which it never read

## [0.59.0] - 2026-10-06

### Added
- `desc-subjects_qc.json` describes every column of the cohort table

### Changed
- Quality records keep their arrays in TSV tables beside the JSON: channel-by-window matrices headed by channel name, per-window series, per-channel values; older records are refused
- `fnirs-qc prep-raw` writes its channel table as `_desc-rawchannel_qc.tsv`, so a tree both commands wrote keeps both verdicts
- Every TSV writes a missing or non-applicable value as `n/a`, as BIDS requires, instead of an empty cell

### Fixed
- A flat channel passed the coupled-window screening, mne-nirs scoring its rounding residue as coupled
- `fnirs-hyper-pairnull` refused a stand-in whose padded cut reached its recording's last sample
- `fnirs-hyper-pairnull` failed when the two cuts of a condition rounded one sample apart
- Derivative SNIRF renumbered the optodes after a missing source or detector, renaming later channels
- `--no-report` left no `_desc-channel_qc.tsv`, so the subject index and dyad pages lost `is_bad`; the record writer now writes it

## [0.58.0] - 2026-10-05

### Added
- GLM sidecars record the boxcar length or the events table, and an FIR model's delays
- The screening sidecar records whether the coupled share was counted over task blocks or the whole run
- WTC sidecars record `wtc_cond_pad_s`, the per-condition route (null when windowed out of the whole run)
- Quality records keep IMU gyroscope speed and accelerometer jerk (mean, p95) per run, condition and window, and their frame-level agreement with GVTD
- Subject, raw and cohort reports show the IMU summaries; cohort boxes mark the mean beside the median
- Methods describe ALFF/fALFF and functional connectivity when a run computed them
- Methods describe the re-paired null: who stood in, how many, how each condition was cut, and how stand-ins were summarised
- Methods describe the phase-scrambled nulls of the coherence and of the correlation
- Methods describe the relative phase per frequency
- WTC sidecars record `wtc_mc_count` and `wtc_seed` when `--wtc-significance` ran, and Methods describe that level
- `fnirs-hyper-groupnull` writes its cohort test's Methods paragraph to `logs/<cohort table>_methods.md`

### Changed
- Sidecar `Sources` are BIDS URIs (`bids::`, `bids:raw:`, `bids:preprocessed:`) rather than absolute paths
- `dataset_description.json` links its source trees by relative path in `DatasetLinks`, keeping links other commands wrote
- A tree whose `Sources` are not BIDS URIs is refused where they are read; rerun the command that wrote it
- Coherence labels and help call the edge-affected region the cone of influence, so kept cells read "outside COI"
- Stage comparisons band-limit with the pipeline's own filter design rather than MNE's default FIR
- The pass/fail grids, brain views and optode flat map colour with the run's own screening lines
- The grand mean averages the good channels only
- Methods state the wavelet design, the boxcar length or FIR delays, and the screening scope actually counted
- Methods say whether per-condition WTC was read out of the whole-run transform or transformed on its own
- Methods say bad channels were marked rather than excluded, and that a GLM fits and flags them
- Methods give an FIR model's boxcar durations and name no input file
- The filter-edge caveat states the high-pass's reach, one period of the cutoff, instead of an edge-to-middle RMS ratio
- Condition-page GVTD tooltips name the corrected file their windows come from
- Mean SNR has no colour cutoffs
- The reported GVTD threshold is the mode plus 10 left-side SDs, as censoring uses, and its share is labelled "% above threshold"
- The GVTD motion band is 0.02-0.5 Hz, the published task band, and `gvtd_threshold` defaults to the reported multiplier
- The cohort outlier score and table cover the charted metrics only
- The cohort over-time grid, by-condition panels and each run's condition timeline draw the motion-band GVTD, not the unfiltered one
- Reports show only the motion-band GVTD; the unfiltered values stay in the record, TSV and database for comparison with other implementations

### Removed
- The spline motion-correction Methods sentence, and the unreachable caveats for a missing short channel and for few trials

### Fixed
- `--mode glm` without `--stim-dur` or `--events-path` failed per subject after preprocessing instead of stopping first
- The broadband rest residual's sidecar recorded a passband it never went through
- `fnirs-qc hyper-raw --tstart` drew the usable-time carpet and head maps tstart seconds off the other panels
- The too-short window warning never showed when `--wtc-band-fmin` was left unset
- The unfiltered-ISC caveat showed even when `--isc-fmin` high-passed the correlation
- A rejected pair's Status in the channel decisions tables did not name the criterion
- Condition pages printed an empty trial window and measured the filter's stopband in their greyed stage panels
- Raw condition pages dropped the motion numbers on a montage without short channels
- A flat channel at a non-zero level lowered `cv_mean`
- The orphan-separation caveat counted each wavelength and suggested bounds the CLI refuses
- The chunked-trials caveat showed when no annotation was long enough to cut
- The raw-signal caption claimed uncorrected data when it had fallen back to `desc-preproc`
- The cohort report charted censoring columns under "Other", settings included

## [0.57.0] - 2026-10-05

### Added
- A crossed WTC null also writes a null for each dyad's crossed ROI x ROI matrix
- Dyad WTC sidecars record the channel crossing, the ROI minimum, the input stage and the rejection scope
- `fnirs-hyper-groupnull` corrects the crossed ROI null's cells in its per-cell table, as a family of their own
- The GLM fit table and residual record the design's conditions, and its HRF when it holds any

### Changed
- **`--wtc-roi-min-channels` defaults to 1, so one surviving channel keeps a region; pass 2 for the earlier rule**
- `fnirs-hyper-groupnull` leaves a thin region out only for the occasions and conditions whose own ROI table blanks it
- `fnirs-hyper-merge` refuses dyads run on a different stage, rejection scope or ROI minimum
- **`fnirs-hyper-pairnull` takes its stage, rejection scope, crossing and ROI minimum from the real tables, refusing tables that lack them or a retyped value that differs**
- `fnirs-hyper-groupnull` keeps a region by the ROI minimum the real tables recorded, refusing tables that lack it
- The crossed ROI table, not the homologous one, is described as the ROI number to report
- The Hyper Analysis re-paired null form drops the fields it now reads off the tables and offers a homologous-only switch
- A dyad's Methods follow the member files the analysis read, so `--desc` sets which preprocessing they describe
- Methods name the cardiac band, screening scope, hand-marked channels, every nuisance regressor, the ISC options and the COI exclusion
- `fnirs-qc prep-raw` and `fnirs-qc hyper-raw` reports carry no Methods paragraph
- Report captions and metric tooltips that disagreed with the computation are rewritten
- Spike count has no colour cutoffs; var-normalised GVTD and CP have no better end

### Fixed
- The phase-scrambled ROI null ignored `--wtc-roi-min-channels` and always required two channels per region
- The Methods paragraph called a crossed dyad's coherence homologous
- The `--wtc-channel-cross` help said the coherence maps stay on homologous pairs
- The dyad Methods left out the members' preprocessing
- The Methods said recordings were aligned under `--no-align`, and named `ar_irls` an unspecified noise model
- The raw viewer's Methods called a task GLM a regression with no condition regressors
- Windowed CV cut its windows a sample shorter than SCI and PSP at rates like 7.8125 Hz, so the grids drifted apart

## [0.56.0] - 2026-10-02

### Added
- The Analysis page's cutoff fields take `none`, so a cutoff the mode fills in can be switched off from the GUI
- Subject and dyad reports note each channel the ROI mapping lists in more than one ROI
- `fnirs-hyper-groupnull` tests every ordered region pair against a crossed null, as its own corrected family
- The subject report's run page shows every condition's channel pass/fail grid, one block per metric

### Changed
- The windowed SCI and PSP colour bars are pinned with the threshold mid-scale, as CV's already was
- The 3D brain views are lit with white light and drawn matte, no longer yellow-tinted and glossy
- `--wtc-roi-min-channels` counts each member's channels, not channel pairs, and a thin ROI cell stays in the table as a blank row
- The dyad report draws no ROI coherence map for a cell the ROI table leaves blank
- Resting-state and hyperscanning Fisher z now share one transform, which turns a non-finite r into NaN
- `fnirs-pipe` stops when `--roi-mapping` cannot be read, as `fnirs-hyper` does, instead of running without ROI output
- `--high-pass` and `--low-pass` take a plain decimal or `none`, refusing negative values, `nan`, `inf` and exponent notation
- ISC `null_abs_sd` is the sample SD (ddof=1), as the WTC and group nulls already were
- Montage colours follow the shared long/short split, so a channel between the two ranges is drawn unclassified
- An ROI with no good channel stays in the resting-state ROI matrix, seed map and ALFF table as a blank row
- Fisher z leaves the diagonal of a blank row (a bad channel, an empty ROI) blank instead of 0
- The hyperscanning modules share one band filter, cone-of-influence mask, row z-score, Yule-Walker step and source list instead of copies
- Windowed CV, window centres, the motion-band derivative and band PSD means each have one definition in the QC metrics
- QC figures share one matplotlib-to-PNG encoder, one colour-to-rgba helper and the report's PNG saving
- Sidecar reads, post-stage sidecars, subject folder listing, repeated CLI flags, rating files and the GUI run status each have one implementation
- A SNIRF's sidecar is read as UTF-8 on every platform
- QC metrics and figures that accept intensity or optical density convert through one `as_optical_density`
- A channel's source-detector pair is read through one `pair_of`, replacing about 45 inline splits and four local helpers
- The crop, edit-markers and aligned trees name the dataset they were made from under `SourceDatasets`
- The hyperscanning ROI functions drop an ` hbo`/` hbr` suffix from map entries themselves, not only on load
- Dyad sidecars, the chromophore check, dyad table reads and the run id format each have one definition

### Removed
- `optode_layout_figure`, the Plotly optode map nothing called

### Fixed
- The HbO–HbR correlation panel and its frame shrink and grow with the browser window
- The dyad and subject raw reports draw and embed their provenance graph instead of a placeholder
- Prewhitened ISC paired the two members a few samples apart whenever their highest AR orders differed
- The raw report's 3-D montage drew each source-detector line from the channel midpoint to the source
- The raw report's 2-D montage left out channels marked bad, so they were never drawn grey
- `participants.tsv` ids without `sub-` were matched for bad channels but not for age
- The optode layout passed over an HbO SCI of exactly 0 and took the HbR value instead
- The dyad PSD panels shaded the physiological bands in their own colours rather than the shared ones
- Crop, edit-markers and align kept whichever `dataset_description.json` version stamp they first found
- WTC ROI means counted a channel two ROIs list in the last one only; it now counts in both, as in ISC
- An ROI mapping entry written with an ` hbo` or ` hbr` suffix matched nothing in `fnirs-hyper`

## [0.55.0] - 2026-10-01

### Added
- `--mode` fills in its own defaults for the post settings, under `--config` and the command line
- `--high-pass none` and `--low-pass none` switch off a cutoff the mode or `--config` would fill in
- The run record names the layer that set each post value, and the QC report lists the settings nobody typed
- The Analysis page starts its filter, drift, HRF and noise-model fields empty and shows the chosen mode's defaults in them

### Changed
- `--mode glm` stops before the first subject when its cosine drift model has no `--drift-high-pass`
- Every QC report template reads its section prose from `boilerplate.notes`, now one module per report family
- Run caveats and the phase-arrow caption read their wording from `boilerplate.notes`
- Prose and labels that quote a window length, band, channel share or draw count read it from the constant that sets it

### Removed
- `configs/`: the presets are now the mode defaults shipped inside the package, and the example config is gone

### Fixed
- `drift_order` and `combine_runs` in a `--config` file were overridden by their command-line defaults
- The per-subject run script ignored values set in `--config`
- The grand-mean caption named a −5 s baseline whatever `--epoch-tmin` was
- The cohort dial caption said six dyads when the cohort had fewer

## [0.54.0] - 2026-09-29

### Added
- The carpet and per-channel motion figures draw the gyroscope's angular speed above the GVTD rows when the recording carries one
- The same figures draw the accelerometer's jerk as a second IMU row when the recording carries one
- The dyad motion panel draws each member's gyroscope and accelerometer on the shared clock, each divided by its own median
- IMU rows print the unit the recording stores, on the axis or beside the dyad medians
- `fnirs-qc hyper-raw --derivatives-dir` names the fnirs-pipe tree its motion panel reads corrected recordings from
- `fnirs-rate hyper --derivatives-dir` keeps each member's channel decisions in the fnirs-pipe tree, beside their raw page

### Changed
- Carpets detrend each row linearly before z-scoring, so the grey shows fluctuation rather than drift
- `fnirs-qc hyper-raw` refuses an output directory that is its `--derivatives-dir`, or any fnirs-pipe tree
- `fnirs-qc hyper-raw` without `--derivatives-dir` leaves an existing fnirs-hyper `dataset_description.json` alone

### Fixed
- `fnirs-qc hyper-raw` looked for corrected recordings in the fnirs-hyper tree, so its motion panel never had a corrected side
- The Hyper Preparation page wrote its dyad report into the derivatives tree and restamped it; it now takes a fnirs-hyper directory
- `fnirs-rate hyper` help named the fnirs-pipe tree; its directory is the fnirs-hyper tree holding the raw report

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
