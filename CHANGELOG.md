# Changelog

All notable changes to this project will be documented in this file.

<!-- Format: Keep a Changelog (https://keepachangelog.com/en/1.0.0/) -->

## [Unreleased]

### Added
- **The raw signal quality panel draws the coefficient of variation per channel and per window**, on the same window grid and the same optical density as SCI and PSP. SNR is 1/CV, so it rides that row's hover rather than taking a second one, and the row's colour is pinned to its cutoff instead of to the recording's worst window

### Fixed
- **The channel quality maps show the screening verdict, not the SCI alone.** A channel rejected for being loose through most of the recording was drawn green beside a table calling it BAD, because both the 3D views and the optode flat map coloured by whole-run SCI and screening has been on the coupled-window share. A rejected channel is red now whatever its SCI; the rest are still graded by theirs
- **A per-condition page shows the channel quality maps**, carrying that condition's own SCI and its own rejected channels
- **The grand-mean panel draws one condition per row** instead of squeezing them into one row side by side, on a shared time and concentration scale. Its legend no longer sits on the first panel's title
- **A per-condition page shows the per-channel before/after motion figures and the denoising carpet.** Both sections were blank on it. Each is measured over the whole recording and viewed over the condition, so the conditions stay on one scale
- **A per-condition page reports the GVTD percentiles, the motion-band GVTD, the frame counts and how much of the condition motion correction touched.** All of them read n/a although the run had already measured them per window
- **A per-condition page names the stage each motion row is measured on.** Its GVTD comes from the corrected recording and its spike share from the uncorrected one, and the two stood side by side as bare numbers. The GVTD threshold and the spike count are the run's, and the page says so rather than showing them empty
- **A per-condition page prints its mean CV**, which it had been computing and dropping

### Changed
- **CV and SNR are measured over short windows and averaged, instead of over the whole recording.** A whole-run CV grows with recording length rather than with noise, so stored CV, SNR and `snr_pass_rate` will differ from earlier runs, in the direction of passing more channels
- **The quality record carries CV and SNR per window**, so the group report draws them as heatmaps alongside SCI, PSP and GVTD
- **A per-condition page screens on its own stretch**, so a channel loose in one condition and coupled in another is named in the one it was loose in. The recording is still processed under the run's verdict, which its own page carries
- **Per-condition pages report the GVTD above-threshold share and the spike share**, counted from the run's own flags over that condition's span rather than re-decided on it
- **The windowed GVTD series is stored per separation set**, long, short and all, the way its scalars already are. GVTD averages across channels, so a set's series cannot be recovered from another's. The unsuffixed keys are the long channels now, so stored `gvtd_*_per_window` will differ from earlier runs
- **Per-condition pages carry the global correlation, the HbO-HbR correlation and the cardiac and respiration bands**, recomputed on that condition's own cut. The band metrics are left out for a condition too short to transform on the run's frequency grid, rather than reported on a coarser one

## [0.33.0] - 2026-09-09

### Added
- **`fnirs-pipe --by-condition` writes one QC report page per annotated condition**, beside the run's own as `desc-<condition>`. Every number is sliced out of the run's own windowed pass, so the conditions share one window grid and one filter. The rejected channels stay the run's verdict, and CV and SNR are left out
- **`fnirs-qc prep-raw --by-condition` does the same for the raw QC report**, with the condition as the `task-` entity. CV, SNR and the PSD are left out

### Changed
- **The GLM activation panel switches between conditions instead of stacking them**, on a colour scale shared across them so the switch is still a comparison. Five conditions in one image could not be read or compared without scrolling

### Fixed
- **The optode, flat-map and brain figures colour channels by the run's own SCI threshold** instead of a fixed 0.75, which could put a channel on the wrong side of the verdict the same report printed
- **The per-channel timeseries on a per-condition page no longer draws its data and its condition shading in different time frames**, which squeezed the trace into a corner of the panel
- **The per-channel epoch panel is no longer empty on a per-condition page**, which made a page report no trials on a condition that has them
- **The GVTD panel scales to the bulk of its trace rather than its largest spike.** The rows still share one scale and each row's true maximum is still printed in its label
- **The epoch and HRF preview draws one panel per condition, with HbO red and HbR blue.** Colour used to carry the condition and line style the chromophore
- **A run no longer warns about the columns of its own events file.** A BIDS events table legitimately carries columns the design matrix does not read
- **The group quality table names the metric pairs that straddle the bandpass**, since subtracting such a pair measures the filter rather than the stage
- **`fnirs-pipe` records the coupled-window share it screened on**, which it had been counting, rejecting channels by, and then dropping, so its records disagreed with `fnirs-qc prep-raw`. Records written before this leave the row out rather than recomputing it
- **The coupled-window share appears in the reports' scalar panels.** It is the metric a channel is rejected on, and no report said what the run's own figure was
- **`--min-good-frac` now reaches the per-trial quality panel**, which screened at the default share whatever the flag was set to
- **The channel quality summary no longer describes a rule it stopped using.** It said a channel is rejected for failing SCI or PSP; rejection has been on the coupled-window share

## [0.32.0] - 2026-09-09

### Added
- **`--min-good-frac` sets how much of a recording a channel has to be coupled for**, as a share of windows, default 0.75. `--sci-threshold` and `--psp-threshold` say what a coupled window is; this says how many of them a kept channel needs
- **`--screen-scope task` counts coupled windows only inside the annotated task blocks**, so the lead-in and the gaps between blocks stop being held against a channel that is coupled throughout. Default stays `run`, and a recording carrying only short triggers falls back to it and says so
- **The raw QC record carries the coupled-window share per condition.** Reported only: one channel set still serves every condition, since otherwise a contrast between two conditions is also a contrast between two montages
- **The pseudo-dyad null follows `--wtc-by-condition`.** It was built on the whole recording however the real table was read, so a short condition looked further above chance than it was. It lands in its own table, at no extra iterations
- **`fnirs-prep crop` can cut a processed stage instead of a recording**, with `--input-desc`, so motion correction and the bandpass still see the whole recording. The cut keeps that stage's `desc-` entity, its bandpass and its bad-channel marks

### Changed
- **Channel screening counts coupled windows instead of comparing two whole-run averages.** SCI and PSP are now thresholded inside the same short window and combined there, which is how the two were defined, and a channel that was usable for part of a long recording no longer passes on the average. **Rejected-channel sets will differ from earlier runs**, in both directions
- **`--wtc-by-condition` reads each window off the whole-run transform instead of transforming it on its own**, which changes its numbers. Transforming a window alone inflated the band mean by an amount that tracks window length, and is slower. The channel pattern is unaffected
- **A run whose input was written by `fnirs-prep crop` now stops** instead of preprocessing each condition on its own, and names the order to use. `--allow-cropped-input` runs it anyway, for reproducing an older analysis
- **A run that replaces an earlier one made with a different passband now says so.** Nothing an output directory holds carries the band in its name, so a rest band and a task band mean two directories

### Removed
- **`--short-channel pca` is gone**, leaving `none` and `mean`. Weighting by variance follows the loudest short channel rather than what the short channels share. A run still asking for it stops and says so

### Fixed
- **Censored spans no longer enter the GLM as a task condition.** With `--gvtd-censor` on, a run fitted an extra HRF-convolved regressor over the frames the censoring had flagged as unusable
- **Editing markers no longer drops the recording's auxiliary channels**, which left `--aux-regressors` further down nothing to read
- **Editing markers no longer fails on a derivatives directory that does not exist yet.** A dataset with a `participants.tsv` stopped on the first subject
- **`--wtc-by-condition` no longer takes every window late** on a dyad aligned to its first shared trigger. Only reachable on a recording holding several conditions, which is why it went unseen

## [0.31.0] - 2026-09-08

### Added
- **`--short-max-dist` / `--long-min-dist` / `--long-max-dist` set what counts as a short and a long channel**, in mm. They were fixed in the source, so a montage the defaults do not describe (infant arrays, high-density ones) could not be measured correctly. The run's values are recorded, so an old record still says which separations produced its split
- **Wavelet coherence runs on both chromophores.** `--wtc-chroma {hbo,hbr,both}`, default both. It was HbO only, so the HbO/HbR consistency check was available on ISC and not on WTC; a coupling in HbO with nothing in HbR is a caution flag. One switch in the report moves every coherence panel between the chromophores at once. Both costs twice the time and roughly doubles the page; pass `hbo` for the old behaviour
- **`fnirs-hyper run` reads the separation bands off the members' quality records** rather than being told them, so the dyad metrics can no longer be split one way while the member reports were split another. Members preprocessed with different bands are refused. The three flags stay as an override for a tree preprocessed before the bands were recorded

### Changed
- **The WTC band-mean tables gained a `chromophore` column**, and the saved maps split one archive per chromophore, so tables written either side of this release will not concatenate

### Removed
- **`--gvtd-channels` is gone.** GVTD always covers the long channels now, the set `--long-min-dist` / `--long-max-dist` define and the only set the rest of the analysis uses.

## [0.30.0] - 2026-09-07

### Added
- **The dyad report says so when ISC runs on a stage with no bandpass on record.** `--desc` defaults to the Beer-Lambert output, which still carries its drift, and a whole-record correlation has no frequency axis to keep drift out of it: two members recorded in one room drift alike for reasons that are not neural. The wavelet coherence panels are unaffected and get no note
- **The GVTD panel draws the short channels on their own row**, under the long one and on the same scale, with the carpet split into a long block and a short block beside it. Short-channel quality had no time-resolved view anywhere in the report. The verdict, the threshold and the reported scalars still come from the long row alone, so nothing a run is judged on changes. Each row shades the derivative spikes found on its own channels, which are now detected per separation class rather than on the long ones alone
- **`--psp-threshold` sets the second screening line.** It was fixed at 0.1 with no way to change it, even though a channel failing it is rejected
- **`fnirs-qc hyper-raw` exposes the windows its figures use**. All three were fixed in the source with no way to reach them
- **`--epoch-tmin` / `--epoch-tmax` set the trial window the subject report works in.** Every epoch figure and the per-trial panel were pinned to -5 to 25 s, which suits a single trial and not a 60 s block. Left unset, the report says so when the run's events outrun the window the figures average
- **`--epoch-chunk-duration` cuts a long task annotation into trials the epoch figures can average.** A block design marks one 240 s annotation per condition and nothing can average a single 240 s trial, so that half of the report described the start of each block. Cutting at 25 s gives 9 trials
- **The WTC sidecars record the wavelet grid.** Neither value is configurable, and both decide how many time-frequency cells a band mean averages over
- **The raw QC report and the interface judge long channels separately**, with an All / Long / Short comparison. Averaging short channels in was lifting SCI, PSP and SNR
- **The raw QC report and the interface show every per-channel metric**, not SCI alone, and write the per-channel metrics CSV
- **The subject report gained the event timeline and the per-trial quality panel**
- **The analysis page offers the bandpass design.** `--filter-method` and `--filter-order` were command-line only, so the interface silently pinned every run to the defaults
- **`--gvtd-censor` marks the frames GVTD flags as `BAD_gvtd`, so an analysis can leave them out.** Nothing is cut, so a threshold set too strictly is undone by rerunning. `--gvtd-censor-n-std` and `--gvtd-min-epoch-s` set the threshold and the shortest stretch worth keeping. Off by default: on a high-motion recording it can flag everything
- **The hyperscanning and group reports now end with the same closing sections the subject report does**: what failed, what was left out on purpose, the provenance table, and the software versions. A panel that failed used to appear only in the run log, so a report could be read as complete when it was not
- **The hyperscanning reports carry a Methods paragraph**, in the same four tabs the subject report offers. It continues from a member subject's preprocessing into the alignment and the cross-brain measures, so the paragraph describes one pipeline end to end, and it says so when the two members were not preprocessed the same way. **The wavelet coherence, coherence, ISC and alignment citations are placeholders**: they print as `TODO-ADD-...-REFERENCE` until the real references are filled into `references.bib`
- **The hyperscanning reports carry their provenance**, the post report as a diagram rebuilt from the sidecars its own WTC and ISC passes wrote, and both as the table naming every file and the step behind it
- **`fnirs-hyper run` writes a run record** to `group-<id>/logs/`, the mirror of a subject's: the verbatim command, the machine, and every option the invocation resolved to
- **The raw QC report ends with an errors and warnings panel**, per run. Thirteen of its panels could fail into the log alone, so a viewer missing half its figures looked like a recording with nothing to plot
- **The hyperscanning raw report writes its coherence tables**, whole-record and windowed, so the numbers behind the bar chart and the heatmap can leave the report

### Fixed
- **`--short-channel` built its regressors from every channel on a recording with no registered optode positions**, so the "systemic" signal regressed out of every channel was the whole montage. Such a run now skips the regression
- **The per-channel motion figure measured GVTD over every channel**, while the carpet panel above it and the metrics table beside it measured the long ones, so one report carried two different GVTD traces and two different thresholds with nothing saying why. Each figure now takes the GVTD of the separation class its own channel belongs to, names it, and shades the derivative spikes found on that same class
- **`fnirs-pipe` wrote no SQM record for any run.** The failure was logged and the run carried on, so the group tables and the quality database were left with whatever an earlier run had put on disk
- **"Long channel" meant two different things**: a bounded band to the reports and the GVTD trace, anything over 10 mm to the dyad metrics and the short-channel regressors, so the same channel could be inside the montage in one half of the package and outside it in the other. One rule now, and a channel in neither band is named in a warning instead of silently taking part in nothing
- **Crossing the channels drew both axes from one member of the dyad**, so the crossed WTC table and the ISC matrix were missing every pairing that used a channel the other member kept and this one had rejected. Which member counted depended on the order the pairs table lists them in. The crossed table grows by the pairings it was dropping
- **The dyad matrices are indexed by the montage**, rejected channels included, so every dyad's matrix has one shape and a group analysis can stack them however their rejections differ. A rejection blanks its own row or its own column, never both
- **ISC returned a number for two members recorded at different sampling rates.** It paired the nth sample of one with the nth of the other, which are not the same moment. It refuses now, as the wavelet coherence always has
- **The crossed WTC matrix drew its axis from one member of the dyad**, so a channel only the other member's montage carries never reached the figure. ISC took both montages already; the two share one rule now. Reaches a dyad whose members were not capped alike
- **`fnirs-qc hyper-raw` scored its per-subject quality table over every channel**, while the individual reports score the long ones, so a subject's SCI, CV, SNR and GVTD could not be read across the two. It is the long-channel view now, and says which set it describes
- **`--gvtd-censor` failed on any recording that has event markers**, which is every task run. It worked only on a recording with no markers at all
- **The interface showed no SCI at all in its channel table**, a dash on every row
- **The per-channel PSD shaded the cardiac and respiration bands at fixed frequencies** instead of the run's own, so a study outside the adult range had the stripe over the wrong part of the spectrum
- **The interface's all-channel PSD panel never appeared.** It was never sent a figure to draw, it stayed hidden until a channel was selected, and selecting one replaced it with that channel's spectrum while the label still read "mean across channels". The two spectra are separate panels now
- **The interface's Per-channel Metrics table stayed empty on every run**
- **The same metric could read as passing in one view and failing in another**, and the interface printed record keys with no label, units or verdict
- **A figure that failed took the whole group report with it.** Every other report loses one panel; this one had no error handling at all

### Changed
- **A channel over 45 mm is a long channel now.** The long band's upper edge is off by default; it used to drop those channels out of every section and metric without appearing anywhere as excluded. Screening still judges them on their own SCI and PSP. **On a montage with channels past 45 mm the long-channel metrics all move, GVTD included**
- **`fnirs-qc hyper-raw`'s coherence band is `--coh-fmin` / `--coh-fmax`.** As `--fmin` / `--fmax` it said nothing about which of the report's bands it set. The old names still work
- **The subject report's per-trial panel scores each trial over the event's own duration** when no epoch window is given, which is what the raw viewer has always done. It used a fixed -5 to 25 s window, so on a block design it scored the first 25 s of a block and called that the trial. The panel says which window it used
- **The analysis page offers the PSP threshold and the epoch window**, so the interface can no longer only produce runs that screen and epoch at the defaults
- **`fnirs-hyper run`'s report is named the way every other report is**, `desc-hyperpost`. The two dyad pages followed two conventions
- **The hyperscanning reports print their per-subject metrics the way every other report does**, so a number cannot appear to three decimals in one and four in another. Some labels change
- **The all-channel PSD panel draws a mean per separation group** instead of pooling every channel into one curve. A short channel's spectrum sits well above a long one's, so the single mean described neither, and the groups and their colours are now the ones the rest of the raw report uses
- **The channel quality heatmap prints its channel names in the same colour in the report and in the interface.** Any figure that left its font unset came out one colour when the command line wrote it and another when the interface drew it
- **Every PSD panel says which stage it is measured on**: optical density before Beer-Lambert for the all-channel spectrum, concentration after it for a single channel's. The two look alike and were being read against each other
- **Channel screening rejects a channel that fails SCI *or* PSP**, where it tested SCI alone. Runs will reject at least as many channels as before, and the reports name which criterion failed
- **The CV pass line in the channel quality grid was 50%**, looser than any published threshold, so the CV row passed almost everything. It is 5% now (Lloyd-Fox 2009), measured per wavelength, and the SNR line is derived from it
- **The per-trial quality panel prints its numbers the way the rest of the report does**
- **Every QC report is rendered into one shared page shell**, so the header, the rating bar and the closing sections are the same wherever they appear and a new report cannot ship without them
- **The reports share one stylesheet for prose, tables and figures**; the dashboard pages and the document pages now differ only in density and chrome. The raw viewer joins the other dashboard pages, so its nav wraps rather than overflowing and its tables and images are framed the way every other report frames them

## [0.29.0] - 2026-09-06

### Added
- **The interface exposes the SCI/PSP window and the epoch window**, and can run the per-trial QC panel, which was command-line only before
- **`--gvtd-channels` chooses which channels GVTD covers.** It defaults to the long channels, which is what the reports already drew; the choice is named on the carpet figure and in the parameter table

### Fixed
- **Short channels lost their HbO-HbR correlation** in the per-channel table and the metrics CSV
- **"What each denoising step did" averaged long and short channels together**, so the same metric read one number there and another in the table above it. It is measured on the long channels now
- **A spike on the last sample of a recording was never drawn**
- **The 3D layout never marked the selected channel**
- **Peaks in the GVTD and per-channel derivative traces were drawn slightly late**
- **Clicking a cell in the Signal Topo selected a different channel.** Cells also overlapped wherever two channels share an optode
- **The interface drew its per-channel figures at a fixed DPF of 6**, whatever DPF the run was set to
- **The "Raw Signal" section was not raw.** Its figures came off the motion-corrected recording while the sliding-window SCI/PSP and the SNR/CV numbers on the same page came off the recording before correction. They are all on the uncorrected stage now, and the section says which stage and which DPF it used
- **The interface measured GVTD over every channel** where the subject report and the stored metrics measure it over the long ones, so the two disagreed on the same recording
- **A subject's runs shared one set of ratings**, so rating one run overwrote the last, and on multi-session data the rating bar linked to sections the page did not have. Ratings saved before this release are not read back

### Changed
- **"What each denoising step did" now sits directly under the metrics table**, where the numbers it carries forward are
- **The quantitative metrics table reads across instead of down.** Each metric is a column, the three channel sets are the rows, and the optical density and haemoglobin halves share one table
- **The provenance diagram says less.** The QC record box no longer lists every stage it covers
- **The correlation matrix draws its lower triangle only.** The upper half repeated the same values read the other way round
- **The interface was restyled.** The run picker sits beside the subject it belongs to, and the Signal Topo shares the viewer row evenly with the layout panels
- **`--epoch-tmin` and `--epoch-tmax` now also set the window the epoch figures are drawn over.** The evoked topo and the per-channel epoch preview always showed -5 to 25 s whatever was asked for
- **The motion figures were redrawn.** Spike segments shade the traces themselves instead of sitting on a band below them, the motion-correction footprint moved to a strip directly above, and rows are labelled where the label fits
- **The GVTD and per-channel derivative traces are drawn at their own resolution.** Pooling them to 2000 points turned a 15-minute run's trace into a row of pickets
- **The report's section links and the rating row are one bar.** The links used to vanish behind the rating chips as soon as the page scrolled, and every figure section can be rated now instead of four of them

### Removed
- **The "What the motion correction did, and what it cost" table.** The GVTD panel answers the same question in a form you can read

## [0.28.0] - 2026-09-06

### Fixed
- **A recording whose input was already optical density could lose its whole report.** The metrics panel compared an undefined SNR against its threshold and stopped rendering
- **The per-channel derivative trace showed pulse, not motion.** It was differenced on the already-downsampled signal and never band-limited, so the cardiac rhythm aliased into it and its peaks did not line up with the spike marks drawn underneath. It is now taken at full resolution over the same 0.01-0.5 Hz band the spikes are detected on
- **The low-pass filter barely filtered.** `--low-pass 0.2` attenuated the cardiac band by 4 dB where a working filter manages 60, so every `desc-filtered` and `desc-errts` still carried the heartbeat. The high-pass was unaffected. Every filtered file, and every number measured on one, changes: rerun anything written before this
- **A run whose markers cannot be epoched filled the console with warnings.** A design that only marks where each condition starts and ends carries annotations but no trials, and every epoch figure tried anyway. The run is now recognised up front, said once in the log, and reported as a note

### Added
- **Run notes for a montage the metrics cannot split**: no registered optode positions, channels at a separation the long and short ranges leave out, or short-channel regression requested where there is no usable short channel to build it from
- **Run notes in the subject report**, listing the sections a run left out because its data does not carry what they need, kept apart from the errors list and repeated once at the end of the run
- **`--filter-method` and `--filter-order`.** The bandpass is now a zero-phase Butterworth of order 4, which is what the fNIRS toolboxes use and what a methods section can state. `--filter-method fir` keeps the linear-phase option and refuses, rather than quietly truncating, when the filter would be longer than the recording
- **The filter is recorded on every stage's sidecar**, not only on `desc-filtered`, so a file can say which passband its content is confined to
- **A warning when a coherence range reaches past the bandpass.** Asking for scales above the low-pass reads as a working analysis and is not: those scales carry what the filter removed
- **`--wtc-by-condition`**, which runs the coherence inside each task annotation's window as well as over the whole recording, so a block design gets one result per block. A trigger with a duration uses it, one without runs to the next trigger, and a window too short to carry the lowest frequency asked for is skipped and said so

### Changed
- **The haemoglobin metrics are split by source-detector separation too.** HbO-HbR correlation, global correlation, CNR, band power and drift were averaged over long and short channels together, which moved the correlation that decides whether a run looks usable. Every stage now carries a long-channel and a short-channel section, and the reports read the long one. These numbers change: rerun anything measured before this
- **`fnirs-qc prep-raw` writes the same record shape the pipeline does**, with the `raw`, `raw_long` and `raw_short` sections, instead of one flat all-channel record. Its report and the group table gain the long/short split; existing flat records are still read
- **The quantitative metrics panel reports SCI, PSP, SNR, CV, amplitude and retention in three columns, All / Long / Short**, instead of one long-channel number with nothing to compare it against. The rest of the scalars are grouped by what they were measured on, so it is visible which ones still include short channels
- **The per-channel metrics table is grouped by source-detector separation**, and short channels now carry their SCI, PSP, SNR and CV instead of a dash. The Channel Quality Summary grid is grouped the same way, and the scalar metrics now say that they are long channels only
- **The HbO–HbR correlation panel separates short channels from long ones.** Ranked together, the short channels sat at the top of the list and read as the worst channels on the montage, when the anticorrelation the −0.3 threshold tests for is a property of cortical haemodynamics and says nothing about a channel that only sees scalp. They now form their own group, drawn in grey and left out of the pass/fail colouring, and the correlation matrix is blocked the same way. A montage with no short channels looks as it did
- **The provenance diagram matches the interface's pipeline DAG.** Same palette, outlined boxes instead of filled pastel blocks, and straight arrows in the colour of the stage they feed. Red now means only that a file is missing, and the colour legend is gone: every box already names its own step
- **The subject report's five before/after strips are now one small panel per metric across the stages**, drawing every stage on disk rather than only the two ends, so a step that did nothing is visible as one. They sit in the Channel Quality Summary
- **Metrics compared across the bandpass are measured in a common band.** Compared as stored, most of what looked like denoising was the drift leaving the calculation. These panels will not match the per-stage values in the metrics table, which are unchanged
- **The motion figures drop the unfiltered GVTD trace.** Differencing amplifies the heartbeat well above head motion, so that trace read as pulse and never as movement, and it is the pipeline stage the GVTD paper measured the worst artifact-to-background ratio at. Only the 0.01-0.5 Hz band is drawn; `gvtd_mean` and `gvtd_p95` are unchanged and now sit next to their motion-band counterparts in the report and the per-trial table
- **The per-channel motion row is labelled `|dOD/dt|` rather than TVD.** It is a view of the spike detector's input, not a per-channel GVTD: GVTD is defined across channels and has no single-channel form, and TVD already means total variation denoising elsewhere. The stored `temporal_derivative_variance` is unchanged
- **The PSD panel draws the filter's own response over the filtered signal**, so what the bandpass did is read off the filter rather than guessed from the gap between two noisy curves, and the panel names the filter it drew. Both rows share one power axis, and the panel stops at the bandpass: the confound regression's output is not a spectral question


## [0.27.0] - 2026-09-05

### Added
- **The subject report shows HbO–HbR correlation before and after denoising**, one panel per stage plus a per-channel strip from `desc-preproc` to `desc-errts`. Removing shared systemic signal should push the correlation towards −1; the strip says on which channels it did
- **Contrast-to-noise ratio, per channel and across denoising.** It is the one signal-quality measure that stays honest across a bandpass: anything built from band power improves by construction once the filter has run, whether or not the data got better, while CNR falls if the denoising ate the evoked response. Task runs only, since it needs stimulus markers
- **SCI and PSP before and after motion correction**, as a per-channel strip under the quality panel. Both metrics sit above the frequencies motion correction works on, so a channel that fell is one where the correction removed cardiac pulsation along with the artifact
- **Relative phase, drawn as arrows on the ROI coherence maps.** Right is in phase, up means the first subject leads by a quarter cycle. Coherence on its own cannot tell a pair that moves together from one that moves together a few seconds apart. The maps saved with `--wtc-save-maps` carry the phase too
- **The crossed coherence reaches the report as figures.** `--wtc-channel-cross` computes every pairing across the two brains and wrote them to the TSV, but the page only ever showed the homologous ones. There is now a channel-by-channel matrix of the band means, and every ROI-by-ROI map on one grid
- **`--bad-channels` also takes a table**, one row per subject, for a cohort whose rejections differ. A plain list still applies to everyone
- **One quality-record section per haemoglobin file the run wrote**, each named after it: `preproc`, `filtered`, `resampled`, `errts`. A step that did not run leaves no section, so the bandpass and the confound regression stay separable

### Fixed
- **A hand-picked bad channel could survive into one chromophore.** Naming a single wavelength marked only that one, so after Beer-Lambert the pair's HbO was rejected and its HbR was not. Either wavelength now marks both
- **Connectivity and ALFF counted rejected channels.** A channel preprocessing had thrown out still carried a value, and read as an ordinary result. It is now blank in the tables and grey in the figures. Every FC and ALFF table on a subject with a rejected channel changes

### Changed
- **The coherence maps label their frequency axis at the decades**, and run frequency downward as the wavelet literature draws it. The axis had been left to plotly, which labelled every digit of every decade and switched rule with the band, so two figures in one report ticked differently
- **The `final` section is gone**, replaced by the sections above. It named a position rather than a file, so the same key meant the bandpassed signal on one run and the unfiltered signal on another, and on a run that regressed confounds it pointed at neither: the actual endpoint sat outside it. Group tables and databases written before this hold `final_*` columns and `checkpoint = 'final'` rows that will not be written again
- **The PSD panel plots the files the run actually wrote**, one line per stage, instead of re-filtering the unfiltered signal in memory to invent an "after" curve. The old curve showed the filter's shape rather than the run's, and never reflected resampling. A run with no post-processing still gets the simulated curve, labelled as one
- **The SCI/PSP panel is one stage throughout.** Its heatmap was read off the motion-corrected file while the lollipops beside it came from before the correction, so the two halves of one figure described different data. Everything in the panel is now the uncorrected optical density

## [0.26.0] - 2026-09-04

### Added
- **The subject report shows GVTD before and after motion correction**, as `before → after` on the same five rows it already had, the way global correlation shows the regression either side. Both sides are measured on the same channel set: a long-channel GVTD read against an all-channel one differs several-fold on the same recording, which would show up as an improvement the correction never made
- **The carpet + GVTD figure carries the corrected recording too**, a second trace in each GVTD panel and a second carpet under the first. Both carpets are on one colour scale and the threshold line stays the uncorrected one, so the panel is read against a fixed yardstick rather than a rescaled one

### Fixed
- **A `--no-report` run left the dyad analysis with nothing excluded.** Rejected channels were read from the per-channel CSV, which the report writes, so preprocessing that skipped the report produced a tree where `--desc errts` analysed every channel including the bad ones, and `--bads-scope subject` did nothing at any stage. Neither said so. Rejection now comes from the `desc-sci` sidecar, which prep writes either way, and the log names the source and the count. Every coherence value on a dyad with a rejected channel changes

### Added
- **`fnirs-hyper run` says what the metrics will run on before it starts**: long channels kept, channels rejected, mean SCI, and which runs the rejections came from, per subject. `--check-only` stops after that and writes nothing, so a cohort can be looked over before committing to a run that with `--wtc-pseudo` takes hours
- **`fnirs-hyper`, a tool for dyad analysis.** `run` writes the WTC + ISC report per dyad, `band` re-averages saved coherence maps over another frequency band, and `merge` concatenates the per-dyad tables into one long table per kind. Every subcommand takes one derivatives directory and reads no BIDS input

- `fnirs-hyper run` ends by saying whether the merged tables are behind the per-dyad ones, and with what command to catch them up. It does not merge on its own: a run often covers one dyad, and merging the whole tree after it would fail on a band a later run legitimately changed

### Changed
- **The carpet + GVTD figure is interactive**, an in-page plot rather than a PNG, so the before and after traces can be toggled from the legend where they lie on top of each other, and hovering names the channel and the time. It reaches the subject report, the raw viewer and the Data Preparation page alike
- **The dyad analysis left `fnirs-qc`.** Wavelet coherence, inter-brain correlation and the pseudo-dyad null are results, not quality checks, and hyperscanning has its own input and its own unit of analysis, so it is its own tool now. `hyper-raw` stays in `fnirs-qc`: a pre-flight look at raw dyad data is quality control. No output file changes name
- **The pseudo-dyad null is a flag on the dyad run again, `--wtc-pseudo N`,** and writes nothing unless you ask. Nine parameters had to match between `hyper-null` and the run it was the null for, and nothing on disk checked it: a null averaged over one band could sit beside a real table averaged over another. Shared by construction now. Its crossing stays its own decision, `--wtc-pseudo-cross`, which is what the split was for
- **One name per parameter for the frequency band.** `--band-fmin`, `--band-fmax` and `--mask-coi` were second names for `--wtc-band-fmin`, `--wtc-band-fmax` and `--wtc-mask-coi`, and the docs had to say they must agree. `--wtc-suffix` is the only flag `band` keeps to itself
- **`fnirs-pipe ... group` no longer demands the preprocessing flags.** `--dpf`, `--sci-threshold` and the four band bounds describe preprocessing and were required of every invocation, including a level that never preprocesses anything. A bad command line is also rejected before the pipeline imports rather than after
- The QC Reports page drives both tools and asks for one directory, since neither reads BIDS. Its Session Label field is gone: the pairs table names the session, and the field filtered nothing

### Removed
- **`fnirs-qc hyper-post`, `hyper-null`, `wtc-band` and `group-hyper-wtc` are gone**, with no forwarding. Use `fnirs-hyper run`, `--wtc-pseudo`, `fnirs-hyper band` and `fnirs-hyper merge`
- The dyad commands took `--session-label` and `--skip-bids-validation` and used neither

## [0.25.0] - 2026-09-04

### Fixed
- **The raw signal panel named its traces in the wrong order.** Channel names came from a legend whose order was the reverse of the stacking, so the name beside a trace belonged to another channel. Names are now axis ticks sitting on the trace
- **The 2D optode layout drew every channel link at half length.** It joined the source to the channel midpoint instead of to the detector, and drew no optodes
- **Optodes rendered off the brain when the montage was already in MNI.** The 3D layout, the quality brain views and the GLM surface projection applied fsaverage's head-to-MRI transform regardless of the source frame. The frame is now read from the file. Datasets in head space are unaffected
- **The correlation matrix labelled only every other channel**, so rows and labels appeared to disagree
- **`fnirs-qc group-hyper-wtc` merged only the channel table.** It asked for an ROI table removed in 0.24.0 and stopped there, so the ROI and null tables were never merged
- **`--bads-scope` never reached the coherence.** Rejected channels were recorded and kept out of ISC, but WTC ran on the full montage whatever was asked for. Every WTC value on a dyad with a rejected long channel changes

### Added
- `fnirs-qc hyper-post` / `hyper-null` / `hyper-raw` take `--tstart` / `--tend`, restricting the synchrony metrics to one window of the aligned recording. The per-subject quality record still describes the whole recording
- `fnirs-qc prep-raw --epoch-qc` adds a per-trial section to the raw report: each event window is scored on its own and shown as a trial x metric heatmap. `--epoch-tmin` / `--epoch-tmax` set a fixed window relative to onset; without them each event's own duration is used
- `fnirs-prep crop --align trigger --trigger-name TEXT` measures `--tmin` / `--tmax` and every segment onset from a named annotation instead of from the recording start, so one window selects the same stretch of task in subjects whose recordings started at different moments

### Removed
- **`fnirs-qc window-raw` is gone.** Cropping with `fnirs-prep crop` and then running `prep-raw` + `group-raw` over the cropped tree gives the same report, and unlike `window-raw` it leaves the windowed metrics on disk where the group aggregation and the provenance graph can see them. `crop` gains the `--align trigger` that only `window-raw` had
- **`fnirs-qc epoch` is gone**, replaced by `prep-raw --epoch-qc`. The trial scores now sit in the subject's raw report next to everything else measured on that recording, instead of in a separate pair of files

### Changed
- **The 2D optode layout is projected onto a head outline**, using the projection the topographies already use, with sources and detectors drawn and named. The axes are gone: projected distances are not millimetres, so read source-detector distance from the channel table instead
- **The hyperscanning reports and `hyper-raw_sqm.tsv` carried SCI and nothing else.** Both computed the full metric set and wrote three columns of it, and SCI is amplitude-invariant, so a run whose cardiac pulse had collapsed read as clean. The TSV now carries every scalar in the record, and both hyper reports gain a per-subject quality table (PSP, CV, SNR, GVTD, motion footprint, HbO-HbR). Nothing is recomputed
- **The raw QC report moves into `sub-<id>/`** with the subject's other reports, instead of loose in the derivatives root
- **The pseudo-dyad null is its own command, `fnirs-qc hyper-null`**, and no longer inherits `--wtc-channel-cross` from the real run. Leaving the null homologous costs 0.6 h per dyad against 8.4 h crossed. `hyper-post --wtc-pseudo` is gone; the table and its columns are unchanged
- The null records how many iterations it ran, and `group-hyper-wtc` refuses to merge nulls of different lengths

## [0.24.0] - 2026-09-02

### Changed
- **The cone of influence is no longer masked by default**, which is what the field does. `--wtc-mask-coi` restores it, and `n_valid_frac` reports the share inside the cone either way. Every coherence value moves, short conditions most
- **`hyper-wtc-roi.tsv` and `--wtc-roi-cross` are gone.** Averaging the signals into one ROI trace detects less than averaging per-channel coherences. `hyper-wtc-roichan.tsv` is the ROI table and `--wtc-channel-cross` builds the cross-ROI matrix
- **The subject QC report is now one report per run.** A subject with several tasks used to get one `sub-<id>_qc.html` built from whichever run finished last, with nothing on the page saying which. Each run now writes `sub-<id>_task-<task>_qc.html` with its own figures, and `sub-<id>_qc.html` becomes an index over them
- WTC computes only the wavelet scales its frequency range keeps, 1.8x faster on a 900 s recording. The coherences are unchanged bit for bit; `--no-wtc-limit-scales` restores the old behaviour
- **A group's outputs live in `group-<id>/` now**, the way a subject's live in `sub-<id>/`: tables and sidecars under `group-<id>/nirs/`, the two HTML reports beside them, figures where they already were. A study of 25 dyads over 5 tasks used to put two thousand loose files in the derivatives root. Filenames are unchanged, and everything that reads these files searches recursively, so an existing tree is still found
- **The quality record is drawn as a node rather than a step in the provenance diagram.** It measures every stage, so it had an arrow from each of them crossing the whole figure and burying the chain underneath. Its own box says which stages it measured. The provenance table names those stages too, in place of the full list of metric names, which stay in the record

### Added
- **`--wtc-pseudo N` writes a pseudo-dyad table**: the same band means against a phase-scrambled partner. Coherence between two unrelated recordings is not zero, so this is the null a real value has to be read against
- WTC band-mean tables gain `coherence_z`, the Fisher r-to-z group statistics should average
- `--wtc-roi-min-channels` drops an ROI cell resting on too few channel pairs
- **The subject page says which run is the odd one out, and which channels each run rejected.** A cell more than 3.5 median absolute deviations from the subject's other runs is marked, and a table names every source-detector pair that any run rejected together with the runs that rejected it, which is the set `--bads-scope subject` unions. Each run also links to its MNE report, provenance diagram, channel metrics and aux table
- **`hyper-post` writes `hyper-wtc-roichan.tsv` whenever `--roi-mapping` is given**: coherence per channel pair, then averaged within each ROI. This is the ROI number the WTC literature reports, and now the only ROI table. `n_ch` says how many channels backed each mean
- `fnirs-qc hyper-post --wtc-channel-cross` crosses every long channel with every other across the two brains: 196 values instead of 14. Single channels are noisy, so the off-diagonal is exploratory
- `fnirs-qc hyper-post --wtc-save-maps` keeps the full time-frequency maps, and `fnirs-qc wtc-band` re-averages them over another band without a second wavelet transform
- `fnirs-qc hyper-post --bads-scope subject` unions each subject's rejected channels over their runs, so conditions are compared on one channel set
- `fnirs-qc hyper-post` writes `hyper-bads.tsv`: which channels the inter-brain metrics excluded, and which run rejected each
- The group CSV accepts optional `session` and `run` columns, for a subject with more than one recording of a task
- The QC Reports page offers `wtc-band`, the saved WTC maps and the two new `hyper-post` options, so nothing added this release is command line only
- **`--aux-regressors` puts the recording's auxiliary channels into the confound regression**, in every mode that regresses. Accelerometers, gyroscopes, a pulse trace, whatever the device wrote to the snirf `aux` group. MNE never reads that group, so preprocessing now extracts it to `desc-aux_timeseries.tsv.gz` first, at the rate it was recorded and with the recorded timestamps. The regression resamples it onto the data's time axis with an anti-alias filter and band-limits it to the same bandpass the data went through, so regressors and data sit in one frequency band
- `--aux-channels` picks individual aux channels by name, for an aux group that also holds something that is not a confound

### Fixed
- **The provenance diagram drew steps that had not been run.** A sidecar whose output file was deleted, which is what switching a tree from one mode to another leaves behind, was read as an ordinary node. Those nodes are drawn dashed and labelled `file missing` instead of being believed
- **`fnirs-prep crop` dropped the aux group.** Cropping writes through MNE, which cannot carry an aux channel, so every cropped recording reached preprocessing with no accelerometers and `--aux-regressors` had nothing to regress. The segment's aux is cut from the source and written back at the rate it was recorded
- **`hyper-post` excluded the wrong run's bad channels.** A subject with five tasks had four of them analysed with a fifth task's rejections
- **A group member with two sessions, or two runs of one task, had one silently analysed and the other dropped.** The ambiguity is refused now, naming the CSV column that resolves it
- **A missing task could be analysed as a different one.** Looking up a processing stage fell back to any task of that subject when the requested one had no file
- Per-channel metrics CSVs are written for every run, not only the last one processed
- A `--task-label` run no longer rewrites the quality records of the tasks it did not process
- `fnirs-qc group-hyper-wtc` refuses to merge crossed and homologous channel tables, as it already did for ROI tables
- The provenance diagram is drawn per run. A subject with five tasks got five identical-looking chains in one figure
- `--mode glm` warns when the data is high-passed but the drift model does not span that band. Task betas were underestimated with nothing saying so, mildly for short blocks and severely for long ones

## [0.23.0] - 2026-08-30

### Added
- `--mode denoise --fc` writes the connectivity products rest mode writes, taken from the confound residual, or from the bandpassed data itself when no regression was asked for. Bandpass then correlate had no route through the package that did not also run a GLM and write an ALFF nobody asked for
- `fnirs-qc group-hyper-wtc` merges every dyad's `hyper-wtc.tsv` and `hyper-wtc-roi.tsv` into one long table carrying `group_id` and `task`, so a study with many dyads or many conditions has a single file to take into stats. It refuses to merge tables averaged over different frequency bands rather than warning: once the rows are concatenated, nothing downstream can tell which band a row used
- A **QC Reports page** in the GUI, covering the `fnirs-qc` layer that was command line only: hyperscanning post-analysis, the three group aggregations, provenance graphs and the windowed drill-down. The report each run produces is shown inline on the page
- The Analysis page's pipeline diagram switches to the **real provenance graph** once a run has written sidecars, instead of always drawing the same schematic of the configured modes
- The GUI's Analysis page offers the drift model, its cutoff and order, the ROI mapping and the connectivity flag, and its dropdowns now list exactly the choices the CLI accepts. The HRF list was two models short of the CLI's seven

### Fixed
- **The GUI could never produce a working `--mode glm` or `--mode rest` command.** Both require a drift model and the page had no control for one, so either choice built a command that stopped in postprocessing. Anything beyond a plain bandpass had to be run from the CLI by hand
- The GUI offers short-channel regression in every mode rather than GLM alone. rest has always honoured it, and denoise since 0.22.0
- The GUI's Stim Duration box was drawn but never read, so no task duration ever reached a command
- The subject report's navigation bar links to the FC and GLM panels whatever mode drew them. A `--mode glm --fc` run drew the connectivity panel with no way to reach it from the top of the page
- The reproduction script written to `logs/sub-<id>_script.py` named the residual step differently from the pipeline it mirrors, so the sidecars it produced had no method prose and their provenance went unrecognised

## [0.22.0] - 2026-08-30

### Added
- `--mode glm --fc` writes the same connectivity products rest mode writes, from the GLM residual. The task sits in the design matrix, so what correlates is what the model did not explain, which is what makes it connectivity rather than a map of who responded to the same stimulus
- The subject report draws the ROI-to-ROI connectivity matrix. Those numbers were already written to `_fcroi.tsv` and nothing displayed them
- The subject report draws ALFF and fALFF on the optode layout, next to the per-channel bars. Low-frequency amplitude is a spatial claim, and a bar chart ordered by channel name cannot be read as one
- `fnirs-qc hyper-post` writes `group-<id>_task-<task>_hyper-isc-<chromophore>.tsv`, the inter-brain correlation matrix behind the ISC panel. It used to exist only as a picture, so the numbers could not be taken into a group analysis
- `--mode denoise` honours `--short-channel` (and `--drift-model`): the confound regression runs after the bandpass and the residual is written as `desc-errts`. No task model, no ALFF, no FC, no second broadband regression. Task data whose systemic physiology has to go without the task being modelled no longer has to borrow `--mode rest` and throw half its output away. `--drift-model` stays optional here, unlike in `rest`
- The quality record gains a `motion_post` section: GVTD, spikes, SCI and PSP measured again on the motion-corrected file. Set against the same keys in `raw`, it says whether the correction reduced motion and whether it cost any cardiac signal
- The quality record gains a `windowed` section: the per-window SCI, PSP and GVTD series, with the window length they were binned on. They used to exist only in what `fnirs-qc prep-raw` wrote
- Metric tooltips in the subject report now say which processing stage the number was measured on, which matters for the keys that appear in two sections
- `fnirs-recon --optode-frame` names the space the SNIRF's optode coordinates were measured in. SNIRF does not record it, and without it mne-bids writes no `_optodes.tsv` or `_coordsystem.json`, both of which BIDS requires
- `fnirs-prep crop` accepts a `task` column in the segments table: each segment is written under that task entity instead of `_seg-NN`, so a recording holding several conditions becomes a BIDS dataset the pipeline can read back one condition at a time
- `fnirs-qc hyper-post --wtc-roi-cross` crosses the two brains' ROIs instead of pairing each with its counterpart, so one person's PFC can be tested against the other's TPJ. `hyper-wtc-roi.tsv` gains a `label2` column, the report an ROI × ROI matrix. Needs `--roi-mapping`
- The Methods paragraph now describes the confound regression, naming the short-channel strategy and the drift basis it actually used. `rest` and `denoise` runs used to skip straight from the filter sentence to the references
- `fnirs-qc hyper-post --wtc-mc-count` sets how many surrogate series stand behind each significance contour (default 300, unchanged). It is what the runtime is spent on and it scales with the value, so a run can be previewed cheaply and settled expensively

### Changed
- `snr_pass_rate` counts every channel, not only the ones with a finite SNR. A flat or saturated channel has no SNR at all and used to drop out of the fraction entirely, so a recording whose channels were dying read as one whose channels were passing. The new `n_flat_channels` says how many those are. **`snr_pass_rate` falls for any run that has them**
- The subject report reads the spike and motion-correction spans from the quality record instead of detecting them a second time, and reads the optical density either side of the correction back from disk. Prep no longer keeps a copy of it in memory
- The subject report's per-window SCI and PSP panel reads the quality record instead of values passed from prep, and the series are measured on the motion-corrected file. Prep no longer computes them
- Tables you hand the package read by extension: `.tsv` tab-separated, `.csv` comma-separated, anything else sniffed. Applies to events, segments, the pairs file and `participants.tsv`. What the package writes stays tab-separated
- zALFF standardizes by the population SD (n) instead of the sample SD (n-1). Values grow by `sqrt(n/(n-1))`.

### Fixed
- Accented author names reach the Methods text as letters rather than LaTeX, so the reference list reads `Yücel` instead of `Y{\"u}cel`
- `--config` takes flat top-level keys, and that is now what the reference and the presets show. `configs/presets/denoise_standard.toml` used `[workflow]` / `[denoising]` sections, which the loader never reads, so the whole file was silently ignored. The contrast file is flat too: a `[contrasts]` section hid every contrast in it
- `fnirs-prep crop`, `fnirs-prep edit-markers apply`, `fnirs-prep align` and their two GUI equivalents wrote SNIRF through an MNE export format that does not exist, so **none of them ever produced a file**. They now use the same writer as the pipeline
- Writing a cropped recording put its markers at their position in the original recording, so a segment taken from late in a run lost every marker to the reader's range check. Markers now start where the segment does, in the SNIRF and in the `_events.tsv` beside it
- `crop` names each segment's sidecars after the segment, not after the file it was cut from, and finds `_optodes.tsv` / `_coordsystem.json`, which sit at subject level and were never matched
- `crop`, `edit-markers` and `align` carry `participants.tsv` and `README` into their output, which is what makes those directories readable as BIDS datasets on their own

## [0.21.0] - 2026-08-24

### Added
- Topographic maps of the condition-averaged response in the subject report, at time points after onset, one row per condition and chromophore. The first spatial view of activity that does not wait for the GLM
- `fnirs-qc hyper-post --desc` picks which per-subject stage the inter-brain metrics read (default `preproc`; `errts` reads the confound-regression residual, so short-channel regression can precede a coherence analysis). Optical-density stages are refused
- `fnirs-qc hyper-post` writes `group-<id>_task-<task>_hyper-wtc.tsv` (and `hyper-wtc-roi.tsv` with `--roi-mapping`): one coherence value per pair and channel, averaged over the requested band inside the cone of influence. The figures and a group analysis now read the same numbers
- Rest mode writes `desc-<chromo>_fcseed.tsv` and `_fcseedz.tsv` when `--roi-mapping` is given: each ROI's mean signal against every channel. Cells for a seed's own channels are blank
- The subject report draws the seed maps as flat maps, one per seed ROI and chromophore, each channel coloured by its correlation with that seed. Channels inside the seed are grey rather than zero-coloured, rejected channels are faded, and short channels are left out

### Fixed
- ISC, band coherence and windowed coherence pair channels by S-D label instead of by position, so a dyad whose members had different channels rejected is no longer compared off-by-one. **These values change for any such dyad**
- The `raw_long` and `raw_short` quality sections keep rejected channels, which used to make their mean SCI and channel retention read better than the run was
- Provenance is no longer empty when the pipeline is handed a recording it did not read from disk
- A run where no channel passes the SCI threshold now stops there and says so, instead of failing later inside Beer-Lambert
- `--motion-correction wavelet` sets its outlier threshold per wavelet scale over the whole recording, not per time window
- Rest mode no longer writes ALFF/fALFF when the drift model leaves linear drift in the data (`--drift-model none`, or `polynomial` with `--drift-order 0`); that drift's leakage falls inside the ALFF band
- The windowed SCI and PSP heatmaps plotted each window at roughly half its true time, so the report's time axis covered only the first half of the recording
- GVTD per-window series share the time axis of the windowed SCI and PSP instead of drifting apart from it over a run
- A cardiac band the filter cannot use no longer takes the GVTD per-window series down with the SCI and PSP ones
- A metrics database written by an older version gains any column it is missing instead of failing every insert
- Fisher z no longer zeroes the leading diagonal of a non-square matrix, where the cells are ordinary values rather than self-correlations
- The seed-map sidecar lists the channels each seed was built from, not the ones the ROI mapping asked for

### Changed
- The subject report's trial images are split by condition, with the pooled image kept first. Stacking every condition's trials together hides a response only one condition drives. They were called erpimages; the figure files are now `trialimage_*.html`
- The per-channel layout figure shows the condition-averaged response, not the continuous signal. Runs without events keep the continuous view
- The hyperscanning WTC, coherence and ISC read long channels only. Short channels carry scalp physiology that two people in one room share whatever their brains are doing. **Any montage with short channels loses those rows from its hyper figures**
- Inter-brain metrics stop when the members of a group were sampled at different rates, instead of applying the first member's rate to everyone
- Functional connectivity is plain Pearson, not the shrinkage estimate inherited from a library default, which shrank hardest for subjects with shorter runs or more rejected channels. **Every FC and FCZ value changes; weak connections change most**
- Quality records store `qc_window_s`, so a group report whose subjects were run at different `--window-length` values says so instead of stacking incompatible rows in one heatmap
- `Mean PSP` is labelled with its 10 s window in the report. The window is pinned there and does not follow `--window-length`, which still sets the windowed PSP heatmap
- zALFF standardizes by the sample standard deviation. Values shrink by sqrt((n-1)/n), where n counts the channels of one chromophore: 5.1% at 10 per chromophore, 2.6% at 20
- `--motion-correction wavelet` reaches artifacts up to about 25 s long; it used to stop at about 1.6 s

## [0.20.0] - 2026-08-11

### Added
- `<sub>_<task>_desc-sqm_nirs.json`: one quality record per run, grouped into sections by what each metric was measured on (`raw`, `raw_long`, `raw_short`, `motion`, `preproc`, `final`). A subject with several tasks gets one record per task. The `raw*` sections include rejected channels, the rest exclude them
- Sidecars record per-channel SCI and the respiration band, so a record can be rebuilt from an output directory alone
- The QC report's metrics panel explains itself: hover any metric for what it is and which way is good, and the five that decide whether a run is usable are marked. The explanations used to live only in a separate guide

### Changed
- Quality metrics are measured once and written once. They used to be computed three to four times per subject, and the number shown in the report did not match the number on disk
- `sub-XX_sqm.toml` and `sub-XX_sqm_raw.toml` are gone, replaced by the per-run JSON. Group aggregation now sees pipeline runs, not only `fnirs-qc prep-raw` output
- The metrics database stores one row per section and fills in the task
- `fnirs-qc prep-raw` writes `<sub>_<task>_desc-sqmraw_nirs.json`. It used to write `desc-sqm`, the same name the pipeline uses, so running both into one output directory left whichever finished last. The group report reads both and prefers the pipeline's record for a run that has both

### Fixed
- The resting-state FC heatmap and connectogram lost their HbR half, or failed to render at all
- A single failing metric no longer discards a run's whole quality record; only the section it belongs to is lost
- The QC report says so when a run's quality record is unreadable, instead of showing an empty metrics panel
- The provenance table lists the metric names again for the quality record
- `<sub>_channel_metrics.csv` gains the run's task and session, so a subject with several tasks keeps one file per task instead of only the last
- Short and long channels are decided the same way everywhere. The QC report, the `prep-raw` figures and the quality record each used a different separation cutoff, so the same channel could be short in one and long in another
- Cardiac Power was silently unavailable on any recording with a rejected channel: `cp_mean`, `cp_per_channel` and `cp_pass_rate` all came back empty
- Peak spectral power is measured on optical density, matching the windowed PSP series shown beside it in the report. The two were measured on different signals; values shift in the fourth decimal
- Rebuilding a quality record from an existing derivatives tree lost every SCI metric when the sidecar carried no stored per-channel scores
- The group report's boxplots grouped every metric under "Other". Grouping matched bare metric names and the columns had gained their section prefix
- Per-wavelength CV (`cv_mean_760` and the like) vanished from a record whenever the CV/SNR step failed, instead of being reported as missing like every other metric
- A derivatives tree that has been moved, or is read on another machine, keeps its raw-signal quality metrics. They were silently skipped because the sidecars name the original recording by an absolute path that no longer resolved

## [0.19.0] - 2026-07-31

### Added
- `fnirs-qc hyper-post --wtc-seed` makes the wavelet-coherence significance contour reproducible; without it the Monte Carlo surrogates, and pycwt's on-disk cache of them, made two runs on the same data disagree
- Sidecars record the shape of the data each step left behind: channel count, how many were marked bad, sampling rate and duration
- The provenance diagram shows the settings each step used (filter band, dpf, motion-correction method, GLM models) and the data shape at every node, instead of the step name alone
- `*_alff.tsv` and `*_glm_results.csv` gain a `bad` column, and connectivity, ALFF and GLM sidecars list the rejected channels: results for rejected channels are kept and flagged rather than dropped, so the table shape stays predictable for group analysis
- The SQM checkpoints appear in the provenance diagram; the raw one points back at the original recording it measured, not at the file it happens to sit beside

### Changed
- The provenance diagram moves from `sub-XX/logs/` to `sub-XX/figures/provenance.png` and is shown in the QC report; `fnirs-qc provenance` writes to the same place, so re-rendering refreshes the diagram an existing report displays
- The two SQM checkpoints name the metric families they computed, so the diagram distinguishes the raw checkpoint from the thinner one post-processing leaves behind, instead of showing both as `sqm`
- The QC report's Provenance section lists every output with the step that made it and a line saying what that step did; the SQM rows expand to the full list of metric names each checkpoint recorded
- The Methods paragraph is built from the sidecars the run wrote rather than from the configuration, so it describes what actually ran
- The provenance diagram renders at 300 dpi, matching the other report figures
- Per-subject log, run record, script and provenance diagram lose the timestamp in their names (`sub-01.log`, `sub-01.toml`); a re-run overwrites them
- The 3-view brain figure colours the source-detector links by SCI, like the flat map beside it, instead of spheres at channel midpoints; the surface is opaque so the far side of the head no longer shows through, and optodes are red (source) / blue (detector)
- Writing two different files under one pipeline stage now fails instead of silently crediting an output to the wrong source
- Postprocessing now rejects a `desc-` that disagrees with the stage stamped on the data, matching the check preprocessing already had
- GLM outputs carry the subject and task of their input: `sub-01_task-tapping_design_matrix.csv`, and likewise for `glm_results.csv` and `contrasts.csv`

### Fixed
- A subject whose short channels were all rejected for one chromophore crashed the GLM instead of continuing without short-channel regressors
- `--drift-model cosine` without `--drift-high-pass` now says so before the run starts, instead of failing minutes later inside nilearn with a `NoneType` multiplication error
- The Methods paragraph stopped after the Beer-Lambert sentence: filtering, resampling and the GLM were never described, because the report built the text without the post-processing configuration
- Channels marked bad during preprocessing were unmarked again as soon as postprocessing reloaded the data, so filtering, resampling and the GLM all treated SCI-rejected channels as good; the marks now travel with the file
- A short channel rejected by SCI was still averaged into the short-channel regressor, and that regressor sits in the design matrix, so one bad channel shifted the fit of every channel
- mALFF and zALFF were standardized against a mean and standard deviation taken over all channels, rejected ones included, distorting the value of every good channel
- ROI connectivity averaged rejected channels into their ROI signal, carrying them into every correlation that ROI took part in
- Brain figure drew each channel from its midpoint to its source instead of source to detector
- Design matrix labels overlapped when regressors were numerous
- Postprocessing ran once per matching subject found anywhere under the output directory, so a nested output tree from an earlier run was silently reprocessed and its results overwrote the real ones
- With more than one task per subject, every task wrote the same `design_matrix.csv` and `glm_results.csv`, leaving only the last

## [0.18.0] - 2026-07-28

### Added
- Sidecars now record the file each output came from (`Sources`) and the step that produced it
- ALFF, connectivity, GLM and hyperscanning outputs now have sidecars
- Rest mode writes `desc-filtered` and `desc-errtsbroad`, previously discarded
- Each run writes a provenance flow diagram (PNG + mermaid) to `sub-XX/logs/`, showing which file every output came from and via which step
- `fnirs-qc provenance <output_dir>` renders that diagram for any past run, reading only the sidecars already on disk

### Changed
- Cardiac/respiration band power and low-frequency drift are now written only to the preprocessing checkpoint; on filtered data they describe the filter, not the recording

### Removed
- Durbin–Watson on GLM residuals: ~2 by construction once prewhitening runs

### Fixed
- Cardiac/respiration band power was silently missing on recordings with bad channels
- Rest mode overwrote its SQM file, losing the metrics computed earlier in the run

## [0.17.0] - 2026-07-28

### Added
- Standardized ALFF outputs `malff` (channel value over the chromophore mean) and `zalff` (per-chromophore z-score) in `*_alff.tsv`, for group-level comparison

### Changed
- The per-subject reproducible script (`sub-XX/logs/*_script.py`) is now a step-by-step transcript of the run: one block per processing stage (OD, SCI, motion, Beer-Lambert, filter, GLM/rest) with its parameters and outputs, instead of an opaque call into the pipeline
- ALFF/fALFF are now computed on a broadband residual (drift removed, not band-pass filtered) so fALFF spans the full spectrum as defined; functional connectivity keeps the band-pass filtered residual
- Functional connectivity (channel `*_fc.tsv`, ROI `*_fcroi.tsv`, and their Fisher-z variants) is now written per chromophore as separate HbO/HbR matrices (`desc-hbo` / `desc-hbr`) instead of one matrix mixing both; HbO and HbR anti-correlate, so a mixed matrix was not meaningful
- Cardiac and respiration band power/fraction are now reported per chromophore (`*_hbo` / `*_hbr`)

### Fixed
- fALFF was close to 1 on nearly every channel because it was computed on band-pass filtered data, leaving no out-of-band power in its denominator; it is now valid
- Subject report showed a blank low-frequency drift (HbO) value due to a stale metric key

## [0.16.0] - 2026-07-08

### Added
- Motion-correction footprint (experimental): which timepoints a motion correction actually repaired, summarised like framewise motion (`motion_corrected_pct` / `_num` / `_n_segments`) and drawn as a coloured band beneath the carpet and per-channel motion figures
- Spike frame metrics: fraction of all channel-samples flagged (`spike_pct`) and fraction of timepoints where many channels spike together (`spike_pct_frames`), alongside the existing spike count; spikes are also drawn as a band on the motion figures
- erpimage in the subject report for task data: single-trial HbO heatmaps per channel and per ROI (each stimulus repetition is a row, with a trial-average trace below), computed on the denoised signal and smoothed across trials
- Fisher z-transformed functional connectivity output (`*_fcz.tsv`, ROI `*_fcroiz.tsv`) for group-level statistics, alongside the raw correlation matrices

### Changed
- Windowed SCI/PSP/GVTD now share one sliding-window length (default 10 s, previously a mix of 30 s / 10 s), exposed as `--window-length` on `fnirs-prep`, `fnirs-qc prep-raw`, and `fnirs-qc window-raw`
- Spike detection now runs on the motion-band-filtered optical density (cardiac removed first), so the count reflects motion rather than cardiac pulsation
- Global correlation is now reported before → after the short-channel regression (the step that removes global/systemic signal) instead of before/after the bandpass, which inflated it; a single value is shown when no short-channel regression ran
- The GVTD carpet figure splits the raw and filtered traces into separate panels, with the plotted GVTD downsampled by max-pooling so it stays readable (metric values unchanged)
- Global correlation and spike metrics are now marked experimental
- QC figures render at ≥300 dpi, and the before/after denoising carpet no longer has heavy panel borders

### Removed
- tSNR (haemoglobin mean/std): ΔHbO/ΔHbR has no stable baseline, so its mean/std is not a meaningful signal-to-noise ratio; raw-intensity SNR already covers channel signal stability

## [0.15.0] - 2026-07-06

### Added
- Wavelet motion correction (`--motion-correction wavelet`): zeroes outlier wavelet detail coefficients per channel in OD space
- Filtered GVTD (0.01–0.5 Hz motion band) reported alongside the raw GVTD, plus per-run motion summaries above an adaptive (histogram-mode) threshold: number of motion frames, percent of run, and the threshold value
- Motion figures (carpet + per-channel detail) overlay the raw and filtered GVTD traces and draw the motion threshold; group reports gain a filtered-GVTD time × subject heatmap
- Cardiac and respiration band power are also reported as a fraction of total spectral power (`cardiac_band_frac` / `resp_band_frac`), comparable across subjects regardless of overall amplitude
- `fnirs-qc epoch`: per-trial QC — one window per task event (read from the SNIRF or a `--events-csv`), recomputes SQM per trial and renders a trial × metric TSV + HTML. `--mode epoch|duration` is required (epoch mode needs `--tmin`/`--tmax`); SQM values are reported per trial without a good/bad label
- Channel-standardized GVTD (`gvtd_vstd_*`): each channel's temporal derivative is divided by its own SD before the cross-channel RMS, so high-dynamic-range channels no longer dominate the motion index
- Global correlation per chromophore (`gcor_hbo` / `gcor_hbr`): mean pairwise channel correlation; the subject report shows it before → after denoising (a drop means systemic/global signal was removed)
- Durbin–Watson on GLM residuals (`durbin_watson_mean`): residual autocorrelation check (~2 = white residuals); shown in the GLM report section and saved with the post-processing SQM
- Per-window GVTD now reports p95 (worst-moment) alongside the mean, so a brief motion burst is not averaged away; group reports gain p95 time × subject heatmaps and the subject report shows the GVTD p95 scalar

### Changed
- The cardiac band (`--cardiac-l-freq` / `--cardiac-h-freq`) is now required, with no default: it is population-dependent (adult vs infant heart rate) and now consistently sets the band for channel SCI, PSP, Cardiac Power, and cardiac band power, so a non-adult band is no longer silently ignored by some metrics
- The respiration band (`--resp-l-freq` / `--resp-h-freq`) is now a required option too (population-dependent, same rationale as the cardiac band)
- Spectral confound metrics renamed `residual_cardiac_power` / `residual_resp_power` → `cardiac_band_power` / `resp_band_power` (they measure band power present, not a post-filter residual)
- Spike count now uses a robust (MAD-based) threshold, so a few large spikes no longer inflate the threshold and hide themselves
- Cardiac Power is marked experimental (overlaps PSP; may be removed); SCI and PSP remain the primary channel-quality metrics
- Group report distributions are now split into per-scale charts (coupling, GVTD amplitude, counts, drift, etc.) showing real values with one colour per metric; clicking a point opens that subject's report

### Fixed
- fALFF is now the fraction of total spectral amplitude in the low band (sum/sum, in [0, 1]), matching its definition; was previously a ratio of mean amplitudes. ALFF unchanged (validated against a reference implementation).
- Cardiac Power now sums band power (was averaging) and is computed on optical density, so it lies in [0, 1] and matches the CP ≥ 0.5 quality gate
- `pct_data_retained` no longer double-counts overlapping bad segments
- Low-frequency drift amplitude is now estimated from a polynomial trend fit, avoiding the filter edge artifacts of the previous 0.01 Hz low-pass

---

## [0.14.0] - 2026-07-04

### Added
- ROI-level WTC in the hyperscanning post report: coherence on HbO averaged within each ROI from `--roi-mapping`, excluding the union of all subjects' bad channels so both use the same channel set
- Hyper post report marks bad channels: ISC matrix blanks bad rows/columns and drops their connectogram arcs; per-channel WTC selector tags bad channels
- Denoising before/after carpet in the subject QC report: HbO and HbR shown separately, both scaled by the pre-denoising SD so reduced fluctuation renders paler
- `--roi-mapping` on the main pipeline: a JSON file mapping ROI labels to channel lists, used to group the denoising carpet by ROI and to compute ROI-level connectivity
- ROI-level functional connectivity (`_fcroi.tsv`) in rest mode when `--roi-mapping` is given: each ROI's HbO channels are averaged, then correlated between ROIs; channel-level FC is unchanged

### Changed
- `--exclude-channels` renamed to `--bad-channels`: channels are now marked bad (kept in the data, unioned with SCI-detected bads) instead of being dropped
- Quality metrics renamed from IQM to SQM (signal quality metrics): sidecars are now `sub-<id>_sqm.toml` and `sub-<id>_sqm_raw.toml`
- Motion QC carpet now shows all channels (both wavelengths) instead of a single wavelength

### Fixed
- PSD before/after panel and the run-record database now reflect bandpass cutoffs set in a TOML config file; previously only command-line `--high-pass` / `--low-pass` were read, so a TOML-configured filter showed as no filter
- GVTD trace in the motion carpet is now computed at full resolution instead of on the down-sampled display data, so its spikes and p95 line match the reported GVTD metric

### Known Issues
- `compute_alff` numerical validation pending

---

## [0.13.0] - 2026-07-03

### Changed
- All CLIs reworked to follow the BIDS App convention: `--participant-label` and other list options now accept space-separated values (e.g. `--participant-label 01 02 03`) as well as repeated flags
- `--help` output is grouped into labelled sections

---

## [0.12.0] - 2026-07-03

### Added
- New GUI "Recon" page: detect raw snirf files in a folder, assign BIDS metadata per file in a table, preview the commands, and batch-convert to BIDS
- Command preview (Analysis and Recon pages) has a shell selector so line-continuation matches bash / cmd / PowerShell
- Pipeline auto-detects already-OD input, skips the OD-conversion step, and marks intensity-only QC metrics as unavailable

### Changed
- `fnirs-recon --subject` help clarifies that labels are alphanumeric and can encode group, e.g. `patient01`

### Fixed
- GUI "generate command" emitted invalid multi-subject commands; participant/session/task labels now repeat the flag per value
- QC no longer crashes on low sampling-rate data — PSD and cardiac-power frequency limits clamp to the Nyquist frequency
- QC brain views no longer fail when the static-image export backend was missing (now a hard dependency)

---

## [0.11.0] - 2026-05-25

### Added
- Self-contained rating viewers: rating bar inlined into the QC HTML template; static open works read-only, `fnirs-rate` open enables writes
- Section ratings + channel decisions persist via `/save_*` REST endpoints only — HTML never modified

### Changed
- `HyperRatingApp` / `RawRatingApp` / `FNIRSRatingApp` refactored: no more regex reverse-parsing or string-concat injection

### Fixed
- `HyperRatingApp` channel-decisions sidecar is now session-aware

---

## [0.10.0] - 2026-05-30

### Added
- `fnirs-qc group-raw` and `group-hyper-raw` aggregate per-subject / per-dyad SQM into cohort HTML reports (heatmap, boxplots, outliers, sortable table)
- `fnirs-qc window-raw` crops each subject's raw to a time window and re-runs the group-raw layout — for "is this minute dropping for everyone?"
- prep-raw SQM JSON now stores `sci_per_window` / `psp_per_window` / `gvtd_per_window`; group reports render time × subject heatmaps for these

### Changed
- QC output directory follows a BIDS-derivatives layout: per-subject / per-group `figures/` and `[ses-XX/]nirs/` subdirs
- QC filenames standardised to `_nirs.*` suffix with BIDS `desc` entity
- Hyper raw report converted to an iframe shell with auto-resize; Plotly loaded from CDN

### Fixed
- `fnirs-qc` CLI now initialises logging

---

## [0.8.0] - 2026-05-23

### Added
- `fnirs-log merge` CLI + `utils/job_db.py`: JSONL run events consolidated into a SQLite database (`pipeline_executions`, `runs`, `sqm`, `command_outputs`)
- `PrepResult` exposes `sqm_raw` and `sqm_final` for downstream use
- Paragraph-style Methods boilerplate with a "Rendered" tab in the HTML report

### Removed
- `--mode connectivity` (was never implemented)
- `--ica` postprocessing flag (ICA for fNIRS is experimental; removed to simplify the post pipeline)
- `--segments-path` / `--crop-tmin` / `--crop-tmax` on `fnirs-pipe` (superseded by `fnirs-prep crop`)

---

## [0.7.0] - 2026-05-14

### Added
- `fnirs-gui` CLI entry point launches a Dash multi-page application
- Data Preparation page: SNIRF file loader, editable stimulus marker table, per-subject SQM display
- Analysis page skeleton wired into the sidebar router

---

## [0.6.0] - 2026-05-13

### Added
- Resting-state analysis mode added to the hyperscanning pipeline (`--mode rest`)
- ALFF and fALFF computation (`pipeline/restingstate.py`)
- Resting-state functional connectivity (Pearson correlation matrix across channels)
- Resting-state QC figures and report section
- `--trim` option for hyperscanning pipeline to discard leading/trailing seconds before analysis

---

## [0.5.0] - 2026-05-13

### Added
- ISC (inter-subject correlation) computation and heatmap visualization in hyperscanning post report
- WTC (wavelet transform coherence) computation and figure builders
- Connectogram visualization for hyperscanning functional connectivity (`qc/figures/connectogram.py`)
- Hyperscanning post-processing QC report (`qc/hyper_report.py`) with WTC, ISC, and connectogram sections
- Hyperscanning group-level raw QC report: Plotly figure builders for multi-dyad channel summaries
- Normalization option for hyperscanning raw data; `z_score` utility added

### Changed
- Hyperscanning functionality consolidated from a standalone CLI into `pipeline/hyperscanning.py`; `fnirs-hyper` CLI removed and subsumed under `fnirs-qc`

---

## [0.4.0] - 2026-05-12

### Added
- `fnirs-qc` CLI (`cli/qc.py`) with `prep-raw` and hyperscanning subcommands
- Interactive raw QC viewer: standalone HTML page with Plotly channel traces (`qc/app.py`)
- SQM expansion: cardiac power band metrics and tSNR per channel (`qc/quantitative_metrics.py`)
- Channel quality summary figure integrated into prep-raw report and templates

### Changed
- SCI computation corrected; figure axis labels clarified across multiple panels

### Fixed
- Empty epoch subsets no longer crash `build_channel_figure`
- Unused imports and variables cleaned up across figure modules

---

## [0.3.0] - 2026-05-05

### Added
- Task GLM postprocessing via nilearn (`pipeline/glm.py`): design matrix construction, contrast estimation
- Denoise mode: bandpass filter + optional resampling (`pipeline/denoise.py`)
- GLM QC section in per-subject report: design matrix timeseries figure + nilearn activation heatmap
- Raw residuals written as a separate SNIRF file after GLM fitting
- Motion correction figures improved: bad-segment zoom panel now runs independently of the motion carpet

### Fixed
- `_section_motion` bad-segment zoom was incorrectly nested inside the motion panel's `except` block; now executes unconditionally after the main panel

---

## [0.2.0] - 2026-05-05

### Added
- Per-subject HTML QC report generated via Jinja2 + Plotly (`qc/prep_raw_report.py`)
- Report sections: OD traces, SCI/PSP heatmaps, GVTD timeseries, motion carpet, HbO/HbR correlation panel, PSD, brain views, short-channel PSD, SQM table
- SQM sidecar files: `sub-{id}_sqm.toml` + `sub-{id}_channel_metrics.csv`
- `_guard` context manager for uniform per-section error handling; failed sections render a warning card rather than crashing the report

---

## [0.1.0] - 2026-05-04

### Added
- End-to-end single-subject preprocessing: OD conversion, SCI/PSP channel QC, motion detection and correction (TDDR), Beer-Lambert law
- Pipeline reorganized into `pipeline/` module: `prep_pipeline.py`, `post_pipeline.py`, `glm.py`, `denoise.py`
- BIDS Derivatives sidecar writing; intermediate SNIRF written at each prep step
- `fnirs-pipe` CLI via Typer with all planned preprocessing flags wired
- `fnirs-recon` CLI for raw → BIDS conversion
- Run record and script utilities (`utils/run_record.py`, `utils/run_script.py`)
- Logging setup (`utils/logging.py`)
