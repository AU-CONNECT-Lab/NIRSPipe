---
title: Processing workflow
---

# Processing workflow

`nirspipe ... participant` runs each subject through preprocessing and then, when `--mode` is given, through postprocessing. The order of the steps is fixed: motion correction always runs on optical density, before the bandpass, and the short-channel regression after the bandpass. `nirspipe ... group` gathers the per-run quality records into the cohort pages. Dyad analysis is a separate set of commands, described in [Hyperscanning](hyperscanning.md). Every flag named below is listed in [Usage](usage.md), and every file in [Outputs](outputs.md).

In the diagrams, rounded boxes are files written to `sub-<label>/nirs/`, plain boxes are operations that leave no file of their own, and dashed edges are optional. They show every branch the code can take. What one particular run did is drawn by the provenance graph it writes, `sub-<label>/figures/<run>_desc-provenance_nirs.png`, which [`nirspipe-qc provenance`](usage.md#nirspipe-qc) can regenerate for any past run.

## Preprocessing

Preprocessing turns raw intensity into haemoglobin concentration in five steps that always run in this order. Each step writes a SNIRF with a JSON sidecar recording its settings and its source files. The only branch is the input check: a recording that is already optical density skips the conversion.

```mermaid
flowchart TD
    IN[/"BIDS input<br/>sub-XX_task-YY_nirs.snirf"/]

    IN --> ISOD{"already optical<br/>density?"}
    ISOD -->|no| CONV["intensity → OD<br/><small>od_conversion</small>"]
    ISOD -->|yes| PASS["passthrough<br/><small>od_passthrough</small>"]

    CONV --> OD(["desc-od"])
    PASS --> OD

    OD --> SCI["windowed coupling<br/><small>SCI and PSP per window, cardiac band</small>"]
    MANUAL[/"--bad-channels"/] -.-> MERGE
    SCI --> MERGE["mark bad channels<br/><small>passing windows &lt; min-good-frac ∪ manual</small>"]
    MERGE --> SCIF(["desc-sci"])

    SCIF --> MOT["motion correction<br/><small>tddr | wavelet | tpca | spline | spline_sg | wavelet_tddr | spline_wavelet | none</small>"]
    MOT --> MOTF(["desc-motcorrected"])

    MOTF --> BL["modified Beer-Lambert<br/><small>OD → HbO/HbR, dpf</small>"]
    BL --> PRE(["desc-preproc"])

    IN -.-> AUXX["read the snirf aux group<br/><small>only if the file has one</small>"]
    AUXX -.-> AUX(["desc-aux<br/>timeseries.tsv.gz"])

    PRE ==> POST(["to postprocessing"])
    AUX -.-> POST

    classDef haemo fill:#fadbd8,stroke:#c0392b
    classDef od fill:#fdebd0,stroke:#e67e22
    classDef side fill:#eaeded,stroke:#7f8c8d
    class OD,SCIF,MOTF od
    class PRE,POST haemo
    class AUX side
```

No quality metric is measured inside this diagram. The run's metrics are measured once, after preprocessing and postprocessing have both finished, from the files they left behind; see [QC](qc.md).

### BIDS validation

Before any data is read, the input is checked with bids-validator when one is installed. An error stops the run, naming the error codes and the first files each one names; warnings do not. Without a validator the run goes on with a warning, and `--skip-bids-validation` skips the check. Either way, files BIDS does not recognise are left out when the dataset is indexed.

### Optical density

Raw intensity is converted to optical density, $\Delta OD_\lambda = -\log(I / I_0)$, with MNE-Python, where $I_0$ is the channel's mean. Writes `desc-od`.

Before the conversion, frames at the start or end of the recording in which most channels measured no light (an intensity missing or not above zero) take the values of the nearest frame that did, and are marked `BAD_edge` ([Excluded time](#excluded-time)). Some devices write such a frame when they start or stop. Left in, one missing sample makes the channel's mean, and so its whole optical density, undefined; and the bandpass pads each end by mirroring the signal about its first and last frames, so a dark frame there becomes a step that spreads across the run. The sidecar of `desc-od` records how many frames were held at each end.

### Channel screening

The scalp coupling index (SCI) measures the strength of the cardiac pulsation in the optical density, as a proxy for how well the optode touches the scalp. Peak spectral power (PSP) measures the same pulsation in the frequency domain, computed as QT-NIRS computes it. Both are measured in the cardiac band set by `--cardiac-l-freq` / `--cardiac-h-freq`, which has no default because it depends on the population: choose it around the participants' heart rate, about 1 to 1.7 Hz for adults and 2 to 3 Hz for infants. The band's top edge plus the filter's 0.3 Hz roll-off has to stay below half the sampling rate, and a band that does not is refused with the highest top edge the recording allows.

Screening counts 10 s windows; the window is fixed and does not follow `--window-length`. A window *passes* when the channel's SCI clears `--sci-threshold` in it, and a channel is marked bad when it passes in fewer than `--min-good-frac` (0.75 by default) of the windows that count. PSP is computed in the same windows and drawn in every report without a line. `--psp-threshold` has no default; given, a window has to clear it as well as the SCI line, inside the same window, as QT-NIRS does. Its 0.1 comes from PHOEBE, a tool for fitting the headgear before recording, and its authors have used 0.075 on recorded data; on expert-rated recordings either line rejected far more channels than the raters did. `--screen-scope` says which windows count: `run`, the whole recording, or `task`, only the annotated blocks, so the lead-in and the gaps between blocks are not held against a channel. `task` falls back to `run` when no annotation is long enough to hold two windows. A channel passing for only part of a recording can therefore be rejected even when its whole-run SCI passes. The same criteria are used by every path that screens: the pipeline, the raw QC report, the per-trial scoring and the dyad commands.

A recording screening cannot judge, one with fewer than two windows outside its `BAD_` spans or one whose windowed SCI cannot be measured, stops preprocessing with the reason. The raw QC report and the dyad raw QC still draw it, reject none of its channels on screening, and say why above the channel table.

```{figure} figures/screening_scipsp.png
:alt: Windowed SCI, PSP and CV heatmaps, channels by time
:width: 100%

Windowed SCI, PSP and CV of one recording, channels by time, with each channel's mean and the line it is held to beside the heatmap.
```

Bad channels are **marked, not removed or interpolated**. Three sources are unioned into one set:

- the channels screening rejected;
- the channels named by `--bad-channels`, either as source-detector labels or as a table with optional `participant_id`, `session`, `task`, `acq` and `run` columns ([Data preparation](data-preparation.md#tables-that-name-recordings));
- the channels the recording's BIDS `channels.tsv` marks `bad` in its `status` column. The nearest `channels.tsv` is read, as BIDS inheritance allows, `status` is compared without regard to case, and a channel marked `good` is still screened like any other. `--ignore channel-status` turns this off. The `desc-sci` sidecar records them as `parameters.channels_tsv_bads`, and the report shows them as `BAD (manual)`.

Naming either wavelength of a pair rejects the pair. A pair in the table's `keep_channels` column leaves screening's rejections, not the other two sources; the `desc-sci` sidecar records it as `parameters.kept_channels` and the Methods name it. A kept pair also carries a run in which screening rejected every channel. When the listed channels leave no usable channel, the run stops. A window touching any `BAD_` span is not counted ([Excluded time](#excluded-time)), so the run has one channel set, chosen on the time it keeps. Writes `desc-sci`.

### Motion correction

Motion artifacts are corrected in optical density, before the conversion to haemoglobin. `--motion-correction` picks the method; `tddr` is the default.

| `--motion-correction` | Method | Reference |
|---|---|---|
| `tddr` | Temporal derivative distribution repair, as implemented in MNE-Python | Fishburn et al. (2019) |
| `wavelet` | Wavelet filtering | Molavi and Dumont (2012) |
| `tpca` | Targeted principal component analysis | Yücel et al. (2014) |
| `spline` | Spline interpolation | Scholkmann et al. (2010) |
| `spline_sg` | Spline interpolation followed by Savitzky-Golay smoothing | Jahani et al. (2018) |
| `wavelet_tddr` | `wavelet`, then `tddr` on its output | |
| `spline_wavelet` | `spline`, then `wavelet` on its output | Di Lorenzo et al. (2019) |
| `none` | No correction; `desc-motcorrected` is not written | |

`--motion-std-thresh` and `--wavelet-iqr` set the spline detection threshold and the wavelet fence of the methods that use them; their per-method defaults are in [Usage](usage.md#nirspipe). The detection threshold is measured against each channel's sample-to-sample change, so the same value marks more of a faster recording; when `spline` or `spline_wavelet` marks over 10% of the samples, the log warns and the report adds a note.

The methods differ in how much of the response they keep, not only in how much noise they remove. `tddr` weighs every sample's derivative by a robust weight below one, not only the outlying ones, so a small slow response comes out smaller even in a recording without motion; `wavelet_tddr` and `spline_wavelet` shrink it at least as much, `wavelet` less. Response amplitudes are comparable only between recordings corrected with the same method, so a study comparing conditions or groups should correct all of its data one way.

`spline_sg` ends with a Savitzky-Golay smoothing that low-passes every channel near 0.1 Hz. An AR noise model fitted after it fits the smoothing rather than the noise, as after a `--low-pass`: the GLM's betas hold but its t values come out anti-conservative. Before a GLM with an AR noise model, prefer another method, or use `--noise-model ols` and treat the t values as uncalibrated.

Writes `desc-motcorrected`. The before-and-after comparison the report draws is described in [QC](qc.md#motion-correction-footprint).

```{figure} figures/motion_gvtd_carpet.png
:alt: Motion-band GVTD traces above carpets of the optical density before and after correction
:width: 100%

The motion-band GVTD of the long and the short channels above a carpet of the optical density, before and after correction, on one scale; the dashed line is the run's threshold.
```

### Beer-Lambert law

The corrected optical density is converted to HbO and HbR concentration changes with the modified Beer-Lambert law, $\Delta c = \Delta OD / (\varepsilon \cdot DPF \cdot d)$, where $\varepsilon$ is the molar extinction coefficient and $d$ the source-detector separation. `--dpf` takes one differential pathlength factor for every wavelength, one per wavelength, or `age`, which computes one per wavelength from each participant's age in `participants.tsv` by the general equation of Scholkmann and Wolf (2013), $DPF(\lambda, A) = 223.3 + 0.05624\,A^{0.8493} - 5.723 \times 10^{-7}\lambda^3 + 0.001245\,\lambda^2 - 0.9025\,\lambda$, with $\lambda$ in nm and $A$ in years. The equation is a surface fitted to the frontal-head DPF that Duncan et al. (1995) measured in 283 people from 1 day to 50 years old at 690, 744, 807 and 832 nm, and checked against six other data sets. Its authors recommend it for ages 0 to 70 years and wavelengths 690 to 832 nm, safe on to 950 nm, and for the frontal and frontotemporal head only; outside that range the run goes on, with a warning in the log and a note on the report. At 760 and 850 nm it gives about 5.3 and 4.2 for a newborn, 5.7 and 4.6 at 10 years, 6.2 and 5.1 at 25, and 6.9 and 5.8 at 50. The age is read in years unless `participants.json` gives its `Units` as months, weeks or days, and a participant with no age stops the run before its first subject. The sidecar records the factors used and the age. Under `--dpf age` the head the optodes are placed on follows the age too ([Optode placement](#optode-placement)). Writes `desc-preproc`, the end of preprocessing.

### Auxiliary channels

A SNIRF file may carry non-optical channels in its `aux` group, such as accelerometers, gyroscopes or a pulse trace. MNE does not read them, so preprocessing copies them from the input into `desc-aux_timeseries.tsv.gz`, at the rate they were recorded, with the recorded timestamps in a `time` column. No derivative SNIRF holds them, so this table is the only copy carried forward. The report draws the gyroscope and accelerometer rows on its motion figures from it, and `nirspipe-prep crop` carries the aux group into each segment it cuts.

### Optode placement

SNIRF does not record which space its optode coordinates are in, and a dataset's `_coordsystem.json` is not always right about it, so preprocessing places the optodes on a head template from the recording itself and writes the result once, as `space-<template>_channels.tsv`.

The template is fsaverage, an adult head, unless `--template` names one of MNE's infant templates (O'Reilly et al. 2021, built on the MRI of Richards et al. 2016): `2wk`, `1mo`, `2mo`, `3mo`, `4.5mo`, `6mo`, `7.5mo`, `9mo`, `10.5mo`, `12mo`, `15mo`, `18mo` or `2yr`, each with its own scalp, cortex and 10-10 positions. With `--dpf age` and no `--template`, each participant takes the template nearest their age in `participants.tsv`: the infant template of the closest age up to 2 years, and fsaverage from 18. No template fits between 2 and 18 years, so such a participant stops the run before its first subject, with a message asking for `--template`. Each participant is placed on their own template, so a parent and an infant in one dyad are each drawn on their own head. An infant template is downloaded the first time a run needs it, before the first subject, into the folder MNE keeps fsaverage in (about 630 MB each). On a machine without internet access, run `mne.datasets.fetch_infant_template("<age>", subjects_dir)` on another machine and copy the template's folder over.

When the file carries at least three landmarks named as in the 10-05 system (the nasion and preauricular points count), they are registered onto the same points on the template by a rotation, a translation and one scale, and the result is kept when the optodes then lie within 5 mm of the template's scalp (median over optodes). An infant template carries 10-10 positions only, so other 10-05 names do not count there. Otherwise the coordinates are read in each frame they could be in, and the reading that puts the optodes on the scalp is kept. On fsaverage these are its head frame and its surface RAS. On an infant template they are its own head frame, for positions digitised on the infant, and fsaverage's head frame and surface RAS carried onto the infant head by the affine that best maps fsaverage's 10-10 positions onto the template's, for a cap stored on an adult head, as many acquisition programs export it. A recording that no reading places, such as one whose probe holds a flat 2-D layout, gets `n/a` positions and a warning, its brain figures draw the coordinates as they stand, and a run note on its reports says why it was not placed.

The table holds one row per channel, at the midpoint of its source and detector, in the template's surface RAS (fsaverage's is MNI305) and in mm. Each row also names the cortical region of the pial vertex nearest that point: its Desikan-Killiany region, `aparc` (Desikan et al., 2006), on every template, and its Brodmann area, `brodmann`, from the PALS-B12 atlas (Van Essen, 2005), on fsaverage, which alone ships one; the hemisphere is suffixed, as in `precentral-lh` or `Brodmann.4-lh`, and an unplaced channel has `n/a`. It is the region nearest the channel, not a model of where its light travels. BIDS names fsaverage but no infant template, so an infant table's sidecar gives `NIRSCoordinateSystem` as `Other` and names the template in `NIRSCoordinateSystemDescription`; its `space` label is the MNE template's name with the hyphen written `p` (`ANTS4p5Months3T` for `4.5mo`), since a BIDS label is alphanumeric. The sidecar records which route placed the optodes, how close they came to the scalp, and the transform, and the run's Methods paragraph says the same in a sentence: the landmarks registered, or the frame the coordinates were read in, with the median distance to the scalp. An unplaced run's Methods say nothing about placement. Every later file of the run, the brain figures, the head maps and the dyad tables take the placement from this record, and `nirspipe-prep crop` and `edit-markers` copy the source's landmarks into what they write. A rerun on another template replaces the table, so a run has one. The 3-D views and the cortex projections draw on the template's own pial surface, still spreading each channel within 3 cm, and the head maps draw a head outline shrunk with the template, so a cap sits inside it as it does on fsaverage.

## Postprocessing

`--mode` picks the branch: `denoise`, `glm` or `rest`. The bandpass and the resampling are shared by all three, and each mode fills in its own defaults for the settings below, which `--config` and the command line override ([Mode defaults](usage.md#mode-defaults)). Postprocessing measures no quality metric of its own; the run's quality record gets one section per stage it wrote (`filtered`, `resampled`, `errts`).

```mermaid
flowchart TD
    PRE(["desc-preproc"]) --> BP{"--high-pass /<br/>--low-pass given?"}
    BP -->|yes| BPF["bandpass"]
    BPF --> FILT(["desc-filtered"])
    BP -->|no| RS
    FILT --> RS{"--resample-sfreq<br/>given?"}
    RS -->|yes| RSF["resample"]
    RSF --> RESA(["desc-resampled"])
    RS -->|no| MODE
    RESA --> MODE{"--mode"}

    MODE -->|denoise| DSC{"--short-channel or<br/>--drift-model given?"}
    DSC -->|no| DONE["no regression"]
    DSC -->|yes| DREG["confound regression"]
    DREG --> DERR(["desc-errts"])
    DONE -.->|--fc| DFC["connectivity<br/><small>as in rest mode</small>"]
    DERR -.->|--fc| DFC
    MODE -->|glm| GLM["GLM"]
    MODE -->|rest| REST["confound regression<br/>see Mode: rest"]

    classDef haemo fill:#fadbd8,stroke:#c0392b
    class PRE,FILT,RESA,DERR haemo
```

### Filling corrupted spans

When the recording carries a corrupted `BAD_` span ([Excluded time](#excluded-time)), every channel is filled across it before the bandpass, so the artifact does not spread through the filter into kept time. `--censor-fill` picks the fill: `linear` (the default) joins the kept samples either side, `spline` runs one cubic spline through every kept sample, and `lomb` rebuilds each span from a Lomb-Scargle fit to the kept samples. A span at the edge of a recording takes the nearest kept value under `linear` and `spline`. The spans stay marked and are left out of every later estimate, and `desc-preproc` keeps the measured data.

### Bandpass and resampling

The bandpass uses `--high-pass` and `--low-pass`; `none` switches off a cutoff the mode would fill in. `--filter-method iir` (the default) is a zero-phase Butterworth of order `--filter-order` (4 by default) applied forward and backward, so its rolloff is twice the order and its cutoff sits at -6 dB. `--filter-method fir` is a Hamming-windowed linear-phase filter, refused when its length exceeds the recording. Writes `desc-filtered`. `--resample-sfreq` then resamples, writing `desc-resampled`.

:::{warning}
A low-pass, and a resampling, which anti-aliases at the new Nyquist frequency, both undermine prewhitening: on a low-passed series an autoregressive `--noise-model` describes the filter rather than the noise, and the residual stays autocorrelated. In a task GLM the betas hold but the standard errors come out too small. In `rest` or `denoise` the residual `desc-errts` itself changes. Every mode warns about it.
:::

Every filtered stage also opens and closes with the filter settling rather than with anything measured, the more so the lower the cutoff; the report notes how far that reaches into each end.

### Mode: denoise

Applies the bandpass and resampling and stops there, for analyses done outside the package. Given `--short-channel` or `--drift-model`, it also regresses those confounds out of the bandpassed signal, with no task model, and writes the residual as `desc-errts`. `--drift-model` is optional here, since the bandpass already detrends. With `--fc` it writes the connectivity products of rest mode, from the residual or, when no regression ran, from the bandpassed data.

### Mode: glm

A first-level GLM, built with nilearn through MNE-NIRS. It writes the residual as `desc-errts`, the design matrix as `design.tsv`, the fit as `desc-glm_nirsmap.tsv` and each condition's block average as `desc-blockavg_timeseries.tsv`.

**Events.** The design is built from the SNIRF annotations with every event given the duration `--stim-dur`, or from a BIDS `_events.tsv` named by `--events-path` with its own onsets and durations. One of the two is required, and they are mutually exclusive. Every duration must be above 0: nilearn models a zero-duration event as a single sample, so its beta would grow with the sampling rate and would not be in concentration units, and the run refuses such an event. Give each event its real length; a button press can take 0.1 s.

**Block average.** Each condition's trials averaged per channel, on the series the model is fitted to, over `--epoch-tmin` to `--epoch-tmax` (-5 to 25 s when neither is given, the report's epoch window), each trial less its mean before onset. The trials are the design's, so the `--events-path` table when one is given, cut by `--epoch-chunk-duration` when set; an event annotated twice (same condition and onset) is one trial, as in the design, and two conditions starting on one sample each keep it. A trial whose window runs off the recording or touches a `BAD_` span is left out. The table gives, per condition, channel, chromophore and time, the mean, its standard error and the number of trials, with rejected channels flagged as in the GLM tables. Unlike the report's epoch figures, it is not low-passed for display.

**Design.** `--hrf-model` sets the HRF basis (`spm` by default; `glover`, the derivative and dispersion variants, or `fir` with `--fir-delays`). `--drift-model` sets the drift basis (`cosine` by default, or `polynomial` of order `--drift-order`, or `none`).

```{figure} figures/glm_design.png
:alt: A GLM design matrix drawn as an image
:width: 100%

A design matrix as the report draws it: the condition regressors, the short-channel regressors, the cosine drift basis and the constant, one row per sample.
```

**Cosine drift cutoff.** `--drift-high-pass` has no default in this mode, because it is bounded by the design. The cosine basis spans every frequency below its cutoff, so the cutoff has to sit at or above the data's `--high-pass`, so the task regressors are detrended the way the data was, and below the rate at which a condition repeats, or the basis fits that condition away as drift. A condition whose trials are at most `T` seconds apart repeats at `1/T` Hz, and `1/(2T)` is a cutoff that clears it. The run warns when either bound is crossed, naming each condition the basis would absorb, and suggests a cutoff that clears the slowest.

**Noise model.** `--noise-model` sets the residual autocorrelation model: `ols`, `arN` for an AR model of order N, `auto` (the default) for an AR order of four times the sampling rate, `ar_irls`, or `worsley`. `ar_irls` alternates the AR whitening with a robust refit that down-weights outlying samples such as residual motion, until the betas stop moving; the order is chosen per channel by BIC, capped at the same bound `auto` uses or at N with `ar_irlsN`. `worsley` keeps `auto`'s order (N with `worsleyN`) and takes the coefficients from the residual's autocovariances after correcting the bias the fit leaves in them (Worsley et al., 2002); a channel whose corrected autocovariances do not admit that order is fitted as `auto`, and the sidecars list it as `worsley_fitted_as_auto`. Every channel keeps its own AR coefficients. Every mode fits the noise model, `denoise` and `rest` on their confound regression.

**Short-channel regression.** Short channels (at or under `--short-max-dist`) see the scalp only. `--short-channel mean` adds the mean of the good short channels as one regressor per chromophore; `pca` adds one orthogonalised column per good short channel, HbO and HbR decomposed together, which fits the same as entering every short channel. A short channel the regressors span is fitted against itself, so its residual is zero and its GLM statistics, connectivity and amplitude values are left blank. A montage with no short channel stops the run; a subject whose short channels were all rejected runs without the regressor, and its record and Methods text say so.

**Auxiliary regressors (experimental).** `--aux-regressors` adds the recording's aux channels to the confound regression, each resampled onto the data's time axis with an anti-alias filter, band-limited to `--high-pass` / `--low-pass` and z-scored. `--aux-channels` picks them by name. Both are withheld from `--help` while they are evaluated.

**Contrasts.** `--contrast-file` names a TOML file of linear contrasts over the condition regressors ([Contrast file](usage.md#contrast-file)), written to `desc-contrast_nirsmap.tsv`.

**P values.** Both tables give each test's uncorrected `p_value`. `--p-correction` (`fdr_bh`, `fdr_by`, `holm` or `bonferroni`; none by default) adds `p_<method>` beside it, corrected within one regressor or contrast and one chromophore over the long channels not marked bad; short and rejected channels keep their `p_value` and get no corrected one. `family` gives the number of tests a correction runs over, and is written whether or not one ran. The Methods paragraph says whether p was corrected, and how. MNE-NIRS's own `Significant` column, p below 0.05 uncorrected, is not written.

Every channel is fitted, rejected ones included; their rows carry `bad = True` rather than being dropped, so the table shape stays the same across subjects.

### Mode: rest

Resting-state analysis: a confound regression with no task regressors, then ALFF and functional connectivity on the residuals. The mode fills a 0.01 to 0.08 Hz bandpass and a 0.01 Hz cosine drift. The confound regression runs **twice, on different inputs**:

```mermaid
flowchart TD
    PRE(["desc-preproc"])
    PRE --> A["bandpass + resample"]
    PRE --> B["resample only<br/><small>no bandpass</small>"]

    A --> AR(["desc-resampled"])
    B --> BR["broadband copy<br/><small>in memory, never written</small>"]

    AR --> REG1["confound regression<br/><small>empty events, drift + short channels</small>"]
    BR --> REG2["same regression<br/><small>identical settings</small>"]

    REG1 --> E1(["desc-errts"])
    REG2 --> E2(["desc-errtsbroad"])

    E1 --> FC["functional connectivity<br/><small>per chromophore</small>"]
    FC --> FCF(["chromo-hbo/hbr_stat-pearson_relmat.tsv<br/>chromo-hbo/hbr_stat-fisherz_relmat.tsv"])
    FC --> ROI{"--roi-mapping"}
    ROI -->|given| ROIF(["seg-&lt;map&gt;_agg-roi / agg-seed relmat.tsv<br/><small>bad channels excluded from ROI means</small>"])

    E2 --> ALFF["ALFF / fALFF<br/><small>mALFF, zALFF standardized<br/>against good channels only</small>"]
    ALFF --> ALFFF(["stat-alff_nirsmap.tsv"])
    ALFF --> AROI{"--roi-mapping"}
    AROI -->|given| AROIF(["seg-&lt;map&gt;_agg-roi_stat-alff_nirsmap.tsv<br/><small>measured per channel, then averaged</small>"])

    classDef haemo fill:#fadbd8,stroke:#c0392b
    classDef derivative fill:#d6eaf8,stroke:#2874a6
    classDef input fill:#ecf0f1,stroke:#7f8c8d
    class PRE,AR,E1,E2 haemo
    class FCF,ROIF,ALFFF,AROIF derivative
    class BR input
```

fALFF is the band's share of the total spectral amplitude, so ALFF and fALFF are read from a residual that was detrended but never low-passed (`desc-errtsbroad`), while connectivity is read from the bandpassed one (`desc-errts`).

**Connectivity.** Pearson correlation between every pair of channels, per chromophore, with its Fisher z. A rejected channel is a blank row and column, so every subject's matrix keeps one shape and one channel order.

**Amplitude.** Per channel, from the amplitude spectrum of the broadband residual:

| Column | Definition |
|--------|------------|
| `alff` | Mean amplitude across the `[high_pass, low_pass]` band |
| `falff` | Band amplitude over total amplitude, DC excluded, in `[0, 1]` |
| `malff` | `alff` divided by the mean `alff` of its own chromophore |
| `zalff` | `alff` z-scored within its own chromophore |

ALFF needs both cutoffs; with either switched off it is skipped, and connectivity is still computed.

```{figure} figures/rest_connectivity.png
:alt: HbO and HbR channel correlation matrices above ALFF and fALFF per channel
:width: 100%

The rest-mode panel of the report: HbO and HbR channel correlation matrices on one scale, and ALFF and fALFF per channel for each chromophore.
```

**Regions.** With `--roi-mapping`, each region's good channels are averaged into one signal, and the region-by-region correlation and each region's seed map (region against every channel) are computed from those signals. The amplitude measures are averaged within each region after they are measured, the opposite order, with `n_channels` beside them. A region left with no good channel stays in every output as a blank entry, so every subject's tables have the same rows. Each ROI product's sidecar records, as `roi_channels`, which channels the region was actually averaged from.

### Connectivity on task data

`--fc` writes the connectivity products of rest mode in the other two modes: in `glm` from the residual the task was modelled out of, in `denoise` from its confound residual or its bandpassed data. ALFF is not written, since it needs a broadband residual. Give `--high-pass` / `--low-pass` too; without a bandpass the correlations are dominated by drift.

### Combining runs

`--combine-runs` joins the runs of a task once each has been postprocessed. Files of one subject that differ only in `run` are joined in run order: each run's final series (the residual, or the bandpassed or resampled data when no regression ran) is z-scored per channel over its frames outside `BAD_` spans, so a run with a larger variance does not outweigh the others, and the joined series is written without the `run` entity, its sidecar naming each run and where it starts. A channel takes part only in the runs where it was not rejected: its frames are NaN in the others, as XCP-D keeps a region missing from a run, and the sidecar's `channel_runs` names the runs each channel took part in. It is marked bad in the joined series only when every run rejected it. FC, ROI FC and seed FC are computed on the joined series wherever the mode writes FC, each pair over the kept frames both have; an ROI is averaged per run over that run's own good channels before the z-score. ALFF and the GLM's tables stay per run. A run with less kept time than `--min-time` is left out, and a task with a single run is not joined. The [subject index](qc.md) tables each joined run, draws the joined carpet and the joined FC.

### Chromophores

`--chromophores` names what the analysis tables carry: one or more of `hbo`, `hbr` and `hbt`, `hbo hbr` by default. `hbt`, total haemoglobin, is HbO plus HbR of each channel, sample by sample, and is bad wherever either half is. MNE has no HbT channel type, so HbT is never written to a stage file: each analysis forms it from the processed pair it reads, and every stage file keeps HbO and HbR whatever the flag says. Every step after Beer-Lambert is linear, so summing the processed pair gives what carrying HbT from the start would.

- **GLM.** Every channel is fitted as before, and HbT is fitted as a series of its own with the same design and noise model; the GLM, contrast and whitening tables carry the chromophores asked for. Under `ols` the HbT beta equals the sum of the HbO and HbR betas and only the t differs; under an AR model the beta differs slightly too, since HbT has its own noise. The short-channel regressors already hold both chromophores, so HbT needs none of its own.
- **Block average.** HbT's trial average is the sum of the pair's.
- **Connectivity and amplitude.** FC and its region and seed products take HbT from the residual pair they read, `desc-errts` (or the last stage when no regression ran), and ALFF from `desc-errtsbroad`; mALFF and zALFF scale HbT within HbT. With `--combine-runs` HbT is summed in each run before the run's z-score.

The report follows the flag: the activation and ALFF brains draw a row or block per chromophore, each on its own colour scale, and the connectivity figures draw each chromophore the tables hold. The raw quality panels keep HbO and HbR. The dyad commands take the same choice through `--wtc-chroma` ([Hyperscanning](hyperscanning.md)).

## Excluded time

A run is always processed whole. Time that should not count is marked with an annotation whose description starts with `BAD_`, and every estimate leaves those frames out. Nothing is cut: every output keeps the run's full length, the spans travel with it as annotations, and every sidecar lists them under `excluded_spans`. With no span at all, every step takes its plain route.

| Kind | Annotation | Comes from | The data there |
|------|------------|------------|----------------|
| unselected | `BAD_unselected` | `--keep-spans`: everything outside the stretches its table lists | Intact measurements you chose not to count |
| corrupted | every other `BAD_` description | `BAD_manual` from `--bad-spans`, `BAD_gvtd` from `--gvtd-censor`, `BAD_edge` where most channels measured no light at the recording's start or end ([Optical density](#optical-density)), or any `BAD_` span the input file carries | Damaged by motion or another artifact, or not measured |

The two kinds part ways only before the filters: corrupted time is filled first ([Filling corrupted spans](#filling-corrupted-spans)), intact time is not. From the first estimate on, both are left out alike. An unknown `BAD_` description counts as corrupted.

`--keep-spans` and `--bad-spans` take tables of `onset` and `duration` in seconds on each recording's own time axis, with optional columns limiting a row to some recordings ([Data preparation](data-preparation.md#choosing-stretches-of-a-recording)). `--gvtd-censor` marks the frames where the global variance of the temporal derivative (GVTD) exceeds `--gvtd-censor-n-std` standard deviations of its left tail above its mode, on the channels screening kept, and censors any surviving stretch shorter than `--gvtd-min-epoch-s` with it. It is off by default.

| Step | What it does with `BAD_` time |
|------|-------------------------------|
| Channel screening | A window touching a span is not counted. `BAD_gvtd` is made after screening, so it never changes the channel set. A run left with fewer than two windows stops. |
| Motion correction | Runs over the whole recording. |
| Fill, bandpass, resampling | Corrupted spans are filled, then everything is filtered whole. |
| GLM and confound regression | The design is built on the full time axis, and the `BAD_` rows are dropped from data and design at fit time. AR coefficients come only from samples whose lags lie in kept time. The fit is applied to the whole run, so `desc-errts` stays full length. A fit with fewer than three kept frames per regressor is refused, and a condition with no kept rows is dropped and named. |
| Connectivity | Pearson r over the kept frames. |
| ALFF, fALFF | A Lomb-Scargle periodogram of the kept samples. |
| `--min-time` | A run with fewer kept seconds than this gets no connectivity or ALFF file. Off by default. |
| Dyad coherence and correlation | Each pairing leaves out the union of both members' spans ([Hyperscanning](hyperscanning.md#excluded-time)). |
| QC report | Left out of the condition shares, the trials and the spectra, and listed in an Excluded section. |

Values at `BAD_` frames are never NaN: unselected frames hold the processed measurement, corrupted frames the filtered fill. Read the annotations, or the sidecar's `excluded_spans`, to leave them out. For a tool that ignores annotations, `--export-mean-filled` also writes `desc-meanfilled` beside the final residual, every `BAD_` frame set to its channel's mean over the kept frames. A correlation over it equals one over the kept frames; an amplitude spectrum, a GLM or a coherence over it does not.

A fill keeps an artifact out of the filter but does not restore the signal, and spans are recording-wide: a channel cannot keep a stretch another channel loses. Per-channel problems are what bad channels are for.

## Group level

`nirspipe BIDS_DIR OUTPUT_DIR group` reads the quality records under `OUTPUT_DIR`, not the BIDS input, and needs no preprocessing flag. It writes the cohort page over every run, `desc-subjects_report.html` with its table `desc-subjects_qc.tsv`, and, where the tree also holds dyad records, the dyad cohort page `desc-groups_report.html` with `desc-groups_qc.tsv`. Both are described in [QC](qc.md#cohort-reports).
