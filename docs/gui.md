---
title: GUI
---

# GUI

`nirspipe-gui` is a local web app that runs in your browser. Each page holds the form of one command-line tool, built from that tool's own flags: it shows the command the form makes, runs it and streams what it prints, so anything done in the GUI can be repeated from a terminal with the command it shows. The same app serves the rating viewers and the raw viewer's edit mode.

```bash
nirspipe-gui                  # port 8050, or the next free one above it
nirspipe-gui --port 8051      # exactly this port, or stop if it is taken
```

It opens your browser on Data Preparation. Keep the terminal open while you work: errors the app itself meets are printed there, and Ctrl+C stops it. The sidebar folds to a strip with the ☰ button.

## Pages

The sidebar has two groups. **Interactive** holds Data Preparation, which opens one recording to look at and edit. **Batch** holds the pages that run the command-line tools over many subjects or dyads, one tab per command.

| Page | Tabs | Runs |
|------|------|------|
| **Data Preparation** | prep-raw | `nirspipe-qc prep-raw`, then the raw viewer's edit mode: `nirspipe-prep edit-markers apply` and `nirspipe-prep crop` |
| **SNIRF to BIDS** | snirf2bids, batch | `nirspipe-snirf2bids`, once per file on the batch tab |
| **Batch Prep** | crop, align, edit-markers, keep-spans | `nirspipe-prep` |
| **Raw QC** | prep-raw, hyper-raw, rate-raw, rate-hyper, export | `nirspipe-qc prep-raw` / `hyper-raw`, the rating viewers, `nirspipe-rate export` |
| **Analysis** | run | `nirspipe` |
| **Hyper Analysis** | run, pair-null, group-null, band, merge, index | `nirspipe-hyper` and its companions |
| **Cohort Reports** | cohort, cohort-hyper, provenance, rate | `nirspipe-qc cohort` / `cohort-hyper` / `provenance`, `nirspipe-rate rate` |

**Commands**, under **All**, lists every command and subcommand with a form for each, including the few no page has a tab for.

A study usually moves down the table: convert with SNIRF to BIDS, fix events on Data Preparation and roll the fix out with Batch Prep, write the raw reports on Raw QC and rate them, then run Analysis, Hyper Analysis and Cohort Reports. Channel decisions are taken in the rating viewers only; the other pages record none.

The screenshots on this page come from the [tutorial](tutorial.md) run through the GUI, its folder mapped to the drive `S:` so the paths stay short.

## Forms

A field is a flag, named as the command line names it. The **?** beside it shows the flag's help; what a flag does, its default and the values it takes are also on the [Usage](usage.md) page. A new flag of a command appears on its page without any change to the GUI.

- **Required first.** The flags a command cannot run without lead in a group of their own, marked with a red star. The other groups keep the command's own grouping.
- **Folding.** Click a group's header to fold it. A folded group lists what is set inside it as the command line reads it, for example `--sci-threshold 0.8 · --skip-bids-validation`. On a first visit Required is open and the rest folded; after that each page keeps its folds.
- **Empty fields.** An empty field sends nothing, so the command's default applies. Where the default depends on another setting, such as a `--mode` default, the field shows it in grey.
- **Lists.** A field that takes several values takes them separated by spaces, or one per line for paths, since a path may hold spaces.
- **Memory.** Each page keeps its last values in this browser. The BIDS and output folders follow you from page to page: the last one typed on any page fills the others.
- **Detect.** Beside `--participant-label`, on a page that also has the BIDS folder, **Detect** lists the dataset's subjects as checkboxes. It ticks the ones already typed, or all of them when the field is empty, and the field follows the ticks.

## Running a command

The Run panel sits to the right of the form, and under it on a narrow window, and stays in view while the form scrolls.

- **Command** shows the exact command the form makes and follows every change. The **for** picker sets quoting and line continuation for PowerShell, bash or cmd, so the command pastes into that shell as it stands. Each positional argument gets its own line and each flag sits beside its values.
- The form is checked by the command's own parser as you type, so a missing or malformed flag shows the command line's own message under **Run**.
- **Run** runs what **Command** shows. The output appears under it as it is printed; the last 2000 lines are kept.
- **Stop** ends the running command. Closing the page does not.

After a run that writes a report, the report opens below the form, with its path and an **Open in a new tab** link above it. The frame is as tall as the window, so the report scrolls inside it and keeps its own tabs in view. A command that writes tables says so instead, and one that should have written a page and did not says what it looked for.

The rating tabs (Raw QC's rate-raw and rate-hyper, Cohort Reports' rate) have **Open** in place of **Run**: they open the rating viewer in the app and list the pages they opened.

## Data Preparation

The **Data Preparation** page opens a run's raw report in edit mode, where its events can be edited and the recording cropped. The report must exist first: either find it in a Raw QC tree, or make it on the page.

- **Find reports** takes a Raw QC tree (the output folder `prep-raw` wrote) and the BIDS dataset it read, and lists the tree's run pages. Each link opens one in edit mode in a new tab.
- **Make reports** is the `nirspipe-qc prep-raw` form. When the run finishes, each report it wrote is listed under **Edit events** above the framed report. `--run-label` gives one run a page of its own.

```{figure} figures/gui_dataprep.png
:alt: The Data Preparation page after a prep-raw run, with the Edit events link and the raw report framed below the form
:width: 100%

Data Preparation after `prep-raw` on sub-101: the run panel on the right, the Edit events link and the framed raw report below.
```

In edit mode the page is the raw report as the rating viewer shows it, read only for ratings and channel decisions, which are taken in `nirspipe-rate raw`. Two sections are added under **Raw Signal**.

```{figure} figures/gui_edit.png
:alt: The raw report in edit mode: the Raw Signal plot, the Edit events table and the Crop section with their forms
:width: 100%

sub-101's drawing run in edit mode: the four renamed blocks in the Edit events table and on the plot, and the Crop section below.
```

### Edit events

The table holds the run's events (onset, duration, name), read with `nirspipe-prep edit-markers export`. Every edit shows at once on the Raw Signal plot and in the report's Events list.

| Control | Effect |
|---------|--------|
| **Step (s)** with **− All** / **+ All** | Shift every onset by the step; onsets stop at 0. |
| **− Picked** / **+ Picked** | The same shift, for the ticked rows only. |
| **Rename** … **to** … **Rename all** | Rename every event of one name. |
| **+ Add**, **×** | Add a row; delete a row. |
| **Download TSV** | Download the table as `events.tsv`; nothing is written to disk by the app. |
| **Save** | Run `nirspipe-prep edit-markers apply --tsv` with the table, into the output folder of the form below it. |

Under the table the page gives the cosine drift cutoff a GLM on this run could use, from the slowest repeat of each condition ([Choosing the drift cutoff](#choosing-the-drift-cutoff)).

### Crop

- **One stretch** keeps the recording between **Start** and **End** (seconds; either may be left open). **Draw on plot** lets you drag a box on the Raw Signal plot; typing a value moves the box, and **Use plot zoom** takes the range the plot shows.
- **Segments** keeps several stretches, each a row of onset, duration and an optional task name, with **+ Segment**. `--combine` on the form joins them into one recording; without it each segment is written as a recording of its own.

**Crop** runs `nirspipe-prep crop` on the recording the page shows. The form below the section holds crop's other flags, among them `--align trigger` for times counted from a trigger.

### What gets written

Both actions run on the shown recording alone: its subject, session, task and run become the selection flags.

| Action | Writes |
|--------|--------|
| Save | `<output_dir>/marker_edited/sub-<id>/[ses-<id>/]nirs/`, the recording with its sidecars and the new `events.tsv` |
| Crop | `<output_dir>/cropped/sub-<id>/[ses-<id>/]nirs/`, one recording, or one per segment |

When either finishes, a link opens the `prep-raw` form on the tree it wrote, with the output folder cleared: pick a new one, since the edited recording's report has the same name as the first. The recordings in the BIDS dataset are never modified.

## SNIRF to BIDS

The **snirf2bids** tab is the `nirspipe-snirf2bids` form for one file. The **batch** tab converts a folder of SNIRF files, one `nirspipe-snirf2bids` per file.

**Find files** searches the folder and its subfolders and lists each `.snirf` as a row. **subject** is guessed from the file name with everything but letters and digits removed; check it. **task** is required and starts empty; **session** and **run** are optional. The form above the table holds the flags every file shares: the output BIDS folder, `--optode-frame` and `--overwrite`.

**Run** converts the rows one after another under one log and one **Stop**. Each file starts with a line such as `[2/5] sub01.snirf -> sub-sub01`, one failing file does not stop the rest, and the last line gives the totals. A row without a subject or a task is skipped and named.

:::{tip}
Convert a couple of files first and open the resulting tree before running the whole folder. A wrong subject or task propagates into every derivative and report downstream, and the fix is a re-conversion.
:::

## Batch Prep

The **Batch Prep** tabs apply one data-preparation step to many subjects at once, without preprocessing: once Data Preparation has shown what a run needs, roll the same edit out across the study. Each tab is one `nirspipe-prep` subcommand ([Usage](usage.md#nirspipe-prep)).

| Tab | Runs |
|-----|------|
| crop | `nirspipe-prep crop` |
| align | `nirspipe-prep align`: dyad recordings aligned by a shared trigger ([alignment](hyperscanning.md#alignment)) |
| edit-markers | `nirspipe-prep edit-markers apply`: one of `--shift`, `--set-duration`, `--rename` or a `--tsv` table per run |
| keep-spans | `nirspipe-prep keep-spans`: a table for `nirspipe --keep-spans` ([Excluded time](workflows.md#excluded-time)) |

## Raw QC

The **Raw QC** page writes the quality reports on the recordings as delivered: one per subject with **prep-raw**, one per dyad with **hyper-raw**. See [Raw QC reports](qc.md) for what they hold.

```{figure} figures/gui_rawqc.png
:alt: The Raw QC hyper-raw tab after a dyad run, with the form, the run panel and the dyad report framed below
:width: 100%

Raw QC after `nirspipe-qc hyper-raw` on the tutorial's three dyads, with the newest dyad report below the form.
```

After a prep-raw run, a **Rate** link per report opens it in the rating viewer in the app. There, bad stretches can be marked on the raw signal of a run page. The **rate-hyper** tab opens a dyad report, and **export** runs `nirspipe-rate export`, which writes the `--bad-channels` table for Analysis, and the `--bad-spans` table when stretches were marked ([Rating](qc.md#rating)).

## Analysis

The **Analysis** page runs `nirspipe`, preprocessing and, with `--mode`, postprocessing.

```{figure} figures/gui_analysis.png
:alt: The Analysis page after a denoise run: the stages drawn above the form, the run panel on the right and the run report below
:width: 100%

The Analysis page set up as the tutorial's preprocessing step, after the run: the denoise stages above, the folded groups' settings, and the run report below.
```

### Run record

Give **Run record** a run's record (`<output_dir>/sub-<id>/logs/sub-<id>.toml`) or any `--config` file, and **Load into the form** fills every field it holds, read as `nirspipe --config` reads it: a file the command line would refuse shows its message. A record holds no folders, so the two folder fields keep the last ones used; check them.

### Stages

Above the form, the stages the chosen mode runs, in order: preprocessing in orange, postprocessing in blue, the GLM in green. Denoise draws the confound regression only when a drift model or short-channel regression gives it something to remove. **Download PNG** saves the drawing at twice its size. After a run, the report's Provenance section holds the graph of what the run actually did.

### What the form adjusts

The form follows the choices that change what other flags mean:

- With a mode chosen, the filter, drift, HRF and noise-model fields show that mode's defaults in grey ([Mode defaults](usage.md#mode-defaults)). Type `none` into a cutoff field to switch off a cutoff the mode would fill in.
- A flag the current choices do not read is hidden and left out of the command: the GLM flags without `--mode glm`, `--fir-delays` without `--hrf-model fir`, the drift settings of the other drift model, the motion settings of the other methods, the GVTD settings without censoring.
- Notes under the form warn when a cosine drift model has no cutoff, and when the drift cutoff is below the high-pass, where the basis misses part of what the filter removed.

### Choosing the drift cutoff

The cosine drift basis spans every frequency below its cutoff, which puts the cutoff between two bounds:

```
data high-pass   <=   drift cutoff   <   1 / longest repeat interval
```

Below the high-pass, the basis does not cover what the filter removed and the betas come out too small. At or above `1/T`, for a condition repeating every `T` seconds, the basis fits the condition away as drift; the usual choice is `1 / (2T)`. The run checks both bounds and names each condition the basis would absorb ([Choosing the cosine drift cutoff](workflows.md#mode-glm)). The Edit events section of [Data Preparation](#edit-events) reads the number off a run's events.

## Hyper Analysis

The **Hyper Analysis** page runs `nirspipe-hyper` and its companions on what `nirspipe` wrote; see [Hyperscanning](hyperscanning.md).

```{figure} figures/gui_hyper.png
:alt: The Hyper Analysis run tab after a dyad run, with the run panel and a dyad landing page framed below
:width: 100%

Hyper Analysis after the tutorial's dyad analysis, with the newest dyad's landing page below the form.
```

| Tab | Runs | After a run |
|-----|------|-------------|
| run | `nirspipe-hyper` | the newest dyad's landing page |
| pair-null | `nirspipe-hyper-pairnull` | tables |
| group-null | `nirspipe-hyper-groupnull` | the cohort test page, `task-<task>_desc-groupnull_report.html` |
| band | `nirspipe-hyper-band` | tables |
| merge | `nirspipe-hyper-merge` | tables |
| index | `nirspipe-hyper-index` | the newest dyad's landing page |

## Cohort Reports

The **Cohort Reports** page rolls the quality records of a finished tree up into cohort pages, and redraws provenance graphs.

```{figure} figures/gui_cohort.png
:alt: The Cohort Reports cohort tab after a run, with the cohort report framed below
:width: 100%

Cohort Reports after `nirspipe-qc cohort` on the tutorial's six members.
```

| Tab | Runs | After a run |
|-----|------|-------------|
| cohort | `nirspipe-qc cohort` | `desc-subjects_report.html` |
| cohort-hyper | `nirspipe-qc cohort-hyper` | `desc-groups_report.html` |
| provenance | `nirspipe-qc provenance` | graphs, no page |
| rate | `nirspipe-rate rate` | the rating viewer for any report |
