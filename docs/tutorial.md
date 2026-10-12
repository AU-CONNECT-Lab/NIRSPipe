---
title: Tutorial
---

# Tutorial

This page runs three dyads from a public hyperscanning dataset through every command, from download to the cohort null tests. It takes about 11 minutes on a desktop and 1.1 GB of disk. NIRSPipe must be [installed](installation.md) first.

The commands are written for a Mac or Linux terminal and Git Bash. In Windows PowerShell, end each continued line with a backtick (`` ` ``) instead of `\`.

## The data

The InterGenSynchrony dataset (Moffat and Cross, OpenNeuro ds008192, CC0): adult pairs with caps over the left and right inferior frontal gyrus and temporo-parietal junction, drawing alone, drawing together twice, then a short collaborative task. The tutorial uses session 2 of dyads 1001, 1002 and 1003, 23 MB in all.

## 1. Download

Make a working folder; every later command runs from it.

```bash
mkdir fnirs_demo
cd fnirs_demo
pip install openneuro-py
openneuro-py download --dataset ds008192 --target-dir ds008192 \
    --include sub-101/ses-02 --include sub-201/ses-02 \
    --include sub-102/ses-02 --include sub-202/ses-02 \
    --include sub-105/ses-02 --include sub-203/ses-02
```

Save {download}`pairs.csv <files/pairs.csv>` and {download}`roi.json <files/roi.json>` into `fnirs_demo`. The first names the dyads ([Pairs file](data-preparation.md#pairs-file)), the second groups the channels into four regions ([ROI mapping file](data-preparation.md#roi-mapping-file)).

## 2. Name the conditions

The recordings mark each block with a bare number and a 1-second trigger. Rename the markers, then set every block to 300 s; each command makes one edit and writes a new copy ([Editing markers](data-preparation.md#editing-markers)).

```bash
nirspipe-prep edit-markers apply ds008192 renamed \
    --participant-label 101 201 102 202 105 203 \
    --rename 1:alone 2:together1 3:together2 4:collab \
             1.0:alone 2.0:together1 3.0:together2 4.0:collab \
    --skip-bids-validation
nirspipe-prep edit-markers apply renamed/marker_edited timed \
    --participant-label 101 201 102 202 105 203 \
    --set-duration 300 --skip-bids-validation
```

From here on the input dataset is `timed/marker_edited`.

## 3. Preprocess each person

```bash
nirspipe timed/marker_edited derivatives participant \
    --participant-label 101 201 102 202 105 203 \
    --dpf 6.0 --sci-threshold 0.8 \
    --cardiac-l-freq 0.7 --cardiac-h-freq 2.0 \
    --resp-l-freq 0.1 --resp-h-freq 0.5 \
    --short-max-dist 12 \
    --mode denoise --short-channel pca \
    --high-pass 0.01 --low-pass none \
    --resample-sfreq 5 \
    --by-condition --fc --roi-mapping roi.json \
    --skip-bids-validation --n-jobs 3
```

About 2.5 minutes. Three settings are specific to this dataset:

| Option | Why |
|---|---|
| `--short-max-dist 12` | The short channels sit right at the 10 mm default edge |
| `--resample-sfreq 5` | The two caps of a pair recorded at slightly different rates, and the dyad analysis needs one rate |
| `--by-condition` | Writes a report page per condition as well as per run |

Each person's pages start at `derivatives/sub-101/sub-101_desc-index_report.html`.

## 4. Cohort page

```bash
nirspipe timed/marker_edited derivatives group --skip-bids-validation
```

Writes `derivatives/desc-subjects_report.html` and `derivatives/desc-subjects_qc.tsv`.

## 5. Dyad analysis

```bash
nirspipe-hyper derivatives hyper group \
    --pairs-csv pairs.csv --desc errts --roi-mapping roi.json \
    --wtc-band-fmin 0.02 --wtc-band-fmax 0.10 \
    --isc-fmin 0.02 --isc-fmax 0.10 \
    --wtc-chroma hbo \
    --wtc-phase-null 5 --isc-phase-null 100 --wtc-seed 1
```

About 6 minutes. `--wtc-chroma hbo` keeps coherence and correlation to HbO, which halves the time, and five WTC scrambles is a tutorial setting, far below what a study needs. Each dyad's pages start at `hyper/group-1001/group-1001_desc-index_report.html`.

Then the raw quality page for each dyad, under a minute:

```bash
nirspipe-qc hyper-raw timed/marker_edited hyper group \
    --pairs-csv pairs.csv --derivatives-dir derivatives \
    --dpf 6.0 --sci-threshold 0.8 \
    --cardiac-l-freq 0.7 --cardiac-h-freq 2.0 \
    --short-max-dist 12 --skip-bids-validation
```

```{note}
The members' markers in dyads 1002 and 1003 disagree by 5 to 30 s after alignment, a fault in the recordings. Their condition results are not usable; the dyad pages list the differences under the alignment check.
```

## 6. Nulls

The re-paired null, about a minute, then the cohort test against both nulls:

```bash
nirspipe-hyper-pairnull derivatives hyper group \
    --pairs-csv pairs.csv --roi-mapping roi.json --chroma hbo
nirspipe-hyper-groupnull hyper group --task-label drawing \
    --roi-mapping roi.json --seed 1
```

The cohort test writes `hyper/task-drawing_desc-groupnull_report.html` and its `chromo-hbo_*` tables.

## 7. Merge and the dyad cohort page

```bash
nirspipe-hyper-merge hyper group
nirspipe-qc cohort-hyper hyper
```

The merged tables land at the top of `hyper/`, one per kind (`stat-wtc_relmat.tsv`, `stat-isc_relmat.tsv`, ...), and the cohort page is `hyper/desc-groups_report.html`. [Outputs](outputs.md) lists every file both trees hold, and [QC](qc.md) and [Hyperscanning](hyperscanning.md) explain the pages.
