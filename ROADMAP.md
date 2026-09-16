# Roadmap

Where this package is and where it is going. Released changes are in [CHANGELOG.md](CHANGELOG.md); this file is only the shape of the work.

---

## What it does today

**Preprocessing.** Raw SNIRF through optical density, channel screening, motion correction and haemoglobin conversion. Every stage is written to disk, and each output records how it was made and from what, so a number in a report can be traced back to the file it came from.

**Analysis.** Task GLM, denoising and resting state for one person at a time. For two people recorded together, wavelet coherence and inter-subject correlation, each reported against a null to judge it by rather than as a bare number.

**Quality control.** A report for a single recording, for a pair, and for a whole cohort. Each opens in a browser with nothing running behind it, and reads the quality record it was drawn from, so a figure and the matching number cannot disagree. Recordings can be rated, and the ratings are kept alongside the data.

**Interfaces.** Everything is available from the command line. A browser interface covers the steps that benefit from being interactive: preparing data, editing markers, launching batches, aligning pairs, and running the pipeline while watching its output arrive.

**Reproducibility.** Runs are logged, and the logs consolidate into a database that answers questions across a whole study. The numerical results have been checked against independent implementations of the same algorithms.

---

## In progress

Nothing. The last piece of planned work closed on 2026-09-14.

---

## Considered, not committed

Ordered by how often each has come up, not by when it might happen.

**Analysis**

- Spline motion correction. Two motion correction methods already ship; this would be a third, and it needs its own artifact detection built alongside it
- External regressors in confound regression. Built, then withheld: whether it helps has not been shown on enough data to turn on

**Quality control**

- Projecting resting-state maps onto a brain surface rather than a flat layout
- Deriving region mappings from the montage automatically, instead of writing them by hand
- Comparing one subject's repeated runs side by side, to show how stable a measure is within a person
- Comparing the signal before and after every processing step, not only the two that have it

**Under the hood**

- Moving SNIRF reading onto a maintained library. Blocked upstream: the fix exists but has not been released

---

Anything more specific than this lives with the project notes, outside the repository.
