---
title: API
---

# API

Generated from the source by static analysis: every page below comes from the docstrings and signatures in `nirspipe/`.

| Module | Description |
|--------|-------------|
| `nirspipe.pipeline` | Preprocessing, postprocessing, dyad analysis, cropping, marker editing |
| `nirspipe.io` | BIDS querying, SNIRF read and write, derivative names and paths |
| `nirspipe.qc` | Quality records, metrics, figures, report writers, rating apps, Methods text |

```{toctree}
:maxdepth: 2

_generated/nirspipe.pipeline/nirspipe.pipeline
_generated/nirspipe.io/nirspipe.io
_generated/nirspipe.qc/nirspipe.qc
```

`nirspipe.utils` (logging, run records, the event stream and database) is internal; its user-facing surface is `nirspipe-log`. The command-line front-end (`nirspipe.cli`) and the GUI (`nirspipe.interface`) build on these subpackages and are documented as commands in [Usage](../usage.md) and [GUI](../gui.md). The API is not yet stable.

## Package layout

```
nirspipe/
├── cli/             ← argparse entry points, one file per command
│   ├── run.py       ← nirspipe (participant / group)
│   ├── workflows.py ← what a run walks: subject resolution, config merge, dispatch
│   ├── recon.py, prep.py, qc.py, hyper.py, rate.py, gui.py, db.py
│
├── pipeline/        ← signal processing
│   ├── prep_pipeline.py     (fixed-order preprocessing)
│   ├── post_pipeline.py     (denoise / glm / rest dispatch)
│   ├── glm.py, glm_censored.py, ar_irls.py
│   ├── denoise.py           (bandpass and resampling)
│   ├── restingstate.py      (ALFF, connectivity)
│   ├── motion.py            (motion correction)
│   ├── censor_fill.py       (filling corrupted BAD_ spans before the bandpass)
│   ├── crop.py, edit_markers.py, keep_spans.py
│   └── hyper/               ← dyad analysis
│       ├── group_io.py          (pairs table, member lookup, stage loading)
│       ├── alignment.py         (the shared clock and the analysis window)
│       ├── group_quality.py     (the dyad's channel set and separation bands)
│       ├── wtc.py, whiten.py, isc.py, roi.py
│       ├── surrogate.py         (the nulls' recomputation)
│       ├── hyper_post.py        (what nirspipe-hyper walks)
│       ├── pair_null.py, pair_null_group.py
│       └── wtc_store.py, wtc_null.py, wtc_aggregate.py
│
├── io/              ← bids.py, snirf.py, naming.py, derivatives.py, auxiliary.py, tables.py
│
├── qc/
│   ├── subject/     ← run report, raw report, condition pages, subject index,
│   │                  quality record, per-trial scoring, cohort writer
│   ├── hyper/       ← dyad raw and analysis reports, dyad index, usable time, dyad cohort writer
│   ├── common/      ← report shell, channel table, windows, provenance graph, figure I/O
│   ├── metrics/     ← coupling, screening, GVTD, motion, haemoglobin, windowed, dyad, IMU
│   ├── figures/     ← figure modules, split common / subject / hyper
│   ├── rating/      ← rating apps
│   ├── boilerplate/ ← Methods text, metric descriptions, references
│   └── templates/   ← HTML templates
│
├── interface/       ← the GUI
├── utils/           ← logging, run_record, job_db, lineage, spans, snirf_prep, net
├── data/            ← the BIDS entities and path patterns names are built from, mode presets
└── exceptions.py    ← AlignmentError, GroupCSVError, MissingDerivativesError, StageError
```

| Module | Responsibility |
|--------|---------------|
| `cli/` | Argument parsing, subject iteration, parallel dispatch. No signal processing. |
| `io/` | BIDS querying, SNIRF read and write, derivative names, paths and sidecars. |
| `pipeline/` | All signal processing. Writes the stage files; owns no report or database writes. |
| `qc/` | Quality records, metrics, reports, rating, Methods text. Reads the pipeline's outputs. |
| `interface/` | The GUI. Builds commands; runs no preprocessing in its own process. |
| `utils/` | Logging, run records, the event stream and database, span handling. |
