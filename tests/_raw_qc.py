"""Raw QC run as a study runs it: every member's participant level, then the group level."""

from nirspipe.cli import qc as qc_cli
from nirspipe.pipeline.prep_pipeline import PrepConfig

SCREENING = ("--dpf", "6", "--sci-threshold", "0.8",
             "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
             "--resp-l-freq", "0.1", "--resp-h-freq", "0.5")


def raw_run(bids, path, task, **settings):
    """The run entry ``nirspipe-qc participant`` hands the raw report, for ``sub-01_task-<task>``."""
    config = PrepConfig(**{"subject": "01", "dpf": [6.0], "sci_threshold": 0.8,
                           "cardiac_l_freq": 0.7, "cardiac_h_freq": 1.5, "resp_l_freq": 0.1,
                           "resp_h_freq": 0.5, "motion_correction": "none", "lenient": True,
                           **settings})
    return {"label": f"sub-01_task-{task}", "subject_id": "01", "snirf_path": path,
            "session": None, "bids_dir": bids, "entities": {"task": task}, "config": config}


def raw_qc(bids, out, pairs=None, participant=SCREENING, group=(), validate=False):
    """Both levels into ``out``; the dyad pages too when ``pairs`` is given."""
    skip = () if validate else ("--skip-bids-validation",)
    qc_cli.main([str(bids), str(out), "participant", *participant, *skip])
    qc_cli.main([str(bids), str(out), "group",
                 *(("--pairs-csv", str(pairs)) if pairs is not None else ()), *group, *skip])
