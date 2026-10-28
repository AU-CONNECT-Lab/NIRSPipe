"""Raw QC run as a study runs it: every member's participant level, then the group level."""

from nirspipe.cli import qc as qc_cli

SCREENING = ("--dpf", "6", "--sci-threshold", "0.8",
             "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
             "--resp-l-freq", "0.1", "--resp-h-freq", "0.5")


def raw_qc(bids, out, pairs=None, participant=SCREENING, group=(), validate=False):
    """Both levels into ``out``; the dyad pages too when ``pairs`` is given."""
    skip = () if validate else ("--skip-bids-validation",)
    qc_cli.main([str(bids), str(out), "participant", *participant, *skip])
    qc_cli.main([str(bids), str(out), "group",
                 *(("--pairs-csv", str(pairs)) if pairs is not None else ()), *group, *skip])
