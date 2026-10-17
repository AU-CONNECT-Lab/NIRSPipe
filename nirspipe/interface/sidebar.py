"""The GUI's pages: each a tab per command, grouped in the sidebar as the CLI's work is."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tab:
    key: str
    prog: str
    sub: "str | None" = None


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    tabs: "tuple[Tab, ...]"


PAGES = (
    Page("recon", "Recon", (Tab("recon", "nirspipe-recon"),)),
    Page("batch-prep", "Batch Prep", (
        Tab("crop", "nirspipe-prep", "crop"),
        Tab("align", "nirspipe-prep", "align"),
        Tab("edit-markers", "nirspipe-prep", "edit-markers"),
        Tab("keep-spans", "nirspipe-prep", "keep-spans"),
    )),
    Page("raw-qc", "Raw QC", (
        Tab("prep-raw", "nirspipe-qc", "prep-raw"),
        Tab("hyper-raw", "nirspipe-qc", "hyper-raw"),
        Tab("rate-raw", "nirspipe-rate", "raw"),
        Tab("rate-hyper", "nirspipe-rate", "hyper"),
        Tab("export", "nirspipe-rate", "export"),
    )),
    Page("analysis", "Analysis", (Tab("run", "nirspipe"),)),
    Page("hyper-analysis", "Hyper Analysis", (
        Tab("run", "nirspipe-hyper"),
        Tab("pair-null", "nirspipe-hyper-pairnull"),
        Tab("group-null", "nirspipe-hyper-groupnull"),
        Tab("band", "nirspipe-hyper-band"),
        Tab("merge", "nirspipe-hyper-merge"),
        Tab("index", "nirspipe-hyper-index"),
    )),
    Page("cohort", "Cohort Reports", (
        Tab("cohort", "nirspipe-qc", "cohort"),
        Tab("cohort-hyper", "nirspipe-qc", "cohort-hyper"),
        Tab("provenance", "nirspipe-qc", "provenance"),
        Tab("rate", "nirspipe-rate", "rate"),
    )),
)
