"""The GUI's pages: each a tab per command, grouped in the sidebar as the CLI's work is."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tab:
    key: str
    prog: str
    # a nested subcommand as a path, "edit-markers/apply"
    sub: "str | None" = None
    # a page of its own instead of the command's generated form
    template: "str | None" = None


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    tabs: "tuple[Tab, ...]"
    # the sidebar heading it sits under: one recording looked at and edited, or commands run over many
    family: str = "Batch"


PAGES = (
    Page("data-prep", "Data Preparation", (
        Tab("prep-raw", "nirspipe-qc", "prep-raw", template="data_prep.html"),
    ), family="Interactive"),
    Page("snirf2bids", "SNIRF to BIDS", (
        Tab("snirf2bids", "nirspipe-snirf2bids"),
        Tab("batch", "nirspipe-snirf2bids", template="snirf2bids_batch.html"),
    )),
    Page("batch-prep", "Batch Prep", (
        Tab("crop", "nirspipe-prep", "crop"),
        Tab("align", "nirspipe-prep", "align"),
        Tab("edit-markers", "nirspipe-prep", "edit-markers/apply"),
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

FAMILIES = tuple(dict.fromkeys(p.family for p in PAGES))
