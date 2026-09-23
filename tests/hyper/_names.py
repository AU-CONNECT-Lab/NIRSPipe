"""A dyad table's name, built the way the writers build it.

Spelling these out in each test is how a check and its writer drift apart: both halves read
right, both agree with each other, and neither is what lands on disk. Everything here goes
through the same config `fnirs_pipe.io.naming` writes against, so a rename breaks the tests
that are actually wrong rather than every test at once.
"""

from fnirs_pipe.io.naming import derivative_path

# the entity sets the WTC pass writes, under the names this suite refers to them by
KINDS = {
    "wtc":                        {"statistic": "wtc"},
    "wtc-roichan":                {"segmentation": "custom", "aggregation": "roi",
                                   "statistic": "wtc"},
    "wtc-roihom":                 {"segmentation": "custom", "aggregation": "homologous",
                                   "statistic": "wtc"},
    "wtc-phasescale":             {"statistic": "wtcphase"},
    "wtcbycond":                  {"condition": "all", "statistic": "wtc"},
    "wtcbycond-roichan":          {"segmentation": "custom", "aggregation": "roi",
                                   "condition": "all", "statistic": "wtc"},
    "wtcbycond-roihom":           {"segmentation": "custom", "aggregation": "homologous",
                                   "condition": "all", "statistic": "wtc"},
    "wtcbycond-phasescale":       {"condition": "all", "statistic": "wtcphase"},
    "wtc-phasenull":              {"nulldist": "phase", "statistic": "wtc"},
    "wtc-roihom-phasenull":       {"segmentation": "custom", "aggregation": "homologous",
                                   "nulldist": "phase", "statistic": "wtc"},
    "wtcbycond-phasenull":        {"condition": "all", "nulldist": "phase",
                                   "statistic": "wtc"},
    "wtcbycond-roihom-phasenull": {"segmentation": "custom", "aggregation": "homologous",
                                   "condition": "all", "nulldist": "phase",
                                   "statistic": "wtc"},
    "wtcbycond-phasenull-draws":  {"condition": "all", "nulldist": "phase",
                                   "statistic": "wtc", "desc": "draws"},
    "wtc-pairnull":               {"nulldist": "pair", "statistic": "wtc"},
    "wtc-roihom-pairnull":        {"segmentation": "custom", "aggregation": "homologous",
                                   "nulldist": "pair", "statistic": "wtc"},
    "wtcbycond-pairnull":         {"condition": "all", "nulldist": "pair",
                                   "statistic": "wtc"},
    "wtcbycond-roihom-pairnull":  {"segmentation": "custom", "aggregation": "homologous",
                                   "condition": "all", "nulldist": "pair",
                                   "statistic": "wtc"},
    "wtcbycond-pairnull-draws":   {"condition": "all", "nulldist": "pair",
                                   "statistic": "wtc", "desc": "draws"},
    "iscpairs":                   {"statistic": "isc"},
    "isc-pairnull":               {"nulldist": "pair", "statistic": "isc"},
    "iscbycond-pairnull":         {"condition": "all", "nulldist": "pair",
                                   "statistic": "isc"},
    "iscbycond-pairnull-draws":   {"condition": "all", "nulldist": "pair",
                                   "statistic": "isc", "desc": "draws"},
}


def name(group: str, task: str, kind: str = "wtc", extension: str = ".tsv",
         **entities) -> str:
    """One dyad table's filename. ``kind`` is a key of :data:`KINDS`.

    An unknown kind raises rather than falling back to a bare name: silently dropping the
    entities gives two kinds one filename, which is a test that passes for the wrong reason.
    """
    if kind not in KINDS:
        raise KeyError(f"{kind!r} is not one of {sorted(KINDS)}")
    return derivative_path("", "relmat", extension, group=group, task=task,
                           **{**KINDS[kind], **entities}).name


def archive(group: str, task: str, chromophore: str, **entities) -> str:
    """One chromophore's saved transform."""
    return name(group, task, extension=".npz", chromophore=chromophore, **entities)


def cohort(nulldist: str, desc: str, task: str = "full", chromophore: str = "hbo") -> str:
    """One cross-dyad verdict table at the tree root, from `fnirs-hyper group-null`.

    No group- and no sub-: having no analysis unit in the name is what marks a table as
    cross-dyad. The task and the chromophore stay, both being a filter the command was given.
    """
    return derivative_path("", "relmat", ".tsv", chromophore=chromophore, task=task,
                           condition="all", nulldist=nulldist, statistic="wtc",
                           desc=desc).name
