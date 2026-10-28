"""Each ``--roi-mapping`` region's long channels kept: the count, and the run report's table of it.

The synthetic montage has four long pairs, S1_D1 to S4_D4, and the short S5_D5; S3_D3's
wavelengths do not couple, so screening rejects it.
"""

from __future__ import annotations

import json

from nirspipe.qc.common.channel_table import region_coverage
from tests.figure_accuracy._dyad import table_under


def test_each_region_counts_its_long_channels_and_names_the_rejected_ones():
    pairs = {
        "S1_D1": {"separation": "long", "is_bad": False},
        "S1_D2": {"separation": "long", "is_bad": True},
        "S2_D2": {"separation": "n/a", "is_bad": False},
        "S1_D9": {"separation": "short", "is_bad": False},
        "S3_D3": {"separation": "unclassified", "is_bad": False},
    }
    roi_map = {"L": ["S1_D1", "S1_D2 hbo", "S1_D2 hbr", "S1_D9"],
               "R": ["S2_D2", "S3_D3", "S7_D7"],
               "gone": ["S1_D2"]}
    assert region_coverage(pairs, roi_map) == [
        # the chromophore suffix drops and the repeat collapses, as every ROI path reads a map
        {"region": "L", "listed": 2, "kept": 1, "rejected": ["S1_D2"],
         "not_counted": {"S1_D9": "short"}},
        # a montage that never split counts as long, as the tables block it
        {"region": "R", "listed": 1, "kept": 1, "rejected": [],
         "not_counted": {"S3_D3": "between the separation bands",
                         "S7_D7": "not in the recording"}},
        {"region": "gone", "listed": 1, "kept": 0, "rejected": ["S1_D2"], "not_counted": {}},
    ]


def test_the_run_report_prints_each_region_s_long_channels_kept(mini_bids, tmp_path):
    from nirspipe.cli.run import main

    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"front": ["S1_D1", "S2_D2"], "back": ["S3_D3 hbo", "S4_D4"],
                               "scalp": ["S5_D5"], "elsewhere": ["S9_D9"]}))
    out = tmp_path / "out"
    main([str(mini_bids), str(out), "participant", "--participant-label", "01",
          "--task-label", "tapping", "--bad-channels", "S1_D1", "--dpf", "6",
          "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.1", "--resp-h-freq", "0.4", "--roi-mapping", str(roi),
          "--skip-bids-validation"])

    page = out / "sub-01" / "sub-01_task-tapping_report.html"
    # S1_D1 by hand, S3_D3 by screening; S5_D5 is short and S9_D9 is not in the montage
    assert table_under(page, "Region coverage") == [
        ["Region", "Long channels kept", "Rejected"],
        ["front", "1 of 2", "S1_D1"],
        ["back", "1 of 2", "S3_D3"],
        ["scalp", "0 of 0", ""],
        ["elsewhere", "0 of 0", ""],
    ]
    html = page.read_text(encoding="utf-8")
    assert "S5_D5 (short), S9_D9 (not in the recording)" in html
    assert (html.index('<section id="RunSummary"') < html.index('<section id="Regions"')
            < html.index('<section id="ChannelTable"'))


def test_without_a_map_the_report_has_no_region_table():
    from nirspipe.qc.subject.report import _section_regions

    assert _section_regions([{"name": "S1_D1 760", "is_bad": False, "separation": "long"}],
                            None) == {"region_coverage": [], "region_not_counted": ""}
