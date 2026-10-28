"""The numeric order of source-detector pairs."""

from nirspipe.utils import pair_order


def test_pairs_sort_by_source_then_detector_number():
    names = ["S10_D1 hbo", "S2_D10 760", "S1_D2", "S2_D9 hbr", "S1_D10", "short1"]
    assert sorted(names, key=pair_order) == ["S1_D2", "S1_D10", "S2_D9 hbr", "S2_D10 760",
                                             "S10_D1 hbo", "short1"]
