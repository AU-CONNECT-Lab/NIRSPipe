"""participants.tsv is matched on the bare label whether or not its ids carry ``sub-``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nirspipe.io.bids import get_participant_age


@pytest.mark.parametrize("pid", ["sub-01", "01"])
def test_the_age_is_found_with_or_without_the_prefix(tmp_path, pid):
    (tmp_path / "participants.tsv").write_text(f"participant_id\tage\n{pid}\t7.5\n")
    layout = SimpleNamespace(root=str(tmp_path))

    assert get_participant_age(layout, "01") == 7.5
    assert get_participant_age(layout, "sub-01") == 7.5
