"""participants.tsv is matched on the bare label whether or not its ids carry ``sub-``, and an age is never guessed."""

from __future__ import annotations

import pytest

from nirspipe.io.bids import get_participant_age


@pytest.mark.parametrize("pid", ["sub-01", "01"])
def test_the_age_is_found_with_or_without_the_prefix(tmp_path, pid):
    (tmp_path / "participants.tsv").write_text(f"participant_id\tage\n{pid}\t7.5\n")

    assert get_participant_age(tmp_path, "01") == 7.5
    assert get_participant_age(tmp_path, "sub-01") == 7.5


def test_an_age_in_months_is_read_in_years(tmp_path):
    (tmp_path / "participants.tsv").write_text("participant_id\tage\nsub-01\t6\n")
    (tmp_path / "participants.json").write_text('{"age": {"Units": "months"}}')
    assert get_participant_age(tmp_path, "01") == pytest.approx(0.5)


@pytest.mark.parametrize("rows", ["participant_id\tage\nsub-01\tn/a\n",
                                  "participant_id\nsub-01\n",
                                  "participant_id\tage\nsub-02\t9\n"])
def test_a_missing_age_is_refused_rather_than_guessed(tmp_path, rows):
    (tmp_path / "participants.tsv").write_text(rows)
    with pytest.raises(ValueError, match="gives no age for sub-01"):
        get_participant_age(tmp_path, "01")


def test_an_unknown_unit_is_refused(tmp_path):
    (tmp_path / "participants.tsv").write_text("participant_id\tage\nsub-01\t6\n")
    (tmp_path / "participants.json").write_text('{"age": {"Units": "decades"}}')
    with pytest.raises(ValueError, match="decades"):
        get_participant_age(tmp_path, "01")
