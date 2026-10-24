"""Which head a participant is placed on: fsaverage, or MNE's infant template nearest their age.

No infant template is downloaded here; the geometry is checked on real recordings outside
the suite, and every placement test runs on fsaverage.
"""

import argparse
import re

import pytest

from nirspipe.cli import _app_options
from nirspipe.io.head_template import (CHOICES, FSAVERAGE, INFANT, resolve_template, space_of,
                                       subject_of, template_for_age, template_of_space)


# ages in years; the nearest template worked out in months by hand
@pytest.mark.parametrize("age, template", [
    (0.0, "2wk"),            # a newborn takes the youngest
    (100 / 365.25, "3mo"),   # 3.29 months: 0.29 from 3, 1.21 from 4.5
    (120 / 365.25, "4.5mo"), # 3.94 months: 0.56 from 4.5, 0.94 from 3
    (0.8, "9mo"),            # 9.6 months: 0.6 from 9, 0.9 from 10.5
    (1.8, "2yr"),            # 21.6 months: 2.4 from 24, 3.6 from 18
    (2.0, "2yr"),
    (18.0, FSAVERAGE),
    (45.0, FSAVERAGE),
])
def test_an_age_takes_the_nearest_template(age, template):
    assert template_for_age(age) == template


@pytest.mark.parametrize("age", [2.01, 5.0, 17.9])
def test_an_age_between_the_infant_templates_and_adulthood_asks_for_one(age):
    with pytest.raises(ValueError, match="pass --template"):
        template_for_age(age)


def test_a_given_template_wins_over_the_age():
    assert resolve_template("6mo", 30.0) == "6mo"
    assert resolve_template(FSAVERAGE, 0.5) == FSAVERAGE
    assert resolve_template(None, 0.5) == "6mo"
    # numbers in --dpf leave the age unread, so the default head is the adult one
    assert resolve_template(None, None) == FSAVERAGE


def test_a_template_the_package_does_not_know_is_refused():
    with pytest.raises(ValueError, match="none of"):
        resolve_template("5yr", None)


def test_every_space_label_is_a_bids_label_and_names_its_template_back():
    for template in CHOICES:
        space = space_of(template)
        assert re.fullmatch(r"[A-Za-z0-9]+", space), space
        assert template_of_space(space) == template
    assert space_of("4.5mo") == "ANTS4p5Months3T"
    assert subject_of("ANTS4p5Months3T") == "ANTS4-5Months3T" == INFANT["4.5mo"][0]
    assert subject_of(FSAVERAGE) == FSAVERAGE


def test_the_flag_takes_the_known_templates_only():
    parser = argparse.ArgumentParser()
    _app_options.add_template(parser)
    assert parser.parse_args([]).template is None
    assert parser.parse_args(["--template", "7.5mo"]).template == "7.5mo"
    with pytest.raises(SystemExit):
        parser.parse_args(["--template", "3yr"])


def test_each_infant_template_is_fetched_once_before_the_first_subject(monkeypatch):
    fetched = []
    monkeypatch.setattr(_app_options, "template_dir", fetched.append)
    chosen = _app_options.templates_for(None, ["01", "02", "03", "04"],
                                        {"01": 0.5, "02": 0.52, "03": 30.0, "04": 1.0})
    assert chosen == {"01": "6mo", "02": "6mo", "03": FSAVERAGE, "04": "12mo"}
    assert fetched == ["ANTS12p0Months3T", "ANTS6p0Months3T"]


def test_fsaverage_alone_fetches_nothing_here(monkeypatch):
    fetched = []
    monkeypatch.setattr(_app_options, "template_dir", fetched.append)
    assert _app_options.templates_for(None, ["01"], {}) == {"01": FSAVERAGE}
    assert fetched == []


def test_a_child_with_no_template_stops_the_run_and_names_the_option(monkeypatch, capsys):
    monkeypatch.setattr(_app_options, "template_dir", lambda space: None)
    with pytest.raises(SystemExit):
        _app_options.templates_for(None, ["01", "02"], {"01": 0.5, "02": 5.0})
    err = capsys.readouterr().err
    assert "sub-02" in err and "--template" in err


def test_a_template_that_cannot_be_fetched_stops_the_run(monkeypatch, capsys):
    def offline(space):
        raise RuntimeError(f"the {space} infant template could not be loaded")

    monkeypatch.setattr(_app_options, "template_dir", offline)
    with pytest.raises(SystemExit):
        _app_options.templates_for("6mo", ["01"], {})
    assert "could not be loaded" in capsys.readouterr().err
