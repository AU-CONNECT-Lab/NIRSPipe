"""The registry of section prose, and its contract with the template that reads it.

No length check here. The three-sentence cap in `test_metric_vocabulary` is a rule about
hover text, which is not read past three sentences; these are paragraphs, they run to four
or five, and how long one should be is a judgement call rather than something to assert.

What is worth asserting is the contract, because it fails silently: a key the template asks
for and the registry does not answer renders an empty paragraph. Nothing raises, nothing
logs, the section quietly loses its explanation, and only a reader who knew the sentence was
there would notice. So the two sides are compared directly.
"""

import re
from pathlib import Path

import fnirs_pipe.qc as qc_pkg
from fnirs_pipe.qc.boilerplate.notes import SECTION_NOTES, section_note

TEMPLATES = Path(qc_pkg.__file__).resolve().parent / "templates"
PACKAGE = Path(qc_pkg.__file__).resolve().parents[1]
CALL = re.compile(r"""section_note\(\s*(['"])([^'"]+)\1""")


def _keys_in(path: Path) -> "set[str]":
    return {key for _quote, key in CALL.findall(path.read_text(encoding="utf-8"))}


def _asked_for() -> "dict[str, set[str]]":
    """Every key the templates ask for, by template name."""
    out = {}
    for path in [*TEMPLATES.glob("*.j2"), *TEMPLATES.glob("*.html")]:
        keys = _keys_in(path)
        if keys:
            out[path.name] = keys
    return out


def _asked_for_by_code() -> "dict[str, set[str]]":
    """Every key the package's Python asks for, by module path."""
    out = {}
    for path in PACKAGE.rglob("*.py"):
        keys = _keys_in(path)
        if keys:
            out[str(path.relative_to(PACKAGE))] = keys
    return out


def test_no_note_is_empty():
    assert [k for k, v in SECTION_NOTES.items() if not v.strip()] == []


# ---- the contract with the templates ----

def test_every_key_a_template_asks_for_is_written():
    # the failure this catches is silent: an unanswered key renders an empty paragraph
    for name, keys in {**_asked_for(), **_asked_for_by_code()}.items():
        missing = keys - set(SECTION_NOTES)
        assert not missing, f"{name} asks for {sorted(missing)}, which nothing answers"


def test_every_note_written_is_asked_for_somewhere():
    asked = set().union(*_asked_for().values(), *_asked_for_by_code().values())
    assert not set(SECTION_NOTES) - asked


def test_a_note_with_slots_is_asked_for_with_them_filled():
    # a slot left unfilled reaches the page as a literal {sci}, which no reader can make
    # sense of and no other check would see. Python is not checked here: a call there that
    # leaves a slot out raises KeyError on the run that reaches it
    for name, keys in _asked_for().items():
        source = (TEMPLATES / name).read_text(encoding="utf-8")
        for key in keys:
            # an unwritten key is the test above's to report, not this one's
            slots = set(re.findall(r"\{([a-z_]+)\}", SECTION_NOTES.get(key, "")))
            if not slots:
                continue
            call = re.search(r"section_note\(\s*'" + re.escape(key)
                             + r"'(.*?)\)\s*(?:\|[^}]*)?\}\}",
                             source, re.S)
            assert call, f"{name}: no call found for {key}"
            passed = set(re.findall(r"(\w+)\s*=", call.group(1)))
            assert slots <= passed, f"{name}: {key} needs {sorted(slots - passed)}"


# ---- the accessor ----

def test_an_unknown_key_is_a_missing_paragraph_rather_than_an_exception():
    assert section_note("no.such.section") == ""


def test_a_slot_is_filled_from_the_caller():
    assert "-5 to 25 s" in section_note("trial_qc", window="-5 to 25 s")


def test_a_note_without_slots_survives_being_handed_values():
    # the template passes what a note needs; a note that stopped needing one must not raise
    plain = section_note("psd_detail.picker")
    assert section_note("psd_detail.picker", unused="x") == plain


def test_the_filter_edge_note_states_the_reach_of_the_high_pass():
    """One period of the cutoff, as the report fills it: no measured ratio to threshold."""
    note = section_note("caveat.filter_edge", edge_s=round(1.0 / 0.01), l_freq=0.01)
    assert "first and last 100 s" in note and "0.01 Hz high-pass" in note
