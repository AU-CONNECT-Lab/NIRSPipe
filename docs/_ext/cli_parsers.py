"""Parser factories for the usage page's argparse directives, one per entry point."""

import argparse
import re

from nirspipe.cli import gui, hyper, prep, qc, rate, run, snirf2bids


def _rst(text):
    # help strings are plain text with Markdown-style `code`; outside the code, reST would
    # read a trailing underscore (BAD_) as a link target, * as emphasis, |r| as a substitution
    if not text or text == argparse.SUPPRESS:
        return text
    parts = re.split(r"`([^`]+)`", text)
    for i, part in enumerate(parts):
        if i % 2:
            parts[i] = f"``{part}``"
        else:
            part = part.replace("*", r"\*").replace("|", r"\|")
            parts[i] = re.sub(r"(\w)_(?=\W|$)", r"\1\\_", part)
    return "".join(parts)


def _escaped(parser):
    parser.description = _rst(parser.description)
    parser.epilog = _rst(parser.epilog)
    for action in parser._actions:
        action.help = _rst(action.help)
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                _escaped(sub)
            for choice in action._choices_actions:
                choice.help = _rst(choice.help)
    return parser


def nirspipe():
    return _escaped(run._build_parser())


def nirspipe_snirf2bids():
    return _escaped(snirf2bids._build_parser())


def nirspipe_prep():
    return _escaped(prep._build_parser())


def nirspipe_qc():
    return _escaped(qc._build_parser())


def nirspipe_hyper():
    return _escaped(hyper._parsers()["nirspipe-hyper"])


def nirspipe_hyper_pairnull():
    return _escaped(hyper._parsers()["nirspipe-hyper-pairnull"])


def nirspipe_hyper_groupnull():
    return _escaped(hyper._parsers()["nirspipe-hyper-groupnull"])


def nirspipe_hyper_band():
    return _escaped(hyper._parsers()["nirspipe-hyper-band"])


def nirspipe_hyper_index():
    return _escaped(hyper._parsers()["nirspipe-hyper-index"])


def nirspipe_hyper_merge():
    return _escaped(hyper._parsers()["nirspipe-hyper-merge"])


def nirspipe_rate():
    return _escaped(rate._build_parser())


def nirspipe_gui():
    return _escaped(gui._build_parser())

