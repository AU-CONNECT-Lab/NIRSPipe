"""Every flag a subcommand parses has to be a parameter its function accepts.

Each CLI dispatches with ``args.func(**kw)`` over every parsed key, so adding a flag to a
shared parent parser adds it to every command built from that parser, whether or not the
command's function grew a parameter for it. The failure is a `TypeError` on the first real
invocation and nothing else: the flag parses, the help lists it, the parser tests pass, and
the command dies the moment somebody runs it.

It has happened: a flag added to the shared screening block broke `fnirs-qc prep-raw` and
`fnirs-qc hyper-raw`, which no test invoked through `main`. This file is the guard, and it is
a signature comparison rather than a run so it stays fast and needs no data.
"""

import importlib
import inspect

import pytest

# the CLIs that dispatch through set_defaults(func=...)
MODULES = ["fnirs_pipe.cli.qc", "fnirs_pipe.cli.prep", "fnirs_pipe.cli.hyper"]


def _commands(module_name):
    """[(command path, function, {dest names the parser produces})] for one CLI module."""
    module = importlib.import_module(module_name)
    parser = module._build_parser()
    out = []

    def walk(p, path):
        for action in p._actions:
            if isinstance(action, __import__("argparse")._SubParsersAction):
                for name, sub in action.choices.items():
                    walk(sub, f"{path} {name}".strip())
                return
        func = p.get_default("func")
        if func is not None:
            dests = {a.dest for a in p._actions
                     if a.dest not in ("help", "func") and a.dest != "=="}
            out.append((path, func, dests))

    walk(parser, module_name.rsplit(".", 1)[-1])
    return out


ALL = [case for m in MODULES for case in _commands(m)]


def test_the_walk_found_the_commands():
    """A guard that finds nothing passes for the wrong reason."""
    assert len(ALL) >= 10
    assert any("prep-raw" in path for path, _f, _d in ALL)


@pytest.mark.parametrize("path,func,dests",
                         ALL, ids=[c[0] for c in ALL])
def test_every_parsed_flag_is_a_parameter(path, func, dests):
    params = set(inspect.signature(func).parameters)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD
           for p in inspect.signature(func).parameters.values()):
        pytest.skip(f"{path} takes **kwargs, so nothing can be unexpected")
    missing = sorted(dests - params)
    assert not missing, (
        f"`{path}` parses {missing} but {func.__name__} has no parameter for them, so the "
        f"command raises TypeError the moment it is invoked"
    )
