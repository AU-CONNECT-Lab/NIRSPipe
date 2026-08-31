"""Auto-generate Methods paragraph and reference list for the report."""

import importlib.metadata
import re
import unicodedata
import sys
from importlib.resources import files
from typing import Any


# ---- Citation rendering ----

def _clean_field(s: str) -> str:
    """Strip LaTeX brace groups, keeping their content (e.g. {TDDR} → TDDR)."""
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'\{([^{}]*)\}', r'\1', s)
    return s.strip()


def _last_name(author_token: str) -> str:
    author_token = author_token.strip()
    if "," in author_token:
        return author_token.split(",")[0].strip()
    parts = author_token.split()
    return parts[-1] if parts else author_token


def _initials(name_part: str) -> str:
    """'Frank A.' or 'Frank Alan' → 'F. A.'"""
    return " ".join(p[0] + "." for p in name_part.split() if p and p[0].isalpha())


def _apa_author(token: str) -> str:
    """Format one BibTeX author token as APA 'Last, F. M.'"""
    token = token.strip()
    if not token or token.lower() == "others":
        return ""
    if "," in token:
        last, first = token.split(",", 1)
        initials = _initials(first.strip())
        return f"{last.strip()}, {initials}" if initials else last.strip()
    parts = token.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-1]}, {_initials(' '.join(parts[:-1]))}"


def _apa_authors(author_field: str) -> str:
    """Format full author list in APA 7 style."""
    tokens = [t.strip() for t in author_field.split(" and ") if t.strip() and t.strip().lower() != "others"]
    formatted = [a for a in (_apa_author(t) for t in tokens) if a]
    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def _cite_plain(entry: dict) -> str:
    """In-text citation: (Author et al., Year)."""
    authors = [a.strip() for a in entry.get("author", "").split(" and ") if a.strip()]
    year = entry.get("year", "")
    if not authors:
        return year
    first = _last_name(authors[0])
    real_authors = [a for a in authors if a.lower() != "others"]
    if len(real_authors) == 1:
        return f"{first}, {year}"
    if len(real_authors) == 2:
        return f"{first} & {_last_name(real_authors[1])}, {year}"
    return f"{first} et al., {year}"


def _fmt_citations(keys: list[str], refs: dict, fmt: str) -> str:
    if not keys:
        return ""
    if fmt == "latex":
        return r"\cite{" + ",".join(keys) + "}"
    parts = [_cite_plain(refs[k]) if k in refs else k for k in keys]
    text = "(" + "; ".join(parts) + ")"
    if fmt == "html":
        return f'<span class="boilerplate-cite">{text}</span>'
    return text


# ---- Resource loading ----

def _load_steps() -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    data = files("fnirs_pipe.qc.boilerplate").joinpath("steps.toml").read_bytes()
    return tomllib.loads(data.decode())


# BibTeX stores accented letters as LaTeX commands, so Yucel is written Y{\"u}cel. Left
# alone it reaches the Methods paragraph a user copies into a manuscript.
_ACCENTS = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303",
    ".": "\u0307", "=": "\u0304", "c": "\u0327", "v": "\u030c", "u": "\u0306",
    "H": "\u030b",
}
_ACCENT_BARE = re.compile(r'\{?\\(["\'`^~.=])\s*\{?([A-Za-z])\}?\}?')
# letter accents need the braces, otherwise \url would match as a breve
_ACCENT_BRACED = re.compile(r'\{?\\([cvuH])\{([A-Za-z])\}\}?')
_LIGATURES = (("ss", "\u00df"), ("aa", "\u00e5"), ("AA", "\u00c5"),
              ("o", "\u00f8"), ("O", "\u00d8"), ("ae", "\u00e6"), ("AE", "\u00c6"))


def _delatex(value: str) -> str:
    """LaTeX-escaped BibTeX field to plain Unicode.

    'Y{\\"u}cel' -> 'Yucel' with an umlaut; '{fNIRS} review' -> 'fNIRS review'
    """
    def _apply(m: "re.Match") -> str:
        return unicodedata.normalize("NFC", m.group(2) + _ACCENTS[m.group(1)])

    value = _ACCENT_BARE.sub(_apply, value)
    value = _ACCENT_BRACED.sub(_apply, value)
    for cmd, ch in _LIGATURES:
        value = re.sub(r"\\" + cmd + r"(?![A-Za-z])", ch, value)
    # what is left is BibTeX case protection, e.g. {fNIRS} or {van der Berg}
    return value.replace("{", "").replace("}", "")


def _load_refs() -> dict[str, dict]:
    import bibtexparser
    text = files("fnirs_pipe.qc.boilerplate").joinpath("references.bib").read_text(encoding="utf-8")
    lib = bibtexparser.parse_string(text)
    return {
        entry.key: {name: _delatex(field.value) for name, field in entry.fields_dict.items()}
        for entry in lib.entries
    }


# ---- Step assembly ----

def _active_steps(prep_config: Any, post_config: Any, mode: str | None) -> list[tuple[str, dict]]:
    dpf_str = ", ".join(str(d) for d in prep_config.dpf)

    result = [
        ("od_conversion", {}),
        ("sci_marking", {
            "threshold": str(prep_config.sci_threshold),
            "action": "marked as bad and excluded from further analysis",
        }),
    ]

    mc = prep_config.motion_correction
    if mc != "none":
        result.append((f"motion_{mc}", {}))

    result.append(("beer_lambert", {"dpf": dpf_str}))

    if post_config is not None:
        hp = getattr(post_config, "high_pass", None)
        lp = getattr(post_config, "low_pass", None)
        if hp and lp:
            result.append(("bandpass", {"l_freq": str(hp), "h_freq": str(lp)}))
        elif hp:
            result.append(("highpass", {"l_freq": str(hp)}))
        elif lp:
            result.append(("lowpass", {"h_freq": str(lp)}))

        if getattr(post_config, "resample_sfreq", None):
            result.append(("resample", {"sfreq": str(post_config.resample_sfreq)}))

        if mode == "glm":
            result.append(("glm", {
                "hrf_model": post_config.hrf_model,
                "noise_model": post_config.noise_model,
                "drift_model": post_config.drift_model,
                "drift_high_pass": str(post_config.drift_high_pass),
            }))
        elif mode in ("rest", "denoise") and (
                post_config.short_channel or post_config.drift_model not in (None, "none")):
            from fnirs_pipe.qc.boilerplate.vocabulary import template_slots
            result.append(("confound_regression", template_slots("confound_regression", {
                "short_channel": post_config.short_channel,
                "drift_model": post_config.drift_model,
                "drift_high_pass": post_config.drift_high_pass,
                "drift_order": post_config.drift_order,
            })))

    return result


def _render_step(template: str, params: dict, citations: str) -> str:
    if citations:
        params = {**params, "citations": citations}
    else:
        template = re.sub(r'\s*\{citations\}', '', template)
    return template.format_map(params)


def _collect_prose(active: list[tuple[str, dict]], steps: dict, refs: dict, fmt: str) -> list[str]:
    out = []
    for key, params in active:
        if key not in steps:
            continue
        step = steps[key]
        cite_str = _fmt_citations(step.get("citations", []), refs, fmt)
        out.append(_render_step(step["plain"], params, cite_str))
    return out


def _with_connectives(sentences: list[str]) -> list[str]:
    """Prepend 'Finally, ' to the last sentence to soften paragraph end."""
    if len(sentences) <= 1:
        return sentences
    last = sentences[-1]
    if last and last[0].isupper():
        last = "Finally, " + last[0].lower() + last[1:]
    else:
        last = "Finally, " + last
    return sentences[:-1] + [last]


def _apa_ref(entry: dict) -> str:
    """Format one BibTeX entry as an APA 7 reference string."""
    authors  = _apa_authors(entry.get("author", ""))
    year     = entry.get("year", "n.d.")
    title    = _clean_field(entry.get("title", ""))
    journal  = _clean_field(entry.get("journal", ""))
    volume   = entry.get("volume", "")
    number   = entry.get("number", "")
    pages    = entry.get("pages", "").replace("--", "–")
    doi      = entry.get("doi", "")

    vol_str = ""
    if volume and number:
        vol_str = f"{volume}({number})"
    elif volume:
        vol_str = volume

    source = journal
    if vol_str and pages:
        source += f", {vol_str}, {pages}"
    elif vol_str:
        source += f", {vol_str}"

    line = f"{authors} ({year}). {title}. {source}."
    if doi:
        line += f" https://doi.org/{doi}"
    return line


def _build_html(header_text: str, refs: dict, active: list[tuple[str, dict]], steps: dict) -> str:
    """Render Methods as inline HTML fragment for embedding in the report."""
    prose_html = _with_connectives(_collect_prose(active, steps, refs, "html"))
    paragraph = " ".join([header_text] + prose_html)
    out = [f'<p class="boilerplate-para">{paragraph}</p>']
    reflist = _build_reflist(active, steps, refs)
    if reflist:
        items = "".join(f"<li>{line}</li>" for line in reflist.splitlines())
        out.append('<h4 class="boilerplate-refs-title">References</h4>')
        out.append(f'<ol class="boilerplate-refs">{items}</ol>')
    return "\n".join(out)


def _build_reflist(active: list[tuple[str, dict]], steps: dict, refs: dict) -> str:
    seen: list[str] = []
    for key, _ in active:
        if key in steps:
            for ck in steps[key].get("citations", []):
                if ck not in seen:
                    seen.append(ck)
    if not seen:
        return ""
    lines = []
    for ck in seen:
        lines.append(_apa_ref(refs[ck]) if ck in refs else ck)
    return "\n".join(lines)


# ---- Public API ----

def generate_methods_text(
    prep_config: Any,
    post_config: Any = None,
    mode: str | None = None,
    versions: dict[str, str] | None = None,
    nirs_dir: Any = None,
) -> dict[str, str]:
    """Methods prose for one run.

    Given nirs_dir, the steps are read from the sidecars that run wrote, so the text
    describes what actually happened; the config is the fallback for a tree with no
    sidecars, and it can only describe what was requested.
    """
    steps = _load_steps()
    refs = _load_refs()
    ver = (versions or {}).get("fnirs-pipe", "unknown")

    active = []
    if nirs_dir is not None:
        from fnirs_pipe.qc.boilerplate.vocabulary import steps_from_sidecars
        active = steps_from_sidecars(nirs_dir, mode)
    if not active:
        active = _active_steps(prep_config, post_config, mode)

    prose_plain = _with_connectives(_collect_prose(active, steps, refs, "plain"))
    prose_md    = _with_connectives(_collect_prose(active, steps, refs, "markdown"))
    prose_latex = _with_connectives(_collect_prose(active, steps, refs, "latex"))
    reflist     = _build_reflist(active, steps, refs)

    header = steps.get("header", {})
    header_plain = header.get("plain", "fNIRS data were preprocessed using fnirs-pipe v{ver}.").format(ver=ver)
    header_md    = header.get("markdown", header_plain).format(ver=ver)
    header_latex = header.get("latex", header_plain).format(ver=ver)

    para_plain = " ".join([header_plain] + prose_plain)
    plain = para_plain
    if reflist:
        plain += "\n\nReferences\n\n" + reflist

    para_md = " ".join([header_md] + prose_md)
    md_refs = ""
    if reflist:
        md_refs = "\n\n### References\n\n" + "\n\n".join(f"- {l}" for l in reflist.splitlines())
    markdown = f"## Methods\n\n{para_md}{md_refs}"

    para_latex = " ".join([header_latex] + prose_latex)
    latex = (
        "\\subsection{fNIRS Preprocessing}\n"
        f"{para_latex}\n"
        "% BibTeX keys: see fnirs_pipe/qc/boilerplate/references.bib"
    )

    html = _build_html(header_plain, refs, active, steps)

    return {"plain": plain, "markdown": markdown, "latex": latex, "html": html}


def step_sentence(step: str | None, params: dict, mode: str | None = None) -> str:
    """One line saying what a step did, for a table rather than a paragraph.

    The Methods sentence where there is one, without its citations; otherwise the plain
    description of a bookkeeping step. Empty for a step this module has never heard of.
    """
    from fnirs_pipe.qc.boilerplate.vocabulary import (
        boilerplate_key, step_summary, template_slots,
    )

    key = boilerplate_key(step, params, mode)
    section = _load_steps().get(key) if key else None
    if not section:
        return step_summary(step)
    return _render_step(section["plain"], template_slots(key, params), "")


def collect_software_versions() -> dict[str, str]:
    from fnirs_pipe import __version__

    out = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "fnirs-pipe": __version__,
    }
    for req in importlib.metadata.requires("fnirs-pipe") or []:
        if "extra ==" in req:
            continue
        name = re.split(r"[>=<!;\s\[]", req)[0].strip()
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "n/a"
    return out
