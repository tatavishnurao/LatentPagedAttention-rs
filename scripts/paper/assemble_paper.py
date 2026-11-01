#!/usr/bin/env python3
"""Inline the generated tables into the paper and estimate its length (no pandoc needed).

Replaces every `<!-- include: generated/<file>.md -->` marker in paper/paper.md with that file
(minus its "generated" comment) and writes two copies, both regenerated, never hand-edited:
- paper/build/paper.md: tables inlined, claim tags kept (the working copy);
- paper/build/paper_submission.md: also with the claim tags ([M:key], [S:key], [H]) and every
  HTML comment stripped, and Appendix B's keyed evidence table replaced by a plain "Evidence
  files" list grouped by topic (the keys mean nothing once the tags are gone).

Image paths: paper/paper.md links figures relative to itself (`../reports/...`, so it renders on
GitHub); both build copies use repository-root-relative paths (`reports/...`), so a pandoc build
must run from the repository root (pandoc resolves images against its working directory by
default). Every image is checked to exist, in the source and in the builds; a missing one fails.

The page estimate is ROUGH and is computed on the submission copy: body words / WORDS_PER_PAGE,
plus each table at its row count (doubled when it must span both columns: more than two columns
and data cells wider than ONE_COLUMN_CHARS in total; headers and two-column tables can wrap),
plus each figure at a fixed fraction of a page, for an ACM-style two-column page. It is a
planning number, not a typeset page count. Every page count quoted from it must say "rough".
The constants are unmeasured heuristics, chosen without a typeset build, in particular
ONE_COLUMN_CHARS = 75: no column width of any venue template was measured. Replace the estimate
with a real build in the venue's template as soon as one is chosen.

usage: assemble_paper.py [--src paper/paper.md] [--out paper/build/paper.md]
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = re.compile(r"<!--\s*include:\s*(\S+)\s*-->")
WORDS_PER_PAGE = 900  # dense two-column text page, rough
ROW_PAGE = 0.012  # one table row in a single column (~1/80 page), rough
TABLE_OVERHEAD = 0.03  # caption, header and rules per table, rough
FIGURE_PAGE = 0.25  # one single-column figure with caption, rough
# Summed data-cell width (+2 per column) above which a table spans both columns.
# UNMEASURED HEURISTIC: no template column width was measured.
ONE_COLUMN_CHARS = 75
TAG = re.compile(r"\s?\[(?:[MS]:[\w-]+(?:\s*,\s*[MS]:[\w-]+)*|H)\]")
APPENDIX_HEADINGS = ("appendix",)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
# Evidence-key topics for the submission copy's "Evidence files" list. An unknown key fails, so a
# new key in the paper must be placed here.
EVIDENCE_TOPICS = (
    ("Timing", ("M:timing", "M:sens", "M:fastacc", "M:prelim")),
    ("Numerics", ("M:scale", "M:emu")),
    ("SASS", ("M:sass",)),
    ("Instrument checks", ("M:void", "M:checks", "M:nvml", "M:t1")),
    ("Sources (code and design documents)", ("S:rules", "S:kern", "S:e4", "S:harness")),
)
EVIDENCE_START = "**Evidence keys.**"


def assemble(src):
    text = src.read_text()

    def inline(m):
        frag = (src.parent / m.group(1)).read_text()
        return re.sub(r"<!--.*?-->\n?", "", frag, flags=re.S).rstrip("\n")

    text = INCLUDE.sub(inline, text)
    return text.replace("](../reports/", "](reports/")


def check_images(text, base):
    """Every image link resolves under base; returns the list, fails on a missing file."""
    links = IMAGE.findall(text)
    missing = [p for p in links if not (base / p).is_file()]
    if missing:
        raise SystemExit(f"missing image(s) relative to {base}: {missing}")
    return links


def evidence_list(text):
    """Replace the keyed evidence table with a key-free list grouped by topic."""
    start = text.find(EVIDENCE_START)
    if start < 0:
        return text
    m = re.compile(r"(?:^\|.*\n)+", re.M).search(text, start)
    if not m:
        raise SystemExit("evidence table not found after the evidence-keys heading")
    files = {}
    for line in m.group(0).splitlines()[2:]:
        key, value = (c.strip() for c in line.strip().strip("|").split("|", 1))
        files[key] = value
    known = {k for _, keys in EVIDENCE_TOPICS for k in keys}
    unknown = sorted(set(files) - known)
    if unknown:
        raise SystemExit(f"evidence keys without a topic in EVIDENCE_TOPICS: {unknown}")
    items = []
    for topic, keys in EVIDENCE_TOPICS:
        present = [files[k] for k in keys if k in files]
        if present:
            items.append(f"- *{topic}:* " + "; ".join(present))
    block = "**Evidence files.** All paths are relative to the repository root.\n\n"
    block += "\n".join(items) + "\n"
    return text[:start] + block + text[m.end() :]


def strip_tags(text):
    """Submission copy: no claim tags, no HTML comments, no evidence keys."""
    text = evidence_list(text)
    text = re.sub(r"\n?<!--.*?-->\n?", "\n", text, flags=re.S)
    text = TAG.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def split_body(text):
    """(body, appendix): the body ends at the first 'Appendix' top-level heading."""
    m = re.search(r"^# (?:%s)" % "|".join(APPENDIX_HEADINGS), text, flags=re.M | re.I)
    return (text[: m.start()], text[m.start() :]) if m else (text, "")


def table_width(lines):
    """Summed widest data cell per column (+2 per column); 0 for two-column tables (they wrap)."""
    cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    data = [r for r in cells[2:] if r]
    if not data or len(cells[0]) <= 2:
        return 0
    cols = len(cells[0])
    return sum(max(len(r[i]) if i < len(r) else 0 for r in data) + 2 for i in range(cols))


def estimate(text):
    text = re.sub(
        r"\A---\n.*?\n---\n", lambda m: m.group(0).split("abstract:", 1)[-1], text, flags=re.S
    )
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    blocks = re.findall(r"(?:^\|.*\n)+", text, flags=re.M)
    tables = len(blocks)
    rows = 0.0
    wide = 0
    for block in blocks:
        lines = block.splitlines()
        span = 2 if table_width(lines) > ONE_COLUMN_CHARS else 1
        wide += span == 2
        rows += max(0, len(lines) - 2) * span
    figures = len(re.findall(r"^!\[", text, flags=re.M))
    prose = "\n".join(
        ln
        for ln in text.splitlines()
        if not ln.lstrip().startswith("|") and not ln.startswith("![")
    )
    words = len(re.findall(r"\S+", prose))
    pages = words / WORDS_PER_PAGE + rows * ROW_PAGE + tables * TABLE_OVERHEAD
    pages += figures * FIGURE_PAGE
    return {
        "words": words,
        "tables": tables,
        "full_width_tables": wide,
        "figures": figures,
        "pages": round(pages, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(ROOT / "paper" / "paper.md"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "build" / "paper.md"))
    ap.add_argument("--submission", default=str(ROOT / "paper" / "build" / "paper_submission.md"))
    a = ap.parse_args(argv)
    src = Path(a.src)
    check_images(src.read_text(), src.parent)
    full = assemble(src)
    images = check_images(full, ROOT)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full)
    clean = strip_tags(full)
    sub = Path(a.submission)
    sub.write_text(clean)
    check_images(clean, ROOT)
    leftover = TAG.findall(clean) + re.findall(r"<!--|\| Key \||\b[MS]:[a-z0-9]+\b", clean)
    if leftover:
        raise SystemExit(f"tags left in the submission copy: {leftover[:5]}")
    body, appendix = split_body(clean)
    b, x = estimate(body), estimate(appendix)
    print(f"ASSEMBLE_OK {out} {sub} images={len(images)} (root-relative, all exist)")
    print(f"body (incl. references): {b}  -> ~{b['pages']} pages (rough)")
    print(f"appendix:                {x}  -> ~{x['pages']} pages (rough)")


if __name__ == "__main__":
    main()
