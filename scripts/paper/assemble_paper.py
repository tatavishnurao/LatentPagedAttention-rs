#!/usr/bin/env python3
"""Inline the generated tables into the paper and estimate its length (no pandoc needed).

Replaces every `<!-- include: generated/<file>.md -->` marker in paper/paper.md with that file
(minus its "generated" comment) and writes two copies, both regenerated, never hand-edited:
- paper/build/paper.md: tables inlined, claim tags kept (the working copy);
- paper/build/paper_submission.md: also with the claim tags ([M:key], [S:key], [H]) and every
  HTML comment stripped, for conversion once a venue is chosen.
Figure paths are rewritten relative to paper/build/.

The page estimate is ROUGH and is computed on the submission copy: body words / WORDS_PER_PAGE,
plus each table at its row count (doubled when it must span both columns: more than two columns
and data cells wider than ONE_COLUMN_CHARS in total; headers and two-column tables can wrap),
plus each figure at a fixed fraction of a page, for an ACM-style two-column page. It is a
planning number, not a typeset page count.

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
ONE_COLUMN_CHARS = (
    75  # summed data-cell width (+2 per column) above which a table spans both columns
)
TAG = re.compile(r"\s?\[(?:[MS]:[\w-]+(?:\s*,\s*[MS]:[\w-]+)*|H)\]")
APPENDIX_HEADINGS = ("appendix",)


def assemble(src):
    text = src.read_text()

    def inline(m):
        frag = (src.parent / m.group(1)).read_text()
        return re.sub(r"<!--.*?-->\n?", "", frag, flags=re.S).rstrip("\n")

    text = INCLUDE.sub(inline, text)
    return text.replace("](../reports/", "](../../reports/")


def strip_tags(text):
    """Submission copy: no claim tags, no HTML comments."""
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
    full = assemble(Path(a.src))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full)
    clean = strip_tags(full)
    sub = Path(a.submission)
    sub.write_text(clean)
    leftover = TAG.findall(clean) + re.findall(r"<!--", clean)
    if leftover:
        raise SystemExit(f"tags left in the submission copy: {leftover[:5]}")
    body, appendix = split_body(clean)
    b, x = estimate(body), estimate(appendix)
    print(f"ASSEMBLE_OK {out} {sub}")
    print(f"body (incl. references): {b}  -> ~{b['pages']} pages (rough)")
    print(f"appendix:                {x}  -> ~{x['pages']} pages (rough)")


if __name__ == "__main__":
    main()
