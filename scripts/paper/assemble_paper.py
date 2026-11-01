#!/usr/bin/env python3
"""Inline the generated tables into the paper and estimate its length (no pandoc needed).

Replaces every `<!-- include: generated/<file>.md -->` marker in paper/paper.md with that file
(minus its "generated" comment) and writes paper/build/paper.md, which pandoc can convert once a
venue is chosen. Figure paths are rewritten relative to paper/build/.

The page estimate is ROUGH: body words / WORDS_PER_PAGE plus each table at its row count and each
figure at a fixed fraction of a page, for an ACM-style two-column page. It is a planning number,
not a typeset page count.

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
APPENDIX_HEADINGS = ("appendix",)


def assemble(src):
    text = src.read_text()

    def inline(m):
        frag = (src.parent / m.group(1)).read_text()
        return re.sub(r"<!--.*?-->\n?", "", frag, flags=re.S).rstrip("\n")

    text = INCLUDE.sub(inline, text)
    return text.replace("](../reports/", "](../../reports/")


def split_body(text):
    """(body, appendix): the body ends at the first 'Appendix' top-level heading."""
    m = re.search(r"^# (?:%s)" % "|".join(APPENDIX_HEADINGS), text, flags=re.M | re.I)
    return (text[: m.start()], text[m.start():]) if m else (text, "")


def estimate(text):
    text = re.sub(r"\A---\n.*?\n---\n", lambda m: m.group(0).split("abstract:", 1)[-1], text,
                  flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    table_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("|")]
    tables = len(re.findall(r"(?:^\|.*\n)+", text, flags=re.M))
    figures = len(re.findall(r"^!\[", text, flags=re.M))
    prose = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("|") and not ln.startswith("![")
    )
    words = len(re.findall(r"\S+", prose))
    rows = max(0, len(table_lines) - 2 * tables)  # minus header and rule lines
    pages = words / WORDS_PER_PAGE + rows * ROW_PAGE + tables * TABLE_OVERHEAD
    pages += figures * FIGURE_PAGE
    return {"words": words, "tables": tables, "table_rows": rows, "figures": figures,
            "pages": round(pages, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(ROOT / "paper" / "paper.md"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "build" / "paper.md"))
    a = ap.parse_args(argv)
    full = assemble(Path(a.src))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full)
    body, appendix = split_body(full)
    b, x = estimate(body), estimate(appendix)
    print(f"ASSEMBLE_OK {out}")
    print(f"body (incl. references): {b}  -> ~{b['pages']} pages (rough)")
    print(f"appendix:                {x}  -> ~{x['pages']} pages (rough)")


if __name__ == "__main__":
    main()
