#!/usr/bin/env python3
"""Build the submission PDF with the unmodified IEEEtran conference class.

Input: paper/build/paper_submission.md (run assemble_paper.py first). Output: paper/ieee/paper.tex
and paper/ieee/paper.pdf. The document class is `\\documentclass[conference]{IEEEtran}` with no
margin, font or spacing changes; the only packages added are for input encoding, Unicode
characters, graphics, table rules and URLs.

Prose goes through pandoc (markdown -> LaTeX). Tables, figures and the reference list are built
here, because pandoc's longtable output does not work in two-column mode:
- each Markdown pipe table plus its following "*Table N: caption*" paragraph becomes a
  table (or table*, for those listed in WIDE_TABLES) with the caption above, IEEE style;
- each image becomes figure (or figure*, WIDE_FIGURES) with \\includegraphics;
- "# References" entries "[n] ..." become a thebibliography list in the same order;
- "Section N", "Table N" and "Figure N" in the text become \\ref to the right label, because
  IEEEtran numbers sections and tables with Roman numerals.

Afterwards it reports the page count and the page on which the references start (from the PDF
text, via pdftotext), so the "everything except references" count is read off the real build.

usage: build_ieee.py [--src paper/build/paper_submission.md] [--out-dir paper/ieee]
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIDE_FIGURES = {1, 2}  # dataflow and clock traces span both columns
# Main results (4) spans both columns as planned; 2, 3 and 5 also must, because their natural
# width is 72-111 pt more than one 252 pt column (measured from the overfull boxes of a build with
# them single-column). Table 6 is 18 pt too wide: a smaller column gap, scoped to that table.
WIDE_TABLES = {2, 3, 4, 5}
TIGHT_TABLES = {6}
KEY_VALUE_TABLES = {1}  # setup table: its values are sentences, so the value column wraps
KEY_VALUE_SPEC = r"lp{0.57\columnwidth}"
SEP = "ZZSEPARATORZZ"

PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{newunicodechar}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{url}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{′}{\ensuremath{'}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{µ}{\textmu}
\newunicodechar{ᵀ}{\ensuremath{^{\mathsf{T}}}}
\providecommand{\tightlist}{}
"""


def pandoc(md):
    out = subprocess.run(
        ["pandoc", "-f", "markdown-auto_identifiers-smart", "-t", "latex", "--wrap=none"],
        input=md,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def pandoc_many(snippets):
    """Convert several inline snippets in one pandoc call."""
    if not snippets:
        return []
    joined = f"\n\n{SEP}\n\n".join(snippets)
    parts = pandoc(joined).split(SEP)
    out = [p.strip() for p in parts]
    if len(out) != len(snippets):
        raise SystemExit("pandoc split mismatch")
    return out


def front_matter(text):
    m = re.match(r"---\n(.*?)\n---\n", text, flags=re.S)
    fm, body = m.group(1), text[m.end() :]
    title = re.search(r'^title: "(.*)"$', fm, flags=re.M).group(1)
    author = re.search(r'^author: "(.*)"$', fm, flags=re.M).group(1)
    abstract = re.search(r"^abstract: \|\n((?:  .*\n?)+)", fm, flags=re.M).group(1)
    abstract = " ".join(line.strip() for line in abstract.splitlines())
    return title, author, abstract, body


def urls_to_autolinks(s):
    return re.sub(r"(?<![<(])(https?://[^\s)]+?)([.,;]?)(?=\s|$|\))", r"<\1>\2", s)


def tabular(lines, two_col_spec=KEY_VALUE_SPEC):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    header, data = rows[0], rows[2:]
    cells = pandoc_many([c if c else "~" for r in [header] + data for c in r])
    it = iter(cells)
    conv = [[next(it) for _ in r] for r in [header] + data]
    spec = two_col_spec if len(header) == 2 and two_col_spec else "l" * len(header)
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    out.append(" & ".join(conv[0]) + r" \\")
    out.append(r"\midrule")
    out += [" & ".join(r) + r" \\" for r in conv[1:]]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def build(src, out_dir):
    title, author, abstract, body = front_matter(src.read_text())
    refs_at = body.index("\n# References")
    body, refs = body[:refs_at], body[refs_at:]
    blocks = re.split(r"\n\s*\n", body)
    keep, floats = [], {}
    t_no = f_no = 0
    i = 0
    while i < len(blocks):
        b = blocks[i].strip("\n")
        if b.startswith("|"):
            t_no += 1
            cap = ""
            if i + 1 < len(blocks) and re.match(r"\*Table \d+:", blocks[i + 1].strip()):
                cap = re.sub(r"^\*Table \d+:\s*|\*$", "", blocks[i + 1].strip())
                i += 1
            floats[f"TABLEPH{t_no}"] = ("table", t_no, b.splitlines(), cap)
            keep.append(f"TABLEPH{t_no}")
        elif re.match(r"!\[", b):
            f_no += 1
            m = re.match(r"!\[(.*)\]\(([^)\s]+)\)\s*$", b, flags=re.S)
            floats[f"FIGUREPH{f_no}"] = ("figure", f_no, m.group(2), m.group(1))
            keep.append(f"FIGUREPH{f_no}")
        else:
            keep.append(urls_to_autolinks(b))
        i += 1
    latex = pandoc("\n\n".join(keep))
    # Section labels, in order; then Section/Table/Figure N -> \ref.
    n = 0

    def label(m):
        nonlocal n
        n += 1
        return f"{m.group(0)}\\label{{sec:{n}}}"

    latex = re.sub(r"\\section\{[^}]*\}", label, latex)
    caps = [f[3] for f in floats.values()]
    caps_tex = pandoc_many([urls_to_autolinks(c) for c in caps])
    for (key, f), cap in zip(floats.items(), caps_tex, strict=True):
        if f[0] == "table":
            env = "table*" if f[1] in WIDE_TABLES else "table"
            tight = "\\setlength{\\tabcolsep}{4pt}\n" if f[1] in TIGHT_TABLES else ""
            spec = KEY_VALUE_SPEC if f[1] in KEY_VALUE_TABLES else None
            tex = (
                f"\\begin{{{env}}}[t]\n\\centering\n{tight}"
                f"\\caption{{{cap}}}\\label{{tab:{f[1]}}}\n"
                f"{tabular(f[2], spec)}\n\\end{{{env}}}"
            )
        else:
            env = "figure*" if f[1] in WIDE_FIGURES else "figure"
            width = r"\textwidth" if f[1] in WIDE_FIGURES else r"\columnwidth"
            path = Path("../..") / f[2]
            tex = (
                f"\\begin{{{env}}}[t]\n\\centering\n\\includegraphics[width={width}]{{{path}}}\n"
                f"\\caption{{{cap}}}\\label{{fig:{f[1]}}}\n\\end{{{env}}}"
            )
        latex = latex.replace(key, tex, 1)
    latex = re.sub(r"Sections? (\d+)\b", lambda m: f"Section~\\ref{{sec:{m.group(1)}}}", latex)
    latex = re.sub(r"Table (\d+)\b", lambda m: f"Table~\\ref{{tab:{m.group(1)}}}", latex)
    latex = re.sub(r"Figure (\d+)", lambda m: f"Fig.~\\ref{{fig:{m.group(1)}}}", latex)
    entries = re.findall(r"^\[(\d+)\] (.+)$", refs, flags=re.M)
    nums = [int(k) for k, _ in entries]
    if nums != list(range(1, len(nums) + 1)):
        raise SystemExit(f"references are not numbered 1..n in order: {nums}")
    items = pandoc_many([urls_to_autolinks(t) for _, t in entries])
    bib = "\\begin{thebibliography}{99}\n" + "\n".join(
        f"\\bibitem{{r{k}}} {t}" for (k, _), t in zip(entries, items, strict=True)
    )
    bib += "\n\\end{thebibliography}\n"
    title_tex, abstract_tex, author_tex = pandoc_many([title, abstract, author])
    doc = (
        PREAMBLE
        + "\\begin{document}\n"
        + f"\\title{{{title_tex}}}\n"
        + f"\\author{{\\IEEEauthorblockN{{{author_tex}}}}}\n"
        + "\\maketitle\n"
        + f"\\begin{{abstract}}\n{abstract_tex}\n\\end{{abstract}}\n\n"
        + latex
        + "\n"
        + bib
        + "\\end{document}\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tex = out_dir / "paper.tex"
    tex.write_text(doc)
    run = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
        cwd=out_dir,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if run.returncode != 0:
        print(run.stdout[-3000:])
        raise SystemExit("latexmk failed")
    return out_dir / "paper.pdf", out_dir / "paper.log"


def page_report(pdf, log):
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, flags=re.M).group(1))
    ref_page = None
    for p in range(1, pages + 1):
        txt = subprocess.run(
            # Raw (non-layout) text: with -layout the heading shares a line with the other column.
            ["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"],
            capture_output=True,
            text=True,
            errors="replace",
        ).stdout
        if re.search(r"^\s*R\s*EFERENCES\s*$", txt, flags=re.M):
            ref_page = p
            break
    text = log.read_text(errors="replace")
    overfull = re.findall(r"^Overfull \\hbox \(([\d.]+)pt too wide\)", text, flags=re.M)
    undefined = re.findall(r"Reference `([^']+)' on page \d+ undefined", text)
    return pages, ref_page, overfull, sorted(set(undefined))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(ROOT / "paper" / "build" / "paper_submission.md"))
    ap.add_argument("--out-dir", default=str(ROOT / "paper" / "ieee"))
    a = ap.parse_args(argv)
    for tool in ("pandoc", "latexmk", "pdfinfo", "pdftotext"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found")
    pdf, log = build(Path(a.src), Path(a.out_dir))
    pages, ref_page, overfull, undefined = page_report(pdf, log)
    print(f"BUILD_IEEE_OK {pdf}")
    print(f"pages total: {pages}; references start on page: {ref_page}")
    print(f"overfull hboxes: {len(overfull)} (max {max(map(float, overfull), default=0):.1f} pt)")
    if undefined:
        print(f"UNDEFINED references: {undefined}")


if __name__ == "__main__":
    main()
