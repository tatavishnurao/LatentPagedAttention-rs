#!/usr/bin/env python3
"""Numbers lint for the paper: every numeric literal in the prose must be traceable.

A literal passes if it is (a) in paper/generated/numbers.json (written by make_tables.py),
(b) in paper/numbers_whitelist.txt (each entry with a written reason), or (c) a numeric token in
a results file under reports/ (.json, .log, .out, .csv, .txt; never prose Markdown, to avoid
circular evidence; xz-compressed samples are read too). Skipped: fenced code blocks, inline code,
HTML comments, link targets, the leading section number of a heading, and the bibliography (a
top-level "# References" section, up to the next top-level heading). A digit followed by
"x" or "×" is checked as the number alone.

Exit status 1 lists every offending line as path:line: literal.

usage: check_numbers.py [FILE ...]   (default: paper/*.md)
"""

import argparse
import json
import lzma
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = re.compile(r"(?<![\w.\-/])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![\w.]*\w)")
RESULT_SUFFIXES = {".json", ".log", ".out", ".csv", ".txt"}


def strip_prose(text):
    """Returns [(line_no, cleaned_line)] with code, comments and link targets removed."""
    text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    out, fenced, biblio = [], False, False
    for no, line in enumerate(text.splitlines(), 1):
        if re.match(r"#\s", line):
            biblio = line[1:].strip().lower() == "references"
        if biblio:
            continue
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = re.sub(r"`[^`]*`", " ", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        line = re.sub(r"https?://\S+", " ", line)
        line = re.sub(r"^(#+\s*)(?:[A-Z]\.)?[\d.]+\s", r"\1", line)
        line = re.sub(r"(\d)\s?[x×](?![A-Za-z])", r"\1 ", line)
        out.append((no, line))
    return out


def literals(line):
    return [m.group(0).lstrip("-") for m in NUM.finditer(line)]


def result_tokens(root=ROOT / "reports"):
    tokens = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in RESULT_SUFFIXES:
            text = path.read_text(errors="replace")
        elif path.name.endswith(".jsonl.xz"):
            text = lzma.open(path, "rt").read()
        elif path.suffix == ".jsonl":
            text = path.read_text(errors="replace")
        else:
            continue
        tokens.update(m.group(0).lstrip("-") for m in NUM.finditer(text))
    return tokens


def load_whitelist(path):
    allowed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                if "#" not in line:
                    raise SystemExit(f"whitelist entry without a reason: {line!r}")
                allowed.add(entry)
    return allowed


def check(files, registry, whitelist, tokens):
    bad = []
    for path in files:
        for no, line in strip_prose(Path(path).read_text()):
            for lit in literals(line):
                if lit in registry or lit in whitelist or lit in tokens:
                    continue
                bad.append(f"{path}:{no}: {lit}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*")
    args = ap.parse_args(argv)
    files = args.files or sorted(str(p) for p in (ROOT / "paper").glob("*.md"))
    reg_path = ROOT / "paper" / "generated" / "numbers.json"
    registry = set(json.loads(reg_path.read_text())) if reg_path.exists() else set()
    whitelist = load_whitelist(ROOT / "paper" / "numbers_whitelist.txt")
    bad = check(files, registry, whitelist, result_tokens())
    if bad:
        print("\n".join(bad))
        print(f"CHECK_NUMBERS_FAIL {len(bad)} untraceable literal(s) in {len(files)} file(s)")
        return 1
    print(f"CHECK_NUMBERS_OK files={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
