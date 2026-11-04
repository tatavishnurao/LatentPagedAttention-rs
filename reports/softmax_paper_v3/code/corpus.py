"""Public-domain text from Project Gutenberg (ids 1342 Pride and Prejudice, 2701 Moby Dick,
84 Frankenstein, 1661 Sherlock Holmes). The Gutenberg header and licence footer are stripped.
Files are read from $CORPUS_DIR (default ../corpus); fetch them with:
  curl -sL -o pg<ID>.txt https://www.gutenberg.org/cache/epub/<ID>/pg<ID>.txt"""
import os
import re

CORPUS_DIR = os.environ.get("CORPUS_DIR", os.path.join(os.path.dirname(__file__), "..", "corpus"))


def load_text(name):
    t = open(os.path.join(CORPUS_DIR, name + ".txt"), encoding="utf-8").read()
    a = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t)
    b = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
    t = t[a.end() if a else 0: b.start() if b else len(t)]
    return re.sub(r"\s+", " ", t).strip()
