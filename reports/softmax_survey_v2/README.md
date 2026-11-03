# Where Softmax Lives — v2 (report sources, code and data)

Author: Tata Vishnu Vardhan Rao. Survey with measurements; see `report.md` (paper) and `blog.md` (short version).

| Path | Contents |
|---|---|
| `report.md`, `blog.md` | Markdown sources |
| `site/` | built HTML (`code/build_html.py`) |
| `code/` | experiment, census and build scripts (one per figure; listed in report Appendix B) |
| `data/` | raw JSON and stdout for every figure; `census.json`, `census_table.md` |
| `figures/` | PNG figures |
| `sources/configs/` | the 44 official config.json files read for the census |
| `sources/arxiv_meta.json`, `sources/references.json` | reference metadata (full author lists from the arXiv API) |

Reproduce (from this folder):

```bash
pip install -r requirements.txt
```

```bash
cd code && for e in e1_dispersion e2_temperature e3_random_logits e4_precision e5_cost_model census; do python $e.py | tee ../data/${e}_stdout.txt; done
```

```bash
python e6_e7_probes.py --smoke
```

```bash
python build_html.py
```

Build scripts: `code/build_html.py` renders `report.md` and `blog.md` to `site/*.html`. It converts
`$...$` maths to MathML (latex2mathml), turns `![alt](src "caption")` into figures, numbers `[@key]`
citations, and writes the reference list from `sources/references.json` + `sources/arxiv_meta.json`.
`code/style.py` holds the plotting style and output paths used by every experiment script.

E6/E7 (`code/e6_e7_probes.py`) were not run: the Hugging Face weight CDN (`us.aws.cdn.hf.co`) was
blocked by the session's network policy. `--smoke` checks the code on random tiny configs only.
