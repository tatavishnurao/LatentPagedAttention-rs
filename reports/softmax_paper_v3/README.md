# Softmax Numerics and Attention Sinks in LLM Serving: v3 (paper, report, code, data)

Author: Tata Vishnu Vardhan Rao.

| Path | Contents |
|---|---|
| `paper.md` | 4–6 page reproduction and measurement paper (Markdown source) |
| `report.md` | companion technical report (v2 survey and census, updated with E6/E7) |
| `blog.md` | short version |
| `profile_note.md` | one-page note on how the paper fits a graduate application (not part of the paper) |
| `sources/genre_notes.md` | structure of 12 recent measurement and reproduction papers, and the template adopted |
| `sources/references.json`, `sources/arxiv_meta.json` | references: venue, full-text summary, full author lists |
| `code/` | v3 experiments: `probe_attention.py`, `models.py`, `validate_models.py`, `e6_probes.py`, `e7_logprobs.py`, `e4_fp64.py`, `e6_e7_figures.py`, `build_html.py`, `corpus.py`, `style.py` |
| `code/v2/` | v2 experiments E1–E5 and census (results copied into `data/` and `figures/`) |
| `code/gpu/` | B4–B6 GPU scripts and README (not run in the cloud session) |
| `data/` | raw JSON and stdout for every figure and table |
| `figures/` | PNG figures |
| `site/` | built HTML |

Reproduce (CPU; from this folder):

```bash
pip install -r requirements.txt
```

```bash
export HF_MODELS=$PWD/hf_models CORPUS_DIR=$PWD/corpus
```

```bash
cd code && python validate_models.py && python e6_probes.py && python e7_logprobs.py && python e4_fp64.py && python e6_e7_figures.py
```

```bash
python build_html.py
```

Models are fetched into `$HF_MODELS` with `huggingface_hub.snapshot_download` (`Qwen/Qwen3-0.6B`, `google/gemma-3-1b-it` (gated), `HuggingFaceTB/SmolLM2-360M`, `zaydzuhri/vanilla-340M-4096-model`, `zaydzuhri/softpick-340M-4096-model`). The corpus is Project Gutenberg books 1342, 2701, 84 and 1661 (see `code/corpus.py`). E6 took about 2 hours on 4 CPU cores.

`build_html.py` renders the Markdown to `site/*.html`, with MathML, figure captions, numbered citations and hover/focus/tap preview cards built from `sources/references.json`.
