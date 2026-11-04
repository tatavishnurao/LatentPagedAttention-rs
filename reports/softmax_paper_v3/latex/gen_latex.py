"""Generate LaTeX inputs from the data, so no number in the PDF is typed by hand.

Writes:
  generated/refs.bib    from ../sources/references.json + ../sources/arxiv_meta.json (full author lists)
  generated/macros.tex  \newcommand per reported number, read from ../data/*.json
  generated/tab_*.tex   tables built from the same data
  figures/*             copied from ../figures (PNG; regenerate from ../code for vector output)
Run: python gen_latex.py"""
import glob
import json
import math
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
GEN = os.path.join(HERE, "generated")
os.makedirs(GEN, exist_ok=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)


def tex_escape(s):
    rep = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(c, c) for c in str(s))


def J(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------- bibliography ----------------
refs = json.load(open(os.path.join(ROOT, "sources", "references.json")))["refs"]
meta = json.load(open(os.path.join(ROOT, "sources", "arxiv_meta.json")))
bib = []
for k, r in refs.items():
    if "arxiv" in r:
        m = meta[r["arxiv"]]
        authors = " and ".join("{" + tex_escape(a.strip()) + "}" if a.strip().endswith("Team") or " " not in a.strip()
                               else tex_escape(a.strip()) for a in m["authors"] if a.strip() not in (":",))
        venue = r["venue"]
        note = f"arXiv:{r['arxiv']} (read as {m['version']})"
        if venue in ("arXiv preprint", "Model card") or venue.startswith("Technical report"):
            bib.append(f"@misc{{{k},\n  author = {{{authors}}},\n  title = {{{{{tex_escape(m['title'])}}}}},\n"
                       f"  year = {{{m['published'][:4]}}},\n  howpublished = {{{tex_escape(venue)}}},\n"
                       f"  note = {{{note}}},\n  url = {{https://arxiv.org/abs/{r['arxiv']}}}\n}}")
        else:
            yr = re.search(r"(19|20)\d\d", venue)
            bib.append(f"@inproceedings{{{k},\n  author = {{{authors}}},\n  title = {{{{{tex_escape(m['title'])}}}}},\n"
                       f"  booktitle = {{{tex_escape(venue)}}},\n  year = {{{yr.group(0) if yr else m['published'][:4]}}},\n"
                       f"  note = {{{note}}},\n  url = {{https://arxiv.org/abs/{r['arxiv']}}}\n}}")
    else:
        authors = " and ".join("{" + tex_escape(a) + "}" for a in r["authors"])
        yr = re.search(r"(19|20)\d\d", r["venue"])
        bib.append(f"@misc{{{k},\n  author = {{{authors}}},\n  title = {{{{{tex_escape(r['title'])}}}}},\n"
                   f"  year = {{{yr.group(0) if yr else '2026'}}},\n  howpublished = {{{tex_escape(r['venue'])}}},\n"
                   f"  url = {{{r['url']}}}\n}}")
open(os.path.join(GEN, "refs.bib"), "w").write("\n\n".join(bib) + "\n")

# ---------------- macros ----------------
M = {}


def sci(v, d=1):
    m, e = f"{v:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def put(name, val, fmt="{:.3f}"):
    if isinstance(val, str):
        M[name] = val
    elif fmt.endswith("e}"):
        M[name] = sci(val, int(fmt[3]))
    else:
        M[name] = fmt.format(val)


col = J("e4_collapse.json")
fp64 = J("e4_fp64.json")
b6 = J("b6_thresholds_cpu.json")
canc = J("e4_cancellation.json")
e3 = J("e3_random_logits.json")
fwd = J("e4_forward_logprob.json")
val = J("validate_models.json")
summ = J("e6_e7_summary.json")

pm = lambda key: col["setups"][key]["first_zero_paper_margin"]
put("cliffBfTwo", pm("C2|bfloat16|pure")); put("cliffFpTwo", pm("C2|float32|pure"))
put("cliffBfMany", pm("C113|bfloat16|pure")); put("cliffFpMany", pm("C113|float32|pure"))
put("cliffHalfMany", pm("C113|float16|pure"))
f64 = {r["C"]: r for r in fp64["rows"]}
put("cliffDblTwo", f64[2]["paper_margin"]); put("cliffDblMany", f64[113]["paper_margin"])
put("cliffDblEarly", f64[2]["paper_margin"] - f64[113]["paper_margin"], "{:.2f}")
put("cliffFpEarly", pm("C2|float32|pure") - pm("C113|float32|pure"), "{:.2f}")
put("fixHalfZeroGap", col["setups"]["C113|float16|fixed"]["first_zero_gap"], "{:.3f}")
put("fixRelErrFp", col["fixed_accuracy_C113"]["float32"]["fixed_max_rel_err_normal"], "{:.1e}")
put("fixRelErrBf", col["fixed_accuracy_C113"]["bfloat16"]["fixed_max_rel_err_normal"], "{:.1e}")
for d, n in (("float32", "Fp"), ("bfloat16", "Bf"), ("float16", "Half")):
    put(f"cancFloor{n}", canc["dtypes"][d]["err"]["all-dtype"][0], "{:.2e}")
put("cancDecompStudent", canc["dtypes"]["bfloat16"]["eps0_decomposition"]["only_student_logsoftmax_in_dtype"], "{:.2e}")
put("cancDecompTeacher", canc["dtypes"]["bfloat16"]["eps0_decomposition"]["only_teacher_probs_in_dtype"], "{:.2e}")
put("cancMixedBf", canc["dtypes"]["bfloat16"]["err"]["teacher-fp32"][0], "{:.1e}")
pi = e3["poly_info"]
put("polyLo", pi["H_min_sharpen"]); put("polyHi", pi["H_max_sharpen"]); put("polyBetaMax", pi["beta_max"], "{:.2f}")
put("logpBfGap", fwd["C151936|bfloat16"]["first_exact_zero_gap"], "{:.2f}")
put("logpFpGap", fwd["C151936|float32"]["first_exact_zero_gap"], "{:.2f}")
for m, n in (("qwen3-0.6b", "Qwen"), ("gemma3-1b-it", "Gemma"), ("smollm2-360m", "Smol"), ("vanilla-340m", "Vanilla"),
             ("softpick-340m", "Softpick")):
    if val and m in val:
        put(f"ppl{n}", val[m]["ppl_1024_eager"], "{:.1f}")
    if summ and m in summ["e6"]:
        rows = summ["e6"][m]["rows"]
        put(f"sinkFirst{n}", 100 * rows[0]["sink_rate_0p3"], "{:.1f}")
        put(f"sinkLast{n}", 100 * rows[-1]["sink_rate_0p3"], "{:.1f}")
        put(f"sinkLastN{n}", f"{rows[-1]['N']:,}")
        nd = summ["e6"][m]["needle"]["rows"]
        put(f"needleFixedFirst{n}", nd[0]["w_fixed"]); put(f"needleFixedLast{n}", nd[-1]["w_fixed"], "{:.4f}")
        put(f"needleBestLast{n}", nd[-1]["w_best"])
        put(f"needleTopOne{n}", f"{sum(x['top1_rate'] * 3 for x in nd):.0f}/{3 * len(nd)}")
open(os.path.join(GEN, "macros.tex"), "w").write(
    "% generated by gen_latex.py from ../data; do not edit\n" +
    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()) + "\n")

# ---------------- tables ----------------
lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"dtype & published & $C{=}2$ & $C{=}113$ & $C{=}1000$ & $C{=}151{,}936$ \\", r"\midrule"]
b6m = {(r["C"], r["dtype"]): r["paper_margin"] for r in b6["rows"] if r["setup"] == "pure"}
for d, pub in (("bfloat16", r"$\approx$5.5"), ("float16", "--"), ("float32", r"$\approx$16.6")):
    lines.append(f"{d} & {pub} & {pm('C2|'+d+'|pure'):.3f} & {pm('C113|'+d+'|pure'):.3f} & "
                 f"{pm('C1000|'+d+'|pure'):.3f} & {b6m[(151936, d)]:.3f} \\\\")
lines.append(f"float64 & $\\approx$36.7 & {f64[2]['paper_margin']:.3f} & {f64[113]['paper_margin']:.3f} & "
             f"{f64[1000]['paper_margin']:.3f} & {f64[151936]['paper_margin']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(GEN, "tab_cliffs.tex"), "w").write("\n".join(lines) + "\n")

if summ:
    lab = {"qwen3-0.6b": "Qwen3-0.6B", "gemma3-1b-it": "Gemma-3-1B-it", "smollm2-360m": "SmolLM2-360M",
           "vanilla-340m": "vanilla-340M", "softpick-340m": "softpick-340M"}
    L = [r"\begin{tabular}{lrrrrrl}", r"\toprule",
         r"model & $N_{\max}$ & sink rate (256) & sink rate ($N_{\max}$) & fixed-head $w$ (256$\to N_{\max}$) & best-head $w$ ($N_{\max}$) & top-1 \\",
         r"\midrule"]
    for m in lab:
        if m not in summ["e6"]:
            L.append(f"{lab[m]} & \\multicolumn{{6}}{{l}}{{\\emph{{pending}}}} \\\\")
            continue
        r = summ["e6"][m]["rows"]; nd = summ["e6"][m]["needle"]["rows"]
        L.append(f"{lab[m]} & {r[-1]['N']:,} & {100*r[0]['sink_rate_0p3']:.1f}\\% & {100*r[-1]['sink_rate_0p3']:.1f}\\% & "
                 f"{nd[0]['w_fixed']:.3f}$\\to${nd[-1]['w_fixed']:.4f} & {nd[-1]['w_best']:.3f} & "
                 f"{sum(x['top1_rate']*3 for x in nd):.0f}/{3*len(nd)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_e6.tex"), "w").write("\n".join(L) + "\n")

for f in ["e4_collapse.png", "e3_random_logits.png", "e6_sinks.png", "e6_needle.png", "e4_cancellation.png", "e7_logprobs.png"]:
    src = os.path.join(ROOT, "figures", f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(HERE, "figures", f))
print(f"wrote refs.bib ({len(bib)} entries), macros.tex ({len(M)} macros), tables, figures")
