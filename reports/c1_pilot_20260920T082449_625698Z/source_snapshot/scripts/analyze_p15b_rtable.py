"""Aggregate bounded P1.5B R-TABLE measurements; no benchmark execution."""
from __future__ import annotations
import glob, json, math
from pathlib import Path
OUT = Path("reports/p15b_rtable")
K = ("A0_score","A1_score","A0_context","A1_context","B0_score","B1_score","B0_context","B1_context")
S = {}
for p in glob.glob(str(OUT / "summary_seq*.json")):
    d=json.loads(Path(p).read_text()); S[d["seq"]]=d
S=dict(sorted(S.items()))
per={"measured_lengths":list(S),"note":"medians are bounded CUDA-event diagnostics; raw samples are retained in per-length summaries and JSONL files","by_seq":{}}
for n,d in S.items():
    per["by_seq"][str(n)]={"kernels":{k:{x:v[x] for x in ("warmup","iterations","median_ms","mean_ms","min_ms","max_ms","samples_ms")} for k,v in d["kernels"].items()},"correctness":d["correctness"]}
base=S[min(S)]
for n in S:
    if n!=min(S): per["by_seq"][str(n)]["growth_vs_1k"]={k:S[n]["kernels"][k]["median_ms"]/base["kernels"][k]["median_ms"] for k in S[n]["kernels"]}
doub={"by_kernel":{k:{} for k in K}}
for k in K:
    ns=sorted(S)
    for a,b in zip(ns,ns[1:]):
        if b==2*a: doub["by_kernel"][k][f"{a}->{b}"]=S[b]["kernels"][k]["median_ms"]/S[a]["kernels"][k]["median_ms"]
exp={"by_kernel":{}}
for k in K:
    ns=sorted(S); vals=[S[n]["kernels"][k]["median_ms"] for n in ns]
    exp["by_kernel"][k]={"from_seq":ns[0],"to_seq":ns[-1],"growth":vals[-1]/vals[0],"log2_exponent":math.log(vals[-1]/vals[0],2)/math.log(ns[-1]/ns[0],2)}
corr={"by_seq":{str(n):d["correctness"] for n,d in S.items()},"note":"A0/B0 and A1/B1 GPU-vs-CPU plus repaired-vs-control metrics are in each summary"}
for name,obj in (("per_kernel_timings.json",per),("doubling_factors.json",doub),("scaling_exponents.json",exp),("correctness.json",corr)):
    (OUT/name).write_text(json.dumps(obj,indent=2))
print(json.dumps({"lengths":list(S),"exponents":{k:v["log2_exponent"] for k,v in exp["by_kernel"].items()}},indent=2))
