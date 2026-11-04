"""B6 (GPU): do the CPU collapse thresholds of E4 carry over to CUDA?

Repeats E4a (first logit gap at which the built-in target gradient is exactly 0) and E4c (first gap at which
log p_target is exactly 0) on the chosen device, for C in {113, 151936} and float32 / bfloat16 / float16,
with the same grid as v2 E4 (step 0.01, refined to 0.001 near the transition).
Usage:  python b6_cuda_thresholds.py --device cuda     (about 1 minute on an RTX 4060)
        python b6_cuda_thresholds.py --device cpu      (should reproduce the v2 CPU numbers; code check)"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def grad_t(gaps, C, dt, dev, setup):
    z = torch.zeros(len(gaps), C, dtype=torch.float64)
    z[:, 0] = torch.as_tensor(gaps, dtype=torch.float64)
    x = z.to(dt).to(dev).requires_grad_(True)
    t = torch.zeros(len(gaps), dtype=torch.long, device=dev)
    F.cross_entropy(x.float() if setup == "upcast" else x, t, reduction="sum").backward()
    return x.grad[:, 0].double().cpu().numpy()


def logp_t(gaps, C, dt, dev):
    z = torch.zeros(len(gaps), C, dtype=torch.float64)
    z[:, 0] = torch.as_tensor(gaps, dtype=torch.float64)
    return F.log_softmax(z.to(dt).to(dev), -1)[:, 0].double().cpu().numpy()


def first_zero(f, lo=0.0, hi=45.0):
    g = np.round(np.arange(lo, hi + 1e-9, 0.01), 6)
    v = np.concatenate([f(g[i:i + 200]) for i in range(0, len(g), 200)])
    if not (v == 0).any():
        return None
    i0 = int(np.argmax(v == 0))
    fine = np.round(np.arange(max(lo, g[i0] - 0.25), g[i0] + 0.25 + 1e-9, 0.001), 6)
    vf = np.concatenate([f(fine[i:i + 200]) for i in range(0, len(fine), 200)])
    return float(fine[np.argmax(vf == 0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    res = dict(device=a.device, torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else None, rows=[])
    for C in (113, 151936):
        for name, dt in DT.items():
            for setup in ("pure", "upcast"):
                if setup == "upcast" and name == "float32":
                    continue
                g0 = first_zero(lambda g: grad_t(g, C, dt, a.device, setup))
                row = dict(C=C, dtype=name, setup=setup, grad_first_zero_gap=g0,
                           paper_margin=None if g0 is None else g0 - float(np.log(C - 1)))
                if setup == "pure":
                    row["logp_first_zero_gap"] = first_zero(lambda g: logp_t(g, C, dt, a.device))
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "..", "data", f"b6_thresholds_{a.device}.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
