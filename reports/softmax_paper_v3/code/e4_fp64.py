"""E4 extension: the float64 collapse threshold (Barron et al. give c64 ~ 36.7) on CPU PyTorch.
Same method as code/gpu/b6_cuda_thresholds.py (grid 0.01 over [0, 52], refined to 0.001), C in {2, 113, 1000, 151936}."""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpu"))
from b6_cuda_thresholds import first_zero, grad_t, logp_t  # noqa: E402

torch.set_num_threads(2)
eps = torch.finfo(torch.float64).eps
rows = []
for C in (2, 113, 1000, 151936):
    g = first_zero(lambda x: grad_t(x, C, torch.float64, "cpu", "pure"), 0.0, 52.0)
    lp = first_zero(lambda x: logp_t(x, C, torch.float64, "cpu"), 0.0, 52.0)
    r = dict(C=C, grad_first_zero_gap=g, logp_first_zero_gap=lp,
             paper_margin=None if g is None else g - float(np.log(C - 1)),
             pred_ln1_over_eps=float(np.log(C - 1) + np.log(1 / eps)), pred_ln2_over_eps=float(np.log(C - 1) + np.log(2 / eps)))
    rows.append(r)
    print(json.dumps(r), flush=True)
json.dump(dict(device="cpu", torch=torch.__version__, rows=rows),
          open(os.path.join(os.path.dirname(__file__), "..", "data", "e4_fp64.json"), "w"), indent=1)
