# C1 milestone progress

Base HEAD: 213d45a (no intervening changes); branch research/c1-rtable.
Untracked manuscript/audit files predate this work and are preserved. No applicable AGENTS.md found.

Completed: source inspection of R-TABLE, existing C, benchmark and correctness wrappers; NVIDIA GPU visible (initial 81 C).
Next: implement C1 using existing C FP32 projected buffer + R-TABLE score/context; bounded harness and validation.
Commands: git status --short; git log -5 --oneline; nvidia-smi; git switch -c research/c1-rtable.
Results will be in reports/c1_* (unique directories). No 16K/32K execution or context parallelization permitted.
