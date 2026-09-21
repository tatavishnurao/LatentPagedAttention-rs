# Thermal protocol correction / retained procedural failure

After the pilot and while the first main attempt was building, inspection of
`scripts/run_source_identified_campaign.py` revealed a newer, stricter policy
than `run_p15b_rtable.py`. The initial protocol had cited only the older guard.
This was an audit omission, not a reason to relax the current guard.

Pilot `c1_pilot_20260920T082449_625698Z` remains a diagnostic under the older
84/86 C policy, not accepted main evidence. First main attempt
`c1_main_20260920T082902_048752Z` was stopped during build before any benchmark
process or GPU timing. It is retained as PROCEDURAL_ABORT (no samples).

Superseding thermal rules: use the current source-identified idle preflight
(10 samples, <=73 C final, <=3 C range, <=5% utilization, no unrelated compute,
no active thermal/hardware slowdown), confirm AC + High performance power
scheme, abort at >=82 C or active thermal/hardware slowdown. Unavailable
critical fields fail closed. Between-process cooldown target is <=73 C with
300 s bound; next idle gate remains mandatory. No threshold raised or disabled.
All other budget, timing and correctness rules remain unchanged. If this gate
cannot pass, retain the skipped attempt and report the environmental blocker.
The initial pilot is excluded from inferential/main results regardless of outcome.
