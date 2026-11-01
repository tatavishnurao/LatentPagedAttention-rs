#!/usr/bin/env bash
# Host check run before every E3/E4 process on the WSL2 laptop (docs/LOCAL_4060_PLAN.md,
# addendum item 8): fails if Brave, EA Desktop or EALocalHostSvc is running on the Windows side.
# E0B_HOST_CHECK_PROCS overrides the process list (used by tests).
set -euo pipefail
procs="${E0B_HOST_CHECK_PROCS:-brave,EADesktop,EALocalHostSvc}"
command -v powershell.exe >/dev/null || { echo "HOST_CHECK_FAILED: powershell.exe not found"; exit 2; }
running="$(powershell.exe -NoProfile -c "Get-Process -Name $procs -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name" 2>/dev/null \
  | tr -d '\r' | grep -v '^\s*$' | sort -u | tr '\n' ' ' || true)"
if [[ -n "$running" ]]; then
  echo "HOST_CHECK_FAILED: running: $running"
  exit 1
fi
echo "HOST_CHECK_OK"
