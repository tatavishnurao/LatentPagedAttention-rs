#!/usr/bin/env bash
set -euo pipefail

script="scripts/validate_release.sh"

bash -n "$script"

set +e
usage_output="$(bash "$script" --invalid 2>&1)"
usage_status=$?
set -e
test "$usage_status" -eq 2
grep -Fq "usage: $script [--gpu]" <<<"$usage_output"

cpu_output="$(VALIDATE_RELEASE_TEST_MODE=1 bash "$script")"
grep -Fxq "CPU_RELEASE_VALIDATION_OK=1" <<<"$cpu_output"
grep -Fxq "RELEASE_VALIDATION_OK=1" <<<"$cpu_output"
! grep -Fq "GPU_VALIDATION_START=1" <<<"$cpu_output"
test "$(tail -n 1 <<<"$cpu_output")" = "RELEASE_VALIDATION_OK=1"

gpu_output="$(VALIDATE_RELEASE_TEST_MODE=1 bash "$script" --gpu)"
grep -Fxq "CPU_RELEASE_VALIDATION_OK=1" <<<"$gpu_output"
grep -Fxq "GPU_VALIDATION_START=1" <<<"$gpu_output"
grep -Fxq "GPU_RELEASE_VALIDATION_OK=1" <<<"$gpu_output"
grep -Fxq "RELEASE_VALIDATION_OK=1" <<<"$gpu_output"
test "$(grep -nFx "CPU_RELEASE_VALIDATION_OK=1" <<<"$gpu_output" | cut -d: -f1)" -lt \
  "$(grep -nFx "GPU_VALIDATION_START=1" <<<"$gpu_output" | cut -d: -f1)"
test "$(tail -n 1 <<<"$gpu_output")" = "RELEASE_VALIDATION_OK=1"

echo "VALIDATE_RELEASE_SCRIPT_TEST_OK=1"
