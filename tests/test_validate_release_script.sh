#!/usr/bin/env bash
set -euo pipefail

script="scripts/validate_release.sh"

bash -n "$script"

source "$script"

recorded_commands=()
run_step() {
  recorded_commands+=("$(printf '%q ' "$@")")
}

test -z "${VALIDATE_RELEASE_TEST_MODE+x}"

set +e
usage_output="$(main --invalid 2>&1)"
usage_status=$?
set -e
test "$usage_status" -eq 2
grep -Fq "usage: $script [--gpu]" <<<"$usage_output"

cpu_output_file="$(mktemp)"
main >"$cpu_output_file"
cpu_output="$(<"$cpu_output_file")"
rm "$cpu_output_file"
grep -Fxq "CPU_RELEASE_VALIDATION_OK=1" <<<"$cpu_output"
grep -Fxq "RELEASE_VALIDATION_OK=1" <<<"$cpu_output"
! grep -Fq "GPU_VALIDATION_START=1" <<<"$cpu_output"
test "$(tail -n 1 <<<"$cpu_output")" = "RELEASE_VALIDATION_OK=1"
grep -Fq "uv sync" <<<"${recorded_commands[*]}"

test -n "${recorded_commands[*]}"
recorded_commands=()

fake_cuda_dir="$(mktemp -d)"
trap 'rm -rf "$fake_cuda_dir"' EXIT
mkdir "$fake_cuda_dir/bin"
touch "$fake_cuda_dir/bin/nvcc"
chmod +x "$fake_cuda_dir/bin/nvcc"

gpu_output_file="$(mktemp)"
CUDA_TOOLKIT_PATH="$fake_cuda_dir" main --gpu >"$gpu_output_file"
gpu_output="$(<"$gpu_output_file")"
rm "$gpu_output_file"
grep -Fxq "CPU_RELEASE_VALIDATION_OK=1" <<<"$gpu_output"
grep -Fxq "GPU_RELEASE_VALIDATION_OK=1" <<<"$gpu_output"
grep -Fxq "RELEASE_VALIDATION_OK=1" <<<"$gpu_output"
test "$(grep -nFx "CPU_RELEASE_VALIDATION_OK=1" <<<"$gpu_output" | cut -d: -f1)" -lt \
  "$(grep -nFx "GPU_RELEASE_VALIDATION_OK=1" <<<"$gpu_output" | cut -d: -f1)"
test "$(tail -n 1 <<<"$gpu_output")" = "RELEASE_VALIDATION_OK=1"
grep -Fq "cargo check" <<<"${recorded_commands[*]}"

echo "VALIDATE_RELEASE_SCRIPT_TEST_OK=1"
