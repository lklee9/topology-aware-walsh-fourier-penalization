#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CALL_DIR="$(pwd)"

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' is not installed or not on PATH." >&2
  exit 1
fi

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [output-zip-path]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  if [[ "$1" = /* ]]; then
    OUTPUT_ZIP="$1"
  else
    OUTPUT_ZIP="${CALL_DIR}/$1"
  fi
else
  OUTPUT_ZIP="${REPO_ROOT}/dwave-bench-min.zip"
fi

INCLUDE_PATHS=(
  # Top-level run instructions and pinned Python environment.
  "README.md"
  "requirements.txt"
  # D-Wave benchmark driver and shared experiment constants.
  "experiments/6_dwave_bench.py"
  "experiments/experiment_config.py"
  # Runtime helper modules imported directly or transitively by the driver.
  "experiments/utils"
  "fourier_projection"
  # Family-level tuning summaries consumed by --tuning-dir defaults.
  "experiments/tunings/projected_penalty_tuning_summary.csv"
  "experiments/tunings/unbalanced_penalty_tuning_summary.csv"
  # Benchmark datasets consumed by the driver's default input paths.
  "data/tsp_instances_small"
  "data/mis_instances"
  "data/mdkp_instances"
)

for path in "${INCLUDE_PATHS[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
    echo "Error: missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${OUTPUT_ZIP}")"
rm -f "${OUTPUT_ZIP}"

cd "${REPO_ROOT}"

zip -r "${OUTPUT_ZIP}" \
  "${INCLUDE_PATHS[@]}" \
  -x '*/__pycache__/*' '*.pyc' '*/.DS_Store' '*/.~/*' '*~' \
    'experiments/results/*'

echo "Created: ${OUTPUT_ZIP}"
