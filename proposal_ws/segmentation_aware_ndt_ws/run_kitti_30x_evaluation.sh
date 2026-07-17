#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${KITTI_AUTOLOG_ROOT:-${ROOT}/kitti_output/sequence_00/autolog_0706}"
REPETITIONS="${KITTI_REPETITIONS:-30}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

summarize() {
  python3 "${ROOT}/evaluation/summarize_kitti_repeated.py" \
    "${OUTPUT_ROOT}" --expected-runs "${REPETITIONS}" || true
}
stop_batch() {
  trap - EXIT
  summarize
  exit 130
}
trap summarize EXIT
trap stop_batch INT TERM

run_one() {
  local trial="$1"
  local label="$2"
  shift 2
  local log_path="${LOG_ROOT}/${label}_try${trial}.log"
  local completed_pattern=(
    "${OUTPUT_ROOT}/${label}/kitti_00_${label}_try${trial}_"*/evaluation/summary.txt
  )

  if [[ -e "${completed_pattern[0]}" ]]; then
    echo "SKIP completed ${label} try${trial}"
    return 0
  fi

  echo "============================================================"
  echo "KITTI ${label} try${trial}/${REPETITIONS}"
  echo "log: ${log_path}"
  if KITTI_EVALUATION_BASE="${OUTPUT_ROOT}" \
      KITTI_RUN_SUFFIX="try${trial}" \
      "${ROOT}/run_kitti_localization_evaluation.sh" "$@" \
      >"${log_path}" 2>&1; then
    echo "Completed ${label} try${trial}"
  else
    echo "FAILED ${label} try${trial}; see ${log_path}" >&2
  fi
}

for trial_number in $(seq 1 "${REPETITIONS}"); do
  trial=$(printf "%02d" "${trial_number}")
  run_one "${trial}" classical classical
  run_one "${trial}" weighted_ground100 weighted ground_keep_ratio:=1.0
  run_one "${trial}" weighted_ground50 weighted ground_keep_ratio:=0.5
done

trap - EXIT INT TERM
summarize
echo "KITTI 30-run evaluation completed: ${OUTPUT_ROOT}"
