#!/usr/bin/env bash

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${KITTI_DATASET_ROOT:-${HOME}/datasets/kitti/odometry/dataset}"
COMPLETE_MARKER="${ROOT}/kitti_output/.mapping_00_10_complete"

source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"
set -u

if [[ ! -f "${COMPLETE_MARKER}" ]]; then
  echo "Mapping 00-10 is not complete. Run ./run_kitti_00_10_mapping.sh first." >&2
  exit 1
fi

for number in $(seq 0 10); do
  sequence=$(printf "%02d" "${number}")
  output_dir="${ROOT}/kitti_output/sequence_${sequence}"
  evaluation_dir="${output_dir}/evaluation_00_10"

  echo "============================================================"
  echo "KITTI sequence ${sequence}: Classical NDT"
  "${ROOT}/run_kitti_localization_evaluation.sh" classical \
    dataset_root:="${DATASET_ROOT}" sequence:="${sequence}" \
    output_dir:="${output_dir}" evaluation_root:="${evaluation_dir}"

  echo "KITTI sequence ${sequence}: Weighted NDT, ground 100%"
  "${ROOT}/run_kitti_localization_evaluation.sh" weighted \
    dataset_root:="${DATASET_ROOT}" sequence:="${sequence}" \
    output_dir:="${output_dir}" evaluation_root:="${evaluation_dir}" \
    ground_keep_ratio:=1.0

  echo "KITTI sequence ${sequence}: Weighted NDT, ground 50%"
  "${ROOT}/run_kitti_localization_evaluation.sh" weighted \
    dataset_root:="${DATASET_ROOT}" sequence:="${sequence}" \
    output_dir:="${output_dir}" evaluation_root:="${evaluation_dir}" \
    ground_keep_ratio:=0.5
done

python3 "${ROOT}/evaluation/summarize_kitti_sequences.py"
echo "KITTI sequences 00-10 localization completed."
