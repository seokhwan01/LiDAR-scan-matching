#!/usr/bin/env bash

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:-weighted}"
if [[ "${METHOD}" != "weighted" && "${METHOD}" != "classical" \
    && "${METHOD}" != "classical_online" \
    && "${METHOD}" != "unweighted_segmented" ]]; then
  echo "usage: $0 [weighted|classical|classical_online|unweighted_segmented] [launch arguments...]" >&2
  exit 2
fi
shift || true

SEQUENCE="00"
OUTPUT_DIR=""
EVALUATION_ROOT_ARGUMENT=""
RESULT_METHOD_ARGUMENT=""
LAUNCH_ARGUMENTS=()
for argument in "$@"; do
  case "${argument}" in
    sequence:=*) SEQUENCE="${argument#sequence:=}" ;;
    output_dir:=*) OUTPUT_DIR="${argument#output_dir:=}" ;;
    evaluation_root:=*)
      EVALUATION_ROOT_ARGUMENT="${argument#evaluation_root:=}"
      ;;
    result_method:=*)
      RESULT_METHOD_ARGUMENT="${argument#result_method:=}"
      ;;
    *) LAUNCH_ARGUMENTS+=("${argument}") ;;
  esac
done
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${ROOT}/kitti_output/sequence_${SEQUENCE}"
fi

RESULT_METHOD="${METHOD}"
if [[ "${METHOD}" == "weighted" ]]; then
  GROUND_KEEP_RATIO="0.5"
  for argument in "$@"; do
    if [[ "${argument}" == ground_keep_ratio:=* ]]; then
      GROUND_KEEP_RATIO="${argument#ground_keep_ratio:=}"
    fi
  done
  case "${GROUND_KEEP_RATIO}" in
    1|1.0|1.00|1.000) RESULT_METHOD="weighted_ground100" ;;
    .5|0.5|0.50|0.500) RESULT_METHOD="weighted_ground50" ;;
    *) RESULT_METHOD="weighted_ground${GROUND_KEEP_RATIO//./p}" ;;
  esac
fi
if [[ -n "${RESULT_METHOD_ARGUMENT}" ]]; then
  RESULT_METHOD="${RESULT_METHOD_ARGUMENT}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_SUFFIX="${KITTI_RUN_SUFFIX:-}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_SUFFIX="_${RUN_SUFFIX}"
fi
BAG_NAME="kitti_${SEQUENCE}_${RESULT_METHOD}${RUN_SUFFIX}_${STAMP}"
if [[ -n "${KITTI_EVALUATION_BASE:-}" ]]; then
  EVALUATION_BASE="${KITTI_EVALUATION_BASE}"
elif [[ -n "${EVALUATION_ROOT_ARGUMENT}" ]]; then
  EVALUATION_BASE="${EVALUATION_ROOT_ARGUMENT}"
else
  EVALUATION_BASE="${OUTPUT_DIR}/evaluation"
fi
BAG_ROOT="${EVALUATION_BASE}/${RESULT_METHOD}"
BAG_PATH="${BAG_ROOT}/${BAG_NAME}"

source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"
mkdir -p "${BAG_ROOT}"

ros2 launch segmentation_aware_ndt kitti_localization.launch.py \
  method:="${METHOD}" \
  sequence:="${SEQUENCE}" \
  output_dir:="${OUTPUT_DIR}" \
  evaluation_root:="${BAG_ROOT}" \
  evaluation_bag_name:="${BAG_NAME}" \
  "${LAUNCH_ARGUMENTS[@]}"

if [[ ! -f "${BAG_PATH}/metadata.yaml" ]]; then
  echo "KITTI evaluation bag was not finalized: ${BAG_PATH}" >&2
  exit 1
fi
python3 "${ROOT}/evaluation/evaluate_localization.py" "${BAG_PATH}"
