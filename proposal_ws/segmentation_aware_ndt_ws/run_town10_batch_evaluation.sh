#!/usr/bin/env bash

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_ROOT="${HOME}/CARLA/rosbag_record"
RESULT_ROOT="${WORKSPACE_DIR}/evaluation/town10_0_40_30runs"
SKIP_MISSING_INPUTS="${SKIP_MISSING_INPUTS:-false}"

if [[ "${1:-}" == "--existing-only" ]]; then
  SKIP_MISSING_INPUTS="true"
elif [[ -n "${1:-}" ]] && [[ "${1}" != "--check-only" ]]; then
  INPUT_ROOT="${1}"
fi

# ROUTES=(0_40 30_70 50_90 70_110 90_130)
ROUTES=(0_40)
INPUT_BAGS=()
for route in "${ROUTES[@]}"; do
  for try_index in $(seq 1 30); do
    INPUT_BAGS+=("town10_localization_${route}_try${try_index}")
  done
done
METHODS=(classical_raw weighted_100 weighted_source50)

source /opt/ros/humble/setup.bash
source "${WORKSPACE_DIR}/install/setup.bash"
mkdir -p "${RESULT_ROOT}"

if [[ "${1:-}" == "--check-only" ]]; then
  test -f "${WORKSPACE_DIR}/semantic_weighted_map_town10_global.pcd"
  test -f "${WORKSPACE_DIR}/weighted_ndt_voxels_town10_global.pcd"
  test -f "${WORKSPACE_DIR}/classical_pointcloud_map_town10_global.pcd"
  test -f "${WORKSPACE_DIR}/classical_ndt_voxels_town10_global.pcd"
  echo "BATCH_PERMISSION_CHECK_OK result=${RESULT_ROOT}"
  exit 0
fi

LAUNCH_PID=""
cleanup() {
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT -- "-${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  LAUNCH_PID=""
}
stop_batch() {
  cleanup
  exit 130
}
trap cleanup EXIT
trap stop_batch INT TERM

wait_for_diagnostics_publisher() {
  local attempt
  for attempt in $(seq 1 150); do
    if ros2 topic info /ndt_localization/diagnostics 2>/dev/null \
        | grep -qE 'Publisher count: [1-9]'; then
      return 0
    fi
    if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
      echo "Localization launch exited before diagnostics became available." >&2
      return 1
    fi
    sleep 0.2
  done
  echo "Timed out waiting for /ndt_localization/diagnostics publisher." >&2
  return 1
}

run_one() {
  local method="$1"
  local input_name="$2"
  local input_bag="${INPUT_ROOT}/${input_name}"
  local output_dir="${RESULT_ROOT}/${method}/${input_name}"
  local output_name="${method}_${input_name}"
  local output_bag="${output_dir}/${output_name}"
  local initial_x initial_y initial_z initial_roll initial_pitch initial_yaw

  if [[ ! -f "${input_bag}/metadata.yaml" ]]; then
    if [[ "${SKIP_MISSING_INPUTS}" == "true" ]]; then
      echo "SKIP missing input bag: ${input_bag}"
      return 0
    fi
    echo "Input bag not found: ${input_bag}" >&2
    return 1
  fi
  if [[ -e "${output_bag}" ]]; then
    if [[ -f "${output_bag}/metadata.yaml" ]] \
        && [[ -f "${output_bag}/evaluation/summary.txt" ]]; then
      echo "SKIP completed evaluation: ${output_bag}"
      return 0
    fi
    echo "Output already exists; refusing to overwrite: ${output_bag}" >&2
    return 1
  fi

  read -r initial_x initial_y initial_z initial_roll initial_pitch initial_yaw < <(
    python3 "${WORKSPACE_DIR}/evaluation/extract_initial_pose.py" "${input_bag}" \
      | tail -n 1
  )
  mkdir -p "${output_dir}"
  local common_args=(
    town:=town10
    initial_x:="${initial_x}"
    initial_y:="${initial_y}"
    initial_z:="${initial_z}"
    initial_roll:="${initial_roll}"
    initial_pitch:="${initial_pitch}"
    initial_yaw:="${initial_yaw}"
    record_evaluation:=true
    evaluation_root:="${output_dir}"
    evaluation_bag_name:="${output_name}"
  )

  echo "============================================================"
  echo "method=${method} bag=${input_name}"
  echo "initial pose: ${initial_x} ${initial_y} ${initial_z} yaw=${initial_yaw}"

  if [[ "${method}" == "classical_raw" ]]; then
    # Standard Classical baseline: raw source scan against Gaussian voxels
    # precomputed from the raw global point map. Semantic/PCA weights and
    # source segmentation are all disabled.
    setsid ros2 launch segmentation_aware_ndt semantic_localization.launch.py \
      map_path:="${WORKSPACE_DIR}/classical_pointcloud_map_town10_global.pcd" \
      ndt_map_path:="${WORKSPACE_DIR}/classical_ndt_voxels_town10_global.pcd" \
      use_precomputed_ndt_map:=true \
      use_target_ndt_weight:=false \
      use_source_semantic_weight:=false \
      use_source_segmentation:=false \
      use_source_ground_downsampling:=false \
      ground_keep_ratio:=1.0 \
      "${common_args[@]}" &
  else
    local keep_ratio="1.0"
    if [[ "${method}" == "weighted_source50" ]]; then
      keep_ratio="0.5"
    fi
    setsid ros2 launch segmentation_aware_ndt semantic_localization.launch.py \
      map_path:="${WORKSPACE_DIR}/semantic_weighted_map_town10_global.pcd" \
      ndt_map_path:="${WORKSPACE_DIR}/weighted_ndt_voxels_town10_global.pcd" \
      use_precomputed_ndt_map:=true \
      use_target_ndt_weight:=true \
      use_source_semantic_weight:=true \
      use_source_segmentation:=true \
      use_source_ground_downsampling:=true \
      ground_keep_ratio:="${keep_ratio}" \
      "${common_args[@]}" &
  fi
  LAUNCH_PID=$!

  wait_for_diagnostics_publisher
  ros2 bag play "${input_bag}" --clock \
    --topics /point_cloud /tf /tf_static \
    --remap /tf:=/ground_truth/tf_raw

  # Send Ctrl+C to the complete launch process group. This lets rosbag write
  # metadata.yaml before launch exits, exactly like an interactive Ctrl+C.
  kill -INT -- "-${LAUNCH_PID}"
  wait "${LAUNCH_PID}" || true
  LAUNCH_PID=""

  if [[ ! -f "${output_bag}/metadata.yaml" ]]; then
    echo "Recorded evaluation bag was not finalized: ${output_bag}" >&2
    return 1
  fi
  python3 "${WORKSPACE_DIR}/evaluation/evaluate_localization.py" "${output_bag}"
}

for method in "${METHODS[@]}"; do
  for input_name in "${INPUT_BAGS[@]}"; do
    run_one "${method}" "${input_name}"
  done
done

for route in "${ROUTES[@]}"; do
  python3 "${WORKSPACE_DIR}/evaluation/summarize_town10_batch.py" \
    --root "${RESULT_ROOT}" \
    --route "${route}" \
    --methods "${METHODS[@]}" \
    --expected-tries 30
done

trap - EXIT INT TERM
echo "Batch evaluation pass completed: ${RESULT_ROOT}"
