#!/usr/bin/env bash

set -e

CARLA_DIR="/home/seokhwan/CARLA"
OUTPUT_ROOT="${CARLA_DIR}/rosbag_record"
LOG_ROOT="${OUTPUT_ROOT}/town10_generation_logs"
export MPLCONFIGDIR="/tmp/matplotlib"
export PYTHONPATH="${CARLA_DIR}:${CARLA_DIR}/PythonAPI/carla:${PYTHONPATH}"

source /opt/ros/humble/setup.bash
mkdir -p "${LOG_ROOT}"

# ROUTES=(0_40 30_70 50_90 70_110 90_130)
ROUTES=(0_40)
SCRIPTS=(
  town10_spawn_000_to_040_lidar_localization_ros2.py
)

if [[ "${1:-}" == "--check-only" ]]; then
  for script in "${SCRIPTS[@]}"; do
    test -f "${CARLA_DIR}/${script}"
  done
  test -w "${OUTPUT_ROOT}"
  echo "GENERATION_PERMISSION_CHECK_OK output=${OUTPUT_ROOT}"
  exit 0
fi

RECORDER_PID=""
cleanup_recorder() {
  if [[ -n "${RECORDER_PID}" ]] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
    # Background jobs inherit SIGINT as ignored from bash. rosbag2 handles
    # SIGTERM gracefully and writes metadata.yaml before exiting.
    kill -TERM "${RECORDER_PID}" 2>/dev/null || true
    wait "${RECORDER_PID}" 2>/dev/null || true
  fi
  RECORDER_PID=""
}
stop_generation() {
  cleanup_recorder
  exit 130
}
trap cleanup_recorder EXIT
trap stop_generation INT TERM

for route_index in "${!ROUTES[@]}"; do
  route="${ROUTES[$route_index]}"
  script="${CARLA_DIR}/${SCRIPTS[$route_index]}"
  if [[ ! -f "${script}" ]]; then
    echo "Scenario script not found: ${script}" >&2
    exit 1
  fi

  # for try_index in 1 2 3 4 5; do
  for try_index in $(seq 6 30); do
    bag_name="town10_localization_${route}_try${try_index}"
    bag_path="${OUTPUT_ROOT}/${bag_name}"
    log_path="${LOG_ROOT}/${bag_name}.log"

    # Bags created before this script gained completion markers are valid when
    # metadata exists. Only a bag whose log explicitly records a scenario
    # failure without a later completion is treated as incomplete.
    logged_failure=false
    if [[ -f "${bag_path}/.generation_failed" ]]; then
      logged_failure=true
    elif grep -q "^Scenario failed: ${bag_name}" "${log_path}" 2>/dev/null && \
        ! grep -q "^Completed ${bag_name}$" "${log_path}" 2>/dev/null; then
      logged_failure=true
    fi
    if [[ -f "${bag_path}/.generation_complete" ]] || \
        { [[ -f "${bag_path}/metadata.yaml" ]] && \
          [[ "${logged_failure}" != "true" ]]; }; then
      echo "SKIP existing bag: ${bag_path}"
      continue
    fi
    if [[ -e "${bag_path}" ]]; then
      failed_path="${bag_path}.failed_$(date +%Y%m%d_%H%M%S)"
      echo "Archiving incomplete bag: ${bag_path} -> ${failed_path}"
      mv "${bag_path}" "${failed_path}"
    fi

    completed=false
    for attempt in 1 2 3; do
      echo "============================================================" | tee -a "${log_path}"
      echo "Generating ${bag_name} (attempt ${attempt}/3)" | tee -a "${log_path}"
      ros2 bag record -o "${bag_path}" /point_cloud /tf /tf_static \
        >>"${log_path}" 2>&1 &
      RECORDER_PID=$!
      sleep 2

      if /usr/bin/python3 "${script}" >>"${log_path}" 2>&1; then
        cleanup_recorder
        if [[ ! -f "${bag_path}/metadata.yaml" ]]; then
          echo "Rosbag was not finalized: ${bag_path}" >&2
          exit 1
        fi
        touch "${bag_path}/.generation_complete"
        echo "Completed ${bag_name}" | tee -a "${log_path}"
        completed=true
        break
      fi

      echo "Scenario failed: ${bag_name} attempt ${attempt}; see ${log_path}" >&2
      cleanup_recorder
      if [[ -e "${bag_path}" ]]; then
        touch "${bag_path}/.generation_failed"
        failed_path="${bag_path}.failed_attempt${attempt}_$(date +%Y%m%d_%H%M%S)"
        mv "${bag_path}" "${failed_path}"
        echo "Archived failed attempt: ${failed_path}" | tee -a "${log_path}"
      fi
    done
    if [[ "${completed}" != "true" ]]; then
      echo "Scenario failed three times: ${bag_name}" >&2
      exit 1
    fi
  done
done

trap - EXIT INT TERM
echo "All Town10 localization bags are ready under ${OUTPUT_ROOT}."
