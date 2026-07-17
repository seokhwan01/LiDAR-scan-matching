#!/usr/bin/env bash

set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${KITTI_DATASET_ROOT:-${HOME}/datasets/kitti/odometry/dataset}"
FRAME_STEP="${KITTI_MAP_FRAME_STEP:-1}"
PUBLISH_RATE="${KITTI_MAP_PUBLISH_RATE:-10.0}"

source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"
set -u

for number in $(seq 0 10); do
  sequence=$(printf "%02d" "${number}")
  output_dir="${ROOT}/kitti_output/sequence_${sequence}"
  mkdir -p "${output_dir}"

  echo "============================================================"
  echo "KITTI sequence ${sequence}: semantic weighted map"
  if [[ -s "${output_dir}/semantic_weighted_map.pcd" \
      && -s "${output_dir}/weighted_ndt_voxels.pcd" ]]; then
    echo "SKIP existing semantic maps"
  else
    "${ROOT}/run_kitti_mapping.sh" \
      dataset_root:="${DATASET_ROOT}" sequence:="${sequence}" \
      output_dir:="${output_dir}" frame_step:="${FRAME_STEP}" \
      publish_rate:="${PUBLISH_RATE}" build_semantic_map:=true \
      build_classical_map:=false wait_for_point_subscribers:=1
  fi

  echo "KITTI sequence ${sequence}: classical point map"
  if [[ -s "${output_dir}/classical_pointcloud_map.pcd" ]]; then
    echo "SKIP existing classical point map"
  else
    "${ROOT}/run_kitti_mapping.sh" \
      dataset_root:="${DATASET_ROOT}" sequence:="${sequence}" \
      output_dir:="${output_dir}" frame_step:="${FRAME_STEP}" \
      publish_rate:="${PUBLISH_RATE}" build_semantic_map:=false \
      build_classical_map:=true wait_for_point_subscribers:=1
  fi

  echo "KITTI sequence ${sequence}: classical Gaussian voxel map"
  if [[ -s "${output_dir}/classical_ndt_voxels.pcd" ]]; then
    echo "SKIP existing classical NDT voxel map"
  else
    ros2 launch segmentation_aware_ndt build_ndt_voxel_map.launch.py \
      input_map_path:="${output_dir}/classical_pointcloud_map.pcd" \
      output_ndt_map_path:="${output_dir}/classical_ndt_voxels.pcd" \
      map_leaf_size:=0.0 ndt_resolution:=1.0
  fi
done

for number in $(seq 0 10); do
  sequence=$(printf "%02d" "${number}")
  output_dir="${ROOT}/kitti_output/sequence_${sequence}"
  for file in semantic_weighted_map.pcd weighted_ndt_voxels.pcd \
              classical_pointcloud_map.pcd classical_ndt_voxels.pcd; do
    if [[ ! -s "${output_dir}/${file}" ]]; then
      echo "Missing map: ${output_dir}/${file}" >&2
      exit 1
    fi
  done
done

touch "${ROOT}/kitti_output/.mapping_00_10_complete"
echo "KITTI sequences 00-10 mapping completed."
