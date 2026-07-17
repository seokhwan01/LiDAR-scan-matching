#!/usr/bin/env bash

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"

exec ros2 launch segmentation_aware_ndt kitti_mapping.launch.py "$@"
