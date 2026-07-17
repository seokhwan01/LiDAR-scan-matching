# Segmentation-Aware Weighted NDT Localization

ROS 2 package for building NDT maps and evaluating scan-to-map LiDAR
localization with segmentation-aware weighting. The package supports classical
NDT baselines, segmented unweighted ablations, and weighted NDT localization on
KITTI odometry sequences.

## Overview

The method projects each LiDAR scan onto a range image, separates ground and
non-ground measurements, and builds precomputed NDT voxel maps. During online
localization, source ground points can be deterministically subsampled while
structural, geometric, and compatibility weights are applied to the NDT matching
score.

Main comparison modes:

- `classical`: precomputed classical NDT map without segmentation weights
- `unweighted_segmented`: segmented source/map with weighting disabled
- `weighted`: segmentation-aware weighted NDT

## Environment

Tested with:

- Ubuntu 22.04
- ROS 2 Humble
- PCL
- OpenMP
- KITTI odometry dataset

The launch files currently assume the KITTI dataset is located at:

```bash
/home/seokhwan/datasets/kitti/odometry/dataset
```

If the dataset is stored elsewhere, pass `dataset_root:=...` to the mapping and
localization scripts.

When setting `ROS_DOMAIN_ID` manually, keep it below 200.

## Build

From this workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Release mode is recommended because localization runtime is part of the
evaluation.

## KITTI Map Generation

Build both classical and weighted maps for one sequence:

```bash
./run_kitti_mapping.sh \
  sequence:=00 \
  output_dir:=$PWD/kitti_output_relaxed/sequence_00 \
  build_classical_map:=true \
  build_semantic_map:=true \
  preserve_unlabeled_points:=false \
  use_map_ground_downsampling:=false \
  map_leaf_size:=0.0 \
  ground_weight:=0.5 \
  nonground_weight:=1.0 \
  semantic_mismatch_weight:=0.1 \
  segment_valid_point_num:=3 \
  segment_valid_line_num:=2 \
  frame_step:=1 \
  max_frames:=-1 \
  publish_rate:=10.0 \
  wait_for_point_subscribers:=2
```

Build representative KITTI sequences used in the paper:

```bash
for SEQ in 00 01 03; do
  ./run_kitti_mapping.sh \
    sequence:=$SEQ \
    output_dir:=$PWD/kitti_output_relaxed/sequence_$SEQ \
    build_classical_map:=true \
    build_semantic_map:=true \
    preserve_unlabeled_points:=false \
    use_map_ground_downsampling:=false \
    map_leaf_size:=0.0 \
    ground_weight:=0.5 \
    nonground_weight:=1.0 \
    segment_valid_point_num:=3 \
    segment_valid_line_num:=2 \
    frame_step:=1 \
    max_frames:=-1 \
    publish_rate:=10.0 \
    wait_for_point_subscribers:=2
done
```

Generated maps are stored under the selected `output_dir`, for example:

- `classical_pointcloud_map.pcd`
- `classical_ndt_voxels.pcd`
- `semantic_weighted_map.pcd`
- `weighted_ndt_voxels.pcd`

## KITTI Localization Evaluation

Weighted-S50 evaluation:

```bash
./run_kitti_localization_evaluation.sh weighted \
  sequence:=00 \
  output_dir:=$PWD/kitti_output_relaxed/sequence_00 \
  evaluation_root:=$PWD/kitti_output_relaxed/sequence_00/evaluation_direct7 \
  result_method:=weighted_s50_direct7 \
  ndt_map_path:=$PWD/kitti_output_relaxed/sequence_00/weighted_ndt_voxels.pcd \
  ground_keep_ratio:=0.5 \
  weighted_use_target_ndt_weight:=true \
  weighted_use_source_semantic_weight:=true \
  semantic_mismatch_weight:=0.1 \
  ndt_neighbor_search:=DIRECT7 \
  segment_valid_point_num:=3 \
  segment_valid_line_num:=2 \
  max_correction_distance_m:=3.0 \
  correction_prediction_ratio:=0.5 \
  reinitialize_after_rejections:=10 \
  max_consecutive_rejections:=30
```

Weighted-S100 evaluation:

```bash
./run_kitti_localization_evaluation.sh weighted \
  sequence:=00 \
  output_dir:=$PWD/kitti_output_relaxed/sequence_00 \
  evaluation_root:=$PWD/kitti_output_relaxed/sequence_00/evaluation_direct7 \
  result_method:=weighted_s100_direct7 \
  ndt_map_path:=$PWD/kitti_output_relaxed/sequence_00/weighted_ndt_voxels.pcd \
  ground_keep_ratio:=1.0 \
  weighted_use_target_ndt_weight:=true \
  weighted_use_source_semantic_weight:=true \
  semantic_mismatch_weight:=0.1 \
  ndt_neighbor_search:=DIRECT7 \
  segment_valid_point_num:=3 \
  segment_valid_line_num:=2 \
  max_correction_distance_m:=3.0 \
  correction_prediction_ratio:=0.5 \
  reinitialize_after_rejections:=10 \
  max_consecutive_rejections:=30
```

Unweighted-S50 ablation:

```bash
./run_kitti_localization_evaluation.sh weighted \
  sequence:=00 \
  output_dir:=$PWD/kitti_output_relaxed/sequence_00 \
  evaluation_root:=$PWD/kitti_output_relaxed/sequence_00/evaluation_direct7 \
  result_method:=unweighted_s50_direct7 \
  ndt_map_path:=$PWD/kitti_output_relaxed/sequence_00/weighted_ndt_voxels.pcd \
  ground_keep_ratio:=0.5 \
  weighted_use_target_ndt_weight:=false \
  weighted_use_source_semantic_weight:=false \
  semantic_mismatch_weight:=1.0 \
  ndt_neighbor_search:=DIRECT7 \
  segment_valid_point_num:=3 \
  segment_valid_line_num:=2 \
  max_correction_distance_m:=3.0 \
  correction_prediction_ratio:=0.5 \
  reinitialize_after_rejections:=10 \
  max_consecutive_rejections:=30
```

Classical NDT evaluation:

```bash
./run_kitti_localization_evaluation.sh classical \
  sequence:=00 \
  output_dir:=$PWD/kitti_output_relaxed/sequence_00 \
  evaluation_root:=$PWD/kitti_output_relaxed/sequence_00/evaluation_direct7 \
  result_method:=classical_direct7 \
  classical_ndt_map_path:=$PWD/kitti_output_relaxed/sequence_00/classical_ndt_voxels.pcd \
  ndt_neighbor_search:=DIRECT7 \
  max_correction_distance_m:=3.0 \
  correction_prediction_ratio:=0.5 \
  reinitialize_after_rejections:=10 \
  max_consecutive_rejections:=30
```

Each evaluation records a ROS bag and then runs:

```bash
python3 evaluation/evaluate_localization.py <bag_path>
```

The evaluator reports:

- accepted / rejected registrations
- processing time mean / P95
- ATE/RMSE
- position median / P95 / max
- yaw MAE / P95

## Map Inspection

Inspect a weighted NDT voxel map:

```bash
python3 view_ndt_map.py kitti_output_relaxed/sequence_00/weighted_ndt_voxels.pcd --no-view
```

Inspect the semantic weighted point map:

```bash
python3 view_map.py kitti_output_relaxed/sequence_00/semantic_weighted_map.pcd
```

## Trajectory Figures

Generate KITTI trajectory PDFs for the paper:

```bash
python3 make_kitti_trajectory_figs.py \
  --output-dir /home/seokhwan/LiDAR-scan-matching-ictc2026/figs
```

Font sizes can be adjusted with:

```bash
python3 make_kitti_trajectory_figs.py \
  --axis-label-size 18 \
  --tick-size 16 \
  --legend-size 17
```

## Notes on Generated Files

The following files and directories are generated artifacts and should not be
committed:

- `build/`, `install/`, `log/`
- `kitti_output*/`
- `evaluation*/`
- ROS bag databases such as `*.db3` and `*.mcap`
- generated PCD maps such as `*.pcd`

These patterns should be covered by the repository `.gitignore`.
