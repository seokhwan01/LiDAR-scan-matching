from datetime import datetime
import math
import os

import numpy as np
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


ROOT = "/home/seokhwan/LiDAR-scan-matching/proposal_ws/segmentation_aware_ndt_ws"
DATASET = "/home/seokhwan/datasets/kitti/odometry/dataset"


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def initial_pose(dataset_root, sequence, frame):
    sequence_dir = os.path.join(dataset_root, "sequences", sequence)
    poses = np.loadtxt(os.path.join(dataset_root, "poses", f"{sequence}.txt"), ndmin=2)
    with open(os.path.join(sequence_dir, "calib.txt"), encoding="utf-8") as stream:
        tr_line = next(line for line in stream if line.startswith("Tr:"))
    tr = np.eye(4)
    tr[:3, :4] = np.asarray([float(v) for v in tr_line.split()[1:]]).reshape(3, 4)
    camera_pose = np.eye(4)
    camera_pose[:3, :4] = poses[frame].reshape(3, 4)
    pose = np.linalg.inv(tr) @ camera_pose @ tr
    rotation = pose[:3, :3]
    # Both localizers compose the initial rotation as Rx(roll) * Ry(pitch) *
    # Rz(yaw), not the more common Rz * Ry * Rx convention.
    pitch = math.asin(max(-1.0, min(1.0, rotation[0, 2])))
    roll = math.atan2(-rotation[1, 2], rotation[2, 2])
    yaw = math.atan2(-rotation[0, 1], rotation[0, 0])
    return tuple(float(value) for value in (
        pose[0, 3], pose[1, 3], pose[2, 3], roll, pitch, yaw
    ))


def launch_setup(context):
    dataset_root = LaunchConfiguration("dataset_root").perform(context)
    sequence = LaunchConfiguration("sequence").perform(context)
    start_frame = int(LaunchConfiguration("start_frame").perform(context))
    output_dir = LaunchConfiguration("output_dir").perform(context)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LaunchConfiguration("evaluation_root").perform(context), exist_ok=True)
    classical_map_path = LaunchConfiguration("classical_map_path").perform(context)
    if not classical_map_path:
        classical_map_path = os.path.join(output_dir, "classical_pointcloud_map.pcd")
    classical_ndt_map_path = LaunchConfiguration(
        "classical_ndt_map_path"
    ).perform(context)
    if not classical_ndt_map_path:
        classical_ndt_map_path = os.path.join(
            output_dir, "classical_ndt_voxels.pcd"
        )
    ndt_map_path = LaunchConfiguration("ndt_map_path").perform(context)
    if not ndt_map_path:
        ndt_map_path = os.path.join(output_dir, "weighted_ndt_voxels.pcd")
    x, y, z, roll, pitch, yaw = initial_pose(dataset_root, sequence, start_frame)
    method = LaunchConfiguration("method")
    common = {
        "input_topic": "/point_cloud",
        "map_frame_id": "map",
        "base_frame_id": "base_link",
        "use_sim_time": True,
        "reliable_input_qos": True,
        "initial_x": x, "initial_y": y, "initial_z": z,
        "initial_roll": roll, "initial_pitch": pitch, "initial_yaw": yaw,
        "source_downsample_resolution": typed("source_downsample_resolution", float),
        "map_downsample_resolution": typed("map_downsample_resolution", float),
        "ndt_resolution": typed("ndt_resolution", float),
        "ndt_step_size": 0.1,
        "ndt_num_threads": 4,
        "ndt_neighbor_search": LaunchConfiguration("ndt_neighbor_search"),
        "transformation_epsilon": 0.01,
        "max_iterations": 64,
        "min_transformation_probability": typed(
            "min_transformation_probability", float
        ),
        "use_constant_velocity_prediction": True,
        "max_prediction_horizon_sec": typed(
            "max_prediction_horizon_sec", float
        ),
        "max_prediction_scale": typed("max_prediction_scale", float),
        "max_prediction_distance_m": typed(
            "max_prediction_distance_m", float
        ),
        "max_consecutive_rejections": typed(
            "max_consecutive_rejections", int
        ),
        "reinitialize_after_rejections": typed(
            "reinitialize_after_rejections", int
        ),
        "use_correction_limit": typed("use_correction_limit", bool),
        "max_correction_distance_m": typed("max_correction_distance_m", float),
        "correction_prediction_ratio": typed("correction_prediction_ratio", float),
        "use_velocity_consistency_check": typed(
            "use_velocity_consistency_check", bool
        ),
        "max_accepted_speed_mps": typed("max_accepted_speed_mps", float),
        "max_accepted_yaw_rate_dps": typed("max_accepted_yaw_rate_dps", float),
        "use_adaptive_recovery": typed("use_adaptive_recovery", bool),
        "adaptive_recovery_after_rejections": typed(
            "adaptive_recovery_after_rejections", int
        ),
        "adaptive_recovery_exit_accepts": typed(
            "adaptive_recovery_exit_accepts", int
        ),
        "recovery_use_target_ndt_weight": typed(
            "recovery_use_target_ndt_weight", bool
        ),
        "recovery_use_source_semantic_weight": typed(
            "recovery_use_source_semantic_weight", bool
        ),
        "recovery_use_source_ground_downsampling": typed(
            "recovery_use_source_ground_downsampling", bool
        ),
        "recovery_ground_keep_ratio": typed("recovery_ground_keep_ratio", float),
        "recovery_semantic_mismatch_weight": typed(
            "recovery_semantic_mismatch_weight", float
        ),
        "recovery_min_transformation_probability": typed(
            "recovery_min_transformation_probability", float
        ),
        "segment_valid_point_num": typed("segment_valid_point_num", int),
        "segment_valid_line_num": typed("segment_valid_line_num", int),
        "transform_timeout_sec": 0.2,
    }
    classical_online = Node(
        package="segmentation_aware_ndt", executable="classical_ndt_localizer",
        name="kitti_classical_online_ndt_localizer", output="screen",
        condition=IfCondition(PythonExpression(["'", method, "' == 'classical_online'"])),
        parameters=[common, {
            "map_path": classical_map_path,
            "source_downsample_resolution": typed(
                "classical_source_downsample_resolution", float
            ),
        }],
    )
    classical_precomputed = Node(
        package="segmentation_aware_ndt", executable="semantic_ndt_localizer",
        name="kitti_classical_ndt_localizer", output="screen",
        condition=IfCondition(PythonExpression(["'", method, "' == 'classical'"])),
        parameters=[
            ParameterFile(
                f"{ROOT}/segmentation_aware_ndt/config/kitti_64.yaml",
                allow_substs=True,
            ),
            common,
            {
                "map_path": os.path.join(output_dir, "semantic_weighted_map.pcd"),
                "ndt_map_path": classical_ndt_map_path,
                "use_precomputed_ndt_map": True,
                "require_weighted_map": True,
                "use_target_ndt_weight": False,
                "use_source_semantic_weight": False,
                "use_source_segmentation": False,
                "use_source_ground_downsampling": False,
                "source_downsample_resolution": typed(
                    "classical_source_downsample_resolution", float
                ),
            },
        ],
    )
    weighted = Node(
        package="segmentation_aware_ndt", executable="semantic_ndt_localizer",
        name="kitti_weighted_ndt_localizer", output="screen",
        condition=IfCondition(PythonExpression([
            "'", method, "' == 'weighted' or '", method,
            "' == 'unweighted_segmented'"
        ])),
        parameters=[ParameterFile(f"{ROOT}/segmentation_aware_ndt/config/kitti_64.yaml", allow_substs=True),
                    common, {
                        "map_path": os.path.join(output_dir, "semantic_weighted_map.pcd"),
                        "ndt_map_path": ndt_map_path,
                        "use_precomputed_ndt_map": True,
                        "require_weighted_map": True,
                        "use_target_ndt_weight": typed(
                            "weighted_use_target_ndt_weight", bool
                        ),
                        "use_source_semantic_weight": typed(
                            "weighted_use_source_semantic_weight", bool
                        ),
                        "use_source_segmentation": typed(
                            "weighted_use_source_segmentation", bool
                        ),
                        "use_source_ground_downsampling": typed(
                            "weighted_use_source_ground_downsampling", bool
                        ),
                        "ground_keep_ratio": typed("ground_keep_ratio", float),
                        "ground_weight": typed("ground_weight", float),
                        "nonground_weight": typed("nonground_weight", float),
                        "neutral_weight": typed("neutral_weight", float),
                        "preserve_unlabeled_points": typed(
                            "preserve_unlabeled_points", bool
                        ),
                        "semantic_class_threshold": typed("semantic_class_threshold", float),
                        "semantic_mismatch_weight": typed(
                            "semantic_mismatch_weight", float
                        ),
                    }],
    )
    return [classical_online, classical_precomputed, weighted]


def generate_launch_description():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arguments = [
        DeclareLaunchArgument("dataset_root", default_value=DATASET),
        DeclareLaunchArgument("sequence", default_value="00"),
        DeclareLaunchArgument(
            "method", default_value="weighted",
            choices=[
                "classical", "classical_online",
                "unweighted_segmented", "weighted"
            ]
        ),
        DeclareLaunchArgument("start_frame", default_value="0"),
        DeclareLaunchArgument("end_frame", default_value="-1"),
        DeclareLaunchArgument("frame_step", default_value="1"),
        DeclareLaunchArgument("max_frames", default_value="-1"),
        DeclareLaunchArgument("publish_rate", default_value="10.0"),
        DeclareLaunchArgument("output_dir", default_value=f"{ROOT}/kitti_output/sequence_00"),
        DeclareLaunchArgument("classical_map_path", default_value=""),
        DeclareLaunchArgument("classical_ndt_map_path", default_value=""),
        DeclareLaunchArgument("ndt_map_path", default_value=""),
        DeclareLaunchArgument("evaluation_root", default_value=f"{ROOT}/kitti_output/sequence_00/evaluation"),
        DeclareLaunchArgument("evaluation_bag_name", default_value=f"kitti_00_{stamp}"),
        DeclareLaunchArgument("record_evaluation", default_value="true"),
        DeclareLaunchArgument("source_downsample_resolution", default_value="0.2"),
        DeclareLaunchArgument(
            "classical_source_downsample_resolution", default_value="0.2"
        ),
        DeclareLaunchArgument("map_downsample_resolution", default_value="0.2"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument(
            "ndt_neighbor_search", default_value="DIRECT7",
            choices=["DIRECT1", "DIRECT7", "DIRECT26"],
        ),
        DeclareLaunchArgument("min_transformation_probability", default_value="0.2"),
        DeclareLaunchArgument("ground_keep_ratio", default_value="0.5"),
        DeclareLaunchArgument("ground_weight", default_value="0.5"),
        DeclareLaunchArgument("nonground_weight", default_value="1.0"),
        DeclareLaunchArgument("neutral_weight", default_value="-1.0"),
        DeclareLaunchArgument("preserve_unlabeled_points", default_value="false"),
        DeclareLaunchArgument("segment_valid_point_num", default_value="3"),
        DeclareLaunchArgument("segment_valid_line_num", default_value="2"),
        DeclareLaunchArgument("semantic_class_threshold", default_value="0.75"),
        DeclareLaunchArgument("semantic_mismatch_weight", default_value="0.1"),
        DeclareLaunchArgument("weighted_use_target_ndt_weight", default_value="true"),
        DeclareLaunchArgument(
            "weighted_use_source_semantic_weight", default_value="true"
        ),
        DeclareLaunchArgument("weighted_use_source_segmentation", default_value="true"),
        DeclareLaunchArgument(
            "weighted_use_source_ground_downsampling", default_value="true"
        ),
        # KITTI sequence 00 reaches roughly 10 m/s. Keep the last accepted
        # velocity through short NDT rejection bursts so the initial guess does
        # not freeze several metres behind the vehicle.
        DeclareLaunchArgument("max_prediction_horizon_sec", default_value="2.0"),
        DeclareLaunchArgument("max_prediction_scale", default_value="20.0"),
        DeclareLaunchArgument("max_prediction_distance_m", default_value="25.0"),
        DeclareLaunchArgument("max_consecutive_rejections", default_value="30"),
        DeclareLaunchArgument("reinitialize_after_rejections", default_value="10"),
        DeclareLaunchArgument("use_correction_limit", default_value="true"),
        DeclareLaunchArgument("max_correction_distance_m", default_value="3.0"),
        DeclareLaunchArgument("correction_prediction_ratio", default_value="0.5"),
        DeclareLaunchArgument("use_velocity_consistency_check", default_value="true"),
        DeclareLaunchArgument("max_accepted_speed_mps", default_value="40.0"),
        DeclareLaunchArgument("max_accepted_yaw_rate_dps", default_value="120.0"),
        DeclareLaunchArgument("use_adaptive_recovery", default_value="false"),
        DeclareLaunchArgument("adaptive_recovery_after_rejections", default_value="3"),
        DeclareLaunchArgument("adaptive_recovery_exit_accepts", default_value="10"),
        DeclareLaunchArgument("recovery_use_target_ndt_weight", default_value="false"),
        DeclareLaunchArgument("recovery_use_source_semantic_weight", default_value="false"),
        DeclareLaunchArgument(
            "recovery_use_source_ground_downsampling", default_value="false"
        ),
        DeclareLaunchArgument("recovery_ground_keep_ratio", default_value="1.0"),
        DeclareLaunchArgument("recovery_semantic_mismatch_weight", default_value="1.0"),
        DeclareLaunchArgument(
            "recovery_min_transformation_probability", default_value="-1.0"
        ),
    ]
    publisher = ExecuteProcess(cmd=[
        "ros2", "run", "segmentation_aware_ndt", "kitti_sequence_publisher",
        "--dataset-root", LaunchConfiguration("dataset_root"),
        "--sequence", LaunchConfiguration("sequence"),
        "--start-frame", LaunchConfiguration("start_frame"),
        "--end-frame", LaunchConfiguration("end_frame"),
        "--frame-step", LaunchConfiguration("frame_step"),
        "--max-frames", LaunchConfiguration("max_frames"),
        "--publish-rate", LaunchConfiguration("publish_rate"),
        "--wait-for-point-subscribers", "1",
    ], output="screen")
    recorder = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o",
             PathJoinSubstitution([LaunchConfiguration("evaluation_root"),
                                   LaunchConfiguration("evaluation_bag_name")]),
             "/ndt_localization/pose", "/ndt_localization/odom",
             "/ndt_localization/diagnostics", "/ground_truth/tf_raw"],
        condition=IfCondition(LaunchConfiguration("record_evaluation")), output="screen",
        sigterm_timeout="10", sigkill_timeout="5",
    )
    shutdown = RegisterEventHandler(OnProcessExit(
        target_action=publisher,
        on_exit=[EmitEvent(event=Shutdown(reason="KITTI localization input completed"))],
    ))
    return LaunchDescription(arguments + [OpaqueFunction(function=launch_setup), recorder,
                                           TimerAction(period=2.0, actions=[publisher]), shutdown])
