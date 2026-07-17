import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


ROOT = "/home/seokhwan/LiDAR-scan-matching/proposal_ws/segmentation_aware_ndt_ws"
DATASET = "/home/seokhwan/datasets/kitti/odometry/dataset"


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def make_output_directory(context):
    os.makedirs(LaunchConfiguration("output_dir").perform(context), exist_ok=True)
    return []


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("dataset_root", default_value=DATASET),
        DeclareLaunchArgument("sequence", default_value="00"),
        DeclareLaunchArgument("start_frame", default_value="0"),
        DeclareLaunchArgument("end_frame", default_value="-1"),
        DeclareLaunchArgument("frame_step", default_value="1"),
        DeclareLaunchArgument("max_frames", default_value="-1"),
        DeclareLaunchArgument("publish_rate", default_value="10.0"),
        DeclareLaunchArgument("output_dir", default_value=f"{ROOT}/kitti_output/sequence_00"),
        DeclareLaunchArgument("map_leaf_size", default_value="0.0"),
        DeclareLaunchArgument("keyframe_delta_trans", default_value="1.0"),
        DeclareLaunchArgument("keyframe_delta_angle", default_value="0.17"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument("ndt_min_points", default_value="6"),
        DeclareLaunchArgument("use_map_ground_downsampling", default_value="false"),
        DeclareLaunchArgument("ground_keep_ratio", default_value="0.5"),
        DeclareLaunchArgument("ground_weight", default_value="0.5"),
        DeclareLaunchArgument("nonground_weight", default_value="1.0"),
        DeclareLaunchArgument("neutral_weight", default_value="-1.0"),
        DeclareLaunchArgument("preserve_unlabeled_points", default_value="false"),
        DeclareLaunchArgument("segment_valid_point_num", default_value="3"),
        DeclareLaunchArgument("segment_valid_line_num", default_value="2"),
        DeclareLaunchArgument("build_semantic_map", default_value="true"),
        DeclareLaunchArgument("build_classical_map", default_value="true"),
        DeclareLaunchArgument("wait_for_point_subscribers", default_value="2"),
    ]
    kitti_config = f"{ROOT}/segmentation_aware_ndt/config/kitti_64.yaml"
    publisher = ExecuteProcess(
        cmd=[
            "ros2", "run", "segmentation_aware_ndt", "kitti_sequence_publisher",
            "--dataset-root", LaunchConfiguration("dataset_root"),
            "--sequence", LaunchConfiguration("sequence"),
            "--start-frame", LaunchConfiguration("start_frame"),
            "--end-frame", LaunchConfiguration("end_frame"),
            "--frame-step", LaunchConfiguration("frame_step"),
            "--max-frames", LaunchConfiguration("max_frames"),
            "--publish-rate", LaunchConfiguration("publish_rate"),
            "--wait-for-point-subscribers",
            LaunchConfiguration("wait_for_point_subscribers"),
            "--publish-mapping-tf",
            "--mapping-tf-warmup-cycles", "5",
        ],
        output="screen",
    )
    common = {
        "cloud_topic": "/point_cloud",
        "map_frame_id": "map",
        "use_sim_time": True,
        "map_leaf_size": typed("map_leaf_size", float),
        "keyframe_delta_trans": typed("keyframe_delta_trans", float),
        "keyframe_delta_angle": typed("keyframe_delta_angle", float),
        "transform_timeout_sec": 0.2,
    }
    semantic_builder = Node(
        package="segmentation_aware_ndt",
        executable="semantic_map_builder",
        name="kitti_semantic_map_builder",
        output="screen",
        condition=IfCondition(LaunchConfiguration("build_semantic_map")),
        parameters=[ParameterFile(kitti_config, allow_substs=True), common, {
            "map_path": PathJoinSubstitution([LaunchConfiguration("output_dir"), "semantic_weighted_map.pcd"]),
            "ndt_map_path": PathJoinSubstitution([LaunchConfiguration("output_dir"), "weighted_ndt_voxels.pcd"]),
            "publish_map": False,
            "ndt_resolution": typed("ndt_resolution", float),
            "ndt_min_points": typed("ndt_min_points", int),
            "use_map_ground_downsampling": typed("use_map_ground_downsampling", bool),
            "ground_keep_ratio": typed("ground_keep_ratio", float),
            "ground_weight": typed("ground_weight", float),
            "nonground_weight": typed("nonground_weight", float),
            "neutral_weight": typed("neutral_weight", float),
            "preserve_unlabeled_points": typed("preserve_unlabeled_points", bool),
            "segment_valid_point_num": typed("segment_valid_point_num", int),
            "segment_valid_line_num": typed("segment_valid_line_num", int),
        }],
    )
    classical_builder = Node(
        package="segmentation_aware_ndt",
        executable="pointcloud_map_builder",
        name="kitti_classical_map_builder",
        output="screen",
        condition=IfCondition(LaunchConfiguration("build_classical_map")),
        parameters=[common, {
            "map_path": PathJoinSubstitution([LaunchConfiguration("output_dir"), "classical_pointcloud_map.pcd"]),
            "publish_map": False,
            "compact_after_save": True,
        }],
    )
    shutdown = RegisterEventHandler(OnProcessExit(
        target_action=publisher,
        on_exit=[EmitEvent(event=Shutdown(reason="KITTI mapping input completed"))],
    ))
    return LaunchDescription(
        arguments + [OpaqueFunction(function=make_output_directory), semantic_builder,
                     classical_builder, TimerAction(period=2.0, actions=[publisher]), shutdown]
    )
