from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def town_pose_value(town01_value, town10_value):
    return PythonExpression([
        "'", LaunchConfiguration("town"), "' == 'town10' and '",
        town10_value, "' or '", town01_value, "'",
    ])


def generate_launch_description():
    evaluation_bag_name = (
        "localization_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    arguments = [
        DeclareLaunchArgument(
            "map_path",
            default_value=(
                "/home/seokhwan/LiDAR-scan-matching/proposal_ws/"
                "segmentation_aware_ndt_ws/semantic_weighted_map.pcd"
            ),
        ),
        DeclareLaunchArgument(
            "ndt_map_path",
            default_value=(
                "/home/seokhwan/LiDAR-scan-matching/proposal_ws/"
                "segmentation_aware_ndt_ws/weighted_ndt_voxels.pcd"
            ),
        ),
        DeclareLaunchArgument("input_topic", default_value="/point_cloud"),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("base_link_frame", default_value="base_link"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("record_evaluation", default_value="true"),
        DeclareLaunchArgument(
            "evaluation_bag_name", default_value=evaluation_bag_name
        ),
        DeclareLaunchArgument(
            "evaluation_root",
            default_value=(
                "/home/seokhwan/LiDAR-scan-matching/proposal_ws/"
                "segmentation_aware_ndt_ws/evaluation/weighted_ndt"
            ),
        ),
        DeclareLaunchArgument(
            "town", default_value="town01", choices=["town01", "town10"]
        ),
        DeclareLaunchArgument(
            "initial_x", default_value=town_pose_value("335.48999", "-64.644844")
        ),
        DeclareLaunchArgument(
            "initial_y", default_value=town_pose_value("-273.74335", "-24.471010")
        ),
        DeclareLaunchArgument(
            "initial_z", default_value=town_pose_value("0.0015", "0.546100")
        ),
        DeclareLaunchArgument("initial_roll", default_value="0.0"),
        DeclareLaunchArgument("initial_pitch", default_value="0.0"),
        DeclareLaunchArgument(
            "initial_yaw", default_value=town_pose_value("-1.570796", "-0.002779")
        ),
        DeclareLaunchArgument("n_scan", default_value="32"),
        DeclareLaunchArgument("horizon_scan", default_value="1084"),
        DeclareLaunchArgument("ang_res_x", default_value=str(360.0 / 1084.0)),
        DeclareLaunchArgument("ang_res_y", default_value=str(40.0 / 31.0)),
        DeclareLaunchArgument("ang_bottom", default_value="30.0"),
        DeclareLaunchArgument("ground_scan_ind", default_value="20"),
        DeclareLaunchArgument(
            "weights_config",
            default_value=(
                "/home/seokhwan/LiDAR-scan-matching/proposal_ws/"
                "segmentation_aware_ndt_ws/segmentation_aware_ndt/config/"
                "weighted_ndt_weights.yaml"
            ),
        ),
        DeclareLaunchArgument("min_transformation_probability", default_value="0.2"),
        DeclareLaunchArgument("map_downsample_resolution", default_value="0.2"),
        DeclareLaunchArgument("source_downsample_resolution", default_value="0.2"),
        DeclareLaunchArgument("ndt_step_size", default_value="0.1"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument("ndt_min_points", default_value="6"),
        DeclareLaunchArgument("use_precomputed_ndt_map", default_value="true"),
        DeclareLaunchArgument("use_target_ndt_weight", default_value="true"),
        DeclareLaunchArgument("use_source_semantic_weight", default_value="true"),
        DeclareLaunchArgument("use_source_segmentation", default_value="true"),
        DeclareLaunchArgument(
            "use_source_ground_downsampling", default_value="true"
        ),
        DeclareLaunchArgument("ground_keep_ratio", default_value="0.5"),
        DeclareLaunchArgument("use_constant_velocity_prediction", default_value="true"),
        DeclareLaunchArgument("max_prediction_horizon_sec", default_value="0.5"),
        DeclareLaunchArgument("max_prediction_scale", default_value="3.0"),
        DeclareLaunchArgument("max_prediction_distance_m", default_value="1.5"),
        DeclareLaunchArgument("max_consecutive_rejections", default_value="3"),
    ]

    localization_node = Node(
        package="segmentation_aware_ndt",
        executable="semantic_ndt_localizer",
        name="semantic_ndt_localizer",
        output="screen",
        parameters=[ParameterFile(
            LaunchConfiguration("weights_config"), allow_substs=True
        ), {
            "input_topic": LaunchConfiguration("input_topic"),
            "map_path": LaunchConfiguration("map_path"),
            "ndt_map_path": LaunchConfiguration("ndt_map_path"),
            "map_frame_id": LaunchConfiguration("map_frame"),
            "base_frame_id": LaunchConfiguration("base_link_frame"),
            "use_sim_time": typed("use_sim_time", bool),
            "initial_x": typed("initial_x", float),
            "initial_y": typed("initial_y", float),
            "initial_z": typed("initial_z", float),
            "initial_roll": typed("initial_roll", float),
            "initial_pitch": typed("initial_pitch", float),
            "initial_yaw": typed("initial_yaw", float),
            "n_scan": typed("n_scan", int),
            "horizon_scan": typed("horizon_scan", int),
            "ang_res_x": typed("ang_res_x", float),
            "ang_res_y": typed("ang_res_y", float),
            "ang_bottom": typed("ang_bottom", float),
            "ground_scan_ind": typed("ground_scan_ind", int),
            "sensor_minimum_range": 1.0,
            "sensor_mount_angle": 0.0,
            "segment_theta": 60.0 * 3.141592653589793 / 180.0,
            "segment_valid_point_num": 5,
            "segment_valid_line_num": 3,
            "map_downsample_resolution": typed("map_downsample_resolution", float),
            "source_downsample_resolution": typed(
                "source_downsample_resolution", float
            ),
            "ndt_resolution": typed("ndt_resolution", float),
            "ndt_min_points": typed("ndt_min_points", int),
            "ndt_step_size": typed("ndt_step_size", float),
            "ndt_num_threads": 4,
            "transformation_epsilon": 0.01,
            "max_iterations": 64,
            "min_transformation_probability": typed(
                "min_transformation_probability", float
            ),
            "transform_timeout_sec": 0.2,
            "require_weighted_map": True,
            "use_precomputed_ndt_map": typed("use_precomputed_ndt_map", bool),
            "use_target_ndt_weight": typed("use_target_ndt_weight", bool),
            "use_source_semantic_weight": typed(
                "use_source_semantic_weight", bool
            ),
            "use_source_segmentation": typed("use_source_segmentation", bool),
            "use_source_ground_downsampling": typed(
                "use_source_ground_downsampling", bool
            ),
            "ground_keep_ratio": typed("ground_keep_ratio", float),
            "use_constant_velocity_prediction": typed(
                "use_constant_velocity_prediction", bool
            ),
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
            "use_runtime_distance_weight": False,
        }],
    )

    evaluation_recorder = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            "-o",
            PathJoinSubstitution([
                LaunchConfiguration("evaluation_root"),
                LaunchConfiguration("evaluation_bag_name"),
            ]),
            "/ndt_localization/pose",
            "/ndt_localization/odom",
            "/ndt_localization/diagnostics",
            "/ground_truth/tf_raw",
        ],
        condition=IfCondition(LaunchConfiguration("record_evaluation")),
        output="screen",
        sigterm_timeout="10",
        sigkill_timeout="5",
    )

    return LaunchDescription(arguments + [localization_node, evaluation_recorder])
