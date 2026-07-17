from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROOT = "/home/seokhwan/LiDAR-scan-matching/proposal_ws/segmentation_aware_ndt_ws"


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def town_pose_value(town01_value, town10_value):
    return PythonExpression([
        "'", LaunchConfiguration("town"), "' == 'town10' and '",
        town10_value, "' or '", town01_value, "'",
    ])


def generate_launch_description():
    bag_name = "classical_ndt_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    arguments = [
        # Default to the same segmented target geometry used by Weighted NDT.
        # The Classical localizer ignores intensity/semantic weights.
        DeclareLaunchArgument("map_path", default_value=f"{ROOT}/semantic_weighted_map.pcd"),
        DeclareLaunchArgument("input_topic", default_value="/point_cloud"),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("base_link_frame", default_value="base_link"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("record_evaluation", default_value="true"),
        DeclareLaunchArgument(
            "evaluation_root", default_value=f"{ROOT}/evaluation/classical_ndt"
        ),
        DeclareLaunchArgument("evaluation_bag_name", default_value=bag_name),
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
        DeclareLaunchArgument("map_downsample_resolution", default_value="0.2"),
        DeclareLaunchArgument("source_downsample_resolution", default_value="0.2"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument("ndt_step_size", default_value="0.1"),
        DeclareLaunchArgument("min_transformation_probability", default_value="0.2"),
        DeclareLaunchArgument("use_constant_velocity_prediction", default_value="true"),
        DeclareLaunchArgument("max_prediction_horizon_sec", default_value="0.5"),
        DeclareLaunchArgument("max_prediction_scale", default_value="3.0"),
        DeclareLaunchArgument("max_prediction_distance_m", default_value="1.5"),
        DeclareLaunchArgument("max_consecutive_rejections", default_value="3"),
    ]
    node = Node(
        package="segmentation_aware_ndt",
        executable="classical_ndt_localizer",
        name="classical_ndt_localizer",
        output="screen",
        parameters=[{
            "input_topic": LaunchConfiguration("input_topic"),
            "map_path": LaunchConfiguration("map_path"),
            "map_frame_id": LaunchConfiguration("map_frame"),
            "base_frame_id": LaunchConfiguration("base_link_frame"),
            "use_sim_time": typed("use_sim_time", bool),
            "initial_x": typed("initial_x", float),
            "initial_y": typed("initial_y", float),
            "initial_z": typed("initial_z", float),
            "initial_roll": typed("initial_roll", float),
            "initial_pitch": typed("initial_pitch", float),
            "initial_yaw": typed("initial_yaw", float),
            "map_downsample_resolution": typed("map_downsample_resolution", float),
            "source_downsample_resolution": typed("source_downsample_resolution", float),
            "ndt_resolution": typed("ndt_resolution", float),
            "ndt_step_size": typed("ndt_step_size", float),
            "ndt_num_threads": 4,
            "transformation_epsilon": 0.01,
            "max_iterations": 64,
            "min_transformation_probability": typed("min_transformation_probability", float),
            "use_constant_velocity_prediction": typed(
                "use_constant_velocity_prediction", bool
            ),
            "max_prediction_horizon_sec": typed("max_prediction_horizon_sec", float),
            "max_prediction_scale": typed("max_prediction_scale", float),
            "max_prediction_distance_m": typed("max_prediction_distance_m", float),
            "max_consecutive_rejections": typed("max_consecutive_rejections", int),
            "transform_timeout_sec": 0.2,
        }],
    )
    recorder = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record", "-o",
            PathJoinSubstitution([
                LaunchConfiguration("evaluation_root"),
                LaunchConfiguration("evaluation_bag_name"),
            ]),
            "/ndt_localization/pose", "/ndt_localization/odom",
            "/ndt_localization/diagnostics",
            "/ground_truth/tf_raw",
        ],
        condition=IfCondition(LaunchConfiguration("record_evaluation")),
        output="screen",
        sigterm_timeout="10",
        sigkill_timeout="5",
    )
    return LaunchDescription(arguments + [node, recorder])
