from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROOT = "/home/seokhwan/LiDAR-scan-matching/proposal_ws/segmentation_aware_ndt_ws"


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("cloud_topic", default_value="/point_cloud"),
        DeclareLaunchArgument("map_path", default_value=f"{ROOT}/classical_pointcloud_map.pcd"),
        DeclareLaunchArgument("map_frame_id", default_value="map"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("map_leaf_size", default_value="0.2"),
        DeclareLaunchArgument("keyframe_delta_trans", default_value="1.0"),
        DeclareLaunchArgument("keyframe_delta_angle", default_value="0.17"),
    ]
    node = Node(
        package="segmentation_aware_ndt",
        executable="pointcloud_map_builder",
        name="classical_pointcloud_map_builder",
        output="screen",
        parameters=[{
            "cloud_topic": LaunchConfiguration("cloud_topic"),
            "map_path": LaunchConfiguration("map_path"),
            "map_frame_id": LaunchConfiguration("map_frame_id"),
            "publish_map": False,
            "use_sim_time": typed("use_sim_time", bool),
            "map_leaf_size": typed("map_leaf_size", float),
            "keyframe_delta_trans": typed("keyframe_delta_trans", float),
            "keyframe_delta_angle": typed("keyframe_delta_angle", float),
            "transform_timeout_sec": 0.2,
        }],
    )
    return LaunchDescription(arguments + [node])
