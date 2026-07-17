from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROOT = "/home/seokhwan/LiDAR-scan-matching/proposal_ws/segmentation_aware_ndt_ws"


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "input_map_path", default_value=f"{ROOT}/semantic_weighted_map.pcd"
        ),
        DeclareLaunchArgument(
            "output_ndt_map_path", default_value=f"{ROOT}/weighted_ndt_voxels.pcd"
        ),
        # semantic point map은 저장할 때 이미 downsampling된다.
        # 외부 raw point map을 변환할 때만 이 값을 0보다 크게 설정한다.
        DeclareLaunchArgument("map_leaf_size", default_value="0.0"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument("ndt_min_points", default_value="6"),
        Node(
            package="segmentation_aware_ndt",
            executable="semantic_ndt_voxel_builder",
            name="semantic_ndt_voxel_builder",
            output="screen",
            parameters=[{
                "input_map_path": LaunchConfiguration("input_map_path"),
                "output_ndt_map_path": LaunchConfiguration("output_ndt_map_path"),
                "map_leaf_size": typed("map_leaf_size", float),
                "ndt_resolution": typed("ndt_resolution", float),
                "ndt_min_points": typed("ndt_min_points", int),
            }],
        ),
    ])
