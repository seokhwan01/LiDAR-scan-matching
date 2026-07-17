from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


def typed(name, value_type):
    """Convert a launch string explicitly to the ROS parameter's declared type."""
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    # 센서 기본값은 CARLA publisher 설정과 동일하다.
    # 32개 ring, 수직 FOV [-30, 10]도, tick마다 ring당 약 1084개 ray를 사용한다.
    arguments = [
        DeclareLaunchArgument("cloud_topic", default_value="/point_cloud"),
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
        DeclareLaunchArgument("map_frame_id", default_value="map"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
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
        DeclareLaunchArgument("map_leaf_size", default_value="0.2"),
        DeclareLaunchArgument("keyframe_delta_trans", default_value="1.0"),
        DeclareLaunchArgument("keyframe_delta_angle", default_value="0.17"),
        DeclareLaunchArgument("ndt_resolution", default_value="1.0"),
        DeclareLaunchArgument("ndt_min_points", default_value="6"),
        DeclareLaunchArgument("use_map_ground_downsampling", default_value="false"),
        DeclareLaunchArgument("ground_keep_ratio", default_value="0.5"),
    ]

    mapping_node = Node(
        package="segmentation_aware_ndt",
        executable="semantic_map_builder",
        name="semantic_map_builder",
        output="screen",
        parameters=[ParameterFile(
            LaunchConfiguration("weights_config"), allow_substs=True
        ), {
            "cloud_topic": LaunchConfiguration("cloud_topic"),
            "map_path": LaunchConfiguration("map_path"),
            "ndt_map_path": LaunchConfiguration("ndt_map_path"),
            "map_frame_id": LaunchConfiguration("map_frame_id"),
            "publish_map": False,
            "use_sim_time": typed("use_sim_time", bool),
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
            "map_leaf_size": typed("map_leaf_size", float),
            "keyframe_delta_trans": typed("keyframe_delta_trans", float),
            "keyframe_delta_angle": typed("keyframe_delta_angle", float),
            "ndt_resolution": typed("ndt_resolution", float),
            "ndt_min_points": typed("ndt_min_points", int),
            "use_map_ground_downsampling": typed(
                "use_map_ground_downsampling", bool
            ),
            "ground_keep_ratio": typed("ground_keep_ratio", float),
        }],
    )

    return LaunchDescription(arguments + [mapping_node])
