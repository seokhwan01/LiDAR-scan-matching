from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ndt_localization",
            executable="ndt_localizer_node",
            name="ndt_localizer_node",
            output="screen",
            parameters=[{
                # PCD map path
                "map_path": "/home/seokhwan/CARLA_0.9.16/PythonAPI/examples/maps/carla_map_20260506_195251/carla_ndt_map_voxel_0.15_20260506_195251.pcd",

                # Current LiDAR PointCloud2 topic
                "scan_topic": "/carla/ego_vehicle/front_lidar/point_cloud",

                # Debug GT pose topic from gt_pose_publisher.py
                "use_gt_initial_guess": True,
                "gt_pose_topic": "/carla_lidar_gt_pose",

                # Frame name used by /ndt_pose and /ndt_aligned_cloud
                "map_frame": "map",

                # Fallback initial pose
                # spawn_index=0:
                # vehicle: x=-64.645, y=24.471, z=0.600, yaw=0.159 deg
                # lidar is mounted at vehicle z + 2.0m
                "initial_x": -64.645,
                "initial_y": 24.471,
                "initial_z": 2.600,
                "initial_roll_deg": 0.0,
                "initial_pitch_deg": 0.0,
                "initial_yaw_deg": 0.159,

                # NDT parameters
                "ndt_resolution": 2.0,
                "step_size": 0.2,
                "trans_eps": 0.01,
                "max_iter": 30,

                # Current scan downsampling
                "scan_voxel_size": 0.7,

                # Warning threshold
                "fitness_warn_threshold": 20.0,
            }]
        )
    ])