#!/usr/bin/env python3

"""Publish KITTI odometry scans and Velodyne ground truth as ROS 2 topics."""

import argparse
import math
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def load_matrix(line):
    values = np.asarray([float(value) for value in line.split()[1:]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = values.reshape(3, 4)
    return matrix


def rotation_to_quaternion(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ])
    axis = int(np.argmax(np.diag(rotation)))
    if axis == 0:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
        ])
    elif axis == 1:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quaternion = np.array([
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
        ])
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quaternion = np.array([
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    return quaternion / np.linalg.norm(quaternion)


def seconds_to_stamp(seconds):
    nanoseconds = int(round(seconds * 1_000_000_000.0))
    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


class KittiSequencePublisher(Node):
    def __init__(self, args):
        super().__init__("kitti_sequence_publisher")
        self.args = args
        sequence_dir = args.dataset_root / "sequences" / args.sequence
        pose_path = args.dataset_root / "poses" / f"{args.sequence}.txt"
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"KITTI sequence directory not found: {sequence_dir}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"KITTI ground-truth pose not found: {pose_path}")

        self.scan_paths = sorted((sequence_dir / "velodyne").glob("*.bin"))
        self.times = np.loadtxt(sequence_dir / "times.txt", dtype=np.float64, ndmin=1)
        camera_poses = np.loadtxt(pose_path, dtype=np.float64, ndmin=2).reshape(-1, 3, 4)
        with (sequence_dir / "calib.txt").open(encoding="utf-8") as stream:
            calibration = {
                line.split(":", 1)[0]: load_matrix(line)
                for line in stream if ":" in line
            }
        if "Tr" not in calibration:
            raise RuntimeError(f"Tr is missing from {sequence_dir / 'calib.txt'}")
        camera_from_velodyne = calibration["Tr"]
        velodyne_from_camera = np.linalg.inv(camera_from_velodyne)

        count = min(len(self.scan_paths), len(self.times), len(camera_poses))
        end = count if args.end_frame < 0 else min(args.end_frame + 1, count)
        self.indices = list(range(args.start_frame, end, args.frame_step))
        if args.max_frames > 0:
            self.indices = self.indices[:args.max_frames]
        if not self.indices:
            raise RuntimeError("selected KITTI frame range is empty")

        self.velodyne_poses = []
        for pose in camera_poses:
            world_from_camera = np.eye(4, dtype=np.float64)
            world_from_camera[:3, :4] = pose
            # KITTI poses are camera_i -> camera_0. Express ground truth in the
            # first Velodyne frame so map z is up and frame 0 is identity.
            self.velodyne_poses.append(
                velodyne_from_camera @ world_from_camera @ camera_from_velodyne
            )

        sensor_qos = QoSProfile(depth=5)
        # KITTI HDL-64 PointCloud2 messages are roughly 2 MB. BEST_EFFORT can
        # lose fragmented localhost samples, which invalidates offline frame-by-frame
        # evaluation. RELIABLE still matches SensorDataQoS subscribers and ensures
        # every published scan reaches the localizer.
        sensor_qos.reliability = ReliabilityPolicy.RELIABLE
        self.cloud_pub = self.create_publisher(PointCloud2, args.point_topic, sensor_qos)
        self.gt_pub = self.create_publisher(TFMessage, args.ground_truth_topic, 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform()

        self.position = 0
        self.finished = False
        self.wait_log_counter = 0
        self.mapping_tf_warmup_remaining = args.mapping_tf_warmup_cycles
        self.timer = self.create_timer(1.0 / args.publish_rate, self.publish_next)
        self.get_logger().info(
            f"KITTI sequence {args.sequence}: selected {len(self.indices)} frames "
            f"({self.indices[0]}..{self.indices[-1]}, step={args.frame_step})"
        )

    def publish_static_transform(self):
        transform = TransformStamped()
        transform.header.stamp = seconds_to_stamp(1.0)
        transform.header.frame_id = self.args.base_frame
        transform.child_frame_id = self.args.lidar_frame
        transform.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(transform)

    def pose_transform(self, index, stamp):
        pose = self.velodyne_poses[index]
        quaternion = rotation_to_quaternion(pose[:3, :3])
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.args.map_frame
        transform.child_frame_id = self.args.base_frame
        transform.transform.translation.x = float(pose[0, 3])
        transform.transform.translation.y = float(pose[1, 3])
        transform.transform.translation.z = float(pose[2, 3])
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])
        return transform

    def point_cloud(self, index, stamp):
        points = np.fromfile(self.scan_paths[index], dtype=np.float32)
        if points.size % 4 != 0:
            raise RuntimeError(f"invalid KITTI scan: {self.scan_paths[index]}")
        points = np.ascontiguousarray(points.reshape(-1, 4), dtype=np.float32)
        message = PointCloud2()
        message.header = Header(stamp=stamp, frame_id=self.args.lidar_frame)
        message.height = 1
        message.width = points.shape[0]
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = message.point_step * message.width
        message.is_dense = bool(np.isfinite(points).all())
        message.data = points.tobytes()
        return message

    def publish_next(self):
        if self.finished:
            return
        subscriber_count = self.cloud_pub.get_subscription_count()
        if subscriber_count < self.args.wait_for_point_subscribers:
            self.wait_log_counter += 1
            if self.wait_log_counter == 1 or self.wait_log_counter % 50 == 0:
                self.get_logger().info(
                    "waiting for point-cloud subscribers: "
                    f"{subscriber_count}/{self.args.wait_for_point_subscribers}"
                )
            return
        index = self.indices[self.position]
        # Avoid zero ROS time, which means "uninitialized" to parts of tf2.
        stamp_seconds = 1.0 + float(self.times[index])
        stamp = seconds_to_stamp(stamp_seconds)
        self.clock_pub.publish(Clock(clock=stamp))
        transform = self.pose_transform(index, stamp)
        self.gt_pub.publish(TFMessage(transforms=[transform]))
        if self.args.publish_mapping_tf:
            self.tf_broadcaster.sendTransform(transform)

        # /tf and /point_cloud use different DDS streams. Repeat the initial
        # transform before frame 0 so tf2 has map<-base_link cached before the
        # map builders receive the first cloud. Localization keeps this at zero.
        if self.mapping_tf_warmup_remaining > 0:
            self.mapping_tf_warmup_remaining -= 1
            if self.mapping_tf_warmup_remaining == 0:
                self.get_logger().info("initial mapping TF warm-up completed")
            return
        self.cloud_pub.publish(self.point_cloud(index, stamp))

        self.position += 1
        if self.position >= len(self.indices):
            self.finished = True
            self.timer.cancel()
            self.get_logger().info("KITTI sequence publication completed")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--publish-rate", type=float, default=10.0)
    parser.add_argument("--wait-for-point-subscribers", type=int, default=1)
    parser.add_argument("--publish-mapping-tf", action="store_true")
    parser.add_argument("--mapping-tf-warmup-cycles", type=int, default=0)
    parser.add_argument("--point-topic", default="/point_cloud")
    parser.add_argument("--ground-truth-topic", default="/ground_truth/tf_raw")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--lidar-frame", default="velodyne")
    args = parser.parse_args()
    if (args.start_frame < 0 or args.frame_step < 1 or args.publish_rate <= 0.0
            or args.wait_for_point_subscribers < 0
            or args.mapping_tf_warmup_cycles < 0):
        parser.error("start-frame, frame-step, and publish-rate are invalid")
    return args


def main():
    args = parse_arguments()
    rclpy.init()
    node = KittiSequencePublisher(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
