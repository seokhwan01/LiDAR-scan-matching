#ifndef ROS_UTILS_HPP
#define ROS_UTILS_HPP

#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace lidar_odometry {

inline geometry_msgs::msg::TransformStamped matrix2transform(
    const rclcpp::Time& stamp,
    const Eigen::Matrix4f& pose,
    const std::string& frame_id,
    const std::string& child_frame_id) {
  Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
  quat.normalize();

  geometry_msgs::msg::TransformStamped odom_trans;
  odom_trans.header.stamp = stamp;
  odom_trans.header.frame_id = frame_id;
  odom_trans.child_frame_id = child_frame_id;
  odom_trans.transform.translation.x = pose(0, 3);
  odom_trans.transform.translation.y = pose(1, 3);
  odom_trans.transform.translation.z = pose(2, 3);
  odom_trans.transform.rotation.x = quat.x();
  odom_trans.transform.rotation.y = quat.y();
  odom_trans.transform.rotation.z = quat.z();
  odom_trans.transform.rotation.w = quat.w();
  return odom_trans;
}

inline Eigen::Isometry3d odom2isometry(const nav_msgs::msg::Odometry& odom_msg) {
  const auto& orientation = odom_msg.pose.pose.orientation;
  const auto& position = odom_msg.pose.pose.position;

  Eigen::Quaterniond quat(orientation.w, orientation.x, orientation.y, orientation.z);
  quat.normalize();

  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = quat.toRotationMatrix();
  isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return isometry;
}

}  // namespace lidar_odometry

#endif  // ROS_UTILS_HPP
