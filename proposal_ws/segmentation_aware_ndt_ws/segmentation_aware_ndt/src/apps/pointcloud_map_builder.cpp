#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace {

class PointcloudMapBuilder : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  PointcloudMapBuilder()
      : rclcpp::Node("pointcloud_map_builder"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_),
        map_cloud_(new pcl::PointCloud<PointT>()) {
    initializeParams();

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&PointcloudMapBuilder::callback, this, std::placeholders::_1));

    if (publish_map_) {
      map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          map_topic_, rclcpp::QoS(1).transient_local());
      publish_timer_ = create_wall_timer(
          std::chrono::duration<double>(publish_interval_),
          std::bind(&PointcloudMapBuilder::publishTimer, this));
    }
    save_timer_ = create_wall_timer(
        std::chrono::duration<double>(save_interval_),
        std::bind(&PointcloudMapBuilder::saveTimer, this));
  }

  ~PointcloudMapBuilder() override {
    saveMap();
  }

private:
  void initializeParams() {
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/point_cloud");
    map_topic_ = declare_parameter<std::string>("map_topic", "/pointcloud_map");
    keyframe_delta_trans_ = declare_parameter<double>("keyframe_delta_trans", 1.0);
    keyframe_delta_angle_ = declare_parameter<double>("keyframe_delta_angle", 0.17);
    publish_interval_ = declare_parameter<double>("publish_interval", 5.0);
    save_interval_ = declare_parameter<double>("save_interval", 30.0);
    publish_map_ = declare_parameter<bool>("publish_map", false);
    compact_after_save_ = declare_parameter<bool>("compact_after_save", false);
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    map_path_ = declare_parameter<std::string>("map_path", "pointcloud_map.pcd");
    map_leaf_size_ = declare_parameter<double>("map_leaf_size", 0.2);
    transform_timeout_sec_ = declare_parameter<double>("transform_timeout_sec", 0.2);
  }

  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    auto cloud = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
      return;
    }

    Eigen::Isometry3d map_to_cloud;
    try {
      const auto tf_msg = tf_buffer_.lookupTransform(
          map_frame_id_, cloud_msg->header.frame_id, cloud_msg->header.stamp,
          tf2::durationFromSec(transform_timeout_sec_));
      map_to_cloud = transformToEigen(tf_msg);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "failed to lookup transform %s <- %s: %s",
          map_frame_id_.c_str(), cloud_msg->header.frame_id.c_str(), ex.what());
      return;
    }

    if (has_last_keyframe_) {
      const Eigen::Isometry3d delta = last_keyframe_pose_.inverse() * map_to_cloud;
      const double trans = delta.translation().norm();
      const Eigen::AngleAxisd aa(delta.linear());
      if (trans < keyframe_delta_trans_ && std::abs(aa.angle()) < keyframe_delta_angle_) {
        return;
      }
    }

    auto transformed = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::transformPointCloud(*cloud, *transformed, map_to_cloud.matrix().cast<float>());

    {
      std::lock_guard<std::mutex> lock(mutex_);
      // 원본 스캔 밀도를 유지하기 위해 voxel/downsample 없이 그대로 누적한다.
      *map_cloud_ += *transformed;
      last_keyframe_pose_ = map_to_cloud;
      has_last_keyframe_ = true;
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "raw point cloud accumulated in %s: %zu points",
          map_frame_id_.c_str(), map_cloud_->size());
    }
  }

  Eigen::Isometry3d transformToEigen(
      const geometry_msgs::msg::TransformStamped& tf_msg) const {
    const auto& t = tf_msg.transform.translation;
    const auto& r = tf_msg.transform.rotation;
    Eigen::Quaterniond quat(r.w, r.x, r.y, r.z);
    quat.normalize();

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = quat.toRotationMatrix();
    transform.translation() = Eigen::Vector3d(t.x, t.y, t.z);
    return transform;
  }

  void publishTimer() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (map_cloud_->empty()) {
      return;
    }

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*map_cloud_, map_msg);
    map_msg.header.frame_id = map_frame_id_;
    map_msg.header.stamp = now();
    map_pub_->publish(map_msg);
  }

  void saveTimer() {
    saveMap();
  }

  void saveMap() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (map_cloud_->empty()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "map is empty; nothing to save yet");
      return;
    }
    // pcl::VoxelGrid uses a dense integer grid index and overflows for a
    // kilometre-scale map with a 0.2 m leaf. ApproximateVoxelGrid uses hashing
    // and therefore supports the KITTI map extent.
    pcl::ApproximateVoxelGrid<PointT> voxel_grid;
    voxel_grid.setLeafSize(map_leaf_size_, map_leaf_size_, map_leaf_size_);
    voxel_grid.setInputCloud(map_cloud_);
    pcl::PointCloud<PointT> downsampled;
    voxel_grid.filter(downsampled);
    if (pcl::io::savePCDFileBinary(map_path_, downsampled) != 0) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "failed to save map: %s", map_path_.c_str());
    } else {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "saved raw point cloud map: %s (%zu points)",
          map_path_.c_str(), downsampled.size());
      if (compact_after_save_) {
        map_cloud_ = std::make_shared<pcl::PointCloud<PointT>>(std::move(downsampled));
      }
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr save_timer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::mutex mutex_;

  pcl::PointCloud<PointT>::Ptr map_cloud_;
  std::string cloud_topic_;
  std::string map_topic_;
  double keyframe_delta_trans_;
  double keyframe_delta_angle_;
  double publish_interval_;
  double save_interval_;
  bool publish_map_{false};
  bool compact_after_save_{false};
  double transform_timeout_sec_;
  float map_leaf_size_{0.2F};
  std::string map_frame_id_;
  std::string map_path_;
  bool has_last_keyframe_{false};
  Eigen::Isometry3d last_keyframe_pose_{Eigen::Isometry3d::Identity()};
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointcloudMapBuilder>());
  rclcpp::shutdown();
  return 0;
}
