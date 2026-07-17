#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <ndt_mapping_localization/weighted_map.hpp>

namespace {

class WeightedNdtMapBuilder : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  WeightedNdtMapBuilder()
      : rclcpp::Node("weighted_ndt_map_builder"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_),
        map_cloud_(new pcl::PointCloud<PointT>()) {
    initializeParams();

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&WeightedNdtMapBuilder::callback, this, std::placeholders::_1));

    if (publish_map_) {
      map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          "/ndt_map", rclcpp::QoS(1).transient_local());
      publish_timer_ = create_wall_timer(
          std::chrono::duration<double>(publish_interval_),
          std::bind(&WeightedNdtMapBuilder::publishTimer, this));
    }
    save_timer_ = create_wall_timer(
        std::chrono::duration<double>(save_interval_),
        std::bind(&WeightedNdtMapBuilder::saveTimer, this));
  }

  ~WeightedNdtMapBuilder() override {
    saveMap();
  }

private:
  void initializeParams() {
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/filtered_points");
    keyframe_delta_trans_ = declare_parameter<double>("keyframe_delta_trans", 1.0);
    keyframe_delta_angle_ = declare_parameter<double>("keyframe_delta_angle", 0.17);
    // NDT target resolution(기본 1.0m)보다 충분히 조밀해야 voxel covariance를 계산할 수 있다.
    map_resolution_ = declare_parameter<double>("map_resolution", 0.2);
    publish_interval_ = declare_parameter<double>("publish_interval", 5.0);
    publish_map_ = declare_parameter<bool>("publish_map", false);
    save_interval_ = declare_parameter<double>("save_interval", 30.0);
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    map_path_ = declare_parameter<std::string>("map_path", "weighted_ndt_map.pcd");
    weight_resolution_ = declare_parameter<double>("weight_resolution", 1.0);
    weight_min_points_ = declare_parameter<int>("weight_min_points", 6);
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
      *map_cloud_ += *transformed;
      downsampleMap();
      last_keyframe_pose_ = map_to_cloud;
      has_last_keyframe_ = true;
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "mapping cloud accumulated in %s: %zu points",
          map_frame_id_.c_str(), map_cloud_->size());
    }
  }

  Eigen::Isometry3d transformToEigen(const geometry_msgs::msg::TransformStamped& tf_msg) const {
    const auto& t = tf_msg.transform.translation;
    const auto& r = tf_msg.transform.rotation;
    Eigen::Quaterniond quat(r.w, r.x, r.y, r.z);
    quat.normalize();

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.linear() = quat.toRotationMatrix();
    transform.translation() = Eigen::Vector3d(t.x, t.y, t.z);
    return transform;
  }

  void downsampleMap() {
    pcl::VoxelGrid<PointT> voxelgrid;
    voxelgrid.setLeafSize(map_resolution_, map_resolution_, map_resolution_);
    voxelgrid.setInputCloud(map_cloud_);
    auto filtered = std::make_shared<pcl::PointCloud<PointT>>();
    voxelgrid.filter(*filtered);
    map_cloud_ = filtered;
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
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "map is empty; nothing to save yet");
      return;
    }
    pcl::PointCloud<PointXYZIWeight> weighted_cloud;
    const std::size_t valid_weighted_points = ndt_mapping_localization::buildWeightedMap(
        map_cloud_, weight_resolution_, weight_min_points_, weighted_cloud);
    if (pcl::io::savePCDFileBinary(map_path_, weighted_cloud) != 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "failed to save map: %s", map_path_.c_str());
    } else {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "saved weighted map: %s (%zu points, %zu valid weighted points)",
          map_path_.c_str(), map_cloud_->size(), valid_weighted_points);
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
  double keyframe_delta_trans_;
  double keyframe_delta_angle_;
  double map_resolution_;
  double publish_interval_;
  bool publish_map_{false};
  double save_interval_;
  double transform_timeout_sec_;
  double weight_resolution_;
  int weight_min_points_;
  std::string map_frame_id_;
  std::string map_path_;
  bool has_last_keyframe_{false};
  Eigen::Isometry3d last_keyframe_pose_{Eigen::Isometry3d::Identity()};
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WeightedNdtMapBuilder>());
  rclcpp::shutdown();
  return 0;
}
