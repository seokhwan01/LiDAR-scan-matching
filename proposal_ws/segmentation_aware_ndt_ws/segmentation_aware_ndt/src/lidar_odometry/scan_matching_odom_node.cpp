#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <lidar_odometry/ros_utils.hpp>
#include <ndt_pca/ndt_pca.h>

namespace lidar_odometry {

class ScanMatchingOdomNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ScanMatchingOdomNode()
      : rclcpp::Node("scan_matching_odom_node"),
        tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)) {
    initializeParams();

    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::SensorDataQoS(),
        std::bind(&ScanMatchingOdomNode::cloudCallback, this, std::placeholders::_1));
    read_until_pub_ = create_publisher<std_msgs::msg::Header>("/scan_matching_odom/read_until", 256);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 256);
  }

  ~ScanMatchingOdomNode() override {
    if (fp_odom_) {
      std::fclose(fp_odom_);
    }
  }

private:
  void initializeParams() {
    input_topic_ = declare_parameter<std::string>("input_topic", "/filtered_points");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    odom_frame_id_ = declare_parameter<std::string>("odom_frame_id", "odom");
    keyframe_delta_trans_ = declare_parameter<double>("keyframe_delta_trans", 5.0);
    keyframe_delta_angle_ = declare_parameter<double>("keyframe_delta_angle", 0.17);
    keyframe_delta_time_ = declare_parameter<double>("keyframe_delta_time", 1.0);

    const auto calib_file = declare_parameter<std::string>("calib_file", "");
    const auto odom_file = declare_parameter<std::string>("odom_file", "");

    tf_velo2cam_.setIdentity();
    if (!calib_file.empty()) {
      std::ifstream fin(calib_file);
      if (fin) {
        std::string tmp;
        for (int i = 0; i < 4; ++i) {
          std::getline(fin, tmp);
        }
        fin >> tmp >> tf_velo2cam_(0, 0) >> tf_velo2cam_(0, 1) >> tf_velo2cam_(0, 2) >> tf_velo2cam_(0, 3) >>
            tf_velo2cam_(1, 0) >> tf_velo2cam_(1, 1) >> tf_velo2cam_(1, 2) >> tf_velo2cam_(1, 3) >>
            tf_velo2cam_(2, 0) >> tf_velo2cam_(2, 1) >> tf_velo2cam_(2, 2) >> tf_velo2cam_(2, 3);
      } else {
        RCLCPP_WARN(get_logger(), "failed to open calib_file '%s'; using identity transform", calib_file.c_str());
      }
    }

    if (!odom_file.empty()) {
      fp_odom_ = std::fopen(odom_file.c_str(), "w");
      if (!fp_odom_) {
        RCLCPP_WARN(get_logger(), "failed to open odom_file '%s'", odom_file.c_str());
      }
    }

    reg_s2k_.setResolution(declare_parameter<double>("ndt_resolution", 1.0));
    reg_s2k_.setNumThreads(declare_parameter<int>("ndt_num_threads", 4));
    reg_s2k_.setNeighborhoodSearchMethod(pclpca::DIRECT1);
    reg_s2k_.setTransformationEpsilon(declare_parameter<double>("transformation_epsilon", 0.01));
    reg_s2k_.setMaximumIterations(declare_parameter<int>("max_iterations", 64));

    odom_velo_.setIdentity();
    tf_s2k_error_.setIdentity();
    scan_count_ = 0;
    key_id_ = 0;
    total_processing_time_sec_ = 0.0;
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    auto cloud = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
      return;
    }

    const Eigen::Matrix4d pose = matchingS2K(cloud_msg->header.stamp, cloud);
    writeOdomFile(pose);
    publishOdometry(cloud_msg->header.stamp, cloud_msg->header.frame_id, pose);
    ++scan_count_;

    std_msgs::msg::Header read_until;
    read_until.stamp = rclcpp::Time(cloud_msg->header.stamp) + rclcpp::Duration::from_seconds(1.0);
    read_until.frame_id = "/velodyne_points";
    read_until_pub_->publish(read_until);
    read_until.frame_id = input_topic_;
    read_until_pub_->publish(read_until);
  }

  Eigen::Matrix4d matchingS2K(
      const rclcpp::Time& stamp,
      const pcl::PointCloud<PointT>::ConstPtr& cloud) {
    filtered_ = cloud;
    if (scan_count_ == 0) {
      key_ = filtered_;
      reg_s2k_.setInputTarget(key_);
      key_id_ = scan_count_;
      guess_trans_.setIdentity();
      guess_trans_(0, 3) = 1.5;
      pre_tf_s2k_.setIdentity();
      key_pose_.setIdentity();
      keyframe_stamp_ = stamp;
      return Eigen::Matrix4d::Identity();
    }

    const auto start = now();
    auto matched = std::make_shared<pcl::PointCloud<PointT>>();

    reg_s2k_.setInputSource(filtered_);
    reg_s2k_.align(*matched, guess_trans_.cast<float>());
    tf_s2k_ = reg_s2k_.getFinalTransformation().cast<double>();
    if (scan_count_ == 1) {
      reg_s2k_.align(*matched, tf_s2k_.cast<float>());
      tf_s2k_ = reg_s2k_.getFinalTransformation().cast<double>();
    }

    tf_s2s_ = pre_tf_s2k_.inverse() * tf_s2k_;
    tf_s2k_error_ = tf_s2k_;
    odom_velo_ = key_pose_ * tf_s2k_;

    const double dx_s2k = tf_s2k_.block<3, 1>(0, 3).norm();
    const double qw = std::clamp(Eigen::Quaterniond(tf_s2k_.block<3, 3>(0, 0)).w(), -1.0, 1.0);
    const double da_s2k = 2.0 * std::acos(qw);
    const double dt_s2k = (stamp - keyframe_stamp_).seconds();
    if (dx_s2k > keyframe_delta_trans_ || da_s2k > keyframe_delta_angle_ || dt_s2k > keyframe_delta_time_) {
      key_ = filtered_;
      reg_s2k_.setInputTarget(key_);
      key_id_ = scan_count_;
      tf_s2k_.setIdentity();
      key_pose_ = odom_velo_;
      keyframe_stamp_ = stamp;
    }

    pre_tf_s2k_ = tf_s2k_;
    guess_trans_ = pre_tf_s2k_ * tf_s2s_;

    const double elapsed = (now() - start).seconds();
    total_processing_time_sec_ += elapsed;
    RCLCPP_INFO(
        get_logger(), "OdomCount %d processing %.3f ms avg %.3f ms",
        scan_count_, elapsed * 1000.0, (total_processing_time_sec_ / scan_count_) * 1000.0);

    return odom_velo_;
  }

  void writeOdomFile(const Eigen::Matrix4d& pose) {
    if (!fp_odom_) {
      return;
    }

    const Eigen::Matrix4d odom = tf_velo2cam_ * pose * tf_velo2cam_.inverse();
    std::fprintf(
        fp_odom_, "%le %le %le %le %le %le %le %le %le %le %le %le\n",
        odom(0, 0), odom(0, 1), odom(0, 2), odom(0, 3),
        odom(1, 0), odom(1, 1), odom(1, 2), odom(1, 3),
        odom(2, 0), odom(2, 1), odom(2, 2), odom(2, 3));
  }

  void publishOdometry(
      const rclcpp::Time& stamp,
      const std::string& base_frame_id,
      const Eigen::Matrix4d& pose) {
    const auto odom_trans = matrix2transform(stamp, pose.cast<float>(), odom_frame_id_, base_frame_id);
    tf_broadcaster_->sendTransform(odom_trans);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_id_;
    odom.child_frame_id = base_frame_id;
    odom.pose.pose.position.x = pose(0, 3);
    odom.pose.pose.position.y = pose(1, 3);
    odom.pose.pose.position.z = pose(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;
    odom_pub_->publish(odom);
  }

  std::string input_topic_;
  std::string odom_topic_;
  std::string odom_frame_id_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::Header>::SharedPtr read_until_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  double keyframe_delta_trans_;
  double keyframe_delta_angle_;
  double keyframe_delta_time_;
  rclcpp::Time keyframe_stamp_;

  pclpca::NormalDistributionsTransform<PointT, PointT> reg_s2k_;
  FILE* fp_odom_{nullptr};
  Eigen::Matrix4d tf_velo2cam_;
  double total_processing_time_sec_;
  Eigen::Matrix4d guess_trans_;
  int scan_count_;
  int key_id_;
  pcl::PointCloud<PointT>::ConstPtr filtered_;
  pcl::PointCloud<PointT>::ConstPtr key_;
  Eigen::Matrix4d tf_s2s_;
  Eigen::Matrix4d tf_s2k_;
  Eigen::Matrix4d key_pose_;
  Eigen::Matrix4d pre_tf_s2k_;
  Eigen::Matrix4d odom_velo_;
  Eigen::Matrix4d tf_s2k_error_;
};

}  // namespace lidar_odometry

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lidar_odometry::ScanMatchingOdomNode>());
  rclcpp::shutdown();
  return 0;
}
