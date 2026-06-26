#include <chrono>
#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

using std::placeholders::_1;

class NDTLocalizerNode : public rclcpp::Node
{
public:
  NDTLocalizerNode()
  : Node("ndt_localizer_node")
  {
    // -----------------------------
    // Parameters
    // -----------------------------
    map_path_ = this->declare_parameter<std::string>(
      "map_path",
      "maps/carla_map_20260504_193051/carla_ndt_map_voxel_0.15_20260504_193051.pcd"
    );

    scan_topic_ = this->declare_parameter<std::string>(
      "scan_topic",
      "/carla/front_lidar/point_cloud"
    );

    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");

    use_gt_initial_guess_ = this->declare_parameter<bool>(
      "use_gt_initial_guess",
      false
    );

    gt_pose_topic_ = this->declare_parameter<std::string>(
      "gt_pose_topic",
      "/carla_lidar_gt_pose"
    );

    initial_x_ = this->declare_parameter<double>("initial_x", 0.0);
    initial_y_ = this->declare_parameter<double>("initial_y", 0.0);
    initial_z_ = this->declare_parameter<double>("initial_z", 0.0);
    initial_roll_deg_ = this->declare_parameter<double>("initial_roll_deg", 0.0);
    initial_pitch_deg_ = this->declare_parameter<double>("initial_pitch_deg", 0.0);
    initial_yaw_deg_ = this->declare_parameter<double>("initial_yaw_deg", 0.0);

    ndt_resolution_ = this->declare_parameter<double>("ndt_resolution", 1.0);
    step_size_ = this->declare_parameter<double>("step_size", 0.1);
    trans_eps_ = this->declare_parameter<double>("trans_eps", 0.01);
    max_iter_ = this->declare_parameter<int>("max_iter", 40);

    scan_voxel_size_ = this->declare_parameter<double>("scan_voxel_size", 0.3);

    fitness_warn_threshold_ = this->declare_parameter<double>(
      "fitness_warn_threshold",
      2.0
    );

    // -----------------------------
    // Load map
    // -----------------------------
    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_path_, *map_cloud_) < 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load PCD map: %s", map_path_.c_str());
      throw std::runtime_error("Failed to load PCD map");
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Loaded map: %s, points: %zu",
      map_path_.c_str(),
      map_cloud_->size()
    );

    // -----------------------------
    // Setup NDT
    // -----------------------------
    ndt_.setTransformationEpsilon(trans_eps_);
    ndt_.setStepSize(step_size_);
    ndt_.setResolution(ndt_resolution_);
    ndt_.setMaximumIterations(max_iter_);
    ndt_.setInputTarget(map_cloud_);

    // -----------------------------
    // Fallback initial guess
    // -----------------------------
    previous_pose_ = makeTransform(
      initial_x_,
      initial_y_,
      initial_z_,
      deg2rad(initial_roll_deg_),
      deg2rad(initial_pitch_deg_),
      deg2rad(initial_yaw_deg_)
    );

    has_previous_pose_ = true;

    RCLCPP_INFO(
      this->get_logger(),
      "Fallback initial pose: x=%.3f y=%.3f z=%.3f roll=%.3f pitch=%.3f yaw=%.3f deg",
      initial_x_,
      initial_y_,
      initial_z_,
      initial_roll_deg_,
      initial_pitch_deg_,
      initial_yaw_deg_
    );

    RCLCPP_INFO(
      this->get_logger(),
      "use_gt_initial_guess: %s",
      use_gt_initial_guess_ ? "true" : "false"
    );

    // -----------------------------
    // ROS interfaces
    // -----------------------------
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/ndt_pose",
      10
    );

    aligned_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/ndt_aligned_cloud",
      10
    );

    gt_initial_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/gt_initial_aligned_cloud",
      10
    );

    rclcpp::SensorDataQoS sensor_qos;

    scan_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      scan_topic_,
      sensor_qos,
      std::bind(&NDTLocalizerNode::scanCallback, this, _1)
    );

    if (use_gt_initial_guess_) {
      gt_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        gt_pose_topic_,
        10,
        std::bind(&NDTLocalizerNode::gtPoseCallback, this, _1)
      );

      RCLCPP_INFO(
        this->get_logger(),
        "Subscribed GT pose topic: %s",
        gt_pose_topic_.c_str()
      );
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Subscribed scan topic: %s",
      scan_topic_.c_str()
    );
  }

private:
  double deg2rad(double deg)
  {
    return deg * M_PI / 180.0;
  }

  Eigen::Matrix4f makeTransform(
    double x,
    double y,
    double z,
    double roll,
    double pitch,
    double yaw
  )
  {
    Eigen::AngleAxisf roll_angle(static_cast<float>(roll), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch_angle(static_cast<float>(pitch), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yaw_angle(static_cast<float>(yaw), Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f rotation = (yaw_angle * pitch_angle * roll_angle).matrix();

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rotation;
    transform(0, 3) = static_cast<float>(x);
    transform(1, 3) = static_cast<float>(y);
    transform(2, 3) = static_cast<float>(z);

    return transform;
  }

  Eigen::Matrix4f poseMsgToMatrix(
    const geometry_msgs::msg::PoseStamped & msg
  )
  {
    Eigen::Quaternionf q(
      static_cast<float>(msg.pose.orientation.w),
      static_cast<float>(msg.pose.orientation.x),
      static_cast<float>(msg.pose.orientation.y),
      static_cast<float>(msg.pose.orientation.z)
    );

    q.normalize();

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = q.toRotationMatrix();

    transform(0, 3) = static_cast<float>(msg.pose.position.x);
    transform(1, 3) = static_cast<float>(msg.pose.position.y);
    transform(2, 3) = static_cast<float>(msg.pose.position.z);

    return transform;
  }

  geometry_msgs::msg::PoseStamped matrixToPoseMsg(
    const Eigen::Matrix4f & transform,
    const rclcpp::Time & stamp
  )
  {
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = map_frame_;

    Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
    Eigen::Quaternionf q(rotation);
    q.normalize();

    pose_msg.pose.position.x = transform(0, 3);
    pose_msg.pose.position.y = transform(1, 3);
    pose_msg.pose.position.z = transform(2, 3);

    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();

    return pose_msg;
  }

  void gtPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    latest_gt_pose_ = poseMsgToMatrix(*msg);
    has_gt_pose_ = true;

    gt_msg_count_++;

    if (gt_msg_count_ % 20 == 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "GT initial guess received: x=%.3f y=%.3f z=%.3f",
        latest_gt_pose_(0, 3),
        latest_gt_pose_(1, 3),
        latest_gt_pose_(2, 3)
      );
    }
  }

  Eigen::Matrix4f selectInitialGuess()
  {
    if (use_gt_initial_guess_ && has_gt_pose_) {
      initial_guess_source_ = "gt_pose";
      return latest_gt_pose_;
    }

    if (use_gt_initial_guess_ && !has_gt_pose_) {
      no_gt_warn_count_++;

      if (no_gt_warn_count_ % 30 == 1) {
        RCLCPP_WARN(
          this->get_logger(),
          "use_gt_initial_guess=true but no GT pose received yet. Falling back to previous_pose."
        );
      }
    }

    initial_guess_source_ = "previous_pose";
    return previous_pose_;
  }

  void publishInitialGuessCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & scan_filtered,
    const Eigen::Matrix4f & initial_guess,
    const rclcpp::Time & stamp
  )
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr gt_initial_cloud(
      new pcl::PointCloud<pcl::PointXYZ>
    );

    pcl::transformPointCloud(
      *scan_filtered,
      *gt_initial_cloud,
      initial_guess
    );

    sensor_msgs::msg::PointCloud2 gt_initial_msg;
    pcl::toROSMsg(*gt_initial_cloud, gt_initial_msg);
    gt_initial_msg.header.stamp = stamp;
    gt_initial_msg.header.frame_id = map_frame_;

    gt_initial_cloud_pub_->publish(gt_initial_msg);
  }

  void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_raw(
      new pcl::PointCloud<pcl::PointXYZ>
    );

    pcl::fromROSMsg(*msg, *scan_raw);

    if (scan_raw->empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty scan.");
      return;
    }

    // -----------------------------
    // Downsample current scan
    // -----------------------------
    pcl::PointCloud<pcl::PointXYZ>::Ptr scan_filtered(
      new pcl::PointCloud<pcl::PointXYZ>
    );

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(scan_raw);
    voxel_filter.setLeafSize(
      static_cast<float>(scan_voxel_size_),
      static_cast<float>(scan_voxel_size_),
      static_cast<float>(scan_voxel_size_)
    );
    voxel_filter.filter(*scan_filtered);

    if (scan_filtered->empty()) {
      RCLCPP_WARN(this->get_logger(), "Filtered scan is empty.");
      return;
    }

    // -----------------------------
    // Select initial guess
    // -----------------------------
    Eigen::Matrix4f initial_guess = selectInitialGuess();

    // -----------------------------
    // Debug cloud:
    // current scan transformed only by initial guess
    // -----------------------------
    publishInitialGuessCloud(
      scan_filtered,
      initial_guess,
      msg->header.stamp
    );

    // -----------------------------
    // NDT alignment
    // -----------------------------
    ndt_.setInputSource(scan_filtered);

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(
      new pcl::PointCloud<pcl::PointXYZ>
    );

    ndt_.align(*aligned_cloud, initial_guess);

    if (!ndt_.hasConverged()) {
      RCLCPP_WARN(this->get_logger(), "NDT did not converge.");
      return;
    }

    Eigen::Matrix4f ndt_pose = ndt_.getFinalTransformation();
    double fitness_score = ndt_.getFitnessScore();

    previous_pose_ = ndt_pose;
    has_previous_pose_ = true;

    RCLCPP_INFO(
      this->get_logger(),
      "NDT converged. source=%s fitness=%.4f, x=%.3f, y=%.3f, z=%.3f",
      initial_guess_source_.c_str(),
      fitness_score,
      ndt_pose(0, 3),
      ndt_pose(1, 3),
      ndt_pose(2, 3)
    );

    if (fitness_score > fitness_warn_threshold_) {
      RCLCPP_WARN(
        this->get_logger(),
        "High fitness score: %.4f. Localization may be unstable.",
        fitness_score
      );
    }

    // -----------------------------
    // Publish pose
    // -----------------------------
    auto pose_msg = matrixToPoseMsg(ndt_pose, msg->header.stamp);
    pose_pub_->publish(pose_msg);

    // -----------------------------
    // Publish NDT aligned cloud
    // -----------------------------
    sensor_msgs::msg::PointCloud2 aligned_msg;
    pcl::toROSMsg(*aligned_cloud, aligned_msg);
    aligned_msg.header.stamp = msg->header.stamp;
    aligned_msg.header.frame_id = map_frame_;
    aligned_pub_->publish(aligned_msg);
  }

private:
  std::string map_path_;
  std::string scan_topic_;
  std::string map_frame_;

  bool use_gt_initial_guess_{false};
  std::string gt_pose_topic_;
  bool has_gt_pose_{false};
  Eigen::Matrix4f latest_gt_pose_{Eigen::Matrix4f::Identity()};
  int gt_msg_count_{0};
  int no_gt_warn_count_{0};
  std::string initial_guess_source_{"previous_pose"};

  double initial_x_;
  double initial_y_;
  double initial_z_;
  double initial_roll_deg_;
  double initial_pitch_deg_;
  double initial_yaw_deg_;

  double ndt_resolution_;
  double step_size_;
  double trans_eps_;
  int max_iter_;
  double scan_voxel_size_;
  double fitness_warn_threshold_;

  bool has_previous_pose_{false};
  Eigen::Matrix4f previous_pose_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gt_pose_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gt_initial_cloud_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<NDTLocalizerNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}