#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <ndt_omp/ndt_omp.h>

namespace {

pclomp::NeighborSearchMethod neighborSearchMethod(const std::string& name) {
  if (name == "DIRECT1") return pclomp::DIRECT1;
  if (name == "DIRECT7") return pclomp::DIRECT7;
  if (name == "DIRECT26") return pclomp::DIRECT26;
  throw std::invalid_argument(
      "ndt_neighbor_search must be DIRECT1, DIRECT7, or DIRECT26");
}

Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(),
      vector.z(), 0.0, -vector.x(),
      -vector.y(), vector.x(), 0.0;
  return matrix;
}

Eigen::Matrix3d so3LeftJacobian(const Eigen::Vector3d& omega) {
  const double theta = omega.norm();
  const Eigen::Matrix3d omega_matrix = skew(omega);
  const Eigen::Matrix3d omega_squared = omega_matrix * omega_matrix;
  if (theta < 1.0e-7) {
    return Eigen::Matrix3d::Identity() + 0.5 * omega_matrix +
        (1.0 / 6.0) * omega_squared;
  }
  return Eigen::Matrix3d::Identity() +
      ((1.0 - std::cos(theta)) / (theta * theta)) * omega_matrix +
      ((theta - std::sin(theta)) / (theta * theta * theta)) * omega_squared;
}

Eigen::Matrix4f scaleSe3Motion(const Eigen::Matrix4f& relative_motion, double scale) {
  const Eigen::Matrix3d rotation =
      relative_motion.block<3, 3>(0, 0).cast<double>();
  const Eigen::Vector3d translation =
      relative_motion.block<3, 1>(0, 3).cast<double>();

  const Eigen::AngleAxisd angle_axis(rotation);
  const Eigen::Vector3d omega = angle_axis.axis() * angle_axis.angle();
  const Eigen::Matrix3d jacobian = so3LeftJacobian(omega);
  const Eigen::Vector3d linear_component = jacobian.fullPivLu().solve(translation);

  const Eigen::Vector3d scaled_omega = scale * omega;
  const double scaled_angle = scaled_omega.norm();
  Eigen::Matrix3d scaled_rotation = Eigen::Matrix3d::Identity();
  if (scaled_angle > 1.0e-12) {
    scaled_rotation = Eigen::AngleAxisd(
        scaled_angle, scaled_omega / scaled_angle).toRotationMatrix();
  }
  const Eigen::Vector3d scaled_translation =
      so3LeftJacobian(scaled_omega) * (scale * linear_component);

  Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
  result.block<3, 3>(0, 0) = scaled_rotation.cast<float>();
  result.block<3, 1>(0, 3) = scaled_translation.cast<float>();
  return result;
}

class ClassicalNdtLocalizer : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  ClassicalNdtLocalizer()
      : rclcpp::Node("classical_ndt_localizer"),
        map_cloud_(new pcl::PointCloud<PointT>()),
        aligned_(new pcl::PointCloud<PointT>()),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_),
        tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)) {
    initializeParams();
    loadMap();
    initializeRegistration();

    // ROS2 입력 흐름: /filtered_points -> weighted_ndt_localizer -> pose/odom/aligned_points
    rclcpp::QoS input_qos = rclcpp::SensorDataQoS();
    if (reliable_input_qos_) {
      input_qos.reliable();
    }
    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, input_qos,
        std::bind(&ClassicalNdtLocalizer::callback, this, std::placeholders::_1));
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/ndt_localization/odom", 10);
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/ndt_localization/pose", 10);
    aligned_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/ndt_localization/aligned_points", 1);
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/ndt_localization/diagnostics", 50);

    // 첫 정합의 초기값이다. 실제 시작 위치가 맵 원점과 다르면 launch에서 initial_* 값을 맞춘다.
    current_pose_ = composeInitialPose();
  }

private:
  void initializeParams() {
    input_topic_ = declare_parameter<std::string>("input_topic", "/point_cloud");
    reliable_input_qos_ = declare_parameter<bool>("reliable_input_qos", false);
    map_path_ = declare_parameter<std::string>("map_path", "");
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    base_frame_id_ = declare_parameter<std::string>("base_frame_id", "base_link");
    downsample_resolution_ = declare_parameter<double>("source_downsample_resolution", 0.2);
    map_downsample_resolution_ = declare_parameter<double>("map_downsample_resolution", 0.2);
    ndt_resolution_ = declare_parameter<double>("ndt_resolution", 1.0);
    ndt_step_size_ = declare_parameter<double>("ndt_step_size", 0.1);
    ndt_num_threads_ = declare_parameter<int>("ndt_num_threads", 4);
    ndt_neighbor_search_ =
        declare_parameter<std::string>("ndt_neighbor_search", "DIRECT1");
    transformation_epsilon_ = declare_parameter<double>("transformation_epsilon", 0.01);
    max_iterations_ = declare_parameter<int>("max_iterations", 64);
    min_transformation_probability_ =
        declare_parameter<double>("min_transformation_probability", 0.2);
    use_constant_velocity_prediction_ =
        declare_parameter<bool>("use_constant_velocity_prediction", true);
    max_prediction_horizon_sec_ =
        declare_parameter<double>("max_prediction_horizon_sec", 0.5);
    max_prediction_scale_ = declare_parameter<double>("max_prediction_scale", 3.0);
    max_prediction_distance_m_ =
        declare_parameter<double>("max_prediction_distance_m", 1.5);
    max_consecutive_rejections_ =
        declare_parameter<int>("max_consecutive_rejections", 3);
    if (max_prediction_horizon_sec_ <= 0.0 || max_prediction_scale_ <= 0.0 ||
        max_prediction_distance_m_ <= 0.0 || max_consecutive_rejections_ < 1) {
      throw std::invalid_argument("constant-velocity prediction limits must be positive");
    }
    transform_timeout_sec_ = declare_parameter<double>("transform_timeout_sec", 0.2);
    initial_x_ = declare_parameter<double>("initial_x", 0.0);
    initial_y_ = declare_parameter<double>("initial_y", 0.0);
    initial_z_ = declare_parameter<double>("initial_z", 0.0);
    initial_roll_ = declare_parameter<double>("initial_roll", 0.0);
    initial_pitch_ = declare_parameter<double>("initial_pitch", 0.0);
    initial_yaw_ = declare_parameter<double>("initial_yaw", 0.0);
  }

  void loadMap() {
    // 먼저 PCD 맵을 로드한다. map_path가 비어 있거나 파일을 못 읽으면 위치추정을 시작할 수 없다.
    pcl::PCLPointCloud2 map_blob;
    if (map_path_.empty() || pcl::io::loadPCDFile(map_path_, map_blob) != 0) {
      throw std::runtime_error("failed to load map_pcd: " + map_path_);
    }
    pcl::fromPCLPointCloud2(map_blob, *map_cloud_);

    // 맵도 입력 스캔과 같은 해상도 기준으로 줄여 NDT target 구성 비용을 낮춘다.
    pcl::VoxelGrid<PointT> voxelgrid;
    voxelgrid.setLeafSize(
        map_downsample_resolution_, map_downsample_resolution_, map_downsample_resolution_);
    voxelgrid.setInputCloud(map_cloud_);
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid.filter(*filtered);
    map_cloud_ = filtered;
    RCLCPP_INFO(get_logger(), "loaded map '%s' (%zu points)", map_path_.c_str(), map_cloud_->size());
  }

  void initializeRegistration() {
    // Classical OpenMP NDT: semantic/PCA/distance weight를 사용하지 않는다.
    reg_.setResolution(ndt_resolution_);
    reg_.setStepSize(ndt_step_size_);
    reg_.setNumThreads(ndt_num_threads_);
    reg_.setNeighborhoodSearchMethod(neighborSearchMethod(ndt_neighbor_search_));
    reg_.setTransformationEpsilon(transformation_epsilon_);
    reg_.setMaximumIterations(max_iterations_);
    reg_.setInputTarget(map_cloud_);
    RCLCPP_INFO(
        get_logger(), "Classical OpenMP NDT initialized (no weights, search=%s)",
        ndt_neighbor_search_.c_str());
  }

  Eigen::Matrix4f composeInitialPose() const {
    Eigen::Affine3f pose =
        Eigen::Translation3f(initial_x_, initial_y_, initial_z_) *
        Eigen::AngleAxisf(initial_roll_, Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(initial_pitch_, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(initial_yaw_, Eigen::Vector3f::UnitZ());
    return pose.matrix();
  }

  struct AcceptedPose {
    Eigen::Matrix4f pose{Eigen::Matrix4f::Identity()};
    std::int64_t stamp_ns{0};
  };

  Eigen::Matrix4f predictPose(const rclcpp::Time& stamp) const {
    if (!use_constant_velocity_prediction_ || accepted_pose_history_.size() < 2) {
      return current_pose_;
    }

    const auto& previous = accepted_pose_history_[0];
    const auto& latest = accepted_pose_history_[1];
    const double history_dt =
        static_cast<double>(latest.stamp_ns - previous.stamp_ns) * 1.0e-9;
    double prediction_dt =
        static_cast<double>(stamp.nanoseconds() - latest.stamp_ns) * 1.0e-9;
    if (!(history_dt > 0.0) || !(prediction_dt > 0.0)) {
      return current_pose_;
    }

    prediction_dt = std::min(prediction_dt, max_prediction_horizon_sec_);
    const double scale = std::min(prediction_dt / history_dt, max_prediction_scale_);
    const Eigen::Matrix4f relative_motion = previous.pose.inverse() * latest.pose;
    Eigen::Matrix4f predicted_pose = latest.pose * scaleSe3Motion(relative_motion, scale);
    const double predicted_distance =
        (predicted_pose.block<3, 1>(0, 3) - latest.pose.block<3, 1>(0, 3)).norm();
    if (predicted_distance > max_prediction_distance_m_) {
      const Eigen::Matrix4f predicted_motion = latest.pose.inverse() * predicted_pose;
      predicted_pose = latest.pose * scaleSe3Motion(
          predicted_motion, max_prediction_distance_m_ / predicted_distance);
    }
    return predicted_pose;
  }

  void rememberAcceptedPose(const Eigen::Matrix4f& pose, const rclcpp::Time& stamp) {
    const std::int64_t stamp_ns = stamp.nanoseconds();
    if (!accepted_pose_history_.empty() &&
        stamp_ns <= accepted_pose_history_.back().stamp_ns) {
      accepted_pose_history_.clear();
    }
    accepted_pose_history_.push_back(AcceptedPose{pose, stamp_ns});
    while (accepted_pose_history_.size() > 2) {
      accepted_pose_history_.pop_front();
    }
  }

  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    const auto processing_start = std::chrono::steady_clock::now();
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
      publishDiagnostics(
          cloud_msg->header.stamp, false, false, 0,
          std::numeric_limits<double>::quiet_NaN(), 0, 0, 0.0, 0.0,
          elapsedMilliseconds(processing_start), "empty_input");
      return;
    }
    const std::size_t input_points = cloud->size();

    if (cloud_msg->header.frame_id != base_frame_id_) {
      try {
        const auto transform = tf_buffer_.lookupTransform(
            base_frame_id_, cloud_msg->header.frame_id, cloud_msg->header.stamp,
            tf2::durationFromSec(transform_timeout_sec_));
        Eigen::Quaternionf quaternion(
            transform.transform.rotation.w, transform.transform.rotation.x,
            transform.transform.rotation.y, transform.transform.rotation.z);
        quaternion.normalize();
        Eigen::Matrix4f base_from_sensor = Eigen::Matrix4f::Identity();
        base_from_sensor.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
        base_from_sensor(0, 3) = transform.transform.translation.x;
        base_from_sensor(1, 3) = transform.transform.translation.y;
        base_from_sensor(2, 3) = transform.transform.translation.z;
        pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*cloud, *transformed, base_from_sensor);
        cloud = transformed;
      } catch (const tf2::TransformException& exception) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000, "failed to lookup %s <- %s: %s",
            base_frame_id_.c_str(), cloud_msg->header.frame_id.c_str(), exception.what());
        publishDiagnostics(
            cloud_msg->header.stamp, false, false, 0,
            std::numeric_limits<double>::quiet_NaN(), input_points, 0, 0.0, 0.0,
            elapsedMilliseconds(processing_start), "transform_unavailable");
        return;
      }
    }

    // 매 스캔을 downsample한 뒤 직전 pose를 초기 추정값으로 사용해 맵에 정합한다.
    pcl::VoxelGrid<PointT> voxelgrid;
    voxelgrid.setLeafSize(downsample_resolution_, downsample_resolution_, downsample_resolution_);
    voxelgrid.setInputCloud(cloud);
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxelgrid.filter(*filtered);

    if (filtered->empty()) {
      publishDiagnostics(
          cloud_msg->header.stamp, false, false, 0,
          std::numeric_limits<double>::quiet_NaN(), input_points, 0, 0.0, 0.0,
          elapsedMilliseconds(processing_start), "downsample_empty");
      return;
    }

    const Eigen::Matrix4f predicted_pose = predictPose(cloud_msg->header.stamp);
    const double prediction_distance =
        (predicted_pose.block<3, 1>(0, 3) - current_pose_.block<3, 1>(0, 3)).norm();
    reg_.setInputSource(filtered);
    reg_.align(*aligned_, predicted_pose);
    const double probability = reg_.getTransformationProbability();
    const int iterations = reg_.getFinalNumIteration();
    const bool converged = reg_.hasConverged();
    const Eigen::Matrix4f candidate_pose = reg_.getFinalTransformation();
    const double correction_distance =
        (candidate_pose.block<3, 1>(0, 3) - predicted_pose.block<3, 1>(0, 3)).norm();
    // NDT 내부에서 max_iterations를 이미 적용한다. 내부 counter가 설정값보다 크게
    // 보고될 수 있으므로 iteration 수는 진단값으로만 기록한다.
    if (!converged || !std::isfinite(probability) ||
        probability < min_transformation_probability_) {
      std::string reason = "registration_rejected";
      if (!converged) {
        reason = "not_converged";
      } else if (!std::isfinite(probability)) {
        reason = "invalid_probability";
      } else if (probability < min_transformation_probability_) {
        reason = "low_probability";
      }
      publishDiagnostics(
          cloud_msg->header.stamp, false, converged, iterations, probability,
          input_points, filtered->size(), prediction_distance, correction_distance,
          elapsedMilliseconds(processing_start), reason);
      ++consecutive_rejections_;
      if (consecutive_rejections_ == max_consecutive_rejections_ &&
          !accepted_pose_history_.empty()) {
        accepted_pose_history_.clear();
        RCLCPP_WARN(
            get_logger(),
            "cleared constant-velocity history after %d consecutive rejections",
            max_consecutive_rejections_);
      }
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "classical ndt rejected (converged=%d, iterations=%d, probability=%.6f)",
          converged, iterations, probability);
      return;
    }

    // 수렴한 결과를 다음 프레임의 초기값으로 재사용한다.
    current_pose_ = candidate_pose;
    consecutive_rejections_ = 0;
    rememberAcceptedPose(current_pose_, cloud_msg->header.stamp);
    publishPose(cloud_msg->header.stamp);
    publishAligned(cloud_msg->header.stamp);
    publishDiagnostics(
        cloud_msg->header.stamp, true, converged, iterations, probability,
        input_points, filtered->size(), prediction_distance, correction_distance,
        elapsedMilliseconds(processing_start), "accepted");
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "classical ndt accepted (iterations=%d, probability=%.6f)",
        iterations, probability);
  }

  static double elapsedMilliseconds(
      const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }

  void publishDiagnostics(
      const rclcpp::Time& stamp, bool accepted, bool converged, int iterations,
      double probability, std::size_t input_points, std::size_t filtered_points,
      double prediction_distance, double correction_distance,
      double processing_time_ms, const std::string& reason) {
    diagnostic_msgs::msg::DiagnosticArray message;
    message.header.stamp = stamp;
    message.header.frame_id = map_frame_id_;

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.level = accepted
        ? diagnostic_msgs::msg::DiagnosticStatus::OK
        : diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.name = "classical_ndt_registration";
    status.hardware_id = "classical_ndt_omp";
    status.message = reason;

    const auto add_value = [&status](const std::string& key, const std::string& value) {
      diagnostic_msgs::msg::KeyValue item;
      item.key = key;
      item.value = value;
      status.values.push_back(item);
    };
    add_value("accepted", accepted ? "true" : "false");
    add_value("converged", converged ? "true" : "false");
    add_value("iterations", std::to_string(iterations));
    add_value("probability", std::to_string(probability));
    add_value("input_points", std::to_string(input_points));
    add_value("projected_points", std::to_string(filtered_points));
    add_value("ground_points", "0");
    add_value("nonground_points", "0");
    add_value("rejected_points", "0");
    add_value("prediction_distance_m", std::to_string(prediction_distance));
    add_value("correction_distance_m", std::to_string(correction_distance));
    add_value("processing_time_ms", std::to_string(processing_time_ms));
    add_value("localizer_type", "classical_ndt");
    add_value("map_path", map_path_);
    add_value("ndt_neighbor_search", ndt_neighbor_search_);
    add_value("ndt_resolution", std::to_string(ndt_resolution_));
    add_value("source_downsample_resolution", std::to_string(downsample_resolution_));
    message.status.push_back(status);
    diagnostics_pub_->publish(message);
  }

  geometry_msgs::msg::TransformStamped matrixToTransform(const rclcpp::Time& stamp) const {
    Eigen::Quaternionf quat(current_pose_.block<3, 3>(0, 0));
    quat.normalize();

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = map_frame_id_;
    tf_msg.child_frame_id = base_frame_id_;
    tf_msg.transform.translation.x = current_pose_(0, 3);
    tf_msg.transform.translation.y = current_pose_(1, 3);
    tf_msg.transform.translation.z = current_pose_(2, 3);
    tf_msg.transform.rotation.x = quat.x();
    tf_msg.transform.rotation.y = quat.y();
    tf_msg.transform.rotation.z = quat.z();
    tf_msg.transform.rotation.w = quat.w();
    return tf_msg;
  }

  void publishPose(const rclcpp::Time& stamp) {
    // TF와 Odometry/Pose 토픽을 같은 pose에서 생성해 RViz와 다른 노드가 동일한 좌표계를 보게 한다.
    const auto tf_msg = matrixToTransform(stamp);
    tf_broadcaster_->sendTransform(tf_msg);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = map_frame_id_;
    odom.child_frame_id = base_frame_id_;
    odom.pose.pose.position.x = current_pose_(0, 3);
    odom.pose.pose.position.y = current_pose_(1, 3);
    odom.pose.pose.position.z = current_pose_(2, 3);
    odom.pose.pose.orientation = tf_msg.transform.rotation;
    odom_pub_->publish(odom);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom.header;
    pose.pose = odom.pose.pose;
    pose_pub_->publish(pose);
  }

  void publishAligned(const rclcpp::Time& stamp) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*aligned_, msg);
    msg.header.frame_id = map_frame_id_;
    msg.header.stamp = stamp;
    aligned_pub_->publish(msg);
  }

  std::string input_topic_;
  bool reliable_input_qos_{false};
  std::string map_path_;
  std::string map_frame_id_;
  std::string base_frame_id_;
  double downsample_resolution_;
  double map_downsample_resolution_;
  double ndt_resolution_;
  double ndt_step_size_;
  int ndt_num_threads_;
  std::string ndt_neighbor_search_;
  double transformation_epsilon_;
  int max_iterations_;
  double min_transformation_probability_;
  bool use_constant_velocity_prediction_;
  double max_prediction_horizon_sec_;
  double max_prediction_scale_;
  double max_prediction_distance_m_;
  int max_consecutive_rejections_;
  double transform_timeout_sec_;
  double initial_x_;
  double initial_y_;
  double initial_z_;
  double initial_roll_;
  double initial_pitch_;
  double initial_yaw_;

  pcl::PointCloud<PointT>::Ptr map_cloud_;
  pcl::PointCloud<PointT>::Ptr aligned_;
  pclomp::NormalDistributionsTransform<PointT, PointT> reg_;
  Eigen::Matrix4f current_pose_;
  std::deque<AcceptedPose> accepted_pose_history_;
  int consecutive_rejections_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<ClassicalNdtLocalizer>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("classical_ndt_localizer"), "%s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
