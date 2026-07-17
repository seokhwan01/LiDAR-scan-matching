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
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <ndt_omp/ndt_omp.h>
#include <segmentation_aware_ndt/lego_loam_segmentation.hpp>
#include <segmentation_aware_ndt/weighted_ndt_voxel.hpp>

namespace {

pclomp::NeighborSearchMethod neighborSearchMethod(const std::string& name) {
  if (name == "DIRECT1") return pclomp::DIRECT1;
  if (name == "DIRECT7") return pclomp::DIRECT7;
  if (name == "DIRECT26") return pclomp::DIRECT26;
  throw std::invalid_argument(
      "ndt_neighbor_search must be DIRECT1, DIRECT7, or DIRECT26");
}

// 아래 SE(3) 지수/로그 연산에서 사용하는 SO(3) skew 행렬이다.
Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(),
      vector.z(), 0.0, -vector.x(),
      -vector.y(), vector.x(), 0.0;
  return matrix;
}

// Exp([v, omega])의 translation이 V * v가 되게 하는 left Jacobian V(omega)이다.
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

// SE(3)의 상대 rigid-body motion에 scale을 적용한다. 이 연산은
// Exp(scale * Log(relative_motion))과 같으며 결합된 translation/rotation을 보존한다.
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

// NDT 양쪽의 기하 class를 비교하는 pre-built map localizer이다.
//
// 입력 cloud는 먼저 lidar 좌표계에서 segmentation된다. 정합 전 정적
// base_link <- lidar extrinsic으로 변환하므로 최적화하는 6개 파라미터는
// map <- base_link를 직접 표현한다. 따라서 CARLA의 ground-truth
// map -> base_link TF는 localizer에서 사용하지 않는다.
class SemanticNdtLocalizer : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<PointT>;

  SemanticNdtLocalizer()
      : rclcpp::Node("semantic_ndt_localizer"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_),
        tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
        map_cloud_(new Cloud()),
        aligned_cloud_(new Cloud()) {
    initializeParameters();
    if (use_precomputed_ndt_map_) {
      loadPrecomputedNdtMap();
    } else {
      loadMap();
    }
    initializeRegistration();
    current_pose_ = composeInitialPose();

    rclcpp::QoS input_qos = rclcpp::SensorDataQoS();
    if (reliable_input_qos_) {
      input_qos.reliable();
    }
    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, input_qos,
        std::bind(&SemanticNdtLocalizer::cloudCallback, this, std::placeholders::_1));
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/ndt_localization/odom", 10);
    pose_pub_ =
        create_publisher<geometry_msgs::msg::PoseStamped>("/ndt_localization/pose", 10);
    aligned_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/ndt_localization/aligned_points", 1);
    segmented_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/ndt_localization/segmented_points", rclcpp::SensorDataQoS());
    diagnostics_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "/ndt_localization/diagnostics", 50);

    RCLCPP_INFO(
        get_logger(),
        "source segmentation: %d rings x %d columns; frame output is %s",
        segmentation_parameters_.n_scan, segmentation_parameters_.horizon_scan,
        base_frame_id_.c_str());
  }

private:
  void initializeParameters() {
    input_topic_ = declare_parameter<std::string>("input_topic", "/point_cloud");
    reliable_input_qos_ = declare_parameter<bool>("reliable_input_qos", false);
    map_path_ = declare_parameter<std::string>("map_path", "./semantic_weighted_map.pcd");
    ndt_map_path_ =
        declare_parameter<std::string>("ndt_map_path", "./weighted_ndt_voxels.pcd");
    use_precomputed_ndt_map_ =
        declare_parameter<bool>("use_precomputed_ndt_map", true);
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    base_frame_id_ = declare_parameter<std::string>("base_frame_id", "base_link");
    transform_timeout_sec_ = declare_parameter<double>("transform_timeout_sec", 0.2);

    map_downsample_resolution_ =
        declare_parameter<double>("map_downsample_resolution", 0.2);
    source_downsample_resolution_ =
        declare_parameter<double>("source_downsample_resolution", 0.2);
    ndt_resolution_ = declare_parameter<double>("ndt_resolution", 1.0);
    ndt_min_points_ = declare_parameter<int>("ndt_min_points", 6);
    ndt_step_size_ = declare_parameter<double>("ndt_step_size", 0.1);
    ndt_num_threads_ = declare_parameter<int>("ndt_num_threads", 4);
    ndt_neighbor_search_ =
        declare_parameter<std::string>("ndt_neighbor_search", "DIRECT1");
    transformation_epsilon_ = declare_parameter<double>("transformation_epsilon", 0.01);
    max_iterations_ = declare_parameter<int>("max_iterations", 64);
    min_transformation_probability_ =
        declare_parameter<double>("min_transformation_probability", 0.2);
    require_weighted_map_ = declare_parameter<bool>("require_weighted_map", true);
    use_constant_velocity_prediction_ =
        declare_parameter<bool>("use_constant_velocity_prediction", true);
    max_prediction_horizon_sec_ =
        declare_parameter<double>("max_prediction_horizon_sec", 0.5);
    max_prediction_scale_ = declare_parameter<double>("max_prediction_scale", 3.0);
    max_prediction_distance_m_ =
        declare_parameter<double>("max_prediction_distance_m", 1.5);
    max_consecutive_rejections_ =
        declare_parameter<int>("max_consecutive_rejections", 3);
    reinitialize_after_rejections_ =
        declare_parameter<int>("reinitialize_after_rejections", 3);
    use_correction_limit_ =
        declare_parameter<bool>("use_correction_limit", true);
    max_correction_distance_m_ =
        declare_parameter<double>("max_correction_distance_m", 1.0);
    correction_prediction_ratio_ =
        declare_parameter<double>("correction_prediction_ratio", 0.2);
    use_velocity_consistency_check_ =
        declare_parameter<bool>("use_velocity_consistency_check", true);
    max_accepted_speed_mps_ =
        declare_parameter<double>("max_accepted_speed_mps", 40.0);
    max_accepted_yaw_rate_dps_ =
        declare_parameter<double>("max_accepted_yaw_rate_dps", 120.0);
    use_adaptive_recovery_ =
        declare_parameter<bool>("use_adaptive_recovery", false);
    adaptive_recovery_after_rejections_ =
        declare_parameter<int>("adaptive_recovery_after_rejections", 3);
    adaptive_recovery_exit_accepts_ =
        declare_parameter<int>("adaptive_recovery_exit_accepts", 10);
    recovery_use_target_ndt_weight_ =
        declare_parameter<bool>("recovery_use_target_ndt_weight", false);
    recovery_use_source_semantic_weight_ =
        declare_parameter<bool>("recovery_use_source_semantic_weight", false);
    recovery_use_source_ground_downsampling_ =
        declare_parameter<bool>("recovery_use_source_ground_downsampling", false);
    recovery_ground_keep_ratio_ =
        declare_parameter<double>("recovery_ground_keep_ratio", 1.0);
    recovery_semantic_mismatch_weight_ =
        declare_parameter<double>("recovery_semantic_mismatch_weight", 1.0);
    recovery_min_transformation_probability_ =
        declare_parameter<double>("recovery_min_transformation_probability", -1.0);
    if (max_prediction_horizon_sec_ <= 0.0 || max_prediction_scale_ <= 0.0 ||
        max_prediction_distance_m_ <= 0.0 || max_consecutive_rejections_ < 1 ||
        reinitialize_after_rejections_ < 1 || max_correction_distance_m_ <= 0.0 ||
        correction_prediction_ratio_ < 0.0 || max_accepted_speed_mps_ <= 0.0 ||
        max_accepted_yaw_rate_dps_ <= 0.0 || adaptive_recovery_after_rejections_ < 1 ||
        adaptive_recovery_exit_accepts_ < 1) {
      throw std::invalid_argument("constant-velocity prediction limits must be positive");
    }
    if (!(recovery_ground_keep_ratio_ > 0.0 && recovery_ground_keep_ratio_ <= 1.0)) {
      throw std::invalid_argument("recovery_ground_keep_ratio must be in (0, 1]");
    }
    if (recovery_semantic_mismatch_weight_ < 0.0 ||
        recovery_semantic_mismatch_weight_ > 1.0) {
      throw std::invalid_argument("recovery_semantic_mismatch_weight must be in [0, 1]");
    }

    // source-map semantic 비교가 제안 방식이다. 통제된 ablation test를 위해
    // runtime distance weighting은 별도 설정으로 켜고 끌 수 있다.
    use_source_semantic_weight_ =
        declare_parameter<bool>("use_source_semantic_weight", true);
    use_source_segmentation_ =
        declare_parameter<bool>("use_source_segmentation", true);
    preserve_unlabeled_points_ =
        declare_parameter<bool>("preserve_unlabeled_points", false);
    use_source_ground_downsampling_ =
        declare_parameter<bool>("use_source_ground_downsampling", true);
    ground_keep_ratio_ = declare_parameter<double>("ground_keep_ratio", 0.5);
    use_target_ndt_weight_ =
        declare_parameter<bool>("use_target_ndt_weight", true);
    semantic_class_threshold_ =
        declare_parameter<double>("semantic_class_threshold", 0.75);
    semantic_mismatch_weight_ =
        declare_parameter<double>("semantic_mismatch_weight", 0.1);
    use_continuous_semantic_compatibility_ =
        declare_parameter<bool>("use_continuous_semantic_compatibility", true);
    dimension_scale_linear_ =
        declare_parameter<double>("dimension_scale_linear", 0.75);
    dimension_scale_planar_ =
        declare_parameter<double>("dimension_scale_planar", 1.25);
    dimension_scale_volumetric_ =
        declare_parameter<double>("dimension_scale_volumetric", 1.0);
    use_runtime_distance_weight_ =
        declare_parameter<bool>("use_runtime_distance_weight", false);

    initial_x_ = declare_parameter<double>("initial_x", 0.0);
    initial_y_ = declare_parameter<double>("initial_y", 0.0);
    initial_z_ = declare_parameter<double>("initial_z", 0.0);
    initial_roll_ = declare_parameter<double>("initial_roll", 0.0);
    initial_pitch_ = declare_parameter<double>("initial_pitch", 0.0);
    initial_yaw_ = declare_parameter<double>("initial_yaw", 0.0);

    // 이 기본값은 semantic_map_builder와 공유하며 CARLA publisher의 32채널
    // 센서 설정과 일치한다.
    segmentation_parameters_.n_scan = declare_parameter<int>("n_scan", 32);
    segmentation_parameters_.horizon_scan = declare_parameter<int>("horizon_scan", 1084);
    segmentation_parameters_.ang_res_x =
        declare_parameter<double>("ang_res_x", 360.0 / 1084.0);
    segmentation_parameters_.ang_res_y =
        declare_parameter<double>("ang_res_y", 40.0 / 31.0);
    segmentation_parameters_.ang_bottom = declare_parameter<double>("ang_bottom", 30.0);
    segmentation_parameters_.ground_scan_ind = declare_parameter<int>("ground_scan_ind", 20);
    segmentation_parameters_.sensor_minimum_range =
        declare_parameter<double>("sensor_minimum_range", 1.0);
    segmentation_parameters_.sensor_mount_angle =
        declare_parameter<double>("sensor_mount_angle", 0.0);
    segmentation_parameters_.segment_theta =
        declare_parameter<double>("segment_theta", 60.0 * M_PI / 180.0);
    segmentation_parameters_.segment_valid_point_num =
        declare_parameter<int>("segment_valid_point_num", 5);
    segmentation_parameters_.segment_valid_line_num =
        declare_parameter<int>("segment_valid_line_num", 3);
    segmentation_parameters_.ground_weight =
        static_cast<float>(declare_parameter<double>("ground_weight", 0.5));
    segmentation_parameters_.nonground_weight =
        static_cast<float>(declare_parameter<double>("nonground_weight", 1.0));
    const double neutral_weight = declare_parameter<double>("neutral_weight", -1.0);
    segmentation_parameters_.neutral_weight = static_cast<float>(
        neutral_weight >= 0.0
            ? neutral_weight
            : 0.5 * (segmentation_parameters_.ground_weight +
                     segmentation_parameters_.nonground_weight));

    segmenter_ = std::make_unique<segmentation_aware_ndt::LegoLoamSegmentation>(
        segmentation_parameters_);

    if (semantic_class_threshold_ <= segmentation_parameters_.ground_weight ||
        semantic_class_threshold_ >= segmentation_parameters_.nonground_weight) {
      throw std::invalid_argument(
          "semantic_class_threshold must lie between ground_weight and nonground_weight");
    }
    if (semantic_mismatch_weight_ < 0.0 || semantic_mismatch_weight_ > 1.0) {
      throw std::invalid_argument("semantic_mismatch_weight must be in [0, 1]");
    }
    if (dimension_scale_linear_ <= 0.0 || dimension_scale_planar_ <= 0.0 ||
        dimension_scale_volumetric_ <= 0.0) {
      throw std::invalid_argument("PCA dimension scales must be positive");
    }
    if (!use_source_segmentation_ && use_source_semantic_weight_) {
      throw std::invalid_argument(
          "source semantic weighting requires source segmentation");
    }
    if (!(ground_keep_ratio_ > 0.0 && ground_keep_ratio_ <= 1.0)) {
      throw std::invalid_argument("ground_keep_ratio must be in (0, 1]");
    }
    if (!use_source_segmentation_ && use_source_ground_downsampling_) {
      throw std::invalid_argument(
          "source ground downsampling requires source segmentation");
    }
    if (!use_source_segmentation_ && recovery_use_source_ground_downsampling_) {
      throw std::invalid_argument(
          "recovery source ground downsampling requires source segmentation");
    }
  }

  void loadMap() {
    pcl::PCLPointCloud2 map_blob;
    if (map_path_.empty() || pcl::io::loadPCDFile(map_path_, map_blob) != 0) {
      throw std::runtime_error("failed to load map_pcd: " + map_path_);
    }

    bool has_intensity = false;
    for (const auto& field : map_blob.fields) {
      has_intensity = has_intensity || field.name == "intensity";
    }
    if (require_weighted_map_ && !has_intensity) {
      throw std::runtime_error(
          "map has no semantic weight in its intensity field: " + map_path_);
    }
    pcl::fromPCLPointCloud2(map_blob, *map_cloud_);
    if (map_cloud_->empty()) {
      throw std::runtime_error("semantic map is empty: " + map_path_);
    }

    // 임의의 PointXYZI map이 자동으로 semantic map이 되는 것은 아니다.
    // intensity가 semantic_map_builder의 encoding 범위를 벗어나는 기존
    // distance-weighted map이나 reflectivity map은 거부한다.
    float minimum_weight = std::numeric_limits<float>::infinity();
    float maximum_weight = -std::numeric_limits<float>::infinity();
    bool has_ground_class = false;
    bool has_nonground_class = false;
    for (const auto& point : map_cloud_->points) {
      if (!std::isfinite(point.intensity)) {
        throw std::runtime_error("semantic map contains a non-finite weight: " + map_path_);
      }
      minimum_weight = std::min(minimum_weight, point.intensity);
      maximum_weight = std::max(maximum_weight, point.intensity);
      has_ground_class =
          has_ground_class || point.intensity <= semantic_class_threshold_;
      has_nonground_class =
          has_nonground_class || point.intensity > semantic_class_threshold_;
    }
    constexpr float kWeightTolerance = 1.0e-3F;
    if (require_weighted_map_ &&
        (minimum_weight < segmentation_parameters_.ground_weight - kWeightTolerance ||
         maximum_weight > segmentation_parameters_.nonground_weight + kWeightTolerance)) {
      throw std::runtime_error(
          "map intensity is outside the semantic weight range [" +
          std::to_string(segmentation_parameters_.ground_weight) + ", " +
          std::to_string(segmentation_parameters_.nonground_weight) + "]: " + map_path_);
    }
    if (require_weighted_map_ && (!has_ground_class || !has_nonground_class)) {
      throw std::runtime_error(
          "map does not contain both ground and non-ground semantic classes: " + map_path_);
    }

    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setLeafSize(
        map_downsample_resolution_, map_downsample_resolution_, map_downsample_resolution_);
    voxel_grid.setInputCloud(map_cloud_);
    auto downsampled = Cloud::Ptr(new Cloud());
    voxel_grid.filter(*downsampled);
    map_cloud_ = downsampled;

    RCLCPP_INFO(
        get_logger(), "loaded semantic map '%s' (%zu points, weight range %.3f..%.3f)",
        map_path_.c_str(), map_cloud_->size(), minimum_weight, maximum_weight);
  }

  void loadPrecomputedNdtMap() {
    pcl::PCLPointCloud2 map_blob;
    if (ndt_map_path_.empty() || pcl::io::loadPCDFile(ndt_map_path_, map_blob) != 0) {
      throw std::runtime_error("failed to load weighted NDT voxel map: " + ndt_map_path_);
    }

    const std::vector<std::string> required_fields{
        "x", "y", "z", "cov_xx", "cov_xy", "cov_xz", "cov_yy", "cov_yz",
        "cov_zz", "semantic_weight", "ndt_weight", "resolution",
        "voxel_x", "voxel_y", "voxel_z", "dimension_label", "point_count",
        "format_version"};
    for (const auto& required : required_fields) {
      bool found = false;
      for (const auto& field : map_blob.fields) {
        found = found || field.name == required;
      }
      if (!found) {
        throw std::runtime_error(
            "weighted NDT voxel map is missing field '" + required + "': " +
            ndt_map_path_);
      }
    }

    pcl::PointCloud<WeightedNdtVoxelPoint> voxel_cloud;
    pcl::fromPCLPointCloud2(map_blob, voxel_cloud);
    if (voxel_cloud.empty()) {
      throw std::runtime_error("weighted NDT voxel map is empty: " + ndt_map_path_);
    }

    const float stored_resolution = voxel_cloud.front().resolution;
    if (!(stored_resolution > 0.0F) || !std::isfinite(stored_resolution)) {
      throw std::runtime_error("weighted NDT voxel map has an invalid resolution");
    }
    if (std::abs(stored_resolution - ndt_resolution_) > 1.0e-5F) {
      throw std::runtime_error(
          "NDT resolution mismatch: file=" + std::to_string(stored_resolution) +
          ", parameter=" + std::to_string(ndt_resolution_));
    }

    float minimum_weight = std::numeric_limits<float>::infinity();
    float maximum_weight = -std::numeric_limits<float>::infinity();
    constexpr float kWeightTolerance = 1.0e-3F;
    const bool validate_semantic_weights =
        use_target_ndt_weight_ || use_source_semantic_weight_;
    const float minimum_allowed_semantic_weight =
        std::min(segmentation_parameters_.ground_weight,
                 segmentation_parameters_.neutral_weight);
    const float maximum_allowed_semantic_weight =
        std::max(segmentation_parameters_.nonground_weight,
                 segmentation_parameters_.neutral_weight);
    for (const auto& voxel : voxel_cloud.points) {
      if (std::abs(voxel.resolution - stored_resolution) > 1.0e-5F ||
          voxel.format_version != segmentation_aware_ndt::kOmpNdtVoxelFormatVersion ||
          !std::isfinite(voxel.x) || !std::isfinite(voxel.y) || !std::isfinite(voxel.z) ||
          !std::isfinite(voxel.semantic_weight) || !std::isfinite(voxel.ndt_weight) ||
          (validate_semantic_weights &&
           (voxel.semantic_weight <
                minimum_allowed_semantic_weight - kWeightTolerance ||
            voxel.semantic_weight >
                maximum_allowed_semantic_weight + kWeightTolerance)) ||
          (use_target_ndt_weight_ && voxel.ndt_weight <= 0.0F) ||
          voxel.dimension_label < 1U ||
          voxel.dimension_label > 3U || voxel.point_count < static_cast<std::uint32_t>(ndt_min_points_)) {
        throw std::runtime_error("weighted NDT voxel map contains inconsistent fields");
      }
      minimum_weight = std::min(minimum_weight, voxel.semantic_weight);
      maximum_weight = std::max(maximum_weight, voxel.semantic_weight);
    }

    precomputed_ndt_leaves_ =
        segmentation_aware_ndt::decodeWeightedNdtVoxelCloud(voxel_cloud);
    RCLCPP_INFO(
        get_logger(),
        "loaded precomputed weighted NDT map '%s' (%zu voxels, resolution %.3f m, semantic %.3f..%.3f)",
        ndt_map_path_.c_str(), precomputed_ndt_leaves_.size(), stored_resolution,
        minimum_weight, maximum_weight);
    RCLCPP_INFO(
        get_logger(), "PCA dimension scales: linear=%.3f planar=%.3f volumetric=%.3f",
        dimension_scale_linear_, dimension_scale_planar_, dimension_scale_volumetric_);
  }

  void initializeRegistration() {
    reg_.setResolution(ndt_resolution_);
    reg_.setStepSize(ndt_step_size_);
    reg_.setNumThreads(ndt_num_threads_);
    reg_.setNeighborhoodSearchMethod(neighborSearchMethod(ndt_neighbor_search_));
    reg_.setTransformationEpsilon(transformation_epsilon_);
    reg_.setMaximumIterations(max_iterations_);
    reg_.setTargetNdtWeight(use_target_ndt_weight_);
    reg_.setSourceSemanticWeight(use_source_semantic_weight_);
    reg_.setSemanticClassThreshold(semantic_class_threshold_);
    reg_.setSemanticMismatchWeight(semantic_mismatch_weight_);
    reg_.setContinuousSemanticCompatibility(use_continuous_semantic_compatibility_);
    reg_.setSemanticWeightRange(
        segmentation_parameters_.nonground_weight - segmentation_parameters_.ground_weight);
    reg_.setDimensionScales(
        dimension_scale_linear_, dimension_scale_planar_,
        dimension_scale_volumetric_);
    if (use_precomputed_ndt_map_) {
      reg_.setPrecomputedTarget(precomputed_ndt_leaves_, ndt_resolution_);
    } else {
      reg_.setInputTarget(map_cloud_);
    }

    RCLCPP_INFO(
        get_logger(),
        "NDT weights: target=%s source_semantic=%s compatibility=%s class_threshold=%.3f mismatch=%.3f distance=%s",
        use_target_ndt_weight_ ? "on" : "off",
        use_source_semantic_weight_ ? "on" : "off",
        use_continuous_semantic_compatibility_ ? "continuous" : "hard",
        semantic_class_threshold_, semantic_mismatch_weight_,
        use_runtime_distance_weight_ ? "on" : "off");
    if (use_adaptive_recovery_) {
      RCLCPP_INFO(
          get_logger(),
          "adaptive recovery: after=%d target=%s source_semantic=%s ground_downsample=%s ground_keep=%.3f mismatch=%.3f min_prob=%.3f",
          adaptive_recovery_after_rejections_,
          recovery_use_target_ndt_weight_ ? "on" : "off",
          recovery_use_source_semantic_weight_ ? "on" : "off",
          recovery_use_source_ground_downsampling_ ? "on" : "off",
          recovery_ground_keep_ratio_, recovery_semantic_mismatch_weight_,
          recovery_min_transformation_probability_);
    }
  }

  Eigen::Matrix4f composeInitialPose() const {
    const Eigen::Affine3f pose =
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
    if (!use_constant_velocity_prediction_ || accepted_pose_history_.size() < 2 ||
        consecutive_rejections_ >= reinitialize_after_rejections_) {
      return current_pose_;
    }

    const auto& previous = accepted_pose_history_[0];
    const auto& latest = accepted_pose_history_[1];
    const double history_dt = static_cast<double>(latest.stamp_ns - previous.stamp_ns) * 1.0e-9;
    double prediction_dt = static_cast<double>(stamp.nanoseconds() - latest.stamp_ns) * 1.0e-9;
    if (!(history_dt > 0.0) || !(prediction_dt > 0.0)) {
      return current_pose_;
    }

    // 긴 reject 구간에서 검증되지 않은 속도를 무한히 외삽하지 않도록
    // 경과 시간과 상대 scale을 모두 제한한다.
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

  static double yawFromPose(const Eigen::Matrix4f& pose) {
    return std::atan2(static_cast<double>(pose(1, 0)), static_cast<double>(pose(0, 0)));
  }

  static double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
  }

  bool violatesCorrectionLimit(
      double prediction_distance,
      double correction_distance,
      double* allowed_correction) const {
    const double allowed = std::max(
        max_correction_distance_m_,
        correction_prediction_ratio_ * std::max(0.0, prediction_distance));
    if (allowed_correction != nullptr) {
      *allowed_correction = allowed;
    }
    return use_correction_limit_ && correction_distance > allowed;
  }

  bool violatesVelocityConsistency(
      const Eigen::Matrix4f& candidate_pose,
      const rclcpp::Time& stamp,
      double* speed_mps,
      double* yaw_rate_dps) const {
    if (!use_velocity_consistency_check_ || accepted_pose_history_.empty()) {
      return false;
    }

    const auto& latest = accepted_pose_history_.back();
    const double dt = static_cast<double>(stamp.nanoseconds() - latest.stamp_ns) * 1.0e-9;
    if (!(dt > 0.0)) {
      return false;
    }

    const double distance =
        (candidate_pose.block<3, 1>(0, 3) - latest.pose.block<3, 1>(0, 3)).norm();
    const double speed = distance / dt;
    const double yaw_rate =
        std::abs(normalizeAngle(yawFromPose(candidate_pose) - yawFromPose(latest.pose))) /
        dt * 180.0 / M_PI;
    if (speed_mps != nullptr) {
      *speed_mps = speed;
    }
    if (yaw_rate_dps != nullptr) {
      *yaw_rate_dps = yaw_rate;
    }
    return speed > max_accepted_speed_mps_ || yaw_rate > max_accepted_yaw_rate_dps_;
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

  void updateAdaptiveRecoveryMode() {
    if (use_adaptive_recovery_ && !recovery_mode_latched_ &&
        consecutive_rejections_ >= adaptive_recovery_after_rejections_) {
      recovery_mode_latched_ = true;
      recovery_accepted_count_ = 0;
    }
    effective_recovery_mode_ = use_adaptive_recovery_ && recovery_mode_latched_;

    effective_use_target_ndt_weight_ = effective_recovery_mode_
        ? recovery_use_target_ndt_weight_ : use_target_ndt_weight_;
    effective_use_source_semantic_weight_ = effective_recovery_mode_
        ? recovery_use_source_semantic_weight_ : use_source_semantic_weight_;
    effective_use_source_ground_downsampling_ = effective_recovery_mode_
        ? recovery_use_source_ground_downsampling_ : use_source_ground_downsampling_;
    effective_ground_keep_ratio_ = effective_recovery_mode_
        ? recovery_ground_keep_ratio_ : ground_keep_ratio_;
    effective_semantic_mismatch_weight_ = effective_recovery_mode_
        ? recovery_semantic_mismatch_weight_ : semantic_mismatch_weight_;
    effective_min_transformation_probability_ =
        (effective_recovery_mode_ && recovery_min_transformation_probability_ >= 0.0)
            ? recovery_min_transformation_probability_
            : min_transformation_probability_;

    reg_.setTargetNdtWeight(effective_use_target_ndt_weight_);
    reg_.setSourceSemanticWeight(effective_use_source_semantic_weight_);
    reg_.setSemanticMismatchWeight(effective_semantic_mismatch_weight_);

    if (effective_recovery_mode_ && !was_recovery_mode_) {
      RCLCPP_WARN(
          get_logger(),
          "adaptive recovery enabled after %d consecutive rejections: target=%s source_semantic=%s ground_keep=%.3f mismatch=%.3f min_prob=%.3f",
          consecutive_rejections_,
          effective_use_target_ndt_weight_ ? "on" : "off",
          effective_use_source_semantic_weight_ ? "on" : "off",
          effective_ground_keep_ratio_, effective_semantic_mismatch_weight_,
          effective_min_transformation_probability_);
    } else if (!effective_recovery_mode_ && was_recovery_mode_) {
      RCLCPP_INFO(get_logger(), "adaptive recovery disabled after accepted registration");
    }
    was_recovery_mode_ = effective_recovery_mode_;
  }

  void noteAcceptedRegistrationInRecovery(double probability) {
    if (!recovery_mode_latched_) {
      return;
    }
    ++recovery_accepted_count_;
    if (recovery_accepted_count_ >= adaptive_recovery_exit_accepts_ &&
        probability >= min_transformation_probability_) {
      recovery_mode_latched_ = false;
      recovery_accepted_count_ = 0;
    }
  }

  void noteRejectedRegistrationInRecovery() {
    if (recovery_mode_latched_) {
      recovery_accepted_count_ = 0;
    }
  }

  Cloud::Ptr downsampleSourceGround(
      const Cloud::Ptr& input,
      segmentation_aware_ndt::SegmentationStatistics& statistics) const {
    statistics.ground_points_kept = statistics.ground_points;
    statistics.ground_points_dropped = 0;
    if (!effective_use_source_ground_downsampling_ || effective_ground_keep_ratio_ >= 1.0) {
      return input;
    }

    auto output = Cloud::Ptr(new Cloud());
    output->reserve(input->size());
    double accumulator = 0.0;
    std::size_t kept = 0;
    for (const auto& point : input->points) {
      const bool is_ground = point.intensity <= semantic_class_threshold_;
      if (!is_ground) {
        output->push_back(point);
        continue;
      }

      accumulator += effective_ground_keep_ratio_;
      if (accumulator + 1.0e-12 >= 1.0) {
        output->push_back(point);
        accumulator -= 1.0;
        ++kept;
      }
    }
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = input->is_dense;
    statistics.ground_points_kept = kept;
    statistics.ground_points_dropped = statistics.ground_points - kept;
    return output;
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    const auto processing_start = std::chrono::steady_clock::now();
    updateAdaptiveRecoveryMode();
    auto raw_lidar = Cloud::Ptr(new Cloud());
    pcl::fromROSMsg(*cloud_msg, *raw_lidar);
    if (raw_lidar->empty()) {
      publishDiagnostics(
          cloud_msg->header.stamp, false, false, 0,
          std::numeric_limits<double>::quiet_NaN(), {}, 0.0, 0.0,
          elapsedMilliseconds(processing_start), "empty_input");
      return;
    }

    // range-image segmentation은 lidar 원점을 사용해야 한다. 2.4 m extrinsic을
    // 먼저 적용하면 모든 수직각이 달라져 ring ID가 잘못된다.
    segmentation_aware_ndt::SegmentationStatistics statistics;
    Cloud::Ptr segmented_lidar;
    if (use_source_segmentation_) {
      segmented_lidar = preserve_unlabeled_points_
          ? segmenter_->labelPreservingGeometry(*raw_lidar, &statistics)
          : segmenter_->segment(*raw_lidar, &statistics);
      segmented_lidar = downsampleSourceGround(segmented_lidar, statistics);
    } else {
      segmented_lidar = Cloud::Ptr(new Cloud(*raw_lidar));
      statistics.input_points = raw_lidar->size();
      statistics.projected_points = raw_lidar->size();
      statistics.nonground_points = raw_lidar->size();
    }
    if (segmented_lidar->empty()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "source segmentation rejected all points (input=%zu projected=%zu)",
          statistics.input_points, statistics.projected_points);
      publishDiagnostics(
          cloud_msg->header.stamp, false, false, 0,
          std::numeric_limits<double>::quiet_NaN(), statistics, 0.0, 0.0,
          elapsedMilliseconds(processing_start), "segmentation_empty");
      return;
    }

    // /tf_static에서 base_link <- lidar를 명시적으로 가져온다. PCL/NDT는 ROS
    // TF tree를 직접 조회하지 않으므로 TF를 발행하는 것만으로는 부족하다.
    Eigen::Matrix4f base_from_sensor = Eigen::Matrix4f::Identity();
    if (cloud_msg->header.frame_id != base_frame_id_) {
      try {
        base_from_sensor = transformMatrix(tf_buffer_.lookupTransform(
            base_frame_id_, cloud_msg->header.frame_id, cloud_msg->header.stamp,
            tf2::durationFromSec(transform_timeout_sec_)));
      } catch (const tf2::TransformException& exception) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000, "failed to lookup %s <- %s: %s",
            base_frame_id_.c_str(), cloud_msg->header.frame_id.c_str(), exception.what());
        publishDiagnostics(
            cloud_msg->header.stamp, false, false, 0,
            std::numeric_limits<double>::quiet_NaN(), statistics, 0.0, 0.0,
            elapsedMilliseconds(processing_start), "transform_unavailable");
        return;
      }
    }

    auto segmented_base = Cloud::Ptr(new Cloud());
    pcl::transformPointCloud(*segmented_lidar, *segmented_base, base_from_sensor);

    // VoxelGrid는 intensity를 평균내므로 ground/non-ground 경계에서 semantic
    // confidence를 버리지 않고 연속값으로 보존한다.
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setLeafSize(
        source_downsample_resolution_, source_downsample_resolution_,
        source_downsample_resolution_);
    voxel_grid.setInputCloud(segmented_base);
    auto ndt_input = Cloud::Ptr(new Cloud());
    voxel_grid.filter(*ndt_input);
    if (ndt_input->empty()) {
      publishDiagnostics(
          cloud_msg->header.stamp, false, false, 0,
          std::numeric_limits<double>::quiet_NaN(), statistics, 0.0, 0.0,
          elapsedMilliseconds(processing_start), "downsample_empty");
      return;
    }
    publishSegmented(*ndt_input, cloud_msg->header.stamp);

    // source가 base_link 좌표로 표현되므로 current_pose_와 최종 transformation은
    // 모두 완전한 6DOF map <- base_link를 나타낸다.
    const Eigen::Matrix4f predicted_pose = predictPose(cloud_msg->header.stamp);
    const double prediction_distance =
        (predicted_pose.block<3, 1>(0, 3) - current_pose_.block<3, 1>(0, 3)).norm();
    reg_.setInputSource(ndt_input);
    reg_.align(*aligned_cloud_, predicted_pose);
    const double probability = reg_.getTransformationProbability();
    const int iterations = reg_.getFinalNumIteration();
    const bool converged = reg_.hasConverged();
    const Eigen::Matrix4f candidate_pose = reg_.getFinalTransformation();
    const double correction_distance =
        (candidate_pose.block<3, 1>(0, 3) - predicted_pose.block<3, 1>(0, 3)).norm();
    double allowed_correction = 0.0;
    const bool correction_too_large = violatesCorrectionLimit(
        prediction_distance, correction_distance, &allowed_correction);
    double estimated_speed_mps = 0.0;
    double estimated_yaw_rate_dps = 0.0;
    const bool velocity_inconsistent = violatesVelocityConsistency(
        candidate_pose, cloud_msg->header.stamp,
        &estimated_speed_mps, &estimated_yaw_rate_dps);
    // reg_ 자체가 max_iterations에서 최적화를 종료한다. 구현 내부 counter는 설정값
    // 64에 대해 66을 반환할 수 있으므로 이 값을 다시 reject 조건으로 사용하면
    // 수렴했고 score도 높은 결과까지 버려 연속 prediction 실패를 유발한다.
    const bool accepted = converged && std::isfinite(probability) &&
        probability >= effective_min_transformation_probability_ &&
        !correction_too_large && !velocity_inconsistent;
    if (!accepted) {
      std::string reason = "registration_rejected";
      if (!converged) {
        reason = "not_converged";
      } else if (!std::isfinite(probability)) {
        reason = "invalid_probability";
      } else if (probability < effective_min_transformation_probability_) {
        reason = "low_probability";
      } else if (correction_too_large) {
        reason = "large_correction";
      } else if (velocity_inconsistent) {
        reason = "velocity_inconsistent";
      }
      publishDiagnostics(
          cloud_msg->header.stamp, false, converged, iterations, probability,
          statistics, prediction_distance, correction_distance,
          elapsedMilliseconds(processing_start), reason);
      ++consecutive_rejections_;
      noteRejectedRegistrationInRecovery();
      if (consecutive_rejections_ == reinitialize_after_rejections_ &&
          !accepted_pose_history_.empty()) {
        accepted_pose_history_.clear();
        RCLCPP_WARN(
            get_logger(),
            "reinitialized prediction from last accepted pose after %d consecutive rejections",
            reinitialize_after_rejections_);
      } else if (consecutive_rejections_ == max_consecutive_rejections_ &&
          !accepted_pose_history_.empty()) {
        accepted_pose_history_.clear();
        RCLCPP_WARN(
            get_logger(),
            "cleared constant-velocity history after %d consecutive rejections",
            max_consecutive_rejections_);
      }
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "semantic NDT rejected: reason=%s converged=%d iterations=%d probability=%.6f prediction=%.3f m correction=%.3f/%.3f m speed=%.3f m/s yaw_rate=%.3f deg/s",
          reason.c_str(), converged, iterations, probability, prediction_distance,
          correction_distance, allowed_correction, estimated_speed_mps,
          estimated_yaw_rate_dps);
      return;
    }

    current_pose_ = candidate_pose;
    noteAcceptedRegistrationInRecovery(probability);
    consecutive_rejections_ = 0;
    updateAdaptiveRecoveryMode();
    rememberAcceptedPose(current_pose_, cloud_msg->header.stamp);
    publishPose(cloud_msg->header.stamp);
    publishAligned(cloud_msg->header.stamp);
    publishDiagnostics(
        cloud_msg->header.stamp, true, converged, iterations, probability,
        statistics, prediction_distance, correction_distance,
        elapsedMilliseconds(processing_start), "accepted");
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "semantic NDT accepted: ground=%zu kept=%zu nonground=%zu rejected=%zu iterations=%d probability=%.6f prediction=%.3f m correction=%.3f m",
        statistics.ground_points, statistics.ground_points_kept, statistics.nonground_points,
        statistics.rejected_points, iterations, probability,
        prediction_distance, correction_distance);
  }

  static double elapsedMilliseconds(
      const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
  }

  void publishDiagnostics(
      const rclcpp::Time& stamp,
      bool accepted,
      bool converged,
      int iterations,
      double probability,
      const segmentation_aware_ndt::SegmentationStatistics& statistics,
      double prediction_distance,
      double correction_distance,
      double processing_time_ms,
      const std::string& reason) {
    diagnostic_msgs::msg::DiagnosticArray message;
    message.header.stamp = stamp;
    message.header.frame_id = map_frame_id_;

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.level = accepted
        ? diagnostic_msgs::msg::DiagnosticStatus::OK
        : diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.name = "semantic_ndt_registration";
    status.hardware_id = "weighted_ndt";
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
    add_value("input_points", std::to_string(statistics.input_points));
    add_value("projected_points", std::to_string(statistics.projected_points));
    add_value("ground_points", std::to_string(statistics.ground_points));
    add_value("ground_points_kept", std::to_string(statistics.ground_points_kept));
    add_value("ground_points_dropped", std::to_string(statistics.ground_points_dropped));
    add_value("nonground_points", std::to_string(statistics.nonground_points));
    add_value("rejected_points", std::to_string(statistics.rejected_points));
    add_value("prediction_distance_m", std::to_string(prediction_distance));
    add_value("correction_distance_m", std::to_string(correction_distance));
    add_value("processing_time_ms", std::to_string(processing_time_ms));
    add_value("localizer_type", "weighted_ndt");
    add_value("map_path", map_path_);
    add_value("ndt_map_path", ndt_map_path_);
    add_value("use_precomputed_ndt_map", use_precomputed_ndt_map_ ? "true" : "false");
    add_value("adaptive_recovery_mode", effective_recovery_mode_ ? "true" : "false");
    add_value(
        "use_target_ndt_weight",
        effective_use_target_ndt_weight_ ? "true" : "false");
    add_value(
        "use_source_semantic_weight",
        effective_use_source_semantic_weight_ ? "true" : "false");
    add_value("use_source_segmentation", use_source_segmentation_ ? "true" : "false");
    add_value(
        "use_source_ground_downsampling",
        effective_use_source_ground_downsampling_ ? "true" : "false");
    add_value("ground_keep_ratio", std::to_string(effective_ground_keep_ratio_));
    add_value("ground_weight", std::to_string(segmentation_parameters_.ground_weight));
    add_value("nonground_weight", std::to_string(segmentation_parameters_.nonground_weight));
    add_value("semantic_class_threshold", std::to_string(semantic_class_threshold_));
    add_value("semantic_mismatch_weight", std::to_string(effective_semantic_mismatch_weight_));
    add_value(
        "min_transformation_probability",
        std::to_string(effective_min_transformation_probability_));
    add_value(
        "continuous_semantic_compatibility",
        use_continuous_semantic_compatibility_ ? "true" : "false");
    add_value("dimension_scale_linear", std::to_string(dimension_scale_linear_));
    add_value("dimension_scale_planar", std::to_string(dimension_scale_planar_));
    add_value("dimension_scale_volumetric", std::to_string(dimension_scale_volumetric_));
    add_value("ndt_resolution", std::to_string(ndt_resolution_));
    add_value("ndt_neighbor_search", ndt_neighbor_search_);
    add_value("source_downsample_resolution", std::to_string(source_downsample_resolution_));
    message.status.push_back(status);
    diagnostics_pub_->publish(message);
  }

  static Eigen::Matrix4f transformMatrix(
      const geometry_msgs::msg::TransformStamped& transform_message) {
    const auto& translation = transform_message.transform.translation;
    const auto& rotation = transform_message.transform.rotation;
    Eigen::Quaternionf quaternion(
        static_cast<float>(rotation.w), static_cast<float>(rotation.x),
        static_cast<float>(rotation.y), static_cast<float>(rotation.z));
    quaternion.normalize();

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
    transform(0, 3) = static_cast<float>(translation.x);
    transform(1, 3) = static_cast<float>(translation.y);
    transform(2, 3) = static_cast<float>(translation.z);
    return transform;
  }

  geometry_msgs::msg::TransformStamped poseTransform(const rclcpp::Time& stamp) const {
    Eigen::Quaternionf quaternion(current_pose_.block<3, 3>(0, 0));
    quaternion.normalize();

    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = map_frame_id_;
    transform.child_frame_id = base_frame_id_;
    transform.transform.translation.x = current_pose_(0, 3);
    transform.transform.translation.y = current_pose_(1, 3);
    transform.transform.translation.z = current_pose_(2, 3);
    transform.transform.rotation.x = quaternion.x();
    transform.transform.rotation.y = quaternion.y();
    transform.transform.rotation.z = quaternion.z();
    transform.transform.rotation.w = quaternion.w();
    return transform;
  }

  void publishPose(const rclcpp::Time& stamp) {
    const auto transform = poseTransform(stamp);
    tf_broadcaster_->sendTransform(transform);

    nav_msgs::msg::Odometry odometry;
    odometry.header = transform.header;
    odometry.child_frame_id = base_frame_id_;
    odometry.pose.pose.position.x = current_pose_(0, 3);
    odometry.pose.pose.position.y = current_pose_(1, 3);
    odometry.pose.pose.position.z = current_pose_(2, 3);
    odometry.pose.pose.orientation = transform.transform.rotation;
    odom_pub_->publish(odometry);

    geometry_msgs::msg::PoseStamped pose;
    pose.header = odometry.header;
    pose.pose = odometry.pose.pose;
    pose_pub_->publish(pose);
  }

  void publishSegmented(const Cloud& cloud, const rclcpp::Time& stamp) {
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(cloud, message);
    message.header.frame_id = base_frame_id_;
    message.header.stamp = stamp;
    segmented_pub_->publish(message);
  }

  void publishAligned(const rclcpp::Time& stamp) {
    sensor_msgs::msg::PointCloud2 message;
    pcl::toROSMsg(*aligned_cloud_, message);
    message.header.frame_id = map_frame_id_;
    message.header.stamp = stamp;
    aligned_pub_->publish(message);
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  Cloud::Ptr map_cloud_;
  Cloud::Ptr aligned_cloud_;
  pclomp::NormalDistributionsTransform<PointT, PointT> reg_;
  Eigen::Matrix4f current_pose_{Eigen::Matrix4f::Identity()};
  segmentation_aware_ndt::SegmentationParameters segmentation_parameters_;
  std::unique_ptr<segmentation_aware_ndt::LegoLoamSegmentation> segmenter_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;

  std::string input_topic_;
  bool reliable_input_qos_{false};
  std::string map_path_;
  std::string ndt_map_path_;
  std::string map_frame_id_;
  std::string base_frame_id_;
  double transform_timeout_sec_{0.2};
  float map_downsample_resolution_{0.2F};
  float source_downsample_resolution_{0.2F};
  float ndt_resolution_{1.0F};
  int ndt_min_points_{6};
  double ndt_step_size_{0.1};
  int ndt_num_threads_{4};
  std::string ndt_neighbor_search_{"DIRECT1"};
  double transformation_epsilon_{0.01};
  int max_iterations_{64};
  double min_transformation_probability_{0.2};
  bool require_weighted_map_{true};
  bool use_constant_velocity_prediction_{true};
  double max_prediction_horizon_sec_{0.5};
  double max_prediction_scale_{3.0};
  double max_prediction_distance_m_{1.5};
  int max_consecutive_rejections_{3};
  int reinitialize_after_rejections_{3};
  bool use_correction_limit_{true};
  double max_correction_distance_m_{1.0};
  double correction_prediction_ratio_{0.2};
  bool use_velocity_consistency_check_{true};
  double max_accepted_speed_mps_{40.0};
  double max_accepted_yaw_rate_dps_{120.0};
  bool use_adaptive_recovery_{false};
  int adaptive_recovery_after_rejections_{3};
  int adaptive_recovery_exit_accepts_{10};
  bool recovery_use_target_ndt_weight_{false};
  bool recovery_use_source_semantic_weight_{false};
  bool recovery_use_source_ground_downsampling_{false};
  double recovery_ground_keep_ratio_{1.0};
  double recovery_semantic_mismatch_weight_{1.0};
  double recovery_min_transformation_probability_{-1.0};
  bool use_precomputed_ndt_map_{true};
  bool use_source_semantic_weight_{true};
  bool use_source_segmentation_{true};
  bool preserve_unlabeled_points_{false};
  bool use_source_ground_downsampling_{true};
  double ground_keep_ratio_{0.5};
  bool use_target_ndt_weight_{true};
  double semantic_class_threshold_{0.75};
  double semantic_mismatch_weight_{0.1};
  bool use_continuous_semantic_compatibility_{true};
  double dimension_scale_linear_{0.75};
  double dimension_scale_planar_{1.25};
  double dimension_scale_volumetric_{1.0};
  bool use_runtime_distance_weight_{false};
  bool effective_recovery_mode_{false};
  bool was_recovery_mode_{false};
  bool recovery_mode_latched_{false};
  int recovery_accepted_count_{0};
  bool effective_use_target_ndt_weight_{true};
  bool effective_use_source_semantic_weight_{true};
  bool effective_use_source_ground_downsampling_{true};
  double effective_ground_keep_ratio_{0.5};
  double effective_semantic_mismatch_weight_{0.1};
  double effective_min_transformation_probability_{0.2};
  float initial_x_{0.0F};
  float initial_y_{0.0F};
  float initial_z_{0.0F};
  float initial_roll_{0.0F};
  float initial_pitch_{0.0F};
  float initial_yaw_{0.0F};
  std::deque<AcceptedPose> accepted_pose_history_;
  int consecutive_rejections_{0};
  std::vector<segmentation_aware_ndt::PrecomputedNdtLeaf> precomputed_ndt_leaves_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SemanticNdtLocalizer>());
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("semantic_ndt_localizer"), "%s", exception.what());
  }
  rclcpp::shutdown();
  return 0;
}
