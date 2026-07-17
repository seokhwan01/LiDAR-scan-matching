#include <chrono>
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
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <segmentation_aware_ndt/lego_loam_segmentation.hpp>
#include <segmentation_aware_ndt/weighted_ndt_voxel.hpp>

namespace {

// segmentation-aware NDT에서 사용할 고정 target map을 생성한다.
//
// Mapping에는 CARLA ground-truth map -> base_link 변환과 정적 base_link -> lidar
// 변환이 필요하다. 이 node는 raw LiDAR scan마다 다음 작업을 수행한다.
//   1. range-image 기하를 보존하기 위해 원본 lidar frame에서 segmentation 수행
//   2. ground/non-ground confidence를 PointXYZI::intensity에 저장
//   3. cloud timestamp에서 map <- lidar 변환 조회
//   4. 변환한 point를 pre-built PCD map에 누적
class SemanticMapBuilder : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<PointT>;

  SemanticMapBuilder()
      : rclcpp::Node("semantic_map_builder"),
        tf_buffer_(get_clock()),
        tf_listener_(tf_buffer_),
        accumulated_map_(new Cloud()) {
    initializeParameters();

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&SemanticMapBuilder::cloudCallback, this, std::placeholders::_1));
    if (publish_map_) {
      map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          map_topic_, rclcpp::QoS(1).transient_local());

      publish_timer_ = create_wall_timer(
          std::chrono::duration<double>(publish_interval_),
          std::bind(&SemanticMapBuilder::publishMap, this));
    }
    save_timer_ = create_wall_timer(
        std::chrono::duration<double>(save_interval_),
        std::bind(&SemanticMapBuilder::saveMap, this));

    RCLCPP_INFO(
        get_logger(),
        "CARLA segmentation: %d rings x %d columns, dV=%.6f deg, dH=%.6f deg",
        segmentation_parameters_.n_scan, segmentation_parameters_.horizon_scan,
        segmentation_parameters_.ang_res_y, segmentation_parameters_.ang_res_x);
  }

  ~SemanticMapBuilder() override {
    saveMap();
  }

private:
  void initializeParameters() {
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/point_cloud");
    map_topic_ = declare_parameter<std::string>("map_topic", "/semantic_weighted_map");
    map_frame_id_ = declare_parameter<std::string>("map_frame_id", "map");
    map_path_ = declare_parameter<std::string>("map_path", "semantic_weighted_map.pcd");
    ndt_map_path_ =
        declare_parameter<std::string>("ndt_map_path", "weighted_ndt_voxels.pcd");
    transform_timeout_sec_ = declare_parameter<double>("transform_timeout_sec", 0.2);
    publish_interval_ = declare_parameter<double>("publish_interval", 5.0);
    publish_map_ = declare_parameter<bool>("publish_map", false);
    save_interval_ = declare_parameter<double>("save_interval", 30.0);
    map_leaf_size_ = declare_parameter<double>("map_leaf_size", 0.2);
    keyframe_delta_trans_ = declare_parameter<double>("keyframe_delta_trans", 1.0);
    keyframe_delta_angle_ = declare_parameter<double>("keyframe_delta_angle", 0.17);
    ndt_resolution_ = declare_parameter<double>("ndt_resolution", 1.0);
    ndt_min_points_ = declare_parameter<int>("ndt_min_points", 6);
    dimension_scale_linear_ = declare_parameter<double>("dimension_scale_linear", 0.75);
    dimension_scale_planar_ = declare_parameter<double>("dimension_scale_planar", 1.25);
    dimension_scale_volumetric_ =
        declare_parameter<double>("dimension_scale_volumetric", 1.0);
    use_map_ground_downsampling_ =
        declare_parameter<bool>("use_map_ground_downsampling", false);
    preserve_unlabeled_points_ =
        declare_parameter<bool>("preserve_unlabeled_points", false);
    ground_keep_ratio_ = declare_parameter<double>("ground_keep_ratio", 0.5);
    if (!(ground_keep_ratio_ > 0.0 && ground_keep_ratio_ <= 1.0)) {
      throw std::invalid_argument("ground_keep_ratio must be in (0, 1]");
    }

    // 기본값은 CARLA HDL-32 유사 publisher 설정과 정확히 일치한다.
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
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
    auto raw_cloud = Cloud::Ptr(new Cloud());
    pcl::fromROSMsg(*cloud_msg, *raw_cloud);
    if (raw_cloud->empty()) {
      return;
    }

    Eigen::Matrix4f map_from_lidar;
    try {
      map_from_lidar = transformMatrix(tf_buffer_.lookupTransform(
          map_frame_id_, cloud_msg->header.frame_id, cloud_msg->header.stamp,
          tf2::durationFromSec(transform_timeout_sec_)));
    } catch (const tf2::TransformException& exception) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000, "failed to lookup %s <- %s: %s",
          map_frame_id_.c_str(), cloud_msg->header.frame_id.c_str(), exception.what());
      return;
    }

    // bag에는 긴 초기 정지 구간을 포함한 10 Hz cloud stream이 들어 있다.
    // 모든 scan을 보관하면 이 경로에서 1억 개 이상의 중복 point가 추가된다.
    // 맵 범위는 유지하면서 CPU와 메모리를 제한하도록 segmentation 전에
    // 결정론적인 GT keyframe을 선택한다.
    if (!acceptKeyframe(map_from_lidar)) {
      return;
    }

    // 2.4 m lidar extrinsic을 적용하기 전에 segmentation을 수행해야 한다.
    // 그렇지 않으면 수직각이 lidar 원점이 아닌 base_link를 기준으로 계산되어
    // ring projection이 잘못된다.
    segmentation_aware_ndt::SegmentationStatistics statistics;
    auto segmented_lidar = preserve_unlabeled_points_
        ? segmenter_->labelPreservingGeometry(*raw_cloud, &statistics)
        : segmenter_->segment(*raw_cloud, &statistics);
    segmented_lidar = downsampleMapGround(segmented_lidar, statistics);
    if (segmented_lidar->empty()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "segmentation produced no map points (input=%zu, projected=%zu)",
          statistics.input_points, statistics.projected_points);
      return;
    }

    auto segmented_map = Cloud::Ptr(new Cloud());
    pcl::transformPointCloud(*segmented_lidar, *segmented_map, map_from_lidar);

    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      *accumulated_map_ += *segmented_map;
    }

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "segmented map: ground=%zu kept=%zu nonground=%zu rejected=%zu accumulated=%zu",
        statistics.ground_points, statistics.ground_points_kept,
        statistics.nonground_points,
        statistics.rejected_points, accumulated_map_->size());
  }

  Cloud::Ptr downsampleMapGround(
      const Cloud::Ptr& input,
      segmentation_aware_ndt::SegmentationStatistics& statistics) const {
    statistics.ground_points_kept = statistics.ground_points;
    statistics.ground_points_dropped = 0;
    if (!use_map_ground_downsampling_ || ground_keep_ratio_ >= 1.0) {
      return input;
    }

    auto output = Cloud::Ptr(new Cloud());
    output->reserve(input->size());
    double accumulator = 0.0;
    std::size_t kept = 0;
    for (const auto& point : input->points) {
      if (point.intensity > segmentation_parameters_.ground_weight) {
        output->push_back(point);
        continue;
      }
      accumulator += ground_keep_ratio_;
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

  bool acceptKeyframe(const Eigen::Matrix4f& current_pose) {
    if (!has_last_keyframe_) {
      last_keyframe_pose_ = current_pose;
      has_last_keyframe_ = true;
      return true;
    }

    const Eigen::Vector3f translation_delta =
        current_pose.block<3, 1>(0, 3) - last_keyframe_pose_.block<3, 1>(0, 3);
    const Eigen::Matrix3f rotation_delta =
        last_keyframe_pose_.block<3, 3>(0, 0).transpose() *
        current_pose.block<3, 3>(0, 0);
    const float rotation_angle = Eigen::AngleAxisf(rotation_delta).angle();

    if (translation_delta.norm() < keyframe_delta_trans_ &&
        std::abs(rotation_angle) < keyframe_delta_angle_) {
      return false;
    }

    last_keyframe_pose_ = current_pose;
    return true;
  }

  void publishMap() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (accumulated_map_->empty()) {
      return;
    }

    sensor_msgs::msg::PointCloud2 map_message;
    pcl::toROSMsg(*accumulated_map_, map_message);
    map_message.header.frame_id = map_frame_id_;
    map_message.header.stamp = now();
    map_pub_->publish(map_message);
  }

  void saveMap() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (accumulated_map_->empty()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "semantic map is empty; nothing to save");
      return;
    }

    auto downsampled = Cloud::Ptr(new Cloud());
    if (map_leaf_size_ > 0.0F) {
      // PCL은 voxel filtering 과정에서 intensity를 평균낸다. 여기서는 intensity가
      // semantic weight이므로 저장값은 map voxel에 기여한 모든 관측의 평균 class
      // confidence가 된다.
      pcl::VoxelGrid<PointT> voxel_grid;
      voxel_grid.setLeafSize(map_leaf_size_, map_leaf_size_, map_leaf_size_);
      voxel_grid.setInputCloud(accumulated_map_);
      voxel_grid.filter(*downsampled);
    } else {
      *downsampled = *accumulated_map_;
    }

    if (pcl::io::savePCDFileBinary(map_path_, *downsampled) != 0) {
      RCLCPP_ERROR(get_logger(), "failed to save semantic map: %s", map_path_.c_str());
      return;
    }
    RCLCPP_INFO(
        get_logger(), "saved semantic weighted map: %s (%zu points)",
        map_path_.c_str(), downsampled->size());

    pcl::PointCloud<WeightedNdtVoxelPoint> ndt_voxels;
    const auto voxel_count = segmentation_aware_ndt::buildWeightedNdtVoxelCloud(
        downsampled, ndt_resolution_, ndt_min_points_, ndt_voxels,
        dimension_scale_linear_, dimension_scale_planar_,
        dimension_scale_volumetric_);
    if (voxel_count == 0) {
      RCLCPP_ERROR(get_logger(), "weighted NDT voxel map is empty; not saving");
      return;
    }
    if (pcl::io::savePCDFileBinary(ndt_map_path_, ndt_voxels) != 0) {
      RCLCPP_ERROR(
          get_logger(), "failed to save weighted NDT voxel map: %s", ndt_map_path_.c_str());
      return;
    }
    RCLCPP_INFO(
        get_logger(), "saved weighted NDT voxel map: %s (%zu voxels, resolution %.3f m)",
        ndt_map_path_.c_str(), voxel_count, ndt_resolution_);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr save_timer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex map_mutex_;
  Cloud::Ptr accumulated_map_;
  segmentation_aware_ndt::SegmentationParameters segmentation_parameters_;
  std::unique_ptr<segmentation_aware_ndt::LegoLoamSegmentation> segmenter_;

  std::string cloud_topic_;
  std::string map_topic_;
  std::string map_frame_id_;
  std::string map_path_;
  std::string ndt_map_path_;
  double transform_timeout_sec_{0.2};
  double publish_interval_{5.0};
  bool publish_map_{false};
  double save_interval_{30.0};
  float map_leaf_size_{0.2F};
  float keyframe_delta_trans_{1.0F};
  float keyframe_delta_angle_{0.17F};
  float ndt_resolution_{1.0F};
  int ndt_min_points_{6};
  float dimension_scale_linear_{0.75F};
  float dimension_scale_planar_{1.25F};
  float dimension_scale_volumetric_{1.0F};
  bool use_map_ground_downsampling_{false};
  bool preserve_unlabeled_points_{false};
  double ground_keep_ratio_{0.5};
  bool has_last_keyframe_{false};
  Eigen::Matrix4f last_keyframe_pose_{Eigen::Matrix4f::Identity()};
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SemanticMapBuilder>());
  rclcpp::shutdown();
  return 0;
}
