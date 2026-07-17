#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace lidar_odometry {

class PrefilteringNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  PrefilteringNode()
      : rclcpp::Node("prefiltering_node"),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {
    initializeParams();

    // ROS2 입력 흐름: /point_cloud -> prefiltering_node -> /filtered_points
    points_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_, rclcpp::SensorDataQoS(),
        std::bind(&PrefilteringNode::cloudCallback, this, std::placeholders::_1));
    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, rclcpp::SensorDataQoS());
  }

private:
  void initializeParams() {
    input_topic_ = declare_parameter<std::string>("input_topic", "/point_cloud");
    output_topic_ = declare_parameter<std::string>("output_topic", "/filtered_points");

    // 다운샘플링은 계산량을 줄이는 핵심 단계다.
    const auto downsample_method = declare_parameter<std::string>("downsample_method", "VOXELGRID");
    const auto downsample_resolution = declare_parameter<double>("downsample_resolution", 0.2);

    if (downsample_method == "VOXELGRID") {
      auto voxelgrid = std::make_shared<pcl::VoxelGrid<PointT>>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter_ = voxelgrid;
    } else if (downsample_method == "APPROX_VOXELGRID") {
      auto approx_voxelgrid = std::make_shared<pcl::ApproximateVoxelGrid<PointT>>();
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter_ = approx_voxelgrid;
    } else if (downsample_method != "NONE") {
      RCLCPP_WARN(get_logger(), "unknown downsample_method '%s'; downsampling disabled", downsample_method.c_str());
    }

    // 이상치 제거는 sparse한 노이즈 포인트를 줄여 NDT 정합 안정성을 높인다.
    const auto outlier_removal_method = declare_parameter<std::string>("outlier_removal_method", "RADIUS");
    if (outlier_removal_method == "STATISTICAL") {
      const auto mean_k = declare_parameter<int>("statistical_mean_k", 20);
      const auto stddev_mul_thresh = declare_parameter<double>("statistical_stddev", 1.0);
      auto sor = std::make_shared<pcl::StatisticalOutlierRemoval<PointT>>();
      sor->setMeanK(mean_k);
      sor->setStddevMulThresh(stddev_mul_thresh);
      outlier_removal_filter_ = sor;
    } else if (outlier_removal_method == "RADIUS") {
      const auto radius = declare_parameter<double>("radius_radius", 0.5);
      const auto min_neighbors = declare_parameter<int>("radius_min_neighbors", 5);
      auto rad = std::make_shared<pcl::RadiusOutlierRemoval<PointT>>();
      rad->setRadiusSearch(radius);
      rad->setMinNeighborsInRadius(min_neighbors);
      outlier_removal_filter_ = rad;
    } else if (outlier_removal_method != "NONE") {
      RCLCPP_WARN(get_logger(), "unknown outlier_removal_method '%s'; outlier removal disabled",
                  outlier_removal_method.c_str());
    }

    // 거리 필터는 차량 주변 너무 가까운 점과 너무 먼 점을 제외한다.
    use_distance_filter_ = declare_parameter<bool>("use_distance_filter", true);
    distance_near_thresh_ = declare_parameter<double>("distance_near_thresh", 0.5);
    distance_far_thresh_ = declare_parameter<double>("distance_far_thresh", 100.0);

    // base_link_frame을 지정하면 입력 cloud를 해당 좌표계 기준으로 변환한다.
    base_link_frame_ = declare_parameter<std::string>("base_link_frame", "");
    use_angle_calibration_ = declare_parameter<bool>("use_angle_calibration", false);
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr input_msg) {
    sensor_msgs::msg::PointCloud2 cloud_msg = *input_msg;
    if (!base_link_frame_.empty() && cloud_msg.header.frame_id != base_link_frame_) {
      try {
        const auto original_stamp = cloud_msg.header.stamp;
        const auto transform = tf_buffer_.lookupTransform(
            base_link_frame_, cloud_msg.header.frame_id, cloud_msg.header.stamp,
            tf2::durationFromSec(0.2));
        tf2::doTransform(cloud_msg, cloud_msg, transform);
        // Static TF lookup results use time zero. doTransform copies that stamp
        // into the output cloud, so restore the sensor acquisition timestamp.
        cloud_msg.header.stamp = original_stamp;
        cloud_msg.header.frame_id = base_link_frame_;
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "failed to transform cloud: %s", ex.what());
        return;
      }
    }

    auto src_cloud = std::make_shared<pcl::PointCloud<PointT>>();
    pcl::fromROSMsg(cloud_msg, *src_cloud);
    if (src_cloud->empty()) {
      return;
    }

    pcl::PointCloud<PointT>::ConstPtr filtered;
    if (use_angle_calibration_) {
      filtered = verticalAngleCalibration(src_cloud);
      filtered = distanceFilter(filtered);
    } else {
      filtered = distanceFilter(src_cloud);
    }

    // 전처리 순서: 거리 필터 -> 다운샘플링 -> 이상치 제거.
    filtered = downsample(filtered);
    filtered = outlierRemoval(filtered);

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*filtered, output_msg);
    output_msg.header = cloud_msg.header;
    points_pub_->publish(output_msg);
  }

  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!downsample_filter_) {
      return cloud;
    }

    auto filtered = std::make_shared<pcl::PointCloud<PointT>>();
    downsample_filter_->setInputCloud(cloud);
    downsample_filter_->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr outlierRemoval(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!outlier_removal_filter_) {
      return cloud;
    }

    auto filtered = std::make_shared<pcl::PointCloud<PointT>>();
    outlier_removal_filter_->setInputCloud(cloud);
    outlier_removal_filter_->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr distanceFilter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!use_distance_filter_) {
      return cloud;
    }

    auto filtered = std::make_shared<pcl::PointCloud<PointT>>();
    filtered->reserve(cloud->size());
    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      const double d = p.getVector3fMap().norm();
      return d > distance_near_thresh_ && d < distance_far_thresh_;
    });

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;
    filtered->header = cloud->header;
    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr verticalAngleCalibration(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    auto filtered = std::make_shared<pcl::PointCloud<PointT>>();
    filtered->reserve(cloud->size());

    constexpr double delta = 0.11 * M_PI / 180.0;
    for (const auto& p : cloud->points) {
      Eigen::Vector3d axis = Eigen::Vector3d(p.x, p.y, p.z).cross(Eigen::Vector3d(0.0, 0.0, 1.0));
      if (axis.norm() < 1.0e-6) {
        filtered->push_back(p);
        continue;
      }
      axis.normalize();
      const Eigen::Vector3d corrected = Eigen::AngleAxisd(delta, axis) * Eigen::Vector3d(p.x, p.y, p.z);
      PointT out = p;
      out.x = corrected.x();
      out.y = corrected.y();
      out.z = corrected.z();
      filtered->push_back(out);
    }

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;
    filtered->header = cloud->header;
    return filtered;
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string base_link_frame_;
  bool use_distance_filter_;
  double distance_near_thresh_;
  double distance_far_thresh_;
  bool use_angle_calibration_;

  pcl::Filter<PointT>::Ptr downsample_filter_;
  pcl::Filter<PointT>::Ptr outlier_removal_filter_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

}  // namespace lidar_odometry

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lidar_odometry::PrefilteringNode>());
  rclcpp::shutdown();
  return 0;
}
