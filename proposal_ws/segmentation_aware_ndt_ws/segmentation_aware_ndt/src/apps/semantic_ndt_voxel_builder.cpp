#include <memory>
#include <stdexcept>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>

#include <segmentation_aware_ndt/weighted_ndt_voxel.hpp>

namespace {

// mapping bag을 재생하지 않고 직렬화된 NDT map을 다시 만드는 일회성 변환기이다.
// 일반 mapping launch는 두 파일을 모두 저장한다. 이 실행 파일은 NDT resolution
// 또는 voxel 최소 point 수만 변경할 때 사용한다.
class SemanticNdtVoxelBuilder : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<PointT>;

  SemanticNdtVoxelBuilder() : rclcpp::Node("semantic_ndt_voxel_builder") {
    const auto input_path = declare_parameter<std::string>(
        "input_map_path", "semantic_weighted_map.pcd");
    const auto output_path = declare_parameter<std::string>(
        "output_ndt_map_path", "weighted_ndt_voxels.pcd");
    const auto map_leaf_size = declare_parameter<double>("map_leaf_size", 0.0);
    const auto ndt_resolution = declare_parameter<double>("ndt_resolution", 1.0);
    const auto ndt_min_points = declare_parameter<int>("ndt_min_points", 6);
    const auto dimension_scale_linear =
        declare_parameter<double>("dimension_scale_linear", 0.75);
    const auto dimension_scale_planar =
        declare_parameter<double>("dimension_scale_planar", 1.25);
    const auto dimension_scale_volumetric =
        declare_parameter<double>("dimension_scale_volumetric", 1.0);

    auto semantic_map = Cloud::Ptr(new Cloud());
    if (pcl::io::loadPCDFile(input_path, *semantic_map) != 0 || semantic_map->empty()) {
      throw std::runtime_error("failed to load semantic point map: " + input_path);
    }

    // semantic_map_builder는 map_leaf_size filtering을 적용한 point map을 저장한다.
    // 평균 point가 voxel 경계를 넘어가 mapping 종료 때 생성한 NDT map과 달라질 수
    // 있으므로 기본 동작에서는 두 번째 filtering을 수행하지 않는다.
    auto ndt_input = semantic_map;
    if (map_leaf_size > 0.0) {
      pcl::VoxelGrid<PointT> voxel_grid;
      voxel_grid.setLeafSize(map_leaf_size, map_leaf_size, map_leaf_size);
      voxel_grid.setInputCloud(semantic_map);
      ndt_input = Cloud::Ptr(new Cloud());
      voxel_grid.filter(*ndt_input);
    }

    pcl::PointCloud<WeightedNdtVoxelPoint> ndt_voxels;
    const auto count = segmentation_aware_ndt::buildWeightedNdtVoxelCloud(
        ndt_input, ndt_resolution, ndt_min_points, ndt_voxels,
        dimension_scale_linear, dimension_scale_planar,
        dimension_scale_volumetric);
    if (count == 0 || pcl::io::savePCDFileBinary(output_path, ndt_voxels) != 0) {
      throw std::runtime_error("failed to save weighted NDT voxel map: " + output_path);
    }

    RCLCPP_INFO(
        get_logger(),
        "converted semantic map '%s' (%zu points) -> NDT map '%s' (%zu voxels, %.3f m)",
        input_path.c_str(), ndt_input->size(), output_path.c_str(), count, ndt_resolution);
  }
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<SemanticNdtVoxelBuilder>();
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("semantic_ndt_voxel_builder"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
