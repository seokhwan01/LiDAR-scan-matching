#ifndef SEGMENTATION_AWARE_NDT__WEIGHTED_NDT_VOXEL_HPP_
#define SEGMENTATION_AWARE_NDT__WEIGHTED_NDT_VOXEL_HPP_

#include <cmath>
#include <cstdint>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include <ndt_omp/voxel_grid_covariance_omp.h>

// PCD point 하나가 완전한 target NDT Gaussian 하나를 나타낸다. x/y/z에는
// 원본 LiDAR return 대신 voxel 평균을 저장한다. 공분산 행렬은 대칭이므로
// 서로 다른 여섯 개 원소만 저장한다.
struct EIGEN_ALIGN16 WeightedNdtVoxelPoint {
  PCL_ADD_POINT4D;
  float cov_xx{0.0F};
  float cov_xy{0.0F};
  float cov_xz{0.0F};
  float cov_yy{0.0F};
  float cov_yz{0.0F};
  float cov_zz{0.0F};
  float semantic_weight{0.0F};
  float ndt_weight{0.0F};
  float resolution{0.0F};
  std::int32_t voxel_x{0};
  std::int32_t voxel_y{0};
  std::int32_t voxel_z{0};
  std::uint32_t dimension_label{0};
  std::uint32_t point_count{0};
  std::uint32_t format_version{0};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    WeightedNdtVoxelPoint,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, cov_xx, cov_xx)
    (float, cov_xy, cov_xy)
    (float, cov_xz, cov_xz)
    (float, cov_yy, cov_yy)
    (float, cov_yz, cov_yz)
    (float, cov_zz, cov_zz)
    (float, semantic_weight, semantic_weight)
    (float, ndt_weight, ndt_weight)
    (float, resolution, resolution)
    (std::int32_t, voxel_x, voxel_x)
    (std::int32_t, voxel_y, voxel_y)
    (std::int32_t, voxel_z, voxel_z)
    (std::uint32_t, dimension_label, dimension_label)
    (std::uint32_t, point_count, point_count)
    (std::uint32_t, format_version, format_version))

namespace segmentation_aware_ndt {

constexpr std::uint32_t kOmpNdtVoxelFormatVersion = 2U;

using NdtTargetGrid = pclomp::VoxelGridCovariance<pcl::PointXYZI>;
using PrecomputedNdtLeaf = NdtTargetGrid::PrecomputedLeaf;

inline std::size_t buildWeightedNdtVoxelCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& semantic_map,
    float resolution,
    int min_points,
    pcl::PointCloud<WeightedNdtVoxelPoint>& output,
    float dimension_scale_linear = 0.75F,
    float dimension_scale_planar = 1.25F,
    float dimension_scale_volumetric = 1.0F) {
  NdtTargetGrid grid;
  grid.setLeafSize(resolution, resolution, resolution);
  grid.setMinPointPerVoxel(min_points);
  grid.setInputCloud(semantic_map);
  grid.filter(false);

  output.clear();
  output.reserve(grid.getLeaves().size());
  for (const auto& entry : grid.getLeaves()) {
    const auto& leaf = entry.second;
    if (leaf.getPointCount() < min_points || leaf.getDimensionLabel() < 1 ||
        leaf.getDimensionLabel() > 3 || leaf.getSemanticWeight() <= 0.0 ||
        !leaf.getMean().allFinite() ||
        !leaf.getCov().allFinite() || !leaf.getInverseCov().allFinite()) {
      continue;
    }

    const auto mean = leaf.getMean();
    const auto covariance = leaf.getCov();
    WeightedNdtVoxelPoint point;
    point.x = static_cast<float>(mean.x());
    point.y = static_cast<float>(mean.y());
    point.z = static_cast<float>(mean.z());
    point.cov_xx = static_cast<float>(covariance(0, 0));
    point.cov_xy = static_cast<float>(covariance(0, 1));
    point.cov_xz = static_cast<float>(covariance(0, 2));
    point.cov_yy = static_cast<float>(covariance(1, 1));
    point.cov_yz = static_cast<float>(covariance(1, 2));
    point.cov_zz = static_cast<float>(covariance(2, 2));
    point.semantic_weight = static_cast<float>(leaf.getSemanticWeight());
    float dimension_scale = dimension_scale_volumetric;
    if (leaf.getDimensionLabel() == 1) {
      dimension_scale = dimension_scale_linear;
    } else if (leaf.getDimensionLabel() == 2) {
      dimension_scale = dimension_scale_planar;
    }
    point.ndt_weight = static_cast<float>(leaf.getSemanticWeight()) * dimension_scale;
    point.resolution = resolution;
    point.voxel_x = static_cast<std::int32_t>(std::floor(mean.x() / resolution));
    point.voxel_y = static_cast<std::int32_t>(std::floor(mean.y() / resolution));
    point.voxel_z = static_cast<std::int32_t>(std::floor(mean.z() / resolution));
    point.dimension_label = static_cast<std::uint32_t>(leaf.getDimensionLabel());
    point.point_count = static_cast<std::uint32_t>(leaf.getPointCount());
    point.format_version = kOmpNdtVoxelFormatVersion;
    output.push_back(point);
  }
  output.width = static_cast<std::uint32_t>(output.size());
  output.height = 1;
  output.is_dense = true;
  return output.size();
}

inline std::vector<PrecomputedNdtLeaf> decodeWeightedNdtVoxelCloud(
    const pcl::PointCloud<WeightedNdtVoxelPoint>& cloud) {
  std::vector<PrecomputedNdtLeaf> leaves;
  leaves.reserve(cloud.size());
  for (const auto& point : cloud.points) {
    PrecomputedNdtLeaf leaf;
    leaf.voxel_index = Eigen::Vector3i(point.voxel_x, point.voxel_y, point.voxel_z);
    leaf.mean = Eigen::Vector3d(point.x, point.y, point.z);
    leaf.covariance <<
        point.cov_xx, point.cov_xy, point.cov_xz,
        point.cov_xy, point.cov_yy, point.cov_yz,
        point.cov_xz, point.cov_yz, point.cov_zz;
    leaf.point_count = static_cast<int>(point.point_count);
    leaf.dimension_label = static_cast<int>(point.dimension_label);
    leaf.semantic_weight = point.semantic_weight;
    leaves.push_back(leaf);
  }
  return leaves;
}

}  // namespace segmentation_aware_ndt

#endif  // SEGMENTATION_AWARE_NDT__WEIGHTED_NDT_VOXEL_HPP_
