#pragma once

#include <cstdint>
#include <cstddef>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>

#include <ndt_pca/voxel_grid_covariance_pca.h>

struct EIGEN_ALIGN16 PointXYZIWeight {
  PCL_ADD_POINT4D;
  float intensity{0.0F};
  float weight{0.0F};
  std::uint32_t dimension_label{0};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIWeight,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, weight, weight)
    (std::uint32_t, dimension_label, dimension_label))

namespace ndt_mapping_localization {

inline std::size_t buildWeightedMap(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& map,
    double ndt_resolution,
    int min_points,
    pcl::PointCloud<PointXYZIWeight>& output) {
  pclpca::VoxelGridCovariance<pcl::PointXYZI> ndt_voxels;
  ndt_voxels.setLeafSize(ndt_resolution, ndt_resolution, ndt_resolution);
  ndt_voxels.setMinPointPerVoxel(min_points);
  ndt_voxels.setInputCloud(map);
  ndt_voxels.filter(false);

  output.clear();
  output.reserve(map->size());
  std::size_t valid_weighted_points = 0;
  for (auto& point : map->points) {
    PointXYZIWeight weighted;
    weighted.x = point.x;
    weighted.y = point.y;
    weighted.z = point.z;
    weighted.intensity = point.intensity;

    const auto* leaf = ndt_voxels.getLeaf(point);
    if (leaf != nullptr && leaf->getPointCount() >= min_points &&
        leaf->getDimensionLabel() > 0) {
      weighted.weight = static_cast<float>(leaf->getDimension2d());
      weighted.dimension_label = static_cast<std::uint32_t>(leaf->getDimensionLabel());
      ++valid_weighted_points;
    }
    output.push_back(weighted);
  }
  output.width = static_cast<std::uint32_t>(output.size());
  output.height = 1;
  output.is_dense = map->is_dense;
  return valid_weighted_points;
}

}  // namespace ndt_mapping_localization
