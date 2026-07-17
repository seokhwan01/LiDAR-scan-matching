#include <cstdlib>
#include <iostream>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ndt_mapping_localization/weighted_map.hpp>

int main(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr
        << "usage: weighted_map_converter INPUT.pcd OUTPUT.pcd "
        << "[map_resolution=0.2] [ndt_resolution=1.0] [min_points=6]\n";
    return 2;
  }

  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const double map_resolution = argc > 3 ? std::stod(argv[3]) : 0.2;
  const double ndt_resolution = argc > 4 ? std::stod(argv[4]) : 1.0;
  const int min_points = argc > 5 ? std::stoi(argv[5]) : 6;
  if (map_resolution <= 0.0 || ndt_resolution <= 0.0 || min_points < 3) {
    std::cerr << "resolutions must be positive and min_points must be at least 3\n";
    return 2;
  }

  using PointT = pcl::PointXYZI;
  pcl::PointCloud<PointT>::Ptr input(new pcl::PointCloud<PointT>());
  if (pcl::io::loadPCDFile(input_path, *input) != 0 || input->empty()) {
    std::cerr << "failed to load input map: " << input_path << '\n';
    return 3;
  }

  pcl::VoxelGrid<PointT> voxel;
  voxel.setLeafSize(map_resolution, map_resolution, map_resolution);
  voxel.setInputCloud(input);
  pcl::PointCloud<PointT>::Ptr map(new pcl::PointCloud<PointT>());
  voxel.filter(*map);

  pcl::PointCloud<PointXYZIWeight> weighted;
  const std::size_t valid = ndt_mapping_localization::buildWeightedMap(
      map, ndt_resolution, min_points, weighted);
  if (pcl::io::savePCDFileBinary(output_path, weighted) != 0) {
    std::cerr << "failed to save weighted map: " << output_path << '\n';
    return 4;
  }

  std::cout << "input points: " << input->size() << '\n'
            << "downsampled points: " << map->size() << '\n'
            << "valid weighted points: " << valid << '\n'
            << "saved: " << output_path << '\n';
  return 0;
}
