cmake_minimum_required(VERSION 3.8)
project(segmentation_aware_ndt)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
  add_link_options(${OpenMP_CXX_FLAGS})
endif()

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(ndt_pca
  src/ndt_pca/voxel_grid_covariance_pca.cpp
  src/ndt_pca/ndt_pca.cpp
)
target_link_libraries(ndt_pca ${PCL_LIBRARIES})

add_library(ndt_omp
  src/ndt_omp/voxel_grid_covariance_omp.cpp
  src/ndt_omp/ndt_omp.cpp
)
target_link_libraries(ndt_omp ${PCL_LIBRARIES})

add_executable(semantic_map_builder src/apps/semantic_map_builder.cpp)
ament_target_dependencies(semantic_map_builder
  rclcpp
  sensor_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_sensor_msgs
)
target_link_libraries(semantic_map_builder
  ${PCL_LIBRARIES}
  ndt_omp
)

add_executable(semantic_ndt_localizer src/apps/semantic_ndt_localizer.cpp)
ament_target_dependencies(semantic_ndt_localizer
  rclcpp
  diagnostic_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  pcl_conversions
  tf2
  tf2_ros
)
target_link_libraries(semantic_ndt_localizer
  ${PCL_LIBRARIES}
  ndt_omp
)

add_executable(pointcloud_map_builder src/apps/pointcloud_map_builder.cpp)
ament_target_dependencies(pointcloud_map_builder
  rclcpp sensor_msgs geometry_msgs pcl_conversions tf2 tf2_ros
)
target_link_libraries(pointcloud_map_builder ${PCL_LIBRARIES})

add_executable(classical_ndt_localizer src/apps/weighted_ndt_localizer.cpp)
ament_target_dependencies(classical_ndt_localizer
  rclcpp diagnostic_msgs sensor_msgs nav_msgs geometry_msgs pcl_conversions tf2 tf2_ros
)
target_link_libraries(classical_ndt_localizer ${PCL_LIBRARIES} ndt_omp)

add_executable(semantic_ndt_voxel_builder src/apps/semantic_ndt_voxel_builder.cpp)
ament_target_dependencies(semantic_ndt_voxel_builder
  rclcpp
)
target_link_libraries(semantic_ndt_voxel_builder
  ${PCL_LIBRARIES}
  ndt_omp
)

install(TARGETS
  ndt_pca
  ndt_omp
  semantic_map_builder
  semantic_ndt_localizer
  semantic_ndt_voxel_builder
  pointcloud_map_builder
  classical_ndt_localizer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

install(PROGRAMS
  scripts/kitti_sequence_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME kitti_sequence_publisher
)

ament_package()
