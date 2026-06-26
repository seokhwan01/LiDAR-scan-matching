cmake_minimum_required(VERSION 3.8)
project(ndt_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_executable(ndt_localizer_node src/ndt_localizer_node.cpp)

ament_target_dependencies(
  ndt_localizer_node
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
)

target_link_libraries(
  ndt_localizer_node
  ${PCL_LIBRARIES}
)

install(
  TARGETS ndt_localizer_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()