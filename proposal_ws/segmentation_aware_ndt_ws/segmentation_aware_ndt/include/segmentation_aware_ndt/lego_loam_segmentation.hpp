#ifndef SEGMENTATION_AWARE_NDT__LEGO_LOAM_SEGMENTATION_HPP_
#define SEGMENTATION_AWARE_NDT__LEGO_LOAM_SEGMENTATION_HPP_

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace segmentation_aware_ndt {

// 맵 생성과 online localization에 공통으로 사용하는 결정론적 range-image
// segmentation 파라미터이다. 기본값은 실험용 CARLA 센서와 동일하다.
//   32채널, 수직 FOV [-30, 10]도, 10 Hz에서 초당 347 k point
// 두 경로가 하나의 파라미터 구조체를 공유하므로 pre-built map과 live scan에
// 서로 다른 센서 모델로 label이 할당되는 문제를 방지한다.
struct SegmentationParameters {
  int n_scan{32};
  int horizon_scan{1084};
  double ang_res_x{360.0 / 1084.0};
  double ang_res_y{40.0 / 31.0};
  double ang_bottom{30.0};
  int ground_scan_ind{20};
  double sensor_minimum_range{1.0};
  double sensor_mount_angle{0.0};
  double segment_theta{60.0 * M_PI / 180.0};
  int segment_valid_point_num{5};
  int segment_valid_line_num{3};
  float ground_weight{0.5F};
  float nonground_weight{1.0F};
  float neutral_weight{0.75F};
};

struct SegmentationStatistics {
  std::size_t input_points{0};
  std::size_t projected_points{0};
  std::size_t ground_points{0};
  std::size_t ground_points_kept{0};
  std::size_t ground_points_dropped{0};
  std::size_t nonground_points{0};
  std::size_t rejected_points{0};
};

// LeGO-LOAM 방식의 기하학적 segmentation이다.
//
// 학습 기반 semantic network가 아니며 ground, 유효하게 연결된 non-ground
// segment, rejected/outlier의 세 기하 class를 생성한다. 출력에는 range-image
// cell마다 가장 가까운 point 하나가 들어간다. 동일 표현을 PCD로 저장하고 custom
// weighted NDT에서 사용하도록 PointXYZI::intensity를 class weight로 덮어쓴다.
class LegoLoamSegmentation {
public:
  using PointT = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<PointT>;

  explicit LegoLoamSegmentation(const SegmentationParameters& parameters)
      : parameters_(parameters) {
    validateParameters();
  }

  const SegmentationParameters& parameters() const {
    return parameters_;
  }

  Cloud::Ptr segment(
      const Cloud& input,
      SegmentationStatistics* statistics = nullptr) const {
    SegmentationStatistics local_statistics;
    local_statistics.input_points = input.size();

    RangeImage frame(parameters_.n_scan, parameters_.horizon_scan);
    projectPointCloud(input, frame);
    removeGround(input, frame);
    labelNonGroundSegments(frame);

    auto output = Cloud::Ptr(new Cloud());
    output->reserve(frame.cell_point_index.size());

    for (std::size_t index = 0; index < frame.cell_point_index.size(); ++index) {
      const int point_index = frame.cell_point_index[index];
      if (point_index < 0) {
        continue;
      }
      ++local_statistics.projected_points;

      float semantic_weight = 0.0F;
      if (frame.ground[index] == 1) {
        semantic_weight = parameters_.ground_weight;
        ++local_statistics.ground_points;
        ++local_statistics.ground_points_kept;
      } else if (frame.label[index] > 0 && frame.label[index] != kOutlierLabel) {
        semantic_weight = parameters_.nonground_weight;
        ++local_statistics.nonground_points;
      } else {
        ++local_statistics.rejected_points;
        continue;
      }

      PointT point = input.points[static_cast<std::size_t>(point_index)];
      point.intensity = semantic_weight;
      output->push_back(point);
    }

    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = input.is_dense;

    if (statistics != nullptr) {
      *statistics = local_statistics;
    }
    return output;
  }

  // Keep the original cloud geometry and use segmentation only as a label
  // generator. Points selected by the range image receive ground/non-ground
  // labels. Rejected or non-projected raw points are kept with neutral weight.
  Cloud::Ptr labelPreservingGeometry(
      const Cloud& input,
      SegmentationStatistics* statistics = nullptr) const {
    SegmentationStatistics local_statistics;
    local_statistics.input_points = input.size();

    RangeImage frame(parameters_.n_scan, parameters_.horizon_scan);
    projectPointCloud(input, frame);
    removeGround(input, frame);
    labelNonGroundSegments(frame);

    std::vector<float> labels(input.size(), parameters_.neutral_weight);
    for (std::size_t index = 0; index < frame.cell_point_index.size(); ++index) {
      const int point_index = frame.cell_point_index[index];
      if (point_index < 0) {
        continue;
      }
      ++local_statistics.projected_points;

      float semantic_weight = parameters_.neutral_weight;
      if (frame.ground[index] == 1) {
        semantic_weight = parameters_.ground_weight;
        ++local_statistics.ground_points;
        ++local_statistics.ground_points_kept;
      } else if (frame.label[index] > 0 && frame.label[index] != kOutlierLabel) {
        semantic_weight = parameters_.nonground_weight;
        ++local_statistics.nonground_points;
      } else {
        ++local_statistics.rejected_points;
      }
      labels[static_cast<std::size_t>(point_index)] = semantic_weight;
    }

    auto output = Cloud::Ptr(new Cloud());
    output->reserve(input.size());
    for (std::size_t point_index = 0; point_index < input.size(); ++point_index) {
      const auto& source = input.points[point_index];
      if (!std::isfinite(source.x) || !std::isfinite(source.y) ||
          !std::isfinite(source.z)) {
        continue;
      }
      PointT point = source;
      point.intensity = labels[point_index];
      output->push_back(point);
    }
    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = input.is_dense;

    if (statistics != nullptr) {
      *statistics = local_statistics;
    }
    return output;
  }

private:
  static constexpr int kOutlierLabel = 999999;

  struct RangeImage {
    RangeImage(int rows, int columns)
        : rows(rows),
          columns(columns),
          range(static_cast<std::size_t>(rows * columns), FLT_MAX),
          label(static_cast<std::size_t>(rows * columns), 0),
          ground(static_cast<std::size_t>(rows * columns), 0),
          cell_point_index(static_cast<std::size_t>(rows * columns), -1) {}

    int rows;
    int columns;
    std::vector<float> range;
    std::vector<int> label;
    std::vector<std::int8_t> ground;
    std::vector<int> cell_point_index;
  };

  void validateParameters() const {
    if (parameters_.n_scan < 2 || parameters_.horizon_scan < 2) {
      throw std::invalid_argument("segmentation image dimensions must be >= 2");
    }
    if (parameters_.ang_res_x <= 0.0 || parameters_.ang_res_y <= 0.0) {
      throw std::invalid_argument("segmentation angular resolutions must be positive");
    }
    if (parameters_.ground_scan_ind < 0 ||
        parameters_.ground_scan_ind >= parameters_.n_scan - 1) {
      throw std::invalid_argument("ground_scan_ind must be in [0, n_scan - 2]");
    }
    if (parameters_.ground_weight <= 0.0F || parameters_.nonground_weight <= 0.0F) {
      throw std::invalid_argument("semantic weights must be positive");
    }
    if (parameters_.ground_weight >= parameters_.nonground_weight) {
      throw std::invalid_argument(
          "ground_weight must be smaller than nonground_weight for class decoding");
    }
    if (parameters_.neutral_weight <= 0.0F) {
      throw std::invalid_argument("neutral_weight must be positive");
    }
  }

  int cellIndex(int row, int column) const {
    return row * parameters_.horizon_scan + column;
  }

  void projectPointCloud(const Cloud& input, RangeImage& frame) const {
    for (std::size_t point_index = 0; point_index < input.size(); ++point_index) {
      const auto& point = input.points[point_index];
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
        continue;
      }

      const double horizontal_range = std::hypot(point.x, point.y);
      const double range = std::hypot(horizontal_range, static_cast<double>(point.z));
      if (range < parameters_.sensor_minimum_range) {
        continue;
      }

      const double vertical_angle =
          std::atan2(point.z, horizontal_range) * 180.0 / M_PI;
      const int row = static_cast<int>(std::round(
          (vertical_angle + parameters_.ang_bottom) / parameters_.ang_res_y));
      if (row < 0 || row >= parameters_.n_scan) {
        continue;
      }

      // LeGO-LOAM과 동일한 column 규칙이다. CARLA publisher가 출력하는 ROS
      // 센서 좌표계(x 전방, y 좌측, z 상방)와 호환된다.
      const double horizon_angle = std::atan2(point.x, point.y) * 180.0 / M_PI;
      int column = static_cast<int>(
          -std::round((horizon_angle - 90.0) / parameters_.ang_res_x) +
          parameters_.horizon_scan / 2);
      column %= parameters_.horizon_scan;
      if (column < 0) {
        column += parameters_.horizon_scan;
      }

      const int index = cellIndex(row, column);
      if (range < frame.range[static_cast<std::size_t>(index)]) {
        frame.range[static_cast<std::size_t>(index)] = static_cast<float>(range);
        frame.cell_point_index[static_cast<std::size_t>(index)] =
            static_cast<int>(point_index);
      }
    }
  }

  void removeGround(const Cloud& input, RangeImage& frame) const {
    for (int column = 0; column < parameters_.horizon_scan; ++column) {
      for (int row = 0; row < parameters_.ground_scan_ind; ++row) {
        const int lower_index = cellIndex(row, column);
        const int upper_index = cellIndex(row + 1, column);
        const int lower_point = frame.cell_point_index[static_cast<std::size_t>(lower_index)];
        const int upper_point = frame.cell_point_index[static_cast<std::size_t>(upper_index)];
        if (lower_point < 0 || upper_point < 0) {
          continue;
        }

        const auto& lower = input.points[static_cast<std::size_t>(lower_point)];
        const auto& upper = input.points[static_cast<std::size_t>(upper_point)];
        const float dx = upper.x - lower.x;
        const float dy = upper.y - lower.y;
        const float dz = upper.z - lower.z;
        const double angle = std::atan2(dz, std::hypot(dx, dy)) * 180.0 / M_PI;

        if (std::abs(angle - parameters_.sensor_mount_angle) <= 10.0) {
          frame.ground[static_cast<std::size_t>(lower_index)] = 1;
          frame.ground[static_cast<std::size_t>(upper_index)] = 1;
        }
      }
    }
  }

  void labelNonGroundSegments(RangeImage& frame) const {
    int next_label = 1;
    for (int row = 0; row < parameters_.n_scan; ++row) {
      for (int column = 0; column < parameters_.horizon_scan; ++column) {
        const int index = cellIndex(row, column);
        const std::size_t storage_index = static_cast<std::size_t>(index);
        if (frame.cell_point_index[storage_index] < 0 ||
            frame.ground[storage_index] == 1 || frame.label[storage_index] != 0) {
          continue;
        }
        labelComponent(row, column, next_label, frame);
      }
    }
  }

  void labelComponent(int seed_row, int seed_column, int& next_label, RangeImage& frame) const {
    static const std::pair<int, int> kNeighborOffsets[] = {
        {0, -1}, {0, 1}, {-1, 0}, {1, 0}};

    std::vector<int> queue;
    std::vector<int> component;
    std::vector<bool> lines_used(static_cast<std::size_t>(parameters_.n_scan), false);
    queue.reserve(256);
    component.reserve(256);

    const int seed_index = cellIndex(seed_row, seed_column);
    frame.label[static_cast<std::size_t>(seed_index)] = next_label;
    queue.push_back(seed_index);
    component.push_back(seed_index);
    lines_used[static_cast<std::size_t>(seed_row)] = true;

    for (std::size_t queue_index = 0; queue_index < queue.size(); ++queue_index) {
      const int from_index = queue[queue_index];
      const int from_row = from_index / parameters_.horizon_scan;
      const int from_column = from_index % parameters_.horizon_scan;

      for (const auto& offset : kNeighborOffsets) {
        const int neighbor_row = from_row + offset.first;
        if (neighbor_row < 0 || neighbor_row >= parameters_.n_scan) {
          continue;
        }

        int neighbor_column = from_column + offset.second;
        if (neighbor_column < 0) {
          neighbor_column = parameters_.horizon_scan - 1;
        } else if (neighbor_column >= parameters_.horizon_scan) {
          neighbor_column = 0;
        }

        const int neighbor_index = cellIndex(neighbor_row, neighbor_column);
        const std::size_t neighbor_storage = static_cast<std::size_t>(neighbor_index);
        if (frame.cell_point_index[neighbor_storage] < 0 ||
            frame.ground[neighbor_storage] == 1 || frame.label[neighbor_storage] != 0) {
          continue;
        }

        const float from_range = frame.range[static_cast<std::size_t>(from_index)];
        const float neighbor_range = frame.range[neighbor_storage];
        const double d1 = std::max(from_range, neighbor_range);
        const double d2 = std::min(from_range, neighbor_range);
        const double alpha = (offset.first == 0)
            ? parameters_.ang_res_x * M_PI / 180.0
            : parameters_.ang_res_y * M_PI / 180.0;
        const double angle =
            std::atan2(d2 * std::sin(alpha), d1 - d2 * std::cos(alpha));

        if (angle > parameters_.segment_theta) {
          frame.label[neighbor_storage] = next_label;
          queue.push_back(neighbor_index);
          component.push_back(neighbor_index);
          lines_used[static_cast<std::size_t>(neighbor_row)] = true;
        }
      }
    }

    bool valid_segment = component.size() >= 30;
    if (!valid_segment &&
        component.size() >= static_cast<std::size_t>(parameters_.segment_valid_point_num)) {
      const int line_count = static_cast<int>(
          std::count(lines_used.begin(), lines_used.end(), true));
      valid_segment = line_count >= parameters_.segment_valid_line_num;
    }

    if (valid_segment) {
      ++next_label;
      return;
    }

    for (const int index : component) {
      frame.label[static_cast<std::size_t>(index)] = kOutlierLabel;
    }
  }

  SegmentationParameters parameters_;
};

}  // namespace segmentation_aware_ndt

#endif  // SEGMENTATION_AWARE_NDT__LEGO_LOAM_SEGMENTATION_HPP_
