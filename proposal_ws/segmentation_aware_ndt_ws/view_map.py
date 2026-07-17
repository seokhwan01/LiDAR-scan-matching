#!/usr/bin/env python3

"""Segmentation-aware weighted PCD map을 표시하는 Open3D viewer이다.

Map builder는 PointXYZI::intensity에 semantic weight를 저장한다.
  - ground: 0.5
  - non-ground: 1.0
  - 중간값: voxel 평균 class confidence
"""

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="A NumPy version")

import numpy as np
import open3d as o3d


DEFAULT_MAP_PATH = Path(__file__).resolve().parent / "semantic_weighted_map.pcd"


def _numpy_type(field_type, field_size):
    type_table = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("I", 1): "<i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
        ("I", 8): "<i8",
        ("U", 1): "<u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("U", 8): "<u8",
    }
    try:
        return type_table[(field_type, field_size)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported PCD field type: TYPE={field_type}, SIZE={field_size}"
        ) from exc


def read_pcd(path):
    """ASCII 또는 압축하지 않은 binary PCD의 field를 읽는다."""
    metadata = {}
    header_line_count = 0

    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("invalid PCD: DATA header was not found")
            header_line_count += 1
            text = line.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#"):
                continue

            key, *values = text.split()
            metadata[key.upper()] = values
            if key.upper() == "DATA":
                break

        fields = metadata["FIELDS"]
        sizes = [int(value) for value in metadata["SIZE"]]
        types = metadata["TYPE"]
        counts = [int(value) for value in metadata.get("COUNT", ["1"] * len(fields))]
        point_count = int(metadata.get("POINTS", metadata["WIDTH"])[0])
        data_type = metadata["DATA"][0].lower()

        if any(count != 1 for count in counts):
            raise ValueError("PCD fields with COUNT other than 1 are not supported")

        if data_type == "binary":
            dtype = np.dtype([
                (field, _numpy_type(field_type, size))
                for field, field_type, size in zip(fields, types, sizes)
            ])
            raw = np.fromfile(stream, dtype=dtype, count=point_count)
            return {field: raw[field] for field in fields}

        if data_type == "ascii":
            raw = np.loadtxt(path, comments="#", skiprows=header_line_count)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            return {field: raw[:, index] for index, field in enumerate(fields)}

        if data_type == "binary_compressed":
            raise ValueError("binary_compressed PCD is not supported")
        raise ValueError(f"unsupported PCD DATA type: {data_type}")


def semantic_colors(weights, ground_weight, nonground_weight):
    """ground는 파랑, non-ground는 빨강으로 표시하고 중간값은 연속 보간한다."""
    denominator = max(nonground_weight - ground_weight, np.finfo(np.float64).eps)
    normalized = np.clip((weights - ground_weight) / denominator, 0.0, 1.0)

    ground_color = np.array([0.05, 0.35, 1.0], dtype=np.float64)
    nonground_color = np.array([1.0, 0.15, 0.05], dtype=np.float64)
    return (
        (1.0 - normalized[:, np.newaxis]) * ground_color
        + normalized[:, np.newaxis] * nonground_color
    )


def height_colors(points):
    z = points[:, 2]
    low, high = np.percentile(z, [2.0, 98.0])
    normalized = np.clip((z - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    return np.column_stack((normalized, 1.0 - np.abs(2.0 * normalized - 1.0), 1.0 - normalized))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd_path", nargs="?", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument("--color", choices=("semantic", "height", "none"), default="semantic")
    parser.add_argument("--ground-weight", type=float, default=0.5)
    parser.add_argument("--nonground-weight", type=float, default=1.0)
    parser.add_argument("--class-threshold", type=float, default=0.75)
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="Open3D display downsampling size in metres; 0 keeps the saved map unchanged",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--no-view", action="store_true", help="load and print statistics only")
    args = parser.parse_args()

    map_path = args.pcd_path.expanduser().resolve()
    if not map_path.is_file():
        raise FileNotFoundError(f"map was not found: {map_path}")

    print(f"loading: {map_path}")
    data = read_pcd(map_path)
    required_fields = {"x", "y", "z"}
    if not required_fields.issubset(data):
        raise ValueError("PCD must contain x, y and z fields")

    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    if "intensity" in data:
        finite &= np.isfinite(data["intensity"])
    points = points[finite]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    print(f"fields: {list(data.keys())}")
    print(f"points: {len(points)}")

    if args.color == "semantic":
        if "intensity" not in data:
            raise ValueError("semantic coloring requires the PCD intensity field")
        weights = np.asarray(data["intensity"], dtype=np.float64)[finite]
        cloud.colors = o3d.utility.Vector3dVector(
            semantic_colors(weights, args.ground_weight, args.nonground_weight)
        )

        ground_count = int(np.count_nonzero(weights <= args.class_threshold))
        nonground_count = int(len(weights) - ground_count)
        print(
            "semantic weight min/mean/max: "
            f"{weights.min():.6f} / {weights.mean():.6f} / {weights.max():.6f}"
        )
        print(f"ground points: {ground_count}")
        print(f"non-ground points: {nonground_count}")
        print("colors: blue=ground, red=non-ground, intermediate=mixed voxel")
    elif args.color == "height":
        cloud.colors = o3d.utility.Vector3dVector(height_colors(points))
    else:
        cloud.paint_uniform_color([0.85, 0.85, 0.85])

    if args.voxel > 0.0:
        cloud = cloud.voxel_down_sample(args.voxel)
        print(f"display points after {args.voxel:.3f} m voxel downsampling: {len(cloud.points)}")

    if args.no_view:
        return

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=f"Segmentation-aware NDT map - {map_path.name}",
        width=1280,
        height=900,
    )
    visualizer.add_geometry(cloud)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.array([0.08, 0.08, 0.08])
    render_option.point_size = args.point_size
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
