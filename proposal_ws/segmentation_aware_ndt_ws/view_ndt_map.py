#!/usr/bin/env python3

"""직렬화된 weighted NDT Gaussian voxel을 표시하는 Open3D viewer이다."""

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message="A NumPy version")

import numpy as np
import open3d as o3d

from view_map import read_pcd, semantic_colors


DEFAULT_MAP_PATH = Path(__file__).resolve().parent / "weighted_ndt_voxels_town10_global.pcd"


def scalar_colors(values):
    low, high = np.percentile(values, [2.0, 98.0])
    normalized = np.clip((values - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    return np.column_stack(
        (normalized, 1.0 - np.abs(2.0 * normalized - 1.0), 1.0 - normalized)
    )


def dimension_colors(labels):
    colors = np.full((len(labels), 3), [0.5, 0.5, 0.5], dtype=np.float64)
    colors[labels == 1] = [0.1, 0.9, 0.2]   # 선형 구조
    colors[labels == 2] = [1.0, 0.2, 0.1]   # 평면 구조
    colors[labels == 3] = [0.15, 0.35, 1.0]  # 구형/분산 구조
    return colors


def covariance_matrix(data, index):
    return np.array(
        [
            [data["cov_xx"][index], data["cov_xy"][index], data["cov_xz"][index]],
            [data["cov_xy"][index], data["cov_yy"][index], data["cov_yz"][index]],
            [data["cov_xz"][index], data["cov_yz"][index], data["cov_zz"][index]],
        ],
        dtype=np.float64,
    )


def make_ellipsoid(mean, covariance, color, sigma_scale):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or np.min(eigenvalues) <= 0.0:
        return None

    transform = np.eye(4)
    transform[:3, :3] = eigenvectors @ np.diag(sigma_scale * np.sqrt(eigenvalues))
    transform[:3, 3] = mean
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=6)
    mesh.transform(transform)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcd_path", nargs="?", type=Path, default=DEFAULT_MAP_PATH)
    parser.add_argument(
        "--color",
        choices=("semantic", "ndt-weight", "dimension", "point-count"),
        default="semantic",
    )
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument(
        "--ellipsoids",
        type=int,
        default=0,
        help="draw this many evenly sampled covariance ellipsoids; 0 disables them",
    )
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    map_path = args.pcd_path.expanduser().resolve()
    if not map_path.is_file():
        raise FileNotFoundError(f"weighted NDT voxel map was not found: {map_path}")

    data = read_pcd(map_path)
    required = {
        "x", "y", "z", "cov_xx", "cov_xy", "cov_xz", "cov_yy", "cov_yz",
        "cov_zz", "semantic_weight", "ndt_weight", "resolution",
        "dimension_label", "point_count",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"not a weighted NDT voxel map; missing fields: {missing}")

    means = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)
    semantic_weight = np.asarray(data["semantic_weight"], dtype=np.float64)
    ndt_weight = np.asarray(data["ndt_weight"], dtype=np.float64)
    labels = np.asarray(data["dimension_label"], dtype=np.int32)
    point_count = np.asarray(data["point_count"], dtype=np.float64)

    if args.color == "semantic":
        colors = semantic_colors(semantic_weight, 0.5, 1.0)
    elif args.color == "ndt-weight":
        colors = scalar_colors(ndt_weight)
    elif args.color == "dimension":
        colors = dimension_colors(labels)
    else:
        colors = scalar_colors(point_count)

    centroids = o3d.geometry.PointCloud()
    centroids.points = o3d.utility.Vector3dVector(means)
    centroids.colors = o3d.utility.Vector3dVector(colors)

    print(f"loading: {map_path}")
    print(f"NDT voxels: {len(means)}")
    print(f"resolution: {float(data['resolution'][0]):.3f} m")
    print(
        "semantic weight min/mean/max: "
        f"{semantic_weight.min():.6f} / {semantic_weight.mean():.6f} / "
        f"{semantic_weight.max():.6f}"
    )
    print(
        "NDT weight min/mean/max: "
        f"{ndt_weight.min():.6f} / {ndt_weight.mean():.6f} / {ndt_weight.max():.6f}"
    )
    print(
        "dimension voxels (linear/planar/spherical): "
        + " / ".join(str(int(np.count_nonzero(labels == value))) for value in (1, 2, 3))
    )

    if args.no_view:
        return

    geometries = [centroids]
    if args.ellipsoids > 0:
        sample_count = min(args.ellipsoids, len(means))
        indices = np.linspace(0, len(means) - 1, sample_count, dtype=np.int64)
        for index in indices:
            ellipsoid = make_ellipsoid(
                means[index], covariance_matrix(data, index), colors[index], args.sigma
            )
            if ellipsoid is not None:
                geometries.append(ellipsoid)
        print(f"covariance ellipsoids: {len(geometries) - 1}")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=f"Weighted NDT voxel map - {map_path.name}",
        width=1280,
        height=900,
    )
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.array([0.08, 0.08, 0.08])
    render_option.point_size = args.point_size
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
