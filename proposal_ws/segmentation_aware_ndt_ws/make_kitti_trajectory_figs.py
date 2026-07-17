#!/usr/bin/env python3

"""Generate KITTI trajectory PDF figures from localization evaluation CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUNS = {
    "00": Path(
        "kitti_output_relaxed/sequence_00/evaluation_direct7/weighted_relaxed_direct7/"
        "kitti_00_weighted_relaxed_direct7_20260713_195534/evaluation/matched_poses.csv"
    ),
    "01": Path(
        "kitti_output_relaxed/sequence_01/evaluation_direct7/weighted_relaxed_direct7/"
        "kitti_01_weighted_relaxed_direct7_20260713_200355/evaluation/matched_poses.csv"
    ),
    "03": Path(
        "kitti_output_relaxed/sequence_03/evaluation_direct7/weighted_relaxed_direct7/"
        "kitti_03_weighted_relaxed_direct7_20260713_200552/evaluation/matched_poses.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 00/01/03 KITTI trajectory PDF figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/seokhwan/LiDAR-scan-matching-ictc2026/figs"),
        help="Directory to write *_trajectory_xy.pdf files.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=14.0,
        help="Base font size.",
    )
    ####################################
    parser.add_argument(
        "--axis-label-size",
        type=float,
        default=10.0,
        help="Axis label font size.",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=10.0,
        help="Tick label font size.",
    )
    parser.add_argument(
        "--legend-size",
        type=float,
        default=11.0,
        help="Legend font size.",
    )
    ######################################
    parser.add_argument(
        "--legend-loc",
        default="upper right",
        help="Matplotlib legend location, e.g., 'upper right', 'lower right'.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=6.2,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=4.8,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--gt-label",
        default="Ground truth",
        help="Legend label for ground truth trajectory.",
    )
    parser.add_argument(
        "--estimate-label",
        default="Weighted-S50",
        help="Legend label for estimated trajectory.",
    )
    parser.add_argument(
        "--gt-linewidth",
        type=float,
        default=2.4,
        help="Ground truth line width.",
    )
    parser.add_argument(
        "--estimate-linewidth",
        type=float,
        default=1.9,
        help="Estimated trajectory line width.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["00", "01", "03"],
        choices=sorted(DEFAULT_RUNS),
        help="Sequences to generate.",
    )
    return parser.parse_args()


def configure_matplotlib(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.labelsize": args.axis_label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.legend_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_sequence(seq: str, csv_path: Path, args: argparse.Namespace) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing matched pose CSV: {csv_path}")

    data = pd.read_csv(csv_path)
    output_path = args.output_dir / f"{seq}_trajectory_xy.pdf"

    figure, axis = plt.subplots(figsize=(args.fig_width, args.fig_height))
    axis.plot(
        data["gt_x"],
        data["gt_y"],
        label=args.gt_label,
        linewidth=args.gt_linewidth,
    )
    axis.plot(
        data["estimate_x"],
        data["estimate_y"],
        label=args.estimate_label,
        linewidth=args.estimate_linewidth,
    )
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    axis.axis("equal")
    axis.grid(True, linewidth=0.6, alpha=0.55)
    axis.legend(loc=args.legend_loc, frameon=True)
    figure.tight_layout(pad=0.35)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(args)

    for seq in args.sequences:
        output_path = plot_sequence(seq, DEFAULT_RUNS[seq], args)
        print(output_path)


if __name__ == "__main__":
    main()
