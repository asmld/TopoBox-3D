"""Render mesh previews and protocol summary plots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def _boundary_surfaces(vtu_path: Path) -> dict[str, pv.PolyData]:
    npz_path = vtu_path.with_suffix(".npz")
    with np.load(npz_path) as data:
        points = data["points"]
        triangles = data["boundary_triangles"]
        masks = data["boundary_triangle_mask"]
    surfaces: dict[str, pv.PolyData] = {}
    for name, bit in (("outer", 1), ("tunnel", 2), ("cavity", 4)):
        selected = triangles[(masks & bit) != 0]
        if len(selected):
            faces = np.column_stack((np.full(len(selected), 3), selected)).ravel()
            surfaces[name] = pv.PolyData(points, faces)
    return surfaces


def _add_geometry(plotter: pv.Plotter, vtu_path: Path, show_edges: bool) -> None:
    surfaces = _boundary_surfaces(vtu_path)
    if "outer" in surfaces:
        plotter.add_mesh(surfaces["outer"], color="#4C78A8", opacity=0.16, show_edges=False)
    if "tunnel" in surfaces:
        plotter.add_mesh(
            surfaces["tunnel"], color="#F58518", opacity=1.0, show_edges=show_edges,
            edge_color="#5A3510", line_width=0.35, label="Tunnel wall",
        )
    if "cavity" in surfaces:
        plotter.add_mesh(
            surfaces["cavity"], color="#54A24B", opacity=1.0, show_edges=show_edges,
            edge_color="#1D4E25", line_width=0.35, label="Cavity wall",
        )


def render_sample(vtu_path: Path, output_png: Path, show_edges: bool = True) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 700))
    plotter.set_background("white")
    plotter.enable_depth_peeling()
    _add_geometry(plotter, vtu_path, show_edges)
    plotter.add_axes()
    plotter.add_legend(face=None, bcolor=None)
    plotter.view_isometric()
    plotter.camera.zoom(1.25)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(output_png)
    plotter.close()


def render_protocol_splits(dataset_root: Path, protocol: str, output_png: Path) -> None:
    split_labels = (
        ("train", "Train"),
        ("validation", "Validation"),
        ("test_iid", "Test-IID"),
        ("test_ood", "Test-OOD"),
    )
    selected: list[tuple[str, Path]] = []
    for split, label in split_labels:
        candidates = sorted((dataset_root / f"protocol_{protocol}" / split).glob("*/mesh.vtu"))
        if not candidates:
            raise FileNotFoundError(f"No mesh.vtu found for Protocol {protocol} {split}")
        described = []
        for candidate in candidates:
            metadata = json.loads((candidate.parent / "metadata.json").read_text(encoding="utf-8"))
            described.append((metadata["beta1"] + metadata["beta2"], metadata["beta1"], metadata["beta2"], candidate))
        _, beta1, beta2, candidate = max(described, key=lambda item: item[:3])
        selected.append((f"{label}  (b1={beta1}, b2={beta2})", candidate))
    plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1500, 1000))
    plotter.set_background("white")
    plotter.enable_depth_peeling()
    for idx, (title, path) in enumerate(selected):
        plotter.subplot(idx // 2, idx % 2)
        plotter.add_text(title, font_size=12, color="black")
        _add_geometry(plotter, path, show_edges=True)
        plotter.view_isometric()
        plotter.camera.zoom(1.2)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(output_png)
    plotter.close()


def plot_dataset_summary(dataset_root: Path, output_png: Path) -> None:
    with (dataset_root / "manifest.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    vertices = np.array([int(row["n_vertices"]) for row in rows])
    quality = np.array([float(row["quality_min"]) for row in rows])
    clearance = np.array([float(row["min_clearance"]) for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].hist(vertices, bins=min(15, max(3, len(vertices) // 4)))
    axes[0].set(xlabel="Vertices per mesh", ylabel="Count")
    axes[1].hist(quality, bins=min(15, max(3, len(quality) // 4)))
    axes[1].set(xlabel="Minimum mean-ratio quality")
    axes[2].hist(clearance, bins=min(15, max(3, len(clearance) // 4)))
    axes[2].axvline(0.10, color="red", linestyle="--", linewidth=1)
    axes[2].set(xlabel="Minimum primitive clearance")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--sample", type=Path, help="A mesh.vtu file to render")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.dataset_root / "visualizations"
    if args.sample:
        render_sample(args.sample, output / f"{args.sample.parent.name}.png")
    for protocol in "ABCD":
        render_protocol_splits(args.dataset_root, protocol, output / f"protocol_{protocol}_splits.png")
    plot_dataset_summary(args.dataset_root, output / "dataset_summary.png")


if __name__ == "__main__":
    main()
