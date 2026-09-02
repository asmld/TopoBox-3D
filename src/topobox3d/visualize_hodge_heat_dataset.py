"""Select representative Betti types and render TopoBox-3D Hodge-heat results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pyvista as pv

from .hodge_heat import load_geometry
from .run_hodge_heat_demo import (
    BOUNDARY_STYLES,
    _add_boundary_surfaces,
    _axis_style,
    _field_line_figure,
    _pyvista_boundary,
)


TARGET_PAIRS = (
    (0, 0), (1, 0), (0, 1),
    (1, 1), (2, 0), (0, 2),
    (3, 0), (0, 3), (3, 3),
)
FIELD_LINE_PAIRS = ((0, 0), (1, 0), (0, 1), (1, 1), (3, 3))


def _select(records: list[dict]) -> dict[tuple[int, int], dict]:
    selected = {}
    split_priority = {"test_ood": 0, "test_iid": 1, "validation": 2, "train": 3}
    records = sorted(
        records,
        key=lambda item: (
            split_priority.get(item["split"], 9),
            item["geometry_id"],
        ),
    )
    for pair in TARGET_PAIRS:
        selected[pair] = next(
            (
                record for record in records
                if (record["beta1"], record["beta2"]) == pair
            ),
            None,
        )
    return {pair: record for pair, record in selected.items() if record}


def _sample_dir(geometry_root: Path, record: dict) -> Path:
    return (
        geometry_root
        / f"protocol_{record['protocol']}"
        / record["split"]
        / record["geometry_id"]
    )


def _geometry_montage(
    geometry_root: Path,
    selected: dict[tuple[int, int], dict],
    path: Path,
) -> None:
    plotter = pv.Plotter(
        shape=(3, 3),
        off_screen=True,
        window_size=(2100, 1350),
        border=False,
    )
    plotter.set_background("white")
    plotter.enable_depth_peeling()
    for index, pair in enumerate(TARGET_PAIRS):
        plotter.subplot(index // 3, index % 3)
        record = selected.get(pair)
        if record is None:
            plotter.add_text(
                rf"$\beta_1={pair[0]},\ \beta_2={pair[1]}$ (unavailable)",
                font_size=13,
                color="#263238",
            )
            continue
        geometry, _ = load_geometry(_sample_dir(geometry_root, record))
        for bit, (_, color, _) in BOUNDARY_STYLES.items():
            opacity = {1: 0.08, 2: 0.72, 4: 0.78}[bit]
            boundary = _pyvista_boundary(geometry, bit)
            if boundary.n_cells:
                plotter.add_mesh(
                    boundary, color=color, opacity=opacity, show_edges=False
                )
        plotter.add_text(
            rf"$\beta_1={pair[0]},\ \beta_2={pair[1]}$",
            position="upper_left",
            font_size=14,
            color="#263238",
        )
        plotter.camera_position = [
            (3.55, -3.35, 2.65),
            (1.0, 0.5, 0.48),
            (0.0, 0.0, 1.0),
        ]
        plotter.camera.zoom(1.12)
    plotter.screenshot(path)
    plotter.close()


def _read_fields(
    solution_root: Path,
    record: dict,
    config_index: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    with h5py.File(solution_root / record["shard"], "r") as handle:
        group = handle[record["group"]]
        initials = {
            degree: np.asarray(group[f"k{degree}/w0"][config_index])
            for degree in range(3)
        }
        finals = {
            degree: np.asarray(group[f"k{degree}/wT"][config_index])
            for degree in range(3)
        }
    return initials, finals


def _k0_scalar_montage(
    geometry_root: Path,
    solution_root: Path,
    selected: dict[tuple[int, int], dict],
    config_index: int,
    path: Path,
) -> None:
    representatives = [
        (pair, selected[pair])
        for pair in FIELD_LINE_PAIRS if pair in selected
    ]
    loaded = []
    global_limit = 0.0
    for pair, record in representatives:
        geometry, _ = load_geometry(_sample_dir(geometry_root, record))
        initials, finals = _read_fields(solution_root, record, config_index)
        values = (initials[0], finals[0])
        global_limit = max(
            global_limit,
            float(np.max(np.abs(np.concatenate(values)))),
        )
        loaded.append((pair, geometry, values))
    if not loaded:
        return

    figure = plt.figure(
        figsize=(4.0 * len(loaded), 7.4),
        constrained_layout=True,
    )
    color_norm = colors.TwoSlopeNorm(
        vmin=-global_limit, vcenter=0.0, vmax=global_limit
    )
    scalar_map = plt.cm.ScalarMappable(norm=color_norm, cmap="coolwarm")
    axes = []
    for column, (pair, geometry, values) in enumerate(loaded):
        points = geometry["points"].astype(np.float64)
        for row, (field, time_label) in enumerate(zip(values, (r"$\omega_0$", r"$\omega_T$"))):
            axis = figure.add_subplot(
                2, len(loaded), row * len(loaded) + column + 1,
                projection="3d",
            )
            axes.append(axis)
            axis.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=field, cmap="coolwarm", norm=color_norm,
                s=3.2, alpha=0.74, linewidths=0,
            )
            _add_boundary_surfaces(axis, geometry)
            _axis_style(
                axis,
                rf"$\beta_1={pair[0]},\ \beta_2={pair[1]}$: {time_label}",
            )
    figure.colorbar(
        scalar_map,
        ax=axes,
        location="right",
        shrink=0.64,
        pad=0.015,
        label=r"scalar value $\omega$",
    )
    figure.savefig(path, dpi=190)
    plt.close(figure)


def render(
    geometry_root: Path,
    solution_root: Path,
    output_root: Path,
    config_name: str,
) -> dict:
    records = json.loads(
        (solution_root / "index.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (solution_root / "manifest.json").read_text(encoding="utf-8")
    )
    config_index = manifest["config_names"].index(config_name)
    selected = _select(records)
    output_root.mkdir(parents=True, exist_ok=True)
    _geometry_montage(
        geometry_root, selected, output_root / "representative_topologies.png"
    )
    _k0_scalar_montage(
        geometry_root,
        solution_root,
        selected,
        config_index,
        output_root / f"{config_name}_k0_scalar_fields.png",
    )
    rendered = {}
    for pair in FIELD_LINE_PAIRS:
        record = selected.get(pair)
        if record is None:
            continue
        sample_dir = _sample_dir(geometry_root, record)
        geometry, _ = load_geometry(sample_dir)
        initials, finals = _read_fields(solution_root, record, config_index)
        sample_output = output_root / record["geometry_id"]
        sample_output.mkdir(parents=True, exist_ok=True)
        path = sample_output / f"{config_name}_field_lines.png"
        _field_line_figure(
            geometry, initials, finals, path, line_count=64
        )
        rendered[f"{pair[0]},{pair[1]}"] = {
            "geometry_id": record["geometry_id"],
            "protocol": record["protocol"],
            "split": record["split"],
            "image": str(path),
        }
    summary = {
        "config_name": config_name,
        "selected": {
            f"{pair[0]},{pair[1]}": record
            for pair, record in selected.items()
        },
        "rendered_field_lines": rendered,
    }
    (output_root / "representatives.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-root", type=Path, default=Path("data/TopoBox-3D")
    )
    parser.add_argument(
        "--solution-root", type=Path, default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/TopoBox-3D-HodgeHeat-representatives"),
    )
    parser.add_argument("--config", default="balanced")
    args = parser.parse_args()
    summary = render(
        args.geometry_root, args.solution_root, args.output, args.config
    )
    print(
        f"Rendered {len(summary['rendered_field_lines'])} field-line representatives "
        f"to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
