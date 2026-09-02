"""Render comparable representative predictions for every PyTorch baseline.

For each protocol and degree, the same three balanced-condition geometries are
used across models: a typical IID case, a typical OOD case, and a challenging
OOD case.  Representatives are selected from the model/seed-averaged test
errors, so model comparisons are not confounded by different geometries.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pyvista as pv
import torch

from topobox3d.pde_dataset import TopoBoxPDEDataset
from topobox3d.run_hodge_heat_demo import (
    _add_boundary_surfaces,
    _axis_style,
    _farthest_seeds,
    _pyvista_boundary,
    _pyvista_volume_mesh,
    _vertex_vector_field,
)

from .metrics import sample_metrics
from .model_registry import (
    BFLOAT16_MODEL_NAMES,
    MODEL_OUTPUT_NAMES,
    TORCH_MODEL_NAMES,
    build_torch_model,
    forward_torch_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "runs" / "topobox3d"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "model_fit_visualizations"
GEOMETRY_ROOT = PROJECT_ROOT / "data" / "TopoBox-3D" / "packed"
SOLUTION_ROOT = PROJECT_ROOT / "data" / "TopoBox-3D-HodgeHeat"
PROTOCOLS = ("A", "B", "C", "D")
DEGREES = (0, 1, 2)
MODEL_RESULT_DIRS = {
    "mgn-lite": "MGN-lite",
    "rigno": "rigno",
    "transolver": "Transolver",
    "gnot": "GNOT",
    "gaot": "GAOT",
    "tno": "TNO",
}


def _jsonable_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "harmonic_coefficient_error"
    }


def _record_errors(protocol: str, degree: int, split: str) -> dict[str, list[float]]:
    by_geometry: dict[str, list[float]] = {}
    for result_dir in MODEL_RESULT_DIRS.values():
        base = RESULTS_ROOT / result_dir / f"protocol_{protocol}" / f"k{degree}"
        if not base.exists():
            continue
        for record_path in base.glob(f"seed_*/{split}.jsonl"):
            with record_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if record["config_name"] != "balanced":
                        continue
                    by_geometry.setdefault(record["geometry_id"], []).append(
                        float(record["relative_l2"])
                    )
    return by_geometry


def _quantile_representative(
    by_geometry: dict[str, list[float]], quantile: float
) -> tuple[str, float]:
    values = sorted(
        (
            geometry_id,
            float(np.mean(errors)),
        )
        for geometry_id, errors in by_geometry.items()
    )
    values.sort(key=lambda item: (item[1], item[0]))
    target = float(np.quantile([item[1] for item in values], quantile))
    return min(values, key=lambda item: abs(item[1] - target))


def build_selection_manifest() -> dict[str, dict[str, dict[str, object]]]:
    manifest: dict[str, dict[str, dict[str, object]]] = {}
    for protocol in PROTOCOLS:
        manifest[protocol] = {}
        for degree in DEGREES:
            iid = _record_errors(protocol, degree, "test_iid")
            ood = _record_errors(protocol, degree, "test_ood")
            iid_typical = _quantile_representative(iid, 0.50)
            ood_typical = _quantile_representative(ood, 0.50)
            ood_challenging = _quantile_representative(ood, 0.85)
            manifest[protocol][str(degree)] = {
                "iid_typical": {
                    "split": "test_iid",
                    "geometry_id": iid_typical[0],
                    "ensemble_mean_relative_l2": iid_typical[1],
                    "quantile": 0.50,
                },
                "ood_typical": {
                    "split": "test_ood",
                    "geometry_id": ood_typical[0],
                    "ensemble_mean_relative_l2": ood_typical[1],
                    "quantile": 0.50,
                },
                "ood_challenging": {
                    "split": "test_ood",
                    "geometry_id": ood_challenging[0],
                    "ensemble_mean_relative_l2": ood_challenging[1],
                    "quantile": 0.85,
                },
            }
    return manifest


def _load_selected_samples(
    protocol: str,
    degree: int,
    selection: dict[str, dict[str, object]],
) -> list[tuple[str, object]]:
    loaded = []
    datasets: dict[str, TopoBoxPDEDataset] = {}
    try:
        for label, selected in selection.items():
            split = str(selected["split"])
            if split not in datasets:
                datasets[split] = TopoBoxPDEDataset(
                    GEOMETRY_ROOT,
                    SOLUTION_ROOT,
                    protocol=protocol,
                    split=split,
                    degrees=(degree,),
                    configs=("balanced",),
                    cache_derived=True,
                    cache_adjacency=False,
                )
            dataset = datasets[split]
            geometry_id = str(selected["geometry_id"])
            index = next(
                index
                for index, item in enumerate(dataset.items)
                if item[1]["geometry_id"] == geometry_id
            )
            loaded.append((label, dataset[index]))
    finally:
        # Samples own NumPy views into HDF5 datasets, so materialize the fields
        # used after closing. Geometry arrays have already been read eagerly.
        for _, sample in loaded:
            sample.w0 = np.asarray(sample.w0).copy()
            sample.wT = np.asarray(sample.wT).copy()
            sample.mass = np.asarray(sample.mass).copy()
            sample.harmonic_basis = np.asarray(sample.harmonic_basis).copy()
        for dataset in datasets.values():
            dataset.close()
    return loaded


def _symmetric_norm(values: np.ndarray) -> colors.TwoSlopeNorm:
    limit = max(float(np.quantile(np.abs(values), 0.995)), 1e-8)
    return colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _select_indices(count: int, maximum: int = 9000) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _render_scalar(
    model_label: str,
    protocol: str,
    loaded: list[tuple],
    path: Path,
) -> None:
    figure = plt.figure(figsize=(14.5, 11.5), constrained_layout=True)
    for row, (case_label, sample, target, prediction, metrics) in enumerate(loaded):
        coordinates = sample.simplex_token_arrays[0] * np.asarray(
            [2.0, 1.0, 1.0], dtype=np.float32
        )
        error = prediction - target
        shared_norm = _symmetric_norm(np.concatenate((target, prediction)))
        error_norm = _symmetric_norm(error)
        indices = _select_indices(len(target))
        for column, (values, field_label, norm) in enumerate(
            (
                (target, "Target", shared_norm),
                (prediction, "Prediction", shared_norm),
                (error, "Error", error_norm),
            )
        ):
            axis = figure.add_subplot(3, 3, row * 3 + column + 1, projection="3d")
            scatter = axis.scatter(
                coordinates[indices, 0],
                coordinates[indices, 1],
                coordinates[indices, 2],
                c=values[indices],
                cmap="coolwarm",
                norm=norm,
                s=3.0,
                alpha=0.78,
                linewidths=0,
                rasterized=True,
            )
            _add_boundary_surfaces(
                axis, sample.geometry.data, outer_alpha_scale=0.45
            )
            title = (
                f"{case_label.replace('_', ' ').upper()} | {field_label}\n"
                f"{sample.geometry_id}"
            )
            if column == 2:
                title += f" | rel. L2={metrics['relative_l2']:.3f}"
            _axis_style(axis, title)
            figure.colorbar(scatter, ax=axis, shrink=0.56, pad=0.01)
    figure.suptitle(
        f"{model_label} | Protocol {protocol} | k=0 | balanced",
        fontsize=16,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_vector(
    model_label: str,
    protocol: str,
    degree: int,
    loaded: list[tuple],
    path: Path,
) -> None:
    plotter = pv.Plotter(
        shape=(3, 3),
        off_screen=True,
        window_size=(2100, 1660),
        border=False,
    )
    plotter.set_background("white")
    plotter.enable_depth_peeling()
    seed_count = 64 if degree == 1 else 96

    for row, (case_label, sample, target, prediction, metrics) in enumerate(loaded):
        geometry = sample.geometry.data
        volume = _pyvista_volume_mesh(geometry)
        boundaries = {
            bit: _pyvista_boundary(geometry, bit) for bit in (1, 2, 4)
        }
        error = prediction - target
        vectors = [
            _vertex_vector_field(degree, geometry, values)
            for values in (target, prediction, error)
        ]
        shared_magnitudes = np.concatenate(
            (
                np.linalg.norm(vectors[0], axis=1),
                np.linalg.norm(vectors[1], axis=1),
            )
        )
        shared_limit = max(float(np.quantile(shared_magnitudes, 0.99)), 1e-30)
        error_limit = max(
            float(np.quantile(np.linalg.norm(vectors[2], axis=1), 0.99)),
            1e-30,
        )
        for column, (field, field_label) in enumerate(
            zip(vectors, ("Target", "Prediction", "Error"))
        ):
            plotter.subplot(row, column)
            field_mesh = volume.copy()
            field_mesh.point_data["field"] = field
            magnitude = np.linalg.norm(field, axis=1)
            field_mesh.point_data["magnitude"] = magnitude
            centers = field_mesh.cell_centers()
            center_weights = np.asarray(centers.sample(field_mesh)["magnitude"])
            seeds = pv.PolyData(
                _farthest_seeds(centers.points, center_weights, seed_count)
            )
            limit = error_limit if column == 2 else shared_limit
            streamlines = field_mesh.streamlines_from_source(
                seeds,
                vectors="field",
                integration_direction="both",
                integrator_type=45,
                initial_step_length=0.025,
                max_step_length=0.06,
                max_time=4.5,
                terminal_speed=limit * 1e-5,
                compute_vorticity=False,
            )
            if streamlines.n_points:
                scalar_name = "error magnitude" if column == 2 else "magnitude"
                streamlines[scalar_name] = np.linalg.norm(
                    np.asarray(streamlines["field"]), axis=1
                )
                plotter.add_mesh(
                    streamlines.tube(radius=0.0038, n_sides=7),
                    scalars=scalar_name,
                    cmap="magma" if column == 2 else "viridis",
                    clim=(0.0, limit),
                    smooth_shading=True,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        "title": "",
                        "n_labels": 3,
                        "label_font_size": 8,
                        "fmt": "%.2g",
                    },
                )
            for bit, color, opacity in (
                (1, "#aeb6bf", 0.055),
                (2, "#e67e22", 0.22),
                (4, "#27ae60", 0.30),
            ):
                if boundaries[bit].n_cells:
                    plotter.add_mesh(
                        boundaries[bit],
                        color=color,
                        opacity=opacity,
                        show_edges=False,
                    )
            title = (
                f"{case_label.replace('_', ' ').upper()} | {field_label}\n"
                f"{sample.geometry_id}"
            )
            if column == 2:
                title += f" | rel. L2={metrics['relative_l2']:.3f}"
            plotter.add_text(
                title,
                position="upper_left",
                font_size=11,
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


@torch.inference_mode()
def render_combination(
    model_name: str,
    protocol: str,
    degree: int,
    selection: dict[str, dict[str, object]],
    output_root: Path,
    seed: int,
) -> dict[str, object]:
    model_label = MODEL_OUTPUT_NAMES[model_name]
    run_dir = (
        RESULTS_ROOT
        / MODEL_RESULT_DIRS[model_name]
        / f"protocol_{protocol}"
        / f"k{degree}"
        / f"seed_{seed}"
    )
    checkpoint = torch.load(
        run_dir / "best.pt", map_location="cuda", weights_only=False
    )
    model = build_torch_model(model_name, degree).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    selected_samples = _load_selected_samples(protocol, degree, selection)
    loaded = []
    records = {}
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if model_name in BFLOAT16_MODEL_NAMES
        else nullcontext()
    )
    for case_label, sample in selected_samples:
        batch = sample.for_model(model_name)
        with autocast:
            prediction = forward_torch_model(
                model_name, model, batch, torch.device("cuda")
            )
        prediction_np = sample.prediction_to_cochain(prediction)
        target_np = np.asarray(sample.wT)
        metrics = sample_metrics(
            prediction,
            batch["target_cochain"].cuda(),
            batch["mass"].cuda(),
            batch["target_harmonic_basis"].cuda(),
        )
        loaded.append(
            (case_label, sample, target_np, prediction_np, metrics)
        )
        records[case_label] = {
            **selection[case_label],
            "beta1": sample.beta1,
            "beta2": sample.beta2,
            "metrics": _jsonable_metrics(metrics),
        }

    output_dir = (
        output_root / model_label / f"protocol_{protocol}" / f"k{degree}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "representative_fits.png"
    if degree == 0:
        _render_scalar(model_label, protocol, loaded, image_path)
    else:
        _render_vector(model_label, protocol, degree, loaded, image_path)
    result = {
        "model": model_label,
        "protocol": protocol,
        "degree": degree,
        "seed": seed,
        "image": str(image_path),
        "cases": records,
    }
    (output_dir / "representative_fits.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TORCH_MODEL_NAMES,
        default=list(TORCH_MODEL_NAMES),
    )
    parser.add_argument(
        "--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS)
    )
    parser.add_argument(
        "--degrees", nargs="+", type=int, choices=DEGREES, default=list(DEGREES)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "representative_selection.json"
    if manifest_path.exists():
        selection_manifest = json.loads(manifest_path.read_text("utf-8"))
    else:
        selection_manifest = build_selection_manifest()
        manifest_path.write_text(
            json.dumps(selection_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    completed = []
    for model_name in args.models:
        for protocol in args.protocols:
            for degree in args.degrees:
                output_path = (
                    args.output
                    / MODEL_OUTPUT_NAMES[model_name]
                    / f"protocol_{protocol}"
                    / f"k{degree}"
                    / "representative_fits.png"
                )
                if args.skip_existing and output_path.exists():
                    continue
                result = render_combination(
                    model_name,
                    protocol,
                    degree,
                    selection_manifest[protocol][str(degree)],
                    args.output,
                    args.seed,
                )
                completed.append(result)
                print(
                    json.dumps(
                        {
                            "model": result["model"],
                            "protocol": protocol,
                            "degree": degree,
                            "image": result["image"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    (args.output / "torch_visualization_index.json").write_text(
        json.dumps(completed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
