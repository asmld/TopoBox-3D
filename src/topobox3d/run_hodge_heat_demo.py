"""Run and visualize one TopoBox-3D Hodge-heat example for k=0,1,2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv

from .hodge_heat import (
    build_hodge_systems,
    generate_initial_condition,
    harmonic_basis,
    load_geometry,
    smallest_generalized_eigenvalues,
    solve_crank_nicolson,
    validate_system,
)


def _vector_proxy(degree: int, geometry: dict[str, np.ndarray], cochain: np.ndarray):
    points = geometry["points"].astype(np.float64)
    if degree == 1:
        simplices = geometry["edges"].astype(np.int64)
        centers = points[simplices].mean(axis=1)
        measure_vector = geometry["edge_vectors"].astype(np.float64)
    elif degree == 2:
        simplices = geometry["faces"].astype(np.int64)
        centers = points[simplices].mean(axis=1)
        measure_vector = geometry["face_area_vectors"].astype(np.float64)
    else:
        raise ValueError("Vector proxies are only defined for k=1,2")
    denominator = np.einsum("ij,ij->i", measure_vector, measure_vector)
    vectors = cochain[:, None] * measure_vector / denominator[:, None]
    return centers, vectors


def _vertex_vector_field(
    degree: int,
    geometry: dict[str, np.ndarray],
    cochain: np.ndarray,
) -> np.ndarray:
    """Least-squares reconstruction of a cochain as a vertex vector field."""
    points = geometry["points"].astype(np.float64)
    if degree == 1:
        simplices = geometry["edges"].astype(np.int64)
        measure_vectors = geometry["edge_vectors"].astype(np.float64)
    elif degree == 2:
        simplices = geometry["faces"].astype(np.int64)
        measure_vectors = geometry["face_area_vectors"].astype(np.float64)
    else:
        raise ValueError("Vector reconstruction is only defined for k=1,2")

    incident: list[list[int]] = [[] for _ in range(len(points))]
    for simplex_index, simplex in enumerate(simplices):
        for vertex in simplex:
            incident[int(vertex)].append(simplex_index)

    result = np.zeros((len(points), 3), dtype=np.float64)
    for vertex, indices in enumerate(incident):
        if not indices:
            continue
        selected = np.asarray(indices, dtype=np.int64)
        matrix = measure_vectors[selected]
        rhs = cochain[selected]
        # A small isotropic ridge makes boundary/corner reconstructions stable.
        gram = matrix.T @ matrix
        ridge = max(float(np.trace(gram)) * 1e-8, 1e-14)
        result[vertex] = np.linalg.solve(
            gram + ridge * np.eye(3),
            matrix.T @ rhs,
        )
    return result


def _pyvista_volume_mesh(geometry: dict[str, np.ndarray]) -> pv.UnstructuredGrid:
    points = geometry["points"].astype(np.float64)
    tetra = geometry["oriented_tetra"].astype(np.int64)
    cells = np.column_stack((np.full(len(tetra), 4, dtype=np.int64), tetra)).ravel()
    cell_types = np.full(len(tetra), pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, points)


def _pyvista_boundary(
    geometry: dict[str, np.ndarray],
    mask_bit: int,
) -> pv.PolyData:
    points = geometry["points"].astype(np.float64)
    triangles = geometry["boundary_triangles"].astype(np.int64)
    masks = geometry["boundary_triangle_mask"].astype(np.uint8)
    selected = triangles[masks == mask_bit]
    faces = np.column_stack(
        (np.full(len(selected), 3, dtype=np.int64), selected)
    ).ravel()
    return pv.PolyData(points, faces)


def _farthest_seeds(
    points: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> np.ndarray:
    """Magnitude-aware, spatially spread streamline seed selection."""
    count = min(count, len(points))
    if count == 0:
        return points[:0]
    normalized = weights / max(float(np.max(weights)), 1e-30)
    chosen = [int(np.argmax(normalized))]
    distance2 = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    span2 = max(float(np.sum(np.ptp(points, axis=0) ** 2)), 1e-30)
    for _ in range(1, count):
        score = np.sqrt(distance2 / span2) * (0.30 + 0.70 * normalized)
        score[np.asarray(chosen)] = -1.0
        next_index = int(np.argmax(score))
        chosen.append(next_index)
        distance2 = np.minimum(
            distance2,
            np.sum((points - points[next_index]) ** 2, axis=1),
        )
    return points[np.asarray(chosen)]


def _field_line_figure(
    geometry: dict[str, np.ndarray],
    initials: dict[int, np.ndarray],
    finals: dict[int, np.ndarray],
    path: Path,
    line_count: int = 64,
) -> None:
    """Render k=1,2 using continuous streamlines / magnetic field lines."""
    volume = _pyvista_volume_mesh(geometry)
    boundary = {
        bit: _pyvista_boundary(geometry, bit)
        for bit in BOUNDARY_STYLES
    }
    plotter = pv.Plotter(
        shape=(2, 2),
        off_screen=True,
        window_size=(1800, 1220),
        border=False,
    )
    plotter.set_background("white")
    plotter.enable_depth_peeling()

    for row, degree in enumerate((1, 2)):
        fields = (
            initials[degree],
            finals[degree],
        )
        reconstructed = [
            _vertex_vector_field(degree, geometry, values) for values in fields
        ]
        shared_limit = float(np.max(np.concatenate([
            np.linalg.norm(reconstructed[0], axis=1),
            np.linalg.norm(reconstructed[1], axis=1),
        ])))
        titles = (r"$\omega_0$", r"$\omega_T$")

        for column, (vectors, title) in enumerate(zip(reconstructed, titles)):
            plotter.subplot(row, column)
            field_mesh = volume.copy()
            field_mesh.point_data["field"] = vectors
            magnitude = np.linalg.norm(vectors, axis=1)
            field_mesh.point_data["magnitude"] = magnitude
            centers = field_mesh.cell_centers()
            center_weights = centers.sample(field_mesh)["magnitude"]
            seeds = pv.PolyData(
                _farthest_seeds(
                    centers.points,
                    np.asarray(center_weights),
                    line_count if degree == 1 else int(round(1.5 * line_count)),
                )
            )
            streamlines = field_mesh.streamlines_from_source(
                seeds,
                vectors="field",
                integration_direction="both",
                integrator_type=45,
                initial_step_length=0.025,
                max_step_length=0.06,
                max_length=4.5,
                terminal_speed=shared_limit * 1e-5,
                compute_vorticity=False,
            )
            limit = shared_limit
            cmap = "viridis"
            if streamlines.n_points:
                speed = np.linalg.norm(np.asarray(streamlines["field"]), axis=1)
                scalar_name = f"absolute magnitude k={degree}"
                streamlines[scalar_name] = speed
                tubes = streamlines.tube(radius=0.0038, n_sides=7)
                actor = plotter.add_mesh(
                    tubes,
                    scalars=scalar_name,
                    cmap=cmap,
                    clim=(0.0, max(limit, 1e-30)),
                    smooth_shading=True,
                    show_scalar_bar=False,
                )
                if column == 1:
                    plotter.add_scalar_bar(
                        title=f"k={degree} magnitude",
                        mapper=actor.mapper,
                        vertical=True,
                        width=0.024,
                        height=0.36,
                        position_x=0.90,
                        position_y=0.10,
                        title_font_size=10,
                        label_font_size=10,
                        n_labels=3,
                    )
            plotter.add_mesh(
                boundary[1], color="#aeb6bf", opacity=0.055,
                show_edges=False,
            )
            plotter.add_mesh(
                boundary[2], color="#e67e22", opacity=0.22,
                show_edges=False,
            )
            plotter.add_mesh(
                boundary[4], color="#27ae60", opacity=0.30,
                show_edges=False,
            )
            field_name = "flow lines" if degree == 1 else "magnetic field lines"
            plotter.add_text(
                f"k={degree}  {field_name}: {title}",
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


def _axis_style(axis, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.set_box_aspect((2, 1, 1))
    axis.view_init(elev=22, azim=-58)


BOUNDARY_STYLES = {
    1: ("outer wall", "#7f8c8d", 0.035),
    2: ("through-tunnel wall", "#e67e22", 0.34),
    4: ("closed-cavity wall", "#27ae60", 0.48),
}


def _add_boundary_surfaces(axis, geometry: dict[str, np.ndarray], outer_alpha_scale: float = 1.0):
    points = geometry["points"].astype(np.float64)
    triangles = geometry["boundary_triangles"].astype(np.int64)
    masks = geometry["boundary_triangle_mask"].astype(np.uint8)
    for bit, (_, color, alpha) in BOUNDARY_STYLES.items():
        selected = triangles[masks == bit]
        if not len(selected):
            continue
        actual_alpha = alpha * outer_alpha_scale if bit == 1 else alpha
        collection = Poly3DCollection(
            points[selected],
            facecolor=color,
            edgecolor=color if bit != 1 else "none",
            linewidth=0.12 if bit != 1 else 0.0,
            alpha=actual_alpha,
        )
        collection.set_rasterized(True)
        axis.add_collection3d(collection)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])


def _boundary_legend_handles():
    return [
        Patch(facecolor=color, edgecolor=color, alpha=max(alpha, 0.25), label=label)
        for label, color, alpha in BOUNDARY_STYLES.values()
    ]


def _geometry_figure(geometry: dict[str, np.ndarray], path: Path) -> None:
    figure = plt.figure(figsize=(12.5, 5.2), constrained_layout=True)
    views = ((23, -58, "front view"), (18, 128, "reverse view"))
    for index, (elevation, azimuth, title) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        _add_boundary_surfaces(axis, geometry, outer_alpha_scale=1.8)
        _axis_style(axis, title)
        axis.view_init(elev=elevation, azim=azimuth)
        if index == 1:
            axis.legend(handles=_boundary_legend_handles(), loc="upper left", frameon=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _field_figure(
    geometry: dict[str, np.ndarray],
    initials: dict[int, np.ndarray],
    finals: dict[int, np.ndarray],
    path: Path,
    masses: dict[int, np.ndarray] | None = None,
    arrow_count: int = 180,
) -> None:
    points = geometry["points"].astype(np.float64)
    figure = plt.figure(figsize=(18.5, 12), constrained_layout=True)
    rng = np.random.default_rng(20260722)

    for degree in range(3):
        initial = initials[degree]
        final = finals[degree]
        difference = final - initial
        fields = (initial, final, difference)
        if masses is not None:
            mass = masses[degree]
            initial_norm = np.sqrt(np.dot(initial * mass, initial))
            relative_change = np.sqrt(np.dot(difference * mass, difference)) / initial_norm
            difference_title = f"difference (relative M change={relative_change:.1%})"
        else:
            difference_title = "difference"
        if degree == 0:
            shared_limit = float(np.quantile(np.abs(np.concatenate((initial, final))), 0.99))
            difference_limit = float(np.quantile(np.abs(difference), 0.99))
            titles = ("w0", "wT", difference_title)
            for column, (values, title) in enumerate(zip(fields, titles)):
                axis = figure.add_subplot(3, 3, 3 * degree + column + 1, projection="3d")
                limit = difference_limit if column == 2 else shared_limit
                color_norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
                scatter = axis.scatter(
                    points[:, 0], points[:, 1], points[:, 2], c=values,
                    cmap="coolwarm", norm=color_norm, s=4, alpha=0.78, linewidths=0,
                )
                _add_boundary_surfaces(axis, geometry)
                figure.colorbar(scatter, ax=axis, shrink=0.58, pad=0.02, label="value")
                _axis_style(axis, f"k=0: {title}")
        else:
            proxies = [_vector_proxy(degree, geometry, values) for values in fields]
            shared_magnitudes = np.concatenate([
                np.linalg.norm(proxies[0][1], axis=1),
                np.linalg.norm(proxies[1][1], axis=1),
            ])
            shared_limit = float(np.quantile(shared_magnitudes, 0.97))
            difference_magnitude = np.linalg.norm(proxies[2][1], axis=1)
            difference_limit = float(np.quantile(difference_magnitude, 0.97))
            sample_count = min(arrow_count, len(proxies[0][0]))
            important_count = min(int(round(0.50 * sample_count)), len(difference_magnitude))
            important = np.argpartition(difference_magnitude, -important_count)[-important_count:]
            remaining = np.setdiff1d(np.arange(len(difference_magnitude)), important, assume_unique=False)
            random_count = min(sample_count - important_count, len(remaining))
            random_selected = rng.choice(remaining, size=random_count, replace=False)
            selected = np.sort(np.concatenate((important, random_selected)))
            titles = ("w0", "wT", difference_title)
            for column, ((centers, vectors), title) in enumerate(zip(proxies, titles)):
                axis = figure.add_subplot(3, 3, 3 * degree + column + 1, projection="3d")
                chosen_centers = centers[selected]
                chosen_vectors = vectors[selected]
                magnitude = np.linalg.norm(chosen_vectors, axis=1)
                safe = np.maximum(magnitude, 1e-30)
                direction = chosen_vectors / safe[:, None]
                limit = difference_limit if column == 2 else shared_limit
                arrow_length = 0.22 * np.minimum(magnitude / max(limit, 1e-30), 1.0)
                scaled = direction * arrow_length[:, None]
                cmap = plt.cm.magma if column == 2 else plt.cm.viridis
                axis.quiver(
                    chosen_centers[:, 0], chosen_centers[:, 1], chosen_centers[:, 2],
                    scaled[:, 0], scaled[:, 1], scaled[:, 2],
                    colors=cmap(np.clip(magnitude / max(limit, 1e-30), 0, 1)),
                    linewidth=0.72, arrow_length_ratio=0.30,
                )
                _add_boundary_surfaces(axis, geometry)
                scalar_map = plt.cm.ScalarMappable(
                    norm=colors.Normalize(0.0, limit), cmap=cmap
                )
                figure.colorbar(
                    scalar_map, ax=axis, shrink=0.58, pad=0.02,
                    label="proxy magnitude",
                )
                _axis_style(axis, f"k={degree}: {title}")
                if degree == 1 and column == 0:
                    axis.legend(handles=_boundary_legend_handles()[1:], loc="upper right", frameon=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _decay_figure(solutions: dict[int, object], final_time: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    markers = ("o", "s", "^")
    for degree in range(3):
        solution = solutions[degree]
        axis.plot(
            solution.times, solution.relative_mass_norm,
            marker=markers[degree], markersize=4, linewidth=1.8, label=f"k={degree}",
        )
    axis.set_xlim(0.0, final_time)
    axis.set_ylim(bottom=0.0, top=1.03)
    axis.set_xlabel("time")
    axis.set_ylabel(r"$\|\omega(t)\|_{M_k}/\|\omega_0\|_{M_k}$")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    sample_dir = Path(args.sample).resolve()
    output_dir = Path(args.output).resolve() if args.output else (
        Path("outputs") / "hodge_heat_demo" / sample_dir.name
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry, metadata = load_geometry(sample_dir)
    systems, derivatives, masses = build_hodge_systems(geometry, metadata)
    initials = {}
    solutions = {}
    diagnostics: dict[str, object] = {
        "geometry_id": metadata["geometry_id"],
        "sample_dir": str(sample_dir),
        "kappa": args.kappa,
        "final_time": args.final_time,
        "diffusion_length_sqrt_2kT": float(np.sqrt(2.0 * args.kappa * args.final_time)),
        "steps": args.steps,
        "initial_filter_time": args.initial_filter_time,
        "initial_filter_passes": args.initial_filter_passes,
        "initial_preset": args.initial_preset,
        "rbf_centers": args.rbf_centers,
        "mass_discretization": "positive diagonal Whitney mass; vertex-lumped M0",
        "boundary_condition": "homogeneous absolute (natural full-cochain weak form)",
        "degrees": {},
    }

    saved = {}
    scalar_harmonic_fraction = 0.0
    if args.initial_preset == "paper":
        energy_fractions = (0.48, 0.48, 0.04)
    elif args.initial_preset == "strong-harmonic-paper":
        energy_fractions = (0.10, 0.10, 0.80)
        scalar_harmonic_fraction = 0.80
    else:
        energy_fractions = (1 / 3, 1 / 3, 1 / 3)
    for degree, system in enumerate(systems):
        stored_guess = geometry.get(f"harmonic_basis_{degree}")
        basis = harmonic_basis(system, stored_guess)
        initial = generate_initial_condition(
            degree,
            geometry,
            system,
            derivatives,
            masses[degree + 1] if degree < 3 else None,
            basis,
            seed=args.seed,
            correlation_length=args.correlation_length,
            energy_fractions=energy_fractions,
            filter_time=args.initial_filter_time,
            filter_passes=args.initial_filter_passes,
            rbf_centers=args.rbf_centers,
            scalar_harmonic_fraction=scalar_harmonic_fraction,
        )
        solution = solve_crank_nicolson(
            system,
            initial.values,
            final_time=args.final_time,
            steps=args.steps,
            kappa=args.kappa,
            snapshots=args.snapshots,
        )
        low_eigenvalues = smallest_generalized_eigenvalues(system)
        system_checks = validate_system(system)
        harmonic_residual = (
            float(np.max(np.linalg.norm(system.stiffness @ basis, axis=0), initial=0.0))
            if basis.shape[1] else 0.0
        )
        degree_diagnostics = {
            **system_checks,
            "dimension": system.dimension,
            "expected_nullity": system.nullity,
            "computed_harmonic_vectors": int(basis.shape[1]),
            "harmonic_stiffness_residual": harmonic_residual,
            "initial_mass_norm": system.mass_norm(initial.values),
            "final_mass_norm": system.mass_norm(solution.final),
            "relative_final_mass_norm": float(solution.relative_mass_norm[-1]),
            "requested_energy_fractions": initial.requested_energy_fractions,
            "realized_energy_fractions": initial.realized_energy_fractions,
            "lowest_positive_generalized_eigenvalues": low_eigenvalues.tolist(),
        }
        diagnostics["degrees"][str(degree)] = degree_diagnostics
        initials[degree] = initial.values
        solutions[degree] = solution
        saved[f"w0_k{degree}"] = initial.values.astype(np.float32)
        saved[f"wT_k{degree}"] = solution.final.astype(np.float32)
        saved[f"mass_k{degree}"] = system.mass.astype(np.float32)
        saved[f"times_k{degree}"] = solution.times.astype(np.float32)
        saved[f"relative_mass_norm_k{degree}"] = solution.relative_mass_norm.astype(np.float32)

    np.savez_compressed(output_dir / "solution.npz", **saved)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _field_figure(
        geometry,
        initials,
        {degree: solutions[degree].final for degree in range(3)},
        output_dir / "fields_w0_wT.png",
        {degree: systems[degree].mass for degree in range(3)},
        args.arrow_count,
    )
    _field_line_figure(
        geometry,
        initials,
        {degree: solutions[degree].final for degree in range(3)},
        output_dir / "field_lines_w0_wT.png",
        args.line_count,
    )
    _geometry_figure(geometry, output_dir / "boundary_surfaces.png")
    _decay_figure(solutions, args.final_time, output_dir / "mass_norm_decay.png")
    return output_dir


def parser() -> argparse.ArgumentParser:
    default_sample = (
        Path("data") / "TopoBox-3D-mini" / "protocol_A" / "train" / "PA_train_0000_b11"
    )
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sample", default=str(default_sample), help="Raw TopoBox-3D sample directory")
    result.add_argument("--output", default=None, help="Output directory")
    result.add_argument("--kappa", type=float, default=1.0)
    result.add_argument("--final-time", type=float, default=0.1)
    result.add_argument("--steps", type=int, default=80)
    result.add_argument("--snapshots", type=int, default=17)
    result.add_argument("--seed", type=int, default=20260722)
    result.add_argument("--correlation-length", type=float, default=0.22)
    result.add_argument("--initial-filter-time", type=float, default=0.03)
    result.add_argument("--initial-filter-passes", type=int, default=2)
    result.add_argument(
        "--initial-preset",
        choices=("balanced", "paper", "strong-harmonic-paper"),
        default="balanced",
    )
    result.add_argument("--rbf-centers", type=int, default=24)
    result.add_argument("--arrow-count", type=int, default=180)
    result.add_argument("--line-count", type=int, default=64)
    return result


if __name__ == "__main__":
    destination = run(parser().parse_args())
    print(destination)
