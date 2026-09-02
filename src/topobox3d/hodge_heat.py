"""Prototype Hodge-heat data generation on TopoBox-3D tetrahedral meshes.

The implementation uses the full cochain complex, so homogeneous absolute
boundary conditions are the natural boundary conditions of the discrete weak
problem.  Positive diagonal mass matrices are assembled from the diagonal of
the Whitney-form mass matrices (with the usual vertex-lumped scalar mass).

This module is intentionally a single-geometry reference implementation.  It
keeps the numerical choices explicit so that a later production generator can
replace the diagonal masses with consistent FEEC masses without changing the
dataset-facing API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass(frozen=True)
class HodgeSystem:
    """Mass and stiffness matrices for one cochain degree."""

    degree: int
    mass: np.ndarray
    stiffness: sparse.csr_matrix
    dimension: int
    nullity: int

    def mass_norm(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        return float(np.sqrt(np.dot(values * self.mass, values)))

    def energy(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float64)
        return 0.5 * float(np.dot(values * self.mass, values))


@dataclass(frozen=True)
class InitialCondition:
    """A normalized initial cochain and its realized Hodge components."""

    values: np.ndarray
    exact: np.ndarray
    coexact: np.ndarray
    harmonic: np.ndarray
    requested_energy_fractions: tuple[float, float, float]
    realized_energy_fractions: tuple[float, float, float]


@dataclass(frozen=True)
class TransientSolution:
    """Snapshots returned by the transient solver."""

    times: np.ndarray
    states: np.ndarray
    relative_mass_norm: np.ndarray

    @property
    def final(self) -> np.ndarray:
        return self.states[-1]


def _load_incidence(data: np.lib.npyio.NpzFile, prefix: str) -> sparse.csr_matrix:
    shape = tuple(int(v) for v in data[f"{prefix}_shape"])
    return sparse.coo_matrix(
        (
            data[f"{prefix}_value"].astype(np.float64),
            (data[f"{prefix}_row"], data[f"{prefix}_col"]),
        ),
        shape=shape,
    ).tocsr()


def load_geometry(sample_dir: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load one raw TopoBox-3D geometry and its metadata."""

    sample_dir = Path(sample_dir)
    with np.load(sample_dir / "mesh.npz", allow_pickle=False) as archive:
        geometry = {key: archive[key] for key in archive.files}
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    return geometry, metadata


def incidence_matrices(geometry: dict[str, np.ndarray]) -> tuple[sparse.csr_matrix, ...]:
    """Return coboundaries D0, D1, D2 from the stored boundary matrices."""

    class _MappingArchive:
        def __init__(self, mapping: dict[str, np.ndarray]):
            self.mapping = mapping

        def __getitem__(self, key: str) -> np.ndarray:
            return self.mapping[key]

    data = _MappingArchive(geometry)
    b1 = _load_incidence(data, "incidence_1")
    b2 = _load_incidence(data, "incidence_2")
    b3 = _load_incidence(data, "incidence_3")
    return b1.T.tocsr(), b2.T.tocsr(), b3.T.tocsr()


def _barycentric_gradients(tet_points: np.ndarray) -> np.ndarray:
    """Gradients of the four barycentric coordinates of a positive tetrahedron."""

    transform = np.column_stack((np.ones(4), tet_points))
    inverse = np.linalg.inv(transform)
    # lambda_i(x) = inverse[0, i] + inverse[1:, i] dot x
    return inverse[1:, :].T


def whitney_diagonal_masses(geometry: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Assemble positive diagonal Whitney mass approximations for k=0..3.

    M0 uses the standard vertex-lumped volume already stored by TopoBox-3D.
    M1 and M2 are the diagonals of the consistent local Whitney mass matrices.
    M3 is exact because there is one discontinuous Whitney 3-form per tetrahedron.
    """

    points = geometry["points"].astype(np.float64)
    tetra = geometry["oriented_tetra"].astype(np.int64)
    edges = geometry["edges"].astype(np.int64)
    faces = geometry["faces"].astype(np.int64)
    volumes = geometry["tetra_volumes"].astype(np.float64)

    edge_lookup = {tuple(edge): index for index, edge in enumerate(edges)}
    face_lookup = {tuple(face): index for index, face in enumerate(faces)}
    m1 = np.zeros(len(edges), dtype=np.float64)
    m2 = np.zeros(len(faces), dtype=np.float64)

    # Symmetric degree-two tetrahedral quadrature: four equal weights.
    a = 0.5854101966249685
    b = 0.1381966011250105
    barycentric_q = np.full((4, 4), b, dtype=np.float64)
    np.fill_diagonal(barycentric_q, a)

    local_edge_positions = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    local_face_positions = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

    for tet_vertices, volume in zip(tetra, volumes):
        tet_points = points[tet_vertices]
        gradients = _barycentric_gradients(tet_points)
        global_to_local = {int(vertex): i for i, vertex in enumerate(tet_vertices)}

        for p, q in local_edge_positions:
            vertex_pair = tuple(sorted((int(tet_vertices[p]), int(tet_vertices[q]))))
            i, j = (global_to_local[v] for v in vertex_pair)
            basis = (
                barycentric_q[:, i, None] * gradients[j]
                - barycentric_q[:, j, None] * gradients[i]
            )
            diagonal_entry = volume * float(np.mean(np.einsum("qi,qi->q", basis, basis)))
            m1[edge_lookup[vertex_pair]] += diagonal_entry

        for p, q, r in local_face_positions:
            vertex_triple = tuple(sorted((
                int(tet_vertices[p]), int(tet_vertices[q]), int(tet_vertices[r])
            )))
            i, j, k = (global_to_local[v] for v in vertex_triple)
            # Vector proxy of the Whitney 2-form under the Euclidean Hodge star.
            basis = 2.0 * (
                barycentric_q[:, i, None] * np.cross(gradients[j], gradients[k])
                - barycentric_q[:, j, None] * np.cross(gradients[i], gradients[k])
                + barycentric_q[:, k, None] * np.cross(gradients[i], gradients[j])
            )
            diagonal_entry = volume * float(np.mean(np.einsum("qi,qi->q", basis, basis)))
            m2[face_lookup[vertex_triple]] += diagonal_entry

    m0 = geometry["vertex_lumped_volume"].astype(np.float64)
    m3 = 1.0 / volumes
    masses = (m0, m1, m2, m3)
    for degree, mass in enumerate(masses):
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise RuntimeError(f"Non-positive or non-finite diagonal mass for k={degree}")
    return masses


def build_hodge_systems(
    geometry: dict[str, np.ndarray], metadata: dict
) -> tuple[
    tuple[HodgeSystem, ...],
    tuple[sparse.csr_matrix, ...],
    tuple[np.ndarray, ...],
]:
    """Construct symmetric weak Hodge-Laplacian systems for k=0,1,2."""

    d0, d1, d2 = incidence_matrices(geometry)
    m0, m1, m2, m3 = whitney_diagonal_masses(geometry)
    masses = (m0, m1, m2, m3)
    derivatives = (d0, d1, d2)

    diagonal = tuple(sparse.diags(mass, format="csr") for mass in masses)
    inverse_diagonal = tuple(sparse.diags(1.0 / mass, format="csr") for mass in masses)

    stiffness: list[sparse.csr_matrix] = []
    for degree in range(3):
        upper = derivatives[degree].T @ diagonal[degree + 1] @ derivatives[degree]
        if degree == 0:
            lower = sparse.csr_matrix(upper.shape, dtype=np.float64)
        else:
            lower = (
                diagonal[degree]
                @ derivatives[degree - 1]
                @ inverse_diagonal[degree - 1]
                @ derivatives[degree - 1].T
                @ diagonal[degree]
            )
        matrix = (upper + lower).tocsr()
        matrix = (0.5 * (matrix + matrix.T)).tocsr()
        stiffness.append(matrix)

    nullities = (1, int(metadata["beta1"]), int(metadata["beta2"]))
    systems = tuple(
        HodgeSystem(k, masses[k], stiffness[k], len(masses[k]), nullities[k])
        for k in range(3)
    )
    return systems, derivatives, masses


def _smooth_rbf_values(
    evaluation_points: np.ndarray,
    box: np.ndarray,
    rng: np.random.Generator,
    correlation_length: float,
    components: int,
    centers_count: int = 24,
) -> np.ndarray:
    """Evaluate a smooth random RBF scalar/vector field in physical coordinates."""

    centers = rng.uniform(np.zeros(3), box, size=(centers_count, 3))
    coefficients = rng.normal(size=(centers_count, components))
    squared_distance = np.sum(
        (evaluation_points[:, None, :] - centers[None, :, :]) ** 2, axis=2
    )
    weights = np.exp(-0.5 * squared_distance / correlation_length**2)
    values = weights @ coefficients / np.sqrt(centers_count)
    return values[:, 0] if components == 1 else values


def _smooth_cochain(
    degree: int,
    geometry: dict[str, np.ndarray],
    rng: np.random.Generator,
    correlation_length: float,
    centers_count: int,
) -> np.ndarray:
    """Project a smooth ambient differential form onto primal simplices."""

    points = geometry["points"].astype(np.float64)
    box = np.ptp(points, axis=0)
    if degree == 0:
        return _smooth_rbf_values(
            points, box, rng, correlation_length, 1, centers_count
        )
    if degree == 1:
        edges = geometry["edges"].astype(np.int64)
        centers = points[edges].mean(axis=1)
        vector = _smooth_rbf_values(
            centers, box, rng, correlation_length, 3, centers_count
        )
        return np.einsum("ij,ij->i", vector, geometry["edge_vectors"])
    if degree == 2:
        faces = geometry["faces"].astype(np.int64)
        centers = points[faces].mean(axis=1)
        vector = _smooth_rbf_values(
            centers, box, rng, correlation_length, 3, centers_count
        )
        return np.einsum("ij,ij->i", vector, geometry["face_area_vectors"])
    if degree == 3:
        tetra = geometry["oriented_tetra"].astype(np.int64)
        centers = points[tetra].mean(axis=1)
        scalar = _smooth_rbf_values(
            centers, box, rng, correlation_length, 1, centers_count
        )
        return scalar * geometry["tetra_volumes"]
    raise ValueError(f"Unsupported cochain degree {degree}")


def _normalize(values: np.ndarray, mass: np.ndarray, tolerance: float = 1e-13) -> np.ndarray:
    norm = float(np.sqrt(np.dot(values * mass, values)))
    if not np.isfinite(norm) or norm <= tolerance:
        raise RuntimeError("Generated a numerically zero Hodge component")
    return values / norm


def _resolvent_filter(
    values: np.ndarray,
    system: HodgeSystem,
    harmonic: np.ndarray,
    filter_time: float,
    passes: int,
    solve=None,
) -> np.ndarray:
    """Suppress mesh-scale modes while preserving the Hodge subspace."""

    if filter_time <= 0.0 or passes <= 0:
        filtered = np.asarray(values, dtype=np.float64).copy()
    else:
        if solve is None:
            mass_matrix = sparse.diags(system.mass, format="csc")
            solve = spla.factorized(
                (mass_matrix + filter_time * system.stiffness).tocsc()
            )
        filtered = np.asarray(values, dtype=np.float64).copy()
        for _ in range(passes):
            filtered = np.asarray(solve(system.mass * filtered)).ravel()
    if harmonic.shape[1]:
        filtered -= harmonic @ (harmonic.T @ (system.mass * filtered))
    return _normalize(filtered, system.mass)


def build_resolvent_solver(
    system: HodgeSystem,
    filter_time: float,
    passes: int,
):
    """Factor the initial-condition smoothing operator once for reuse."""

    if filter_time <= 0.0 or passes <= 0:
        return None
    mass_matrix = sparse.diags(system.mass, format="csc")
    return spla.factorized(
        (mass_matrix + filter_time * system.stiffness).tocsc()
    )


def harmonic_basis(
    system: HodgeSystem,
    initial_guess: np.ndarray | None = None,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Compute an M-orthonormal basis of the weighted harmonic space."""

    if system.nullity == 0:
        return np.empty((system.dimension, 0), dtype=np.float64)
    if system.degree == 0:
        constant = np.ones((system.dimension, 1), dtype=np.float64)
        return constant / np.sqrt(constant.T @ (system.mass[:, None] * constant))

    count = min(system.dimension - 1, system.nullity + 3)
    mass_matrix = sparse.diags(system.mass, format="csc")
    # A small negative shift avoids factorizing the exactly singular stiffness.
    eigen_kwargs = {
        "k": count,
        "M": mass_matrix,
        "which": "LM",
        "v0": None if initial_guess is None or initial_guess.size == 0 else initial_guess[:, 0],
    }
    try:
        values, vectors = spla.eigsh(
            system.stiffness.tocsc(),
            sigma=-1e-9,
            tol=tolerance,
            maxiter=5000,
            **eigen_kwargs,
        )
    except (RuntimeError, spla.ArpackNoConvergence):
        # Rare ill-conditioned meshes can make a near-zero shift difficult.
        # A slightly larger negative shift preserves the requested low modes.
        values, vectors = spla.eigsh(
            system.stiffness.tocsc(),
            sigma=-1e-7,
            tol=max(tolerance, 1e-7),
            maxiter=15000,
            **eigen_kwargs,
        )
    order = np.argsort(values)
    basis = vectors[:, order[: system.nullity]]
    gram = basis.T @ (system.mass[:, None] * basis)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    basis = basis @ (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    residual = np.linalg.norm(system.stiffness @ basis, axis=0)
    scale = np.maximum(np.linalg.norm(system.mass[:, None] * basis, axis=0), 1e-30)
    if float(np.max(residual / scale, initial=0.0)) > 1e-5:
        raise RuntimeError(f"Weighted harmonic basis residual is too large for k={system.degree}")
    return basis


def harmonic_basis_and_spectrum(
    system: HodgeSystem,
    initial_guess: np.ndarray | None = None,
    nonzero_count: int = 6,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the weighted harmonic basis and low positive spectrum together.

    Batch generation needs both quantities.  A single generalized eigensolve is
    substantially cheaper than calling :func:`harmonic_basis` and
    :func:`smallest_generalized_eigenvalues` independently for every geometry.
    """

    if system.dimension <= 1:
        basis = np.ones((system.dimension, system.nullity), dtype=np.float64)
        if basis.size:
            basis /= np.sqrt(basis.T @ (system.mass[:, None] * basis))
        return basis, np.empty(0, dtype=np.float64)

    count = min(
        system.dimension - 1,
        system.nullity + nonzero_count + 2,
    )
    mass_matrix = sparse.diags(system.mass, format="csc")
    values, vectors = spla.eigsh(
        system.stiffness.tocsc(),
        k=count,
        M=mass_matrix,
        sigma=-1e-9,
        which="LM",
        tol=tolerance,
        maxiter=5000,
        v0=None if initial_guess is None or initial_guess.size == 0 else initial_guess[:, 0],
    )
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)

    if system.degree == 0 and system.nullity:
        basis = np.ones((system.dimension, 1), dtype=np.float64)
        basis /= np.sqrt(basis.T @ (system.mass[:, None] * basis))
    elif system.nullity:
        basis = vectors[:, :system.nullity]
        gram = basis.T @ (system.mass[:, None] * basis)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        basis = basis @ (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    else:
        basis = np.empty((system.dimension, 0), dtype=np.float64)

    if basis.shape[1]:
        residual = np.linalg.norm(system.stiffness @ basis, axis=0)
        scale = np.maximum(
            np.linalg.norm(system.mass[:, None] * basis, axis=0),
            1e-30,
        )
        if float(np.max(residual / scale, initial=0.0)) > 1e-5:
            raise RuntimeError(
                f"Weighted harmonic basis residual is too large for k={system.degree}"
            )

    values = np.maximum(values, 0.0)
    threshold = max(1e-7, 1e-6 * float(values[-1]))
    positive = values[values > threshold][:nonzero_count]
    return basis, positive


def generate_initial_condition(
    degree: int,
    geometry: dict[str, np.ndarray],
    system: HodgeSystem,
    derivatives: tuple[sparse.csr_matrix, ...],
    next_mass: np.ndarray | None,
    harmonic: np.ndarray,
    seed: int = 20260722,
    correlation_length: float = 0.22,
    energy_fractions: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    filter_time: float = 0.03,
    filter_passes: int = 2,
    rbf_centers: int = 24,
    scalar_harmonic_fraction: float = 0.0,
    filter_solver=None,
) -> InitialCondition:
    """Generate one smooth, unit-M-norm initial cochain.

    For k=0 the main benchmark uses a smooth mass-mean-zero scalar field and
    suppresses the constant harmonic mode.  For k=1,2 the requested exact,
    coexact and harmonic fractions are realized whenever the harmonic space is
    non-empty; otherwise harmonic energy is redistributed equally.
    """

    requested = np.asarray(energy_fractions, dtype=np.float64)
    if requested.shape != (3,) or np.any(requested < 0.0) or not np.isclose(requested.sum(), 1.0):
        raise ValueError("energy_fractions must be three nonnegative values summing to one")
    rng = np.random.default_rng(seed + 1009 * degree)
    zeros = np.zeros(system.dimension, dtype=np.float64)

    if degree == 0:
        values = _smooth_cochain(
            0, geometry, rng, correlation_length, rbf_centers
        )
        weighted_mean = np.dot(system.mass, values) / np.sum(system.mass)
        coexact = _resolvent_filter(
            values - weighted_mean,
            system,
            harmonic,
            filter_time,
            filter_passes,
            filter_solver,
        )
        if not 0.0 <= scalar_harmonic_fraction < 1.0:
            raise ValueError("scalar_harmonic_fraction must lie in [0,1)")
        if scalar_harmonic_fraction > 0.0:
            harmonic_component = _normalize(
                np.ones(system.dimension, dtype=np.float64), system.mass
            )
            values = (
                np.sqrt(1.0 - scalar_harmonic_fraction) * coexact
                + np.sqrt(scalar_harmonic_fraction) * harmonic_component
            )
            realized = (0.0, 1.0 - scalar_harmonic_fraction, scalar_harmonic_fraction)
        else:
            harmonic_component = zeros.copy()
            values = coexact.copy()
            realized = (0.0, 1.0, 0.0)
        return InitialCondition(
            values,
            zeros.copy(),
            coexact,
            harmonic_component,
            tuple(requested),
            realized,
        )

    exact_potential = _smooth_cochain(
        degree - 1, geometry, rng, correlation_length, rbf_centers
    )
    exact = derivatives[degree - 1] @ exact_potential
    exact = _resolvent_filter(
        np.asarray(exact).ravel(),
        system,
        harmonic,
        filter_time,
        filter_passes,
        filter_solver,
    )

    if next_mass is None:
        raise ValueError("next_mass is required for k=1,2 coexact generation")
    coexact_potential = _smooth_cochain(
        degree + 1, geometry, rng, correlation_length, rbf_centers
    )
    right_hand_side = derivatives[degree].T @ (
        next_mass * coexact_potential
    )
    coexact = np.asarray(right_hand_side).ravel() / system.mass
    # Numerical cleanup; exact and coexact should already be M-orthogonal by d^2=0.
    coexact -= exact * np.dot(exact * system.mass, coexact)
    coexact = _resolvent_filter(
        coexact,
        system,
        harmonic,
        filter_time,
        filter_passes,
        filter_solver,
    )
    coexact -= exact * np.dot(exact * system.mass, coexact)
    coexact = _normalize(coexact, system.mass)

    if harmonic.shape[1] == 0:
        realized_weights = np.array((0.5, 0.5, 0.0), dtype=np.float64)
        harmonic_component = zeros.copy()
    else:
        coefficients = rng.normal(size=harmonic.shape[1])
        coefficients /= np.linalg.norm(coefficients)
        harmonic_component = harmonic @ coefficients
        harmonic_component = _normalize(harmonic_component, system.mass)
        realized_weights = requested.copy()

    values = (
        np.sqrt(realized_weights[0]) * exact
        + np.sqrt(realized_weights[1]) * coexact
        + np.sqrt(realized_weights[2]) * harmonic_component
    )
    values = _normalize(values, system.mass)
    return InitialCondition(
        values=values,
        exact=exact,
        coexact=coexact,
        harmonic=harmonic_component,
        requested_energy_fractions=tuple(float(v) for v in requested),
        realized_energy_fractions=tuple(float(v) for v in realized_weights),
    )


def solve_crank_nicolson(
    system: HodgeSystem,
    initial: np.ndarray,
    final_time: float,
    steps: int = 80,
    kappa: float = 1.0,
    snapshots: int = 17,
) -> TransientSolution:
    """Advance M w_t + kappa K w = 0 with Crank--Nicolson."""

    if final_time <= 0.0 or steps <= 0 or kappa <= 0.0:
        raise ValueError("final_time, steps and kappa must be positive")
    dt = final_time / steps
    mass_matrix = sparse.diags(system.mass, format="csc")
    left = (mass_matrix + 0.5 * kappa * dt * system.stiffness).tocsc()
    right = (mass_matrix - 0.5 * kappa * dt * system.stiffness).tocsr()
    solve = spla.factorized(left)

    requested_steps = np.unique(np.rint(np.linspace(0, steps, snapshots)).astype(int))
    states: list[np.ndarray] = []
    times: list[float] = []
    values = np.asarray(initial, dtype=np.float64).copy()
    initial_norm = system.mass_norm(values)
    request_index = 0
    for step in range(steps + 1):
        if request_index < len(requested_steps) and step == requested_steps[request_index]:
            states.append(values.copy())
            times.append(step * dt)
            request_index += 1
        if step < steps:
            values = np.asarray(solve(right @ values)).ravel()

    state_array = np.stack(states)
    relative = np.asarray([system.mass_norm(state) / initial_norm for state in state_array])
    return TransientSolution(np.asarray(times), state_array, relative)


def solve_fixed_time_batch(
    system: HodgeSystem,
    initials: np.ndarray,
    final_time: float,
    steps: int = 100,
    kappa: float = 1.0,
) -> np.ndarray:
    """Advance several initial cochains to one fixed time with one factorization."""

    values = np.asarray(initials, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != system.dimension:
        raise ValueError(
            f"initials must have shape (batch,{system.dimension}), got {values.shape}"
        )
    if final_time <= 0.0 or steps <= 0 or kappa <= 0.0:
        raise ValueError("final_time, steps and kappa must be positive")
    dt = final_time / steps
    mass_matrix = sparse.diags(system.mass, format="csc")
    left = (mass_matrix + 0.5 * kappa * dt * system.stiffness).tocsc()
    right = (mass_matrix - 0.5 * kappa * dt * system.stiffness).tocsr()
    solve = spla.factorized(left)
    result = values.copy()
    for _ in range(steps):
        result = np.asarray(solve(right @ result.T)).T
    return result


def smallest_generalized_eigenvalues(
    system: HodgeSystem, nonzero_count: int = 6
) -> np.ndarray:
    """Return a few low generalized eigenvalues, including the known nullity."""

    count = min(system.dimension - 1, system.nullity + nonzero_count + 2)
    values = spla.eigsh(
        system.stiffness.tocsc(),
        k=count,
        M=sparse.diags(system.mass, format="csc"),
        sigma=-1e-9,
        which="LM",
        return_eigenvectors=False,
        tol=1e-8,
        maxiter=5000,
    )
    values = np.sort(np.maximum(values, 0.0))
    threshold = max(1e-7, 1e-6 * float(values[-1]))
    return values[values > threshold][:nonzero_count]


def validate_system(system: HodgeSystem, tolerance: float = 1e-10) -> dict[str, float]:
    """Basic symmetry and positivity diagnostics for one assembled system."""

    asymmetry = system.stiffness - system.stiffness.T
    symmetry_error = float(np.max(np.abs(asymmetry.data), initial=0.0))
    diagonal_minimum = float(system.stiffness.diagonal().min(initial=np.inf))
    if symmetry_error > tolerance:
        raise RuntimeError(f"Stiffness symmetry check failed for k={system.degree}")
    if diagonal_minimum < -tolerance:
        raise RuntimeError(f"Negative stiffness diagonal for k={system.degree}")
    return {
        "mass_min": float(system.mass.min()),
        "mass_max": float(system.mass.max()),
        "stiffness_symmetry_error": symmetry_error,
        "stiffness_diagonal_min": diagonal_minimum,
    }
