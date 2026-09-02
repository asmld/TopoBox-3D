"""Analytic geometry fields for mesh nodes and the regular latent grid."""

from __future__ import annotations

import numpy as np

from .geometry import GeometrySpec


REGULAR_GRID_FEATURE_NAMES = (
    "x_normalized", "y_normalized", "z_normalized",
    "is_boundary", "analytic_domain_sdf",
)

# Canonical node geometry input shared by all irregular-mesh models.  Richer
# geometry fields remain stored separately in NPZ/HDF5 for QA and ablations.
NODE_GEOMETRY_FEATURE_NAMES = (
    "x_normalized", "y_normalized", "z_normalized",
    "is_boundary", "analytic_domain_sdf",
)


def node_geometry_features(
    normalized_xyz: np.ndarray,
    is_boundary: np.ndarray,
    analytic_domain_sdf: np.ndarray,
) -> np.ndarray:
    """Build the canonical five-channel node geometry input."""
    return np.column_stack(
        [normalized_xyz, is_boundary, analytic_domain_sdf]
    ).astype(np.float32)


def regular_grid_geometry_features(
    normalized_xyz: np.ndarray,
    analytic_domain_sdf: np.ndarray,
    box: tuple[float, float, float],
    resolution: tuple[int, int, int],
) -> np.ndarray:
    """Build five grid channels using a half-voxel-diagonal boundary band."""
    spacing = np.asarray(box, dtype=np.float64) / (np.asarray(resolution, dtype=np.float64) - 1.0)
    half_voxel_diagonal = 0.5 * np.linalg.norm(spacing)
    is_boundary = (np.abs(analytic_domain_sdf) <= half_voxel_diagonal).astype(np.uint8)
    return np.column_stack([normalized_xyz, is_boundary, analytic_domain_sdf]).astype(np.float32)


def _box_sdf_positive_inside(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Euclidean signed distance to an axis-aligned box, positive inside."""
    lower = -points
    upper = points - box
    outside_vector = np.maximum(np.maximum(lower, upper), 0.0)
    outside_distance = np.linalg.norm(outside_vector, axis=1)
    inside_distance = np.minimum(points, box - points).min(axis=1)
    outside = np.any((points < 0.0) | (points > box), axis=1)
    return np.where(outside, -outside_distance, inside_distance)


def _tunnel_sdf_positive_outside(points: np.ndarray, start: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
    axis = end - start
    # Through-tunnels are cylinders continued beyond both z faces in the CAD;
    # use the supporting line rather than a capped finite segment.
    parameter = ((points - start) @ axis) / np.dot(axis, axis)
    closest = start + parameter[:, None] * axis
    return np.linalg.norm(points - closest, axis=1) - radius


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0 or angle == 0.0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s, one_c = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x*x*one_c, x*y*one_c - z*s, x*z*one_c + y*s],
        [y*x*one_c + z*s, c + y*y*one_c, y*z*one_c - x*s],
        [z*x*one_c - y*s, z*y*one_c + x*s, c + z*z*one_c],
    ])


def _cavity_sdf_positive_outside(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    rotation_axis: np.ndarray,
    rotation_angle: float,
) -> np.ndarray:
    """Sign-exact radial distance approximation for a rotated ellipsoid.

    It is exact for spheres. For ellipsoids the zero level set and sign are
    exact; the magnitude is the radial surface distance rather than the more
    expensive closest-point Euclidean distance.
    """
    rotation = _rotation_matrix(rotation_axis, rotation_angle)
    local = (points - center) @ rotation
    rho = np.sqrt(np.sum((local / axes) ** 2, axis=1))
    surface = local / np.maximum(rho[:, None], 1e-12)
    radial_distance = np.linalg.norm(local - surface, axis=1)
    center_distance = float(np.min(axes))
    return np.where(rho < 1e-12, -center_distance, np.sign(rho - 1.0) * radial_distance)


def analytic_sdf(points: np.ndarray, spec: GeometrySpec, box: tuple[float, float, float]) -> dict[str, np.ndarray]:
    """Return signed geometry fields; every SDF is positive in the solid domain."""
    points = np.asarray(points, dtype=np.float64)
    box_array = np.asarray(box, dtype=np.float64)
    outer = _box_sdf_positive_inside(points, box_array)
    absent = float(np.linalg.norm(box_array))

    tunnel_fields = [
        _tunnel_sdf_positive_outside(points, np.asarray(item.start), np.asarray(item.end), item.radius)
        for item in spec.tunnels
    ]
    cavity_fields = [
        _cavity_sdf_positive_outside(
            points, np.asarray(item.center), np.asarray(item.axes),
            np.asarray(item.rotation_axis), item.rotation_angle,
        )
        for item in spec.cavities
    ]
    tunnel = np.min(tunnel_fields, axis=0) if tunnel_fields else np.full(len(points), absent)
    cavity = np.min(cavity_fields, axis=0) if cavity_fields else np.full(len(points), absent)
    internal = np.minimum(tunnel, cavity)
    domain = np.minimum(outer, internal)
    return {
        "analytic_domain_sdf": domain.astype(np.float32),
        "analytic_outer_box_sdf": outer.astype(np.float32),
        "analytic_internal_void_sdf": internal.astype(np.float32),
        "analytic_tunnel_sdf": tunnel.astype(np.float32),
        "analytic_cavity_sdf": cavity.astype(np.float32),
        "has_tunnel": np.full(len(points), bool(spec.tunnels), dtype=np.uint8),
        "has_cavity": np.full(len(points), bool(spec.cavities), dtype=np.uint8),
        "is_in_domain_analytic": (domain >= -1e-6).astype(np.uint8),
    }


def regular_grid_fields(
    spec: GeometrySpec,
    box: tuple[float, float, float],
    resolution: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    axes = [np.linspace(0.0, length, count, dtype=np.float32) for length, count in zip(box, resolution)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    flat = grid.reshape(-1, 3)
    sdf = analytic_sdf(flat, spec, box)
    normalized = flat / np.asarray(box, dtype=np.float32)
    features = regular_grid_geometry_features(
        normalized, sdf["analytic_domain_sdf"], box, resolution
    )
    shape = (*resolution, len(REGULAR_GRID_FEATURE_NAMES))
    return {
        "regular_grid_xyz": grid.astype(np.float32),
        "regular_grid_normalized_xyz": normalized.reshape(*resolution, 3).astype(np.float32),
        "regular_grid_features": features.reshape(shape),
        "regular_grid_feature_names": np.asarray(REGULAR_GRID_FEATURE_NAMES),
        "regular_grid_resolution": np.asarray(resolution, dtype=np.int32),
    }
