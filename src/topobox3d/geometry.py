"""Constrained sampling of tunnels and closed cavities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GeometryConfig


@dataclass(frozen=True)
class Tunnel:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "circular_cylinder", "start": self.start, "end": self.end, "radius": self.radius}


@dataclass(frozen=True)
class Cavity:
    center: tuple[float, float, float]
    axes: tuple[float, float, float]
    rotation_axis: tuple[float, float, float]
    rotation_angle: float

    @property
    def bounding_radius(self) -> float:
        return max(self.axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "sphere" if max(self.axes) - min(self.axes) < 1e-10 else "ellipsoid",
            "center": self.center,
            "axes": self.axes,
            "rotation_axis": self.rotation_axis,
            "rotation_angle": self.rotation_angle,
        }


@dataclass(frozen=True)
class GeometrySpec:
    tunnels: tuple[Tunnel, ...]
    cavities: tuple[Cavity, ...]
    family: str
    min_clearance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "tunnels": [x.to_dict() for x in self.tunnels],
            "cavities": [x.to_dict() for x in self.cavities],
            "measured_min_clearance": self.min_clearance,
        }


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * ab)))


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    # Closest distance between two line segments (Ericson, Real-Time Collision Detection).
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    eps = 1e-14
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom != 0 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _clearances(tunnels: list[Tunnel], cavities: list[Cavity], box: np.ndarray) -> list[float]:
    values: list[float] = []
    for i, tunnel in enumerate(tunnels):
        a, b = np.asarray(tunnel.start), np.asarray(tunnel.end)
        # z=0 and z=L are intentional openings; enforce clearance to x/y walls.
        for dim in (0, 1):
            values.extend([min(a[dim], b[dim]) - tunnel.radius, box[dim] - max(a[dim], b[dim]) - tunnel.radius])
        for other in tunnels[i + 1 :]:
            values.append(
                _segment_distance(a, b, np.asarray(other.start), np.asarray(other.end))
                - tunnel.radius
                - other.radius
            )
    for i, cavity in enumerate(cavities):
        center, radius = np.asarray(cavity.center), cavity.bounding_radius
        values.extend((center - radius).tolist())
        values.extend((box - center - radius).tolist())
        for other in cavities[i + 1 :]:
            values.append(float(np.linalg.norm(center - np.asarray(other.center))) - radius - other.bounding_radius)
        for tunnel in tunnels:
            values.append(
                _point_segment_distance(center, np.asarray(tunnel.start), np.asarray(tunnel.end))
                - radius
                - tunnel.radius
            )
    return values


def sample_geometry(
    rng: np.random.Generator,
    beta1: int,
    beta2: int,
    family: str,
    config: GeometryConfig,
) -> GeometrySpec:
    """Sample primitives with conservative clearance checks.

    Family A uses axis-aligned circular tunnels and spherical cavities. Family B
    uses inclined tunnels and high-eccentricity, rotated ellipsoids while keeping
    the same topology.
    """
    box = np.asarray(config.box, dtype=float)
    gap = config.min_clearance

    # The (3, 3) composition case is a deliberately crowded packing problem.
    # Pure rejection sampling is both slow and biased toward rare accidental
    # layouts, so start from a safe stratified construction and add small,
    # reproducible perturbations. Mirroring keeps this family geometrically
    # varied while preserving a conservative 0.10 clearance certificate.
    if beta1 == 3 and beta2 == 3 and family == "A":
        mirror = -1.0 if rng.random() < 0.5 else 1.0
        base_xy = np.array(((0.25, 0.20), (0.65, 0.50), (1.05, 0.20)))
        if mirror < 0:
            base_xy[:, 1] = 1.0 - base_xy[:, 1]
        tunnels = [
            Tunnel((float(x), float(y), 0.0), (float(x), float(y), box[2]), 0.10)
            for x, y in base_xy
        ]
        cavity_centers = np.array(((1.45, 0.25, 0.25), (1.70, 0.70, 0.50), (1.45, 0.25, 0.75)))
        if mirror < 0:
            cavity_centers[:, 1] = 1.0 - cavity_centers[:, 1]
        cavity_centers += rng.uniform(-0.012, 0.012, size=(3, 3))
        cavities = [
            Cavity((float(x), float(y), float(z)), (0.12, 0.12, 0.12), (0.0, 0.0, 1.0), 0.0)
            for x, y, z in cavity_centers
        ]
        clearances = _clearances(tunnels, cavities, box)
        measured = min(clearances)
        if measured >= gap - 1e-12:
            return GeometrySpec(tuple(tunnels), tuple(cavities), family, measured)

    for _ in range(config.max_sampling_attempts):
        tunnels: list[Tunnel] = []
        cavities: list[Cavity] = []

        for _tunnel_idx in range(beta1):
            radius = float(rng.uniform(*config.tunnel_radius))
            margin = radius + gap
            x0 = float(rng.uniform(margin, box[0] - margin))
            y0 = float(rng.uniform(margin, box[1] - margin))
            if family == "B":
                # End-point shifts make the tunnel genuinely inclined, not just translated.
                shift = rng.uniform((-0.28, -0.18), (0.28, 0.18))
                x1 = float(np.clip(x0 + shift[0], margin, box[0] - margin))
                y1 = float(np.clip(y0 + shift[1], margin, box[1] - margin))
                if np.linalg.norm([x1 - x0, y1 - y0]) < 0.07:
                    x1 = float(np.clip(x0 + (0.09 if x0 < box[0] / 2 else -0.09), margin, box[0] - margin))
            else:
                x1, y1 = x0, y0
            candidate = Tunnel((x0, y0, 0.0), (x1, y1, box[2]), radius)
            tunnels.append(candidate)

        for _cavity_idx in range(beta2):
            if family == "B":
                small = float(rng.uniform(0.12, 0.14))
                axes_arr = np.array([small, rng.uniform(0.16, 0.19), rng.uniform(0.21, 0.25)])
                rng.shuffle(axes_arr)
                axes = tuple(float(x) for x in axes_arr)
                axis_arr = rng.normal(size=3)
                axis_arr /= np.linalg.norm(axis_arr)
                angle = float(rng.uniform(0.25, 1.15))
            else:
                radius = float(rng.uniform(*config.cavity_radius))
                axes = (radius, radius, radius)
                axis_arr, angle = np.array([0.0, 0.0, 1.0]), 0.0
            bound = max(axes)
            low = np.full(3, bound + gap)
            high = box - low
            if np.any(low >= high):
                break
            center = tuple(float(x) for x in rng.uniform(low, high))
            cavities.append(Cavity(center, axes, tuple(float(x) for x in axis_arr), angle))

        if len(cavities) != beta2:
            continue
        clearances = _clearances(tunnels, cavities, box)
        measured = min(clearances) if clearances else float(min(box))
        if measured >= gap - 1e-12:
            return GeometrySpec(tuple(tunnels), tuple(cavities), family, measured)

    raise RuntimeError(
        f"Could not sample geometry beta1={beta1}, beta2={beta2}, family={family} "
        f"with clearance {gap} after {config.max_sampling_attempts} attempts"
    )
