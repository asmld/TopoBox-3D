"""Dataset configuration and the four benchmark protocol definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeshConfig:
    bulk_size: float = 0.115
    feature_size: float = 0.070
    refinement_distance: float = 0.22
    optimize: bool = True
    algorithm_3d: int = 10  # HXT


@dataclass(frozen=True)
class InputConfig:
    regular_grid_resolution: tuple[int, int, int] = (32, 16, 16)
    regular_grid_boundary_band: str = "abs(domain_sdf) <= half_voxel_diagonal"
    node_geometry_feature_names: tuple[str, ...] = (
        "x_normalized", "y_normalized", "z_normalized",
        "is_boundary", "analytic_domain_sdf",
    )
    save_tno_complex: bool = True
    save_harmonic_basis: bool = True


@dataclass(frozen=True)
class GeometryConfig:
    box: tuple[float, float, float] = (2.0, 1.0, 1.0)
    tunnel_axis: str = "z"
    min_clearance: float = 0.10
    tunnel_radius: tuple[float, float] = (0.10, 0.135)
    cavity_radius: tuple[float, float] = (0.12, 0.165)
    max_sampling_attempts: int = 30000


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "TopoBox-3D-mini"
    seed: int = 20260721
    train: int = 20
    validation: int = 5
    test_iid: int = 5
    test_ood: int = 5
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    inputs: InputConfig = field(default_factory=InputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROTOCOLS = {
    "A": {
        "name": "fixed_topology_geometry_ood",
        "iid_pairs": [(1, 1)],
        "ood_pairs": [(1, 1)],
        "iid_family": "A",
        "ood_family": "B",
    },
    "B": {
        "name": "beta1_ood",
        "iid_pairs": [(0, 0), (1, 0), (2, 0)],
        "ood_pairs": [(3, 0)],
        "iid_family": "A",
        "ood_family": "A",
    },
    "C": {
        "name": "beta2_ood",
        "iid_pairs": [(0, 0), (0, 1), (0, 2)],
        "ood_pairs": [(0, 3)],
        "iid_family": "A",
        "ood_family": "A",
    },
    "D": {
        "name": "mix_ood",
        "iid_pairs": [(g, c) for g in range(3) for c in range(3)],
        "ood_pairs": [(3, 3)],
        "iid_family": "A",
        "ood_family": "A",
    },
}


def split_size(config: DatasetConfig, split: str) -> int:
    return {
        "train": config.train,
        "validation": config.validation,
        "test_iid": config.test_iid,
        "test_ood": config.test_ood,
    }[split]


def save_config(config: DatasetConfig, output_root: Path) -> None:
    import json

    output_root.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict() | {"protocols": PROTOCOLS}
    (output_root / "dataset_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
