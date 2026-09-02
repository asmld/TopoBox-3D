"""Fast validation of the canonical node geometry input in raw and packed data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from topobox3d.model_inputs import TopoBoxGeometry
    from topobox3d.sdf import NODE_GEOMETRY_FEATURE_NAMES
else:
    from .model_inputs import TopoBoxGeometry
    from .sdf import NODE_GEOMETRY_FEATURE_NAMES


EXPECTED = list(NODE_GEOMETRY_FEATURE_NAMES)


def _decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _check_arrays(data, label: str, errors: list[str]) -> None:
    features = np.asarray(data["geometry_features"])
    expected = np.column_stack([
        data["normalized_xyz"], data["is_boundary"], data["analytic_domain_sdf"],
    ]).astype(np.float32)
    if features.shape != expected.shape or not np.array_equal(features, expected):
        errors.append(f"{label}: canonical geometry_features mismatch")
    for rich_field in (
        "boundary_mask", "is_outer_boundary", "is_tunnel_boundary", "is_cavity_boundary",
        "distance_to_boundary", "distance_to_outer_boundary", "distance_to_internal_boundary",
        "distance_to_tunnel_boundary", "distance_to_cavity_boundary",
        "analytic_outer_box_sdf", "analytic_internal_void_sdf", "analytic_tunnel_sdf",
        "analytic_cavity_sdf", "has_tunnel", "has_cavity",
    ):
        if rich_field not in data:
            errors.append(f"{label}: missing preserved field {rich_field}")


def _check_regular_grid(features, names, normalized, resolution, box, label, errors) -> None:
    features = np.asarray(features)
    resolution = tuple(int(value) for value in resolution)
    expected_names = [
        "x_normalized", "y_normalized", "z_normalized",
        "is_boundary", "analytic_domain_sdf",
    ]
    if _decode(names) != expected_names or features.shape != (*resolution, 5):
        errors.append(f"{label}: wrong regular-grid schema")
        return
    if not np.array_equal(features[..., :3], normalized):
        errors.append(f"{label}: regular-grid normalized xyz mismatch")
    spacing = np.asarray(box, dtype=np.float64) / (np.asarray(resolution, dtype=np.float64) - 1.0)
    threshold = 0.5 * np.linalg.norm(spacing)
    expected_boundary = (np.abs(features[..., 4]) <= threshold).astype(np.float32)
    if not np.array_equal(features[..., 3], expected_boundary):
        errors.append(f"{label}: regular-grid boundary band mismatch")


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    raw_paths = sorted(root.glob("protocol_*/*/*/mesh.npz"))
    for path in raw_paths:
        with np.load(path, allow_pickle=False) as data:
            if data["geometry_feature_names"].tolist() != EXPECTED:
                errors.append(f"{path}: wrong feature names")
            _check_arrays(data, str(path), errors)
        metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as data:
            _check_regular_grid(
                data["regular_grid_features"], data["regular_grid_feature_names"],
                data["regular_grid_normalized_xyz"], data["regular_grid_resolution"],
                metadata["box"], str(path), errors,
            )
        if metadata["boundary_encoding"]["geometry_feature_names"] != EXPECTED:
            errors.append(f"{path}: metadata feature names mismatch")

    index = json.loads((root / "packed" / "index.json").read_text(encoding="utf-8"))
    if len(index) != len(raw_paths):
        errors.append(f"packed index has {len(index)} records; raw has {len(raw_paths)}")
    representatives: dict[tuple[str, str], dict] = {}
    shards: dict[str, list[dict]] = {}
    for record in index:
        representatives.setdefault((record["protocol"], record["split"]), record)
        shards.setdefault(record["shard"], []).append(record)
    for relative, records in shards.items():
        with h5py.File(root / "packed" / relative, "r") as handle:
            names = _decode(handle["shared/geometry_feature_names"][()])
            if names != EXPECTED:
                errors.append(f"{relative}: wrong shared feature names")
            for record in records:
                group = handle[record["group"]]
                label = f"{relative}:{record['group']}"
                _check_arrays(group, label, errors)
                metadata = json.loads(group["metadata_json"][()].decode("utf-8"))
                _check_regular_grid(
                    group["regular_grid_features"], handle["shared/regular_grid_feature_names"][()],
                    handle["shared/regular_grid_normalized_xyz"],
                    handle["shared/regular_grid_resolution"], metadata["box"], label, errors,
                )

    for record in representatives.values():
        raw = root / f"protocol_{record['protocol']}" / record["split"] / record["geometry_id"]
        geometry = TopoBoxGeometry(raw)
        for model in ("mgn-lite", "rigno", "transolver", "gnot", "gaot", "tno"):
            adapted = geometry.for_model(model)
            if adapted["node_geometry"].shape[1] != 5:
                errors.append(f"{record['geometry_id']}:{model}: node geometry is not 5D")
        tno = geometry.for_model("tno")["cochains"]
        if [cochain.shape[1] for cochain in tno] != [5, 9, 9]:
            errors.append(f"{record['geometry_id']}: TNO cochain dimensions are not [5, 9, 9]")
        geometry.data.close()
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    errors = validate(args.dataset_root)
    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(1)
    print(
        "Canonical 5D node/grid inputs, the grid boundary band, raw/packed data, "
        "and all model adapters passed."
    )


if __name__ == "__main__":
    main()
