"""Validate every generated sample without invoking Gmsh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from topobox3d.topology import compute_topology
else:
    from .topology import compute_topology


EXPECTED_GEOMETRY_FEATURE_NAMES = [
    "x_normalized", "y_normalized", "z_normalized",
    "is_boundary", "analytic_domain_sdf",
]


def validate_dataset(root: Path) -> list[str]:
    errors: list[str] = []
    for metadata_path in sorted(root.glob("protocol_*/*/*/metadata.json")):
        sample = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(sample / "mesh.npz") as data:
                topology, _ = compute_topology(data["points"], data["tetra"])
                expected = (1, metadata["beta1"], metadata["beta2"], 0)
                actual = (topology.beta0, topology.beta1, topology.beta2, topology.beta3)
                if actual != expected:
                    errors.append(f"{sample}: topology {actual} != {expected}")
                if float(metadata["geometry"]["measured_min_clearance"]) < 0.10 - 1e-9:
                    errors.append(f"{sample}: clearance below 0.10")
                if not np.all(np.isfinite(data["geometry_features"])):
                    errors.append(f"{sample}: non-finite geometry features")
                if data["points"].shape[0] != data["geometry_features"].shape[0]:
                    errors.append(f"{sample}: node feature count mismatch")
                feature_names = data["geometry_feature_names"].tolist()
                if feature_names != EXPECTED_GEOMETRY_FEATURE_NAMES:
                    errors.append(f"{sample}: unexpected geometry features {feature_names}")
                if data["geometry_features"].shape[1] != len(EXPECTED_GEOMETRY_FEATURE_NAMES):
                    errors.append(f"{sample}: expected five node geometry channels")
                required = {
                    "edges", "faces", "oriented_tetra", "incidence_1_row", "incidence_2_row",
                    "incidence_3_row", "harmonic_basis_0", "harmonic_basis_1", "harmonic_basis_2",
                    "regular_grid_normalized_xyz", "regular_grid_features", "analytic_domain_sdf",
                }
                missing = required.difference(data.files)
                if missing:
                    errors.append(f"{sample}: missing model inputs {sorted(missing)}")
                    continue
                matrices = {}
                for rank in (1, 2, 3):
                    prefix = f"incidence_{rank}"
                    matrices[rank] = sparse.coo_matrix(
                        (data[f"{prefix}_value"], (data[f"{prefix}_row"], data[f"{prefix}_col"])),
                        shape=tuple(data[f"{prefix}_shape"]),
                    ).tocsr()
                if (matrices[1] @ matrices[2]).nnz or (matrices[2] @ matrices[3]).nnz:
                    errors.append(f"{sample}: incidence matrices do not form a chain complex")
                expected_harmonic = (1, metadata["beta1"], metadata["beta2"])
                actual_harmonic = tuple(data[f"harmonic_basis_{rank}"].shape[1] for rank in range(3))
                if actual_harmonic != expected_harmonic:
                    errors.append(f"{sample}: harmonic dimensions {actual_harmonic} != {expected_harmonic}")
                max_harmonic_residual = max(
                    float(np.max(data["harmonic_residual_1"], initial=0.0)),
                    float(np.max(data["harmonic_residual_2"], initial=0.0)),
                )
                if max_harmonic_residual > 1e-6:
                    errors.append(f"{sample}: harmonic residual {max_harmonic_residual:.3e} is too large")
                if data["regular_grid_normalized_xyz"].shape != (32, 16, 16, 3):
                    errors.append(f"{sample}: unexpected regular-grid shape")
                if data["regular_grid_features"].shape[:3] != (32, 16, 16):
                    errors.append(f"{sample}: regular-grid feature shape mismatch")
                expected_grid_names = [
                    "x_normalized", "y_normalized", "z_normalized",
                    "is_boundary", "analytic_domain_sdf",
                ]
                if data["regular_grid_feature_names"].tolist() != expected_grid_names:
                    errors.append(f"{sample}: unexpected regular-grid features")
                if data["regular_grid_features"].shape[-1] != 5:
                    errors.append(f"{sample}: expected five regular-grid channels")
                if np.min(data["analytic_domain_sdf"]) < -5e-5:
                    errors.append(f"{sample}: mesh node classified outside analytic domain")
        except Exception as exc:
            errors.append(f"{sample}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    errors = validate_dataset(args.dataset_root)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("All samples passed topology, clearance, SDF, chain-complex, harmonic-basis, and feature checks.")


if __name__ == "__main__":
    main()
