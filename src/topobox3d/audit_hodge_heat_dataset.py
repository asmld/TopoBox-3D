"""Audit completeness and numerical invariants of Hodge-heat HDF5 shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_SPLIT_COUNTS = {
    "train": 800,
    "validation": 120,
    "test_iid": 200,
    "test_ood": 200,
}


def audit(
    root: Path,
    require_complete: bool = False,
    deep: bool = False,
) -> dict:
    errors = []
    counts = Counter()
    betti_counts = Counter()
    relative_norms = {degree: [] for degree in range(3)}
    shard_paths = sorted(
        path for path in root.glob("protocol_*/*/shard_*.h5")
        if not path.name.endswith(".tmp.h5")
    )

    for shard_path in shard_paths:
        try:
            with h5py.File(shard_path, "r") as handle:
                if handle.attrs.get("format") != "TopoBox-3D-HodgeHeat-v1":
                    errors.append(f"{shard_path}: wrong format")
                    continue
                config_names = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in handle["config_names"][()]
                ]
                if len(config_names) != 4:
                    errors.append(f"{shard_path}: expected four configurations")
                for geometry_id, group in handle["samples"].items():
                    protocol = str(group.attrs["protocol"])
                    split = str(group.attrs["split"])
                    beta1 = int(group.attrs["beta1"])
                    beta2 = int(group.attrs["beta2"])
                    counts[(protocol, split)] += 1
                    betti_counts[(protocol, split, beta1, beta2)] += 1
                    expected_nullity = (1, beta1, beta2)
                    for degree in range(3):
                        degree_group = group[f"k{degree}"]
                        w0 = degree_group["w0"]
                        wT = degree_group["wT"]
                        mass = degree_group["mass"]
                        harmonic = degree_group["harmonic_basis"]
                        if w0.shape != wT.shape or w0.shape[0] != 4:
                            errors.append(
                                f"{geometry_id}: k={degree} wrong field shapes"
                            )
                        if w0.shape[1] != mass.shape[0]:
                            errors.append(
                                f"{geometry_id}: k={degree} field/mass mismatch"
                            )
                        if harmonic.shape != (mass.shape[0], expected_nullity[degree]):
                            errors.append(
                                f"{geometry_id}: k={degree} wrong harmonic nullity"
                            )
                        relative = np.asarray(
                            degree_group["relative_final_mass_norm"]
                        )
                        if relative.shape != (4,) or np.any(~np.isfinite(relative)):
                            errors.append(
                                f"{geometry_id}: k={degree} invalid final norms"
                            )
                        elif np.any(relative > 1.0 + 2e-5) or np.any(relative < 0.0):
                            errors.append(
                                f"{geometry_id}: k={degree} non-contractive result"
                            )
                        relative_norms[degree].extend(relative.tolist())
                        realized = np.asarray(
                            degree_group["realized_energy_fractions"]
                        )
                        if realized.shape != (4, 3) or not np.allclose(
                            realized.sum(axis=1), 1.0, atol=2e-5
                        ):
                            errors.append(
                                f"{geometry_id}: k={degree} invalid energy fractions"
                            )
                        if deep:
                            w0_values = np.asarray(w0, dtype=np.float64)
                            wT_values = np.asarray(wT, dtype=np.float64)
                            mass_values = np.asarray(mass, dtype=np.float64)
                            norm0 = np.sqrt(
                                np.sum(w0_values * w0_values * mass_values[None, :], axis=1)
                            )
                            if not np.allclose(norm0, 1.0, atol=1e-5):
                                errors.append(
                                    f"{geometry_id}: k={degree} initial M-norm is not one"
                                )
                            if harmonic.shape[1]:
                                harmonic_values = np.asarray(
                                    harmonic, dtype=np.float64
                                )
                                coefficients0 = harmonic_values.T @ (
                                    mass_values[:, None] * w0_values.T
                                )
                                coefficientsT = harmonic_values.T @ (
                                    mass_values[:, None] * wT_values.T
                                )
                                coefficient_error = np.max(
                                    np.abs(coefficientsT - coefficients0),
                                    initial=0.0,
                                )
                                if coefficient_error > 2e-4:
                                    errors.append(
                                        f"{geometry_id}: k={degree} harmonic coefficients drifted "
                                        f"by {coefficient_error:.3e}"
                                    )
                                observed_energy = np.sum(
                                    coefficients0 * coefficients0, axis=0
                                )
                                if not np.allclose(
                                    observed_energy,
                                    realized[:, 2],
                                    atol=3e-4,
                                ):
                                    errors.append(
                                        f"{geometry_id}: k={degree} harmonic energy mismatch"
                                    )
        except OSError as exc:
            errors.append(f"{shard_path}: unreadable ({exc})")

    if require_complete:
        for protocol in "ABCD":
            for split, expected in EXPECTED_SPLIT_COUNTS.items():
                actual = counts[(protocol, split)]
                if actual != expected:
                    errors.append(
                        f"protocol={protocol} split={split}: expected {expected}, got {actual}"
                    )

    summary = {
        "format": "TopoBox-3D-HodgeHeat-audit-v1",
        "root": str(root.resolve()),
        "complete_required": require_complete,
        "deep": deep,
        "geometry_count": int(sum(counts.values())),
        "shard_count": len(shard_paths),
        "counts": {
            f"{protocol}/{split}": count
            for (protocol, split), count in sorted(counts.items())
        },
        "betti_counts": {
            f"{protocol}/{split}/b{beta1}{beta2}": count
            for (protocol, split, beta1, beta2), count in sorted(betti_counts.items())
        },
        "relative_final_mass_norm": {
            f"k{degree}": {
                "minimum": float(np.min(values)) if values else None,
                "mean": float(np.mean(values)) if values else None,
                "maximum": float(np.max(values)) if values else None,
            }
            for degree, values in relative_norms.items()
        },
        "error_count": len(errors),
        "errors": errors[:500],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.require_complete, args.deep)
    output = args.output or args.root / "audit_report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Audited {report['geometry_count']} geometries in {report['shard_count']} shards; "
        f"errors={report['error_count']}"
    )
    if report["errors"]:
        print("\n".join(report["errors"][:50]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
