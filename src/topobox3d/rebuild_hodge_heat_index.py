"""Rebuild the global Hodge-heat index after parallel protocol generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py

from .generate_hodge_heat_dataset import CONFIG_NAMES, ENERGY_FRACTIONS


def rebuild(root: Path, source_root: Path) -> list[dict]:
    records = []
    shard_paths = sorted(
        path for path in root.glob("protocol_*/*/shard_*.h5")
        if not path.name.endswith(".tmp.h5")
    )
    numerical = None
    for shard_path in shard_paths:
        with h5py.File(shard_path, "r") as handle:
            current = {
                "kappa": float(handle.attrs["kappa"]),
                "final_time": float(handle.attrs["final_time"]),
                "steps": int(handle.attrs["steps"]),
            }
            if numerical is None:
                numerical = current
            elif current != numerical:
                raise RuntimeError(
                    f"Inconsistent numerical parameters in {shard_path}: {current}"
                )
            for geometry_id, group in handle["samples"].items():
                records.append({
                    "geometry_id": geometry_id,
                    "protocol": str(group.attrs["protocol"]),
                    "split": str(group.attrs["split"]),
                    "beta1": int(group.attrs["beta1"]),
                    "beta2": int(group.attrs["beta2"]),
                    "n_vertices": int(group.attrs["n_vertices"]),
                    "shard": shard_path.relative_to(root).as_posix(),
                    "group": f"samples/{geometry_id}",
                })
    records.sort(key=lambda item: item["geometry_id"])
    if len({record["geometry_id"] for record in records}) != len(records):
        raise RuntimeError("Duplicate geometry IDs found across solution shards")
    (root / "index.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if records:
        with (root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    manifest = {
        "format": "TopoBox-3D-HodgeHeat-v1",
        "source_root": str(source_root.resolve()),
        "geometry_count": len(records),
        "degrees": [0, 1, 2],
        "config_names": list(CONFIG_NAMES),
        "energy_fractions_k1_k2": ENERGY_FRACTIONS.tolist(),
        "k0_policy": "four independent smooth mass-mean-zero fields; constant harmonic mode suppressed",
        **(numerical or {"kappa": 1.0, "final_time": 0.1, "steps": 100}),
        "initial_condition": {
            "correlation_length": 0.40,
            "filter_time": 0.10,
            "filter_passes": 3,
            "rbf_centers": 8,
            "base_seed": 20260723,
        },
        "boundary_condition": "homogeneous absolute",
        "mass_discretization": "positive diagonal Whitney mass; vertex-lumped M0",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/TopoBox-3D")
    )
    args = parser.parse_args()
    records = rebuild(args.root, args.source)
    print(f"Rebuilt index for {len(records)} geometries.")


if __name__ == "__main__":
    main()
