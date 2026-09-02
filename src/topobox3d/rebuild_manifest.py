"""Rebuild one consolidated raw-data manifest by scanning sample metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rebuild(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("protocol_*/*/*/metadata.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        topology, quality = item["topology"], item["mesh_quality"]
        rows.append({
            "geometry_id": item["geometry_id"], "protocol": item["protocol"], "split": item["split"],
            "is_ood": int(item["is_ood"]), "geometry_family": item["geometry_family"],
            "beta1": item["beta1"], "beta2": item["beta2"],
            "n_vertices": topology["n_vertices"], "n_tetrahedra": topology["n_tetrahedra"],
            "min_clearance": item["geometry"]["measured_min_clearance"],
            "quality_min": quality["minimum"], "quality_mean": quality["mean"],
            "relative_path": path.parent.relative_to(root).as_posix(),
        })
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"WROTE {len(rows)} records to {root / 'manifest.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/TopoBox-3D"))
    args = parser.parse_args()
    rebuild(args.root)


if __name__ == "__main__":
    main()
