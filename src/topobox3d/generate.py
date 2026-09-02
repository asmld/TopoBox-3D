"""Command-line entry point for TopoBox-3D generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from topobox3d.config import DatasetConfig, PROTOCOLS, save_config, split_size
    from topobox3d.geometry import sample_geometry
    from topobox3d.mesh import generate_mesh
else:
    from .config import DatasetConfig, PROTOCOLS, save_config, split_size
    from .geometry import sample_geometry
    from .mesh import generate_mesh


SPLITS = ("train", "validation", "test_iid", "test_ood")


def balanced_pairs(pairs: list[tuple[int, int]], count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    order = np.arange(len(pairs))
    rng.shuffle(order)
    offset = int(rng.integers(len(pairs)))
    return [pairs[int(order[(offset + i) % len(pairs)])] for i in range(count)]


def geometry_seed(base_seed: int, protocol: str, split: str, index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{protocol}:{split}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


def generate_dataset(
    output_root: Path,
    config: DatasetConfig,
    protocols: list[str],
    splits: list[str],
    overwrite: bool = False,
    limit: int | None = None,
    manifest_name: str = "manifest.csv",
    write_config: bool = True,
    index_start: int = 0,
    index_end: int | None = None,
) -> None:
    if write_config:
        save_config(config, output_root)
    manifest_rows: list[dict] = []
    existing_manifest = output_root / manifest_name
    if existing_manifest.exists():
        with existing_manifest.open("r", newline="", encoding="utf-8") as handle:
            manifest_rows.extend(csv.DictReader(handle))
    existing_ids = {row["geometry_id"] for row in manifest_rows}

    for protocol in protocols:
        protocol_spec = PROTOCOLS[protocol]
        for split in splits:
            requested = split_size(config, split)
            count = min(requested, limit) if limit is not None else requested
            seed = geometry_seed(config.seed, protocol, split, 0)
            split_rng = np.random.default_rng(seed)
            is_ood = split == "test_ood"
            pairs = protocol_spec["ood_pairs" if is_ood else "iid_pairs"]
            family = protocol_spec["ood_family" if is_ood else "iid_family"]
            targets = balanced_pairs(pairs, count, split_rng)

            stop = min(count, index_end) if index_end is not None else count
            if index_start < 0 or index_start > stop:
                raise ValueError(f"Invalid index range [{index_start}, {stop}) for split size {count}")
            for index in range(index_start, stop):
                beta1, beta2 = targets[index]
                geometry_id = f"P{protocol}_{split}_{index:04d}_b{beta1}{beta2}"
                sample_dir = output_root / f"protocol_{protocol}" / split / geometry_id
                metadata_path = sample_dir / "metadata.json"
                if metadata_path.exists() and not overwrite:
                    if geometry_id not in existing_ids:
                        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                        manifest_rows.append(_manifest_row(saved, sample_dir, output_root))
                        existing_ids.add(geometry_id)
                    print(f"SKIP {geometry_id}", flush=True)
                    continue

                rng = np.random.default_rng(geometry_seed(config.seed, protocol, split, index + 1))
                spec = sample_geometry(rng, beta1, beta2, family, config.geometry)
                metadata = {
                    "geometry_id": geometry_id,
                    "protocol": protocol,
                    "protocol_name": protocol_spec["name"],
                    "split": split,
                    "is_ood": is_ood,
                    "beta1": beta1,
                    "beta2": beta2,
                    "geometry_family": family,
                    "seed": geometry_seed(config.seed, protocol, split, index + 1),
                }
                print(f"MESH {geometry_id}", flush=True)
                result = generate_mesh(spec, config, sample_dir, metadata)
                manifest_rows = [row for row in manifest_rows if row.get("geometry_id") != geometry_id]
                manifest_rows.append(_manifest_row(result, sample_dir, output_root))
                existing_ids.add(geometry_id)
                _write_manifest(existing_manifest, manifest_rows)
    _write_manifest(existing_manifest, manifest_rows)


def _manifest_row(result: dict, sample_dir: Path, root: Path) -> dict:
    topo, quality = result["topology"], result["mesh_quality"]
    return {
        "geometry_id": result["geometry_id"], "protocol": result["protocol"], "split": result["split"],
        "is_ood": int(result["is_ood"]), "geometry_family": result["geometry_family"],
        "beta1": result["beta1"], "beta2": result["beta2"],
        "n_vertices": topo["n_vertices"], "n_tetrahedra": topo["n_tetrahedra"],
        "min_clearance": result["geometry"]["measured_min_clearance"],
        "quality_min": quality["minimum"], "quality_mean": quality["mean"],
        "relative_path": sample_dir.relative_to(root).as_posix(),
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: row["geometry_id"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--formal", action="store_true", help="Use the formal 800/120/200/200 scale.")
    parser.add_argument("--train", type=int)
    parser.add_argument("--validation", type=int)
    parser.add_argument("--test-iid", type=int)
    parser.add_argument("--test-ood", type=int)
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--limit", type=int, help="Generate at most N samples per selected split (smoke testing).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-end", type=int)
    args = parser.parse_args()
    if args.formal:
        config = DatasetConfig(name="TopoBox-3D", train=800, validation=120, test_iid=200, test_ood=200)
        default_output = Path("data/TopoBox-3D")
    else:
        config = DatasetConfig()
        default_output = Path("data/TopoBox-3D-mini")
    overrides = {
        "train": args.train, "validation": args.validation,
        "test_iid": args.test_iid, "test_ood": args.test_ood,
    }
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    output = args.output or default_output
    generate_dataset(
        output, config, args.protocols, args.splits, args.overwrite, args.limit,
        manifest_name=args.manifest_name, write_config=not args.skip_config,
        index_start=args.index_start, index_end=args.index_end,
    )


if __name__ == "__main__":
    main()
