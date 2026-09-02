"""Pack variable-size TopoBox geometries into training-friendly HDF5 shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


SHARED_ARRAYS = {
    "regular_grid_xyz", "regular_grid_normalized_xyz", "regular_grid_resolution",
    "geometry_feature_names", "regular_grid_feature_names",
}


def _write_array(group: h5py.Group, name: str, array: np.ndarray) -> None:
    array = np.asarray(array)
    if array.dtype.kind == "U":
        array = array.astype(h5py.string_dtype(encoding="utf-8"))
    kwargs = {}
    if array.size and array.ndim and array.dtype.kind not in "OUS":
        kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset(name, data=array, **kwargs)


def pack_split(
    source_root: Path,
    packed_root: Path,
    protocol: str,
    split: str,
    shard_size: int,
    overwrite: bool,
) -> list[dict]:
    sample_dirs = sorted((source_root / f"protocol_{protocol}" / split).glob("*"))
    sample_dirs = [path for path in sample_dirs if (path / "mesh.npz").exists()]
    output_dir = packed_root / f"protocol_{protocol}" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for shard_start in range(0, len(sample_dirs), shard_size):
        shard_number = shard_start // shard_size
        shard_path = output_dir / f"shard_{shard_number:04d}.h5"
        members = sample_dirs[shard_start:shard_start + shard_size]
        if shard_path.exists() and not overwrite:
            with h5py.File(shard_path, "r") as handle:
                for geometry_id in handle["samples"]:
                    group = handle["samples"][geometry_id]
                    records.append({
                        "geometry_id": geometry_id, "protocol": protocol, "split": split,
                        "shard": shard_path.relative_to(packed_root).as_posix(),
                        "group": f"samples/{geometry_id}", "n_vertices": int(group.attrs["n_vertices"]),
                        "beta1": int(group.attrs["beta1"]), "beta2": int(group.attrs["beta2"]),
                    })
            continue

        temporary = shard_path.with_suffix(".tmp.h5")
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "TopoBox-3D-HDF5-v1"
            handle.attrs["protocol"] = protocol
            handle.attrs["split"] = split
            samples_group = handle.create_group("samples")
            shared_group = handle.create_group("shared")
            for member_index, sample_dir in enumerate(members):
                metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
                with np.load(sample_dir / "mesh.npz", allow_pickle=False) as source:
                    if member_index == 0:
                        for name in sorted(SHARED_ARRAYS.intersection(source.files)):
                            _write_array(shared_group, name, source[name])
                    group = samples_group.create_group(metadata["geometry_id"])
                    for name in source.files:
                        if name not in SHARED_ARRAYS:
                            _write_array(group, name, source[name])
                group.attrs["n_vertices"] = metadata["topology"]["n_vertices"]
                group.attrs["beta1"] = metadata["beta1"]
                group.attrs["beta2"] = metadata["beta2"]
                group.attrs["is_ood"] = int(metadata["is_ood"])
                group.create_dataset("metadata_json", data=np.bytes_(json.dumps(metadata, ensure_ascii=False)))
                records.append({
                    "geometry_id": metadata["geometry_id"], "protocol": protocol, "split": split,
                    "shard": shard_path.relative_to(packed_root).as_posix(),
                    "group": f"samples/{metadata['geometry_id']}",
                    "n_vertices": metadata["topology"]["n_vertices"],
                    "beta1": metadata["beta1"], "beta2": metadata["beta2"],
                })
        temporary.replace(shard_path)
    return records


def pack_dataset(source_root: Path, packed_root: Path, shard_size: int = 50, overwrite: bool = False) -> None:
    records: list[dict] = []
    for protocol in "ABCD":
        for split in ("train", "validation", "test_iid", "test_ood"):
            print(f"PACK protocol={protocol} split={split}", flush=True)
            records.extend(pack_split(source_root, packed_root, protocol, split, shard_size, overwrite))
    records.sort(key=lambda item: item["geometry_id"])
    with (packed_root / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    if records:
        with (packed_root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=Path("data/TopoBox-3D"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or args.source / "packed"
    pack_dataset(args.source, output, args.shard_size, args.overwrite)


if __name__ == "__main__":
    main()
