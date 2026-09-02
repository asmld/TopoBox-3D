"""Generate fixed-time Hodge-heat labels for the complete TopoBox-3D dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import time
import zlib

import h5py
import numpy as np

from .hodge_heat import (
    build_resolvent_solver,
    build_hodge_systems,
    generate_initial_condition,
    harmonic_basis_and_spectrum,
    load_geometry,
    solve_fixed_time_batch,
    validate_system,
)


CONFIG_NAMES = (
    "non_harmonic",
    "weak_harmonic",
    "balanced",
    "strong_harmonic",
)

# The research-plan table is interpreted as unnormalized energy ratios.
ENERGY_FRACTIONS = np.asarray(
    (
        (1 / 2, 1 / 2, 0.0),
        (4 / 9, 4 / 9, 1 / 9),
        (1 / 3, 1 / 3, 1 / 3),
        (1 / 4, 1 / 4, 1 / 2),
    ),
    dtype=np.float64,
)


def _stable_seed(base_seed: int, geometry_id: str, degree: int, config: int) -> int:
    geometry_code = zlib.crc32(geometry_id.encode("utf-8"))
    return int(
        (base_seed + geometry_code + 1009 * degree + 104729 * config)
        % (2**32 - 1)
    )


def _mass_norm(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(values * values * mass[None, :], axis=1))


def solve_geometry(task: dict) -> dict:
    """Worker-safe solve for one geometry."""

    sample_dir = Path(task["sample_dir"])
    geometry, metadata = load_geometry(sample_dir)
    systems, derivatives, masses = build_hodge_systems(geometry, metadata)
    degrees: dict[str, dict] = {}

    for degree, system in enumerate(systems):
        stored_guess = geometry.get(f"harmonic_basis_{degree}")
        harmonic, spectrum = harmonic_basis_and_spectrum(system, stored_guess)
        filter_solver = build_resolvent_solver(
            system, task["filter_time"], task["filter_passes"]
        )
        initials = []
        requested = []
        realized = []
        seeds = []

        for config_index, fractions in enumerate(ENERGY_FRACTIONS):
            seed = _stable_seed(
                task["seed"], metadata["geometry_id"], degree, config_index
            )
            # k=0 is the connected-domain negative control: four independent
            # smooth zero-mean fields, all with the constant mode suppressed.
            initial = generate_initial_condition(
                degree,
                geometry,
                system,
                derivatives,
                masses[degree + 1] if degree < 3 else None,
                harmonic,
                seed=seed,
                correlation_length=task["correlation_length"],
                energy_fractions=tuple(float(value) for value in fractions),
                filter_time=task["filter_time"],
                filter_passes=task["filter_passes"],
                rbf_centers=task["rbf_centers"],
                scalar_harmonic_fraction=0.0,
                filter_solver=filter_solver,
            )
            initials.append(initial.values)
            requested.append(initial.requested_energy_fractions)
            realized.append(initial.realized_energy_fractions)
            seeds.append(seed)

        w0 = np.stack(initials)
        wT = solve_fixed_time_batch(
            system,
            w0,
            final_time=task["final_time"],
            steps=task["steps"],
            kappa=task["kappa"],
        )
        initial_norm = _mass_norm(w0, system.mass)
        final_norm = _mass_norm(wT, system.mass)
        if not np.allclose(initial_norm, 1.0, atol=5e-6):
            raise RuntimeError(
                f"{metadata['geometry_id']} k={degree}: initial normalization failed"
            )
        if np.any(final_norm > initial_norm + 2e-6):
            raise RuntimeError(
                f"{metadata['geometry_id']} k={degree}: heat flow is not contractive"
            )
        checks = validate_system(system)
        harmonic_residual = (
            float(np.max(np.linalg.norm(system.stiffness @ harmonic, axis=0), initial=0.0))
            if harmonic.shape[1] else 0.0
        )
        degrees[str(degree)] = {
            "w0": w0.astype(np.float32),
            "wT": wT.astype(np.float32),
            "mass": system.mass.astype(np.float32),
            "harmonic_basis": harmonic.astype(np.float32),
            "low_positive_eigenvalues": spectrum.astype(np.float64),
            "requested_energy_fractions": np.asarray(requested, dtype=np.float32),
            "realized_energy_fractions": np.asarray(realized, dtype=np.float32),
            "relative_final_mass_norm": (final_norm / initial_norm).astype(np.float64),
            "seeds": np.asarray(seeds, dtype=np.uint32),
            "harmonic_stiffness_residual": harmonic_residual,
            "system_checks": checks,
        }

    return {
        "geometry_id": metadata["geometry_id"],
        "protocol": metadata["protocol"],
        "split": metadata["split"],
        "beta1": int(metadata["beta1"]),
        "beta2": int(metadata["beta2"]),
        "n_vertices": int(metadata["topology"]["n_vertices"]),
        "sample_dir": str(sample_dir),
        "degrees": degrees,
    }


def _write_array(group: h5py.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    kwargs = {}
    if values.size and values.ndim and values.dtype.kind not in "OUS":
        kwargs = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset(name, data=values, **kwargs)


def _write_result(samples: h5py.Group, result: dict) -> None:
    group = samples.create_group(result["geometry_id"])
    for attribute in (
        "geometry_id", "protocol", "split", "beta1", "beta2", "n_vertices",
    ):
        group.attrs[attribute] = result[attribute]
    for degree, payload in result["degrees"].items():
        degree_group = group.create_group(f"k{degree}")
        for name, values in payload.items():
            if name == "system_checks":
                degree_group.attrs[name] = json.dumps(values)
            elif np.isscalar(values):
                degree_group.attrs[name] = values
            else:
                _write_array(degree_group, name, values)


def _records_from_shard(path: Path, root: Path) -> list[dict]:
    records = []
    with h5py.File(path, "r") as handle:
        for geometry_id, group in handle["samples"].items():
            records.append({
                "geometry_id": geometry_id,
                "protocol": str(group.attrs["protocol"]),
                "split": str(group.attrs["split"]),
                "beta1": int(group.attrs["beta1"]),
                "beta2": int(group.attrs["beta2"]),
                "n_vertices": int(group.attrs["n_vertices"]),
                "shard": path.relative_to(root).as_posix(),
                "group": f"samples/{geometry_id}",
            })
    return records


def generate_split(
    source_root: Path,
    output_root: Path,
    protocol: str,
    split: str,
    args: argparse.Namespace,
) -> list[dict]:
    sample_dirs = sorted(
        path for path in (source_root / f"protocol_{protocol}" / split).glob("*")
        if (path / "mesh.npz").exists()
    )
    if args.limit is not None:
        sample_dirs = sample_dirs[:args.limit]
    output_dir = output_root / f"protocol_{protocol}" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for start in range(0, len(sample_dirs), args.shard_size):
        shard_number = start // args.shard_size
        shard_path = output_dir / f"shard_{shard_number:04d}.h5"
        members = sample_dirs[start:start + args.shard_size]
        if shard_path.exists() and not args.overwrite:
            records.extend(_records_from_shard(shard_path, output_root))
            print(
                f"SKIP protocol={protocol} split={split} shard={shard_number:04d}",
                flush=True,
            )
            continue

        tasks = [{
            "sample_dir": str(member),
            "seed": args.seed,
            "final_time": args.final_time,
            "steps": args.steps,
            "kappa": args.kappa,
            "correlation_length": args.correlation_length,
            "filter_time": args.initial_filter_time,
            "filter_passes": args.initial_filter_passes,
            "rbf_centers": args.rbf_centers,
        } for member in members]
        temporary = shard_path.with_suffix(".tmp.h5")
        if temporary.exists():
            temporary.unlink()
        started = time.perf_counter()
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "TopoBox-3D-HodgeHeat-v1"
            handle.attrs["protocol"] = protocol
            handle.attrs["split"] = split
            handle.attrs["kappa"] = args.kappa
            handle.attrs["final_time"] = args.final_time
            handle.attrs["steps"] = args.steps
            handle.create_dataset(
                "config_names",
                data=np.asarray(CONFIG_NAMES, dtype=h5py.string_dtype("utf-8")),
            )
            samples = handle.create_group("samples")
            if args.workers == 1:
                iterator = map(solve_geometry, tasks)
                for result in iterator:
                    _write_result(samples, result)
                    print(f"  solved {result['geometry_id']}", flush=True)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    for result in executor.map(solve_geometry, tasks):
                        _write_result(samples, result)
                        print(f"  solved {result['geometry_id']}", flush=True)
        temporary.replace(shard_path)
        elapsed = time.perf_counter() - started
        records.extend(_records_from_shard(shard_path, output_root))
        print(
            f"DONE protocol={protocol} split={split} shard={shard_number:04d} "
            f"samples={len(members)} seconds={elapsed:.1f}",
            flush=True,
        )
    return records


def _write_index(output_root: Path, records: list[dict], args: argparse.Namespace) -> None:
    records.sort(key=lambda item: item["geometry_id"])
    manifest = {
        "format": "TopoBox-3D-HodgeHeat-v1",
        "source_root": str(Path(args.source).resolve()),
        "geometry_count": len(records),
        "degrees": [0, 1, 2],
        "config_names": list(CONFIG_NAMES),
        "energy_fractions_k1_k2": ENERGY_FRACTIONS.tolist(),
        "k0_policy": "four independent smooth mass-mean-zero fields; constant harmonic mode suppressed",
        "kappa": args.kappa,
        "final_time": args.final_time,
        "steps": args.steps,
        "initial_condition": {
            "correlation_length": args.correlation_length,
            "filter_time": args.initial_filter_time,
            "filter_passes": args.initial_filter_passes,
            "rbf_centers": args.rbf_centers,
            "base_seed": args.seed,
        },
        "boundary_condition": "homogeneous absolute",
        "mass_discretization": "positive diagonal Whitney mass; vertex-lumped M0",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_root / "index.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if records:
        with (output_root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/TopoBox-3D"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument("--protocols", nargs="+", default=list("ABCD"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test_iid", "test_ood"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--correlation-length", type=float, default=0.40)
    parser.add_argument("--initial-filter-time", type=float, default=0.10)
    parser.add_argument("--initial-filter-passes", type=int, default=3)
    parser.add_argument("--rbf-centers", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for protocol in args.protocols:
        for split in args.splits:
            print(f"START protocol={protocol} split={split}", flush=True)
            records.extend(
                generate_split(args.source, args.output, protocol, split, args)
            )
            _write_index(args.output, records, args)
    _write_index(args.output, records, args)
    print(f"COMPLETE geometries={len(records)} output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
