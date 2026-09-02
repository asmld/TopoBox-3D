"""Linux/CUDA-JAX training entry point for the official RIGNO implementation.

TopoBox geometries have different graph sizes.  Samples are therefore padded
to power-of-two graph buckets before entering jitted train/evaluation steps.
The physical-node mass is zero in the padded region, so padded predictions do
not contribute to the objective.  Graph metadata is cached per geometry to
avoid rebuilding Delaunay graphs on every epoch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time

# JAX reads XLA_FLAGS while it is imported.  The CUDA GEMM autotuner bundled
# with jaxlib 0.4.38 can emit invalid HLO integer literals for RIGNO's large
# irregular graphs, leaving a worker stuck in its first JIT compilation.
# Disable that path before importing JAX; the scheduler also exports the same
# flags so direct and supervised launches behave identically.
_SAFE_XLA_FLAGS = (
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_autotune_level=0",
)
_existing_xla_flags = os.environ.get("XLA_FLAGS", "")
for _flag in _SAFE_XLA_FLAGS:
    if _flag.split("=", 1)[0] not in _existing_xla_flags:
        _existing_xla_flags = f"{_existing_xla_flags} {_flag}".strip()
os.environ["XLA_FLAGS"] = _existing_xla_flags
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.serialization
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "RIGNO"))

from rigno.models.operator import Inputs
from rigno.models.rigno import (
    RIGNO,
    RegionInteractionGraphBuilder,
    RegionInteractionGraphMetadata,
)

CONFIG_NAMES = (
    "non_harmonic",
    "weak_harmonic",
    "balanced",
    "strong_harmonic",
)
RIGNO_CONFIG = {
    "node_latent_size": 128,
    "edge_latent_size": 128,
    "processor_steps": 7,
    "mlp_hidden_layers": 1,
}
COCHAIN_NORMALIZATION = "k2_per_sample_input_rms"


@dataclass(frozen=True, order=True)
class GraphBucket:
    """Static array capacities used as a JAX compilation-cache key."""

    pnodes: int
    rnodes: int
    p2r_edges: int
    r2r_edges: int
    r2p_edges: int


def power_of_two_capacity(size: int) -> int:
    if size < 1:
        raise ValueError(f"bucket size must be positive, got {size}")
    return 1 << (size - 1).bit_length()


def graph_bucket(metadata: RegionInteractionGraphMetadata) -> GraphBucket:
    return GraphBucket(
        pnodes=power_of_two_capacity(metadata.x_pnodes_inp.shape[1] - 1),
        rnodes=power_of_two_capacity(metadata.x_rnodes.shape[1] - 1),
        p2r_edges=power_of_two_capacity(metadata.p2r_edge_indices.shape[1]),
        r2r_edges=power_of_two_capacity(metadata.r2r_edge_indices.shape[1]),
        r2p_edges=(
            power_of_two_capacity(metadata.r2p_edge_indices.shape[1])
            if metadata.r2p_edge_indices is not None
            else 0
        ),
    )


def _pad_nodes(values, real_count: int, capacity: int):
    """Move the builder's dummy node from ``real_count`` to ``capacity``."""
    values = np.asarray(values)
    result = np.zeros(
        (values.shape[0], capacity + 1, *values.shape[2:]),
        dtype=values.dtype,
    )
    result[:, :real_count] = values[:, :real_count]
    return result


def _pad_edges(
    values,
    capacity: int,
    dummy_sender: int,
    dummy_receiver: int,
):
    """Pad edge indices with a relocated dummy edge."""
    values = np.asarray(values)
    result = np.empty(
        (values.shape[0], capacity, values.shape[2]),
        dtype=np.uint32,
    )
    result[..., 0] = dummy_sender
    result[..., 1] = dummy_receiver
    # The builder appends exactly one dummy edge at the end.
    result[:, : values.shape[1] - 1] = values[:, :-1]
    return result


def pad_graph_metadata(
    metadata: RegionInteractionGraphMetadata,
) -> tuple[RegionInteractionGraphMetadata, GraphBucket]:
    """Pad nodes and edges while preserving the builder's dummy-node rules."""
    bucket = graph_bucket(metadata)
    real_pnodes = metadata.x_pnodes_inp.shape[1] - 1
    real_rnodes = metadata.x_rnodes.shape[1] - 1
    p2r = _pad_edges(
        metadata.p2r_edge_indices,
        bucket.p2r_edges,
        bucket.pnodes,
        bucket.rnodes,
    )
    r2r = _pad_edges(
        metadata.r2r_edge_indices,
        bucket.r2r_edges,
        bucket.rnodes,
        bucket.rnodes,
    )
    r2r_domains = np.zeros(
        (metadata.r2r_edge_domains.shape[0], bucket.r2r_edges, 2),
        dtype=np.uint8,
    )
    r2r_domains[:, : metadata.r2r_edge_domains.shape[1] - 1] = np.asarray(
        metadata.r2r_edge_domains[:, :-1]
    )
    if metadata.r2p_edge_indices is None:
        r2p = None
    else:
        r2p = _pad_edges(
            metadata.r2p_edge_indices,
            bucket.r2p_edges,
            bucket.rnodes,
            bucket.pnodes,
        )
    padded = RegionInteractionGraphMetadata(
        x_pnodes_inp=_pad_nodes(
            metadata.x_pnodes_inp, real_pnodes, bucket.pnodes
        ),
        x_pnodes_out=_pad_nodes(
            metadata.x_pnodes_out, real_pnodes, bucket.pnodes
        ),
        x_rnodes=_pad_nodes(
            metadata.x_rnodes, real_rnodes, bucket.rnodes
        ),
        r_rnodes=_pad_nodes(
            metadata.r_rnodes, real_rnodes, bucket.rnodes
        ),
        p2r_edge_indices=p2r,
        r2r_edge_indices=r2r,
        r2r_edge_domains=r2r_domains,
        r2p_edge_indices=r2p,
    )
    return padded, bucket


def cochain_scale(sample: dict, config_index: int) -> np.float32:
    if sample["degree"] != 2:
        return np.float32(1.0)
    values = np.asarray(sample["w0"][config_index], dtype=np.float64)
    return np.float32(max(float(np.sqrt(np.mean(np.square(values)))), 1e-8))


def simplex_arrays(degree: int, group) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(group["normalized_xyz"], dtype=np.float32)
    node_geometry = np.asarray(group["geometry_features"], dtype=np.float32)
    if degree == 0:
        return points, node_geometry
    if degree == 1:
        indices = np.asarray(group["edges"], dtype=np.int64)
        oriented = np.asarray(group["edge_vectors"], dtype=np.float32)
        measure = np.asarray(group["edge_lengths"], dtype=np.float32)[:, None]
    elif degree == 2:
        indices = np.asarray(group["faces"], dtype=np.int64)
        oriented = np.asarray(group["face_area_vectors"], dtype=np.float32)
        measure = np.asarray(group["face_areas"], dtype=np.float32)[:, None]
    else:
        raise ValueError(degree)
    coordinates = points[indices].mean(axis=1)
    geometry = np.concatenate(
        (node_geometry[indices].mean(axis=1), oriented, measure), axis=1
    )
    return coordinates.astype(np.float32), geometry.astype(np.float32)


class H5Cache:
    def __init__(self):
        self.handles: dict[Path, h5py.File] = {}

    def open(self, path: Path):
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        return self.handles[path]

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


class RIGNODataset:
    def __init__(
        self,
        geometry_root: Path,
        solution_root: Path,
        protocol: str,
        split: str,
        degree: int,
        graph_cache_dir: Path | None = None,
    ):
        geometry_records = json.loads(
            (geometry_root / "index.json").read_text(encoding="utf-8")
        )
        solution_records = json.loads(
            (solution_root / "index.json").read_text(encoding="utf-8")
        )
        by_id = {item["geometry_id"]: item for item in geometry_records}
        self.records = [
            (by_id[item["geometry_id"]], item)
            for item in solution_records
            if item["protocol"].upper() == protocol
            and item["split"] == split
        ]
        self.geometry_root = geometry_root
        self.solution_root = solution_root
        self.degree = degree
        self.cache = H5Cache()
        self.graph_cache_dir = graph_cache_dir
        self.graph_cache: dict[
            tuple[int, int], tuple[RegionInteractionGraphMetadata, GraphBucket]
        ] = {}

    def load_geometry(self, index: int) -> dict:
        geometry_record, solution_record = self.records[index]
        geometry_file = self.cache.open(
            self.geometry_root / geometry_record["shard"]
        )
        geometry_group = geometry_file[geometry_record["group"]]
        coordinates, conditions = simplex_arrays(self.degree, geometry_group)
        solution_file = self.cache.open(
            self.solution_root / solution_record["shard"]
        )
        degree_group = solution_file[solution_record["group"]][f"k{self.degree}"]
        return {
            "degree": self.degree,
            "geometry_id": solution_record["geometry_id"],
            "protocol": solution_record["protocol"],
            "split": solution_record["split"],
            "beta1": int(geometry_record["beta1"]),
            "beta2": int(geometry_record["beta2"]),
            "coordinates": coordinates,
            "conditions": conditions,
            "w0": np.asarray(degree_group["w0"], dtype=np.float32),
            "wT": np.asarray(degree_group["wT"], dtype=np.float32),
            "mass": np.asarray(degree_group["mass"], dtype=np.float32),
            "harmonic_basis": np.asarray(
                degree_group["harmonic_basis"], dtype=np.float32
            ),
            "realized_energy_fractions": np.asarray(
                degree_group["realized_energy_fractions"], dtype=np.float32
            ),
        }

    def close(self):
        self.cache.close()
        self.graph_cache.clear()

    def graph_cache_path(self, geometry_index: int, seed: int) -> Path | None:
        if self.graph_cache_dir is None:
            return None
        geometry_id = self.records[geometry_index][1]["geometry_id"]
        safe_id = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in geometry_id
        )
        return (
            self.graph_cache_dir
            / "power2_v1"
            / f"k{self.degree}"
            / f"{safe_id}.seed_{seed}.npz"
        )


def load_graph_metadata_cache(path: Path):
    with np.load(path, allow_pickle=False) as values:
        bucket_values = values["bucket"].tolist()
        metadata = RegionInteractionGraphMetadata(
            x_pnodes_inp=values["x_pnodes_inp"],
            x_pnodes_out=values["x_pnodes_out"],
            x_rnodes=values["x_rnodes"],
            r_rnodes=values["r_rnodes"],
            p2r_edge_indices=values["p2r_edge_indices"],
            r2r_edge_indices=values["r2r_edge_indices"],
            r2r_edge_domains=values["r2r_edge_domains"],
            r2p_edge_indices=(
                values["r2p_edge_indices"]
                if bool(values["has_r2p"].item())
                else None
            ),
        )
    return metadata, GraphBucket(*map(int, bucket_values))


def save_graph_metadata_cache(
    path: Path,
    metadata: RegionInteractionGraphMetadata,
    bucket: GraphBucket,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez(
        temporary,
        x_pnodes_inp=np.asarray(metadata.x_pnodes_inp),
        x_pnodes_out=np.asarray(metadata.x_pnodes_out),
        x_rnodes=np.asarray(metadata.x_rnodes),
        r_rnodes=np.asarray(metadata.r_rnodes),
        p2r_edge_indices=np.asarray(metadata.p2r_edge_indices),
        r2r_edge_indices=np.asarray(metadata.r2r_edge_indices),
        r2r_edge_domains=np.asarray(metadata.r2r_edge_domains),
        r2p_edge_indices=(
            np.asarray(metadata.r2p_edge_indices)
            if metadata.r2p_edge_indices is not None
            else np.empty((0,), dtype=np.uint32)
        ),
        has_r2p=np.asarray(metadata.r2p_edge_indices is not None),
        bucket=np.asarray(
            [
                bucket.pnodes,
                bucket.rnodes,
                bucket.p2r_edges,
                bucket.r2r_edges,
                bucket.r2p_edges,
            ],
            dtype=np.int64,
        ),
    )
    os.replace(temporary, path)


def make_graph(
    dataset: RIGNODataset,
    builder,
    geometry_index: int,
    coordinates: np.ndarray,
    seed: int,
):
    cache_key = (geometry_index, seed)
    cached = dataset.graph_cache.get(cache_key)
    if cached is None:
        disk_path = dataset.graph_cache_path(geometry_index, seed)
        if disk_path is not None and disk_path.exists():
            cached = load_graph_metadata_cache(disk_path)
        else:
            coordinates_jax = jnp.asarray(coordinates)
            metadata = builder.build_metadata(
                coordinates_jax,
                coordinates_jax,
                jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                key=jax.random.PRNGKey(seed),
            )
            cached = pad_graph_metadata(jax.device_get(metadata))
            if disk_path is not None:
                save_graph_metadata_cache(disk_path, *cached)
        dataset.graph_cache[cache_key] = cached
    metadata, bucket = cached
    return builder.build_graphs(metadata), bucket


def _pad_sample_array(values, capacity: int):
    values = np.asarray(values)
    result = np.zeros((capacity, *values.shape[1:]), dtype=values.dtype)
    result[: values.shape[0]] = values
    return result


def make_inputs(sample: dict, config_index: int, capacity: int) -> Inputs:
    coordinates = jnp.asarray(
        _pad_sample_array(sample["coordinates"], capacity)
    )[None, None]
    scale = cochain_scale(sample, config_index)
    return Inputs(
        u=jnp.asarray(
            _pad_sample_array(
                (sample["w0"][config_index] / scale)[:, None], capacity
            )
        )[None, None],
        c=jnp.asarray(
            _pad_sample_array(sample["conditions"], capacity)
        )[None, None],
        x_inp=coordinates,
        x_out=coordinates,
        t=0.0,
        tau=0.1,
    )


def relative_mse(prediction, target, mass):
    prediction = prediction[0, 0, :, 0]
    return jnp.sum(mass * jnp.square(prediction - target)) / (
        jnp.sum(mass * jnp.square(target)) + 1e-8
    )


def make_padded_target(sample: dict, config_index: int, capacity: int):
    scale = cochain_scale(sample, config_index)
    target = _pad_sample_array(
        (sample["wT"][config_index] / scale)[:, None], capacity
    )[:, 0]
    mass = _pad_sample_array(sample["mass"][:, None], capacity)[:, 0]
    return jnp.asarray(target), jnp.asarray(mass)


def create_jitted_steps(model, optimizer):
    @jax.jit
    def train_step(params, opt_state, inputs, graphs, target, mass):
        def loss_fn(current_params):
            prediction = model.apply(
                {"params": current_params},
                inputs,
                graphs=graphs,
                key=None,
            )
            return relative_mse(prediction, target, mass)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    @jax.jit
    def predict_step(params, inputs, graphs):
        return model.apply(
            {"params": params}, inputs, graphs=graphs, key=None
        )

    return train_step, predict_step


def numpy_metrics(prediction, target, mass, basis) -> dict[str, object]:
    error = prediction - target
    norm = lambda x: float(np.sqrt(np.sum(mass * np.square(x))))
    relative_l2 = norm(error) / (norm(target) + 1e-8)
    if basis.shape[1]:
        coefficients = basis.T @ (mass * error)
        target_coefficients = basis.T @ (mass * target)
        harmonic_error = basis @ coefficients
        target_harmonic = basis @ target_coefficients
    else:
        coefficients = np.empty((0,), dtype=np.float32)
        harmonic_error = np.zeros_like(error)
        target_harmonic = np.zeros_like(target)
    return {
        "relative_l2": relative_l2,
        "relative_mse": relative_l2**2,
        "harmonic_relative": norm(harmonic_error)
        / (norm(target_harmonic) + 1e-8),
        "nonharmonic_relative": norm(error - harmonic_error)
        / (norm(target - target_harmonic) + 1e-8),
        "harmonic_coefficient_error": coefficients.tolist(),
    }


def write_bytes_atomic(path: Path, payload: bytes):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def evaluate(
    params,
    dataset,
    builder,
    predict_step,
    jit_stats,
    output_path: Path | None,
    max_geometries: int,
) -> dict:
    values = defaultdict(list)
    by_geometry = defaultdict(list)
    writer = output_path.open("w", encoding="utf-8") if output_path else None
    started = time.perf_counter()
    count = 0
    try:
        limit = len(dataset.records)
        if max_geometries:
            limit = min(limit, max_geometries)
        for geometry_index in range(limit):
            sample = dataset.load_geometry(geometry_index)
            graphs, bucket = make_graph(
                dataset,
                builder,
                geometry_index,
                sample["coordinates"],
                geometry_index,
            )
            for config_index, config_name in enumerate(CONFIG_NAMES):
                inputs = make_inputs(sample, config_index, bucket.pnodes)
                call_started = time.perf_counter()
                output = predict_step(params, inputs, graphs)
                output_host = np.asarray(jax.device_get(output))
                if bucket not in jit_stats["predict_seen"]:
                    jit_stats["predict_seen"].add(bucket)
                    jit_stats["predict_first_call_seconds"][bucket] = (
                        time.perf_counter() - call_started
                    )
                real_count = sample["coordinates"].shape[0]
                prediction = (
                    output_host[0, 0, :real_count, 0]
                    * cochain_scale(sample, config_index)
                )
                metrics = numpy_metrics(
                    prediction,
                    sample["wT"][config_index],
                    sample["mass"],
                    sample["harmonic_basis"],
                )
                for key in (
                    "relative_l2",
                    "relative_mse",
                    "harmonic_relative",
                    "nonharmonic_relative",
                ):
                    values[key].append(float(metrics[key]))
                by_geometry[sample["geometry_id"]].append(
                    float(metrics["relative_l2"])
                )
                if writer:
                    writer.write(
                        json.dumps(
                            {
                                "geometry_id": sample["geometry_id"],
                                "protocol": sample["protocol"],
                                "split": sample["split"],
                                "degree": dataset.degree,
                                "config_name": config_name,
                                "beta1": sample["beta1"],
                                "beta2": sample["beta2"],
                                "realized_energy_fractions": sample[
                                    "realized_energy_fractions"
                                ][config_index].tolist(),
                                **metrics,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                count += 1
    finally:
        if writer:
            writer.close()
    elapsed = time.perf_counter() - started
    geometry_means = [np.mean(items) for items in by_geometry.values()]
    result = {
        key: {
            "mean": float(np.mean(items)),
            "median": float(np.median(items)),
        }
        for key, items in values.items()
    }
    result["geometry_clustered_relative_l2"] = {
        "mean": float(np.mean(geometry_means)),
        "median": float(np.median(geometry_means)),
        "geometry_count": len(geometry_means),
    }
    result.update(
        {
            "samples": count,
            "seconds": elapsed,
            "samples_per_second": count / max(elapsed, 1e-9),
        }
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=("A", "B", "C", "D"))
    parser.add_argument("--degree", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help=(
            "Deprecated compatibility option. Validation patience is reported "
            "but never stops fixed-budget training."
        ),
    )
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--all-configs-per-epoch",
        action="store_true",
        help="Use all four initial conditions instead of cycling one per geometry.",
    )
    parser.add_argument("--max-train-geometries", type=int, default=0)
    parser.add_argument("--max-eval-geometries", type=int, default=0)
    parser.add_argument(
        "--geometry-root", type=Path, default=Path("data/TopoBox-3D/packed")
    )
    parser.add_argument(
        "--solution-root", type=Path, default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/topobox3d"))
    parser.add_argument(
        "--jax-cache-dir",
        type=Path,
        default=Path(".jax_cache/rigno_xla_safe_v1"),
        help="Persistent JAX compilation cache shared by compatible runs.",
    )
    parser.add_argument(
        "--graph-cache-dir",
        type=Path,
        default=Path(".graph_cache/rigno"),
        help="Persistent padded graph-metadata cache shared across runs.",
    )
    parser.add_argument(
        "--precompute-graphs",
        action="store_true",
        help="Fill the persistent graph cache for this degree/seed and exit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume parameters, optimizer, history, and best state if present.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.jax_cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update(
        "jax_compilation_cache_dir", str(args.jax_cache_dir.resolve())
    )
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"RIGNO formal training requires JAX GPU, got {jax.default_backend()}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_dir = (
        args.output_root
        / "rigno"
        / f"protocol_{args.protocol}"
        / f"k{args.degree}"
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: RIGNODataset(
            args.geometry_root,
            args.solution_root,
            args.protocol,
            split,
            args.degree,
            args.graph_cache_dir,
        )
        for split in ("train", "validation", "test_iid", "test_ood")
    }
    cfg = RIGNO_CONFIG
    model = RIGNO(
        num_outputs=1,
        processor_steps=cfg["processor_steps"],
        node_latent_size=cfg["node_latent_size"],
        edge_latent_size=cfg["edge_latent_size"],
        mlp_hidden_layers=cfg["mlp_hidden_layers"],
        concatenate_t=True,
        concatenate_tau=True,
        conditioned_normalization=False,
        p_edge_masking=0.0,
    )
    builder = RegionInteractionGraphBuilder(
        periodic=False,
        rmesh_levels=2,
        subsample_factor=2.0,
        overlap_factor_p2r=1.5,
        overlap_factor_r2p=1.5,
        node_coordinate_freqs=0,
    )
    if args.precompute_graphs:
        limits = {}
        for split, dataset in datasets.items():
            configured_limit = (
                args.max_train_geometries
                if split == "train"
                else args.max_eval_geometries
            )
            limits[split] = (
                min(len(dataset.records), configured_limit)
                if configured_limit
                else len(dataset.records)
            )
        total = sum(limits.values())
        completed = 0
        started = time.perf_counter()
        try:
            for split, dataset in datasets.items():
                for geometry_index in range(limits[split]):
                    sample = dataset.load_geometry(geometry_index)
                    graph_seed = (
                        args.seed + geometry_index
                        if split == "train"
                        else geometry_index
                    )
                    make_graph(
                        dataset,
                        builder,
                        geometry_index,
                        sample["coordinates"],
                        graph_seed,
                    )
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        print(
                            json.dumps(
                                {
                                    "graph_cache": {
                                        "completed": completed,
                                        "total": total,
                                        "seconds": time.perf_counter()
                                        - started,
                                    }
                                }
                            ),
                            flush=True,
                        )
        finally:
            for dataset in datasets.values():
                dataset.close()
        return
    first = datasets["train"].load_geometry(0)
    first_graphs, first_bucket = make_graph(
        datasets["train"], builder, 0, first["coordinates"], args.seed
    )
    params = jax.jit(model.init)(
        {"params": jax.random.PRNGKey(args.seed)},
        make_inputs(first, 0, first_bucket.pnodes),
        graphs=first_graphs,
        key=None,
    )["params"]
    parameter_count = int(
        sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(params))
    )
    train_geometry_count = len(datasets["train"].records)
    if args.max_train_geometries:
        train_geometry_count = min(
            train_geometry_count, args.max_train_geometries
        )
    schedule = optax.cosine_decay_schedule(
        args.learning_rate,
        decay_steps=max(
            args.epochs
            * train_geometry_count
            * (4 if args.all_configs_per_epoch else 1),
            1,
        ),
        alpha=0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)
    train_step, predict_step = create_jitted_steps(model, optimizer)
    jit_stats = {
        "train_seen": set(),
        "predict_seen": set(),
        "train_first_call_seconds": {},
        "predict_first_call_seconds": {},
        "train_calls_by_bucket": defaultdict(int),
    }
    best_validation = math.inf
    best_params = params
    best_epoch = 0
    stale = 0
    start_epoch = 1
    history_path = run_dir / "history.json"
    last_params_path = run_dir / "last_params.msgpack"
    last_opt_state_path = run_dir / "last_opt_state.msgpack"
    last_state_path = run_dir / "last_state.json"
    history = []
    if (
        args.resume
        and last_params_path.exists()
        and last_opt_state_path.exists()
        and last_state_path.exists()
    ):
        last_state = json.loads(last_state_path.read_text(encoding="utf-8"))
        checkpoint_epochs = int(last_state.get("epochs", args.epochs))
        if checkpoint_epochs != args.epochs:
            raise RuntimeError(
                "Cannot resume a checkpoint created for "
                f"{checkpoint_epochs} epochs with --epochs {args.epochs}."
            )
        params = flax.serialization.from_bytes(
            params, last_params_path.read_bytes()
        )
        opt_state = flax.serialization.from_bytes(
            opt_state, last_opt_state_path.read_bytes()
        )
        start_epoch = int(last_state["epoch"]) + 1
        best_validation = float(last_state["best_validation"])
        best_epoch = int(last_state["best_epoch"])
        stale = int(last_state.get("stale", 0))
        if best_epoch and (run_dir / "best.msgpack").exists():
            best_params = flax.serialization.from_bytes(
                params, (run_dir / "best.msgpack").read_bytes()
            )
        else:
            best_params = params
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            order = list(range(len(datasets["train"].records)))
            random.Random(args.seed + epoch).shuffle(order)
            if args.max_train_geometries:
                order = order[: args.max_train_geometries]
            losses = []
            started = time.perf_counter()
            for geometry_index in order:
                sample = datasets["train"].load_geometry(geometry_index)
                graphs, bucket = make_graph(
                    datasets["train"],
                    builder,
                    geometry_index,
                    sample["coordinates"],
                    args.seed + geometry_index,
                )
                if args.all_configs_per_epoch:
                    config_order = list(range(4))
                    random.Random(
                        args.seed + epoch * 1000 + geometry_index
                    ).shuffle(config_order)
                else:
                    config_order = [
                        (epoch - 1 + geometry_index + args.seed) % 4
                    ]
                for config_index in config_order:
                    inputs = make_inputs(
                        sample, config_index, bucket.pnodes
                    )
                    target, mass = make_padded_target(
                        sample, config_index, bucket.pnodes
                    )
                    call_started = time.perf_counter()
                    params, opt_state, loss = train_step(
                        params, opt_state, inputs, graphs, target, mass
                    )
                    losses.append(float(jax.device_get(loss)))
                    if bucket not in jit_stats["train_seen"]:
                        jit_stats["train_seen"].add(bucket)
                        jit_stats["train_first_call_seconds"][bucket] = (
                            time.perf_counter() - call_started
                        )
                    jit_stats["train_calls_by_bucket"][bucket] += 1
            event = {
                "epoch": epoch,
                "train": {
                    "relative_mse": float(np.mean(losses)),
                    "samples": len(losses),
                    "seconds": time.perf_counter() - started,
                },
            }
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                validation = evaluate(
                    params,
                    datasets["validation"],
                    builder,
                    predict_step,
                    jit_stats,
                    None,
                    args.max_eval_geometries,
                )
                event["validation"] = validation
                score = float(validation["relative_mse"]["mean"])
                if score < best_validation:
                    best_validation = score
                    best_epoch = epoch
                    best_params = jax.tree_util.tree_map(
                        lambda value: np.asarray(jax.device_get(value)),
                        params,
                    )
                    stale = 0
                    write_bytes_atomic(
                        run_dir / "best.msgpack",
                        flax.serialization.to_bytes(best_params),
                    )
                    write_json_atomic(
                        run_dir / "best.json",
                        {
                            "epoch": epoch,
                            "best_validation": best_validation,
                            "parameter_count": parameter_count,
                            "cochain_normalization": COCHAIN_NORMALIZATION,
                        },
                    )
                else:
                    stale += 1
            history.append(event)
            write_json_atomic(history_path, history)
            write_bytes_atomic(
                last_params_path,
                flax.serialization.to_bytes(jax.device_get(params)),
            )
            write_bytes_atomic(
                last_opt_state_path,
                flax.serialization.to_bytes(jax.device_get(opt_state)),
            )
            write_json_atomic(
                last_state_path,
                {
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "best_validation": best_validation,
                    "best_epoch": best_epoch,
                    "stale": stale,
                },
            )
            print(json.dumps(event, ensure_ascii=False), flush=True)
        summaries = {}
        for split in ("validation", "test_iid", "test_ood"):
            summaries[split] = evaluate(
                best_params,
                datasets[split],
                builder,
                predict_step,
                jit_stats,
                run_dir / f"{split}.jsonl",
                args.max_eval_geometries,
            )
        memory_stats = jax.devices()[0].memory_stats() or {}
        result = {
            "model": "rigno",
            "protocol": args.protocol,
            "degree": args.degree,
            "seed": args.seed,
            "parameter_count": parameter_count,
            "cochain_normalization": COCHAIN_NORMALIZATION,
            "completed_epoch": int(history[-1]["epoch"]),
            "best_epoch": best_epoch,
            "best_validation_relative_mse": best_validation,
            "splits": summaries,
            "jax_jit_policy": "power_of_two_graph_buckets",
            "jax_compilation_cache_dir": str(args.jax_cache_dir.resolve()),
            "jit_buckets": {
                "train": [
                    {
                        **asdict(bucket),
                        "first_call_seconds": jit_stats[
                            "train_first_call_seconds"
                        ][bucket],
                        "calls": jit_stats["train_calls_by_bucket"][bucket],
                    }
                    for bucket in sorted(jit_stats["train_seen"])
                ],
                "predict": [
                    {
                        **asdict(bucket),
                        "first_call_seconds": jit_stats[
                            "predict_first_call_seconds"
                        ][bucket],
                    }
                    for bucket in sorted(jit_stats["predict_seen"])
                ],
            },
            "device_memory_stats": {
                key: int(value)
                for key, value in memory_stats.items()
                if isinstance(value, (int, np.integer))
            },
        }
        write_json_atomic(run_dir / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
