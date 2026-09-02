"""Inference-only pure-harmonic probes for the JAX RIGNO baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from . import train_rigno as tr


TASKS = ("A-k1", "A-k2", "B-k1", "C-k2", "D-k1", "D-k2")


def mass_norm(values: np.ndarray, mass: np.ndarray) -> float:
    return float(np.sqrt(np.sum(mass * np.square(values))))


def pure_inputs(sample: dict, target: np.ndarray, capacity: int):
    coordinates = jnp.asarray(
        tr._pad_sample_array(sample["coordinates"], capacity)
    )[None, None]
    if sample["degree"] == 2:
        scale = np.float32(
            max(float(np.sqrt(np.mean(np.square(target, dtype=np.float64)))), 1e-8)
        )
    else:
        scale = np.float32(1.0)
    inputs = tr.Inputs(
        u=jnp.asarray(
            tr._pad_sample_array((target / scale)[:, None], capacity)
        )[None, None],
        c=jnp.asarray(
            tr._pad_sample_array(sample["conditions"], capacity)
        )[None, None],
        x_inp=coordinates,
        x_out=coordinates,
        t=0.0,
        tau=0.1,
    )
    return inputs, scale


def selected_indices(dataset: tr.RIGNODataset, maximum: int) -> list[int]:
    positive = []
    for index in range(len(dataset.records)):
        sample = dataset.load_geometry(index)
        if sample["harmonic_basis"].shape[1] > 0:
            positive.append(index)
    if len(positive) <= maximum:
        return positive
    positions = np.linspace(0, len(positive) - 1, maximum, dtype=int)
    return [positive[position] for position in positions]


def build_model():
    cfg = tr.RIGNO_CONFIG
    model = tr.RIGNO(
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
    builder = tr.RegionInteractionGraphBuilder(
        periodic=False,
        rmesh_levels=2,
        subsample_factor=2.0,
        overlap_factor_p2r=1.5,
        overlap_factor_r2p=1.5,
        node_coordinate_freqs=0,
    )
    return model, builder


def evaluate_task(args, task: str, split: str) -> list[dict[str, object]]:
    protocol, degree_text = task.split("-k")
    degree = int(degree_text)
    dataset = tr.RIGNODataset(
        args.geometry_root,
        args.solution_root,
        protocol,
        split,
        degree,
        args.graph_cache_dir,
    )
    model, builder = build_model()
    indices = selected_indices(dataset, args.max_geometries)
    first_index = indices[0]
    first = dataset.load_geometry(first_index)
    first_graphs, first_bucket = tr.make_graph(
        dataset, builder, first_index, first["coordinates"], first_index
    )
    first_basis = first["harmonic_basis"]
    first_coefficients = np.ones(first_basis.shape[1], dtype=np.float32)
    first_coefficients /= np.sqrt(first_basis.shape[1])
    first_target = first_basis @ first_coefficients
    first_inputs, _ = pure_inputs(first, first_target, first_bucket.pnodes)
    params = jax.jit(model.init)(
        {"params": jax.random.PRNGKey(args.seed)},
        first_inputs,
        graphs=first_graphs,
        key=None,
    )["params"]
    run_dir = (
        args.results_root / "rigno" / f"protocol_{protocol}" / f"k{degree}"
        / f"seed_{args.seed}"
    )
    params = flax.serialization.from_bytes(
        params, (run_dir / "best.msgpack").read_bytes()
    )

    @jax.jit
    def predict(current_params, inputs, graphs):
        return model.apply(
            {"params": current_params}, inputs, graphs=graphs, key=None
        )

    rows = []
    try:
        for geometry_index in indices:
            sample = dataset.load_geometry(geometry_index)
            graphs, bucket = tr.make_graph(
                dataset,
                builder,
                geometry_index,
                sample["coordinates"],
                geometry_index,
            )
            basis = np.asarray(sample["harmonic_basis"], dtype=np.float32)
            coefficients = np.ones(basis.shape[1], dtype=np.float32)
            coefficients /= np.sqrt(basis.shape[1])
            target = basis @ coefficients
            inputs, scale = pure_inputs(sample, target, bucket.pnodes)
            output = predict(params, inputs, graphs)
            real_count = len(target)
            prediction = (
                np.asarray(jax.device_get(output))[0, 0, :real_count, 0]
                * scale
            )
            mass = np.asarray(sample["mass"], dtype=np.float32)
            error = prediction - target
            coefficient_error = basis.T @ (mass * error)
            harmonic_error = basis @ coefficient_error
            nonharmonic_error = error - harmonic_error
            rows.append(
                {
                    "model": "RIGNO",
                    "model_key": "rigno",
                    "protocol": protocol,
                    "degree": degree,
                    "task": task,
                    "seed": args.seed,
                    "split": split,
                    "geometry_id": sample["geometry_id"],
                    "beta1": sample["beta1"],
                    "beta2": sample["beta2"],
                    "harmonic_dimension": basis.shape[1],
                    "basis_index": 0,
                    "probe_mode": "combined",
                    "pure_harmonic_identity_error": mass_norm(error, mass),
                    "harmonic_preservation_error": mass_norm(
                        harmonic_error, mass
                    ),
                    "harmonic_to_nonharmonic_leakage": mass_norm(
                        nonharmonic_error, mass
                    ),
                    "predicted_norm": mass_norm(prediction, mass),
                    "coefficient_identity_error": float(
                        np.linalg.norm(coefficient_error)
                    ),
                }
            )
    finally:
        dataset.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument(
        "--splits", nargs="+", choices=("test_iid", "test_ood"),
        default=("test_iid", "test_ood"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-geometries", type=int, default=4)
    parser.add_argument(
        "--geometry-root", type=Path, default=Path("data/TopoBox-3D/packed")
    )
    parser.add_argument(
        "--solution-root", type=Path, default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("runs/topobox3d")
    )
    parser.add_argument(
        "--graph-cache-dir", type=Path, default=Path(".graph_cache/rigno")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/04_hodge_subspace_architecture/supporting_data/"
            "pure_harmonic_controlled_rigno.csv"
        ),
    )
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        pd.read_csv(args.output).to_dict("records")
        if args.append and args.output.exists()
        else []
    )
    for task in args.tasks:
        for split in args.splits:
            rows.extend(evaluate_task(args, task, split))
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"completed rigno {task} seed={args.seed} {split}", flush=True)


if __name__ == "__main__":
    main()
