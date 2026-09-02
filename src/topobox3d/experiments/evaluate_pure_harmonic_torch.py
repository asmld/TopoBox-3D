"""Inference-only pure-harmonic probes for the five PyTorch baselines."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topobox3d.pde_dataset import TopoBoxPDEDataset

from .model_registry import build_torch_model, forward_torch_model


MODEL_DIRS = {
    "mgn-lite": "MGN-lite",
    "transolver": "Transolver",
    "gnot": "GNOT",
    "gaot": "GAOT",
    "tno": "TNO",
}
TASKS = ("A-k1", "A-k2", "B-k1", "C-k2", "D-k1", "D-k2")


def mass_norm(values: np.ndarray, mass: np.ndarray) -> float:
    return float(np.sqrt(np.sum(mass * np.square(values))))


def selected_samples(dataset: TopoBoxPDEDataset, maximum: int):
    candidates = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample.harmonic_basis.shape[1] > 0:
            candidates.append(sample)
    if len(candidates) <= maximum:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, maximum, dtype=int)
    return [candidates[index] for index in indices]


@torch.inference_mode()
def evaluate_model_task(
    model_name: str,
    protocol: str,
    degree: int,
    seed: int,
    split: str,
    maximum_geometries: int,
    geometry_root: Path,
    solution_root: Path,
    results_root: Path,
    device: torch.device,
    probe_mode: str,
) -> list[dict[str, object]]:
    run_dir = (
        results_root / MODEL_DIRS[model_name] / f"protocol_{protocol}"
        / f"k{degree}" / f"seed_{seed}"
    )
    checkpoint = torch.load(
        run_dir / "best.pt", map_location=device, weights_only=False
    )
    model = build_torch_model(model_name, degree).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = TopoBoxPDEDataset(
        geometry_root,
        solution_root,
        protocol=protocol,
        split=split,
        degrees=(degree,),
        configs=("balanced",),
        cache_derived=True,
        cache_adjacency=model_name == "mgn-lite",
    )
    rows = []
    try:
        for sample in selected_samples(dataset, maximum_geometries):
            basis = np.asarray(sample.harmonic_basis, dtype=np.float32)
            mass = np.asarray(sample.mass, dtype=np.float32)
            if probe_mode == "combined":
                desired_vectors = [
                    np.ones(basis.shape[1], dtype=np.float32)
                    / np.sqrt(basis.shape[1])
                ]
            else:
                desired_vectors = [
                    np.eye(basis.shape[1], dtype=np.float32)[index]
                    for index in range(basis.shape[1])
                ]
            for basis_index, desired_coefficients in enumerate(desired_vectors):
                target = basis @ desired_coefficients
                probe = replace(
                    sample,
                    config_name=f"pure_harmonic_{basis_index}",
                    config_index=-1,
                    w0=target,
                    wT=target,
                    requested_energy_fractions=np.asarray(
                        [0.0, 0.0, 1.0], dtype=np.float32
                    ),
                    realized_energy_fractions=np.asarray(
                        [0.0, 0.0, 1.0], dtype=np.float32
                    ),
                    relative_final_mass_norm=1.0,
                )
                batch = probe.for_model(model_name)
                prediction_tensor = forward_torch_model(
                    model_name, model, batch, device
                )
                prediction = probe.prediction_to_cochain(prediction_tensor)
                error = prediction - target
                coefficients = basis.T @ (mass * error)
                harmonic_error = basis @ coefficients
                nonharmonic_error = error - harmonic_error
                predicted_coefficients = basis.T @ (mass * prediction)
                rows.append(
                    {
                        "model": MODEL_DIRS[model_name],
                        "model_key": model_name,
                        "protocol": protocol,
                        "degree": degree,
                        "task": f"{protocol}-k{degree}",
                        "seed": seed,
                        "split": split,
                        "geometry_id": sample.geometry_id,
                        "beta1": sample.beta1,
                        "beta2": sample.beta2,
                        "harmonic_dimension": basis.shape[1],
                        "basis_index": basis_index,
                        "probe_mode": probe_mode,
                        "pure_harmonic_identity_error": mass_norm(error, mass),
                        "harmonic_preservation_error": mass_norm(
                            harmonic_error, mass
                        ),
                        "harmonic_to_nonharmonic_leakage": mass_norm(
                            nonharmonic_error, mass
                        ),
                        "predicted_norm": mass_norm(prediction, mass),
                        "coefficient_identity_error": float(
                            np.linalg.norm(
                                predicted_coefficients - desired_coefficients
                            )
                        ),
                    }
                )
    finally:
        dataset.close()
        del model
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_DIRS),
        default=tuple(MODEL_DIRS),
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--max-geometries", type=int, default=10)
    parser.add_argument(
        "--probe-mode", choices=("combined", "basis"), default="combined"
    )
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
        "--output",
        type=Path,
        default=Path(
            "results/04_hodge_subspace_architecture/supporting_data/"
            "pure_harmonic_controlled_torch.csv"
        ),
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model_name in args.models:
        for task in args.tasks:
            protocol, degree_text = task.split("-k")
            degree = int(degree_text)
            for seed in args.seeds:
                for split in ("test_iid", "test_ood"):
                    rows.extend(
                        evaluate_model_task(
                            model_name,
                            protocol,
                            degree,
                            seed,
                            split,
                            args.max_geometries,
                            args.geometry_root,
                            args.solution_root,
                            args.results_root,
                            device,
                            args.probe_mode,
                        )
                    )
                    pd.DataFrame(rows).to_csv(args.output, index=False)
                    print(
                        f"completed {model_name} {task} seed={seed} {split}",
                        flush=True,
                    )
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
