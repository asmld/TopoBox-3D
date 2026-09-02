"""Validate Hodge-heat shards and all seven model data adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pde_dataset import TopoBoxPDEDataset, make_model_dataloader


MODELS = ("mgn-lite", "rigno", "transolver", "gnot", "gaot", "tno")


def validate(
    geometry_root: Path,
    solution_root: Path,
    protocol: str,
    split: str,
) -> list[str]:
    errors = []
    dataset = TopoBoxPDEDataset(
        geometry_root, solution_root, protocol, split
    )
    seen = set()
    for index in range(min(len(dataset), 12)):
        sample = dataset[index]
        seen.add((sample.degree, sample.config_name))
        expected = len((
            sample.geometry.data["points"],
            sample.geometry.data["edges"],
            sample.geometry.data["faces"],
        )[sample.degree])
        if sample.w0.shape != (expected,) or sample.wT.shape != (expected,):
            errors.append(f"{sample.geometry_id}: wrong k={sample.degree} cochain shape")
        norm0 = np.sqrt(np.dot(sample.w0 * sample.mass, sample.w0))
        normT = np.sqrt(np.dot(sample.wT * sample.mass, sample.wT))
        if not np.isclose(norm0, 1.0, atol=1e-5) or normT > norm0 + 1e-5:
            errors.append(f"{sample.geometry_id}: invalid k={sample.degree} norms")
        for model in MODELS:
            adapted = sample.for_model(model)
            if "target_cochain" not in adapted or "mass" not in adapted:
                errors.append(f"{sample.geometry_id}:{model}: missing targets")
            if adapted["target_simplex_field"].shape != (expected, 1):
                errors.append(
                    f"{sample.geometry_id}:{model}: wrong simplex target"
                )
            if model == "tno":
                rank = sample.degree
                if adapted["input_cochains"][rank].shape[0] != expected:
                    errors.append(f"{sample.geometry_id}:TNO wrong active rank")
                if set(adapted["harmonic_basis"]) != {rank}:
                    errors.append(
                        f"{sample.geometry_id}:TNO wrong harmonic ranks"
                    )
                if set(adapted["harmonic_mass"]) != {rank}:
                    errors.append(
                        f"{sample.geometry_id}:TNO wrong harmonic mass ranks"
                    )
                if adapted["harmonic_mass"][rank].shape != (expected,):
                    errors.append(
                        f"{sample.geometry_id}:TNO wrong harmonic mass shape"
                    )
                continue

            geometry_channels = 5 if sample.degree == 0 else 9
            input_channels = geometry_channels + 1
            if adapted["token_coordinates"].shape != (expected, 3):
                errors.append(
                    f"{sample.geometry_id}:{model}: wrong token coordinates"
                )
            if adapted["token_geometry"].shape != (
                expected, geometry_channels
            ):
                errors.append(
                    f"{sample.geometry_id}:{model}: wrong token geometry"
                )
            if adapted["token_features"].shape != (
                expected, input_channels
            ):
                errors.append(
                    f"{sample.geometry_id}:{model}: wrong token features"
                )
            if adapted["input_field"].shape != (expected, 1):
                errors.append(
                    f"{sample.geometry_id}:{model}: wrong input cochain field"
                )
            if model in ("mgn-lite", "gnot"):
                if adapted["edge_index"].shape[0] != 2:
                    errors.append(
                        f"{sample.geometry_id}:{model}: wrong adjacency"
                    )
                if adapted["edge_attr"].shape != (
                    adapted["edge_index"].shape[1], 4
                ):
                    errors.append(
                        f"{sample.geometry_id}:{model}: "
                        "wrong adjacency features"
                    )
            if model == "mgn-lite":
                if adapted["nodes"].shape != (
                    expected, input_channels
                ):
                    errors.append(f"{sample.geometry_id}:MGN wrong nodes")
            if model == "rigno":
                if adapted["u"].shape != (1, 1, expected, 1):
                    errors.append(f"{sample.geometry_id}:RIGNO wrong u shape")
                if adapted["c"].shape != (
                    1, 1, expected, geometry_channels
                ):
                    errors.append(f"{sample.geometry_id}:RIGNO wrong c shape")
                if adapted["x_batched"].shape != (1, 1, expected, 3):
                    errors.append(
                        f"{sample.geometry_id}:RIGNO wrong batched coordinates"
                    )
            if model == "transolver" and adapted["fx"].shape != (
                1, expected, input_channels
            ):
                errors.append(f"{sample.geometry_id}:Transolver wrong fx")
            if model == "gnot":
                if adapted["query_graph_x"].shape != (
                    expected, geometry_channels
                ):
                    errors.append(
                        f"{sample.geometry_id}:GNOT wrong query tokens"
                    )
                if adapted["input_function_graph_x"].shape != (
                    expected, input_channels
                ):
                    errors.append(
                        f"{sample.geometry_id}:GNOT wrong branch tokens"
                    )
                if adapted["branch_inputs"][0].shape != (
                    1, expected, input_channels
                ):
                    errors.append(
                        f"{sample.geometry_id}:GNOT wrong batched branch"
                    )
                if adapted["global_parameters"].shape != (1, 0):
                    errors.append(
                        f"{sample.geometry_id}:GNOT wrong global parameters"
                    )
            if model == "gaot" and adapted["pndata"].shape != (
                1, expected, input_channels
            ):
                errors.append(f"{sample.geometry_id}:GAOT wrong pndata")
    dataset.close()
    for model in MODELS:
        model_dataset, loader = make_model_dataloader(
            geometry_root,
            solution_root,
            protocol,
            split,
            model,
            degrees=(1,),
            configs=("balanced",),
        )
        batch = next(iter(loader))
        if batch["degree"] != 1 or batch["config_name"] != "balanced":
            errors.append(f"{model}: DataLoader filter/collation failed")
        model_dataset.close()
    if len(seen) < 12:
        errors.append(f"Expected 12 degree/config combinations, found {len(seen)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-root", type=Path, default=Path("data/TopoBox-3D/packed")
    )
    parser.add_argument(
        "--solution-root", type=Path, default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument(
        "--protocol",
        default="A",
        help="A, B, C, D, or 'all' (default: A)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="train, validation, test_iid, test_ood, or 'all' (default: train)",
    )
    args = parser.parse_args()
    protocols = list("ABCD") if args.protocol.lower() == "all" else [args.protocol]
    splits = (
        ["train", "validation", "test_iid", "test_ood"]
        if args.split.lower() == "all"
        else [args.split]
    )
    errors = []
    for protocol in protocols:
        for split in splits:
            errors.extend(
                validate(
                    args.geometry_root,
                    args.solution_root,
                    protocol,
                    split,
                )
            )
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(
        "Hodge-heat shards and all seven model adapters passed "
        f"for {len(protocols) * len(splits)} protocol/split combination(s)."
    )


if __name__ == "__main__":
    main()
