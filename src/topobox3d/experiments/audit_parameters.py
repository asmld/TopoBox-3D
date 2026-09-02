"""Write exact parameter counts for the locked six-model configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from .model_registry import (
    MATCHED_CONFIGS,
    TORCH_MODEL_NAMES,
    build_torch_model,
    count_parameters,
)


def rigno_count(degree: int) -> int:
    rigno_path = str(
        Path(__file__).resolve().parents[3] / "third_party" / "RIGNO"
    )
    if rigno_path not in sys.path:
        sys.path.insert(0, rigno_path)
    from rigno.models.operator import Inputs
    from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder

    axis = jnp.linspace(0.1, 0.9, 3)
    coordinates = jnp.stack(
        jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    builder = RegionInteractionGraphBuilder(
        periodic=False,
        rmesh_levels=2,
        subsample_factor=2.0,
        overlap_factor_p2r=1.5,
        overlap_factor_r2p=1.5,
        node_coordinate_freqs=0,
    )
    metadata = builder.build_metadata(
        coordinates,
        coordinates,
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        key=jax.random.PRNGKey(0),
    )
    graphs = builder.build_graphs(metadata)
    channels = 5 if degree == 0 else 9
    inputs = Inputs(
        u=jnp.ones((1, 1, len(coordinates), 1)),
        c=jnp.ones((1, 1, len(coordinates), channels)),
        x_inp=coordinates[None, None],
        x_out=coordinates[None, None],
        t=0.0,
        tau=0.1,
    )
    cfg = MATCHED_CONFIGS["rigno"]
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
    params = model.init(
        {"params": jax.random.PRNGKey(1)}, inputs, graphs=graphs, key=None
    )["params"]
    return int(
        sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(params))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/parameter_counts.json"),
    )
    args = parser.parse_args()
    counts = {}
    for degree in (0, 1, 2):
        counts[f"k{degree}"] = {
            name: count_parameters(build_torch_model(name, degree))
            for name in TORCH_MODEL_NAMES
        }
        counts[f"k{degree}"]["rigno"] = rigno_count(degree)
    all_counts = [value for row in counts.values() for value in row.values()]
    result = {
        "target": 1_170_000,
        "counts": counts,
        "minimum": min(all_counts),
        "maximum": max(all_counts),
        "max_over_min": max(all_counts) / min(all_counts),
        "configs": MATCHED_CONFIGS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
