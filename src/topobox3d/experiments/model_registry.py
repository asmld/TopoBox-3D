"""Parameter-matched model construction and native forward adapters."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_NAMES = ("mgn-lite", "rigno", "transolver", "gnot", "gaot", "tno")
TORCH_MODEL_NAMES = tuple(name for name in MODEL_NAMES if name != "rigno")
MODEL_OUTPUT_NAMES = {
    "mgn-lite": "MGN-lite",
    "rigno": "RIGNO",
    "transolver": "Transolver",
    "gnot": "GNOT",
    "gaot": "GAOT",
    "tno": "TNO",
}
# Sparse CUDA matrix multiplication in TNO does not support bfloat16 in the
# pinned PyTorch/CUDA stack.
BFLOAT16_MODEL_NAMES = ("mgn-lite", "transolver", "gnot", "gaot")

# These settings give 1.135M--1.212M trainable parameters for k=1/2 inputs.
# They deliberately match parameter count rather than forcing architectures to
# share a width, depth, or tokenization that is not native to the method.
MATCHED_CONFIGS: dict[str, dict[str, Any]] = {
    "mgn-lite": {"width": 160, "layers": 6},
    "rigno": {
        "node_latent_size": 128,
        "edge_latent_size": 128,
        "processor_steps": 7,
        "mlp_hidden_layers": 1,
    },
    "transolver": {
        "n_hidden": 192,
        "n_layers": 6,
        "n_head": 8,
        "slice_num": 32,
    },
    "gnot": {
        "n_hidden": 96,
        "n_layers": 4,
        "n_head": 8,
        "n_experts": 2,
        "n_inner": 2,
    },
    "gaot": {
        "magno_hidden": 48,
        "lifting_channels": 18,
        "transformer_hidden": 128,
        "transformer_layers": 4,
        "n_head": 8,
        "radius": 0.2,
        "latent_tokens_size": (8, 4, 4),
    },
    "tno": {"width": 170, "layers": 4, "use_harmonic": True},
}


def input_channels(degree: int) -> int:
    """Return geometry-plus-field channels for an active k-simplex."""

    if degree == 0:
        return 6
    if degree in (1, 2):
        return 10
    raise ValueError(f"degree must be 0, 1, or 2; got {degree}")


def _prepend(relative: str) -> None:
    path = str(PROJECT_ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class _CochainGAOT(nn.Module):
    """Add a fiber-aware local readout to GAOT for oriented cochains.

    Native GAOT decodes latent features using query coordinates alone.  For
    k=1/2, a canonical cochain value also depends on the queried simplex's
    orientation and measure.  The collocated point-data features contain that
    information, so this small branch restores it without changing GAOT's
    nonlocal encoder/processor/decoder path.
    """

    def __init__(self, gaot: nn.Module, channels: int, hidden_channels: int = 32):
        super().__init__()
        self.gaot = gaot
        self.local_decoder = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, *, pndata: torch.Tensor, **gaot_inputs):
        return self.gaot(pndata=pndata, **gaot_inputs) + self.local_decoder(pndata)


def build_torch_model(name: str, degree: int) -> nn.Module:
    """Construct one of the five PyTorch models at the matched scale."""

    key = name.lower()
    channels = input_channels(degree)
    if key == "mgn-lite":
        from topobox3d.models.mgn import MGNLite

        cfg = MATCHED_CONFIGS[key]
        return MGNLite(
            node_in=channels,
            edge_in=4,
            out_channels=1,
            width=cfg["width"],
            layers=cfg["layers"],
        )
    if key == "transolver":
        _prepend("third_party/Transolver/PDE-Solving-StandardBenchmark")
        from model.Transolver_Irregular_Mesh import Model

        cfg = MATCHED_CONFIGS[key]
        return Model(
            space_dim=3,
            fun_dim=channels,
            out_dim=1,
            n_layers=cfg["n_layers"],
            n_hidden=cfg["n_hidden"],
            n_head=cfg["n_head"],
            slice_num=cfg["slice_num"],
        )
    if key == "gnot":
        os.environ.setdefault("DGLBACKEND", "pytorch")
        _prepend("third_party/GNOT")
        from models.mmgpt import GNOT

        cfg = MATCHED_CONFIGS[key]
        return GNOT(
            trunk_size=channels + 3,
            branch_sizes=[channels + 3],
            space_dim=3,
            output_size=1,
            n_layers=cfg["n_layers"],
            n_hidden=cfg["n_hidden"],
            n_head=cfg["n_head"],
            n_experts=cfg["n_experts"],
            n_inner=cfg["n_inner"],
        )
    if key == "gaot":
        _prepend("third_party/GAOT")
        from src.core.default_configs import ModelArgsConfig, ModelConfig
        from src.model.gaot import GAOT
        from src.model.layers.attn import AttentionConfig, TransformerConfig
        from src.model.layers.magno import MAGNOConfig

        cfg = MATCHED_CONFIGS[key]
        magno = MAGNOConfig(
            coord_dim=3,
            radius=cfg["radius"],
            hidden_size=cfg["magno_hidden"],
            mlp_layers=2,
            lifting_channels=cfg["lifting_channels"],
            neighbor_search_method="torch_cluster",
            use_torch_scatter=True,
        )
        transformer = TransformerConfig(
            patch_size=2,
            hidden_size=cfg["transformer_hidden"],
            num_layers=cfg["transformer_layers"],
            positional_embedding="absolute",
            attn_config=AttentionConfig(
                num_heads=cfg["n_head"],
                num_kv_heads=cfg["n_head"],
            ),
        )
        config = ModelConfig(
            latent_tokens_size=cfg["latent_tokens_size"],
            args=ModelArgsConfig(magno=magno, transformer=transformer),
        )
        gaot = GAOT(input_size=channels, output_size=1, config=config)
        return gaot if degree == 0 else _CochainGAOT(gaot, channels)
    if key == "tno":
        from topobox3d.models.tno import TNO

        cfg = MATCHED_CONFIGS[key]
        return TNO(
            in_channels=[6, 10, 10],
            out_channels=[1, 1, 1],
            width=cfg["width"],
            layers=cfg["layers"],
            use_harmonic=cfg["use_harmonic"],
        )
    if key == "rigno":
        raise ValueError("RIGNO uses the separate JAX entry point")
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def move_supervision(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        batch["target_cochain"].to(device=device, non_blocking=True),
        batch["mass"].to(device=device, non_blocking=True),
    )


def forward_torch_model(
    name: str,
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """Run the native forward path and return the active scalar cochain."""

    key = name.lower()
    output: torch.Tensor
    if key == "mgn-lite":
        output = model(
            batch["nodes"].to(device),
            batch["edge_index"].to(device),
            batch["edge_attr"].to(device),
        )
    elif key == "transolver":
        output = model(batch["x"].to(device), batch["fx"].to(device))
    elif key == "gnot":
        import dgl

        graph = dgl.graph(
            (batch["edge_index"][0], batch["edge_index"][1]),
            num_nodes=len(batch["token_coordinates"]),
        )
        graph.ndata["x"] = batch["query_graph_x"]
        output = model(
            graph,
            batch["global_parameters"],
            batch["branch_inputs"],
        )
    elif key == "gaot":
        # Upstream MAGNO caches neighbor graphs by tensor *shape* only.  That
        # is valid for a fixed mesh, but TopoBox changes geometry every sample;
        # equal-sized meshes would otherwise reuse the wrong graph.  A graph
        # created during inference-mode validation can also make the next
        # training backward fail.  Rebuild neighbors for every geometry.
        gaot = model.gaot if isinstance(model, _CochainGAOT) else model
        gaot.encoder.neighbor_cache.clear()
        gaot.decoder.neighbor_cache.clear()
        output = model(
            latent_tokens_coord=batch["latent_tokens_coord"].to(device),
            xcoord=batch["xcoord"].to(device),
            pndata=batch["pndata"].to(device),
            query_coord=batch["query_coord"].to(device),
        )
    elif key == "tno":
        outputs = model(
            [value.to(device) for value in batch["input_cochains"]],
            {rank: value.to(device) for rank, value in batch["incidence"].items()},
            {
                rank: value.to(device)
                for rank, value in batch["harmonic_basis"].items()
            },
            {
                rank: value.to(device)
                for rank, value in batch["harmonic_mass"].items()
            },
        )
        output = outputs[batch["active_degree"]]
    else:
        raise ValueError(f"{name!r} is not a PyTorch benchmark model")
    scale = batch["cochain_scale"].to(
        device=output.device, dtype=torch.float32
    )
    return output.float() * scale
