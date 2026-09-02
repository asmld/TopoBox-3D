"""Degree-specific MeshGraphNets-style baseline used in the paper."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _mlp(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_features),
    )


class InteractionLayer(nn.Module):
    """MeshGraphNet/Interaction-Network message passing with sum aggregation."""

    def __init__(self, width: int, edge_width: int):
        super().__init__()
        self.edge_mlp = _mlp(2 * width + edge_width, width, width)
        self.node_mlp = _mlp(2 * width, width, width)
        self.edge_norm = nn.LayerNorm(width)
        self.node_norm = nn.LayerNorm(width)

    def forward(self, nodes: Tensor, edge_index: Tensor, edges: Tensor) -> tuple[Tensor, Tensor]:
        senders, receivers = edge_index
        messages = self.edge_mlp(torch.cat((nodes[senders], nodes[receivers], edges), dim=-1))
        edges = self.edge_norm(edges + messages)
        # LayerNorm may promote edge states to float32 under AMP while node
        # states remain bfloat16. Match the source dtype required by
        # ``index_add_``; the following MLP is autocast-aware.
        aggregate = edges.new_zeros(nodes.shape)
        aggregate.index_add_(0, receivers, edges)
        update = self.node_mlp(torch.cat((nodes, aggregate), dim=-1))
        nodes = self.node_norm(nodes + update)
        return nodes, edges


class MGNLite(nn.Module):
    """Topology-unaware MPNN baseline on a degree-specific simplex graph.

    For a k-form task, graph nodes are the active k-simplices and graph edges
    connect simplices sharing a local (k+1)-coface.  The model receives no
    cross-rank incidence operators, Betti numbers, or harmonic bases.
    """

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        out_channels: int,
        width: int = 128,
        layers: int = 6,
    ):
        super().__init__()
        self.node_encoder = _mlp(node_in, width, width)
        self.edge_encoder = _mlp(edge_in, width, width)
        self.processor = nn.ModuleList(InteractionLayer(width, width) for _ in range(layers))
        self.decoder = _mlp(width, width, out_channels)

    def forward(self, nodes: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """Predict one output per active simplex token."""
        nodes = self.node_encoder(nodes)
        edges = self.edge_encoder(edge_attr)
        for layer in self.processor:
            nodes, edges = layer(nodes, edge_index, edges)
        return self.decoder(nodes)
