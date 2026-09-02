"""Executable project implementation of the TNO baseline.

The accompanying paper is used as the architectural reference, while this
module is a project implementation because the authors' repository is not
publicly available. It must not be presented as the authors' official code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


def _apply_incidence(incidence: Tensor, values: Tensor, transpose: bool = False) -> Tensor:
    operator = incidence.transpose(0, 1) if transpose else incidence
    if operator.is_sparse:
        return torch.sparse.mm(operator, values)
    return operator @ values


def _mass_weighted_harmonic_projection(
    values: Tensor,
    basis: Tensor,
    mass: Tensor,
) -> Tensor:
    """Project latent k-cochains onto an M-orthonormal harmonic basis.

    ``basis`` is expected to satisfy ``basis.T @ diag(mass) @ basis = I``.
    The projection in the corresponding Hodge inner product is therefore
    ``basis @ (basis.T @ (diag(mass) @ values))``.
    """

    if basis.ndim != 2 or basis.shape[0] != values.shape[0]:
        raise ValueError(
            "harmonic basis must have shape (n_k, beta_k) matching the "
            f"cochain rows; got {basis.shape} and {values.shape}"
        )
    if mass.ndim == 2 and mass.shape[1] == 1:
        mass = mass[:, 0]
    if mass.ndim != 1 or mass.shape[0] != values.shape[0]:
        raise ValueError(
            "harmonic mass must have shape (n_k,) or (n_k, 1) matching the "
            f"cochain rows; got {mass.shape} and {values.shape}"
        )
    weighted_values = mass[:, None] * values
    return basis @ (basis.transpose(0, 1) @ weighted_values)


class TNOLayer(nn.Module):
    """Cross-rank DEC transport plus learned self and harmonic channels."""

    def __init__(self, max_rank: int, width: int, use_harmonic: bool = True):
        super().__init__()
        self.max_rank = max_rank
        self.use_harmonic = use_harmonic
        self.self_maps = nn.ModuleList(nn.Linear(width, width) for _ in range(max_rank + 1))
        self.down_maps = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(max_rank))
        self.up_maps = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(max_rank))
        self.harmonic_maps = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(max_rank + 1))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(max_rank + 1))
        self.activation = nn.GELU()

    def forward(
        self,
        cochains: Sequence[Tensor],
        incidence: Mapping[int, Tensor],
        harmonic_basis: Mapping[int, Tensor] | None = None,
        harmonic_mass: Mapping[int, Tensor] | None = None,
    ) -> list[Tensor]:
        outputs: list[Tensor] = []
        for rank, values in enumerate(cochains):
            update = self.self_maps[rank](values)
            if rank > 0:
                transported = _apply_incidence(incidence[rank], cochains[rank - 1], transpose=True)
                update = update + self.down_maps[rank - 1](transported)
            if rank < self.max_rank:
                transported = _apply_incidence(incidence[rank + 1], cochains[rank + 1])
                update = update + self.up_maps[rank](transported)
            if self.use_harmonic and harmonic_basis is not None and rank in harmonic_basis:
                if harmonic_mass is None or rank not in harmonic_mass:
                    raise ValueError(
                        f"Missing Hodge mass for harmonic basis at rank {rank}"
                    )
                basis = harmonic_basis[rank]
                harmonic = _mass_weighted_harmonic_projection(
                    values, basis, harmonic_mass[rank]
                )
                update = update + self.harmonic_maps[rank](harmonic)
            outputs.append(self.norms[rank](values + self.activation(update)))
        return outputs


class TNO(nn.Module):
    """Topology-aware neural operator over cochain ranks 0..max_rank."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        width: int = 64,
        layers: int = 4,
        use_harmonic: bool = True,
    ):
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must cover the same ranks")
        self.max_rank = len(in_channels) - 1
        self.encoders = nn.ModuleList(nn.Linear(size, width) for size in in_channels)
        self.layers = nn.ModuleList(
            TNOLayer(self.max_rank, width, use_harmonic=use_harmonic) for _ in range(layers)
        )
        self.decoders = nn.ModuleList(nn.Linear(width, size) for size in out_channels)

    def forward(
        self,
        cochains: Sequence[Tensor],
        incidence: Mapping[int, Tensor],
        harmonic_basis: Mapping[int, Tensor] | None = None,
        harmonic_mass: Mapping[int, Tensor] | None = None,
    ) -> list[Tensor]:
        hidden = [encoder(values) for encoder, values in zip(self.encoders, cochains)]
        for layer in self.layers:
            hidden = layer(hidden, incidence, harmonic_basis, harmonic_mass)
        return [decoder(values) for decoder, values in zip(self.decoders, hidden)]


# Backward-compatible name for existing checkpoints/scripts created before the
# experimental roster adopted the shorter model name "TNO".
TNOLite = TNO
