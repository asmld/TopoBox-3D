"""Mass-weighted scalar, harmonic, and non-harmonic evaluation metrics."""

from __future__ import annotations

import torch


def _vector(values: torch.Tensor) -> torch.Tensor:
    while values.ndim > 1 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"expected one scalar per simplex, got {values.shape}")
    return values


def mass_norm(values: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    values = _vector(values)
    mass = _vector(mass)
    return torch.sqrt(torch.sum(mass * values.square()).clamp_min(0.0))


def sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mass: torch.Tensor,
    harmonic_basis: torch.Tensor,
    epsilon: float = 1e-8,
) -> dict[str, object]:
    """Compute the paper's mass-weighted metrics for one sample."""

    prediction = _vector(prediction).float()
    target = _vector(target).to(prediction).float()
    mass = _vector(mass).to(prediction).float()
    basis = harmonic_basis.to(prediction).float()
    error = prediction - target
    relative_l2 = mass_norm(error, mass) / (mass_norm(target, mass) + epsilon)

    if basis.shape[1] == 0:
        harmonic_error = torch.zeros_like(error)
        target_harmonic = torch.zeros_like(target)
        coefficient_error = error.new_empty((0,))
    else:
        coefficient_error = basis.transpose(0, 1) @ (mass * error)
        target_coefficients = basis.transpose(0, 1) @ (mass * target)
        harmonic_error = basis @ coefficient_error
        target_harmonic = basis @ target_coefficients
    nonharmonic_error = error - harmonic_error
    target_nonharmonic = target - target_harmonic

    harmonic_relative = mass_norm(harmonic_error, mass) / (
        mass_norm(target_harmonic, mass) + epsilon
    )
    nonharmonic_relative = mass_norm(nonharmonic_error, mass) / (
        mass_norm(target_nonharmonic, mass) + epsilon
    )
    return {
        "relative_l2": float(relative_l2.detach().cpu()),
        "relative_mse": float(relative_l2.square().detach().cpu()),
        "harmonic_relative": float(harmonic_relative.detach().cpu()),
        "nonharmonic_relative": float(nonharmonic_relative.detach().cpu()),
        "harmonic_coefficient_error": coefficient_error.detach().cpu().tolist(),
    }
