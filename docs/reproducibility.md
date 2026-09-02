# Reproducibility

Run commands from the repository root on Linux. Python 3.10 is the reference
version. The PyTorch and RIGNO/JAX stacks are intentionally isolated because
their pinned GPU dependencies conflict.

## 1. Install and fetch model code

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/torch-cu124.txt
bash scripts/fetch_upstream_models.sh torch
python -m pytest -q
```

Create a second environment from `requirements/rigno-jax-cuda12.txt` and run
`bash scripts/fetch_upstream_models.sh rigno` for RIGNO.

## 2. Obtain or generate data

The recommended path is the versioned Hugging Face release described in
[data.md](data.md). The complete local generation path is:

1. `python -m topobox3d.generate --formal --output data/TopoBox-3D`
2. `python -m topobox3d.validate data/TopoBox-3D`
3. `python -m topobox3d.pack data/TopoBox-3D`
4. `python -m topobox3d.generate_hodge_heat_dataset`
5. `python -m topobox3d.audit_hodge_heat_dataset data/TopoBox-3D-HodgeHeat --require-complete --deep`

The formal audit should report 5,280 geometries and 63,360 PDE instances.

## 3. Train the 216 runs

The locked matrix is six models by four protocols by three cochain degrees by
three seeds. Every run uses 300 epochs, batch size 1, AdamW or the corresponding
Optax schedule, initial learning rate `2e-4`, weight decay `1e-4`, cosine decay,
gradient clipping at 1, and best in-support validation checkpoint selection.

```bash
bash scripts/train_torch_matrix.sh
bash scripts/train_rigno_matrix.sh  # in the JAX environment
```

Use `TOPOBOX_EPOCHS` only for PyTorch smoke runs; formal runs must retain 300
epochs. Outputs are written below `runs/topobox3d/` and are ignored by Git.

## 4. Evaluation and analysis entry points

| Module under `topobox3d.experiments` | Paper role |
|---|---|
| `aggregate_results` | run completeness and clustered degradation ratios |
| `comprehensive_results_analysis` | full task/model tables and rankings |
| `analyze_error_sources` | geometry, topology, and spectral driver analysis |
| `analyze_relative_jensen` | Rayleigh-controlled Jensen-gap analysis |
| `analyze_hodge_subspace_generalization` | component errors and leakage |
| `build_topology_pressure_framework` | task pressure and model capability audits |
| `evaluate_pure_harmonic_torch` | controlled PyTorch subspace probes |
| `evaluate_pure_harmonic_rigno` | controlled RIGNO subspace probes |
| `visualize_all_model_fits` | deterministic representative predictions |

Compact outputs used by the paper are committed under `results/`. Full
per-sample analysis tables are regenerated from the external data and
checkpoints and are not committed.

## 5. Paper-level checks

- 216 complete 300-epoch runs;
- 5,280 geometries and 63,360 PDE instances;
- matched topology penalty `Q > 1` in 37/45 cells for the five models without
  full chain-complex and harmonic-basis access;
- `Q > 1` in 17/20 corresponding cells where harmonic dimension changes;
- initial Rayleigh quotient is the top single-variable predictor in 43/72
  model-task cells;
- controlled subspace probes contain 286 geometry-level seed-0 records, while
  conventional mixed-input comparisons retain the three-seed evaluation.
