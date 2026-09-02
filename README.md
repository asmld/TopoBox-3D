# TopoBox-3D

Official code repository for **Beyond Arbitrary Geometry: Topology
Generalization in Neural PDE Operators**.

TopoBox-3D is a controlled 3D benchmark that separates fixed-topology geometry
shift from generalization to unseen homological support. It contains four
protocols over box-minus-void domains and fixed-time Hodge heat tasks on vertex,
edge, and face cochains. The paper compares six approximately
capacity-matched neural operators under the same splits, training budget, and
mass-weighted evaluation.

The paper and Hugging Face dataset links will be added after the public
releases are available. This GitHub repository intentionally contains no bulk
meshes, PDE shards, checkpoints, or run logs.

## Repository layout

```text
configs/        locked parameter counts and model settings
data/           the lightweight 5,280-geometry split manifest only
docs/           data, model, schema, and reproduction notes
requirements/   pinned PyTorch/CUDA and RIGNO/JAX environments
results/        compact tables and diagnostics reported in the paper
scripts/        upstream checkout and 216-run launchers
src/topobox3d/  generation, Hodge heat, models, training, and analysis code
tests/          lightweight chain-complex tests
third_party/    local upstream checkouts; ignored by Git
```

## Quick start

Python 3.10 on Linux is the reference environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,analysis,test]"
python -m pytest -q
```

Generate and validate a small local smoke set:

```bash
python -m topobox3d.generate --limit 1 --output data/TopoBox-3D-smoke
python -m topobox3d.validate data/TopoBox-3D-smoke
```

Formal training uses separate PyTorch and JAX environments. Fetch the pinned
upstream implementations before launching the full matrices:

```bash
python -m pip install -r requirements/torch-cu124.txt
bash scripts/fetch_upstream_models.sh torch
bash scripts/train_torch_matrix.sh

# Run in the separate RIGNO/JAX environment.
python -m pip install -r requirements/rigno-jax-cuda12.txt
bash scripts/fetch_upstream_models.sh rigno
bash scripts/train_rigno_matrix.sh
```

See [reproducibility](docs/reproducibility.md), [data](docs/data.md),
[models](docs/models.md), and the [PDE data schema](docs/pde_data_schema.md)
for the complete workflow and implementation boundaries.

## Release policy

The complete dataset will be distributed separately on Hugging Face. Trained
checkpoints should likewise remain outside this repository. The committed
manifest and result summaries are small, inspectable records of the exact
paper split and reported analyses.

The project license and citation metadata should be added when the paper and
data releases become public. Third-party implementations retain their own
licenses; see [third-party code](docs/third_party.md).
