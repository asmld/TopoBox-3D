# Data

TopoBox-3D contains connected box-minus-void domains with independently varied
through-tunnels and enclosed cavities. Each accepted tetrahedral mesh is
checked by reconstructing the oriented chain complex, recomputing Betti
numbers and Euler characteristic, and verifying
`B1 @ B2 = 0` and `B2 @ B3 = 0`.

## Paper split

Each protocol contains 800 training, 120 validation, 200 Test-IID, and 200
Test-OOD geometries. Geometry ID is the atomic split unit.

| Protocol | In-support topology | Test-OOD topology | Shift |
|---|---|---|---|
| A | `(beta1, beta2) = (1, 1)`, family A | `(1, 1)`, family B | fixed-topology geometry |
| B | `beta1 in {0,1,2}, beta2 = 0` | `(3, 0)` | unseen tunnel support |
| C | `beta1 = 0, beta2 in {0,1,2}` | `(0, 3)` | unseen cavity support |
| D | `(beta1, beta2) in {0,1,2}^2` | `(3, 3)` | mixed topology |

The complete benchmark has 5,280 geometries. For each geometry, degrees
`k = 0, 1, 2` and four initial-condition configurations produce 63,360 PDE
instances. The fixed-time targets use the Hodge heat equation with `kappa = 1`,
`T = 0.1`, homogeneous absolute boundary conditions, and 100 Crank-Nicolson
steps.

## What is stored in Git

Only `data/TopoBox-3D/manifest.csv` is versioned here. Its 5,280 rows record the
formal geometry IDs, protocols, splits, topology, mesh statistics, and relative
paths. The repository excludes mesh files, packed HDF5 arrays, Hodge-heat
shards, checkpoints, and run logs.

After the Hugging Face release is available, place or link the downloaded data
at the paths expected by the command-line defaults:

```text
data/TopoBox-3D/packed/       geometry HDF5 shards and index.json
data/TopoBox-3D-HodgeHeat/    PDE HDF5 shards and index.json
```

The full numerical schema is documented in
[pde_data_schema.md](pde_data_schema.md). Public data versions should be
immutable and include generator commit, checksums, compression details, and
the same manifest committed here.

## Local generation pipeline

The code release retains the complete paper data pipeline:

```bash
python -m topobox3d.generate --formal --output data/TopoBox-3D
python -m topobox3d.validate data/TopoBox-3D
python -m topobox3d.pack data/TopoBox-3D
python -m topobox3d.generate_hodge_heat_dataset
python -m topobox3d.audit_hodge_heat_dataset \
  data/TopoBox-3D-HodgeHeat --require-complete --deep
```

Generation is substantially slower and larger than the smoke test. The public
Hugging Face release is the recommended route for reproducing model training.
