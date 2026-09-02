# Models

The paper compares six approximately capacity-matched architectures. The
comparison is not a single-factor ablation: processors, tokenization, adapters,
and native structural access differ.

| Model | Implementation used here | Effective structural access | Parameters |
|---|---|---|---:|
| MGN | project degree-specific MeshGraphNets-style adaptation | same-degree simplex adjacency | 1.162M |
| RIGNO | pinned official JAX implementation | coordinate-reconstructed regional graph | 1.197-1.198M |
| Transolver | pinned official implementation | coordinate tokens and local geometry | 1.210-1.212M |
| GNOT | pinned official implementation | coordinate tokens and local geometry | 1.135-1.136M |
| GAOT | pinned official processor with the paper's cochain readout adapter | coordinate latent graph | 1.186M |
| TNO | project implementation of the published mechanism | `B1`, `B2`, `B3`, and active harmonic basis | 1.168M |

Exact settings and degree-wise parameter counts are stored in
`configs/parameter_counts.json` and constructed by
`src/topobox3d/experiments/model_registry.py`.

## Project implementations

`src/topobox3d/models/mgn.py` implements the paper's MGN baseline. Its graph
nodes are the active degree's vertices, edges, or faces, and its graph edges
encode same-degree adjacency through shared cofaces. It does not receive
cross-degree incidence maps, Betti numbers, or harmonic bases.

`src/topobox3d/models/tno.py` implements the published cross-degree incidence
and harmonic mechanism used in the paper because no official code was
available for the experiment. It is project code, not author code.

## Upstream implementations

RIGNO, Transolver, GNOT, and GAOT are fetched at fixed commits into
`third_party/` by `scripts/fetch_upstream_models.sh`. Those checkouts are
ignored by Git and are not redistributed. All six models use the common paper
splits, mass-weighted loss, evaluation records, and reporting functions in this
repository rather than upstream benchmark metrics.

The model-input assembly is in `src/topobox3d/model_inputs.py`; PyTorch forward
adapters are in `src/topobox3d/experiments/model_registry.py`; the isolated JAX
path is in `src/topobox3d/experiments/train_rigno.py`.
