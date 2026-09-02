# TopoBox-3D Hodge-heat data schema

## Canonical stored data

Geometry and PDE data are stored separately and joined by `geometry_id`.

- Geometry root: `data/TopoBox-3D/packed`
- PDE root: `data/TopoBox-3D-HodgeHeat`
- PDE index: `index.json`
- Numerical manifest: `manifest.json`
- One HDF5 sample group: `samples/<geometry_id>/k{0,1,2}`

Each degree group contains:

| Dataset | Shape | Meaning |
|---|---|---|
| `w0` | `(4, n_k)` | Four initial canonical k-cochains |
| `wT` | `(4, n_k)` | Fixed-time canonical k-cochains at `T=0.1` |
| `mass` | `(n_k,)` | Positive diagonal Hodge mass |
| `harmonic_basis` | `(n_k, beta_k)` | M-orthonormal weighted harmonic basis |
| `low_positive_eigenvalues` | `(<=6,)` | Low nonzero generalized spectrum |
| `requested_energy_fractions` | `(4,3)` | Requested exact/coexact/harmonic energy |
| `realized_energy_fractions` | `(4,3)` | Realized energy after nullity handling |
| `relative_final_mass_norm` | `(4,)` | `||wT||_M / ||w0||_M` |
| `seeds` | `(4,)` | Reproducible initial-condition seeds |

The configuration order is:

1. `non_harmonic`
2. `weak_harmonic`
3. `balanced`
4. `strong_harmonic`

For `k=1,2`, their energy fractions are respectively
`(1/2,1/2,0)`, `(4/9,4/9,1/9)`, `(1/3,1/3,1/3)`, and
`(1/4,1/4,1/2)`.  If the relevant Betti number is zero, harmonic energy cannot
be realized and is redistributed to exact/coexact components.  For `k=0`,
all four fields are independent smooth mass-mean-zero, topology-insensitive
controls.

## Model adapters

`TopoBoxPDEDataset` returns one geometry/degree/configuration sample.
`sample.for_model(name)` adds model-native geometry and the PDE fields.

| Model | Physical input | Main keys |
|---|---|---|
| MGN-lite | Scalar k-simplex tokens | `nodes`, `edge_index`, `edge_attr` |
| RIGNO | Scalar k-simplex tokens | `u`, `c`, `x`, `x_batched` |
| Transolver | Scalar k-simplex tokens | `x`, `fx` |
| GNOT | Scalar k-simplex tokens | `query_graph_x`, `branch_inputs` |
| GAOT | Scalar k-simplex tokens | `pndata`, `xcoord`, `latent_tokens_coord` |
| TNO | Native cochain at active rank | `input_cochains`, `incidence`, `harmonic_basis`, `harmonic_mass` |

All six models predict native scalar canonical cochains.  Tokens are
vertices for `k=0`, oriented edges for `k=1`, and oriented faces for `k=2`.
Their normalized coordinates are respectively vertex coordinates, edge
midpoints, and face centers.  The five ordinary models receive:

- `k=0`: five geometry channels plus scalar `w0`, for six input channels;
- `k=1,2`: five averaged geometry channels, a three-component oriented
  edge/area vector, length/area, and scalar `w0`, for ten input channels.

Their output size is always one scalar per active simplex.  They never receive
Betti labels, cross-rank incidence operators, or harmonic bases as inputs.
MGN-lite only receives same-rank adjacency between k-simplices sharing one
local `(k+1)`-coface.

For RIGNO, the evolving scalar cochain is stored in `u`, while the 5/9
simplex-geometry channels are supplied through official `Inputs.c`.

TNO receives geometry features plus one physics channel at every rank.  The
active rank contains `w0`; inactive-rank physics channels are zero.  It also
receives native incidence operators, the active PDE harmonic basis, and the
active diagonal Hodge mass.  Its harmonic branch uses the weighted projection
`H_k (H_k^T M_k x)`; only the full harmonic TNO configuration is used.

## Uniform evaluation

No vertex reconstruction or lossy decode is used.  A model output with shape
`(n_k,1)` is already the predicted canonical cochain.  Use the same
mass-weighted squared relative loss for every model:

```python
from topobox3d.pde_dataset import mass_weighted_relative_mse

loss = mass_weighted_relative_mse(
    prediction, batch["target_cochain"], batch["mass"]
)
```

Take `sqrt(loss)` only for the reported relative Hodge error.  The active
weighted harmonic basis remains available as `target_harmonic_basis` for
evaluation; ordinary models do not receive it as an input.

Use `make_model_dataloader(...)` for the standard batch-size-one PyTorch
loader.  RIGNO's JAX training loop can iterate `TopoBoxPDEDataset` directly and
convert the returned NumPy/PyTorch arrays to JAX arrays when building its
regional graphs.
