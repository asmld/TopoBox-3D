# Third-party code

The benchmark uses pinned upstream processors but does not vendor their source.
Run `scripts/fetch_upstream_models.sh` to create local checkouts under
`third_party/`.

| Local path | Upstream repository | Pinned commit | License observation |
|---|---|---|---|
| `third_party/Transolver` | `https://github.com/thuml/Transolver` | `75e0f67643806a81cd1d3f6adc88dd8c02416fe7` | MIT |
| `third_party/RIGNO` | `https://github.com/camlab-ethz/rigno` | `3e4b307c90f34237d0c1e5e497d4301116e9c3db` | MIT |
| `third_party/GNOT` | `https://github.com/thu-ml/GNOT` | `5ee2e6925a43f9a340a6016bad4da2c82a452cbe` | no top-level license found in the inspected checkout; verify terms before redistribution |
| `third_party/GAOT` | `https://github.com/camlab-ethz/GAOT` | `549c5a5f7113e23ba5e91469f2f8bbb1567fae46` | no top-level license found in the inspected checkout; verify terms before redistribution |

MGN and TNO are project implementations under `src/topobox3d/models/`; they are
not copied upstream repositories. Benchmark-specific adapters and matched
configurations are project code under `src/topobox3d/experiments/`.
