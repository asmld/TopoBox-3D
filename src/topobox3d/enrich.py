"""Add analytic SDF, model geometry inputs, and TNO complex to existing samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import meshio

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from topobox3d.complex import build_complex
    from topobox3d.geometry import Cavity, GeometrySpec, Tunnel
    from topobox3d.sdf import (
        NODE_GEOMETRY_FEATURE_NAMES, REGULAR_GRID_FEATURE_NAMES, analytic_sdf,
        node_geometry_features, regular_grid_fields, regular_grid_geometry_features,
    )
else:
    from .complex import build_complex
    from .geometry import Cavity, GeometrySpec, Tunnel
    from .sdf import (
        NODE_GEOMETRY_FEATURE_NAMES, REGULAR_GRID_FEATURE_NAMES, analytic_sdf,
        node_geometry_features, regular_grid_fields, regular_grid_geometry_features,
    )


def _spec(metadata: dict) -> GeometrySpec:
    geometry = metadata["geometry"]
    tunnels = tuple(Tunnel(tuple(x["start"]), tuple(x["end"]), x["radius"]) for x in geometry["tunnels"])
    cavities = tuple(Cavity(tuple(x["center"]), tuple(x["axes"]), tuple(x["rotation_axis"]), x["rotation_angle"]) for x in geometry["cavities"])
    return GeometrySpec(tunnels, cavities, geometry["family"], geometry["measured_min_clearance"])


def _set_canonical_node_geometry(payload: dict[str, np.ndarray]) -> None:
    payload["geometry_features"] = node_geometry_features(
        payload["normalized_xyz"], payload["is_boundary"], payload["analytic_domain_sdf"]
    )
    payload["geometry_feature_names"] = np.asarray(NODE_GEOMETRY_FEATURE_NAMES)


def enrich_sample(npz_path: Path, resolution: tuple[int, int, int]) -> None:
    metadata_path = npz_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    points, tetra = payload["points"], payload["tetra"]
    spec, box = _spec(metadata), tuple(metadata["box"])
    analytic = analytic_sdf(points, spec, box)
    regular = regular_grid_fields(spec, box, resolution)
    complex_fields = build_complex(points, tetra, metadata["beta1"], metadata["beta2"])

    payload.update(analytic)
    payload.update(regular)
    payload.update(complex_fields)
    _set_canonical_node_geometry(payload)

    temporary = npz_path.with_name("mesh.enriched.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(npz_path)

    metadata["boundary_encoding"].update({
        "geometry_feature_names": payload["geometry_feature_names"].tolist(),
        "analytic_sdf_sign": "positive in the material domain; negative inside a void or outside the box",
        "ellipsoid_sdf_note": "The ellipsoid field has exact sign and zero set; magnitude uses radial surface distance.",
        "regular_grid_resolution": list(resolution),
    })
    metadata["tno_complex"] = {
        "orientation": "canonical ascending vertices for edges/faces; tetrahedra oriented to positive volume",
        "incidence": "COO arrays incidence_{1,2,3}_{row,col,value,shape}",
        "chain_identity": "incidence_1 @ incidence_2 = 0 and incidence_2 @ incidence_3 = 0",
        "harmonic_basis": "Euclidean combinatorial Hodge bases for geometry-only setup; weighted FEEC/DEC bases are deferred to PDE generation.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    sync_vtu_fields(npz_path)


def refresh_sdf_fields(npz_path: Path, resolution: tuple[int, int, int]) -> None:
    """Refresh node/grid SDF fields without recomputing the mesh complex."""
    metadata_path = npz_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    analytic = analytic_sdf(payload["points"], _spec(metadata), tuple(metadata["box"]))
    regular = regular_grid_fields(_spec(metadata), tuple(metadata["box"]), resolution)
    payload.update(analytic)
    payload.update(regular)
    _set_canonical_node_geometry(payload)
    temporary = npz_path.with_name("mesh.sdf.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(npz_path)
    metadata["boundary_encoding"]["geometry_feature_names"] = list(NODE_GEOMETRY_FEATURE_NAMES)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    sync_vtu_fields(npz_path)


def refresh_node_geometry_input(npz_path: Path) -> None:
    """Replace only the ready-to-use node input; preserve every rich field."""
    metadata_path = npz_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    expected_names = list(NODE_GEOMETRY_FEATURE_NAMES)
    if (
        payload.get("geometry_features", np.empty((0, 0))).shape == (len(payload["points"]), 5)
        and payload.get("geometry_feature_names", np.asarray([])).tolist() == expected_names
        and metadata.get("boundary_encoding", {}).get("geometry_feature_names") == expected_names
    ):
        return
    _set_canonical_node_geometry(payload)
    temporary = npz_path.with_name("mesh.geometry-input.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(npz_path)
    metadata["boundary_encoding"]["geometry_feature_names"] = list(NODE_GEOMETRY_FEATURE_NAMES)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def refresh_regular_grid_input(npz_path: Path) -> None:
    """Compact the regular-grid input without recomputing any analytic SDF."""
    metadata_path = npz_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    expected_names = list(REGULAR_GRID_FEATURE_NAMES)
    old_names = payload["regular_grid_feature_names"].tolist()
    if old_names == expected_names and payload["regular_grid_features"].shape[-1] == 5:
        return
    domain_name = "analytic_domain_sdf" if "analytic_domain_sdf" in old_names else "domain_sdf"
    domain = payload["regular_grid_features"][..., old_names.index(domain_name)]
    resolution = tuple(int(value) for value in payload["regular_grid_resolution"])
    normalized = payload["regular_grid_normalized_xyz"]
    features = regular_grid_geometry_features(
        normalized.reshape(-1, 3), domain.reshape(-1), tuple(metadata["box"]), resolution
    )
    payload["regular_grid_features"] = features.reshape(*resolution, 5)
    payload["regular_grid_feature_names"] = np.asarray(REGULAR_GRID_FEATURE_NAMES)
    temporary = npz_path.with_name("mesh.regular-grid-input.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(npz_path)
    metadata["boundary_encoding"]["regular_grid_feature_names"] = expected_names
    metadata["boundary_encoding"]["regular_grid_boundary_band"] = (
        "abs(analytic_domain_sdf) <= 0.5 * physical voxel diagonal"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def sync_vtu_fields(npz_path: Path) -> None:
    """Copy node-level analytic geometry fields into the tetrahedral VTU."""
    vtu_path = npz_path.with_name("mesh.vtu")
    if not vtu_path.exists():
        return
    with np.load(npz_path, allow_pickle=False) as data:
        fields = {
            name: data[name]
            for name in (
                "analytic_domain_sdf", "analytic_outer_box_sdf", "analytic_internal_void_sdf",
                "analytic_tunnel_sdf", "analytic_cavity_sdf", "has_tunnel", "has_cavity",
                "is_in_domain_analytic",
            )
        }
    mesh = meshio.read(vtu_path)
    if len(mesh.points) != len(next(iter(fields.values()))):
        raise RuntimeError(f"VTU/NPZ point mismatch in {npz_path.parent}")
    mesh.point_data.update(fields)
    mesh.write(vtu_path, binary=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/TopoBox-3D-mini"))
    parser.add_argument("--resolution", nargs=3, type=int, default=(32, 16, 16))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1, help="Number of disjoint path partitions.")
    parser.add_argument("--worker-index", type=int, default=0, help="Zero-based partition handled by this process.")
    parser.add_argument("--vtu-only", action="store_true", help="Only synchronize analytic node fields to VTU.")
    parser.add_argument("--sdf-only", action="store_true", help="Refresh analytic node/grid SDF fields only.")
    parser.add_argument(
        "--geometry-input-only", action="store_true",
        help="Rebuild only the canonical five-channel node geometry input.",
    )
    parser.add_argument(
        "--regular-grid-input-only", action="store_true",
        help="Compact only the regular-grid input to five channels.",
    )
    args = parser.parse_args()
    paths = sorted(args.root.rglob("mesh.npz"))
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("require workers >= 1 and 0 <= worker-index < workers")
    paths = paths[args.worker_index::args.workers]
    if args.limit is not None:
        paths = paths[:args.limit]
    for index, path in enumerate(paths, 1):
        action = (
            "SYNC-VTU" if args.vtu_only else
            "REGULAR-GRID-INPUT" if args.regular_grid_input_only else
            "GEOMETRY-INPUT" if args.geometry_input_only else
            "REFRESH-SDF" if args.sdf_only else "ENRICH"
        )
        print(f"{action} {index}/{len(paths)} {path.parent.name}", flush=True)
        if args.vtu_only:
            sync_vtu_fields(path)
        elif args.regular_grid_input_only:
            refresh_regular_grid_input(path)
        elif args.geometry_input_only:
            refresh_node_geometry_input(path)
        elif args.sdf_only:
            refresh_sdf_fields(path, tuple(args.resolution))
        else:
            enrich_sample(path, tuple(args.resolution))


if __name__ == "__main__":
    main()
