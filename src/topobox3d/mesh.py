"""Gmsh CAD construction, meshing, features, and output writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gmsh
import meshio
import numpy as np
from scipy.spatial import cKDTree

from .config import DatasetConfig
from .complex import build_complex
from .geometry import GeometrySpec
from .sdf import (
    NODE_GEOMETRY_FEATURE_NAMES,
    analytic_sdf,
    node_geometry_features,
    regular_grid_fields,
)
from .topology import compute_topology, tetra_quality


BOUNDARY_BITS = {"outer": 1, "tunnel": 2, "cavity": 4}


def _entity_kind(dimtag: tuple[int, int], box: tuple[float, float, float], tol: float = 1e-5) -> str:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(*dimtag)
    bounds = ((xmin, xmax, box[0]), (ymin, ymax, box[1]), (zmin, zmax, box[2]))
    for lo, hi, length in bounds:
        if abs(hi - lo) < tol and (abs(lo) < tol or abs(lo - length) < tol):
            return "outer"
    if any((lo <= tol and hi >= length - tol) for lo, hi, length in bounds):
        return "tunnel"
    return "cavity"


def _extract_elements(dim: int, entity: int | None, wanted_type: int) -> tuple[np.ndarray, np.ndarray]:
    types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, entity if entity is not None else -1)
    for kind, tags, nodes in zip(types, element_tags, node_tags):
        if kind == wanted_type:
            nper = 4 if wanted_type == 4 else 3
            return np.asarray(tags, dtype=np.int64), np.asarray(nodes, dtype=np.int64).reshape(-1, nper)
    return np.empty(0, dtype=np.int64), np.empty((0, 4 if wanted_type == 4 else 3), dtype=np.int64)


def _distances(points: np.ndarray, boundary_triangles: np.ndarray, triangle_bits: np.ndarray) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    all_nodes = np.unique(boundary_triangles)
    outputs["distance_to_boundary"] = cKDTree(points[all_nodes]).query(points, workers=-1)[0].astype(np.float32)
    groups = {
        "outer": (triangle_bits & BOUNDARY_BITS["outer"]) != 0,
        "tunnel": (triangle_bits & BOUNDARY_BITS["tunnel"]) != 0,
        "cavity": (triangle_bits & BOUNDARY_BITS["cavity"]) != 0,
        "internal": (triangle_bits & (BOUNDARY_BITS["tunnel"] | BOUNDARY_BITS["cavity"])) != 0,
    }
    for name, selected in groups.items():
        if np.any(selected):
            nodes = np.unique(boundary_triangles[selected])
            values = cKDTree(points[nodes]).query(points, workers=-1)[0]
        else:
            values = np.full(len(points), 2.5)
        outputs[f"distance_to_{name}_boundary"] = values.astype(np.float32)
    return outputs


def generate_mesh(
    spec: GeometrySpec,
    config: DatasetConfig,
    sample_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add(metadata["geometry_id"])
        occ = gmsh.model.occ
        lx, ly, lz = config.geometry.box
        domain = (3, occ.addBox(0, 0, 0, lx, ly, lz))
        tools: list[tuple[int, int]] = []
        for tunnel in spec.tunnels:
            start, end = np.asarray(tunnel.start), np.asarray(tunnel.end)
            direction = end - start
            # Extend beyond both intended boundary faces before subtraction. This removes the
            # cylinder end caps from the physical domain, which is essential for
            # inclined through-tunnels and avoids tiny cap/box intersection slivers.
            axis_index = {"x": 0, "y": 1, "z": 2}[config.geometry.tunnel_axis]
            extension_fraction = 0.15 / config.geometry.box[axis_index]
            extended_start = start - extension_fraction * direction
            extended_direction = direction * (1.0 + 2.0 * extension_fraction)
            tools.append((3, occ.addCylinder(*extended_start, *extended_direction, tunnel.radius)))
        for cavity in spec.cavities:
            cx, cy, cz = cavity.center
            sphere = occ.addSphere(cx, cy, cz, 1.0)
            occ.dilate([(3, sphere)], cx, cy, cz, *cavity.axes)
            if cavity.rotation_angle:
                occ.rotate([(3, sphere)], cx, cy, cz, *cavity.rotation_axis, cavity.rotation_angle)
            tools.append((3, sphere))
        if tools:
            cut, _ = occ.cut([domain], tools, removeObject=True, removeTool=True)
            volumes = [tag for dim, tag in cut if dim == 3]
        else:
            volumes = [domain[1]]
        occ.synchronize()
        if len(volumes) != 1:
            raise RuntimeError(f"Boolean operation produced {len(volumes)} volume components")

        surfaces = gmsh.model.getBoundary([(3, volumes[0])], oriented=False, recursive=False)
        surface_groups: dict[str, list[int]] = {"outer": [], "tunnel": [], "cavity": []}
        for dimtag in surfaces:
            surface_groups[_entity_kind(dimtag, config.geometry.box)].append(dimtag[1])
        actual_surface_counts = {name: len(tags) for name, tags in surface_groups.items()}
        classification_ok = (
            actual_surface_counts["outer"] == 6
            and actual_surface_counts["tunnel"] >= len(spec.tunnels)
            and actual_surface_counts["cavity"] >= len(spec.cavities)
            and (len(spec.tunnels) > 0) == (actual_surface_counts["tunnel"] > 0)
            and (len(spec.cavities) > 0) == (actual_surface_counts["cavity"] > 0)
        )
        if not classification_ok:
            raise RuntimeError(
                "CAD boundary classification failed: expected six outer faces and boundary entities "
                f"consistent with ({len(spec.tunnels)} tunnels, {len(spec.cavities)} cavities), "
                f"got {actual_surface_counts}"
            )
        volume_group = gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, volume_group, "domain")
        physical_ids = {"outer": 11, "tunnel": 12, "cavity": 13}
        for name, tags in surface_groups.items():
            if tags:
                group = gmsh.model.addPhysicalGroup(2, tags, physical_ids[name])
                gmsh.model.setPhysicalName(2, group, f"{name}_boundary")

        mesh_cfg = config.mesh
        gmsh.option.setNumber("Mesh.Algorithm3D", mesh_cfg.algorithm_3d)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_cfg.feature_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_cfg.bulk_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 18)
        gmsh.option.setNumber("Mesh.Optimize", int(mesh_cfg.optimize))
        internal_surfaces = surface_groups["tunnel"] + surface_groups["cavity"]
        if internal_surfaces:
            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance_field, "SurfacesList", internal_surfaces)
            gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 80)
            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", distance_field)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMin", mesh_cfg.feature_size)
            gmsh.model.mesh.field.setNumber(threshold, "SizeMax", mesh_cfg.bulk_size)
            gmsh.model.mesh.field.setNumber(threshold, "DistMin", mesh_cfg.feature_size)
            gmsh.model.mesh.field.setNumber(threshold, "DistMax", mesh_cfg.refinement_distance)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.model.mesh.generate(3)
        if mesh_cfg.optimize:
            gmsh.model.mesh.optimize("Netgen")

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        order = np.argsort(node_tags)
        node_tags, points = node_tags[order], points[order]
        max_tag = int(node_tags.max())
        tag_to_index = np.full(max_tag + 1, -1, dtype=np.int64)
        tag_to_index[node_tags] = np.arange(len(node_tags), dtype=np.int64)

        _, tet_tags = _extract_elements(3, volumes[0], 4)
        tetra = tag_to_index[tet_tags]
        if len(tetra) == 0 or np.any(tetra < 0):
            raise RuntimeError("No valid linear tetrahedra were extracted")

        boundary_blocks: list[np.ndarray] = []
        boundary_bits: list[np.ndarray] = []
        for name, tags in surface_groups.items():
            for surface_tag in tags:
                _, triangles_tag = _extract_elements(2, surface_tag, 2)
                if len(triangles_tag):
                    triangles = tag_to_index[triangles_tag]
                    boundary_blocks.append(triangles)
                    boundary_bits.append(np.full(len(triangles), BOUNDARY_BITS[name], dtype=np.uint8))
        boundary_triangles = np.concatenate(boundary_blocks)
        triangle_bits = np.concatenate(boundary_bits)

        node_boundary_mask = np.zeros(len(points), dtype=np.uint8)
        for triangles, bits in zip(boundary_blocks, boundary_bits):
            for bit in np.unique(bits):
                node_boundary_mask[np.unique(triangles[bits == bit])] |= bit
        is_boundary = (node_boundary_mask > 0).astype(np.uint8)
        normalized_xyz = (points / np.asarray(config.geometry.box)).astype(np.float32)
        distances = _distances(points, boundary_triangles, triangle_bits)
        analytic_fields = analytic_sdf(points, spec, config.geometry.box)
        regular_fields = regular_grid_fields(
            spec, config.geometry.box, config.inputs.regular_grid_resolution
        )
        quality = tetra_quality(points, tetra).astype(np.float32)
        topology, computed_boundary = compute_topology(points, tetra)
        expected = (1, metadata["beta1"], metadata["beta2"], 0)
        actual = (topology.beta0, topology.beta1, topology.beta2, topology.beta3)
        if actual != expected or topology.nonmanifold_faces:
            raise RuntimeError(f"Topology check failed: expected {expected}, got {actual}; {topology.to_dict()}")
        if len(computed_boundary) != len(boundary_triangles):
            raise RuntimeError(
                f"Boundary extraction mismatch: tetra complex has {len(computed_boundary)} faces, "
                f"CAD surfaces have {len(boundary_triangles)}"
            )

        mesh_path = sample_dir / "mesh.msh"
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.write(str(mesh_path))

        point_data = {
            "is_boundary": is_boundary,
            "boundary_mask": node_boundary_mask,
            "is_outer_boundary": ((node_boundary_mask & 1) != 0).astype(np.uint8),
            "is_tunnel_boundary": ((node_boundary_mask & 2) != 0).astype(np.uint8),
            "is_cavity_boundary": ((node_boundary_mask & 4) != 0).astype(np.uint8),
            **distances,
            **analytic_fields,
        }
        vtu = meshio.Mesh(
            points=points,
            cells=[("tetra", tetra)],
            point_data=point_data,
            cell_data={"mean_ratio_quality": [quality]},
        )
        vtu.write(sample_dir / "mesh.vtu", binary=True)

        geometry_feature_names = list(NODE_GEOMETRY_FEATURE_NAMES)
        geometry_features = node_geometry_features(
            normalized_xyz, point_data["is_boundary"], analytic_fields["analytic_domain_sdf"]
        )
        complex_fields = build_complex(points, tetra, metadata["beta1"], metadata["beta2"])
        np.savez_compressed(
            sample_dir / "mesh.npz",
            points=points.astype(np.float32),
            normalized_xyz=normalized_xyz,
            tetra=tetra.astype(np.int32),
            boundary_triangles=boundary_triangles.astype(np.int32),
            boundary_triangle_mask=triangle_bits,
            is_boundary=is_boundary,
            boundary_mask=node_boundary_mask,
            is_outer_boundary=point_data["is_outer_boundary"],
            is_tunnel_boundary=point_data["is_tunnel_boundary"],
            is_cavity_boundary=point_data["is_cavity_boundary"],
            **distances,
            **analytic_fields,
            **regular_fields,
            **complex_fields,
            tetra_quality=quality,
            geometry_features=geometry_features,
            geometry_feature_names=np.asarray(geometry_feature_names),
        )

        result = metadata | {
            "geometry": spec.to_dict(),
            "box": list(config.geometry.box),
            "topology": topology.to_dict(),
            "mesh_quality": {
                "minimum": float(quality.min()),
                "p01": float(np.quantile(quality, 0.01)),
                "mean": float(quality.mean()),
                "maximum": float(quality.max()),
            },
            "boundary_encoding": {
                "boundary_mask": "bit flags: outer=1, tunnel=2, cavity=4; 0 means interior",
                "distance_sentinel": 2.5,
                "distance_note": "Distances are nearest-node approximations in box-length units; 2.5 means that boundary class is absent.",
                "geometry_feature_names": geometry_feature_names,
                "analytic_sdf_sign": "positive in the material domain; negative inside a void or outside the box",
                "ellipsoid_sdf_note": "The ellipsoid field has exact sign and zero set; magnitude uses radial surface distance.",
                "regular_grid_resolution": list(config.inputs.regular_grid_resolution),
                "regular_grid_feature_names": regular_fields["regular_grid_feature_names"].tolist(),
                "regular_grid_boundary_band": "abs(analytic_domain_sdf) <= 0.5 * physical voxel diagonal",
            },
            "tno_complex": {
                "orientation": "canonical ascending vertices for edges/faces; tetrahedra oriented to positive volume",
                "incidence": "COO arrays incidence_{1,2,3}_{row,col,value,shape}",
                "chain_identity": "incidence_1 @ incidence_2 = 0 and incidence_2 @ incidence_3 = 0",
                "harmonic_basis": "Euclidean combinatorial Hodge bases for geometry-only model setup; weighted FEEC/DEC bases are deferred to PDE generation.",
            },
            "files": {"msh": "mesh.msh", "vtu": "mesh.vtu", "npz": "mesh.npz"},
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    finally:
        gmsh.finalize()
