"""Combinatorial topology and tetrahedron-quality checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass(frozen=True)
class TopologyResult:
    beta0: int
    beta1: int
    beta2: int
    beta3: int
    euler_characteristic: int
    boundary_components: int
    n_vertices: int
    n_edges: int
    n_faces: int
    n_tetrahedra: int
    nonmanifold_faces: int

    def to_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def tetra_quality(points: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    p = points[tetra]
    volume = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6.0
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_sq = sum(np.sum((p[:, i] - p[:, j]) ** 2, axis=1) for i, j in pairs)
    return 12.0 * np.power(3.0 * volume, 2.0 / 3.0) / np.maximum(edge_sq, 1e-30)


def compute_topology(points: np.ndarray, tetra: np.ndarray) -> tuple[TopologyResult, np.ndarray]:
    n_vertices = len(points)
    edge_local = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    face_local = np.array(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    edges = np.sort(tetra[:, edge_local].reshape(-1, 2), axis=1)
    unique_edges = np.unique(edges, axis=0)
    all_faces = np.sort(tetra[:, face_local].reshape(-1, 3), axis=1)
    unique_faces, face_counts = np.unique(all_faces, axis=0, return_counts=True)
    boundary_faces = unique_faces[face_counts == 1]
    nonmanifold = int(np.count_nonzero(face_counts > 2))

    volume_uf = UnionFind(n_vertices)
    for a, b in unique_edges:
        volume_uf.union(int(a), int(b))
    used = np.unique(tetra)
    beta0 = len({volume_uf.find(int(v)) for v in used})

    boundary_uf = UnionFind(n_vertices)
    for a, b, c in boundary_faces:
        boundary_uf.union(int(a), int(b))
        boundary_uf.union(int(a), int(c))
    boundary_nodes = np.unique(boundary_faces)
    boundary_components = len({boundary_uf.find(int(v)) for v in boundary_nodes})

    chi = int(n_vertices - len(unique_edges) + len(unique_faces) - len(tetra))
    beta3 = 0  # Compact 3-domain with non-empty boundary.
    beta2 = boundary_components - beta0
    beta1 = beta0 + beta2 - beta3 - chi
    result = TopologyResult(
        beta0=beta0,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        euler_characteristic=chi,
        boundary_components=boundary_components,
        n_vertices=n_vertices,
        n_edges=len(unique_edges),
        n_faces=len(unique_faces),
        n_tetrahedra=len(tetra),
        nonmanifold_faces=nonmanifold,
    )
    return result, boundary_faces
