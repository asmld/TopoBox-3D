"""Oriented tetrahedral complex and geometry-only TNO inputs."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def _permutation_sign(values: np.ndarray) -> int:
    inversions = sum(values[i] > values[j] for i in range(len(values)) for j in range(i + 1, len(values)))
    return -1 if inversions % 2 else 1


def _coo_payload(prefix: str, matrix: sparse.spmatrix) -> dict[str, np.ndarray]:
    coo = matrix.tocoo()
    return {
        f"{prefix}_row": coo.row.astype(np.int32),
        f"{prefix}_col": coo.col.astype(np.int32),
        f"{prefix}_value": coo.data.astype(np.int8),
        f"{prefix}_shape": np.asarray(coo.shape, dtype=np.int32),
    }


def _harmonic_basis(laplacian: sparse.spmatrix, nullity: int) -> tuple[np.ndarray, np.ndarray]:
    n = laplacian.shape[0]
    if nullity == 0:
        return np.empty((n, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    if nullity >= n:
        return np.eye(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    count = min(n - 1, nullity + 2)
    # A small negative shift makes the singular zero eigenspace the closest
    # target while keeping the factorized matrix nonsingular. This is more
    # reliable than ``which='SM'`` when several tiny nonzero modes cluster near
    # a multi-dimensional harmonic space.
    values, vectors = eigsh(
        laplacian.astype(np.float64), k=count, sigma=-1e-6, which="LM",
        tol=1e-10, maxiter=20000,
    )
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    basis = vectors[:, :nullity]
    # Re-orthogonalize to make the saved projection stable in float32.
    basis, _ = np.linalg.qr(basis)
    residual = np.linalg.norm(laplacian @ basis, axis=0)
    if float(np.max(residual, initial=0.0)) > 1e-6:
        raise RuntimeError(f"Harmonic eigensolve residual is too large: {residual}")
    return basis.astype(np.float32), residual.astype(np.float32)


def build_complex(points: np.ndarray, tetra: np.ndarray, beta1: int, beta2: int) -> dict[str, np.ndarray]:
    """Construct canonical simplices, incidence matrices and harmonic bases."""
    points = np.asarray(points, dtype=np.float64)
    oriented_tetra = np.asarray(tetra, dtype=np.int64).copy()
    p = points[oriented_tetra]
    determinants = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    flip = determinants < 0
    oriented_tetra[flip, :2] = oriented_tetra[flip, 1::-1]

    edge_local = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    face_local = np.array(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    edges = np.unique(np.sort(oriented_tetra[:, edge_local].reshape(-1, 2), axis=1), axis=0)
    faces = np.unique(np.sort(oriented_tetra[:, face_local].reshape(-1, 3), axis=1), axis=0)
    edge_lookup = {tuple(edge): i for i, edge in enumerate(edges)}
    face_lookup = {tuple(face): i for i, face in enumerate(faces)}

    n_vertices, n_edges, n_faces, n_tetra = len(points), len(edges), len(faces), len(oriented_tetra)
    b1_rows = np.ravel(edges)
    b1_cols = np.repeat(np.arange(n_edges), 2)
    b1_data = np.tile((-1, 1), n_edges)
    b1 = sparse.coo_matrix((b1_data, (b1_rows, b1_cols)), shape=(n_vertices, n_edges)).tocsr()

    b2_rows, b2_cols, b2_data = [], [], []
    for face_index, (a, b, c) in enumerate(faces):
        for edge, sign in (((b, c), 1), ((a, c), -1), ((a, b), 1)):
            b2_rows.append(edge_lookup[tuple(edge)])
            b2_cols.append(face_index)
            b2_data.append(sign)
    b2 = sparse.coo_matrix((b2_data, (b2_rows, b2_cols)), shape=(n_edges, n_faces)).tocsr()

    b3_rows, b3_cols, b3_data = [], [], []
    for tet_index, tet in enumerate(oriented_tetra):
        for omitted in range(4):
            induced = np.delete(tet, omitted)
            canonical = np.sort(induced)
            sign = (-1 if omitted % 2 else 1) * _permutation_sign(induced)
            b3_rows.append(face_lookup[tuple(canonical)])
            b3_cols.append(tet_index)
            b3_data.append(sign)
    b3 = sparse.coo_matrix((b3_data, (b3_rows, b3_cols)), shape=(n_faces, n_tetra)).tocsr()

    product12 = b1 @ b2
    product23 = b2 @ b3
    chain_12_error = float(np.max(np.abs(product12.data), initial=0))
    chain_23_error = float(np.max(np.abs(product23.data), initial=0))
    if chain_12_error != 0.0 or chain_23_error != 0.0:
        raise RuntimeError(f"Invalid chain complex: B1B2={chain_12_error}, B2B3={chain_23_error}")

    lap0 = b1 @ b1.T
    lap1 = b1.T @ b1 + b2 @ b2.T
    lap2 = b2.T @ b2 + b3 @ b3.T
    harmonic0 = np.full((n_vertices, 1), 1.0 / np.sqrt(n_vertices), dtype=np.float32)
    harmonic1, residual1 = _harmonic_basis(lap1, beta1)
    harmonic2, residual2 = _harmonic_basis(lap2, beta2)

    edge_vectors = points[edges[:, 1]] - points[edges[:, 0]]
    face_vectors_1 = points[faces[:, 1]] - points[faces[:, 0]]
    face_vectors_2 = points[faces[:, 2]] - points[faces[:, 0]]
    face_area_vectors = 0.5 * np.cross(face_vectors_1, face_vectors_2)
    tet_points = points[oriented_tetra]
    tet_volumes = np.abs(np.einsum(
        "ij,ij->i",
        np.cross(tet_points[:, 1] - tet_points[:, 0], tet_points[:, 2] - tet_points[:, 0]),
        tet_points[:, 3] - tet_points[:, 0],
    )) / 6.0
    vertex_lumped_volume = np.bincount(oriented_tetra.ravel(), np.repeat(tet_volumes / 4.0, 4), minlength=n_vertices)

    payload = {
        "edges": edges.astype(np.int32),
        "faces": faces.astype(np.int32),
        "oriented_tetra": oriented_tetra.astype(np.int32),
        "edge_vectors": edge_vectors.astype(np.float32),
        "edge_lengths": np.linalg.norm(edge_vectors, axis=1).astype(np.float32),
        "face_area_vectors": face_area_vectors.astype(np.float32),
        "face_areas": np.linalg.norm(face_area_vectors, axis=1).astype(np.float32),
        "tetra_volumes": tet_volumes.astype(np.float32),
        "vertex_lumped_volume": vertex_lumped_volume.astype(np.float32),
        "harmonic_basis_0": harmonic0,
        "harmonic_basis_1": harmonic1,
        "harmonic_basis_2": harmonic2,
        "harmonic_residual_0": np.zeros(1, dtype=np.float32),
        "harmonic_residual_1": residual1,
        "harmonic_residual_2": residual2,
        "chain_complex_error": np.asarray((chain_12_error, chain_23_error), dtype=np.float32),
    }
    payload.update(_coo_payload("incidence_1", b1))
    payload.update(_coo_payload("incidence_2", b2))
    payload.update(_coo_payload("incidence_3", b3))
    return payload
