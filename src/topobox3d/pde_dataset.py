"""Joined geometry/PDE dataset and native adapters for all seven models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model_inputs import TopoBoxGeometry, _tensor


COCHAIN_NORMALIZATION = "k2_per_sample_input_rms"


def cochain_scale(degree: int, input_cochain: np.ndarray) -> np.float32:
    """Scale raw 2-cochains to an O(1) coefficient range for neural models."""

    if degree != 2:
        return np.float32(1.0)
    rms = float(
        np.sqrt(np.mean(np.square(np.asarray(input_cochain, dtype=np.float64))))
    )
    return np.float32(max(rms, 1e-8))


def _read_dataset(dataset) -> np.ndarray:
    value = dataset[()]
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO":
        if value.size == 0:
            return value.astype(str)
        decode = np.vectorize(
            lambda item: item.decode("utf-8") if isinstance(item, bytes) else str(item)
        )
        return decode(value)
    return np.asarray(value)


def mass_weighted_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mass: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Squared relative Hodge error for native scalar k-cochains."""

    while prediction.ndim > target.ndim and prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)
    if prediction.ndim == 2 and prediction.shape[-1] == 1:
        prediction = prediction[:, 0]
    if target.ndim == 2 and target.shape[-1] == 1:
        target = target[:, 0]
    if prediction.shape != target.shape or target.shape != mass.shape:
        raise ValueError(
            "prediction, target, and mass must reduce to the same cochain shape; "
            f"got {prediction.shape}, {target.shape}, and {mass.shape}"
        )
    error_energy = ((prediction - target).square() * mass).sum()
    target_energy = (target.square() * mass).sum()
    return error_energy / (target_energy + epsilon)


def _simplex_token_arrays(
    degree: int,
    geometry: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active-simplex coordinates, geometry features, and indices.

    k=0 tokens are vertices, k=1 tokens are oriented edges, and k=2 tokens
    are oriented faces.  Edge/face geometry keeps orientation information so
    scalar cochain values remain meaningful without a vertex-vector proxy.
    """

    points = np.asarray(geometry["normalized_xyz"], dtype=np.float32)
    node_geometry = np.asarray(geometry["geometry_features"], dtype=np.float32)
    if degree == 0:
        indices = np.arange(len(points), dtype=np.int64)[:, None]
        return points, node_geometry, indices
    if degree == 1:
        indices = np.asarray(geometry["edges"], dtype=np.int64)
        coordinates = points[indices].mean(axis=1)
        local_geometry = node_geometry[indices].mean(axis=1)
        oriented_measure = np.asarray(
            geometry["edge_vectors"], dtype=np.float32
        )
        measure = np.asarray(
            geometry["edge_lengths"], dtype=np.float32
        )[:, None]
    elif degree == 2:
        indices = np.asarray(geometry["faces"], dtype=np.int64)
        coordinates = points[indices].mean(axis=1)
        local_geometry = node_geometry[indices].mean(axis=1)
        oriented_measure = np.asarray(
            geometry["face_area_vectors"], dtype=np.float32
        )
        measure = np.asarray(
            geometry["face_areas"], dtype=np.float32
        )[:, None]
    else:
        raise ValueError("Only k=0,1,2 are supported")
    token_geometry = np.concatenate(
        (local_geometry, oriented_measure, measure), axis=1
    )
    return coordinates, token_geometry, indices


def _simplex_adjacency(
    degree: int,
    geometry: dict[str, np.ndarray],
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Connect k-simplices that share one (k+1)-coface.

    This gives vertex adjacency through edges, edge adjacency through faces,
    and face adjacency through tetrahedra, without exposing Betti numbers or
    harmonic bases to topology-unaware models.
    """

    prefix = f"incidence_{degree + 1}"
    simplex_indices = np.asarray(
        geometry[f"{prefix}_row"], dtype=np.int64
    )
    coface_indices = np.asarray(
        geometry[f"{prefix}_col"], dtype=np.int64
    )
    order = np.argsort(coface_indices, kind="stable")
    simplex_indices = simplex_indices[order]
    coface_indices = coface_indices[order]
    boundaries = np.flatnonzero(np.diff(coface_indices)) + 1
    groups = np.split(simplex_indices, boundaries)
    senders = []
    receivers = []
    for group in groups:
        group = np.unique(group)
        if len(group) < 2:
            continue
        sender = np.repeat(group, len(group))
        receiver = np.tile(group, len(group))
        keep = sender != receiver
        senders.append(sender[keep])
        receivers.append(receiver[keep])
    if not senders:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 4), dtype=np.float32)
        return edge_index, edge_attr
    sender = np.concatenate(senders)
    receiver = np.concatenate(receivers)
    edge_index = np.stack((sender, receiver))
    delta = coordinates[receiver] - coordinates[sender]
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    edge_attr = np.concatenate((delta, length), axis=1).astype(np.float32)
    return edge_index, edge_attr


@dataclass
class TopoBoxPDESample:
    geometry: TopoBoxGeometry
    geometry_id: str
    protocol: str
    split: str
    degree: int
    config_name: str
    config_index: int
    w0: np.ndarray
    wT: np.ndarray
    mass: np.ndarray
    harmonic_basis: np.ndarray
    requested_energy_fractions: np.ndarray
    realized_energy_fractions: np.ndarray
    relative_final_mass_norm: float
    beta1: int
    beta2: int
    precomputed_simplex_token_arrays: (
        tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ) = None
    precomputed_simplex_adjacency_arrays: (
        tuple[np.ndarray, np.ndarray] | None
    ) = None

    @cached_property
    def simplex_token_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.precomputed_simplex_token_arrays is not None:
            return self.precomputed_simplex_token_arrays
        return _simplex_token_arrays(self.degree, self.geometry.data)

    @cached_property
    def simplex_adjacency_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.precomputed_simplex_adjacency_arrays is not None:
            return self.precomputed_simplex_adjacency_arrays
        coordinates, _, _ = self.simplex_token_arrays
        return _simplex_adjacency(
            self.degree, self.geometry.data, coordinates
        )

    def prediction_to_cochain(self, prediction) -> np.ndarray:
        """Convert a native scalar k-simplex prediction to a 1D cochain."""

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        prediction = np.asarray(prediction, dtype=np.float32)
        while prediction.ndim > 1 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.shape == (len(self.wT), 1):
            prediction = prediction[:, 0]
        if prediction.shape != self.wT.shape:
            raise ValueError(
                f"k={self.degree} prediction has shape {prediction.shape}; "
                f"expected {self.wT.shape} or {(len(self.wT), 1)}"
            )
        return prediction

    def _common_targets(self, device=None) -> dict:
        scale = cochain_scale(self.degree, self.w0)
        return {
            "target_simplex_field": _tensor(
                self.wT[:, None], device, torch.float32
            ),
            "target_cochain": _tensor(self.wT, device, torch.float32),
            "input_cochain": _tensor(self.w0, device, torch.float32),
            "normalized_input_cochain": _tensor(
                self.w0 / scale, device, torch.float32
            ),
            "cochain_scale": _tensor(scale, device, torch.float32),
            "mass": _tensor(self.mass, device, torch.float32),
            "target_harmonic_basis": _tensor(
                self.harmonic_basis, device, torch.float32
            ),
            "degree": self.degree,
            "config_name": self.config_name,
            "geometry_id": self.geometry_id,
            "protocol": self.protocol,
            "split": self.split,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "realized_energy_fractions": _tensor(
                self.realized_energy_fractions, device, torch.float32
            ),
        }

    def for_model(self, name: str, device=None) -> dict:
        """Return the native geometry structure plus model-ready PDE fields."""

        key = name.lower()
        targets = self._common_targets(device)
        if key == "tno":
            base = self.geometry.for_model("tno")
            incidence = {
                rank: matrix.to(device=device)
                for rank, matrix in base["incidence"].items()
            }
            cochains = []
            for rank, geometry_features in enumerate(base["cochains"]):
                geometry_features = geometry_features.to(device=device)
                physics = torch.zeros(
                    (geometry_features.shape[0], 1),
                    dtype=geometry_features.dtype,
                    device=geometry_features.device,
                )
                if rank == self.degree:
                    physics[:, 0] = targets["normalized_input_cochain"]
                cochains.append(torch.cat((geometry_features, physics), dim=-1))
            return base | targets | {
                "input_cochains": cochains,
                "incidence": incidence,
                "incidence_3": base["incidence_3"].to(device=device),
                "harmonic_basis": {
                    self.degree: targets["target_harmonic_basis"]
                },
                "harmonic_mass": {
                    self.degree: targets["mass"]
                },
                "active_degree": self.degree,
            }

        coordinates_np, token_geometry_np, simplex_indices_np = (
            self.simplex_token_arrays
        )
        coordinates = _tensor(coordinates_np, device, torch.float32)
        token_geometry = _tensor(token_geometry_np, device, torch.float32)
        simplex_indices = _tensor(simplex_indices_np, device, torch.long)
        input_field = targets["normalized_input_cochain"][:, None]
        features = torch.cat((token_geometry, input_field), dim=-1)
        common = targets | {
            "token_coordinates": coordinates,
            "token_geometry": token_geometry,
            "token_features": features,
            "simplex_indices": simplex_indices,
            "input_field": input_field,
        }
        if key in ("mgn", "mgn-lite"):
            edge_index_np, edge_attr_np = self.simplex_adjacency_arrays
            return common | {
                "nodes": features,
                "edge_index": _tensor(edge_index_np, device, torch.long),
                "edge_attr": _tensor(edge_attr_np, device, torch.float32),
            }
        if key == "rigno":
            return common | {
                "x": coordinates,
                "x_batched": coordinates.unsqueeze(0).unsqueeze(0),
                "domain": torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    dtype=torch.float32,
                    device=device,
                ),
                "u": input_field.unsqueeze(0).unsqueeze(0),
                "c": token_geometry.unsqueeze(0).unsqueeze(0),
                "node_features": features,
            }
        if key == "transolver":
            return common | {
                "x": coordinates.unsqueeze(0),
                "fx": features.unsqueeze(0),
            }
        if key == "gnot":
            # The official GNOT forward only uses DGL as a variable-length
            # node container and never consumes its edges. Avoid constructing
            # the expensive same-rank simplex adjacency for this adapter.
            edge_index_np = np.empty((2, 0), dtype=np.int64)
            edge_attr_np = np.empty((0, 4), dtype=np.float32)
            # GNOT assumes the first ``space_dim`` trunk channels are query
            # coordinates (they drive its spatial MoE gate).  Branch tokens
            # also need coordinates so linear cross-attention can associate
            # local input-cochain values with the corresponding query.
            query_features = torch.cat(
                (coordinates, token_geometry, input_field), dim=-1
            )
            branch_features = torch.cat((coordinates, features), dim=-1)
            return common | {
                "query_graph_x": query_features,
                "input_function_graph_x": features,
                "branch_inputs": [branch_features.unsqueeze(0)],
                "edge_index": _tensor(edge_index_np, device, torch.long),
                "edge_attr": _tensor(edge_attr_np, device, torch.float32),
                "global_parameters": torch.empty(
                    (1, 0), dtype=torch.float32, device=device
                ),
            }
        if key == "gaot":
            latent = np.asarray(
                self.geometry.data["regular_grid_normalized_xyz"][
                    ::4, ::4, ::4
                ].reshape(-1, 3),
                dtype=np.float32,
            )
            return common | {
                "latent_tokens_coord": _tensor(
                    latent, device, torch.float32
                ),
                "latent_tokens_size": (8, 4, 4),
                "xcoord": coordinates,
                "pndata": features.unsqueeze(0),
                "query_coord": coordinates,
            }
        raise ValueError(f"Unknown TopoBox model adapter: {name}")


class TopoBoxPDEDataset(Dataset):
    """Map-style dataset expanded over geometry, degree, and initial condition."""

    def __init__(
        self,
        geometry_packed_root: str | Path,
        solution_root: str | Path,
        protocol: str,
        split: str,
        degrees=(0, 1, 2),
        configs=None,
        device=None,
        preload: bool = False,
        cache_derived: bool = False,
        cache_adjacency: bool = False,
    ):
        self.geometry_root = Path(geometry_packed_root)
        self.solution_root = Path(solution_root)
        self.device = device
        self.cache_derived = cache_derived
        self.cache_adjacency = cache_adjacency
        geometry_records = json.loads(
            (self.geometry_root / "index.json").read_text(encoding="utf-8")
        )
        solution_records = json.loads(
            (self.solution_root / "index.json").read_text(encoding="utf-8")
        )
        geometry_by_id = {
            record["geometry_id"]: record for record in geometry_records
        }
        selected_solutions = [
            record for record in solution_records
            if record["protocol"].upper() == protocol.upper()
            and record["split"] == split
        ]
        manifest = json.loads(
            (self.solution_root / "manifest.json").read_text(encoding="utf-8")
        )
        self.config_names = manifest["config_names"]
        if configs is None:
            config_indices = range(len(self.config_names))
        else:
            config_indices = [
                self.config_names.index(config) if isinstance(config, str) else int(config)
                for config in configs
            ]
        self.items = []
        for solution in selected_solutions:
            geometry = geometry_by_id.get(solution["geometry_id"])
            if geometry is None:
                raise KeyError(f"Missing packed geometry {solution['geometry_id']}")
            for degree in degrees:
                for config_index in config_indices:
                    self.items.append(
                        (geometry, solution, int(degree), int(config_index))
                    )
        self._geometry_handles = {}
        self._solution_handles = {}
        self._geometry_memory: dict[str, TopoBoxGeometry] = {}
        self._solution_memory: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self._simplex_memory: dict[
            tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._adjacency_memory: dict[
            tuple[str, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        if preload:
            self.preload()

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _handle(cache: dict, root: Path, relative: str):
        if relative not in cache:
            cache[relative] = h5py.File(root / relative, "r")
        return cache[relative]

    def preload(self) -> dict[str, int]:
        """Load this split into process RAM and close all HDF5 handles.

        Arrays shared by a shard remain shared Python objects rather than being
        duplicated for every geometry. The returned byte counts describe
        unique NumPy buffers and are useful for logging resource use.
        """

        shared_by_shard: dict[str, dict[str, np.ndarray]] = {}
        seen_geometry: set[str] = set()
        for geometry_record, solution_record, degree, _ in self.items:
            geometry_id = geometry_record["geometry_id"]
            if geometry_id not in seen_geometry:
                geometry_handle = self._handle(
                    self._geometry_handles,
                    self.geometry_root,
                    geometry_record["shard"],
                )
                geometry_group = geometry_handle[geometry_record["group"]]
                arrays = {
                    name: _read_dataset(dataset)
                    for name, dataset in geometry_group.items()
                    if name != "metadata_json"
                }
                shard = geometry_record["shard"]
                if shard not in shared_by_shard:
                    shared_by_shard[shard] = {
                        name: _read_dataset(dataset)
                        for name, dataset in geometry_handle["shared"].items()
                    }
                arrays.update(shared_by_shard[shard])
                self._geometry_memory[geometry_id] = (
                    TopoBoxGeometry.from_arrays(arrays, self.device)
                )
                seen_geometry.add(geometry_id)
            solution_key = (geometry_id, degree)
            if solution_key not in self._solution_memory:
                solution_handle = self._handle(
                    self._solution_handles,
                    self.solution_root,
                    solution_record["shard"],
                )
                group = solution_handle[solution_record["group"]][f"k{degree}"]
                self._solution_memory[solution_key] = {
                    name: _read_dataset(dataset)
                    for name, dataset in group.items()
                }
        self.close()
        buffers: dict[int, int] = {}
        for geometry in self._geometry_memory.values():
            for value in geometry.data.values():
                if isinstance(value, np.ndarray):
                    buffers.setdefault(id(value), value.nbytes)
        for group in self._solution_memory.values():
            for value in group.values():
                if isinstance(value, np.ndarray):
                    buffers.setdefault(id(value), value.nbytes)
        return {
            "geometry_count": len(self._geometry_memory),
            "solution_group_count": len(self._solution_memory),
            "numpy_bytes": sum(buffers.values()),
        }

    def __getitem__(self, index: int) -> TopoBoxPDESample:
        geometry_record, solution_record, degree, config_index = self.items[index]
        geometry_id = geometry_record["geometry_id"]
        if geometry_id in self._geometry_memory:
            geometry = self._geometry_memory[geometry_id]
        else:
            geometry_handle = self._handle(
                self._geometry_handles,
                self.geometry_root,
                geometry_record["shard"],
            )
            geometry_group = geometry_handle[geometry_record["group"]]
            arrays = {
                name: _read_dataset(dataset)
                for name, dataset in geometry_group.items()
                if name != "metadata_json"
            }
            arrays.update(
                {
                    name: _read_dataset(dataset)
                    for name, dataset in geometry_handle["shared"].items()
                }
            )
            geometry = TopoBoxGeometry.from_arrays(arrays, self.device)

        solution_key = (geometry_id, degree)
        if solution_key in self._solution_memory:
            group = self._solution_memory[solution_key]
        else:
            solution_handle = self._handle(
                self._solution_handles,
                self.solution_root,
                solution_record["shard"],
            )
            group = solution_handle[solution_record["group"]][f"k{degree}"]

        simplex_arrays = None
        adjacency_arrays = None
        if self.cache_derived:
            if solution_key not in self._simplex_memory:
                self._simplex_memory[solution_key] = _simplex_token_arrays(
                    degree, geometry.data
                )
            simplex_arrays = self._simplex_memory[solution_key]
            if self.cache_adjacency:
                if solution_key not in self._adjacency_memory:
                    self._adjacency_memory[solution_key] = _simplex_adjacency(
                        degree,
                        geometry.data,
                        simplex_arrays[0],
                    )
                adjacency_arrays = self._adjacency_memory[solution_key]
        return TopoBoxPDESample(
            geometry=geometry,
            geometry_id=solution_record["geometry_id"],
            protocol=solution_record["protocol"],
            split=solution_record["split"],
            degree=degree,
            config_name=self.config_names[config_index],
            config_index=config_index,
            w0=np.asarray(group["w0"][config_index]),
            wT=np.asarray(group["wT"][config_index]),
            mass=np.asarray(group["mass"]),
            harmonic_basis=np.asarray(group["harmonic_basis"]),
            requested_energy_fractions=np.asarray(
                group["requested_energy_fractions"][config_index]
            ),
            realized_energy_fractions=np.asarray(
                group["realized_energy_fractions"][config_index]
            ),
            relative_final_mass_norm=float(
                group["relative_final_mass_norm"][config_index]
            ),
            beta1=int(geometry_record["beta1"]),
            beta2=int(geometry_record["beta2"]),
            precomputed_simplex_token_arrays=simplex_arrays,
            precomputed_simplex_adjacency_arrays=adjacency_arrays,
        )

    def close(self) -> None:
        for cache in (self._geometry_handles, self._solution_handles):
            for handle in cache.values():
                handle.close()
            cache.clear()

    def clear_memory_cache(self) -> None:
        """Release arrays retained by ``preload`` and derived-data caches."""

        self._geometry_memory.clear()
        self._solution_memory.clear()
        self._simplex_memory.clear()
        self._adjacency_memory.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_geometry_handles"] = {}
        state["_solution_handles"] = {}
        return state

    def __del__(self):
        self.close()


def single_pde_sample_collate(batch):
    if len(batch) != 1:
        raise ValueError(
            "Variable-size TopoBox meshes require batch_size=1 or a model-specific collator."
        )
    return batch[0]


@dataclass
class ModelBatchCollator:
    """Pickle-safe batch-size-one adapter for PyTorch DataLoader workers."""

    model_name: str

    def __call__(self, batch):
        sample = single_pde_sample_collate(batch)
        return sample.for_model(self.model_name)


def make_model_dataloader(
    geometry_packed_root: str | Path,
    solution_root: str | Path,
    protocol: str,
    split: str,
    model_name: str,
    degrees=(0, 1, 2),
    configs=None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> tuple[TopoBoxPDEDataset, DataLoader]:
    """Construct the canonical batch-size-one loader for one benchmark model."""

    dataset = TopoBoxPDEDataset(
        geometry_packed_root,
        solution_root,
        protocol,
        split,
        degrees=degrees,
        configs=configs,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ModelBatchCollator(model_name),
        persistent_workers=num_workers > 0,
    )
    return dataset, loader
