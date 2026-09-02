"""One geometry loader with adapters for all TopoBox-3D benchmark models."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def _tensor(array: np.ndarray, device=None, dtype=None) -> torch.Tensor:
    result = torch.from_numpy(np.asarray(array))
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result.to(device=device) if device is not None else result


def _sparse(data, prefix: str, device=None) -> torch.Tensor:
    indices = np.stack((data[f"{prefix}_row"], data[f"{prefix}_col"]))
    tensor = torch.sparse_coo_tensor(
        _tensor(indices, device=device, dtype=torch.long),
        _tensor(data[f"{prefix}_value"], device=device, dtype=torch.float32),
        tuple(int(x) for x in data[f"{prefix}_shape"]),
        device=device,
    )
    return tensor.coalesce()


class TopoBoxGeometry:
    """Load one variable-size sample; use batch size one or a graph batcher."""

    def __init__(self, path: str | Path, device=None):
        path = Path(path)
        if path.is_dir():
            path = path / "mesh.npz"
        self.path = path
        self.data = np.load(path, allow_pickle=False)
        self.device = device

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], device=None) -> "TopoBoxGeometry":
        instance = cls.__new__(cls)
        instance.path = None
        instance.data = arrays
        instance.device = device
        return instance

    @property
    def feature_names(self) -> list[str]:
        return self.data["geometry_feature_names"].tolist()

    def common(self) -> dict[str, torch.Tensor]:
        return {
            "coordinates": _tensor(self.data["normalized_xyz"], self.device, torch.float32),
            "node_geometry": _tensor(self.data["geometry_features"], self.device, torch.float32),
        }

    def mgn(self) -> dict[str, torch.Tensor]:
        edges = self.data["edges"].astype(np.int64)
        directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
        points = self.data["normalized_xyz"]
        delta = points[directed[:, 1]] - points[directed[:, 0]]
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        edge_attr = np.concatenate((delta, length), axis=1)
        common = self.common()
        return common | {
            "nodes": common["node_geometry"],
            "edge_index": _tensor(directed.T, self.device, torch.long),
            "edge_attr": _tensor(edge_attr, self.device, torch.float32),
        }

    def rigno(self) -> dict[str, torch.Tensor]:
        common = self.common()
        return common | {"x": common["coordinates"], "node_features": common["node_geometry"]}

    def transolver(self) -> dict[str, torch.Tensor]:
        common = self.common()
        return common | {"x": common["coordinates"].unsqueeze(0), "fx": common["node_geometry"].unsqueeze(0)}

    def gnot(self) -> dict[str, torch.Tensor]:
        # GNOT uses DGL graphs as variable-length point containers. Its u_p is
        # a per-geometry global parameter vector, not a node field; topology
        # labels are deliberately not exposed as global inputs.
        result = self.mgn()
        return result | {
            "query_graph_x": result["node_geometry"],
            "input_function_graph_x": result["node_geometry"],
            "global_parameters": torch.empty(0, device=result["node_geometry"].device),
        }

    def gaot(self) -> dict[str, torch.Tensor]:
        common = self.common()
        # GAOT applies global attention to regional tokens. A 4x stride gives
        # 8x4x4=128 geometry tokens for the latent graph adapter.
        latent = self.data["regular_grid_normalized_xyz"][::4, ::4, ::4].reshape(-1, 3)
        return common | {
            "latent_tokens_coord": _tensor(latent, self.device, torch.float32),
            "latent_tokens_size": (8, 4, 4),
            "xcoord": common["coordinates"],
            "pndata": common["node_geometry"].unsqueeze(0),
            "query_coord": common["coordinates"],
        }

    def tno(self) -> dict:
        node = self.data["geometry_features"]
        edges, faces = self.data["edges"], self.data["faces"]
        edge_geometry = np.concatenate((node[edges].mean(axis=1), self.data["edge_vectors"], self.data["edge_lengths"][:, None]), axis=1)
        face_geometry = np.concatenate((node[faces].mean(axis=1), self.data["face_area_vectors"], self.data["face_areas"][:, None]), axis=1)
        return {
            "cochains": [
                _tensor(node, self.device, torch.float32),
                _tensor(edge_geometry, self.device, torch.float32),
                _tensor(face_geometry, self.device, torch.float32),
            ],
            "incidence": {rank: _sparse(self.data, f"incidence_{rank}", self.device) for rank in (1, 2)},
            "incidence_3": _sparse(self.data, "incidence_3", self.device),
            "harmonic_basis": {
                rank: _tensor(self.data[f"harmonic_basis_{rank}"], self.device, torch.float32)
                for rank in (0, 1, 2)
            },
            "node_geometry": _tensor(node, self.device, torch.float32),
        }

    def for_model(self, name: str) -> dict:
        aliases = {"mgn-lite": "mgn", "transolver": "transolver", "gnot": "gnot", "rigno": "rigno", "gaot": "gaot", "tno": "tno"}
        key = aliases.get(name.lower(), name.lower())
        if not hasattr(self, key):
            raise ValueError(f"Unknown TopoBox model adapter: {name}")
        return getattr(self, key)()


class TopoBoxShardDataset(Dataset):
    """Lazy map-style Dataset over packed HDF5 shards.

    Each worker keeps its own read-only shard handles. Items are returned as
    ``TopoBoxGeometry`` objects so the same per-model adapters work for raw NPZ
    files and packed training data.
    """

    def __init__(self, packed_root: str | Path, protocol: str, split: str, device=None):
        self.root = Path(packed_root)
        records = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.records = [
            item for item in records
            if item["protocol"].upper() == protocol.upper() and item["split"] == split
        ]
        self.device = device
        self._handles = {}

    def __len__(self) -> int:
        return len(self.records)

    def _handle(self, relative_path: str):
        import h5py
        if relative_path not in self._handles:
            self._handles[relative_path] = h5py.File(self.root / relative_path, "r")
        return self._handles[relative_path]

    @staticmethod
    def _read_dataset(dataset) -> np.ndarray:
        value = dataset[()]
        if isinstance(value, np.ndarray) and value.dtype.kind in "SO":
            if value.size == 0:
                return value.astype(str)
            decode = np.vectorize(lambda item: item.decode("utf-8") if isinstance(item, bytes) else str(item))
            return decode(value)
        return np.asarray(value)

    def __getitem__(self, index: int) -> TopoBoxGeometry:
        record = self.records[index]
        handle = self._handle(record["shard"])
        group = handle[record["group"]]
        arrays = {
            name: self._read_dataset(dataset)
            for name, dataset in group.items() if name != "metadata_json"
        }
        arrays.update({
            name: self._read_dataset(dataset)
            for name, dataset in handle["shared"].items()
        })
        item = TopoBoxGeometry.from_arrays(arrays, self.device)
        item.metadata = json.loads(bytes(group["metadata_json"][()]).decode("utf-8"))
        return item

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        self.close()


def single_geometry_collate(batch):
    if len(batch) != 1:
        raise ValueError("Variable-size TopoBox meshes require batch_size=1 or a model-specific collator.")
    return batch[0]
