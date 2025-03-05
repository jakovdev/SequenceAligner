import logging
import numpy as np
import h5py
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


class AnalysisResults:

    def __init__(self):
        # Basic metadata
        self.timestamp = datetime.now().isoformat()
        self.config = None

        # Storage for large matrices
        self._hdf5_path = None
        self._h5_file = None

        # Small results (dictionaries that don't grow with dataset size)
        self.metadata = {}
        self.summary = {}

    def initialize_storage(self, output_dir: Union[str, Path]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        self._hdf5_path = output_path / "analysis_results.h5"
        self._h5_file = h5py.File(self._hdf5_path, "w")

        self._h5_file.create_group("matrices")
        self._h5_file.create_group("classifications")
        self._h5_file.create_group("clusters")

        self._h5_file["matrices"].attrs["description"] = "Large data matrices"
        self._h5_file["classifications"].attrs["description"] = "Classification labels"
        self._h5_file["clusters"].attrs["description"] = "Clustering results"

        logger.info(f"Initialized results storage at {self._hdf5_path}")

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def set_config(self, config) -> None:
        self.config = config
        self.metadata["config"] = config.dict(exclude={"matrix_presets"})
        self.metadata["timestamp"] = self.timestamp

    def store_similarity_matrix(self, matrix: np.ndarray) -> None:
        self._store_matrix(matrix, "similarity_matrix")
        self.summary["similarity_stats"] = self._compute_matrix_stats(matrix)

    def store_feature_matrix(
        self, matrix: np.ndarray, feature_names: List[str]
    ) -> None:
        self._store_matrix(matrix, "feature_matrix")
        self.summary["feature_names"] = feature_names

    def store_labels(self, labels: np.ndarray) -> None:
        if self._h5_file is None:
            raise ValueError("Storage not initialized.")

        labels_int = labels.astype(np.int32)
        dataset = self._h5_file["classifications"].create_dataset(
            "labels",
            data=labels_int,
            compression="gzip",
            dtype=np.int32,
        )

        dataset.attrs["type"] = "categorical"
        dataset.attrs["unique_values"] = np.unique(labels_int).tolist()

    def _store_matrix(self, matrix: np.ndarray, name: str) -> None:
        if self._h5_file is None:
            raise ValueError("Storage not initialized. Call initialize_storage first.")

        # For feature matrix, store it transposed if it's tall and thin
        if name == "feature_matrix" and matrix.shape[0] > matrix.shape[1]:
            dataset = self._h5_file["matrices"].create_dataset(
                name, data=matrix.T, compression="gzip"
            )
            dataset.attrs["transposed"] = True
            dataset.attrs["original_shape"] = matrix.shape
        else:
            dataset = self._h5_file["matrices"].create_dataset(
                name, data=matrix, compression="gzip"
            )
            dataset.attrs["transposed"] = False

    def _compute_matrix_stats(self, matrix: np.ndarray) -> Dict:
        if len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]:
            # For similarity matrices, compute stats on upper triangle
            triu_indices = np.triu_indices_from(matrix, k=1)
            data = matrix[triu_indices]
        else:
            data = matrix.flatten()

        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "shape": matrix.shape,
        }

    def add_similarity_results(self, results: Dict) -> None:
        if "feature_names" in results:
            self.summary["feature_names"] = results["feature_names"]
            del results["feature_names"]

        self.summary["similarity_analysis"] = results

    def add_clustering_results(
        self, results: Dict, labels_dict: Dict[str, np.ndarray]
    ) -> None:
        for method, labels in labels_dict.items():
            labels_int = labels.astype(np.int32)
            dataset = self._h5_file["clusters"].create_dataset(
                method, data=labels_int, compression="gzip", dtype=np.int32
            )

            dataset.attrs["type"] = "categorical"
            dataset.attrs["method"] = method
            dataset.attrs["unique_values"] = np.unique(labels_int).tolist()

        self.summary["clustering"] = results

    def add_statistics_results(self, results: Dict) -> None:
        self.summary["statistics"] = results

    def add_ml_results(self, results: Dict) -> None:
        self.summary["ml_analysis"] = results

    def get_matrix(self, name: str) -> np.ndarray:
        if self._h5_file is None or not self._h5_file.id.valid:
            self._h5_file = h5py.File(self._hdf5_path, "r")

        dataset = self._h5_file["matrices"][name]
        matrix = dataset[:]

        if dataset.attrs.get("transposed", False):
            return matrix.T
        return matrix

    def get_labels(self, kind: str = "labels") -> np.ndarray:
        if self._h5_file is None or not self._h5_file.id.valid:
            self._h5_file = h5py.File(self._hdf5_path, "r")

        return self._h5_file["classifications"][kind][:]

    def get_cluster_labels(self, method: str) -> np.ndarray:
        if self._h5_file is None or not self._h5_file.id.valid:
            self._h5_file = h5py.File(self._hdf5_path, "r")

        return self._h5_file["clusters"][method][:]

    def get_feature_names(self) -> List[str]:
        return self.summary.get("feature_names", [])

    def save_summary(self, output_path: Optional[Union[str, Path]] = None) -> None:
        if output_path is None and self._hdf5_path is not None:
            output_path = self._hdf5_path.parent / "analysis_summary.json"
        else:
            output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump({"metadata": self.metadata, "summary": self.summary}, f, indent=2)

        logger.info(f"Saved analysis summary to {output_path}")
