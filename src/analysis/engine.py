import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import h5py
import gc

from ..core.config import Config
from ..core.sequence import Sequence, SequenceCollection
from ..core.data_manager import DataManager
from .similarity import SimilarityCalculator
from .clustering import ClusteringAnalyzer
from .statistics import StatisticsAnalyzer
from .ml import MLAnalyzer

logger = logging.getLogger(__name__)


class AnalysisEngine:

    def __init__(self, config: Config, data_manager: Optional[DataManager] = None):
        self.config = config
        self.data_manager = data_manager or DataManager(config)
        self.similarity_calc = SimilarityCalculator(config)
        self.clustering = ClusteringAnalyzer(config)
        self.statistics = StatisticsAnalyzer(config)
        self.ml_analyzer = MLAnalyzer(config)
        self.results = {}
        self.large_data = {}

    def run_analysis(self, sequences: Union[SequenceCollection, str, Path]) -> Dict:
        # TODO: Review
        if isinstance(sequences, (str, Path)):
            logger.info(f"Loading sequences from {sequences}")
            sequences = self.data_manager.load_sequences(str(sequences))

        logger.info(f"Running analysis on {len(sequences)} sequences")

        feature_matrix = self._compute_feature_matrix(sequences)
        self.large_data["feature_matrix"] = feature_matrix

        similarity_matrix = self._compute_similarity_matrix(sequences)
        self.large_data["similarity_matrix"] = similarity_matrix

        clustering_results = self._run_clustering(similarity_matrix, feature_matrix)
        self._extract_large_clustering_data(clustering_results)

        statistics_results = self._run_statistics(
            sequences, similarity_matrix, feature_matrix, clustering_results
        )

        # TODO: Review
        ml_results = None
        if hasattr(sequences, "get_labels") and sequences.get_labels() is not None:
            labels = sequences.get_labels()
            if np.any(labels != None):
                ml_results = self._run_ml_analysis(feature_matrix, labels)
                self.large_data["labels"] = labels

        self.results = {
            "similarity_analysis": self.similarity_calc.get_results(),
            "clustering": clustering_results,
            "statistics": statistics_results,
        }

        if ml_results:
            self.results["ml_analysis"] = ml_results

        gc.collect()

        return self.results

    def _extract_large_clustering_data(self, clustering_results: Dict) -> None:
        self.large_data["cluster_labels"] = {}

        # TODO: Review
        for method, result in clustering_results.items():
            if "labels" in result:
                self.large_data["cluster_labels"][method] = result["labels"]
                label_stats = self._get_label_stats(result["labels"])
                result["labels_info"] = label_stats
                del result["labels"]

    def _get_label_stats(self, labels: List) -> Dict:
        if not labels:
            return {}

        labels_array = np.array(labels)
        unique_labels = np.unique(labels_array)

        return {
            "unique_values": unique_labels.tolist(),
            "counts": {
                str(label): int(np.sum(labels_array == label))
                for label in unique_labels
            },
            "n_samples": len(labels),
        }

    def save_large_data(self, output_path: str) -> None:
        logger.info(f"Saving large data structures to {output_path}")
        # TODO: Review
        try:
            with h5py.File(output_path, "w") as hf:
                if "feature_matrix" in self.large_data:
                    hf.create_dataset(
                        "feature_matrix",
                        data=self.large_data["feature_matrix"],
                        compression="gzip",
                    )

                if "similarity_matrix" in self.large_data:
                    hf.create_dataset(
                        "similarity_matrix",
                        data=self.large_data["similarity_matrix"],
                        compression="gzip",
                    )

                if "cluster_labels" in self.large_data:
                    cluster_group = hf.create_group("cluster_labels")
                    for method, labels in self.large_data["cluster_labels"].items():
                        cluster_group.create_dataset(
                            method, data=labels, compression="gzip"
                        )

                if "labels" in self.large_data:
                    hf.create_dataset(
                        "labels", data=self.large_data["labels"], compression="gzip"
                    )

            logger.info(f"Successfully saved large data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving large data: {e}")

    def _compute_feature_matrix(self, sequences: SequenceCollection) -> np.ndarray:
        logger.info("Computing feature matrix")
        return self.similarity_calc.compute_feature_matrix(sequences)

    def _compute_similarity_matrix(self, sequences: SequenceCollection) -> np.ndarray:
        logger.info("Computing similarity matrix")
        return self.similarity_calc.compute_similarity_matrix(sequences)

    def _run_clustering(
        self, similarity_matrix: np.ndarray, feature_matrix: np.ndarray
    ) -> Dict:
        logger.info("Running clustering analysis")
        return self.clustering.run_clustering(similarity_matrix, feature_matrix)

    def _run_statistics(
        self,
        sequences: SequenceCollection,
        similarity_matrix: np.ndarray,
        feature_matrix: np.ndarray,
        clustering_results: Dict,
    ) -> Dict:
        logger.info("Running statistical analysis")
        # TODO: Review
        labels = None
        if hasattr(sequences, "get_labels"):
            labels = sequences.get_labels()

        return self.statistics.run_statistics(
            sequences, similarity_matrix, feature_matrix, clustering_results, labels
        )

    def _run_ml_analysis(self, feature_matrix: np.ndarray, labels: np.ndarray) -> Dict:
        logger.info("Running ML analysis")
        return self.ml_analyzer.run_ml_analysis(feature_matrix, labels)

    def get_results(self) -> Dict:
        return self.results

    def get_large_data(self) -> Dict:
        return self.large_data
