import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import h5py
import gc

from ..core.config import Config
from ..core.sequence import Sequence, SequenceCollection
from ..core.data_manager import DataManager
from ..core.results import AnalysisResults
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
        self.results = AnalysisResults()

    def run_analysis(
        self, sequences: Union[SequenceCollection, str, Path]
    ) -> AnalysisResults:
        if isinstance(sequences, (str, Path)):
            logger.info(f"Loading sequences from {sequences}")
            sequences = self.data_manager.load_sequences(str(sequences))

        logger.info(f"Running analysis on {len(sequences)} sequences")

        self.results.initialize_storage(self.config.output_dir)
        self.results.set_config(self.config)

        feature_matrix, feature_names = self._compute_feature_matrix(sequences)
        self.results.store_feature_matrix(feature_matrix, feature_names)

        similarity_matrix = self._compute_similarity_matrix(sequences)
        self.results.store_similarity_matrix(similarity_matrix)

        clustering_results, cluster_labels = self._run_clustering(
            similarity_matrix, feature_matrix
        )
        self.results.add_clustering_results(clustering_results, cluster_labels)

        statistics_results = self._run_statistics(
            sequences, similarity_matrix, feature_matrix, clustering_results
        )
        self.results.add_statistics_results(statistics_results)

        # Run ML analysis if labels are present
        if hasattr(sequences, "get_labels") and sequences.get_labels() is not None:
            labels = sequences.get_labels()
            if np.any(labels != None):
                self.results.store_labels(labels)
                ml_results = self._run_ml_analysis(feature_matrix, labels)
                self.results.add_ml_results(ml_results)

        gc.collect()
        return self.results

    def _compute_feature_matrix(self, sequences: SequenceCollection) -> np.ndarray:
        logger.info("Computing feature matrix")
        feature_matrix = self.similarity_calc.compute_feature_matrix(sequences)
        feature_names = self.similarity_calc.FEATURE_NAMES
        similarity_results = self.similarity_calc.get_results()
        self.results.add_similarity_results(similarity_results)

        return feature_matrix, feature_names

    def _compute_similarity_matrix(self, sequences: SequenceCollection) -> np.ndarray:
        logger.info("Computing similarity matrix")
        return self.similarity_calc.compute_similarity_matrix(sequences)

    def _run_clustering(
        self, similarity_matrix: np.ndarray, feature_matrix: np.ndarray
    ) -> Tuple[Dict, Dict[str, np.ndarray]]:
        logger.info("Running clustering analysis")
        raw_results = self.clustering.run_clustering(similarity_matrix, feature_matrix)

        # Extract labels to be stored separately
        cluster_labels = {}
        results_without_labels = {}

        for method, result in raw_results.items():
            method_result = result.copy()
            if "labels" in method_result:
                cluster_labels[method] = np.array(method_result["labels"])
                label_stats = self._get_label_stats(method_result["labels"])
                method_result["labels_info"] = label_stats
                del method_result["labels"]

            results_without_labels[method] = method_result

        return results_without_labels, cluster_labels

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

    def _run_statistics(
        self,
        sequences: SequenceCollection,
        similarity_matrix: np.ndarray,
        feature_matrix: np.ndarray,
        clustering_results: Dict,
    ) -> Dict:
        logger.info("Running statistical analysis")
        labels = None
        if hasattr(sequences, "get_labels"):
            labels = sequences.get_labels()

        return self.statistics.run_statistics(
            sequences, similarity_matrix, feature_matrix, clustering_results, labels
        )

    def _run_ml_analysis(self, feature_matrix: np.ndarray, labels: np.ndarray) -> Dict:
        logger.info("Running ML analysis")
        return self.ml_analyzer.run_ml_analysis(feature_matrix, labels)

    def save_results(self) -> None:
        self.results.save_summary()
        self.results.close()
