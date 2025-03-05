import logging
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import gc
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from ..core.config import Config

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:

    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        self.clustering_methods = {
            "kmeans": self._run_kmeans,
            "dbscan": self._run_dbscan,
            "spectral": self._run_spectral,
            "hierarchical": self._run_hierarchical,
        }

    def run_clustering(
        self, similarity_matrix: np.ndarray, feature_matrix: np.ndarray
    ) -> Dict:
        logger.info("Running clustering analysis")

        self.similarity_matrix = similarity_matrix
        self.feature_matrix = feature_matrix

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        self.results = {}

        method = self.config.clustering.method

        if method == "all":
            for method_name, method_func in self.clustering_methods.items():
                self._run_method(method_name, method_func, scaled_features)
        else:
            if method in self.clustering_methods:
                self._run_method(
                    method, self.clustering_methods[method], scaled_features
                )
            else:
                logger.error(f"Unknown clustering method: {method}")
                self.results["error"] = f"Unknown clustering method: {method}"

        return self.results

    def _run_method(
        self, method_name: str, method_func: Callable, features: np.ndarray
    ) -> None:
        try:
            logger.info(f"Running {method_name} clustering")
            method_func(features)
            gc.collect()
        except Exception as e:
            logger.error(f"Error in {method_name} clustering: {str(e)}")
            self.results[method_name] = {"error": str(e)}

    def _run_kmeans(self, features: np.ndarray) -> None:
        n_clusters = self.config.clustering.n_clusters
        random_seed = self.config.clustering.random_seed

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        labels = kmeans.fit_predict(features)

        self.results["kmeans"] = {
            "labels": labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "inertia": float(kmeans.inertia_),
            "cluster_sizes": self._get_cluster_sizes(labels, n_clusters),
            **self._calculate_clustering_metrics(features, labels),
        }

    def _run_dbscan(self, features: np.ndarray) -> None:
        eps = self.config.clustering.eps
        min_samples = self.config.clustering.min_samples

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        metrics = {}
        if n_clusters > 0:
            metrics = self._calculate_clustering_metrics(
                features, labels, allow_noise=True
            )

        self.results["dbscan"] = {
            "labels": labels.tolist(),
            "n_clusters": int(n_clusters),
            "n_noise": int(np.sum(labels == -1)),
            "cluster_sizes": [
                int(np.sum(labels == i)) for i in unique_labels if i != -1
            ],
            "parameters": {"eps": eps, "min_samples": min_samples},
            **metrics,
        }

    def _run_spectral(self, features: np.ndarray) -> None:
        n_clusters = self.config.clustering.n_clusters
        random_seed = self.config.clustering.random_seed
        affinity_matrix = self._prepare_similarity_matrix()

        spectral = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=random_seed
        )
        labels = spectral.fit_predict(affinity_matrix)
        self.results["spectral"] = {
            "labels": labels.tolist(),
            "n_clusters": int(n_clusters),
            "cluster_sizes": self._get_cluster_sizes(labels, n_clusters),
            **self._calculate_clustering_metrics(features, labels),
        }

    def _run_hierarchical(self, features: np.ndarray) -> None:
        n_clusters = self.config.clustering.n_clusters

        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(features)
        self.results["hierarchical"] = {
            "labels": labels.tolist(),
            "n_clusters": int(n_clusters),
            "cluster_sizes": self._get_cluster_sizes(labels, n_clusters),
            **self._calculate_clustering_metrics(features, labels),
        }

    def _get_cluster_sizes(self, labels: np.ndarray, n_clusters: int) -> List[int]:
        return [int(np.sum(labels == i)) for i in range(n_clusters)]

    def _calculate_clustering_metrics(
        self, features: np.ndarray, labels: np.ndarray, allow_noise: bool = False
    ) -> Dict:
        metrics = {}
        if allow_noise and -1 in labels:
            valid_indices = labels != -1
            if np.sum(valid_indices) < 2:
                return metrics

            features_filtered = features[valid_indices]
            labels_filtered = labels[valid_indices]
        else:
            features_filtered = features
            labels_filtered = labels

        if len(np.unique(labels_filtered)) < 2:
            return metrics

        metric_functions = {
            "silhouette_score": silhouette_score,
            "calinski_harabasz_score": calinski_harabasz_score,
            "davies_bouldin_score": davies_bouldin_score,
        }

        for name, func in metric_functions.items():
            try:
                metrics[name] = float(func(features_filtered, labels_filtered))
            except Exception as e:
                logger.debug(f"Could not calculate {name}: {e}")
                pass

        return metrics

    def _prepare_similarity_matrix(self) -> np.ndarray:
        # Make a copy to avoid modifying the original
        matrix = self.similarity_matrix.copy()

        # Handle NaN values if present
        if np.isnan(matrix).any():
            matrix[np.isnan(matrix)] = np.nanmean(matrix)

        # Make sure the matrix is symmetric
        matrix = (matrix + matrix.T) / 2

        # Make it positive semi-definite by adding a small constant to diagonal
        min_val = np.min(matrix)
        if min_val < 0:
            matrix = matrix - min_val

        # Add small constant to diagonal for numerical stability
        np.fill_diagonal(matrix, np.diag(matrix) + 1e-8)

        return matrix
