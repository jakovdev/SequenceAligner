import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from ..core.config import Config
from ..core.sequence import Sequence, SequenceCollection

logger = logging.getLogger(__name__)


class StatisticsAnalyzer:

    def __init__(self, config: Config):
        self.config = config
        self.results = {}

    def run_statistics(
        self,
        sequences: SequenceCollection,
        similarity_matrix: np.ndarray,
        feature_matrix: np.ndarray,
        clustering_results: Dict,
        labels: Optional[np.ndarray] = None,
    ) -> Dict:
        logger.info("Running statistical analysis")

        self.similarity_matrix = similarity_matrix
        self.feature_matrix = feature_matrix
        self.results = {}

        self._analyze_sequence_properties()
        self._analyze_similarity_distribution()

        # TODO: Review
        if labels is not None and np.any(labels != None):
            self._analyze_labeled_groups(labels)

            if clustering_results:
                self._evaluate_clustering(clustering_results, labels)

        # TODO: Review
        if sequences:
            self._analyze_aa_composition(sequences)

        return self.results

    def _analyze_sequence_properties(self) -> None:
        # TODO: Review
        n_features = self.feature_matrix.shape[1]

        property_stats = []
        for i in range(n_features):
            feature_values = self.feature_matrix[:, i]
            property_stats.append(self._calculate_descriptive_stats(feature_values))

        self.results["property_stats"] = property_stats

    def _analyze_similarity_distribution(self) -> None:
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(self.similarity_matrix, k=1)
        similarities = self.similarity_matrix[triu_indices]

        stats = self._calculate_descriptive_stats(similarities)

        # TODO: Review
        stats["histogram"] = {
            "bins": [
                float(b)
                for b in np.linspace(np.min(similarities), np.max(similarities), 20)
            ],
            "counts": [int(c) for c in np.histogram(similarities, bins=20)[0]],
        }

        self.results["similarity_distribution"] = stats

    def _calculate_descriptive_stats(self, values: np.ndarray) -> Dict[str, Any]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "quartiles": [float(q) for q in np.percentile(values, [25, 50, 75])],
        }

    def _analyze_labeled_groups(self, labels: np.ndarray) -> None:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        # TODO: Review
        if n_labels <= 1:
            return

        n_samples = len(labels)

        group_stats = {
            "n_groups": int(n_labels),
            "counts": {},
            "within_group_similarities": {},
            "between_group_similarities": {},
        }

        # TODO: Review
        group_indices = {
            int(label): np.where(labels == label)[0] for label in unique_labels
        }

        for label, indices in group_indices.items():
            group_stats["counts"][str(label)] = len(indices)

        for label, indices in group_indices.items():
            if len(indices) > 1:
                similarities = self._get_pairwise_similarities(indices, indices)
                if len(similarities) > 0:
                    group_stats["within_group_similarities"][str(label)] = {
                        **self._calculate_descriptive_stats(similarities),
                        "count": len(similarities),
                    }

        for i, label_i in enumerate(unique_labels):
            indices_i = group_indices[int(label_i)]
            for j in range(i + 1, len(unique_labels)):
                label_j = unique_labels[j]
                indices_j = group_indices[int(label_j)]

                similarities = self._get_pairwise_similarities(indices_i, indices_j)
                if len(similarities) > 0:
                    key = f"{int(label_i)}-{int(label_j)}"
                    group_stats["between_group_similarities"][key] = {
                        **self._calculate_descriptive_stats(similarities),
                        "count": len(similarities),
                    }

        # TODO: Review
        if n_labels == 2 and all(
            len(indices) > 1 for indices in group_indices.values()
        ):
            self._perform_binary_group_tests(group_indices, unique_labels, group_stats)

        self.results["group_analysis"] = group_stats

    def _get_pairwise_similarities(
        self, indices1: np.ndarray, indices2: np.ndarray
    ) -> np.ndarray:
        similarities = []

        # TODO: Review
        # Skip self-comparisons when the same indices are provided
        same_sets = np.array_equal(indices1, indices2)

        for i, idx1 in enumerate(indices1):
            start = i + 1 if same_sets else 0
            for j in range(start, len(indices2)):
                idx2 = indices2[j]
                if idx1 != idx2:
                    similarities.append(self.similarity_matrix[idx1, idx2])

        return np.array(similarities)

    def _perform_binary_group_tests(
        self,
        group_indices: Dict[int, np.ndarray],
        unique_labels: np.ndarray,
        group_stats: Dict,
    ) -> None:
        try:
            label0, label1 = int(unique_labels[0]), int(unique_labels[1])
            indices0 = group_indices[label0]
            indices1 = group_indices[label1]

            within_sim_0 = self._get_pairwise_similarities(indices0, indices0)
            within_sim_1 = self._get_pairwise_similarities(indices1, indices1)
            between_sim = self._get_pairwise_similarities(indices0, indices1)

            # TODO: Review
            if len(within_sim_0) > 1 and len(within_sim_1) > 1 and len(between_sim) > 1:
                t_within, p_within = stats.ttest_ind(
                    within_sim_0, within_sim_1, equal_var=False
                )
                t_0_between, p_0_between = stats.ttest_ind(
                    within_sim_0, between_sim, equal_var=False
                )
                t_1_between, p_1_between = stats.ttest_ind(
                    within_sim_1, between_sim, equal_var=False
                )

                significance_threshold = self.config.analysis.significance_threshold
                group_stats["statistical_tests"] = {
                    "within_groups_ttest": {
                        "t_statistic": float(t_within),
                        "p_value": float(p_within),
                        "significant": bool(p_within < significance_threshold),
                    },
                    f"{label0}_vs_between_ttest": {
                        "t_statistic": float(t_0_between),
                        "p_value": float(p_0_between),
                        "significant": bool(p_0_between < significance_threshold),
                    },
                    f"{label1}_vs_between_ttest": {
                        "t_statistic": float(t_1_between),
                        "p_value": float(p_1_between),
                        "significant": bool(p_1_between < significance_threshold),
                    },
                }
        except Exception as e:
            logger.warning(f"Error in statistical tests: {e}")

    def _evaluate_clustering(
        self, clustering_results: Dict, true_labels: np.ndarray
    ) -> None:
        clustering_eval = {}

        # TODO: Review
        for method, result in clustering_results.items():
            if "labels" not in result:
                continue

            pred_labels = result["labels"]

            # TODO: Review
            if len(pred_labels) != len(true_labels):
                logger.warning(
                    f"Label length mismatch for {method}: {len(pred_labels)} vs {len(true_labels)}"
                )
                continue

            metrics = self._calculate_clustering_metrics(true_labels, pred_labels)
            metrics["class_distribution"] = self._get_class_distribution(
                true_labels, pred_labels
            )

            clustering_eval[method] = metrics

        self.results["clustering_evaluation"] = clustering_eval

    def _calculate_clustering_metrics(
        self, true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> Dict:
        metrics = {}

        # TODO: Review
        try:
            ari = adjusted_rand_score(true_labels, pred_labels)
            metrics["adjusted_rand_index"] = float(ari)
        except Exception as e:
            logger.warning(f"Error calculating ARI: {e}")
            metrics["adjusted_rand_index"] = None

        try:
            ami = adjusted_mutual_info_score(true_labels, pred_labels)
            metrics["adjusted_mutual_info"] = float(ami)
        except Exception as e:
            logger.warning(f"Error calculating AMI: {e}")
            metrics["adjusted_mutual_info"] = None

        return metrics

    def _get_class_distribution(
        self, true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> Dict:
        unique_clusters = np.unique(pred_labels)
        class_distribution = {}

        for cluster in unique_clusters:
            cluster_mask = pred_labels == cluster
            classes, counts = np.unique(true_labels[cluster_mask], return_counts=True)

            class_distribution[int(cluster)] = {
                str(int(cls)): int(count) for cls, count in zip(classes, counts)
            }

        return class_distribution

    def _analyze_aa_composition(self, sequences: SequenceCollection) -> None:
        # TODO: Review
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")

        total_aa = {aa: 0 for aa in aa_list}
        total_residues = 0

        # TODO: Review
        for seq in sequences.get_sequences():
            aa_counts = seq.aa_counts
            for aa, count in aa_counts.items():
                if aa in total_aa:
                    total_aa[aa] += count
                    total_residues += count

        aa_frequencies = {
            aa: count / total_residues if total_residues > 0 else 0
            for aa, count in total_aa.items()
        }

        property_counts = {}
        for prop, aa_set in Sequence.PROPERTY_GROUPS.items():
            property_counts[prop] = sum(total_aa[aa] for aa in aa_set if aa in total_aa)

        property_percentages = {
            prop: (count / total_residues * 100) if total_residues > 0 else 0
            for prop, count in property_counts.items()
        }

        self.results["aa_composition"] = {
            "aa_counts": total_aa,
            "aa_frequencies": aa_frequencies,
            "property_counts": property_counts,
            "property_percentages": property_percentages,
            "total_residues": total_residues,
        }
