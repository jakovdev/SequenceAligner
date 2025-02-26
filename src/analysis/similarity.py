from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import numpy as np
import parasail
import logging

from ..core.config import Config
from ..core.sequence import Sequence, SequenceCollection
from ..utils.parallel import process_matrix_chunks

logger = logging.getLogger(__name__)


class SimilarityCalculator:

    def __init__(self, config: Config):
        self.config = config
        self._initialize_matrices()
        self.results = {}

    def _initialize_matrices(self):
        # TODO: Review
        self.matrix_name = self.config.matrix.name
        self.gap_open = self.config.matrix.gap_open
        self.gap_extend = self.config.matrix.gap_extend
        try:
            self.parasail_matrix = parasail.Matrix(self.matrix_name)
        except Exception as e:
            logger.error(f"Failed to initialize scoring matrix: {e}")
            raise

        self.hydrophobicity_scale = Sequence.HYDROPHOBICITY

    def compute_feature_matrix(self, sequences: SequenceCollection) -> np.ndarray:
        n_sequences = len(sequences)
        logger.info(f"Computing feature matrix for {n_sequences} sequences")

        feature_names = [
            "length",
            "molecular_weight",
            "charge",
            "hydrophobicity",
            "percent_hydrophobic",
            "percent_charged",
            "percent_polar",
            "percent_aromatic",
            "percent_positive",
            "percent_negative",
        ]
        n_features = len(feature_names)

        self.results["feature_names"] = feature_names

        # TODO: Review
        feature_matrix = np.zeros((n_sequences, n_features), dtype=np.float32)
        for i, seq in enumerate(sequences.get_sequences()):
            feature_matrix[i] = seq.to_feature_vector()

        self._collect_feature_stats(feature_matrix)

        return feature_matrix

    def _collect_feature_stats(self, feature_matrix: np.ndarray) -> None:
        self.results["feature_stats"] = {
            "mean": np.mean(feature_matrix, axis=0).tolist(),
            "std": np.std(feature_matrix, axis=0).tolist(),
            "min": np.min(feature_matrix, axis=0).tolist(),
            "max": np.max(feature_matrix, axis=0).tolist(),
        }

    def compute_similarity_matrix(self, sequences: SequenceCollection) -> np.ndarray:
        logger.info("Computing sequence similarity matrix")

        sequence_strings = sequences.get_raw_sequences()
        n_sequences = len(sequence_strings)
        matrix_size_mb = (n_sequences * n_sequences * 4) / (1024 * 1024)
        logger.info(
            f"Similarity matrix size: {n_sequences}x{n_sequences} ({matrix_size_mb:.2f} MB)"
        )

        matrix_params = {
            "matrix_name": self.matrix_name,
            "gap_open": self.gap_open,
            "gap_extend": self.gap_extend,
        }

        matrix = process_matrix_chunks(
            func=self._process_similarity_chunk,
            data=sequence_strings,
            # chunk_size=self.config.performance.chunk_size,
            n_workers=self.config.performance.n_workers,
            symmetric=True,
            use_tqdm=True,
            desc="Computing similarity",
            matrix_params=matrix_params,
        )

        self._collect_similarity_stats(matrix)
        return matrix

    @staticmethod
    def _process_similarity_chunk(
        seqs_i: List[str],
        seqs_j: List[str],
        i_start: int,
        i_end: int,
        j_start: int,
        j_end: int,
        matrix_params: Dict,
    ) -> np.ndarray:
        matrix_name = matrix_params["matrix_name"]
        gap_open = matrix_params["gap_open"]
        gap_extend = matrix_params["gap_extend"]
        parasail_matrix = parasail.Matrix(matrix_name)

        # TODO: Review
        chunk_matrix = np.zeros((len(seqs_i), len(seqs_j)), dtype=np.float32)

        for i, seq_i in enumerate(seqs_i):
            for j, seq_j in enumerate(seqs_j):
                # Skip diagonal elements when i_start == j_start
                if i_start == j_start and i == j:
                    chunk_matrix[i, j] = 1.0
                    continue

                chunk_matrix[i, j] = (
                    SimilarityCalculator._calculate_sequence_similarity(
                        seq_i, seq_j, gap_open, gap_extend, parasail_matrix
                    )
                )

        return chunk_matrix

    @staticmethod
    def _calculate_sequence_similarity(
        seq1: str, seq2: str, gap_open: int, gap_extend: int, parasail_matrix
    ) -> float:
        # TODO: Review
        if len(seq1) < 5 or len(seq2) < 5:
            matches = sum(
                a == b
                for a, b in zip(
                    seq1, seq2[: len(seq1)] if len(seq1) < len(seq2) else seq2
                )
            )
            return matches / max(len(seq1), len(seq2))

        try:
            # TODO: Review
            result = parasail.nw_stats(
                seq1, seq2, gap_open, gap_extend, parasail_matrix
            )

            alignment_length = result.length + result.similar + result.matches

            similarity = (
                result.matches / alignment_length if alignment_length > 0 else 0.0
            )

            return max(0.0, min(1.0, similarity))

        except Exception:
            return 0.0

    def _collect_similarity_stats(self, similarity_matrix: np.ndarray) -> None:
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(similarity_matrix, k=1)
        similarities = similarity_matrix[triu_indices]

        self.results["similarity_stats"] = {
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "quartiles": [
                float(np.percentile(similarities, 25)),
                float(np.percentile(similarities, 50)),
                float(np.percentile(similarities, 75)),
            ],
            "histogram": {
                "bins": np.linspace(0, 1, 21).tolist(),
                "counts": np.histogram(similarities, bins=20, range=(0, 1))[0].tolist(),
            },
        }

    def get_results(self) -> Dict:
        return self.results
