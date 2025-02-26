import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings

# TODO: Review deprecated usage


class Sequence:

    # TODO: Review
    VALID_RESIDUES: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")

    # TODO: Review
    PROPERTY_GROUPS = {
        "hydrophobic": set("AILMFWV"),
        "polar": set("NCQSTY"),
        "charged": set("DEKRH"),
        "positive": set("RKH"),
        "negative": set("DE"),
        "aromatic": set("FWY"),
        "small": set("AGST"),
    }

    # TODO: Review
    HYDROPHOBICITY = {  # Kyte-Doolittle
        "A": 1.8,
        "R": -4.5,
        "N": -3.5,
        "D": -3.5,
        "C": 2.5,
        "Q": -3.5,
        "E": -3.5,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "L": 3.8,
        "K": -3.9,
        "M": 1.9,
        "F": 2.8,
        "P": -1.6,
        "S": -0.8,
        "T": -0.7,
        "W": -0.9,
        "Y": -1.3,
        "V": 4.2,
    }

    __slots__ = ("_sequence", "_id", "_label", "_properties", "_aa_counts")

    def __init__(
        self, sequence: str, seq_id: Optional[str] = None, label: Optional[int] = None
    ):
        # TODO: Review
        if not all(aa in self.VALID_RESIDUES for aa in sequence):
            invalid_chars = set(sequence) - self.VALID_RESIDUES
            if invalid_chars:
                raise ValueError(f"Invalid amino acids in sequence: {invalid_chars}")

        self._sequence = sequence
        self._id = seq_id
        self._label = label
        self._properties = {}
        self._aa_counts = None

    @property
    def sequence(self) -> str:
        return self._sequence

    @property
    def id(self) -> Optional[str]:
        return self._id

    @property
    def label(self) -> Optional[int]:
        return self._label

    @label.setter
    def label(self, value: int) -> None:
        self._label = value

    @property
    def length(self) -> int:
        return len(self._sequence)

    @property
    def aa_counts(self) -> Dict[str, int]:
        if self._aa_counts is None:
            self._aa_counts = {aa: 0 for aa in self.VALID_RESIDUES}
            for aa in self._sequence:
                self._aa_counts[aa] += 1

        return self._aa_counts

    def get_molecular_weight(self) -> float:
        if "molecular_weight" not in self._properties:
            try:
                analysis = ProteinAnalysis(self._sequence)
                self._properties["molecular_weight"] = analysis.molecular_weight()
            except Exception:
                # Fallback for sequences with non-standard amino acids
                warnings.warn(
                    f"Could not calculate exact molecular weight for sequence {self._id}"
                )
                # TODO: Review
                # Rough estimate: average AA weight ~ 110 Da
                self._properties["molecular_weight"] = len(self._sequence) * 110.0

        return self._properties["molecular_weight"]

    def get_charge(self, pH: float = 7.0) -> float:
        if f"charge_{pH}" not in self._properties:
            try:
                analysis = ProteinAnalysis(self._sequence)
                self._properties[f"charge_{pH}"] = analysis.charge_at_pH(pH)
            except Exception:
                # TODO: Review
                # Fallback calculation
                pos_count = sum(
                    self.aa_counts.get(aa, 0) for aa in self.PROPERTY_GROUPS["positive"]
                )
                neg_count = sum(
                    self.aa_counts.get(aa, 0) for aa in self.PROPERTY_GROUPS["negative"]
                )
                self._properties[f"charge_{pH}"] = pos_count - neg_count

        return self._properties[f"charge_{pH}"]

    def get_hydrophobicity(self) -> float:
        if "hydrophobicity" not in self._properties:
            total = sum(self.HYDROPHOBICITY.get(aa, 0) for aa in self._sequence)
            self._properties["hydrophobicity"] = (
                total / self.length if self.length > 0 else 0
            )

        return self._properties["hydrophobicity"]

    def get_property_percentage(self, property_name: str) -> float:
        if property_name not in self.PROPERTY_GROUPS:
            raise ValueError(f"Unknown property: {property_name}")

        if f"percent_{property_name}" not in self._properties:
            count = sum(
                self.aa_counts.get(aa, 0) for aa in self.PROPERTY_GROUPS[property_name]
            )
            self._properties[f"percent_{property_name}"] = (
                (count / self.length) * 100 if self.length > 0 else 0
            )

        return self._properties[f"percent_{property_name}"]

    def get_all_properties(self) -> Dict[str, float]:
        properties = {
            "length": self.length,
            "molecular_weight": self.get_molecular_weight(),
            "charge": self.get_charge(),
            "hydrophobicity": self.get_hydrophobicity(),
        }

        for prop in self.PROPERTY_GROUPS:
            properties[f"percent_{prop}"] = self.get_property_percentage(prop)

        return properties

    def to_feature_vector(self) -> np.ndarray:
        props = self.get_all_properties()
        # TODO: Review
        return np.array(
            [
                props["length"],
                props["molecular_weight"],
                props["charge"],
                props["hydrophobicity"],
                props["percent_hydrophobic"],
                props["percent_charged"],
                props["percent_polar"],
                props["percent_aromatic"],
                props["percent_positive"],
                props["percent_negative"],
            ],
            dtype=np.float32,
        )

    def __str__(self) -> str:
        return self._sequence

    def __len__(self) -> int:
        return self.length


# TODO: Review
class SequenceCollection:

    def __init__(self):
        self.sequences: List[Sequence] = []
        self._feature_matrix: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

    def add_sequence(self, sequence: Sequence) -> None:
        self.sequences.append(sequence)
        # Invalidate cached data
        self._feature_matrix = None

    def add_sequences(self, sequences: List[Sequence]) -> None:
        self.sequences.extend(sequences)
        self._feature_matrix = None

    def get_sequences(self) -> List[Sequence]:
        return self.sequences

    def get_raw_sequences(self) -> List[str]:
        return [seq.sequence for seq in self.sequences]

    def get_labels(self) -> np.ndarray:
        labels = [seq.label for seq in self.sequences]
        return np.array(labels, dtype=np.int32)

    def get_feature_matrix(
        self, force_recalculate: bool = False
    ) -> Tuple[np.ndarray, List[str]]:
        if self._feature_matrix is None or force_recalculate:
            if not self.sequences:
                return np.empty((0, 0)), []

            # TODO: Review
            sample_props = self.sequences[0].get_all_properties()
            self._feature_names = list(sample_props.keys())

            n_samples = len(self.sequences)
            n_features = len(self._feature_names)
            self._feature_matrix = np.zeros((n_samples, n_features), dtype=np.float32)

            for i, seq in enumerate(self.sequences):
                props = seq.get_all_properties()
                for j, feature in enumerate(self._feature_names):
                    self._feature_matrix[i, j] = props.get(feature, 0.0)

        return self._feature_matrix, self._feature_names

    def get_subset(self, indices: List[int]) -> "SequenceCollection":
        subset = SequenceCollection()
        subset.add_sequences([self.sequences[i] for i in indices])
        return subset

    def split_by_label(self) -> Dict[int, "SequenceCollection"]:
        result = {}
        for seq in self.sequences:
            label = seq.label
            if label not in result:
                result[label] = SequenceCollection()
            result[label].add_sequence(seq)
        return result

    def __len__(self) -> int:
        return len(self.sequences)
