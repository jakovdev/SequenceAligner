from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Tuple, Union
import yaml
from pathlib import Path
import os

# TODO: Check deprecated fields


class MatrixConfig(BaseModel):

    name: str = Field("blosum62", description="Scoring matrix name")
    gap_open: int = Field(11, description="Gap opening penalty")
    gap_extend: int = Field(1, description="Gap extension penalty")

    @validator("name")
    def validate_matrix(cls, v):
        valid_matrices = [
            "blosum45",
            "blosum50",
            "blosum62",
            "blosum80",
            "pam30",
            "pam70",
            "pam250",
        ]
        if v not in valid_matrices:
            raise ValueError(f"Matrix {v} not supported. Use one of {valid_matrices}")
        return v


class ClusteringConfig(BaseModel):

    n_clusters: int = Field(2, description="Number of clusters to detect")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    min_samples: int = Field(5, description="Minimum samples for DBSCAN")
    eps: float = Field(0.5, description="Maximum distance between DBSCAN samples")
    method: str = Field("kmeans", description="Clustering method to use")


class VisualizationConfig(BaseModel):
    # TODO: Add visualization settings
    pass


class PerformanceConfig(BaseModel):

    n_workers: int = Field(0, description="Number of worker processes (0=auto)")
    # chunk_size: int = Field(64, description="Chunk size for parallel processing")
    memory_limit: str = Field("4GB", description="Memory limit for processing")

    @validator("memory_limit")
    def validate_memory(cls, v):
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

        if isinstance(v, str):
            if v[-2:] in units:
                try:
                    value = float(v[:-2]) * units[v[-2:]]
                    return v
                except ValueError:
                    pass

        raise ValueError("Memory limit must be a string like '4GB'")

    @property
    def memory_bytes(self) -> int:
        units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        value = float(self.memory_limit[:-2]) * units[self.memory_limit[-2:]]
        return int(value)


class IOConfig(BaseModel):

    temp_dir: Optional[str] = Field(
        None, description="Temporary directory for processing"
    )
    backup_results: bool = Field(True, description="Backup previous results")
    sequence_format: str = Field(
        "auto", description="Input sequence format (auto, fasta, csv, tsv)"
    )


class AnalysisConfig(BaseModel):

    calculate_properties: bool = Field(
        True, description="Calculate sequence properties"
    )
    predict_features: bool = Field(True, description="Predict sequence features")
    significance_threshold: float = Field(
        0.05, description="Statistical significance threshold"
    )
    minimum_length: int = Field(5, description="Minimum sequence length")
    maximum_length: int = Field(10000, description="Maximum sequence length")


class Config(BaseModel):

    # User-friendly naming
    project_name: str = Field("Sequence Analysis", description="Project name")
    description: str = Field(
        "Protein sequence analysis pipeline", description="Project description"
    )

    # Input/Output settings
    input_file: Optional[str] = Field(None, description="Input file path")
    output_dir: str = Field("results", description="Output directory")

    # Analysis components
    matrix: MatrixConfig = Field(default_factory=MatrixConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    # Common substitution matrices and their recommended gap penalties
    matrix_presets: Dict[str, Dict] = Field(
        default_factory=lambda: {
            "blosum45": {"name": "blosum45", "gap_open": 14, "gap_extend": 2},
            "blosum50": {"name": "blosum50", "gap_open": 13, "gap_extend": 2},
            "blosum62": {"name": "blosum62", "gap_open": 11, "gap_extend": 1},
            "blosum80": {"name": "blosum80", "gap_open": 10, "gap_extend": 1},
            "pam30": {"name": "pam30", "gap_open": 9, "gap_extend": 1},
            "pam70": {"name": "pam70", "gap_open": 10, "gap_extend": 1},
            "pam250": {"name": "pam250", "gap_open": 13, "gap_extend": 2},
        }
    )

    def apply_matrix_preset(self, matrix_name: str) -> None:
        if matrix_name not in self.matrix_presets:
            raise ValueError(f"Unknown matrix preset: {matrix_name}")

        preset = self.matrix_presets[matrix_name]
        self.matrix = MatrixConfig(**preset)

    @classmethod
    def load_from_file(cls, file_path: str) -> "Config":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(path, "r") as file:
            config_data = yaml.safe_load(file)

        return cls(**config_data)

    def save_to_file(self, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as file:
            yaml.dump(
                self.dict(exclude={"matrix_presets"}), file, default_flow_style=False
            )

    @classmethod
    def get_default_config(cls) -> "Config":
        # Try to load from default location
        default_paths = [
            os.path.join(os.getcwd(), "config", "default_config.yaml"),
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "config",
                "default_config.yaml",
            ),
        ]

        for path in default_paths:
            if os.path.exists(path):
                return cls.load_from_file(path)

        # Return built-in defaults
        return cls()
