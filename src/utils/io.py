import os
import json
import zipfile
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IOManager:

    def __init__(self, config):
        self.config = config
        self._setup_directories()

    def _setup_directories(self) -> None:
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)

    def archive_previous_results(self) -> None:
        if not self.config.io.backup_results:
            return

        output_path = Path(self.config.output_dir)
        if not output_path.exists():
            return

        files = [f for f in output_path.glob("*") if f.is_file()]
        vis_path = output_path / "visualizations"
        vis_files = []
        if vis_path.exists():
            vis_files = [f for f in vis_path.glob("*.html") if f.is_file()]

        if not files and not vis_files:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"results_archive_{timestamp}.zip"
        archive_path = output_path / archive_name

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                if file.suffix not in [".zip", ".py", ".md"]:
                    zipf.write(file, file.name)
                    file.unlink()

            for file in vis_files:
                rel_path = os.path.join("visualizations", file.name)
                zipf.write(file, rel_path)
                file.unlink()

            if vis_path.exists() and not any(vis_path.iterdir()):
                vis_path.rmdir()

        logger.info(f"Archived previous results to {archive_path}")

    def save_results(self, results: Dict) -> None:
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        serializable_results = self._convert_to_serializable(results)
        serializable_results["timestamp"] = datetime.now().isoformat()
        serializable_results["config"] = self.config.model_dump(
            exclude={"matrix_presets"}
        )

        results_path = os.path.join(output_dir, "analysis_results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    def _convert_to_serializable(self, obj: Any) -> Any:
        if isinstance(
            obj,
            (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        elif isinstance(obj, dict):
            return {
                key: self._convert_to_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_serializable(x) for x in obj]
        elif hasattr(obj, "__dict__"):
            # For objects with dict representation
            return {
                key: self._convert_to_serializable(value)
                for key, value in obj.__dict__.items()
                if not key.startswith("_")
            }
        return obj

    # TODO: Review usage
    def save_matrix(self, matrix: np.ndarray, name: str) -> str:
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        matrix_path = os.path.join(output_dir, f"{name}.npz")
        np.savez_compressed(matrix_path, matrix=matrix)
        return matrix_path
