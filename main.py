import os
import sys
import logging
import argparse
from pathlib import Path
import gc
import json
from typing import Dict, Optional, Tuple

from src.core.config import Config
from src.core.data_manager import DataManager
from src.analysis.engine import AnalysisEngine
from src.utils.io import IOManager
from src.visualization.visualizer import Visualizer

# TODO: Cleanup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("SequenceAligner")


# TODO: Cleanup
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SequenceAligner: Analyze and visualize sequence data efficiently."
    )

    parser.add_argument(
        "-i", "--input", type=str, help="Input file with sequences (CSV, FASTA, etc.)"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for analysis results",
    )

    parser.add_argument(
        "-c", "--config", type=str, help="Path to custom configuration file (YAML)"
    )

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Generate visualizations of results",
    )

    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )

    parser.add_argument(
        "--save-config", type=str, help="Save current configuration to specified file"
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Preserve temporary files after analysis",
    )

    return parser.parse_args()


# TODO: Cleanup
def setup_environment(args):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = Config.load_from_file(args.config)
    else:
        logger.info("Using default configuration")
        config = Config.get_default_config()
    if args.input:
        config.input_file = args.input
    if args.output:
        config.output_dir = args.output
    if args.save_config:
        logger.info(f"Saving current configuration to: {args.save_config}")
        config.save_to_file(args.save_config)
    return config


def run_analysis(config: Config) -> Tuple[Dict, AnalysisEngine]:
    logger.info("Initializing components")
    io_manager = IOManager(config)
    data_manager = DataManager(config)
    analysis_engine = AnalysisEngine(config, data_manager)
    if config.io.backup_results:
        io_manager.archive_previous_results()

    logger.info(f"Loading sequences from: {config.input_file}")
    sequences = data_manager.load_sequences(config.input_file)
    if not sequences or len(sequences) == 0:
        logger.error("No sequences loaded. Aborting.")
        return {}, None

    logger.info(f"Loaded {len(sequences)} sequences")
    logger.info("Running analysis pipeline")
    results = analysis_engine.run_analysis(sequences)
    logger.info(f"Saving analysis results to: {config.output_dir}")
    io_manager.save_results(results)
    large_data_path = os.path.join(config.output_dir, "large_data.h5")
    analysis_engine.save_large_data(large_data_path)
    gc.collect()
    return results, analysis_engine


# TODO: Cleanup
def create_visualizations(
    config: Config, results: Dict, analysis_engine: Optional[AnalysisEngine] = None
) -> None:
    logger.info("Creating visualizations")
    visualizer = Visualizer(config)
    visualization_data = results.copy()
    if analysis_engine and hasattr(analysis_engine, "get_large_data"):
        large_data = analysis_engine.get_large_data()
        if "feature_matrix" in large_data:
            visualization_data["feature_matrix"] = large_data["feature_matrix"]
        if "similarity_matrix" in large_data:
            visualization_data["similarity_matrix"] = large_data["similarity_matrix"]
        if "labels" in large_data:
            visualization_data["labels"] = large_data["labels"]
        if "cluster_labels" in large_data and "clustering" in results:
            visualization_data["clustering_results"] = {}
            for method, result in results["clustering"].items():
                if method in large_data["cluster_labels"]:
                    visualization_data["clustering_results"][method] = result.copy()
                    if (
                        "labels_info"
                        in visualization_data["clustering_results"][method]
                    ):
                        del visualization_data["clustering_results"][method][
                            "labels_info"
                        ]
                    visualization_data["clustering_results"][method]["labels"] = (
                        large_data["cluster_labels"][method]
                    )

    if (
        "similarity_analysis" in results
        and "feature_names" in results["similarity_analysis"]
    ):
        visualization_data["feature_names"] = results["similarity_analysis"][
            "feature_names"
        ]

    visualization_paths = visualizer.create_visualizations(visualization_data)
    logger.info(f"Created {len(visualization_paths)} visualizations")
    gc.collect()


# TODO: Cleanup
def main():
    try:
        args = parse_arguments()
        config = setup_environment(args)
        if not config.input_file:
            logger.error(
                "No input file specified. Use --input or configure input_file in config."
            )
            return 1

        results, analysis_engine = run_analysis(config)
        if not results:
            logger.error("Analysis failed or produced no results.")
            return 1

        if args.visualize:
            create_visualizations(config, results, analysis_engine)

        logger.info("Analysis complete!")
        return 0

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return 1

    finally:
        gc.collect()


if __name__ == "__main__":
    sys.exit(main())
