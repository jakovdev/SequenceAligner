import os
import sys
import logging
import argparse
from pathlib import Path
import gc

from src.core.config import Config
from src.core.data_manager import DataManager
from src.analysis.engine import AnalysisEngine
from src.utils.io import IOManager
from src.visualization.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("SequenceAligner")


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

    return parser.parse_args()


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


def run_analysis(config: Config):
    logger.info("Initializing analysis components")

    io_manager = IOManager(config)
    if config.io.backup_results:
        io_manager.archive_previous_results()

    data_manager = DataManager(config)
    analysis_engine = AnalysisEngine(config, data_manager)

    logger.info(f"Loading sequences from: {config.input_file}")
    sequences = data_manager.load_sequences(config.input_file)

    if not sequences or len(sequences) == 0:
        logger.error("No sequences loaded. Aborting.")
        return None

    logger.info(f"Loaded {len(sequences)} sequences")

    logger.info("Running analysis pipeline")
    results = analysis_engine.run_analysis(sequences)

    logger.info(f"Analysis completed. Results saved to: {config.output_dir}")
    analysis_engine.save_results()

    gc.collect()
    return results


def create_visualizations(config: Config, results):
    logger.info("Creating visualizations")
    visualizer = Visualizer(config)
    visualization_paths = visualizer.create_visualizations(results)
    logger.info(
        f"Created {len(visualization_paths)} visualizations in {config.output_dir}/visualizations"
    )
    gc.collect()


def main():
    try:
        args = parse_arguments()
        config = setup_environment(args)

        if not config.input_file:
            logger.error(
                "No input file specified. Use --input or configure input_file in config."
            )
            return 1

        results = run_analysis(config)
        if not results:
            logger.error("Analysis failed or produced no results.")
            return 1

        if args.visualize:
            create_visualizations(config, results)

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
