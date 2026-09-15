#!/usr/bin/env python3
"""
Get predictions for all disease models in Prophet using outcome prediction data.

This script loads outcome prediction data from parquet files and generates
predictions using all available models in Prophet. Results are saved to
a specified output directory with one subdirectory per model.

Usage:
    python scripts/get_all_predictions.py [--output OUTPUT_DIR] [--show-progress] [--return-features]

    --output OUTPUT_DIR    Output directory for predictions (default: results/predictions)
    --show-progress       Show progress bars during prediction
    --return-features     Also save feature data alongside predictions
    --models MODELS       Comma-separated list of models to run (default: all)
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pickle as pkl

import polars as pl
from prophet import Prophet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('get_predictions.log')
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = '/home/niels/Desktop/outcome_nlp'
DEFAULT_OUTPUT_DIR = 'results/predictions'

# Mapping of outcome file suffixes to internal names
DATA_FILE_MAPPING = {
    'notes': 'note',
    'icd': 'icd',
    'med': 'med',
    'cpt': 'cpt',
    'demo': 'demo'
}


def load_outcome_data(data_dir: str) -> dict[str, pl.DataFrame]:
    """
    Load outcome prediction data from parquet files.

    Args:
        data_dir: Directory containing outcome_prediction_*.parquet files

    Returns:
        Dictionary mapping data types to DataFrames
    """
    logger.info(f"Loading outcome prediction data from {data_dir}")
    data = {}

    for suffix, name in DATA_FILE_MAPPING.items():
        file_path = os.path.join(data_dir, f'outcome_prediction_{suffix}.parquet')

        if not os.path.exists(file_path):
            continue

        try:
            logger.info(f"Loading {name} data from {file_path}")
            df = pl.read_parquet(file_path)
            logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            data[name] = df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue

    if not data:
        raise ValueError(f"No outcome prediction files found in {data_dir}")

    logger.info(f"Successfully loaded {len(data)} data sources: {', '.join(data.keys())}")
    return data


def get_models_to_run(available_models: list[str], selected_models: str | None = None) -> list[str]:
    """
    Determine which models to run.

    Args:
        available_models: List of all available models
        selected_models: Comma-separated string of models to run, or None for all

    Returns:
        List of model names to run
    """
    if selected_models is None:
        return available_models

    requested = [m.strip() for m in selected_models.split(',')]
    invalid = [m for m in requested if m not in available_models]

    if invalid:
        logger.warning(f"Invalid model names: {', '.join(invalid)}")
        logger.warning(f"Available models: {', '.join(available_models)}")

    valid = [m for m in requested if m in available_models]
    if not valid:
        raise ValueError(f"No valid models selected from: {requested}")

    return valid


def get_compatible_models(prophet: Prophet, data: dict[str, pl.DataFrame],
                         models_to_run: list[str]) -> list[str]:
    """
    Filter models based on data compatibility.

    Args:
        prophet: Prophet instance
        data: Dictionary of available data
        models_to_run: List of requested models

    Returns:
        List of models that are compatible with the data
    """
    compatible = prophet.find_compatible_models(data)
    logger.info(f"Compatible models with provided data: {compatible}")

    # Filter to only requested models
    to_run = [m for m in models_to_run if m in compatible]

    if not to_run:
        logger.error(f"No requested models are compatible with the data")
        logger.error(f"Requested: {models_to_run}")
        logger.error(f"Compatible: {compatible}")
        raise ValueError("No compatible models found")

    skipped = [m for m in models_to_run if m not in compatible]
    if skipped:
        logger.warning(f"Skipping incompatible models: {', '.join(skipped)}")

    return to_run


def run_predictions(prophet: Prophet, data: dict[str, pl.DataFrame],
                   models: list[str], show_progress: bool = False,
                   return_features: bool = False) -> dict:
    """
    Run predictions for specified models.

    Args:
        prophet: Prophet instance
        data: Dictionary of DataFrames
        models: List of model names to run
        show_progress: Whether to show progress bars
        return_features: Whether to return features alongside predictions

    Returns:
        Dictionary mapping model names to results
    """
    logger.info(f"Running predictions for {len(models)} models: {', '.join(models)}")
    notes = data['note'].clone()

    results = {}
    for i, model_name in enumerate(models, 1):
        if model_name == 'congestive_heart_failure' or model_name == 'epilepsy':
            continue
        logger.info(f"[{i}/{len(models)}] Running prediction for '{model_name}'")

        try:
            try:
                if 'index' in notes.columns:
                    data['note'] = notes.drop('index').clone()
                else:
                    data['note'] = notes.clone()
                data['note'] = data['note'].with_row_index().filter(
                    pl.col('index').is_in(
                        pl.read_parquet(f'/home/niels/Desktop/outcome_nlp/{model_name}/notes.parquet')['index']
                    )
                ).drop('index')
            except Exception as e:
                logger.warning(f"  Could not filter notes for '{model_name}': {e}")
                data['note'] = notes.clone()

            model = prophet.model_creator.get_model(model_name)

            result = model.preprocess(
                data=data,
                show_progress=show_progress,
            )

            result.write_parquet(f'/home/niels/Desktop/outcome_nlp/{model_name}/predictions.parquet')
            
            results[model_name] = result

            logger.info(f"  Successfully generated predictions for '{model_name}'")
        except Exception as e:
            logger.error(f"  Error running prediction for '{model_name}': {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions for all Prophet disease models'
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for predictions (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--data-dir',
        default=DATA_DIR,
        help=f'Directory containing outcome_prediction_*.parquet files (default: {DATA_DIR})'
    )
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Show progress bars during prediction'
    )
    parser.add_argument(
        '--return-features',
        action='store_true',
        help='Also save feature data alongside predictions (some models may not support this)'
    )
    parser.add_argument(
        '--models',
        default=None,
        help='Comma-separated list of models to run (default: all compatible models)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"Starting Prophet prediction pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Show progress: {args.show_progress}")
    logger.info(f"Return features: {args.return_features}")

    try:
        # Load data
        data = load_outcome_data(args.data_dir)
        logger.info(f"Loaded {len(data)} data sources with total {sum(len(df) for df in data.values())} rows")

        # Initialize Prophet
        logger.info("Initializing Prophet")
        prophet = Prophet()

        # Get available models
        available_models = prophet.get_available_models()
        logger.info(f"Available models: {len(available_models)}")

        # Determine which models to run
        if args.models:
            models_to_run = get_models_to_run(available_models, args.models)
        else:
            models_to_run = available_models

        logger.info(f"Requested models to run: {len(models_to_run)}")

        # Filter to compatible models
        # compatible_models = get_compatible_models(prophet, data, models_to_run)
        # logger.info(f"Running predictions for {len(compatible_models)} compatible models")

        # Run predictions
        results = run_predictions(
            prophet,
            data,
            models_to_run,
            show_progress=args.show_progress,
            return_features=args.return_features
        )

        if not results:
            logger.error("No predictions were successfully generated")
            return 1

        logger.info(f"Successfully generated predictions for {len(results)} models")

        # Save results
        logger.info(f"Saving results to {args.output}")
        prophet.save_predictions(results, args.output)

        logger.info("=" * 80)
        logger.info(f"Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
