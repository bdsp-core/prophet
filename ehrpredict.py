import sys
import os
import polars as pl
from typing import Dict, Optional, List, Union, Literal, Tuple
import logging
import time
from datetime import datetime
# TODO: enforce typing via pydantic?
from models._modelcreator import _ModelCreator
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EHRPredict:
    def __init__(self) -> None:
        self.model_creator = _ModelCreator()

    def get_data_format(self, phenotype: str) -> Dict:
        """
        Get the expected data format for a phenotype model
        
        Parameters:
        -----------
        phenotype : str
            Name of the phenotype model
            
        Returns:
        --------
        Dict mapping dataframe names to required columns
        """
        return _ModelCreator.get_data_format(phenotype)
    
    def get_credits(self, phenotype: str) -> Dict:
        """
        Get credits and attribution information for a phenotype model
        
        Parameters:
        -----------
        phenotype : str
            Name of the phenotype model
            
        Returns:
        --------
        Dict with model credits/attribution information
        """
        return _ModelCreator.get_credits(phenotype)
    
    def find_compatible_models(self, data: Dict[str, pl.DataFrame]) -> List[str]:
        """
        Find models compatible with the provided data
        
        Parameters:
        -----------
        data : Dict[str, pl.DataFrame]
            Dictionary of dataframes
            
        Returns:
        --------
        List of compatible model names
        """
        return _ModelCreator.get_compatible_models(data)

    def _prepare_data(self, phenotype: str, data: Optional[Dict[str, Union[pl.DataFrame]]] = None, 
                       **kwargs) -> Dict[str, Union[pl.DataFrame]]:
        """
        Prepare data for prediction
        
        Parameters:
        -----------
        phenotype : str
            Name of phenotype model
        data : Dict[str, Union[pl.DataFrame, pl.LazyFrame]], optional
            Dictionary of dataframes
        **kwargs : 
            Additional dataframes passed as keyword arguments
            
        Returns:
        --------
        Dict[str, Union[pl.DataFrame, pl.LazyFrame]]
            Prepared data dictionary
        """
        # Get required dataframes for this model
        req_dfs = _ModelCreator.get_data_format(phenotype).keys()
        
        # Combine data from dict and kwargs
        if data is None:
            data = {}
        all_dfs = {**data, **kwargs}
        
        # Check if all required dataframes are present
        for df_name in req_dfs:
            if df_name not in all_dfs:
                raise ValueError(f"Required dataframe '{df_name}' is missing from input data")
        
        return all_dfs
        
    def predict(self,
                phenotype: Union[str, List[str]],
                data: Optional[Dict[str, Union[pl.DataFrame, pl.LazyFrame]]] = None,
                show_progress_bar: bool = False,
                return_features: bool = False,
                **kwargs):
        """Make predictions for one or more phenotypes with optional feature return.
        
        Args:
            phenotype: Single phenotype string or list of phenotype strings
            data: Input data dictionary
            show_progress_bar: Whether to show tqdm progress bar, if applicable
            return_features: Whether to return features alongside predictions
            **kwargs: Additional arguments for data preparation
            
        Returns:
            If single phenotype:
                - Predictions DataFrame/LazyFrame if return_features=False
                - Tuple of (features, predictions) if return_features=True
            If multiple phenotypes:
                - Dict of phenotype -> predictions if return_features=False
                - Dict of phenotype -> (features, predictions) if return_features=True
        """
        # Handle single phenotype case by converting to list
        single_phenotype = isinstance(phenotype, str)
        phenotypes = [phenotype] if single_phenotype else phenotype
        
        logger.info(f"Starting prediction for {len(phenotypes)} phenotype(s): {', '.join(phenotypes)}")
        start_time_all = time.time()
        
        results = {}
        
        for pheno in phenotypes:
            pheno_start = time.time()
            logger.info(f"Processing phenotype '{pheno}' started at {datetime.now().strftime('%H:%M:%S')}")
            
            # Prepare data for this phenotype
            logger.info(f"Preparing data for '{pheno}'")
            prepared_data = self._prepare_data(pheno, data, **kwargs)
            
            # Get model for this phenotype
            logger.info(f"Loading model for '{pheno}'")
            model = self.model_creator.get_model(pheno)
            
            # Run prediction
            logger.info(f"Running prediction for '{pheno}'")
            result = model.run(prepared_data, show_progress_bar, return_features)
            
            # Store result for this phenotype
            results[pheno] = result
            
            pheno_end = time.time()
            logger.info(f"Phenotype '{pheno}' completed in {pheno_end - pheno_start:.2f}s")
        
        total_time = time.time() - start_time_all
        logger.info(f"All phenotypes completed in {total_time:.2f}s")
        
        # Return results based on input type
        if single_phenotype:
            return results[phenotype]  # Return just the single result
        else:
            return results  # Return dict of all results

    def save_predictions(self,
                        results,
                        output_path: str):
        """Save prediction results with standardized structure.
        
        Args:
            results: Prediction results from predict() - either a single result or a dict of results
            output_path: Directory path where results will be stored
        """
        logger.info(f"Starting to save prediction results to {output_path}")
        
        # Ensure directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Handle different result formats
        if isinstance(results, dict):
            # Multiple phenotypes
            for phenotype, result in results.items():
                self._save_single_result(phenotype, result, output_path)
        else:
            # Single phenotype result - use 'output' as phenotype name
            self._save_single_result('output', results, output_path)
        
        logger.info(f"All results saved successfully to {output_path}")

    def _save_single_result(self,
                        phenotype: str,
                        result : Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]],
                        output_path: str):
        """Save a single prediction result to the specified directory.
        
        Args:
            phenotype: The phenotype name used for the subdirectory
            result: Either predictions or (features, predictions) tuple
            output_path: Root directory path where results will be stored
        """
        
        # Create phenotype subdirectory
        phenotype_dir = os.path.join(output_path, phenotype)
        os.makedirs(phenotype_dir, exist_ok=True)
        
        # Check if result is a tuple (features, predictions)
        if isinstance(result, tuple) and len(result) == 2:
            features, predictions = result
            has_features = True
        else:
            predictions = result
            has_features = False
        
        # Save predictions
        pred_path = os.path.join(phenotype_dir, "predictions.parquet")
        logger.info(f"Saving predictions to {pred_path}")
        predictions.write_parquet(pred_path)
        
        # Save features if available
        if has_features:
            feat_path = os.path.join(phenotype_dir, "features.parquet")
            logger.info(f"Saving features to {feat_path}")
            features.write_parquet(feat_path)