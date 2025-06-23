from ...base._basemodel import _BaseModel
import logging
from typing import Dict, Any, Union, Tuple
import polars as pl
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import SnowballStemmer
import nltk
import ray
import joblib
from ray.experimental import tqdm_ray
from datetime import datetime
import importlib.resources
import time
import gc

logger = logging.getLogger(__name__)

class MIModelICD(_BaseModel):
    path = importlib.resources.files("prophet.models.mi").joinpath("config.yaml")
    if path.exists():
        DEFAULT_CONFIG_PATH = str(path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    def __init__(self, config_path=None, external_ray=False):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        super().__init__(config_path)

    def load_model(self, model_path: str):
        # no model, rule-based
        return

    def run(self, data: Dict[str, pl.DataFrame], show_progress=False, return_features=False, force_casting=False) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Run the model on the provided data.
        
        Args:
            data: Dictionary of DataFrames containing the required data
            force_casting: Boolean flag to force casting of columns
        
        Returns:
            feat: DataFrame with features
            pred: DataFrame with predictions
        """
        pred = self.preprocess(data, show_progress, force_casting)
        if return_features:
            return None, pred
        else:
            return pred
        
    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        
        icd_feat = data['icd']
        if len(icd_feat) == 0:
            raise ValueError("No ICD data found in the provided data. Please check your input data.")
        icd_feat = icd_feat.filter(
            pl.col('icd').str.contains('|'.join(self.config['parameters']['icd']))
        ).with_columns(
            pl.lit(1).alias('prediction')
        )

        return icd_feat
