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

logger = logging.getLogger(__name__)

class BrainTumorModel(_BaseModel):
    with importlib.resources.files("prophet.models.brain_tumor").joinpath("config.yaml") as path:
        DEFAULT_CONFIG_PATH = str(path)

    def __init__(self, config_path = None):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        super().__init__(config_path)

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

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
        feat = self.preprocess(data, show_progress, force_casting)
        pred = self.predict(feat)
        if return_features:
            return feat, pred
        else:
            return pred



    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        # TODO: fix for user loading their own model?
        with importlib.resources.files("prophet.models.brain_tumor").joinpath(model_path) as path:
            return joblib.load(str(path))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        data['note'] = data['note'].with_row_index()

        feat = data['note'].select(['index', 'id'])
        logger.info(f'Generating features for n = {len(feat)}')

        logger.info(f"Note preprocessing started at {datetime.now()}")
        note_feat = self._preprocess_note(data['note'], feat, show_progress)
        logger.info(f"Note preprocessing finished at {datetime.now()}")

        # join features
        feat = feat.join(
            note_feat,
            on=['index', 'id'],
            how='left'
        ).select(['id'] + self.config['parameters']['final_cols'])

        logger.info(f"Preprocessing finished at {datetime.now()}")
        return feat
    
    def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        note_df = note_df.with_columns(
            pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_lowercase()
        )
        kw_names = {x : set(x.split(' ')) for x in self.config['parameters']['kws']}
        negate_words = set(self.config['parameters']['neg_kws'])


        with ray.init() as ray_context:
            if show_progress:
                remote_tqdm = ray.remote(tqdm_ray.tqdm)
                bar = remote_tqdm.remote(total=len(note_df), desc='Processing notes')
            else:
                bar = None

            @ray.remote
            def process_note(text, stemmer_r, bar):
                feature_vector = dict.fromkeys(kw_names, 0)
                feature_vector.update({f'{x}_neg' : 0 for x in kw_names})
                sentences = sent_tokenize(text)
                for s in sentences:
                    stem_words = set(stemmer_r.stem(word) for word in word_tokenize(s))
                    for bag, words in kw_names.items():
                        if words.issubset(stem_words):
                            if negate_words.intersection(stem_words):
                                feature_vector[f'{bag}_neg'] = 1
                            else:
                                feature_vector[bag] = 1
                if bar:
                    bar.update.remote(1)
                return feature_vector                    
            
            note_feat = ray.get([process_note.remote(text, ray.put(SnowballStemmer('english')), bar) for text in note_df['note']])
            if bar:
                bar.close.remote()
            
        note_feat = pl.DataFrame(note_feat)
        note_feat = note_feat.rename({col : f'{col}_' for col in note_feat.columns})
        note_feat = note_feat.hstack(note_df.select(['index', 'id']))
        note_feat = feat.join(
            note_feat,
            on=['index', 'id'],
            how='left',
            validate='1:1'
        )
        return note_feat
    
    def predict(self, feat : pl.DataFrame) -> pl.DataFrame:
        """
        Run the model on the provided features.
        
        Args:
            feat: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")
        pred = self.model.predict_proba(feat.select(self.config['parameters']['final_cols']))
        pred = pl.DataFrame(pred, schema=['prob_NO', 'prob_YES'], orient='row')
        logger.info(f'Using suggested threshold of {self.config["parameters"]["threshold"]}')
        pred = feat.select(pl.all().exclude(self.config['parameters']['final_cols'])).hstack(
            pred.with_columns(
                pl.when(pl.col('prob_YES') > self.config['parameters']['threshold'])
                .then(1)
                .otherwise(0)
                .alias('prediction')
            )
        )
        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return pred
    
    # TODO: model does not need date, may need to use index + pandas instead
    # def evaluate(self, data: Dict[str, pl.DataFrame], ground_truth: pl.DataFrame):
    #     """
    #     Evaluate model performance against ground truth
        
    #     Args:
    #         data: Dictionary of DataFrames containing the test data
    #         ground_truth: DataFrame with actual outcomes (must have 'id', 'date', and 'actual' columns)
            
    #     Returns:
    #         Dictionary containing evaluation metrics
    #     """
    #     import numpy as np
    #     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
    #     # Run the model
    #     _, predictions = self.run(data)
        
    #     # Join predictions with ground truth
    #     evaluation_df = predictions.join(
    #         ground_truth,
    #         on=['id', 'date'],
    #         how='inner'
    #     )
        
    #     if len(evaluation_df) == 0:
    #         raise ValueError("No matching records found between predictions and ground truth")
        
    #     # Calculate metrics
    #     y_true = evaluation_df['actual'].to_numpy()
    #     y_pred = evaluation_df['prediction'].to_numpy()
    #     y_prob = evaluation_df['prob_YES'].to_numpy()
        
    #     metrics = {
    #         'accuracy': accuracy_score(y_true, y_pred),
    #         'precision': precision_score(y_true, y_pred),
    #         'recall': recall_score(y_true, y_pred),
    #         'f1': f1_score(y_true, y_pred),
    #         'roc_auc': roc_auc_score(y_true, y_prob),
    #         'n_samples': len(evaluation_df)
    #     }
        
    #     # Add confusion matrix elements
    #     tn = np.sum((y_true == 0) & (y_pred == 0))
    #     fp = np.sum((y_true == 0) & (y_pred == 1))
    #     fn = np.sum((y_true == 1) & (y_pred == 0))
    #     tp = np.sum((y_true == 1) & (y_pred == 1))
        
    #     metrics.update({
    #         'true_negatives': int(tn),
    #         'false_positives': int(fp),
    #         'false_negatives': int(fn),
    #         'true_positives': int(tp)
    #     })
        
    #     return metrics

