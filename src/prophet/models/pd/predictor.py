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
# from xgboost import XGBClassifier
import gc

logger = logging.getLogger(__name__)

class PDModel(_BaseModel):
    path = importlib.resources.files("prophet.models.pd").joinpath("config.yaml")
    if path.exists():
        DEFAULT_CONFIG_PATH = str(path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {path}")

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
        path = importlib.resources.files("prophet.models.pd").joinpath(model_path)
        return joblib.load(str(path))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        data['note'] = data['note'].with_row_index()

        feat = data['note'].select(['index', 'id', 'date'])
        logger.info(f'Generating features for n = {len(feat)}')

        # preliminary filter by just ids
        data['icd'] = data['icd'].filter(pl.col('id').is_in(feat['id'].unique()))
        data['med'] = data['med'].filter(pl.col('id').is_in(feat['id'].unique()))

        # begin preprocess
        logger.info(f"ICD preprocessing started at {datetime.now()}")
        icd_feat = self._preprocess_icd(data['icd'], feat)
        logger.info(f"ICD preprocessing finished at {datetime.now()}")

        logger.info(f"Med preprocessing started at {datetime.now()}")
        med_feat = self._preprocess_med(data['med'], feat)
        logger.info(f"Med preprocessing finished at {datetime.now()}")

        logger.info(f"Note preprocessing started at {datetime.now()}")
        note_feat = self._preprocess_note(data['note'], feat, show_progress)
        logger.info(f"Note preprocessing finished at {datetime.now()}")

        # join features
        feat = feat.join(
            icd_feat,
            on=['index', 'id', 'date'],
            how='left'
        ).join(
            med_feat,
            on=['index', 'id', 'date'],
            how='left'
        ).join(
            note_feat,
            on=['index', 'id', 'date'],
            how='left'
        ).select(['id', 'date'] + self.config['parameters']['final_cols'])

        logger.info(f"Preprocessing finished at {datetime.now()}")
        return feat

    def _preprocess_icd(self, icd_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        icd_names = self.config['parameters']['icd']
        dt_offset = self.config['parameters']['dt_offset']
        icd_df = icd_df.filter(pl.col('icd').str.contains('|'.join(icd_names)))

        icd_df = icd_df.with_columns(
            (pl.col('date').dt.offset_by('-' + dt_offset)).alias('date_lower'),
            (pl.col('date').dt.offset_by(dt_offset)).alias('date_upper'),
        )
        icd_feat = feat.join(
            feat.drop('index').join(
                icd_df,
                on='id',
                how='left'
            ).filter(
                (pl.col('date') >= pl.col('date_lower')) &
                (pl.col('date') <= pl.col('date_upper'))
            ).group_by(['id', 'date']).agg(
                pl.lit(1).alias('icd')
            ),
            on=['id', 'date'],
            how='left'
        ).fill_null(0)
        return icd_feat

    
    def _preprocess_med(self, med_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        med_names = self.config['parameters']['med']
        dt_offset = self.config['parameters']['dt_offset']
        med_df = med_df.with_columns(
            pl.col('med').str.to_lowercase()
        ).filter(
            pl.col('med').str.contains('|'.join(med_names))
        )
        med_df = med_df.with_columns(
            (pl.col('date').dt.offset_by('-' + dt_offset)).alias('date_lower'),
            (pl.col('date').dt.offset_by(dt_offset)).alias('date_upper'),
        ).drop('date')
        med_feat = feat.join(
            feat.join(
                med_df,
                on='id',
                how='left'
            ).filter(
                (pl.col('date') >= pl.col('date_lower')) &
                (pl.col('date') <= pl.col('date_upper'))
            ).group_by(['id', 'date']).agg(
                [pl.when(pl.col('med').str.contains(x)).then(1).otherwise(0).max().alias(x) for x in med_names]
            ),
            on=['id', 'date'],
            how='left'
        ).fill_null(0)
        return med_feat
    
    def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        note_df = note_df.with_columns(
            pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_lowercase()
        )
        kw_names = {x : set(x.split(' ')) for x in self.config['parameters']['kws']}

        ray.init()
        try:
            if show_progress:
                remote_tqdm = ray.remote(tqdm_ray.tqdm)
                bar = remote_tqdm.remote(total=len(note_df), desc='Processing notes')
            else:
                bar = None

            negation_triggers = {
                'preceding': ['no', 'deny', 'absence', 'not', 'negative', 'without', 'rule out', 
                            'unlikely', 'free of', 'never', 'unremarkable for'],
                'following': ['ruled out', 'excluded', 'negative', 'absent'],
                'pseudo': ['not only', 'no increase'] 
            }

            # Pre-stem the negation triggers
            stemmer = SnowballStemmer('english')
            stemmed_negation = {}
            stemmed_negation['preceding'] = [' '.join(stemmer.stem(word) for word in phrase.split()) 
                                for phrase in negation_triggers['preceding']]
            stemmed_negation['following'] = [' '.join(stemmer.stem(word) for word in phrase.split()) 
                                for phrase in negation_triggers['following']]
            stemmed_negation['pseudo'] = [' '.join(stemmer.stem(word) for word in phrase.split()) 
                            for phrase in negation_triggers['pseudo']]
            
            @ray.remote
            def process_note(text, stemmer_r, bar, stemmed_negation):
                feature_vector = dict.fromkeys(kw_names, 0)
                feature_vector.update({f'{x}_neg': 0 for x in kw_names})
                
                # Window size for negation scope
                window_size = 5  # words
                
                sentences = sent_tokenize(text)
                for s in sentences:
                    words = word_tokenize(s)
                    stemmed_words = [stemmer_r.stem(word) for word in words]
                    
                    # Find negation triggers with sliding window to catch multi-word phrases
                    neg_indices = []
                    for i in range(len(stemmed_words)):
                        # Check for single-word triggers
                        if stemmed_words[i] in stemmed_negation['preceding']:
                            neg_indices.append((i, min(i + window_size, len(stemmed_words))))
                        elif stemmed_words[i] in stemmed_negation['following']:
                            neg_indices.append((max(0, i - window_size), i))
                        
                        # Check for multi-word triggers
                        for j in range(1, min(4, len(stemmed_words) - i)):  # Check phrases up to 4 words
                            phrase = ' '.join(stemmed_words[i:i+j])
                            if phrase in stemmed_negation['preceding']:
                                neg_indices.append((i+j-1, min(i+j-1 + window_size, len(stemmed_words))))
                            elif phrase in stemmed_negation['following']:
                                neg_indices.append((max(0, i - window_size), i))
                            elif phrase in stemmed_negation['pseudo']:  # Remove negation if pseudo-negation
                                neg_indices = [idx for idx in neg_indices if not (idx[0] <= i and idx[1] >= i+j-1)]
                    
                    # Check for keywords
                    for bag, kw_set in kw_names.items():
                        # Find indices where keywords appear
                        kw_indices = []
                        for i, stemmed in enumerate(stemmed_words):
                            if stemmed in kw_set:
                                kw_indices.append(i)
                        
                        if kw_indices:
                            # Check if any keyword is within negation scope
                            is_negated = any(
                                any(neg_start <= kw_idx <= neg_end for neg_start, neg_end in neg_indices)
                                for kw_idx in kw_indices
                            )
                            
                            if is_negated:
                                feature_vector[f'{bag}_neg'] = 1
                            else:
                                feature_vector[bag] = 1
                                
                if bar:
                    bar.update.remote(1)
                return feature_vector
            
            note_feat = ray.get([process_note.remote(text, stemmer, bar, stemmed_negation) for text in note_df['note']])
            if bar:
                bar.close.remote()
        finally:
            ray.shutdown()
            gc.collect()

        note_feat = pl.DataFrame(note_feat)
        note_feat = note_feat.rename({col : f'{col}_' for col in note_feat.columns})
        note_feat = note_feat.hstack(note_df.select(['index', 'id', 'date']))
        note_feat = feat.join(
            note_feat,
            on=['index', 'id', 'date'],
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
    
    def evaluate(self, data: Dict[str, pl.DataFrame], ground_truth: pl.DataFrame):
        """
        Evaluate model performance against ground truth
        
        Args:
            data: Dictionary of DataFrames containing the test data
            ground_truth: DataFrame with actual outcomes (must have 'id', 'date', and 'actual' columns)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Run the model
        _, predictions = self.run(data)
        
        # Join predictions with ground truth
        evaluation_df = predictions.join(
            ground_truth,
            on=['id', 'date'],
            how='inner'
        )
        
        if len(evaluation_df) == 0:
            raise ValueError("No matching records found between predictions and ground truth")
        
        # Calculate metrics
        y_true = evaluation_df['actual'].to_numpy()
        y_pred = evaluation_df['prediction'].to_numpy()
        y_prob = evaluation_df['prob_YES'].to_numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'n_samples': len(evaluation_df)
        }
        
        # Add confusion matrix elements
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
        
        return metrics

