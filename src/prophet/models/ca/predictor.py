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
import os
from glob import glob
import shutil

logger = logging.getLogger(__name__)

class CAModel(_BaseModel):
    path = importlib.resources.files("prophet.models.ca").joinpath("config.yaml")
    if path.exists():
        DEFAULT_CONFIG_PATH = str(path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {path}")

    def __init__(self, config_path=None, external_ray=False):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        super().__init__(config_path)
        self.external_ray = external_ray
        
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
        path = importlib.resources.files("prophet.models.ca").joinpath(model_path)
        return joblib.load(str(path))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        data['note'] = data['note'].with_row_index()

        feat = data['note'].select(['index', 'id', 'date'])
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data. Please check your input data.")
        logger.info(f'Generating features for n = {len(feat)}')

        # preliminary filter by just ids
        data['icd'] = data['icd'].filter(pl.col('id').is_in(feat['id'].unique()))

        # begin preprocess
        logger.info(f"ICD preprocessing started at {datetime.now()}")
        icd_feat = self._preprocess_icd(data['icd'], feat)
        logger.info(f"ICD preprocessing finished at {datetime.now()}")

        logger.info(f"Note preprocessing started at {datetime.now()}")
        note_feat = self._preprocess_note(data['note'], feat, show_progress)
        logger.info(f"Note preprocessing finished at {datetime.now()}")

        # join features
        feat = feat.join(
            icd_feat,
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
        ).drop('date')
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

    def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        note_df = note_df.with_columns(
            pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_lowercase()
        )
        kw_names = {x : set(x.split(' ')) for x in self.config['parameters']['kws']}
        negate_words = set(['no', 'not', 'dont', 'absent', 'ho', 'pmh', 'negat', 'histori', 'unlik', 'without', 'lack', 'defer'])

        ray_was_initialized = ray.is_initialized()
        
        if not ray_was_initialized and not self.external_ray:
            ray.init()
            should_shutdown = True
        else:
            should_shutdown = False

        try:
            # Use fixed batch size of 1000 rows for consistent memory usage
            batch_size = 1000
            num_cpus = int(ray.available_resources().get("CPU", 1))
            total_notes = len(note_df)
            
            if show_progress:
                remote_tqdm = ray.remote(tqdm_ray.tqdm)
                bar = remote_tqdm.remote(total=len(note_df), desc='Processing notes')
            else:
                bar = None

            @ray.remote
            def process_note_batch(text_batch, stemmer_r, bar):
                """Process a batch of notes instead of individual notes"""
                batch_results = []
                
                for text in text_batch:
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
                    batch_results.append(feature_vector)
                    
                    # Update progress for each note in the batch
                    if bar:
                        bar.update.remote(1)
                
                return batch_results                  
            
            # Split notes into fixed-size batches of 1000 rows each
            notes_list = note_df['note'].to_list()
            note_batches = [notes_list[i:i + batch_size] for i in range(0, len(notes_list), batch_size)]
            
            logger.info(f"Processing {total_notes} notes in {len(note_batches)} batches of {batch_size} notes each using {num_cpus} CPUs")
            
            # Use ray.wait() to limit pending tasks and avoid memory pressure
            MAX_PENDING_TASKS = num_cpus * 2  # Keep 2x CPU cores worth of tasks pending
            batch_counter = 0
            result_refs = []
            note_feat = []
            os.makedirs("/tmp/ca_nax_processing", exist_ok=True)
            
            # Put stemmer in object store once to avoid repeated serialization
            stemmer_ref = ray.put(SnowballStemmer('english'))
            
            for i, batch in enumerate(note_batches):
                # Apply backpressure - wait for tasks to complete if we have too many pending
                if len(result_refs) >= MAX_PENDING_TASKS:
                    ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
                    # Process completed results immediately to free memory
                    completed_results = ray.get(ready_refs)
                    batch_df = pl.DataFrame([item for batch_result in completed_results for item in batch_result])
                    batch_df.write_parquet(f"/tmp/ca_nax_processing/batch_{batch_counter}.parquet")
                    batch_counter += 1
                    
                # Submit new task
                result_refs.append(process_note_batch.remote(batch, stemmer_ref, bar))
            
            # Process any remaining tasks
            if result_refs:
                remaining_results = ray.get(result_refs)
                batch_df = pl.DataFrame([item for batch_result in remaining_results for item in batch_result])
                batch_df.write_parquet(f"/tmp/ca_nax_processing/batch_{batch_counter}.parquet")
            
            if bar:
                bar.close.remote()
                
        except KeyboardInterrupt:
            # Handle interruption gracefully
            raise
        finally:
            # Only shutdown Ray if we initialized it AND we're not using external Ray
            if should_shutdown:
                ray.shutdown()
            gc.collect()
            
        note_feat = pl.read_parquet('/tmp/ca_nax_processing/batch_*.parquet')            
        note_feat = pl.DataFrame(note_feat)
        note_feat = note_feat.rename({col : f'{col}_' for col in note_feat.columns})
        note_feat = note_feat.hstack(note_df.select(['index', 'id', 'date']))
        note_feat = feat.join(
            note_feat,
            on=['index', 'id', 'date'],
            how='left',
            validate='1:1'
        )
        shutil.rmtree('/tmp/ca_nax_processing', ignore_errors=True)
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

