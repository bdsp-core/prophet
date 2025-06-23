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
from glob import glob
import os
import shutil
import psutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class TBIModel(_BaseModel):
    path = importlib.resources.files("prophet.models.tbi").joinpath("config.yaml")
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
        path = importlib.resources.files("prophet.models.tbi").joinpath(model_path)
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
            feat.join(
                icd_df,
                on='id',
                how='left'
            ).filter(
                (pl.col('date') >= pl.col('date_lower')) &
                (pl.col('date') <= pl.col('date_upper'))
            ).group_by(['id', 'date']).agg(
                [pl.when(pl.col('icd').str.contains(x)).then(1).otherwise(0).max().alias(x) for x in icd_names]
            ),
            on=['id', 'date'],
            how='left'
        ).fill_null(0)
        return icd_feat

def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
    # Text preprocessing - TBI model uses standard preprocessing
    note_df = note_df.with_columns(
        pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
        .str.to_lowercase()
    )
    
    # Prepare keyword configurations - TBI model has strike words for head trauma detection
    kw_names = {x: set(x.split(' ')) for x in self.config['parameters']['kws']}
    negate_words = set(self.config['parameters']['neg_kws'])
    strike_words = set(self.config['parameters']['strike_kws'])  # TBI model tracks strike/impact words

    # Ray initialization with memory management
    ray_was_initialized = ray.is_initialized()
    
    if not ray_was_initialized and not self.external_ray:
        # Initialize Ray with memory limits to prevent OOM
        ray.init(
            object_store_memory=int(0.2 * psutil.virtual_memory().total),  # 20% of total memory
            _memory=int(0.3 * psutil.virtual_memory().total),  # 30% for worker processes
            _redis_max_memory=int(0.05 * psutil.virtual_memory().total)
        )
        should_shutdown = True
    else:
        should_shutdown = False

    try:
        # Use smaller batch size for better memory management
        batch_size = 500  # Reduced from 1000
        num_cpus = int(ray.available_resources().get("CPU", 1))
        total_notes = len(note_df)
        
        # Create temporary directory with better path handling
        temp_dir = Path(tempfile.mkdtemp(prefix="tbi_processing_"))
        
        if show_progress:
            # Use tqdm without ray.remote wrapper for simplicity
            from tqdm import tqdm
            progress_bar = tqdm(total=total_notes, desc='Processing TBI notes')
        else:
            progress_bar = None

        @ray.remote(num_cpus=1)  # Explicitly set resource requirements
        class TBIProcessor:
            def __init__(self):
                self.stemmer = SnowballStemmer('english')
                self.kw_names = kw_names
                self.negate_words = negate_words
                self.strike_words = strike_words
            
            def process_batch(self, text_batch, batch_id):
                """Process a batch of notes with explicit memory management"""
                batch_results = []
                
                try:
                    for text in text_batch:
                        # TBI model creates standard pos/neg features PLUS custom strike features
                        feature_vector = dict.fromkeys(self.kw_names, 0)
                        feature_vector.update({f'{x}_neg': 0 for x in self.kw_names})
                        feature_vector.update({'custom_hit_strike': 0, 'custom_hit_strike_neg': 0})  # Custom TBI features
                        
                        sentences = sent_tokenize(text)
                        for s in sentences:
                            stem_words = set(self.stemmer.stem(word) for word in word_tokenize(s))
                            
                            # TBI-specific logic: Check for strike words + "head"
                            if stem_words.intersection(self.strike_words) and 'head' in stem_words:
                                if self.negate_words.intersection(stem_words):
                                    feature_vector['custom_hit_strike_neg'] = 1
                                else:
                                    feature_vector['custom_hit_strike'] = 1
                            
                            # Standard keyword processing
                            for bag, words in self.kw_names.items():
                                if words.issubset(stem_words):
                                    if self.negate_words.intersection(stem_words):
                                        feature_vector[f'{bag}_neg'] = 1
                                    else:
                                        feature_vector[bag] = 1
                        
                        batch_results.append(feature_vector)
                    
                    # Convert to DataFrame and save immediately
                    batch_df = pl.DataFrame(batch_results)
                    output_path = temp_dir / f"batch_{batch_id}.parquet"
                    batch_df.write_parquet(output_path)
                    
                    # Force garbage collection
                    del batch_results, batch_df
                    gc.collect()
                    
                    return len(text_batch), str(output_path)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_id}: {e}")
                    raise

        # Create processor actors with your preferred concurrency
        max_concurrent_actors = max(1, num_cpus - 2)  # Use your preferred num_cpus - 2
        processors = [TBIProcessor.remote() for _ in range(max_concurrent_actors)]
        
        # Split notes into batches
        notes_list = note_df['note'].to_list()
        note_batches = [notes_list[i:i + batch_size] for i in range(0, len(notes_list), batch_size)]
        
        logger.info(f"Processing {total_notes} notes in {len(note_batches)} batches using {max_concurrent_actors} actors")
        
        # Process batches with controlled concurrency
        futures = []
        batch_counter = 0
        completed_files = []
        
        # Submit initial batches
        for i, batch in enumerate(note_batches):
            processor = processors[i % len(processors)]
            future = processor.process_batch.remote(batch, batch_counter)
            futures.append(future)
            batch_counter += 1
            
            # Control memory by limiting pending tasks
            if len(futures) >= max_concurrent_actors * 2:
                # Wait for at least one to complete
                ready, futures = ray.wait(futures, num_returns=1, timeout=None)
                
                # Process completed results
                for ready_ref in ready:
                    try:
                        processed_count, file_path = ray.get(ready_ref)
                        completed_files.append(file_path)
                        
                        if progress_bar:
                            progress_bar.update(processed_count)
                            
                    except Exception as e:
                        logger.error(f"Error getting result: {e}")
                        continue
        
        # Wait for remaining tasks with better handling
        while futures:
            ready, futures = ray.wait(futures, num_returns=len(futures), timeout=60)
            
            for ready_ref in ready:
                try:
                    processed_count, file_path = ray.get(ready_ref, timeout=30)
                    completed_files.append(file_path)
                    
                    if progress_bar:
                        progress_bar.update(processed_count)
                        
                except ray.exceptions.GetTimeoutError:
                    logger.warning("Task timed out during final processing")
                    continue
                except Exception as e:
                    logger.error(f"Error in final processing: {e}")
                    continue
        
        if progress_bar:
            progress_bar.close()
            
        # Clean up actors
        for processor in processors:
            ray.kill(processor)
        
        # Force garbage collection before reading results
        gc.collect()
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Only shutdown Ray if we initialized it AND we're not using external Ray
        if should_shutdown:
            ray.shutdown()
        gc.collect()
    
    # Read and combine results
    try:
        parquet_files = list(temp_dir.glob("batch_*.parquet"))
        if not parquet_files:
            raise ValueError("No batch files were created")
            
        note_feat = pl.read_parquet(parquet_files)
        
        # Rename columns and join with original data - TBI model uses index+id+date
        note_feat = note_feat.rename({col: f'{col}_' for col in note_feat.columns})
        note_feat = note_feat.hstack(note_df.select(['index', 'id', 'date']))  # TBI model includes date
        note_feat = feat.join(
            note_feat,
            on=['index', 'id', 'date'],  # TBI model: 3 columns like many others
            how='left',
            validate='1:1'
        ).fill_null(0)  # TBI model fills null values with 0
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
    
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

