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

class NarcolepsyModel(_BaseModel):
    path = importlib.resources.files("prophet.models.narcolepsy").joinpath("config.yaml")
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



    def load_model(self, model_path):
        """Load narcolepsy models.

        model_path may be:
          - a dict  {task_key: relative_path, ...}  → returns dict of loaded models
          - a str   relative_path                   → returns single loaded model (legacy)
        """
        pkg = importlib.resources.files("prophet.models.narcolepsy")
        if isinstance(model_path, dict):
            models = {}
            for task_key, rel_path in model_path.items():
                logger.info(f"Loading {task_key} model from {rel_path}")
                models[task_key] = joblib.load(str(pkg.joinpath(rel_path)))
            return models
        logger.info(f"Loading model from {model_path}")
        return joblib.load(str(pkg.joinpath(model_path)))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        
        # Create a copy of the note DataFrame with index to avoid modifying the original
        note_df_with_index = data['note'].with_row_index()
        data['note'] = note_df_with_index

        feat = data['note'].select(['index', 'id', 'date'])
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data. Please check your input data.")
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
        )
        if 'final_cols' in self.config['parameters'] and self.config['parameters']['final_cols']:
            feat = feat.select(['id', 'date'] + self.config['parameters']['final_cols'])

        logger.info(f"Preprocessing finished at {datetime.now()}")
        return feat

    def _preprocess_icd(self, icd_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        icd_names = self.config['parameters']['icd']
        if icd_names and isinstance(icd_names[0], list):
            all_icd_names = []
            for sublist in icd_names:
                all_icd_names.extend(sublist)
                grouped = True
        else:
            all_icd_names = icd_names
            grouped = False
        dt_offset = self.config['parameters']['dt_offset']
        icd_df = icd_df.filter(pl.col('icd').str.contains('|'.join(all_icd_names)))

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
                [pl.when(pl.col('icd').str.contains(x)).then(1).otherwise(0).max().alias(x) for x in all_icd_names]
            ),
            on=['id', 'date'],
            how='left'
        ).fill_null(0)
        if grouped:
            for sublist in icd_names:
                if all(x in icd_feat.columns for x in sublist):
                    icd_feat = icd_feat.with_columns(
                        pl.max_horizontal(sublist).alias('|'.join(sublist))
                    ).drop(sublist)
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
        # Text preprocessing
        note_df = note_df.with_columns(
            # pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            # .str.replace_all(r'\s+', ' ')
            # .str.strip_chars()
            # .str.to_lowercase()

            pl.col('note').str.replace_all(r'\bw/\b', ' with ')
            .str.replace_all(r'\bw/o\b', ' without ')
            .str.replace_all(r'[:;()\[\]\\\/]', ' ')
            .str.replace_all(r"\.", ". ")
            .str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
        )
        
        # Prepare keyword configurations
        kw_names = {x: set(x.split(' ')) for x in self.config['parameters']['kws']}
        # kw_names.update({x: set(x.split(' ')) for x in self.config['parameters']['med']}) #FIXME test with meds
        negate_words = set(self.config['parameters']['neg_kws'])

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
            temp_dir = Path(tempfile.mkdtemp(prefix="narcolepsy_processing_"))
            
            if show_progress:
                # Use tqdm without ray.remote wrapper for simplicity
                from tqdm import tqdm
                progress_bar = tqdm(total=total_notes, desc='Processing narcolepsy notes')
            else:
                progress_bar = None

            @ray.remote(num_cpus=1)  # Explicitly set resource requirements
            class NarcolepsyProcessor:
                def __init__(self):
                    self.stemmer = SnowballStemmer('english')
                    self.kw_names = kw_names
                    self.negate_words = negate_words
                
                def process_batch(self, batch_data, batch_id):
                    """Process a batch of notes with explicit memory management"""
                    batch_results = []
                    
                    try:
                        for item in batch_data:
                            text = item['note']
                            index = item['index']
                            
                            feature_vector = dict.fromkeys(self.kw_names, 0)
                            feature_vector.update({f'{x}_neg': 0 for x in self.kw_names})
                            feature_vector['index'] = index  # Include the index
                            
                            sentences = sent_tokenize(text)
                            for s in sentences:
                                stem_words = set(self.stemmer.stem(word) for word in word_tokenize(s))
                                
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
                        
                        return len(batch_data), str(output_path)
                        
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_id}: {e}")
                        raise

            # Create processor actors with limited concurrency
            max_concurrent_actors = max(1, num_cpus - 2)
            processors = [NarcolepsyProcessor.remote() for _ in range(max_concurrent_actors)]
            
            # Split notes into batches with indices
            notes_with_indices = note_df.select(['index', 'note']).to_dicts()
            note_batches = [notes_with_indices[i:i + batch_size] for i in range(0, len(notes_with_indices), batch_size)]
            
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

            # Apply join columns logic
            if 'combine_cols' in self.config['parameters']:
                for cols in self.config['parameters']['combine_cols']:
                    if all(x in note_feat.columns for x in cols):
                        note_feat = note_feat.with_columns(
                            pl.max_horizontal(cols).alias(cols[0])
                        ).drop(cols[1:])
                    cols = [x+'_neg' for x in cols]
                    if all(x in note_feat.columns for x in cols):
                        note_feat = note_feat.with_columns(
                            pl.max_horizontal(cols).alias(cols[0])
                        ).drop(cols[1:])
            
            # Rename feature columns (but not the index column) and join with original data using index
            feature_cols = [col for col in note_feat.columns if col != 'index']
            rename_dict = {col: f'{col}_' for col in feature_cols}
            note_feat = note_feat.rename(rename_dict).cast({'index':pl.UInt32})
            
            # Ensure index column datatypes match before join
            note_df_subset = note_df.select(['index', 'id', 'date'])
            
            # Join with original note data using index, then add id and date
            note_feat = note_feat.join(
                note_df_subset,
                on='index',
                how='left'
            )
            # Finally join with the feat DataFrame
            note_feat = feat.join(
                note_feat,
                on=['index', 'id', 'date'],
                how='left',
                validate='1:1'
            ).fill_null(0)
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return note_feat

    def predict(self, feat: pl.DataFrame) -> pl.DataFrame:
        """
        Run the model(s) on the provided features.

        When model_path is a dict (the standard multi-task setup), each model's
        feature names are read from model.feature_names_in_ so each task uses its
        own chi2-selected feature subset.  Output columns are named
        prob_<task_key> and pred_<task_key> (e.g. prob_nt1_vs_not, pred_nt1_vs_not).

        Falls back to the legacy single-model behaviour when model_path is a string.
        """
        import numpy as np

        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")
        threshold = self.config['parameters']['threshold']
        logger.info(f'Using suggested threshold of {threshold}')

        models = self.model

        if isinstance(models, dict):
            # Multi-task: one binary model per task
            all_feat_cols = self.config['parameters']['final_cols']
            result = feat.select(pl.all().exclude(all_feat_cols))

            for task_key, model in models.items():
                feature_cols = list(model.feature_names_in_)
                X = feat.select(feature_cols).to_numpy()
                proba = model.predict_proba(X)[:, 1]
                preds = (proba >= threshold).astype(int)
                result = result.with_columns([
                    pl.Series(f'prob_{task_key}', proba),
                    pl.Series(f'pred_{task_key}', preds),
                ])
        else:
            # Legacy single-model path
            pred_arr = models.predict_proba(feat.select(self.config['parameters']['final_cols']))
            pred_df  = pl.DataFrame(pred_arr, schema=['prob_NO', 'prob_YES'], orient='row')
            result = feat.select(pl.all().exclude(self.config['parameters']['final_cols'])).hstack(
                pred_df.with_columns(
                    pl.when(pl.col('prob_YES') > threshold)
                    .then(1)
                    .otherwise(0)
                    .alias('prediction')
                )
            )

        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return result
    
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

