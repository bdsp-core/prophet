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
import psutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class BrainTumorModel(_BaseModel):
    path = importlib.resources.files("prophet.models.brain_tumor").joinpath("config.yaml")
    if path.exists():
        DEFAULT_CONFIG_PATH = str(path)
    else:
        raise FileNotFoundError(f"Configuration file not found: {path}")

    def __init__(self, config_path=None, external_ray=False):
        if config_path is None:
            config_path = self.DEFAULT_CONFIG_PATH
        super().__init__(config_path)
        self.external_ray = external_ray  # Flag to indicate if Ray is managed externally
        
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
        path = importlib.resources.files("prophet.models.brain_tumor").joinpath(model_path)
        return joblib.load(str(path))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)

        note_df = data['note'].with_row_index()
        feat = note_df.select(['index', 'id'])
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data. Please check your input data.")
        logger.info(f'Generating features for n = {len(feat)}')

        logger.info(f"Note preprocessing started at {datetime.now()}")
        note_feat = self._preprocess_note(note_df, feat, show_progress)
        logger.info(f"Note preprocessing finished at {datetime.now()}")

        feat = feat.join(
            note_feat,
            on=['index', 'id'],
            how='left'
        ).select(['id', 'index'] + self.config['parameters']['final_cols'])

        logger.info(f"Preprocessing finished at {datetime.now()}")
        return feat
    
    def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        note_df = note_df.with_columns(
            pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_lowercase()
        )
        
        # Prepare keyword configurations
        kw_names = {x: set(x.split(' ')) for x in self.config['parameters']['kws']}
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
            temp_dir = Path(tempfile.mkdtemp(prefix="brain_tumor_processing_"))
            
            if show_progress:
                # Use tqdm without ray.remote wrapper for simplicity
                from tqdm import tqdm
                progress_bar = tqdm(total=total_notes, desc='Processing brain tumor notes')
            else:
                progress_bar = None

            @ray.remote(num_cpus=1)  # Explicitly set resource requirements
            class BrainTumorProcessor:
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
                            note_idx = item['index']

                            feature_vector = dict.fromkeys(self.kw_names, 0)
                            feature_vector.update({f'{x}_neg': 0 for x in self.kw_names})
                            feature_vector['index'] = note_idx
                            
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
            processors = [BrainTumorProcessor.remote() for _ in range(max_concurrent_actors)]
            
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

            if len(note_feat) != len(note_df):
                raise RuntimeError(
                    f"note feature extraction produced {len(note_feat)} rows for "
                    f"{len(note_df)} notes; a Ray batch failed to write its output"
                )

            # Rename feature columns (keep index as the join key)
            feature_cols = [col for col in note_feat.columns if col != 'index']
            note_feat = note_feat.rename({col: f'{col}_' for col in feature_cols})

            # Join back id, then join with feat
            note_feat = note_feat.join(
                note_df.select(['index', 'id']),
                on='index',
                how='left'
            )
            note_feat = feat.join(
                note_feat,
                on=['index', 'id'],
                how='left',
                validate='1:1'
            ).fill_null(0)
            
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
    

