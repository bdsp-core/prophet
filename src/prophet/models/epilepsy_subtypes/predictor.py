from ...base._basemodel import _BaseModel
import logging
from typing import Dict, Union, Tuple
import polars as pl
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
import nltk
import ray
import re
import json
import joblib
import time
import gc
import os
import shutil
import tempfile
import psutil
from pathlib import Path
from datetime import datetime
import importlib.resources

logger = logging.getLogger(__name__)


class EpilepsySubtypesModel(_BaseModel):
    path = importlib.resources.files("prophet.models.epilepsy_subtypes").joinpath("config.yaml")
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

        # Load syndrome-specific data (keywords, ICDs, medication mapping)
        syndrome_data_path = importlib.resources.files(
            "prophet.models.epilepsy_subtypes"
        ).joinpath("syndrome_data.json")
        with open(str(syndrome_data_path), 'r') as f:
            self.syndrome_data = json.load(f)

        # Build union of all keywords and ICDs for feature extraction
        self._all_keywords = set()
        self._all_icds = set()
        for s_data in self.syndrome_data['syndromes'].values():
            self._all_keywords.update(s_data['keywords'])
            self._all_icds.update(s_data['icds'])
        self._all_keywords = sorted(self._all_keywords)
        self._all_icds = sorted(self._all_icds)

        # Medication config
        self._med_config = self.syndrome_data['medications']
        self._canonical_meds = sorted(self._med_config['canonical_names'])

        # Syndromes to predict
        self._syndromes = self.config['parameters']['syndromes']

    def run(self, data: Dict[str, pl.DataFrame], show_progress=False,
            return_features=False, force_casting=False) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        feat = self.preprocess(data, show_progress, force_casting)
        pred = self.predict(feat)
        if return_features:
            return feat, pred
        else:
            return pred

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        path = importlib.resources.files("prophet.models.epilepsy_subtypes").joinpath(model_path)
        return joblib.load(str(path))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False,
                   force_casting=False) -> pl.DataFrame:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)

        # Create feat index: unique (id, date) pairs from notes
        feat = data['note'].select(['id', 'date']).unique()
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data.")
        logger.info(f'Generating features for n = {len(feat)}')

        # Filter auxiliary data to relevant patients
        patient_ids = feat['id'].unique()
        data['icd'] = data['icd'].filter(pl.col('id').is_in(patient_ids))
        data['med'] = data['med'].filter(pl.col('id').is_in(patient_ids))

        # Extract features
        _start = time.time()
        logger.info(f"ICD preprocessing started at {datetime.now()}")
        icd_feat = self._preprocess_icd(data['icd'], feat)
        logger.info(f"ICD preprocessing finished in {time.time() - _start:.2f}s")

        _start = time.time()
        logger.info(f"Med preprocessing started at {datetime.now()}")
        med_feat = self._preprocess_med(data['med'], feat)
        logger.info(f"Med preprocessing finished in {time.time() - _start:.2f}s")

        _start = time.time()
        logger.info(f"Note preprocessing started at {datetime.now()}")
        note_feat = self._preprocess_note(data['note'], feat, show_progress)
        logger.info(f"Note preprocessing finished in {time.time() - _start:.2f}s")

        # Join all features
        feat = feat.join(icd_feat, on=['id', 'date'], how='left') \
                   .join(med_feat, on=['id', 'date'], how='left') \
                   .join(note_feat, on=['id', 'date'], how='left') \
                   .fill_null(0)

        logger.info(f"Preprocessing finished. Feature matrix shape: {feat.shape}")
        return feat

    def _preprocess_icd(self, icd_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        """Extract binary ICD features: 1 if patient has matching ICD within ±1 year."""
        # Create date windows for temporal matching
        icd_df = icd_df.with_columns(
            (pl.col('date').dt.offset_by('-1y')).alias('date_lower'),
            (pl.col('date').dt.offset_by('1y')).alias('date_upper'),
        ).drop('date')

        # Join with feat to get note dates, then filter by temporal window
        # For null-date notes, use all ICDs for that patient (no temporal restriction)
        matched = feat.join(icd_df, on='id', how='left').filter(
            pl.col('date').is_null() |
            ((pl.col('date') >= pl.col('date_lower')) &
             (pl.col('date') <= pl.col('date_upper')))
        ).drop(['date_lower', 'date_upper'])

        # For each ICD code in our union list, create binary feature
        icd_exprs = []
        for icd_code in self._all_icds:
            col_name = f'icd_{icd_code}'
            icd_exprs.append(
                pl.col('icd').str.starts_with(icd_code).cast(pl.Int32).max().alias(col_name)
            )

        if len(matched) > 0:
            icd_feat = matched.group_by(['id', 'date']).agg(icd_exprs)
        else:
            # No ICD matches - create empty feature frame
            icd_feat = feat.select(['id', 'date']).with_columns(
                [pl.lit(0).alias(f'icd_{c}') for c in self._all_icds]
            )

        icd_feat = feat.join(icd_feat, on=['id', 'date'], how='left').fill_null(0)
        return icd_feat

    def _preprocess_med(self, med_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        """Extract binary medication features with consolidation."""
        name_to_canonical = self._med_config['name_to_canonical']

        # Normalize med names
        med_df = med_df.with_columns(pl.col('med').str.to_lowercase().str.strip_chars())

        # Map medications to canonical names via substring matching
        when_chain = pl.when(False).then(None)
        for med_name, canonical in name_to_canonical.items():
            when_chain = when_chain.when(
                pl.col('med').str.contains(f'(?i){re.escape(med_name)}')
            ).then(pl.lit(canonical))
        med_df = med_df.with_columns(
            when_chain.otherwise(None).alias('canonical_med')
        ).filter(pl.col('canonical_med').is_not_null())

        # 2-year lookback from note date: med must be within [note_date - 2y, note_date]
        med_df = med_df.rename({'date': 'med_date'})
        # For null-date notes, use all meds for that patient (no temporal restriction)
        matched = feat.with_columns(
            (pl.col('date').dt.offset_by('-2y')).alias('date_lower'),
        ).join(med_df, on='id', how='left').filter(
            pl.col('date').is_null() |
            ((pl.col('med_date') >= pl.col('date_lower')) &
             (pl.col('med_date') <= pl.col('date')))
        ).drop(['date_lower', 'med_date'])

        if len(matched) > 0:
            # Create binary features per canonical med
            med_exprs = []
            for canonical in self._canonical_meds:
                col_name = f'med_{canonical}'
                med_exprs.append(
                    (pl.col('canonical_med') == canonical).cast(pl.Int32).max().alias(col_name)
                )
            med_feat = matched.group_by(['id', 'date']).agg(med_exprs)
        else:
            med_feat = feat.select(['id', 'date']).with_columns(
                [pl.lit(0).alias(f'med_{c}') for c in self._canonical_meds]
            )

        med_feat = feat.join(med_feat, on=['id', 'date'], how='left').fill_null(0)
        return med_feat

    def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame,
                         show_progress=False) -> pl.DataFrame:
        """Extract keyword features from clinical notes using stemming + regex."""
        # Combine multiple notes per (id, date) pair
        note_df = note_df.with_row_index().with_columns(
            pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_lowercase()
        ).group_by(['id', 'date']).agg(
            pl.col('index').first(),
            pl.col('note').str.concat(delimiter=' ').alias('note')
        )

        all_keywords = self._all_keywords  # sorted list

        # Ray initialization
        ray_was_initialized = ray.is_initialized()
        if not ray_was_initialized and not self.external_ray:
            ray.init(
                object_store_memory=int(0.2 * psutil.virtual_memory().total),
                _memory=int(0.3 * psutil.virtual_memory().total),
                _redis_max_memory=int(0.05 * psutil.virtual_memory().total)
            )
            should_shutdown = True
        else:
            should_shutdown = False

        try:
            batch_size = 500
            num_cpus = int(ray.available_resources().get("CPU", 1))
            total_notes = len(note_df)
            temp_dir = Path(tempfile.mkdtemp(prefix="epilepsy_subtypes_processing_"))

            if show_progress:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_notes, desc='Processing epilepsy subtype notes')
            else:
                progress_bar = None

            @ray.remote(num_cpus=1)
            class KeywordProcessor:
                def __init__(self):
                    self.stemmer = SnowballStemmer('english')
                    # Compile keyword patterns: each keyword may contain \s for whitespace
                    self.keyword_patterns = []
                    for kw in all_keywords:
                        try:
                            pattern = re.compile(rf'\b{kw}\b', re.IGNORECASE)
                            self.keyword_patterns.append((kw, pattern))
                        except re.error:
                            logger.warning(f"Invalid regex keyword: {kw}")
                            self.keyword_patterns.append((kw, None))

                def process_batch(self, batch_data, batch_id):
                    batch_results = []
                    try:
                        for item in batch_data:
                            text = item['note']
                            index = item['index']

                            # Stem the text
                            tokens = word_tokenize(text)
                            stemmed_text = ' '.join(self.stemmer.stem(t) for t in tokens)

                            # Count keyword matches
                            feature_vector = {'index': index}
                            for kw, pattern in self.keyword_patterns:
                                col_name = f'kw_{kw}'
                                if pattern is not None:
                                    matches = pattern.findall(stemmed_text)
                                    feature_vector[col_name] = len(matches)
                                else:
                                    feature_vector[col_name] = 0

                            batch_results.append(feature_vector)

                        batch_df = pl.DataFrame(batch_results)
                        output_path = temp_dir / f"batch_{batch_id}.parquet"
                        batch_df.write_parquet(output_path)

                        del batch_results, batch_df
                        gc.collect()
                        return len(batch_data), str(output_path)
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_id}: {e}")
                        raise

            max_concurrent_actors = max(1, num_cpus - 2)
            processors = [KeywordProcessor.remote() for _ in range(max_concurrent_actors)]

            notes_with_indices = note_df.select(['index', 'note']).to_dicts()
            note_batches = [notes_with_indices[i:i + batch_size]
                           for i in range(0, len(notes_with_indices), batch_size)]

            logger.info(f"Processing {total_notes} notes in {len(note_batches)} batches "
                       f"using {max_concurrent_actors} actors")

            futures = []
            batch_counter = 0
            completed_files = []

            for i, batch in enumerate(note_batches):
                processor = processors[i % len(processors)]
                future = processor.process_batch.remote(batch, batch_counter)
                futures.append(future)
                batch_counter += 1

                if len(futures) >= max_concurrent_actors * 2:
                    ready, futures = ray.wait(futures, num_returns=1, timeout=None)
                    for ready_ref in ready:
                        try:
                            processed_count, file_path = ray.get(ready_ref)
                            completed_files.append(file_path)
                            if progress_bar:
                                progress_bar.update(processed_count)
                        except Exception as e:
                            logger.error(f"Error getting result: {e}")
                            continue

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

            for processor in processors:
                ray.kill(processor)
            gc.collect()

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise
        finally:
            if should_shutdown:
                ray.shutdown()
            gc.collect()

        try:
            parquet_files = list(temp_dir.glob("batch_*.parquet"))
            if not parquet_files:
                raise ValueError("No batch files were created")

            note_feat = pl.read_parquet(parquet_files).with_columns(
                pl.col('index').cast(pl.UInt32)
            )

            # Join with note data to get id/date
            note_feat = note_feat.join(
                note_df.select(['index', 'id', 'date']),
                on='index',
                how='left'
            ).drop('index')

            # Join with feat
            note_feat = feat.join(note_feat, on=['id', 'date'], how='left').fill_null(0)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return note_feat

    def predict(self, feat: pl.DataFrame) -> pl.DataFrame:
        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")

        # The model is a dict: {syndrome_name: {'model': fitted_model, 'threshold': float, 'feature_cols': list}}
        models_dict = self.model

        result = feat.select(['id', 'date'])

        for syndrome in self._syndromes:
            if syndrome not in models_dict:
                logger.warning(f"No trained model for syndrome: {syndrome}")
                continue

            syndrome_info = models_dict[syndrome]
            model = syndrome_info['model']
            threshold = syndrome_info['threshold']
            feature_cols = syndrome_info['feature_cols']

            # Select features that exist in our feature matrix
            available_cols = [c for c in feature_cols if c in feat.columns]
            missing_cols = [c for c in feature_cols if c not in feat.columns]
            if missing_cols:
                logger.warning(f"{syndrome}: {len(missing_cols)} features missing, filling with 0")

            # Create feature matrix with correct columns in order
            X = feat.select(available_cols)
            for mc in missing_cols:
                X = X.with_columns(pl.lit(0).alias(mc))
            X = X.select(feature_cols)  # ensure correct order

            proba = model.predict_proba(X)
            prob_col = f'prob_{syndrome}'
            pred_col = f'pred_{syndrome}'

            # Use positive class probability
            if proba.ndim > 1 and proba.shape[1] > 1:
                proba_pos = proba[:, 1]
            else:
                proba_pos = proba

            result = result.with_columns(
                pl.Series(name=prob_col, values=proba_pos),
                pl.when(pl.Series(values=proba_pos) > threshold)
                .then(1)
                .otherwise(0)
                .alias(pred_col)
            )

        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return result
