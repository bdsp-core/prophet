from ...base._basemodel import _BaseModel
import logging
from typing import Dict, Union, Tuple
import polars as pl
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
import nltk
import ray
import re
import joblib
import time
import gc
import os
import shutil
import psutil
import tempfile
from pathlib import Path
from datetime import datetime
import importlib.resources

logger = logging.getLogger(__name__)

# 41 regex patterns applied to stemmed text, in the exact order used during training.
# Each tuple is (regex_pattern, human_readable_name).
WLST_REGEX_PATTERNS = [
    (r'\bno escal of care\b', 'no_escalation_of_care'),
    (r'\bcmo\b', 'cmo'),
    (r'\barrest\b', 'arrest'),
    (r'\banox\b', 'anoxia'),
    (r'\brewarm\b', 'rewarm'),
    (r'\bhypothermia\b', 'hypothermia'),
    (r'\bcool\b', 'cool'),
    (r'\bttm\b', 'ttm'),
    (r'\bneurolog recoveri\b', 'neurologic_recovery'),
    (r'\bintub\b', 'intubation'),
    (r'\bpalliat extub\b', 'palliative_extubation'),
    (r'\b(no|not) respons\b', 'no_response'),
    (r'\bcardiac arrest\b', 'cardiac_arrest'),
    (r'\bdischarg\b', 'discharge'),
    (r'\bfull code\b', 'full_code'),
    (r'\bwithdraw([(a-z)|\s]{0,20})(care|support)\b', 'withdraw_care_support'),
    (r'\b(wean([(a-z)|\s]{0,20})support\b)|(\bsupport([(a-z)|\s]{0,20})wean\b)', 'wean_support'),
    (r'\b(cvvh|ecmo|ecl)([(a-z)|\s]{0,20})(discontinu|wean)\b', 'discontinue_ecmo_cvvh'),
    (r'\btermin([(a-z)|\s]{0,20})extub\b', 'terminal_extubation'),
    (r'\btransit([(a-z)|\s]{0,20})comfort\b', 'transition_to_comfort'),
    (r'\bcomfort (measur|onli|care|center|base care)', 'comfort_measures'),
    (r'\bexpir\b', 'expired'),
    (r'\bautopsi\b', 'autopsy'),
    (r'\bpass away\b', 'pass_away'),
    (r'\bpass\b', 'pass'),
    (r'\bdeath\b', 'death'),
    (r'\bdie\b', 'die'),
    (r'\bdeceas\b', 'deceased'),
    (r'\bdead\b', 'dead'),
    (r'\bpronounc\b', 'pronounced'),
    (r'\bno heart sound\b', 'no_heart_sound'),
    (r'\bpost mortem\b', 'post_mortem'),
    (r'\bpass out\b', 'pass_out'),
    (r'\bdeath or\b|\bor death\b', 'death_or'),
    (r'\bsuccess extub', 'successful_extubation'),
    (r'\bdue[(a-z)|\s]{0,20}arrest\b', 'due_to_arrest'),
    (r'\bbrain death\b', 'brain_death'),
    (r'\bconvers\b', 'conversation'),
    (r'\banswer question\b', 'answer_questions'),
    (r'\bawak and alert\b', 'awake_and_alert'),
    (r'\bsudden cardiac death\b', 'sudden_cardiac_death'),
]

# Precompile regex patterns
_COMPILED_PATTERNS = [(re.compile(pat), name) for pat, name in WLST_REGEX_PATTERNS]


class WLSTModel(_BaseModel):
    path = importlib.resources.files("prophet.models.wlst").joinpath("config.yaml")
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
        feat = self.preprocess(data, show_progress, force_casting)
        pred = self.predict(feat)
        if return_features:
            return feat, pred
        else:
            return pred

    def load_model(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        path = importlib.resources.files("prophet.models.wlst").joinpath(model_path)
        return joblib.load(str(path))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> pl.DataFrame:
        logger.info(f"Preprocessing started at {datetime.now()}")
        data = super().preprocess(data, show_progress, force_casting)
        data['note'] = data['note'].with_row_index()

        feat = data['note'].select(['index', 'id'])
        if len(feat) == 0:
            raise ValueError("No notes found in the provided data.")
        logger.info(f'Generating features for n = {len(feat)}')

        logger.info(f"Note preprocessing started at {datetime.now()}")
        note_feat = self._preprocess_note(data['note'], feat, show_progress)
        logger.info(f"Note preprocessing finished at {datetime.now()}")

        logger.info(f"Preprocessing finished at {datetime.now()}")
        return note_feat

    def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        # Basic text cleaning
        note_df = note_df.with_columns(
            pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_lowercase()
        )

        feature_names = [name for _, name in WLST_REGEX_PATTERNS]

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

            temp_dir = Path(tempfile.mkdtemp(prefix="wlst_processing_"))

            if show_progress:
                from tqdm import tqdm
                progress_bar = tqdm(total=total_notes, desc='Processing WLST notes')
            else:
                progress_bar = None

            # Serialize patterns for Ray workers
            patterns_list = [(pat, name) for pat, name in WLST_REGEX_PATTERNS]

            @ray.remote(num_cpus=1)
            class WLSTProcessor:
                def __init__(self):
                    self.stemmer = SnowballStemmer('english')
                    self.compiled_patterns = [(re.compile(pat), name) for pat, name in patterns_list]

                def process_batch(self, batch_data, batch_id):
                    batch_results = []
                    try:
                        for item in batch_data:
                            text = item['note']
                            index = item['index']

                            # Stem the entire text
                            tokens = word_tokenize(text)
                            stemmed_text = ' '.join(self.stemmer.stem(t) for t in tokens)

                            # Apply each regex pattern
                            feature_vector = {'index': index}
                            for compiled_pat, name in self.compiled_patterns:
                                feature_vector[name] = 1 if compiled_pat.search(stemmed_text) else 0

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
            processors = [WLSTProcessor.remote() for _ in range(max_concurrent_actors)]

            notes_with_indices = note_df.select(['index', 'note']).to_dicts()
            note_batches = [notes_with_indices[i:i + batch_size] for i in range(0, len(notes_with_indices), batch_size)]

            logger.info(f"Processing {total_notes} notes in {len(note_batches)} batches using {max_concurrent_actors} actors")

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

            note_feat = pl.read_parquet(parquet_files)

            # Join with original note data to get id and date
            note_feat = note_feat.join(
                note_df.select(['index', 'id']),
                on='index',
                how='left'
            )

            # Join with feat to preserve original row structure
            note_feat = feat.join(
                note_feat,
                on=['index', 'id'],
                how='left',
                validate='1:1'
            ).fill_null(0)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return note_feat

    def predict(self, feat: pl.DataFrame) -> pl.DataFrame:
        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")

        feature_names = [name for _, name in WLST_REGEX_PATTERNS]
        pred = self.model.predict_proba(feat.select(feature_names))
        pred = pl.DataFrame(pred, schema=['prob_NO', 'prob_YES'], orient='row')

        threshold = self.config['parameters']['threshold']
        logger.info(f'Using suggested threshold of {threshold}')

        pred = feat.select(pl.all().exclude(feature_names)).hstack(
            pred.with_columns(
                pl.when(pl.col('prob_YES') > threshold)
                .then(1)
                .otherwise(0)
                .alias('prediction')
            )
        )
        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return pred
