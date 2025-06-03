from ...base._basemodel import _BaseModel
import logging
from typing import Dict, Any
import polars as pl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import SnowballStemmer
import nltk
import ray
import joblib
from ray.experimental import tqdm_ray
from datetime import datetime
import time
import importlib.resources
import gc

logger = logging.getLogger(__name__)

class EpilepsyModel(_BaseModel):
    path = importlib.resources.files("prophet.models.epilepsy").joinpath("config.yaml")
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

    def run(self, data: Dict[str, pl.DataFrame], show_progress=False, return_features=False, force_casting=False) -> pl.DataFrame:
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
        path = importlib.resources.files("prophet.models.epilepsy").joinpath(model_path)
        return joblib.load(str(path))

    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        preproc_start = time.time()
        logger.info(f"Preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
        data = super().preprocess(data, show_progress, force_casting)

        feat = data['note'].select(['id', 'date']).unique()
        logger.info(f'Generating features for n = {len(feat)}')

        # preliminary filter by ids
        data['demo'] = data['demo'].filter(pl.col('id').is_in(feat['id'].unique()))
        data['icd'] = data['icd'].filter(pl.col('id').is_in(feat['id'].unique()))
        data['med'] = data['med'].filter(pl.col('id').is_in(feat['id'].unique()))

        # begin preprocess
        _start = time.time()
        logger.info(f"Demo preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
        demo_feat = self._preprocess_demo(data['demo'], feat)
        logger.info(f"Demo preprocessing finished in {time.time() - _start:.2f}s")

        _start = time.time()
        logger.info(f"ICD preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
        icd_feat = self._preprocess_icd(data['icd'], feat)
        logger.info(f"ICD preprocessing finished in {time.time() - _start:.2f}s")

        _start = time.time()
        logger.info(f"Med preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
        med_feat = self._preprocess_med(data['med'], feat)
        logger.info(f"Med preprocessing finished in {time.time() - _start:.2f}s")

        _start = time.time()
        logger.info(f"Note preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
        note_feat = self._preprocess_note(data['note'], feat, show_progress)
        logger.info(f"Note preprocessing finished in {time.time() - _start:.2f}s")

        # join features
        feat = feat.join(
            demo_feat,
            on=['id', 'date'],
            how='left'
        ).join(
            icd_feat,
            on=['id', 'date'],
            how='left'
        ).join(
            med_feat,
            on=['id', 'date'],
            how='left'
        ).join(
            note_feat,
            on=['id', 'date'],
            how='left'
        ).select(['id', 'date'] + self.config['parameters']['final_cols']).drop_nulls()

        logger.info(f"Preprocessing finished in {time.time() - preproc_start:.2f}s")
        return feat

    def _preprocess_demo(self, demo_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        # Ensure unique rows for each 'id'
        unique_demo_df = demo_df.unique()
        id_counts = unique_demo_df['id'].value_counts()
        dup_vals = id_counts.filter(pl.col('count') > 1)
        if not dup_vals.is_empty():
            raise ValueError(f"Duplicate non-unique 'id' values found in demo data: {dup_vals['id'].to_list()}")
        if len(unique_demo_df['id'].unique()) != len(feat['id'].unique()):
            logger.warning(f"Missing demo info for id(s): {set(feat['id']) - set(demo_df['id'])}\nThese will be removed from the dataset.")
        
        demo_feat = feat.join(
            unique_demo_df,
            on='id',
            how='left'
        ).select(
            pl.col('id'),
            pl.col('date'),
            (pl.col('date').dt.year() - pl.col('date_of_birth').dt.year()).alias('age'),
            pl.when((pl.col('sex').str.to_lowercase() == 'm') | (pl.col('sex').str.to_lowercase() == 'male'))
            .then(1)
            .when(pl.col('sex').is_null())
            .then(None)
            .otherwise(0)
            .alias('sex')
        )

        return demo_feat

    def _preprocess_icd(self, icd_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        icd_config = self.config['parameters']['icd']
        icd_df = icd_df.filter(pl.col('icd').str.contains('|'.join(icd_config.values())))

        when_then_chain = pl.when(False).then(None)  # Start with a false condition as placeholder

        # Build the pattern matching chain
        for name, codes in icd_config.items():
            when_then_chain = when_then_chain.when(pl.col('icd').str.contains(codes)).then(pl.lit(name))

        # Apply the when-then chain, defaulting to the original value if no match
        icd_df = icd_df.with_columns(
            when_then_chain.otherwise(pl.col('icd')).alias('icd_group')
        )

        icd_df = icd_df.with_columns(
            (pl.col('date').dt.offset_by('-6mo')).alias('date_lower'),
            (pl.col('date').dt.offset_by('6mo')).alias('date_upper'),
        ).drop('date')
        icd_feat = feat.join(
            feat.join(
                icd_df,
                on='id',
                how='left'
            ).filter(
                (pl.col('date') >= pl.col('date_lower')) &
                (pl.col('date') <= pl.col('date_upper'))
            ).with_columns(
                pl.when(pl.col('icd_group').str.contains('epilepsy_and_recurrent_seizures'))
                .then(1)
                .otherwise(0)
                .alias('epilepsy_and_recurrent_seizures'),
                pl.when(pl.col('icd_group').str.contains('convulsions_seizures'))
                .then(1)
                .otherwise(0)
                .alias('convulsions_seizures'),
                pl.when(pl.col('icd_group').str.contains('syncope'))
                .then(1)
                .otherwise(0)
                .alias('syncope')
            ).group_by(['id', 'date']).agg(
                pl.col('epilepsy_and_recurrent_seizures').max(),
                pl.col('convulsions_seizures').max(),
                pl.col('syncope').max()
            ),
            on=['id', 'date'],
            how='left'
        ).with_columns(
            pl.col('epilepsy_and_recurrent_seizures').fill_null(0),
            pl.col('convulsions_seizures').fill_null(0),
            pl.col('syncope').fill_null(0),
            (pl.col('epilepsy_and_recurrent_seizures') + pl.col('convulsions_seizures')).fill_null(0).alias('n_icds') # does not include syncope
        )
        return icd_feat
    
    def _preprocess_med(self, med_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        med_config = self.config['parameters']['med']
        med_df = med_df.with_columns(
            pl.col('med').str.to_lowercase()
        ).filter(
            pl.col('med').str.contains('|'.join(med_config['names']))
        )
        # Create a when-then expression chain
        when_then_chain = pl.when(False).then(None)  # Start with a false condition as placeholder

        # Build the pattern matching chain
        for pattern, replacement in med_config['groupings'].items():
            when_then_chain = when_then_chain.when(pl.col('med').str.contains(pattern)).then(pl.lit(replacement))

        # Apply the when-then chain, defaulting to the original value if no match
        med_df = med_df.with_columns(
            when_then_chain.otherwise(pl.col('med')).alias('med')
        )
        med_df = med_df.with_columns(
            (pl.col('date').dt.offset_by('-6mo')).alias('date_lower'),
            (pl.col('date').dt.offset_by('6mo')).alias('date_upper'),
        ).drop('date')
        med_feat = feat.join(
            med_df,
            on='id',
            how='left'
        ).filter(
            (pl.col('date') >= pl.col('date_lower')) &
            (pl.col('date') <= pl.col('date_upper'))
        ).to_dummies('med').drop(['date_lower', 'date_upper'])
        med_feat = med_feat.rename({x: x.replace('med_', '') for x in med_feat.columns if 'med_' in x})
        med_feat = med_feat.with_columns(
            pl.lit(0).alias(x) for x in med_config['names'] if x not in med_feat.columns
        ).group_by(['id', 'date']).agg(
            [pl.col(x).max().alias(x) for x in med_config['names']]
        ).with_columns(
            pl.sum_horizontal(pl.all().exclude('id', 'date')).alias('n_meds')
        )
        med_feat = feat.join(
            med_feat,
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
        ).group_by(['id', 'date']).agg(
            pl.col('*').exclude('note'),
            pl.col('note').str.concat(delimiter=' ').alias('note')
        )
        kw_config = self.config['parameters']['kws']

        pro_feat_names = {k : set(v) for k, v in kw_config['pro'].items()}
        neg_feat_names = {k : set(v) for k, v in kw_config['neg'].items()}
        med_feat_names = set(kw_config['med'])
        all_note_feat = list(pro_feat_names.keys()) + list(neg_feat_names.keys()) + list(med_feat_names)


        ray_was_initialized = ray.is_initialized()
        
        if not ray_was_initialized and not self.external_ray:
            ray.init()
            should_shutdown = True
        else:
            should_shutdown = False

        try:
            # Calculate batch size based on available CPUs
            num_cpus = int(ray.available_resources().get("CPU", 1))
            total_notes = len(note_df)
            batch_size = max(1, total_notes // (num_cpus * 4))  # 4x more batches than CPUs for better load balancing
            
            if show_progress:
                remote_tqdm = ray.remote(tqdm_ray.tqdm)
                bar = remote_tqdm.remote(total=total_notes, desc='Processing notes')
            else:
                bar = None

            @ray.remote
            def process_note_batch(text_batch, stemmer_r, bar):
                """Process a batch of notes instead of individual notes"""
                batch_results = []
                
                for text in text_batch:
                    feature_vector = dict.fromkeys(all_note_feat, 0)
                    sentences = sent_tokenize(text)
                    for s in sentences:
                        stem_words = set(stemmer_r.stem(word) for word in word_tokenize(s))

                        # Check for AEDs
                        for word in stem_words.intersection(med_feat_names):
                            feature_vector[word] = 1

                        # Check for anti-epilepsy bag of words and pro-evidences
                        for dictionary in (pro_feat_names, neg_feat_names):
                            for bag, words in dictionary.items():
                                if words.issubset(stem_words):
                                    feature_vector[bag] = 1
                        batch_results.append(feature_vector)
                        
                        # Update progress for each note in the batch
                        if bar:
                            bar.update.remote(1)
                
                return batch_results

            # Split notes into batches
            notes_list = note_df['note'].to_list()
            note_batches = [notes_list[i:i + batch_size] for i in range(0, len(notes_list), batch_size)]
            
            logger.info(f"Processing {total_notes} notes in {len(note_batches)} batches of ~{batch_size} notes each using {num_cpus} CPUs")
            
            # Create one task per batch instead of per note
            futures = [process_note_batch.remote(batch, ray.put(SnowballStemmer('english')), bar) 
                    for batch in note_batches]
            
            # Get results and flatten the list of lists
            batch_results = ray.get(futures)
            note_feat = [item for batch in batch_results for item in batch]
            
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

        note_feat = pl.DataFrame(note_feat)
        for col1, col2 in kw_config['join_columns']:
            if col1 in note_feat.columns and col2 in note_feat.columns:
                note_feat = note_feat.with_columns(
                    pl.max_horizontal(col1, col2).alias(col1)
                ).drop(col2)
        note_feat = note_feat.rename({col : f'{col}_' for col in note_feat.columns})
        note_feat = note_feat.hstack(note_df.select(['id', 'date']))
        note_feat = feat.join(
            note_feat,
            on=['id', 'date'],
            how='left'
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

