from ...base._basemodel import _BaseModel
import logging
from typing import Dict, Any, List, Optional, Union
import polars as pl
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import SnowballStemmer
import nltk
import ray
import joblib
import yaml
import os
import numpy as np
from ray.experimental import tqdm_ray
from datetime import datetime
import time
import importlib.resources
from pathlib import Path

logger = logging.getLogger(__name__)

class TesterModel():
    """
    1. Extract features based on a configuration file
    2. Test different feature configurations quickly
    3. Optionally train models using extracted features
    """
    class PolarsSafeLoader(yaml.SafeLoader):
        pass
    
    # Initialize the loader with the constructor
    @classmethod
    def _init_yaml_loader(cls):
        # Create the constructor for Polars datatypes
        def polars_constructor(loader: yaml.SafeLoader, node: yaml.Node):
            value = loader.construct_scalar(node)
            return getattr(pl, value)
        
        cls.PolarsSafeLoader.add_constructor('!pl', polars_constructor)
    
    @classmethod
    def load_config(cls, config_path):
        """Load config from YAML file with Polars support"""
        # Ensure loader is initialized
        cls._init_yaml_loader()
        try:
            with open(config_path, 'r') as file:
                return yaml.load(file, Loader=cls.PolarsSafeLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the TesterModel.
        
        Args:
            config_path: Path to the configuration YAML file
        """

        self.config = {}
        if config_path:
            self.config = self.load_config(config_path)

        # Install NLTK resources if not already available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('stemmers/snowball')
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            
        self.stemmer = SnowballStemmer("english")
        self.model = None

    def load_model(self, model_path: Optional[str] = None):
        """
        Load a model from a specified path.
        
        Args:
            model_path: Path to the model file
        """
        if model_path is None:
            return None
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model using joblib
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model
        
    def run(self, data: Dict[str, pl.DataFrame], show_progress=False, 
           return_features=True, force_casting=False) -> Union[pl.DataFrame, Dict[str, Any]]:
        """
        Run feature extraction and optionally train a model.
        
        Args:
            data: Dictionary of DataFrames containing the required data
            show_progress: Whether to show a progress bar
            return_features: Whether to return the features (True) or predictions (False)
            force_casting: Boolean flag to force casting of columns
            
        Returns:
            Features DataFrame if return_features=True, else predictions
        """
        feat = self.preprocess(data, show_progress, force_casting)
        
        if return_features:
            return feat
        else:
            if self.model is None:
                logger.warning("No model has been trained or loaded. Returning features instead.")
                return feat
            else:
                return self.predict(feat)
        
    def preprocess(self, data: Dict[str, pl.DataFrame], show_progress=False, force_casting=False) -> pl.DataFrame:
        """
        Extract features based on the current configuration.
        
        Args:
            data: Dictionary of DataFrames containing the required data
            show_progress: Whether to show a progress bar
            force_casting: Boolean flag to force casting of columns
        
        Returns:
            DataFrame with extracted features
        """
        preproc_start = time.time()
        logger.info(f"Preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
        
        
        # Create base feature dataframe with IDs and dates
        feature_sources = []
        
        # Determine which data sources to use based on what's in the data dictionary
        if 'note' in data:
            feat = data['note'].select(['id', 'date']).unique()
        else:
            # Create an empty DataFrame with id and date columns
            feat = pl.DataFrame({'id': [], 'date': []})
            
        logger.info(f'Generating features for n = {len(feat)}')
        
        # Process each data type if present in the data and config
        feature_dfs = []
        
        # Process demographics if available
        if 'demo' in data and 'demographics' in self.config.get('parameters', {}):
            _start = time.time()
            logger.info(f"Demo preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
            demo_feat = self._preprocess_demo(data['demo'], feat)
            feature_dfs.append(demo_feat)
            feature_sources.append('demographics')
            logger.info(f"Demo preprocessing finished in {time.time() - _start:.2f}s")
            
        # Process ICD codes if available
        if 'icd' in data and 'icd' in self.config.get('parameters', {}):
            _start = time.time()
            logger.info(f"ICD preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
            icd_feat = self._preprocess_icd(data['icd'], feat)
            feature_dfs.append(icd_feat)
            feature_sources.append('icd')
            logger.info(f"ICD preprocessing finished in {time.time() - _start:.2f}s")
            
        # Process medications if available
        if 'med' in data and 'med' in self.config.get('parameters', {}):
            _start = time.time()
            logger.info(f"Med preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
            med_feat = self._preprocess_med(data['med'], feat)
            feature_dfs.append(med_feat)
            feature_sources.append('medications')
            logger.info(f"Med preprocessing finished in {time.time() - _start:.2f}s")
            
        # Process clinical notes if available
        if 'note' in data and 'keywords' in self.config.get('parameters', {}):
            _start = time.time()
            logger.info(f"Note preprocessing started at {datetime.now().strftime('%H:%M:%S')}")
            note_feat = self._preprocess_note(data['note'], feat, show_progress)
            feature_dfs.append(note_feat)
            feature_sources.append('notes')
            logger.info(f"Note preprocessing finished in {time.time() - _start:.2f}s")
        
        # Join all features together
        result_df = feat
        for df in feature_dfs:
            result_df = result_df.join(
                df,
                on=['id', 'date'],
                how='left'
            )
            
        # Get list of feature columns from config if available, otherwise use all columns except id and date
        if 'final_cols' in self.config.get('parameters', {}) and self.config['parameters']['final_cols']:
            feature_cols = self.config['parameters']['final_cols']
            result_df = result_df.select(['id', 'date'] + feature_cols).drop_nulls()
        else:
            result_df = result_df.drop_nulls()
            
        logger.info(f"Preprocessing finished in {time.time() - preproc_start:.2f}s")
        logger.info(f"Generated features from: {', '.join(feature_sources)}")
        
        return result_df
    
    def _preprocess_demo(self, demo_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        """
        Extract demographic features.
        
        Args:
            demo_df: DataFrame with demographic data
            feat: Base feature DataFrame with IDs and dates
            
        Returns:
            DataFrame with demographic features
        """
        # Handle potential duplicates
        unique_demo_df = demo_df.unique()
        id_counts = unique_demo_df['id'].value_counts()
        dup_vals = id_counts.filter(pl.col('count') > 1)
        
        if not dup_vals.is_empty():
            logger.warning(f"Found {len(dup_vals)} IDs with duplicate demographic entries. Using first occurrence.")
            # Keep the first occurrence of each ID
            unique_demo_df = unique_demo_df.group_by('id').agg(pl.all().first())
            
        demo_config = self.config['parameters'].get('demographics', {})
        extract_cols = demo_config.get('extract_columns', [])
        
        # Base selection always includes ID
        select_cols = [pl.col('id')]
        
        # Process demographic features based on config
        if 'age' in extract_cols and 'date_of_birth' in demo_df.columns and 'date' in feat.columns:
            select_cols.append((pl.col('date').dt.year() - pl.col('date_of_birth').dt.year()).alias('age'))
            
        if 'sex' in extract_cols and 'sex' in demo_df.columns:
            select_cols.append(
                pl.when((pl.col('sex').str.to_lowercase() == 'm') | (pl.col('sex').str.to_lowercase() == 'male'))
                .then(1)
                .when(pl.col('sex').is_null())
                .then(None)
                .otherwise(0)
                .alias('sex')
            )
            
        # Add any other direct columns to extract
        for column in extract_cols:
            if column not in ['age', 'sex'] and column in demo_df.columns:
                select_cols.append(pl.col(column))
                
        # Join with feat dataframe to get the correct dimensions
        if 'date' in feat.columns and 'date' not in unique_demo_df.columns:
            demo_feat = feat.join(
                unique_demo_df.select(select_cols),
                on='id',
                how='left'
            )
        else:
            demo_feat = unique_demo_df.select(select_cols)
            if 'date' in feat.columns and 'date' not in demo_feat.columns:
                demo_feat = demo_feat.with_columns(pl.lit(datetime.now().date()).alias('date'))
        
        return demo_feat
    
    def _preprocess_icd(self, icd_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        """
        Extract features from ICD codes.
        
        Args:
            icd_df: DataFrame with ICD codes
            feat: Base feature DataFrame with IDs and dates
            
        Returns:
            DataFrame with ICD features
        """
        icd_config = self.config['parameters'].get('icd', {})
        
        # Get list of ICD codes to search for
        icd_codes = icd_config.get('codes', [])
        if not icd_codes:
            logger.warning("No ICD codes specified in configuration. No ICD features will be generated.")
            return feat.select(['id', 'date'])
        
        # Filter by specified ICD codes
        icd_df = icd_df.filter(pl.col('icd').str.contains('|'.join(icd_codes)))
        
        # Apply time window
        time_window = icd_config.get('time_window', '6mo')
        icd_df = icd_df.with_columns(
            (pl.col('date').dt.offset_by(f'-{time_window}')).alias('date_lower'),
            (pl.col('date').dt.offset_by(time_window)).alias('date_upper'),
        ).drop('date')
        
        # Join with feat dataframe and filter by date range
        filtered_df = feat.join(
            icd_df,
            on='id',
            how='left'
        ).filter(
            (pl.col('date') >= pl.col('date_lower')) &
            (pl.col('date') <= pl.col('date_upper'))
        )
        
        # Group by ID and date, and create indicator variables for each ICD code
        if not filtered_df.is_empty():
            # Create binary indicators for each ICD code
            icd_indicators = filtered_df.group_by(['id', 'date']).agg(
                [pl.when(pl.col('icd').str.contains(code)).then(1).otherwise(0).max().alias(code) 
                for code in icd_codes]
            )
            
            # Add count of total ICD codes
            icd_indicators = icd_indicators.with_columns(
                pl.sum_horizontal([pl.col(code) for code in icd_codes]).alias('n_icds')
            )
            
            # Join back to original features
            icd_result = feat.join(
                icd_indicators,
                on=['id', 'date'],
                how='left'
            ).fill_null(0)
        else:
            # If no matches, create empty indicators
            columns = [pl.lit(0).alias(code) for code in icd_codes]
            columns.append(pl.lit(0).alias('n_icds'))  # Add total count column
            icd_result = feat.with_columns(columns)
        
        return icd_result
    
    def _preprocess_med(self, med_df: pl.DataFrame, feat: pl.DataFrame) -> pl.DataFrame:
        """
        Extract features from medications.
        
        Args:
            med_df: DataFrame with medication data
            feat: Base feature DataFrame with IDs and dates
            
        Returns:
            DataFrame with medication features
        """
        med_config = self.config['parameters'].get('med', {})
        
        # Get medication names from config
        med_names = med_config.get('names', [])
        if not med_names:
            logger.warning("No medication names specified in configuration. No medication features will be generated.")
            return feat.select(['id', 'date'])
        
        # Preprocess medication names and filter by specified medications
        med_df = med_df.with_columns(
            pl.col('med').str.to_lowercase()
        ).filter(
            pl.col('med').str.contains('|'.join(med_names))
        )
        
        # Apply time window
        time_window = med_config.get('time_window', '6mo')
        med_df = med_df.with_columns(
            (pl.col('date').dt.offset_by(f'-{time_window}')).alias('date_lower'),
            (pl.col('date').dt.offset_by(time_window)).alias('date_upper'),
        ).drop('date')
        
        # Join with feat dataframe and filter by date range
        filtered_df = feat.join(
            med_df,
            on='id',
            how='left'
        ).filter(
            (pl.col('date') >= pl.col('date_lower')) &
            (pl.col('date') <= pl.col('date_upper'))
        )
        
        # Group by ID and date, and create indicator variables for each medication
        if not filtered_df.is_empty():
            # Create binary indicators for each medication
            med_indicators = filtered_df.group_by(['id', 'date']).agg(
                [pl.when(pl.col('med').str.contains(med)).then(1).otherwise(0).max().alias(med) 
                for med in med_names]
            )
            
            # Add count of total medications
            med_indicators = med_indicators.with_columns(
                pl.sum_horizontal([pl.col(med) for med in med_names]).alias('n_meds')
            )
            
            # Join back to original features
            med_result = feat.join(
                med_indicators,
                on=['id', 'date'],
                how='left'
            ).fill_null(0)
        else:
            # If no matches, create empty indicators
            columns = [pl.lit(0).alias(med) for med in med_names]
            columns.append(pl.lit(0).alias('n_meds'))  # Add total count column
            med_result = feat.with_columns(columns)
        
        return med_result

    def _preprocess_note(self, note_df: pl.DataFrame, feat: pl.DataFrame, show_progress=False) -> pl.DataFrame:
        """
        Extract features from clinical notes based on keyword presence.
        
        Args:
            note_df: DataFrame with clinical notes
            feat: Base feature DataFrame with IDs and dates
            show_progress: Whether to show a progress bar
            
        Returns:
            DataFrame with note features
        """
        # Preprocess the notes - clean and standardize text
        note_df = note_df.with_columns(
            pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
            .str.to_lowercase()
        ).group_by(['id', 'date']).agg(
            pl.col('note').str.concat(delimiter=' ').alias('note')
        )
        
        # Get keyword configuration
        kw_config = self.config['parameters'].get('keywords', {})
        
        # Convert keywords to sets of stemmed words
        stemmer = SnowballStemmer("english")
        
        # Process positive keywords by converting each space-separated group to set of stemmed words
        keywords = kw_config.get('positive', {})
        if isinstance(keywords, dict):
            # When keywords are organized by category
            kw_names = {k: set(stemmer.stem(word) for word in v.split(' ')) 
                    for k, v in keywords.items()}
        else:
            # When keywords are just a list
            kw_names = {kw: set(stemmer.stem(word) for word in kw.split(' ')) 
                    for kw in keywords}
        
        # Get negation words if available
        negate_words = set()
        if 'negative' in kw_config:
            negate_words = set(stemmer.stem(word) for word in kw_config['negative'])
        
        # Use Ray for parallel processing if available
        with ray.init() as ray_context:
            if show_progress:
                remote_tqdm = ray.remote(tqdm_ray.tqdm)
                bar = remote_tqdm.remote(total=len(note_df), desc='Processing notes')
            else:
                bar = None

            @ray.remote
            def process_note(text, stemmer_r, kw_names_r, negate_words_r, bar):
                # Initialize all features to 0
                feature_vector = dict.fromkeys(kw_names_r, 0)
                
                # Only add negation features if negation words are provided
                if negate_words_r:
                    feature_vector.update({f'{x}_neg': 0 for x in kw_names_r})
                
                # Process note sentence by sentence
                sentences = sent_tokenize(text)
                for s in sentences:
                    # Stem words in the sentence
                    stem_words = set(stemmer_r.stem(word) for word in word_tokenize(s))
                    
                    # Check each keyword set
                    for kw_name, words in kw_names_r.items():
                        if words.issubset(stem_words):
                            # Check if keyword is negated
                            if negate_words_r and negate_words_r.intersection(stem_words):
                                feature_vector[f'{kw_name}_neg'] = 1
                            else:
                                feature_vector[kw_name] = 1
                                
                if bar:
                    bar.update.remote(1)
                return feature_vector
            
            # Process all notes in parallel
            note_feat = ray.get([
                process_note.remote(
                    text, 
                    ray.put(stemmer), 
                    ray.put(kw_names), 
                    ray.put(negate_words), 
                    bar
                ) for text in note_df['note']
            ])
            
            if bar and show_progress:
                bar.close.remote()
        
        # Convert results to DataFrame
        note_feat_df = pl.DataFrame(note_feat)
        
        # Add suffix to feature names
        note_feat_df = note_feat_df.rename({col: f'{col}_' for col in note_feat_df.columns})
        
        # Add ID and date information
        result = note_feat_df.hstack(note_df.select(['id', 'date']))
        
        # Join with original features
        result = feat.join(
            result, 
            on=['id', 'date'],
            how='left'
        )
        
        # Convert boolean columns to integers (1/0)
        bool_cols = [col for col in result.columns if result[col].dtype == pl.Boolean]
        if bool_cols:
            result = result.with_columns(
                [pl.col(col).cast(pl.Int8) for col in bool_cols]
            )
            
        # Fill nulls with 0
        feature_cols = [col for col in result.columns if col.endswith('_')]
        if feature_cols:
            result = result.with_columns(
                [pl.col(col).fill_null(0) for col in feature_cols]
            )
                
        return result
    
    def train_model(self, features: pl.DataFrame, target_column: str, output_dir: str = 'model_output',
                   source_column: Optional[str] = None, random_state: int = 42):
        """
        Train a model using the extracted features.
        
        Args:
            features: DataFrame with features
            target_column: Name of the target column
            output_dir: Directory to save model outputs
            source_column: Optional column indicating data source (for cross-source validation)
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with model results
        """
        from ...utils.model_comp import train_best_production_model, cross_source_model_comparison, regular_kfold_validation
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if target column exists
        if target_column not in features.columns:
            raise ValueError(f"Target column '{target_column}' not found in features DataFrame")
        
        # Set up model configurations
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.svm import SVC
        
        models_config = {
            'LogisticRegression': (
                LogisticRegression,
                {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'solver': ['lbfgs'],  # Remove liblinear for simplicity
                    'penalty': ['l2', None],  # Compatible with lbfgs
                    'class_weight': [None, 'balanced'],
                    'random_state': [42],
                    'max_iter': [2000]  # Increased iterations
                }
            ),
            'LogisticRegression_Liblinear': (
                LogisticRegression,
                {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'solver': ['liblinear'],
                    'penalty': ['l1', 'l2'],  # Only these work with liblinear
                    'class_weight': [None, 'balanced'],
                    'random_state': [42],
                    'max_iter': [2000]
                }
            ),
            'RandomForest': (
                RandomForestClassifier,
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced'],
                    'random_state': [42]
                }
            ),
            'GradientBoosting': (
                GradientBoostingClassifier,
                {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'random_state': [42]
                }
            ),
            'XGBoost': (
                XGBClassifier,
                {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    # 'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)],  # For imbalanced datasets
                    'random_state': [42]
                }
            ),
            # 'LightGBM': (
            #     LGBMClassifier,
            #     {
            #         'n_estimators': [100, 200],
            #         'learning_rate': [0.01, 0.1, 0.2],
            #         'max_depth': [3, 5, 7],
            #         'subsample': [0.8, 1.0],
            #         'colsample_bytree': [0.8, 1.0],
            #         'class_weight': [None, 'balanced'],
            #         'random_state': [42]
            #     }
            # ),
            'SVM': (
                SVC,
                {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto'],
                    'probability': [True],
                    'class_weight': [None, 'balanced'],
                    'random_state': [42]
                }
            )
        }
        
        # Train model based on validation strategy
        if source_column is not None and source_column in features.columns:
            logger.info(f"Running cross-source validation with source column: {source_column}")
            results, curves_data = cross_source_model_comparison(
                features, source_column, target_column, models_config, output_dir, random_state
            )
        else:
            logger.info("Running standard k-fold validation")
            results, curves_data = regular_kfold_validation(
                features, target_column, models_config, output_dir, random_state
            )
        
        # Train final production model
        logger.info("Training best production model on all data")
        final_model, best_params, best_model_name = train_best_production_model(
            features, source_column, target_column, results, output_dir, random_state
        )
                
        return results, curves_data, final_model, best_params, best_model_name
    
    def predict(self, feat: pl.DataFrame) -> pl.DataFrame:
        """
        Generate predictions from features.
        
        Args:
            feat: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded")
        
        _start = time.time()
        logger.info(f"Prediction started at {datetime.now().strftime('%H:%M:%S')}")
        
        # Get feature columns from model
        if hasattr(self.model, 'feature_names_in_'):
            feature_cols = list(self.model.feature_names_in_)
        else:
            feature_cols = [col for col in feat.columns if col not in ['id', 'date']]
            
        # Ensure all required features are present
        missing_cols = [col for col in feature_cols if col not in feat.columns]
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}. Adding with zero values.")
            feat = feat.with_columns([pl.lit(0).alias(col) for col in missing_cols])
        
        # Generate predictions
        pred_proba = self.model.predict_proba(feat.select(feature_cols))
        pred_df = pl.DataFrame({
            'prob_NO': pred_proba[:, 0],
            'prob_YES': pred_proba[:, 1]
        })
        
        # Apply threshold
        threshold = getattr(self, 'threshold', 0.5)
        logger.info(f'Using threshold of {threshold}')
        
        pred = feat.select(['id', 'date']).hstack(
            pred_df.with_columns(
                pl.when(pl.col('prob_YES') > threshold)
                .then(1)
                .otherwise(0)
                .alias('prediction')
            )
        )
        
        logger.info(f"Prediction finished in {time.time() - _start:.2f}s")
        return pred
    
    def save_features(self, features: pl.DataFrame, output_path: str) -> None:
        """
        Save extracted features to a file.
        
        Args:
            features: DataFrame with features
            output_path: Path to save the features
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Determine file format based on extension
        if output_path.endswith('.csv'):
            features.write_csv(output_path)
        elif output_path.endswith('.parquet'):
            features.write_parquet(output_path)
        else:
            # Default to CSV
            features.write_csv(output_path)
            
        logger.info(f"Features saved to {output_path}")
    
    def load_features(self, input_path: str) -> pl.DataFrame:
        """
        Load previously saved features.
        
        Args:
            input_path: Path to the features file
            
        Returns:
            DataFrame with features
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Features file not found at {input_path}")
            
        # Determine file format based on extension
        if input_path.endswith('.csv'):
            features = pl.read_csv(input_path)
        elif input_path.endswith('.parquet'):
            features = pl.read_parquet(input_path)
        else:
            # Try CSV as default
            features = pl.read_csv(input_path)
            
        logger.info(f"Loaded features from {input_path} with {features.shape[0]} rows and {features.shape[1]} columns")
        return features