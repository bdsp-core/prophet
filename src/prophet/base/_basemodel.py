from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import polars as pl
import yaml
from ..utils.preprocessing import validate_dataframe
import warnings
import importlib.resources

# Column names that the base class reserves for its own output.
# Input DataFrames must not use these names.
_RESERVED_OUTPUT_COLS = frozenset({'note_idx'})


class _BaseModel(ABC):
    DEFAULT_CONFIG_PATH = None

    # Define the custom YAML loader as a class attribute
    class PolarsSafeLoader(yaml.SafeLoader):
        pass

    def __init_subclass__(cls, **kwargs):
        """Wrap every subclass preprocess() to append note_idx to its output."""
        super().__init_subclass__(**kwargs)
        if 'preprocess' in cls.__dict__:
            _orig = cls.__dict__['preprocess']

            def _preprocess_with_note_idx(self, data, *args, **kwargs):
                feat = _orig(self, data, *args, **kwargs)
                # Only act on the feature DataFrame returned by the subclass.
                # The base-class preprocess() returns a dict; the subclass's
                # outer call returns a pl.DataFrame — that's what we annotate.
                if isinstance(feat, pl.DataFrame) and 'note_idx' not in feat.columns:
                    feat = feat.with_row_index('note_idx')
                return feat

            cls.preprocess = _preprocess_with_note_idx
    
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
    
    def __init__(self, config_path=None):
        """Initialize with config path"""
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = {}

        # Store path for lazy loading — model is not loaded until first prediction
        self._model_path = self.config.get('model_path')
        self._model = None

    @property
    def model(self):
        """Load the model on first access so feature extraction works without loading the model."""
        if self._model is None and self._model_path:
            self._model = self.load_model(self._model_path)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @abstractmethod
    def load_model(self, model_path : Optional[str] = None):
        """Load the model from a given path"""
        pass

    @classmethod
    def get_data_format(cls, *args, config_path=None):
        """
        Dual method to get default data format schema/instantiated schema
        
        Args:
            config_path: Path to the config file, defaults to class's DEFAULT_CONFIG_PATH
            
        Returns:
            Dictionary containing the schema information
        """
        # Check if called on an instance
        if args and isinstance(args[0], cls):
            instance = args[0]
            if instance.config:
                return instance.config.get('schema', {})
        
        # If not called on instance or instance has no config
        if config_path is None:
            if cls.DEFAULT_CONFIG_PATH is None:
                raise ValueError(f"No default config path defined for {cls.__name__}")
            config_path = cls.DEFAULT_CONFIG_PATH
            
        config = cls.load_config(config_path)
        return config.get('schema', {})
    
    @classmethod
    def get_credits(cls, config_path=None):
        """
        Class method to get default credits information
        
        Args:
            config_path: Path to the config file, defaults to class's DEFAULT_CONFIG_PATH
            
        Returns:
            Dictionary containing the schema information
        """
        if config_path is None:
            if cls.DEFAULT_CONFIG_PATH is None:
                raise ValueError(f"No default config path defined for {cls.__name__}")
            config_path = cls.DEFAULT_CONFIG_PATH
            
        config = cls.load_config(config_path)
        return config.get('credits', {})
    
    @classmethod
    def get_paper_link(cls, config_path=None):
        """
        Class method to get default paper link information
        
        Args:
            config_path: Path to the config file, defaults to class's DEFAULT_CONFIG_PATH
            
        Returns:
            Dictionary containing the schema information
        """
        if config_path is None:
            if cls.DEFAULT_CONFIG_PATH is None:
                raise ValueError(f"No default config path defined for {cls.__name__}")
            config_path = cls.DEFAULT_CONFIG_PATH
            
        config = cls.load_config(config_path)
        return config.get('paper_link', {})

    def preprocess(self, data, show_progress=False, force_casting=False) -> Dict[str, pl.DataFrame]:
        '''Check that data is in the correct format, removing any unnecessary or empty columns/rows, and converting to proper types.'''
        # Reject reserved column names in every input DataFrame before doing
        # anything else, so the error is unambiguous.
        for df_name, df in data.items():
            conflicts = _RESERVED_OUTPUT_COLS & set(df.columns)
            if conflicts:
                raise ValueError(
                    f"Input dataframe '{df_name}' contains column(s) {sorted(conflicts)} "
                    f"that are reserved for model output. Rename them before calling the model."
                )

        data_format = self.get_data_format()
        if data_format:
            new_data = {}
            for name, schema in data_format.items():
                if name in data:
                    if force_casting:
                        new_data[name] = validate_dataframe(data[name], schema, force_casting, name)
                    else:
                        validate_dataframe(data[name], schema, force_casting, name)
                        new_data[name] = data[name].clone()
                else:
                    raise ValueError(f"Missing required dataframe: {name}")
        else:
            warnings.warn('No schema provided in config. Skipping validation.')
        return new_data

    @abstractmethod
    def run(self, data: Dict[str, pl.DataFrame], show_progress_bar, return_features) -> pl.DataFrame:
        '''Get predictions from the model using the data.'''
        pass


