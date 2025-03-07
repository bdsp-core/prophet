from abc import ABC, abstractmethod
from typing import Dict, List, Any
import polars as pl
import yaml
from utils.preprocessing import validate_dataframe
import warnings

class _BaseModel(ABC):
    DEFAULT_CONFIG_PATH = None

    # Define the custom YAML loader as a class attribute
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
    
    def __init__(self, config_path=None):
        """Initialize with config path"""
        if config_path:
            self.config = self.load_config(config_path)
        else:
            self.config = {}
        
        # Load model path from config
        model_path = self.config.get('model_path')
        self.model = None
        if model_path:
            self.model = self.load_model(model_path)

    @abstractmethod
    def load_model(self, model_path : str):
        """Load the model from a given path"""
        pass

    @classmethod
    def get_data_format(cls, config_path=None):
        """
        Class method to get default data format schema
        
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

    def preprocess(self, data, force_casting = False) -> Dict[str, pl.DataFrame]:
        '''Check that data is in the correct format, removing any unnecessary or empty columns/rows, and converting to proper types.'''
        data_format = self.get_data_format()
        if data_format:
            for name, schema in data_format.items():
                if name in data:
                    if force_casting:
                        data[name] = validate_dataframe(data[name], schema, force_casting)
                    else:
                        validate_dataframe(data[name], schema)
                else:
                    raise ValueError(f"Missing required dataframe: {name}")
        else:
            warnings.warn('No schema provided in config. Skipping validation.')
        return data

    @abstractmethod
    def run(self, data: Dict[str, pl.DataFrame], show_progress_bar, return_features) -> pl.DataFrame:
        '''Get predictions from the model using the data.'''
        pass


