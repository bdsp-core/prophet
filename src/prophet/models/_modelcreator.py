from typing import List, Dict
import polars as pl
import logging
from ..base._basemodel import _BaseModel

logger = logging.getLogger(__name__)

class _ModelCreator:
    """Internal class responsible for model creation and management"""
    
    # Model registry maps model name to (module_path, class_name)
    _model_registry = {
        "epilepsy": (".models.epilepsy.predictor", "EpilepsyModel"),
        "congestive_heart_failure": (".models.chf.predictor", "CHFModel"),
        # Add more models here following the pattern: "model_name": ("module.path", "ClassName")
    }
    
    def __init__(self):
        self._model_instances = {}
        
    def get_model(self, model_name: str) -> _BaseModel:
        """Get or create model instance"""
        if model_name not in self._model_instances:
            self._model_instances[model_name] = self._create_model(model_name)
        return self._model_instances[model_name]
    
    def _create_model(self, model_name: str):
        """Create a new model instance dynamically"""
        if model_name not in self._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Get module and class information from registry
        module_path, class_name = self._model_registry[model_name]
        
        try:
            # Dynamically import the module and class
            import importlib
            module = importlib.import_module(module_path, 'prophet')
            model_class = getattr(module, class_name)
            
            # Instantiate the model
            return model_class()
        except ImportError as e:
            # Module could not be imported
            logger.error(f"Failed to import module {module_path} for model {model_name}: {str(e)}")
            raise ModuleNotFoundError(f"The module '{module_path}' for model '{model_name}' could not be found. " 
                                     f"Please ensure the module is installed and in your Python path.") from e
        except AttributeError as e:
            # Class not found in module
            logger.error(f"Class {class_name} not found in module {module_path}: {str(e)}")
            raise AttributeError(f"The class '{class_name}' was not found in module '{module_path}'. "
                               f"Please verify the class name in the model registry.") from e
        except Exception as e:
            # Other initialization errors
            logger.error(f"Error creating model {model_name}: {str(e)}")
            if "not implemented yet" in str(e).lower():
                raise NotImplementedError(f"The model '{model_name}' is registered but not fully implemented yet.") from e
            raise RuntimeError(f"Failed to initialize model '{model_name}'. Error: {str(e)}") from e
    
    @classmethod
    def get_data_format(cls, model_name: str) -> Dict:
        """Get required data format for a model"""
        if model_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
            
        try:
            # Dynamically import model class
            import importlib
            module_path, class_name = cls._model_registry[model_name]
            module = importlib.import_module(module_path, 'prophet')
            model_class = getattr(module, class_name)
            
            # Call static method
            return model_class.get_data_format()
        except Exception as e:
            logger.error(f"Error getting data format for {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to get data format for '{model_name}'. Error: {str(e)}") from e
    
    @classmethod
    def get_credits(cls, model_name: str) -> Dict:
        """Get credits/attribution information for a model"""
        if model_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
            
        try:
            # Dynamically import model class
            import importlib
            module_path, class_name = cls._model_registry[model_name]
            module = importlib.import_module(module_path, 'prophet')
            model_class = getattr(module, class_name)
            
            # Call static method
            return model_class.get_credits()
        except Exception as e:
            logger.error(f"Error getting data format for {model_name}: {str(e)}")
            raise RuntimeError(f"Failed to get data format for '{model_name}'. Error: {str(e)}") from e
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available models"""
        return list(cls._model_registry.keys())
    
    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """Check if a model name is valid"""
        return model_name in cls._model_registry
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, str]:
        """Get information about a model from the registry"""
        if model_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
            
        module_path, class_name = cls._model_registry[model_name]
        return {
            "name": model_name,
            "module_path": module_path,
            "class_name": class_name
        }
    
    @classmethod
    def register_model(cls, model_name: str, module_path: str, class_name: str):
        """Register a new model in the registry"""
        cls._model_registry[model_name] = (module_path, class_name)
        logger.info(f"Registered model: {model_name} -> {module_path}.{class_name}")
    
    @classmethod
    def get_compatible_models(cls, data_dict: Dict[str, pl.DataFrame]) -> List[str]:
        """Find models compatible with the provided data"""
        compatible_models = []
        
        # Get available dataframes
        available_dfs = set(data_dict.keys())
        
        for model_name in cls.get_available_models():
            try:
                # Get required dataframes for this model
                required_dfs = set(cls.get_data_format(model_name).keys())
                
                # Check if all required dataframes are available
                if required_dfs.issubset(available_dfs):
                    compatible_models.append(model_name)
            except (ValueError, NotImplementedError, ImportError):
                # Skip models that raise errors
                logger.debug(f"Skipping incompatible model: {model_name}")
                continue
                
        return compatible_models
