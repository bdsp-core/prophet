from typing import List, Dict
import polars as pl
import logging
from ..base._basemodel import _BaseModel

logger = logging.getLogger(__name__)

class _ModelCreator:
    """Internal class responsible for model creation and management"""
    
    # Primary model registry maps canonical model name to (module_path, class_name)
    _model_registry = {
        "epilepsy": (".models.epilepsy.predictor", "EpilepsyModel"),
        "congestive_heart_failure": (".models.chf.predictor", "CHFModel"),
        "parkinsons_disease": (".models.pd.predictor", "PDModel"),
        "cardiac_arrest": (".models.ca.predictor", "CAModel"),
        "brain_tumor": (".models.brain_tumor.predictor", "BrainTumorModel"),
        "subarachnoid_hemorrhage": (".models.sah.predictor", "SAHModel"),
        "subdural_hematoma": (".models.sdh.predictor", "SDHModel"),
        "traumatic_brain_injury": (".models.tbi.predictor", "TBIModel"),
        "intracranial_hemorrhage": (".models.ich.predictor", "ICHModel"),
        # "ischemic_stroke": (".models.is.predictor", "ISModel"),
        "mild_cognitive_impairment": (".models.is.predictor", "ISModel"),
        # Add more models here following the pattern: "model_name": ("module.path", "ClassName")
    }
    
    # Alias registry maps alternative names to canonical model names
    _model_aliases = {
        "chf": "congestive_heart_failure",
        "pd": "parkinsons_disease",
        "ca": "cardiac_arrest",
        "sah": "subarachnoid_hemorrhage",
        "sdh": "subdural_hematoma",
        "tbi": "traumatic_brain_injury",
        "ich": "intracranial_hemorrhage",
        # "is": "ischemic_stroke",
        # Add more aliases here following the pattern: "alias": "canonical_name"
    }
    
    def __init__(self):
        self._model_instances = {}
        
    def get_model(self, model_name: str) -> _BaseModel:
        """Get or create model instance using canonical name or alias"""
        # Resolve alias to canonical name if necessary
        canonical_name = self._resolve_model_name(model_name)
        
        if canonical_name not in self._model_instances:
            self._model_instances[canonical_name] = self._create_model(canonical_name)
        return self._model_instances[canonical_name]
    
    def _create_model(self, model_name: str):
        """Create a new model instance dynamically using canonical name"""
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
    def _resolve_model_name(cls, model_name: str) -> str:
        """Convert alias to canonical model name if necessary"""
        return cls._model_aliases.get(model_name, model_name)
    
    @classmethod
    def get_data_format(cls, model_name: str) -> Dict:
        """Get required data format for a model using canonical name or alias"""
        canonical_name = cls._resolve_model_name(model_name)
        
        if canonical_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
            
        try:
            # Dynamically import model class
            import importlib
            module_path, class_name = cls._model_registry[canonical_name]
            module = importlib.import_module(module_path, 'prophet')
            model_class = getattr(module, class_name)
            
            # Call static method
            return model_class.get_data_format()
        except Exception as e:
            logger.error(f"Error getting data format for {canonical_name}: {str(e)}")
            raise RuntimeError(f"Failed to get data format for '{canonical_name}'. Error: {str(e)}") from e
    
    @classmethod
    def get_credits(cls, model_name: str) -> Dict:
        """Get credits/attribution information for a model using canonical name or alias"""
        canonical_name = cls._resolve_model_name(model_name)
        
        if canonical_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
            
        try:
            # Dynamically import model class
            import importlib
            module_path, class_name = cls._model_registry[canonical_name]
            module = importlib.import_module(module_path, 'prophet')
            model_class = getattr(module, class_name)
            
            # Call static method
            return model_class.get_credits()
        except Exception as e:
            logger.error(f"Error getting credits for {canonical_name}: {str(e)}")
            raise RuntimeError(f"Failed to get credits for '{canonical_name}'. Error: {str(e)}") from e
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of all available canonical model names"""
        return list(cls._model_registry.keys())
    
    @classmethod
    def get_all_model_names(cls) -> List[str]:
        """Get list of all available model names including aliases"""
        return list(cls._model_registry.keys()) + list(cls._model_aliases.keys())
    
    @classmethod
    def is_valid_model(cls, model_name: str) -> bool:
        """Check if a model name (canonical or alias) is valid"""
        canonical_name = cls._resolve_model_name(model_name)
        return canonical_name in cls._model_registry
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, str]:
        """Get information about a model from the registry using canonical name or alias"""
        canonical_name = cls._resolve_model_name(model_name)
        
        if canonical_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}")
            
        module_path, class_name = cls._model_registry[canonical_name]
        info = {
            "name": canonical_name,
            "module_path": module_path,
            "class_name": class_name
        }
        
        # Add alias information if this is a canonical name with aliases
        aliases = [alias for alias, canon in cls._model_aliases.items() if canon == canonical_name]
        if aliases:
            info["aliases"] = aliases
        
        # Add canonical name if this was looked up via an alias
        if model_name != canonical_name:
            info["canonical_name"] = canonical_name
            info["is_alias"] = True
        
        return info
    
    @classmethod
    def register_model(cls, model_name: str, module_path: str, class_name: str, aliases: List[str] = None):
        """Register a new model in the registry with optional aliases"""
        cls._model_registry[model_name] = (module_path, class_name)
        logger.info(f"Registered model: {model_name} -> {module_path}.{class_name}")
        
        # Register any aliases
        if aliases:
            for alias in aliases:
                cls._model_aliases[alias] = model_name
                logger.info(f"Registered alias: {alias} -> {model_name}")
    
    @classmethod
    def register_alias(cls, alias: str, canonical_name: str):
        """Register a new alias for an existing model"""
        if canonical_name not in cls._model_registry:
            raise ValueError(f"Cannot create alias for unknown model: {canonical_name}")
        
        cls._model_aliases[alias] = canonical_name
        logger.info(f"Registered alias: {alias} -> {canonical_name}")
    
    @classmethod
    def get_compatible_models(cls, data_dict: Dict[str, pl.DataFrame]) -> List[str]:
        """Find models compatible with the provided data (returns canonical names only)"""
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