from models.epilepsy.predictor import EpilepsyModel
from typing import Dict

class _ModelCreator:
    @staticmethod
    def get_model(model_name):
        if model_name == "congestive heart failure":
            # return CHFModel()
            pass
        elif model_name == "epilepsy" or model_name == "epilepsy_m":
            return EpilepsyModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_data_format(model_name) -> Dict:
        if model_name == 'congestive heart failure':
            # return CHFModel.get_data_format()
            pass
        elif model_name == 'epilepsy':
            return EpilepsyModel.get_data_format()
        else:
            raise ValueError(f'Unknown model: {model_name}')
        
    @staticmethod
    def get_credits(model_name) -> Dict:
        if model_name == 'congestive heart failure':
            pass
        elif model_name == 'epilepsy':
            return EpilepsyModel.get_credits()
        else:
            raise ValueError(f'Unknown model: {model_name}')