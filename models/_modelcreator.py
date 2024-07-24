# from models.epilepsy_m._epilepsymodel import _EpilepsyModel
from models.chf.chfmodel import CHFModel
from typing import Dict

class _ModelCreator:
    @staticmethod
    def get_model(model_name):
        if model_name == "congestive heart failure":
            return CHFModel()
        elif model_name == "epilepsy" or model_name == "epilepsy_m":
            raise NotImplementedError('not implemented!')
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_data_format(model_name) -> Dict:
        if model_name == 'congestive heart failure':
            return CHFModel.get_data_format()
        elif model_name == 'epilepsy':
            raise NotImplementedError('not implemented!')
        else:
            raise ValueError(f'Unknown model: {model_name}')