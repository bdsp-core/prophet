# from epilepsy.epilepsymodel import EpilepsyModel
from models.chf._chfmodel import _CHFModel
from typing import Dict

class _ModelCreator:
    @staticmethod
    def get_model(model_name):
        if model_name == "congestive heart failure":
            return _CHFModel()
        elif model_name == "epilepsy":
            raise NotImplementedError('not implemented!')
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @staticmethod
    def get_data_format(model_name) -> Dict:
        if model_name == 'congestive heart failure':
            return _CHFModel.get_data_format()
        elif model_name == 'epilepsy':
            raise NotImplementedError('not implemented!')
        else:
            raise ValueError(f'Unknown model: {model_name}')