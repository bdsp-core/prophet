from typing import Dict, Optional, List
import os
import json
import subprocess
import pandas as pd
import logging
from models.chf.chf_model import CHFModel
# from models.epilepsy import EpilepsyModel
# from models.mcid import MCIDModel


logging = logging.getLogger(__name__)

class EHRModel:
    def __init__(self, model_name: str):
        with open('models.json', 'r') as f:
            data = json.load(f)

        if model_name not in data:
            raise ValueError(f"Invalid model: {model_name}. Valid models are: {list(data.keys())}")
        
        for name, v in data[model_name].items():
            setattr(self, name, v)

        self.model_name = model_name
        self.model = self.__load_model()

    def __load_model(self):
        if self.model_name == 'congestive heart failure':
            model = CHFModel(self.config)
            return model
        # elif self.model_name == 'epilepsy':
        #     return EpilepsyModel(self.config)
        # elif self.model_name == 'mcid':
        #     return MCIDModel(self.config)
        else:
            raise ValueError(f"Invalid model: {self.model_name}.")


    def __validate_data(self, data: Dict[str, pd.DataFrame]):
        for name in data:
            if name not in self.data_format:
                raise ValueError(f"Invalid data: {name}. Valid data are: {list(self.data_format.keys())}")
            
        for name, column_names in self.data_format.items():
            if name not in data:
                raise ValueError(f"Missing data: {name}")

            for column_name in column_names:
                if column_name not in data[name]:
                    raise ValueError(f"Missing column: '{column_name}' in {name}")
            
                
    # def __validate_env(self):
    #     if self.environment is None:
    #         raise ValueError("Environment is not set")
        
    #     if self.environment['type'] == 'conda':
    #         if not os.path.exists(self.environment['path']):
    #             raise ValueError(f"Environment not found: {self.environment['path']}")
            
    #     elif self.environment['type'] == 'venv':
    #         if not os.path.exists(self.environment['path']):
    #             raise ValueError(f"Environment not found: {self.environment['path']}")
        

    def get_data_format(self):
        if self.data_format is None:
            raise ValueError("Data format is not set")
        return self.data_format

    def predict(self, data: Optional[Dict[str, pd.DataFrame]] = None, save_dir = None, **kwargs: pd.DataFrame):
        if data is None:
            data = kwargs
        else:
            data.update(kwargs)
        
        self.__validate_data(data)
        return self.model.predict(data)



        


