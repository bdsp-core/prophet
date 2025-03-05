import sys
import os
import pandas as pd
from typing import Dict, Optional, List
# TODO: enforce typing via pydantic?

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models._modelcreator import _ModelCreator

class EHRPredict:
    def __init__(self) -> None:
        self.model_creator = _ModelCreator()

    @staticmethod
    def get_data_format(phenotype : str) -> Dict:
        return _ModelCreator.get_data_format(phenotype)
    
    @staticmethod
    def get_credits(phenotype : str) -> Dict:
        return _ModelCreator.get_credits(phenotype)
    
    @staticmethod
    def get_models() -> List[str]:
        return _ModelCreator.get_models()

    def predict(self, phenotype : str, data: Optional[Dict[str, pd.DataFrame]]=None, **kwargs):
        req_dfs = _ModelCreator.get_data_format(phenotype).keys()
        
        if data is None:
            data = {}
        all_dfs = {**data, **kwargs}

        for df in req_dfs:
            if df not in all_dfs:
                raise ValueError(f"Required dataframe {df} is missing from input data")
        
        model = self.model_creator.get_model(phenotype)
        return model.run(all_dfs)