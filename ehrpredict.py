import sys
import os
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models._modelcreator import _ModelCreator

class EHRPredict:
    def __init__(self) -> None:
        self.model_creator = _ModelCreator()

    @staticmethod
    def get_data_format(phenotype : str) -> Dict:
        return _ModelCreator.get_data_format(phenotype)

    def predict(self, phenotype : str, data):
        model = self.model_creator.get_model(phenotype)
        return model.run(data)