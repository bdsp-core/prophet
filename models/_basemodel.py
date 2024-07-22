import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict

class _BaseModel(ABC):
    @abstractmethod
    def preprocess_data(self, data) -> Dict[str, pd.DataFrame]:
        '''Check that data is in the correct format, removing any unnecessary or empty columns/rows, and converting to proper types.'''
        pass

    @abstractmethod
    def generate_features(self, data : Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''Create the feature matrix. Returns tuple(identifier_df, feature_matrix_df).\n\nThe rows of the identifier dataframe and feature matrix dataframe should align.'''
        pass

    @abstractmethod
    def predict(self, features : pd.DataFrame) -> pd.DataFrame:
        '''Get predictions from the model using the feature matrix. This will be pd.concat with the identifier matrix.'''
        pass

    def run(self, data):
        # TODO: save intermediate steps
        preprocessed_data = self.preprocess_data(data)
        identifier, features = self.generate_features(preprocessed_data)
        predictions = self.predict(features)
        return pd.concat([identifier, predictions], axis=1)
