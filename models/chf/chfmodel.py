'''
Model/code from: Daniel Sumsion
Additional code help from: Niels Turley
'''

from .._basemodel import _BaseModel
from .code.build_feat_mat import build_feat_mat
import os
import pandas as pd
import nltk
import logging
import json
import joblib
from typing import Dict

logger = logging.getLogger(__name__)

class CHFModel(_BaseModel):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', 'data_format.json')) as f:
        _data_format : Dict = json.load(f)

    def __init__(self):
        logger.info("Initializing model")
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.vocab = {
            'icd_codes': pd.read_csv(os.path.join(current_dir, 'files', 'ICD_CHF.txt'))['ICD_cd'].astype(str).tolist(),
            'medications': pd.read_csv(os.path.join(current_dir, 'files', 'MED_CHF.txt'))['Med_name'].astype(str).tolist(),
            'notes': pd.read_csv(os.path.join(current_dir, 'files', 'KW_CHF_BoW.txt'))['keywords'].astype(str).tolist()
        }

        self.model = joblib.load(os.path.join(current_dir, 'files','LR_icd_meds_notes_model.joblib'))

        logger.info('Finished initializing model')

    @classmethod
    def get_data_format(cls):
        return cls._data_format

    def _check_keys(self, data, expected_keys):
        unknown_keys = set(data.keys()) - set(expected_keys)
        missing_keys = set(expected_keys) - set(data.keys())
        return unknown_keys, missing_keys

    def _check_columns(self, df, expected_columns):
        unknown_columns = set(df.columns) - set(expected_columns)
        missing_columns = set(expected_columns) - set(df.columns)
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        return unknown_columns, missing_columns, duplicate_columns

    def preprocess_data(self, data: Dict[str, pd.DataFrame]):
        logger.info('Preprocessing data')

        # Check overall data format
        unknown_keys, missing_keys = self._check_keys(data, self._data_format.keys())
        if unknown_keys:
            for k in unknown_keys:
                logger.warning(f'Ignoring unknown key in data: {k}')
                del data[k]
        if missing_keys:
            raise ValueError(f"Missing required data: {', '.join(missing_keys)}")

        # Check each dataframe
        for k, df in data.items():
            unknown_cols, missing_cols, duplicate_cols = self._check_columns(df, self._data_format[k])
            
            if unknown_cols:
                logger.warning(f"Ignoring unknown columns in {k}: {', '.join(unknown_cols)}")
                data[k] = df.drop(columns=unknown_cols)
            
            if missing_cols:
                raise ValueError(f"{k} is missing required columns: {', '.join(missing_cols)}")
            
            if duplicate_cols:
                raise ValueError(f"{k} has duplicate columns: {', '.join(duplicate_cols)}")


        # Data format is correct, now clean up the dataframes
        meds = data['medications'].reset_index(drop=True)
        meds['date_med_end'] = meds['date_med_end'].fillna(meds['date_med_start']) # set end to start if not available
        icds = data['icd_codes'].reset_index(drop=True)
        notes = data['notes'].reset_index(drop=True)

        for df in [meds, icds, notes]:
            df['bdsp_patient_id'] = df['bdsp_patient_id'].astype(float)

        # Filter out meds / icds we don't care about
        meds = meds[meds['med'].str.contains('|'.join(self.vocab['medications']), na=False, case=False)]
        icds = icds[icds['icd'].str.contains('|'.join(self.vocab['icd_codes']), na=False, case=False)]

        for df, date_cols in [
            (meds, ['date_med_start', 'date_med_end']),
            (icds, ['date_icd']),
            (notes, ['date_note'])
        ]:
            df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='coerce') # ? change to raise

        meds = meds.dropna().drop_duplicates(ignore_index=True)
        icds = icds.dropna().drop_duplicates(ignore_index=True)
        notes = notes.dropna().drop_duplicates(ignore_index=True)

        return {
            'medications': meds,
            'icd_codes': icds,
            'notes': notes
        }
    
    def generate_features(self, data):
        logger.info('Generating features')
        return build_feat_mat(data, self.vocab)

    def predict(self, features):
        logger.info('Getting model predictions')
        y_pred = self.model.predict(features)
        y_prob = self.model.predict_proba(features)
        predictions = pd.DataFrame(y_prob, columns=['prob_chf_0', 'prob_chf_1'])
        predictions['predict_class_chf'] = y_pred
        return predictions

