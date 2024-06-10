from ..basemodel import BaseModel
from .code.build_feat_mat import build_feat_mat
# from .code.LR_bootstrap import LR_bootstrap
import os
import pandas as pd
import nltk
import logging

logger = logging.getLogger(__name__)

class CHFModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        logger.info("Initializing CHFModel")
        
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
    
    def predict(self, data, save_dir=None):
        feat_mat = build_feat_mat(data, self.vocab)
        # TODO: predict using chf model

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            feat_mat.to_csv(os.path.join(save_dir, 'chf_feat_mat.csv'))

        return feat_mat

