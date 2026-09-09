"""
Training script for NIHSS (NIH Stroke Scale) model.
Uses pre-extracted binary n-gram features from prophet_dataset.

Dataset: 155 patients, 4294 annotated notes, 266 binary stemmed n-gram features.
Model: Lasso regression (alpha tuned via patient-level 5-fold CV).
Metrics: RMSE, Spearman correlation.
"""
import sys
import os

# Add prophet source to path for imports
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import polars as pl
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Lasso
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Import feature list from the prophet predictor (single source of truth)
from prophet.models.nihss.predictor import NIHSS_FEATURES as FEATURE_COLS

DATA_DIR = '/home/niels/cdac Dropbox/Niels Turley/prophet_dataset/nih_stroke_scale'
OUTPUT_DIR = os.path.join(_ROOT, 'src', 'prophet', 'models', 'nihss')


def load_data():
    feat = pl.read_parquet(os.path.join(DATA_DIR, 'feat.parquet'))
    annot = pl.read_parquet(os.path.join(DATA_DIR, 'annot.parquet'))
    note = pl.read_parquet(os.path.join(DATA_DIR, 'note.parquet')).with_row_index()

    df = (
        feat
        .join(annot.select(['feat_index', 'annot']), on='feat_index', how='left')
        .join(note.select(['index', 'id']).rename({'index': 'note_index'}),
              left_on='feat_index', right_on='note_index', how='left')
    )
    return df.to_pandas()


def evaluate(y_true, y_pred, label=''):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, pval = spearmanr(y_true, y_pred)
    print(f"{label}  RMSE={rmse:.3f}  Spearman={rho:.3f} (p={pval:.2e})")
    return rmse, rho


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} notes, {df['id'].nunique()} patients")
    print(f"  NIHSS range: {df['annot'].min()}-{df['annot'].max()}, mean={df['annot'].mean():.1f}")

    X = df[FEATURE_COLS].values
    y = df['annot'].values
    groups = df['id'].values  # patient IDs for group CV

    # Patient-level 5-fold CV to find best alpha
    print("\nTuning alpha via patient-level 5-fold CV...")
    alphas = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.5, 1.0]
    gkf = GroupKFold(n_splits=5)

    best_alpha, best_rmse = None, np.inf
    for alpha in alphas:
        fold_rmses = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X[train_idx], y[train_idx])
            pred = np.clip(model.predict(X[test_idx]), 0, 42)
            fold_rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        mean_rmse = np.mean(fold_rmses)
        print(f"  alpha={alpha:.3f}  CV-RMSE={mean_rmse:.3f}")
        if mean_rmse < best_rmse:
            best_rmse, best_alpha = mean_rmse, alpha

    print(f"\nBest alpha: {best_alpha}  (CV-RMSE={best_rmse:.3f})")

    # Hold-out evaluation on last fold for reporting
    splits = list(gkf.split(X, y, groups))
    train_idx, test_idx = splits[-1]
    model_eval = Lasso(alpha=best_alpha, max_iter=10000)
    model_eval.fit(X[train_idx], y[train_idx])
    pred_test = np.clip(model_eval.predict(X[test_idx]), 0, 42)
    evaluate(y[test_idx], pred_test, label='Hold-out fold')

    # Train final model on ALL data
    print("\nTraining final model on all data...")
    final_model = Lasso(alpha=best_alpha, max_iter=10000)
    final_model.fit(X, y)
    pred_all = np.clip(final_model.predict(X), 0, 42)
    evaluate(y, pred_all, label='All-data (train)')
    print(f"  Non-zero coefficients: {(final_model.coef_ != 0).sum()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, 'nihss_model.joblib')
    joblib.dump(final_model, model_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Best alpha for config: {best_alpha}")


if __name__ == '__main__':
    main()
