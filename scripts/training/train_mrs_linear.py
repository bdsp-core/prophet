"""
Training script for mRS (Modified Rankin Score) linear model.
Task: predict the continuous mRS score (0-6) directly, rather than the
binary good/poor split handled by train_mrs.py.

Pipeline (identical feature pipeline to the binary mRS model, different
target/estimator):
  1. Preprocess notes (same as binary mRS: abbreviation expansion, NIHSS n2w,
     stopwords, Porter stemming) via the shared prophet.models.mrs preprocessing
  2. Fit CountVectorizer (1-3 n-grams, binary) on training notes
  3. Sparsity filter (<90% zeros)
  4. Train Lasso regression (patient-level 5-fold CV for alpha), predictions
     clipped to [0, 6]
  5. Save vectorizer + model

Dataset: 5337 patients, 42672 notes. Notes from the same patient on the same day
are concatenated before feature extraction to capture full daily clinical picture.
"""
import sys
import os

# Add prophet source to path for imports
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import polars as pl
import numpy as np
import joblib
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr
from nltk.stem import PorterStemmer
import nltk

# Import preprocessing from the prophet predictor (single source of truth,
# shared with the binary mRS model — same clinical preprocessing pipeline)
from prophet.models.mrs.predictor import _preprocess_text as preprocess_text

DATA_DIR = os.path.join(_ROOT, 'dataset_unused', 'modified_rankin_score')
OUTPUT_DIR = os.path.join(_ROOT, 'src', 'prophet', 'models', 'mrs_linear')

# One stemmer per worker process (PorterStemmer isn't expensive to construct,
# but this avoids re-creating it per note).
_worker_stemmer = None


def _init_worker():
    global _worker_stemmer
    _worker_stemmer = PorterStemmer()


def _preprocess_one(text):
    return preprocess_text(str(text), _worker_stemmer)


def load_data():
    note = pl.read_parquet(os.path.join(DATA_DIR, 'note.parquet'))
    annot = pl.read_parquet(os.path.join(DATA_DIR, 'annot.parquet'))
    # annot's key is 'note_index' (a per-note row index shared with note.parquet),
    # not 'id'+'date' — join on that instead.
    df = note.join(annot.select(['note_index', 'annot']), on='note_index', how='inner')
    # Concatenate notes from the same patient on the same day before feature extraction
    df = (
        df.group_by(['id', 'date'])
        .agg([
            pl.col('note').str.concat(' ').alias('note'),
            pl.col('annot').first().alias('annot'),
        ])
        .sort(['id', 'date'])
    )
    return df.to_pandas()


def evaluate(y_true, y_pred, label=''):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, pval = spearmanr(y_true, y_pred)
    print(f"{label}  RMSE={rmse:.3f}  Spearman={rho:.3f} (p={pval:.2e})")
    return rmse, rho


def main():
    nltk.download('punkt_tab', quiet=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} notes, {df['id'].nunique()} patients")
    print(f"  mRS distribution:\n{df['annot'].value_counts().sort_index()}")

    y = df['annot'].values.astype(float)

    n_workers = max(1, cpu_count() - 1)
    print(f"\nPreprocessing notes across {n_workers} processes...")
    texts = df['note'].tolist()
    processed = [None] * len(texts)
    with Pool(n_workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.imap(_preprocess_one, texts, chunksize=200)):
            processed[i] = result
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(df)}")
    df['processed'] = processed
    print("  Done.")

    # Patient-level 5-fold CV
    groups = df['id'].values
    gkf = GroupKFold(n_splits=5)

    print("\nTuning alpha via patient-level 5-fold CV...")
    alphas = [0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0]

    # Vectorization/variance-filtering doesn't depend on alpha, so fit it once
    # per fold and reuse across all alpha values instead of refitting per alpha.
    rmse_by_alpha = {alpha: [] for alpha in alphas}
    for train_idx, test_idx in gkf.split(df, y, groups):
        X_train_raw = df['processed'].iloc[train_idx].values
        X_test_raw = df['processed'].iloc[test_idx].values
        y_train, y_test = y[train_idx], y[test_idx]

        vec = CountVectorizer(ngram_range=(1, 3), binary=True, min_df=5)
        X_train = vec.fit_transform(X_train_raw)
        X_test = vec.transform(X_test_raw)

        # Sparsity filter
        vt = VarianceThreshold(threshold=0.1 * 0.9)  # ~90% sparse cutoff
        X_train_f = vt.fit_transform(X_train)
        X_test_f = vt.transform(X_test)

        for alpha in alphas:
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_train_f, y_train)
            pred = np.clip(model.predict(X_test_f), 0, 6)
            rmse_by_alpha[alpha].append(np.sqrt(mean_squared_error(y_test, pred)))

    best_alpha, best_rmse = None, np.inf
    for alpha in alphas:
        mean_rmse = np.mean(rmse_by_alpha[alpha])
        print(f"  alpha={alpha}  CV-RMSE={mean_rmse:.4f}")
        if mean_rmse < best_rmse:
            best_rmse, best_alpha = mean_rmse, alpha

    print(f"\nBest alpha: {best_alpha}  (CV-RMSE={best_rmse:.4f})")

    # Hold-out evaluation on last fold
    splits = list(gkf.split(df, y, groups))
    train_idx, test_idx = splits[-1]
    X_train_raw = df['processed'].iloc[train_idx].values
    X_test_raw = df['processed'].iloc[test_idx].values
    y_train, y_test = y[train_idx], y[test_idx]

    vec_eval = CountVectorizer(ngram_range=(1, 3), binary=True, min_df=5)
    X_train_e = vec_eval.fit_transform(X_train_raw)
    X_test_e = vec_eval.transform(X_test_raw)
    vt_eval = VarianceThreshold(threshold=0.1 * 0.9)
    X_train_ef = vt_eval.fit_transform(X_train_e)
    X_test_ef = vt_eval.transform(X_test_e)

    model_eval = Lasso(alpha=best_alpha, max_iter=10000)
    model_eval.fit(X_train_ef, y_train)
    pred_test = np.clip(model_eval.predict(X_test_ef), 0, 6)
    evaluate(y_test, pred_test, label='Hold-out fold')

    # Train final model on all data
    print("\nTraining final model on all data...")
    vec_final = CountVectorizer(ngram_range=(1, 3), binary=True, min_df=5)
    X_all = vec_final.fit_transform(df['processed'].values)
    vt_final = VarianceThreshold(threshold=0.1 * 0.9)
    X_all_f = vt_final.fit_transform(X_all)

    model_final = Lasso(alpha=best_alpha, max_iter=10000)
    model_final.fit(X_all_f, y)
    pred_all = np.clip(model_final.predict(X_all_f), 0, 6)
    evaluate(y, pred_all, label='All-data (train)')
    nz = (model_final.coef_ != 0).sum()
    print(f"  Non-zero coefficients: {nz} / {X_all_f.shape[1]}")
    print(f"  Vocabulary size: {len(vec_final.vocabulary_)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model_final, os.path.join(OUTPUT_DIR, 'mrs_linear_model.joblib'))
    joblib.dump(vec_final, os.path.join(OUTPUT_DIR, 'mrs_linear_vectorizer.joblib'))
    joblib.dump(vt_final, os.path.join(OUTPUT_DIR, 'mrs_linear_variance_filter.joblib'))
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"Best alpha for config: {best_alpha}")


if __name__ == '__main__':
    main()
