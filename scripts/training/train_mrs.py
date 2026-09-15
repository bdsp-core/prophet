"""
Training script for mRS (Modified Rankin Score) binary model.
Task: predict good (mRS 0-2) vs poor (mRS 3-6) outcome.

Pipeline:
  1. Preprocess notes (same as NIHSS: abbreviation expansion, NIHSS n2w, stopwords, Porter stemming)
  2. Fit CountVectorizer (1-3 n-grams, binary) on training notes
  3. Sparsity filter (<90% zeros)
  4. Train LogisticRegression (L1, patient-level 5-fold CV for C)
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
import pandas as pd
import numpy as np
import joblib
import gc
from multiprocessing import Pool, cpu_count
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.feature_selection import VarianceThreshold
from nltk.stem import PorterStemmer
import nltk

# Import preprocessing from the prophet predictor (single source of truth)
from prophet.models.mrs.predictor import _preprocess_text as preprocess_text

DATA_DIR = os.path.join(_ROOT, 'dataset_unused', 'modified_rankin_score')
OUTPUT_DIR = os.path.join(_ROOT, 'src', 'prophet', 'models', 'mrs')

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


def main():
    nltk.download('punkt_tab', quiet=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} notes, {df['id'].nunique()} patients")
    print(f"  mRS distribution:\n{df['annot'].value_counts().sort_index()}")

    # Binary label: 0 = good (mRS 0-2), 1 = poor (mRS 3-6)
    df['label'] = (df['annot'] >= 3).astype(int)
    print(f"  Binary: {df['label'].sum()} poor ({df['label'].mean():.1%}), {(df['label']==0).sum()} good")

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
    y = df['label'].values
    gkf = GroupKFold(n_splits=5)

    print("\nTuning C via patient-level 5-fold CV...")
    Cs = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    # Vectorization/variance-filtering doesn't depend on C, so fit it once per
    # fold and reuse across all C values instead of refitting 7x per fold.
    auc_by_C = {C: [] for C in Cs}
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

        for C in Cs:
            clf = LogisticRegression(C=C, penalty='l1', solver='liblinear',
                                     class_weight='balanced', max_iter=1000)
            clf.fit(X_train_f, y_train)
            prob = clf.predict_proba(X_test_f)[:, 1]
            auc_by_C[C].append(roc_auc_score(y_test, prob))

    best_C, best_auc = None, 0
    for C in Cs:
        mean_auc = np.mean(auc_by_C[C])
        print(f"  C={C}  CV-AUC={mean_auc:.4f}")
        if mean_auc > best_auc:
            best_auc, best_C = mean_auc, C

    print(f"\nBest C: {best_C}  (CV-AUC={best_auc:.4f})")

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

    clf_eval = LogisticRegression(C=best_C, penalty='l1', solver='liblinear',
                                   class_weight='balanced', max_iter=1000)
    clf_eval.fit(X_train_ef, y_train)
    prob_test = clf_eval.predict_proba(X_test_ef)[:, 1]
    auc = roc_auc_score(y_test, prob_test)
    auprc = average_precision_score(y_test, prob_test)

    # Optimal threshold
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_test, prob_test)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thresh = thresholds[np.argmax(f1s[:-1])]
    pred_binary = (prob_test >= best_thresh).astype(int)

    print(f"\nHold-out fold:  AUC={auc:.4f}  AUPRC={auprc:.4f}")
    print(f"  Optimal threshold: {best_thresh:.3f}")
    print(f"  F1={f1_score(y_test, pred_binary):.4f}  "
          f"Precision={precision_score(y_test, pred_binary):.4f}  "
          f"Recall={recall_score(y_test, pred_binary):.4f}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    vec_final = CountVectorizer(ngram_range=(1, 3), binary=True, min_df=5)
    X_all = vec_final.fit_transform(df['processed'].values)
    vt_final = VarianceThreshold(threshold=0.1 * 0.9)
    X_all_f = vt_final.fit_transform(X_all)

    clf_final = LogisticRegression(C=best_C, penalty='l1', solver='liblinear',
                                    class_weight='balanced', max_iter=1000)
    clf_final.fit(X_all_f, y)
    prob_all = clf_final.predict_proba(X_all_f)[:, 1]
    print(f"  All-data AUC={roc_auc_score(y, prob_all):.4f}")
    nz = (clf_final.coef_ != 0).sum()
    print(f"  Non-zero coefficients: {nz} / {X_all_f.shape[1]}")
    print(f"  Vocabulary size: {len(vec_final.vocabulary_)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(clf_final, os.path.join(OUTPUT_DIR, 'mrs_model.joblib'))
    joblib.dump(vec_final, os.path.join(OUTPUT_DIR, 'mrs_vectorizer.joblib'))
    joblib.dump(vt_final, os.path.join(OUTPUT_DIR, 'mrs_variance_filter.joblib'))
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"Threshold to use in config: {best_thresh:.3f}")


if __name__ == '__main__':
    main()
