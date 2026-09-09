"""
Training script for Epilepsy Subtypes model.
Trains one binary classifier per syndrome using ICD + medication + keyword features.
"""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import polars as pl
import numpy as np
import joblib
import os
import json
import re
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, roc_curve
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/niels/cdac Dropbox/Niels Turley/prophet_dataset/epilepsy_subtypes'
OUTPUT_DIR = os.path.join(_ROOT, 'src', 'prophet', 'models', 'epilepsy_subtypes')
RESULTS_DIR = os.path.join(_ROOT, 'results', 'epilepsy_subtypes')
N_FEATURES = 100  # chi-squared top-k pre-filter per syndrome


def load_syndrome_data():
    """Load syndrome configuration from JSON."""
    with open(os.path.join(OUTPUT_DIR, 'syndrome_data.json'), 'r') as f:
        return json.load(f)


def extract_icd_features(icd_df, note_df, all_icds):
    """Extract binary ICD features for each (id, date) pair."""
    print(f"Extracting ICD features for {len(all_icds)} codes...")
    feat = note_df.select(['id', 'date']).unique()

    icd_df = icd_df.with_columns(
        (pl.col('date').dt.offset_by('-1y')).alias('date_lower'),
        (pl.col('date').dt.offset_by('1y')).alias('date_upper'),
    ).drop('date')

    matched = feat.join(icd_df, on='id', how='left').filter(
        ((pl.col('date') >= pl.col('date_lower')) &
         (pl.col('date') <= pl.col('date_upper')))
    ).drop(['date_lower', 'date_upper'])

    icd_exprs = []
    for icd_code in all_icds:
        col_name = f'icd_{icd_code}'
        icd_exprs.append(
            pl.col('icd').str.starts_with(icd_code).cast(pl.Int32).max().alias(col_name)
        )

    if len(matched) > 0:
        icd_feat = matched.group_by(['id', 'date']).agg(icd_exprs)
    else:
        icd_feat = feat.with_columns([pl.lit(0).alias(f'icd_{c}') for c in all_icds])

    icd_feat = feat.join(icd_feat, on=['id', 'date'], how='left', nulls_equal=True).fill_null(0)
    print(f"  ICD features shape: {icd_feat.shape}")
    return icd_feat


def extract_med_features(med_df, note_df, med_config):
    """Extract binary medication features with consolidation."""
    print(f"Extracting medication features for {len(med_config['canonical_names'])} canonical meds...")
    feat = note_df.select(['id', 'date']).unique()
    name_to_canonical = med_config['name_to_canonical']
    canonical_meds = sorted(med_config['canonical_names'])

    med_df = med_df.with_columns(pl.col('med').str.to_lowercase().str.strip_chars())

    when_chain = pl.when(False).then(None)
    for med_name, canonical in name_to_canonical.items():
        when_chain = when_chain.when(
            pl.col('med').str.contains(f'(?i){re.escape(med_name)}')
        ).then(pl.lit(canonical))
    med_df = med_df.with_columns(
        when_chain.otherwise(None).alias('canonical_med')
    ).filter(pl.col('canonical_med').is_not_null())

    # Use 2-year lookback from note date: med must be within [note_date - 2y, note_date]
    med_df = med_df.rename({'date': 'med_date'})
    # For null-date notes, use all meds for that patient (no temporal restriction)
    matched = feat.with_columns(
        (pl.col('date').dt.offset_by('-2y')).alias('date_lower'),
    ).join(med_df, on='id', how='left').filter(
        ((pl.col('med_date') >= pl.col('date_lower')) &
         (pl.col('med_date') <= pl.col('date')))
    ).drop(['date_lower', 'med_date'])

    if len(matched) > 0:
        med_exprs = []
        for canonical in canonical_meds:
            col_name = f'med_{canonical}'
            med_exprs.append(
                (pl.col('canonical_med') == canonical).cast(pl.Int32).max().alias(col_name)
            )
        med_feat = matched.group_by(['id', 'date']).agg(med_exprs)
    else:
        med_feat = feat.with_columns([pl.lit(0).alias(f'med_{c}') for c in canonical_meds])

    med_feat = feat.join(med_feat, on=['id', 'date'], how='left', nulls_equal=True).fill_null(0)
    print(f"  Med features shape: {med_feat.shape}")
    return med_feat


def extract_keyword_features(note_df, all_keywords):
    """Extract keyword count features from stemmed notes."""
    print(f"Extracting keyword features for {len(all_keywords)} keywords...")
    stemmer = SnowballStemmer('english')

    # Combine notes per (id, date)
    note_agg = note_df.with_columns(
        pl.col('note').str.replace_all(r'[^a-zA-Z0-9 \n\.]', '')
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
        .str.to_lowercase()
    ).group_by(['id', 'date']).agg(
        pl.col('note').str.concat(delimiter=' ').alias('note')
    )

    # Compile keyword patterns
    compiled_patterns = []
    for kw in all_keywords:
        try:
            pattern = re.compile(rf'\b{kw}\b', re.IGNORECASE)
            compiled_patterns.append((kw, pattern))
        except re.error:
            compiled_patterns.append((kw, None))

    results = []
    total = len(note_agg)
    for i, row in enumerate(note_agg.iter_rows(named=True)):
        if (i + 1) % 500 == 0:
            print(f"  Processing note {i + 1}/{total}...")
        text = row['note']
        tokens = word_tokenize(text)
        stemmed_text = ' '.join(stemmer.stem(t) for t in tokens)

        feature_vector = {'id': row['id'], 'date': row['date']}
        for kw, pattern in compiled_patterns:
            col_name = f'kw_{kw}'
            if pattern is not None:
                feature_vector[col_name] = len(pattern.findall(stemmed_text))
            else:
                feature_vector[col_name] = 0
        results.append(feature_vector)

    kw_feat = pl.DataFrame(results)
    print(f"  Keyword features shape: {kw_feat.shape}")
    return kw_feat


def find_optimal_threshold_roc(y_true, y_proba):
    """Find optimal threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]


def train_syndrome_model(X, y, syndrome_name):
    """Train a model for one syndrome with GridSearchCV."""
    print(f"\n{'='*60}")
    print(f"Training model for: {syndrome_name}")
    print(f"Samples: {len(y)}, Positives: {int(y.sum())}, Negatives: {int((1-y).sum())}")
    print(f"Features: {X.shape[1]}")

    pipeline = Pipeline([
        ('preprocessor', FunctionTransformer(func=None)),
        ('model', LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=10000,
            random_state=42
        ))
    ])

    param_grid = {
        'model__C': [0.01, 0.1, 1.0, 10.0],
        'model__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
        'model__class_weight': [None, 'balanced'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='average_precision',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    # Get cross-validated predictions for threshold optimization
    from sklearn.model_selection import cross_val_predict
    cv_proba = cross_val_predict(best_model, X, y, cv=cv, method='predict_proba')
    cv_proba_pos = cv_proba[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y, cv_proba_pos)
    auprc = average_precision_score(y, cv_proba_pos)
    threshold = find_optimal_threshold_roc(y, cv_proba_pos)
    cv_pred = (cv_proba_pos >= threshold).astype(int)
    f1 = f1_score(y, cv_pred)

    print(f"CV ROC-AUC: {roc_auc:.4f}")
    print(f"CV AUPRC: {auprc:.4f}")
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"CV F1 at threshold: {f1:.4f}")

    # Retrain on all data with best params
    final_model = best_model
    final_model.fit(X, y)

    return final_model, threshold, roc_auc, auprc, f1


def main():
    print("=" * 60)
    print("Epilepsy Subtypes Model Training")
    print("=" * 60)

    # Load data
    note = pl.read_parquet(os.path.join(DATA_DIR, 'note.parquet'))
    icd = pl.read_parquet(os.path.join(DATA_DIR, 'icd.parquet'))
    med = pl.read_parquet(os.path.join(DATA_DIR, 'med.parquet'))
    annot = pl.read_parquet(os.path.join(DATA_DIR, 'annot.parquet'))

    syndrome_data = load_syndrome_data()
    syndromes = syndrome_data['syndromes']
    med_config = syndrome_data['medications']

    # Compute union of all keywords and ICDs
    all_keywords = set()
    all_icds = set()
    for s_data in syndromes.values():
        all_keywords.update(s_data['keywords'])
        all_icds.update(s_data['icds'])
    all_keywords = sorted(all_keywords)
    all_icds = sorted(all_icds)

    print(f"Total keywords: {len(all_keywords)}")
    print(f"Total ICDs: {len(all_icds)}")
    print(f"Total canonical meds: {len(med_config['canonical_names'])}")

    # Extract features
    print("\n--- Feature Extraction ---")
    icd_feat = extract_icd_features(icd, note, all_icds)
    med_feat = extract_med_features(med, note, med_config)
    kw_feat = extract_keyword_features(note, all_keywords)

    # Join all features
    feat = note.select(['id', 'date']).unique()
    feat = feat.join(icd_feat, on=['id', 'date'], how='left', nulls_equal=True) \
               .join(med_feat, on=['id', 'date'], how='left', nulls_equal=True) \
               .join(kw_feat, on=['id', 'date'], how='left', nulls_equal=True) \
               .fill_null(0)

    print(f"\nCombined feature matrix: {feat.shape}")

    # Join with annotations (map note_index -> id, date)
    note_with_annot = note.select(['note_index', 'id', 'date']).join(
        annot, on='note_index', how='left'
    )

    # For each syndrome, train a model
    feature_cols = [c for c in feat.columns if c.startswith(('icd_', 'med_', 'kw_'))]
    print(f"Total feature columns: {len(feature_cols)}")

    models_dict = {}
    results_summary = []

    for syndrome_name, s_data in syndromes.items():
        annot_col = s_data['annot_col']

        if annot_col not in note_with_annot.columns:
            print(f"\nWARNING: Annotation column '{annot_col}' not found, skipping {syndrome_name}")
            continue

        # Get annotated samples for this syndrome — use max label per (id, date)
        # to avoid losing positives when multiple notes share the same (id, date)
        labeled = note_with_annot.filter(
            pl.col(annot_col).is_not_null()
        ).group_by(['id', 'date']).agg(pl.col(annot_col).max())

        # Join with features (nulls_equal=True because many notes have null dates)
        train_data = labeled.join(feat, on=['id', 'date'], how='inner', nulls_equal=True)
        y = train_data[annot_col].to_numpy().astype(int)
        X = train_data.select(feature_cols).to_pandas()

        if len(np.unique(y)) < 2:
            print(f"\nWARNING: Only one class for {syndrome_name}, skipping")
            continue

        # Chi-squared feature pre-selection (top N_FEATURES) to speed up grid search
        n_select = min(N_FEATURES, len(feature_cols))
        chi2_scores, _ = chi2(np.abs(X.values), y)
        top_idx = np.argsort(chi2_scores)[-n_select:]
        selected_cols = [feature_cols[i] for i in sorted(top_idx)]
        X = X[selected_cols]
        print(f"Features after chi2 selection: {len(selected_cols)}")

        model, threshold, roc_auc, auprc, f1 = train_syndrome_model(X, y, syndrome_name)

        models_dict[syndrome_name] = {
            'model': model,
            'threshold': float(threshold),
            'feature_cols': selected_cols,
            'roc_auc': float(roc_auc),
            'auprc': float(auprc),
            'f1': float(f1),
            'n_samples': len(y),
            'n_positive': int(y.sum()),
        }

        results_summary.append({
            'syndrome': syndrome_name,
            'roc_auc': roc_auc,
            'auprc': auprc,
            'f1': f1,
            'threshold': threshold,
            'n_samples': len(y),
            'n_positive': int(y.sum()),
        })

    # Save models
    model_path = os.path.join(OUTPUT_DIR, 'epilepsy_subtypes_models.joblib')
    joblib.dump(models_dict, model_path)
    print(f"\nModels saved to: {model_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    results_df = pl.DataFrame(results_summary)
    print(results_df)

    # Update config with thresholds
    print("\nThresholds per syndrome:")
    for r in results_summary:
        print(f"  {r['syndrome']}: {r['threshold']:.4f}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df.write_csv(os.path.join(RESULTS_DIR, 'syndrome_results.csv'))

    summary = {
        'validation_type': '5_fold_cv',
        'scoring_metric_used': 'average_precision',
        'n_syndromes_trained': len(results_summary),
        'syndromes': {
            r['syndrome']: {
                'roc_auc': r['roc_auc'],
                'auprc': r['auprc'],
                'f1': r['f1'],
                'threshold': r['threshold'],
                'n_samples': r['n_samples'],
                'n_positive': r['n_positive'],
            }
            for r in results_summary
        }
    }
    with open(os.path.join(RESULTS_DIR, 'final_model_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
