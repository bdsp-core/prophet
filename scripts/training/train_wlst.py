"""
Training script for WLST (Withdrawal of Life Sustaining Therapy) model.
Uses pre-extracted regex features from prophet_dataset.
"""
import sys
import os

# Add this directory to path for model_comparison import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _SCRIPT_DIR)

import polars as pl
import pandas as pd
import numpy as np
from model_comparison import (
    cross_source_model_comparison,
    define_models_config,
    train_final_production_model,
    find_optimal_threshold,
    plot_adaptive_model_curves,
    plot_adaptive_confusion_matrix
)
import joblib

DATA_DIR = '/home/niels/cdac Dropbox/Niels Turley/prophet_dataset/withdrawal_of_life_sustaining_therapy'
ORIGINAL_CSV = '/home/niels/Desktop/pheno_models/CA/NAXCA/CA program files/WLST/WLSTMatrix.csv'
OUTPUT_DIR = os.path.join(_ROOT, 'src', 'prophet', 'models', 'wlst')

def load_training_data():
    """Load and merge features, annotations, and hospital source."""
    feat = pl.read_parquet(os.path.join(DATA_DIR, 'feat.parquet'))
    annot = pl.read_parquet(os.path.join(DATA_DIR, 'annot.parquet'))
    note = pl.read_parquet(os.path.join(DATA_DIR, 'note.parquet'))

    # Get hospital mapping from original CSV
    wlst_csv = pd.read_csv(ORIGINAL_CSV)
    hospital_map = pl.from_pandas(wlst_csv[['BDSPPatientID', 'Hospital']]).with_columns(
        pl.col('BDSPPatientID').cast(pl.String).alias('id')
    )

    # Feature columns (exclude index)
    feature_cols = [c for c in feat.columns if c != 'feat_index']

    # Join feat + annot + id + hospital
    train_df = (
        feat
        .join(annot.select(['feat_index', 'annot']), on='feat_index', how='left')
        .join(note.select(['note_index', 'id']), left_on='feat_index', right_on='note_index', how='left')
        .join(hospital_map.select(['id', 'Hospital']), on='id', how='left')
        .drop(['feat_index', 'id'])
    )

    # Clean column names for sklearn compatibility (replace special chars)
    clean_names = {}
    for col in feature_cols:
        clean = col.replace('\\b', '_B_').replace('\\s', '_S_').replace('|', '_OR_')
        clean = clean.replace('(', '_LP_').replace(')', '_RP_').replace('[', '_LB_').replace(']', '_RB_')
        clean = clean.replace('{', '_LC_').replace('}', '_RC_').replace('+', '_PLUS_')
        clean = clean.replace('?', '_Q_').replace('*', '_STAR_').replace('^', '_CARET_')
        clean_names[col] = clean

    train_df = train_df.rename(clean_names)

    print(f"Training data shape: {train_df.shape}")
    print(f"Target distribution:\n{train_df['annot'].value_counts()}")
    print(f"Hospital distribution:\n{train_df['Hospital'].value_counts()}")

    return train_df, clean_names

def main():
    print("=" * 60)
    print("WLST Model Training")
    print("=" * 60)

    train_df, clean_names = load_training_data()

    # Cross-source model comparison
    models_config = define_models_config()

    print("\n--- Cross-Source Model Comparison ---")
    results_df, curves_data = cross_source_model_comparison(
        df=train_df,
        source_col='Hospital',
        target_col='annot',
        models_config=models_config,
        output_dir=os.path.join(OUTPUT_DIR, 'training_output'),
        random_state=42
    )

    print("\n--- Results Summary ---")
    print(results_df.select(['model', 'train_source', 'test_source', 'roc_auc', 'auprc', 'f1', 'accuracy']))

    # Find best model by average AUPRC across cross-source splits
    best_results = (
        results_df
        .group_by('model')
        .agg([
            pl.col('roc_auc').mean().alias('mean_roc_auc'),
            pl.col('auprc').mean().alias('mean_auprc'),
            pl.col('f1').mean().alias('mean_f1'),
        ])
        .sort('mean_auprc', descending=True)
    )

    print("\n--- Model Rankings (by mean AUPRC) ---")
    print(best_results)

    best_model_name = best_results['model'][0]
    print(f"\nBest model: {best_model_name}")

    # Get best params from the best performing cross-source split
    best_row = results_df.filter(pl.col('model') == best_model_name).sort('auprc', descending=True).row(0, named=True)
    best_params_str = best_row['best_params']
    best_params = eval(best_params_str)
    # Remove model__ prefix for production training
    clean_params = {k.replace('model__', ''): v for k, v in best_params.items()}

    model_class = dict(models_config)[best_model_name][0]

    print(f"Best parameters: {clean_params}")

    # Train final production model on ALL data
    print("\n--- Training Final Production Model ---")
    best_config = (best_model_name, model_class, clean_params)

    final_model, model_info = train_final_production_model(
        df=train_df,
        source_col='Hospital',
        target_col='annot',
        best_model_config=best_config,
        output_dir=os.path.join(OUTPUT_DIR, 'training_output'),
        random_state=42
    )

    # Save as the model file for prophet
    model_path = os.path.join(OUTPUT_DIR, 'wlst_model.joblib')
    joblib.dump(final_model, model_path)
    print(f"\nFinal model saved to: {model_path}")

    # Find optimal threshold using cross-validation predictions
    # Use the best model's predictions from cross-source validation
    print("\n--- Finding Optimal Threshold ---")
    all_y_true = []
    all_y_proba = []
    for key, curve_info in curves_data.get(best_model_name, {}).items():
        y_test = curve_info['y_test']
        y_proba = curve_info['y_proba']
        if hasattr(y_test, 'to_numpy'):
            y_test = y_test.to_numpy()
        all_y_true.extend(y_test)
        if y_proba.ndim > 1:
            all_y_proba.extend(y_proba[:, 1])
        else:
            all_y_proba.extend(y_proba)

    all_y_true = np.array(all_y_true)
    all_y_proba = np.array(all_y_proba)

    threshold, metrics_at_threshold = find_optimal_threshold(all_y_true, all_y_proba, metric='f1')
    print(f"Optimal threshold (F1): {threshold:.4f}")
    print(f"Metrics at threshold: {metrics_at_threshold}")

    # Also save feature name mapping
    mapping_path = os.path.join(OUTPUT_DIR, 'feature_name_mapping.joblib')
    joblib.dump(clean_names, mapping_path)
    print(f"Feature name mapping saved to: {mapping_path}")

    # Save the original regex pattern → clean name mapping for the predictor
    print(f"\nThreshold to use in config: {threshold:.2f}")
    print(f"Feature columns (clean names): {list(clean_names.values())}")

    # Plot curves
    plot_adaptive_model_curves(
        curves_data, results_df,
        output_dir=os.path.join(OUTPUT_DIR, 'training_output'),
        scoring_metric='average_precision'
    )

if __name__ == '__main__':
    main()
