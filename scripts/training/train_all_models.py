#!/usr/bin/env python3
"""
Train all prophet models by merging feat.parquet + annot.parquet for each model
and running model_comparison.py. One log file per model in results/.

Usage:
    python train_all_models.py [--chi2 N]

    --chi2 N   Keep only the top N features by chi-squared score before training.
               Useful for high-dimensional models. Default: use all features.
"""

import os
import re
import sys
import argparse
import subprocess
import numpy as np
import polars as pl
from datetime import datetime

DATASET_DIR = '/home/niels/cdac Dropbox/Niels Turley/prophet_dataset'
RESULTS_DIR = '/home/niels/Desktop/prophet/results'
SCRIPT      = os.path.join(os.path.dirname(__file__), 'model_comparison.py')
MODEL_VERSION = 'v1_2026'

# Models with no annotations or not trainable as classification models
SKIP_MODELS = {
    'epilepsy_subtypes',  # no annot.parquet
    'nih_stroke_scale',
    'narcolepsy',         # multi-task binary training — handled by train_narcolepsy()
    'modified_rankin_score',  # text features require TF-IDF vectorization — not yet supported
}

# Narcolepsy: 3 binary tasks matching retrain_all.py in the NAX-Narcolepsy project.
# Label maps: {original_label: binary_label, ...}; None means drop that class.
NARCOLEPSY_TASKS = {
    'nt1_vs_others':          {1: 1, 2: 0, 3: None, 4: 0},
    'nt2ih_vs_others':        {1: 0, 2: 1, 3: None, 4: 0},
    'any_narcolepsy_vs_others': {1: 1, 2: 1, 3: 1,    4: 0},
}
# Maps task name → model filename (must match narcolepsy/config.yaml model_path keys)
NARCOLEPSY_MODEL_FILES = {
    'nt1_vs_others':          'nt1_vs_not.joblib',
    'nt2ih_vs_others':        'nt2_vs_not.joblib',
    'any_narcolepsy_vs_others': 'nt12_vs_not.joblib',
}
N_FEATURES_NARC = 100  # chi-squared top-k pre-filter (matches retrain_all.py)

# Key columns added by the pipeline — not features, drop before training
KEY_COLS = {'id', 'date', 'note_idx'}


def merge_feat_annot(model_dir: str) -> pl.DataFrame:
    feat  = pl.read_parquet(os.path.join(model_dir, 'feat.parquet'))
    annot = pl.read_parquet(os.path.join(model_dir, 'annot.parquet'))

    # Join on whichever key columns both files share
    join_on = [c for c in ('id', 'date', 'note_idx')
               if c in feat.columns and c in annot.columns]

    merged = feat.join(
        annot.select(join_on + ['annot']),
        on=join_on,
        how='inner',
    )

    # Drop key columns — not features for the model
    drop = [c for c in KEY_COLS if c in merged.columns]
    return merged.drop(drop)


def select_chi2_features(df: pl.DataFrame, n_features: int) -> pl.DataFrame:
    """Return df with only the top n_features columns by chi-squared score."""
    from sklearn.feature_selection import chi2

    feature_cols = [c for c in df.columns if c != 'annot']
    X = df.select(feature_cols).to_numpy().astype(np.float64)
    y = df['annot'].to_numpy()

    # chi2 requires non-negative values — features are counts/binary, so OK
    chi2_scores, _ = chi2(np.abs(X), y)
    top_idx = np.argsort(chi2_scores)[-n_features:]
    selected_cols = [feature_cols[i] for i in sorted(top_idx)]
    return df.select(selected_cols + ['annot'])


def train_model(name: str, model_dir: str, log_path: str,
                n_chi2: int | None = None, save_fold_models: bool = False) -> int:
    merged = merge_feat_annot(model_dir)

    if n_chi2 is not None and n_chi2 < len(merged.columns) - 1:
        merged = select_chi2_features(merged, n_chi2)

    n_rows, n_cols = merged.shape
    n_feat = n_cols - 1  # minus the 'annot' column

    input_path = os.path.join(model_dir, 'train_input.parquet')
    merged.write_parquet(input_path)

    output_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, SCRIPT,
        '--input',          input_path,
        '--output_dir',     output_dir,
        '--target_column',  'annot',
        '--model_version',  MODEL_VERSION,
    ]
    if save_fold_models:
        cmd.append('--save_fold_models')

    with open(log_path, 'w') as f:
        f.write(f"=== {name} ===\n")
        f.write(f"Started:    {datetime.now()}\n")
        f.write(f"Input rows: {n_rows}  features: {n_feat}"
                + (f"  (chi2 top {n_chi2})" if n_chi2 else "") + "\n")
        f.write(f"Command:    {' '.join(cmd)}\n\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\nFinished: {datetime.now()}  exit={result.returncode}\n")

    return result.returncode


def train_narcolepsy(model_dir: str, log_path: str,
                     n_chi2: int = N_FEATURES_NARC,
                     save_fold_models: bool = False) -> dict:
    """Train 3 binary narcolepsy classifiers using the retrain_all.py approach.

    Instead of a single 4-class problem (which breaks XGBoost labels), this runs
    three separate binary tasks:
        nt1_vs_others, nt2ih_vs_others, any_narcolepsy_vs_others

    Returns a dict mapping task_name → subprocess returncode.
    """
    from sklearn.feature_selection import chi2 as chi2_fn

    feat  = pl.read_parquet(os.path.join(model_dir, 'feat.parquet'))
    annot = pl.read_parquet(os.path.join(model_dir, 'annot.parquet'))

    join_on = [c for c in ('id', 'date', 'note_idx')
               if c in feat.columns and c in annot.columns]
    merged = feat.join(annot.select(join_on + ['annot']), on=join_on, how='inner')
    merged = merged.drop([c for c in KEY_COLS if c in merged.columns])

    feature_cols = [c for c in merged.columns if c != 'annot']

    results = {}
    with open(log_path, 'w') as f:
        f.write(f"=== narcolepsy (3-task binary approach) ===\n")
        f.write(f"Started:    {datetime.now()}\n")
        f.write(f"Input rows: {len(merged)}  features: {len(feature_cols)}\n\n")
        f.flush()

        for task_name, label_map in NARCOLEPSY_TASKS.items():
            f.write(f"\n{'='*50}\n")
            f.write(f"  TASK: {task_name}\n")
            f.write(f"{'='*50}\n")

            # Remap labels; None → null (will be dropped)
            task_df = merged.with_columns(
                pl.col('annot').replace(label_map).alias('annot')
            ).filter(pl.col('annot').is_not_null()).with_columns(
                pl.col('annot').cast(pl.Int32)
            )

            n_pos = task_df.filter(pl.col('annot') == 1).shape[0]
            n_neg = task_df.filter(pl.col('annot') == 0).shape[0]
            f.write(f"  Positive: {n_pos}, Negative: {n_neg}, Total: {n_pos + n_neg}\n")

            # Chi-squared feature selection (top N_FEATURES_NARC)
            X_feat  = task_df.select(feature_cols).to_numpy().astype(np.float64)
            y_label = task_df['annot'].to_numpy()
            chi2_scores, _ = chi2_fn(np.abs(X_feat), y_label)
            top_idx = np.argsort(chi2_scores)[-n_chi2:]
            selected_cols = [feature_cols[i] for i in sorted(top_idx)]
            f.write(f"  Features after chi2 selection: {len(selected_cols)}\n")

            # Sanitize feature names for XGBoost (no [ ] < > in names)
            rename_map = {}
            for c in selected_cols:
                clean = re.sub(r'[\[\]<>]', '_', c)
                if clean != c:
                    rename_map[c] = clean
            if rename_map:
                task_df = task_df.rename(rename_map)
                selected_cols = [rename_map.get(c, c) for c in selected_cols]
                f.write(f"  Renamed {len(rename_map)} features with invalid XGBoost chars\n")

            task_df_filtered = task_df.select(selected_cols + ['annot'])
            input_path = os.path.join(model_dir, f'{task_name}_input.parquet')
            task_df_filtered.write_parquet(input_path)

            output_dir = os.path.join(RESULTS_DIR, 'narcolepsy', task_name)
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                sys.executable, SCRIPT,
                '--input',         input_path,
                '--output_dir',    output_dir,
                '--target_column', 'annot',
                '--model_version', MODEL_VERSION,
            ]
            if save_fold_models:
                cmd.append('--save_fold_models')

            f.write(f"  Command: {' '.join(cmd)}\n\n")
            f.flush()

            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            f.write(f"\n{task_name} finished: {datetime.now()}  exit={result.returncode}\n")
            f.flush()
            results[task_name] = result.returncode

    return results


def find_trainable_models() -> list[str]:
    names = []
    for name in sorted(os.listdir(DATASET_DIR)):
        if name in SKIP_MODELS:
            continue
        d = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(d):
            continue
        if not os.path.exists(os.path.join(d, 'feat.parquet')):
            continue
        if not os.path.exists(os.path.join(d, 'annot.parquet')):
            continue
        names.append(name)
    return names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chi2', type=int, default=None, metavar='N',
                        help='Keep top N features by chi-squared score (default: all features)')
    parser.add_argument('--save_fold_models', action='store_true',
                        help='Save the model from each CV fold (passed to model_comparison.py)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = find_trainable_models()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_log = os.path.join(RESULTS_DIR, f'train_all_{timestamp}.log')

    print(f"Training {len(models)} models"
          + (f" (chi2 top {args.chi2} features)" if args.chi2 else ""))
    print(f"Results → {RESULTS_DIR}/")
    print(f"Summary → {summary_log}\n")

    results = []
    with open(summary_log, 'w') as sf:
        sf.write(f"Prophet model training run — {datetime.now()}\n")
        sf.write(f"Models: {models}\n\n")
        sf.flush()

        for name in models:
            model_dir = os.path.join(DATASET_DIR, name)
            log_path  = os.path.join(RESULTS_DIR, f'{name}.log')
            t0 = datetime.now()

            print(f"[{t0.strftime('%H:%M:%S')}] {name} ...", flush=True)

            try:
                rc = train_model(name, model_dir, log_path,
                                n_chi2=args.chi2,
                                save_fold_models=args.save_fold_models)
                status = 'OK' if rc == 0 else f'FAILED (exit {rc})'
            except Exception as e:
                status = f'ERROR: {e}'

            elapsed = (datetime.now() - t0).seconds
            line = f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {status}  ({elapsed}s)  log={log_path}"
            print(line, flush=True)
            sf.write(line + '\n')
            sf.flush()

            results.append((name, status))

    # --- mRS: text-based model, requires its own vectorization pipeline ---
    mrs_dir = os.path.join(DATASET_DIR, 'modified_rankin_score')
    mrs_log = os.path.join(RESULTS_DIR, 'modified_rankin_score.log')
    mrs_script = os.path.join(os.path.dirname(__file__), 'train_mrs.py')
    if os.path.isdir(mrs_dir) and os.path.exists(mrs_script):
        t0 = datetime.now()
        print(f"[{t0.strftime('%H:%M:%S')}] modified_rankin_score ...", flush=True)
        try:
            with open(mrs_log, 'w') as f:
                f.write(f"=== modified_rankin_score ===\nStarted: {datetime.now()}\n\n")
                result = subprocess.run(
                    [sys.executable, mrs_script],
                    stdout=f, stderr=subprocess.STDOUT, text=True,
                )
                f.write(f"\nFinished: {datetime.now()}  exit={result.returncode}\n")
            status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
        except Exception as e:
            status = f'ERROR: {e}'

        elapsed = (datetime.now() - t0).seconds
        line = (f"[{datetime.now().strftime('%H:%M:%S')}] modified_rankin_score: {status}"
                f"  ({elapsed}s)  log={mrs_log}")
        print(line, flush=True)
        with open(summary_log, 'a') as sf:
            sf.write(line + '\n')
        results.append(('modified_rankin_score', status))

    # --- Narcolepsy: 3-task binary training (retrain_all.py approach) ---
    narc_dir  = os.path.join(DATASET_DIR, 'narcolepsy')
    narc_log  = os.path.join(RESULTS_DIR, 'narcolepsy.log')
    if os.path.isdir(narc_dir) and os.path.exists(os.path.join(narc_dir, 'feat.parquet')):
        t0 = datetime.now()
        print(f"[{t0.strftime('%H:%M:%S')}] narcolepsy (3-task binary) ...", flush=True)
        try:
            narc_rc = train_narcolepsy(narc_dir, narc_log,
                                       save_fold_models=args.save_fold_models)
            any_failed = any(rc != 0 for rc in narc_rc.values())
            status = f'FAILED ({narc_rc})' if any_failed else 'OK'
        except Exception as e:
            status = f'ERROR: {e}'

        elapsed = (datetime.now() - t0).seconds
        line = (f"[{datetime.now().strftime('%H:%M:%S')}] narcolepsy: {status}"
                f"  ({elapsed}s)  log={narc_log}")
        print(line, flush=True)
        with open(summary_log, 'a') as sf:
            sf.write(line + '\n')
        results.append(('narcolepsy', status))

    print(f"\n{'='*60}")
    print(f"Done — {sum(1 for _, s in results if s == 'OK')}/{len(results)} succeeded")
    for name, status in results:
        print(f"  {status:30s}  {name}")
    print(f"\nSummary: {summary_log}")
