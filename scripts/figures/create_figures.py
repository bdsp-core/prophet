#!/usr/bin/env python3
"""
PROPHET Paper Figures
Generates all publication figures for the PROPHET paper.

Run with:
    PYENV_VERSION=prophet python3 scripts/figures/create_figures.py

Figures produced:
    1. ehr_inputs.png          — phenotype × EHR data-type grid
    2. performance_heatmap.png — AUROC / AUPRC heatmap (all models)
    3. roc_prc_bootstrap.png   — per-disease AUROC/AUPRC with 95 % CI
    4. swimmer_plot.png        — patient prediction timelines
    5. mrs_classification.png   — mRS per-class performance  [COMMENTED OUT]
    6. nihss_performance.png   — NIHSS regression / bin performance
    7. consort_diagram.png     — cohort composition by phenotype and hospital
"""

import os, json, sys
from pathlib import Path
from glob import glob

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / 'results'
MODELS_DIR  = ROOT / 'src/prophet/models'
OUTCOMES_DIR = ROOT / 'scripts/figures/outcomes'
FIGURES_DIR  = ROOT / 'scripts/figures/output'
FIGURES_DIR.mkdir(exist_ok=True)

# Add prophet source to path for imports
sys.path.insert(0, str(ROOT / 'src'))

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# Consistent palette
PALETTE = {
    'auroc': '#2196F3',
    'auprc': '#4CAF50',
    'accent': '#FF5722',
    'grid':  '#E0E0E0',
    'cell_bdsp':       '#d5e8d4',   # BDSP dataset cells (light green)
    'cell_mimic':      '#fce5cd',   # MIMIC dataset cells (light orange)
    'cell_no_dataset': '#e1d5e7',   # No-dataset cells (light purple)
    'cell_header':     '#dae8fc',   # Header / total cells (light blue)
}

COHORT_LABELS = {
    'I0001': 'MGH',
    'I0002': 'BIDMC',
    'I0003': 'BCH',
    'I0004': 'BWH',
    'I0006': 'Emory',
    'I0007': 'Stanford',
}

# ─────────────────────────────────────────────────────────────────────────────
# Phenotype display names (binary ML models only)
# ─────────────────────────────────────────────────────────────────────────────
BINARY_MODELS = {
    'brain_tumor':                                  'Brain Tumor',
    'cardiac_arrest':                               'Cardiac Arrest',
    'congestive_heart_failure':                     'Congestive Heart Failure',
    'epilepsy':                                     'Epilepsy',
    'intracranial_hemorrhage':                      'Intracerebral Hemorrhage',
    'ischemic_stroke':                              'Ischemic Stroke',
    'mild_cognitive_impairment_alzhiemers_disease': "MCI / Alzheimer's Disease",
    'neuroinfectious_diseases':                     'Neuroinfectious Diseases',
    'parkinsons_disease':                           "Parkinson's Disease",
    'subarachnoid_hemorrhage':                      'Subarachnoid Hemorrhage',
    'subdural_hematoma':                            'Subdural Hematoma',
    'traumatic_brain_injury':                       'Traumatic Brain Injury',
    'withdrawal_of_life_sustaining_therapy':        'Withdrawal of LST',
}

NARCOLEPSY_MODELS = {
    'narcolepsy_nt12': 'Narcolepsy Any Type',
    'narcolepsy_nt1':  'Narcolepsy Type 1',
    'narcolepsy_nt2ih':'Narcolepsy Type 2/IH',
}

# Epilepsy subtypes — multi-label, one row per syndrome
EPILEPSY_SYNDROME_LABELS = {
    'SELECT':        'Epilepsy — SELECTS',
    'focal':         'Epilepsy — Focal',
    'generalized':   'Epilepsy — Generalized',
    'infantilespasm':'Epilepsy — Infantile Spasm',
    'JME':           'Epilepsy — Juvenile Myoclonic',
    'temporal':      'Epilepsy — Temporal Lobe',
}

# Ordinal / regression models — commented out until results are ready
# ORDINAL_MODELS = {
#     'modified_rankin_score': 'Modified Rankin Score',
# }


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_binary_performance():
    """Return dict keyed by display name with all metrics for best model."""
    perf = {}
    for key, label in BINARY_MODELS.items():
        fp = RESULTS_DIR / key / 'final_model_summary.json'
        kfp = RESULTS_DIR / key / 'kfold_results.csv'
        if not fp.exists():
            print(f'  [SKIP] {key}: no final_model_summary.json')
            continue
        with open(fp) as f:
            summary = json.load(f)
        vp = summary['validation_performance']
        best_model = summary['best_model']

        # Per-fold metrics for the best model
        fold_auroc, fold_auprc = [], []
        if kfp.exists():
            df = pl.read_csv(kfp)
            rows = df.filter(pl.col('model') == best_model)
            fold_auroc = rows['roc_auc'].to_list()
            fold_auprc = rows['auprc'].to_list()

        perf[label] = {
            'accuracy':    vp.get('avg_accuracy'),
            'precision':   vp.get('avg_precision'),
            'recall':      vp.get('avg_recall'),
            'f1':          vp.get('avg_f1'),
            'auroc':       vp['avg_roc_auc'],
            'auprc':       vp['avg_auprc'],
            'model':       best_model,
            'n':           summary['production_model_info']['training_samples'],
            'fold_auroc':  fold_auroc,
            'fold_auprc':  fold_auprc,
        }
    return perf


def load_narcolepsy_performance():
    """Return dict keyed by narcolepsy display name with AUROC/AUPRC."""
    perf = {}

    # nt1_vs_others and nt2ih_vs_others: avg_results.csv
    for key, label in [('nt1_vs_others', 'narcolepsy_nt1'),
                        ('nt2ih_vs_others', 'narcolepsy_nt2ih')]:
        fp = RESULTS_DIR / 'narcolepsy' / key / 'avg_results.csv'
        pfp = RESULTS_DIR / 'narcolepsy' / key / 'per_fold_results.csv'
        if not fp.exists():
            continue
        df = pl.read_csv(fp)
        best_row = df.sort('ROC-AUC (mean)', descending=True).row(0, named=True)
        best_model_name = best_row['Model']
        # per-fold AUROC/AUPRC for CI
        fold_auroc, fold_auprc = [], []
        if pfp.exists():
            pfdf = pl.read_csv(pfp)
            if 'Model' in pfdf.columns and 'ROC-AUC' in pfdf.columns:
                rows = pfdf.filter(pl.col('Model') == best_model_name)
                fold_auroc = rows['ROC-AUC'].to_list()
                fold_auprc = rows['PR-AUC'].to_list() if 'PR-AUC' in pfdf.columns else []
        perf[NARCOLEPSY_MODELS[label]] = {
            'accuracy':   best_row['Accuracy (mean)'],
            'precision':  best_row['Precision (mean)'],
            'recall':     best_row['Sensitivity (mean)'],  # Sensitivity = Recall
            'f1':         best_row['F1 (mean)'],
            'auroc':      best_row['ROC-AUC (mean)'],
            'auprc':      best_row['PR-AUC (mean)'],
            'model':      best_model_name,
            'n':          None,
            'fold_auroc': fold_auroc,
            'fold_auprc': fold_auprc,
        }

    # any_narcolepsy_vs_others: full 5-fold CV results (NT1 + NT2 + NT3 vs rest)
    fp = RESULTS_DIR / 'narcolepsy' / 'any_narcolepsy_vs_others' / 'final_model_summary.json'
    kfp = RESULTS_DIR / 'narcolepsy' / 'any_narcolepsy_vs_others' / 'kfold_results.csv'
    if fp.exists():
        with open(fp) as f:
            summary = json.load(f)
        vp = summary['validation_performance']
        best_model = summary['best_model']
        fold_auroc, fold_auprc = [], []
        if kfp.exists():
            df = pl.read_csv(kfp)
            rows = df.filter(pl.col('model') == best_model)
            fold_auroc = rows['roc_auc'].to_list()
            fold_auprc = rows['auprc'].to_list()
        perf[NARCOLEPSY_MODELS['narcolepsy_nt12']] = {
            'accuracy':   vp.get('avg_accuracy'),
            'precision':  vp.get('avg_precision'),
            'recall':     vp.get('avg_recall'),
            'f1':         vp.get('avg_f1'),
            'auroc':      vp['avg_roc_auc'],
            'auprc':      vp['avg_auprc'],
            'model':      best_model,
            'n':          summary['production_model_info']['training_samples'],
            'fold_auroc': fold_auroc,
            'fold_auprc': fold_auprc,
        }

    return perf


def load_epilepsy_subtypes_performance():
    """Return dict keyed by syndrome display name with AUROC/AUPRC/F1."""
    fp = RESULTS_DIR / 'epilepsy_subtypes' / 'final_model_summary.json'
    if not fp.exists():
        print('  [SKIP] epilepsy_subtypes: no final_model_summary.json')
        return {}
    with open(fp) as f:
        summary = json.load(f)
    perf = {}
    for key, label in EPILEPSY_SYNDROME_LABELS.items():
        d = summary['syndromes'].get(key, {})
        if not d:
            continue
        perf[label] = {
            'accuracy':   None,
            'precision':  d.get('precision'),
            'recall':     d.get('recall'),
            'f1':         d.get('f1'),
            'auroc':      d.get('roc_auc'),
            'auprc':      d.get('auprc'),
            'model':      'XGBoost',
            'n':          d.get('n_samples'),
            'fold_auroc': [],
            'fold_auprc': [],
        }
    return perf


def bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    """Bootstrap 95% CI on the mean of `values`."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    if len(arr) < 2:
        return arr.mean(), arr.mean(), arr.mean()
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return arr.mean(), lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — EHR Input Data Types
# ─────────────────────────────────────────────────────────────────────────────
def fig_ehr_inputs():
    """Grid: phenotype (row) × EHR data type (column)."""
    import yaml

    class PolarsSafeLoader(yaml.SafeLoader):
        pass
    yaml.SafeLoader.add_constructor('!pl', lambda loader, node: loader.construct_scalar(node))

    DATA_TYPES  = ['Clinical Notes', 'ICD Codes', 'Medications', 'Demographics', 'CPT Codes']
    SCHEMA_KEYS = ['note',           'icd',       'med',         'demo',         'cpt']

    # models to include (ordered): rule-based first, then ML binary, then ordinal
    MODEL_ORDER = [
        # Rule-based (ICD-only)
        ('af',      'Atrial Fibrillation (ICD)'),
        ('cad',     'Coronary Artery Disease (ICD)'),
        ('dm',      'Diabetes Mellitus (ICD)'),
        ('is_icd',  'Ischemic Stroke (ICD)'),
        ('mi',      'Myocardial Infarction (ICD)'),
        # ML binary
        ('brain_tumor',                                  'Brain Tumor'),
        ('ca',                                           'Cardiac Arrest'),
        ('chf',                                          'Congestive Heart Failure'),
        ('epilepsy',                                     'Epilepsy'),
        ('ich',                                          'Intracerebral Hemorrhage'),
        ('is',                                           'Ischemic Stroke'),
        ('mci',                                          "MCI / Alzheimer's Disease"),
        ('narcolepsy',                                   'Narcolepsy'),
        ('nidx',                                         'Neuroinfectious Diseases'),
        ('pd',                                           "Parkinson's Disease"),
        ('sah',                                          'Subarachnoid Hemorrhage'),
        ('sdh',                                          'Subdural Hematoma'),
        ('tbi',                                          'Traumatic Brain Injury'),
        ('wlst',                                         'Withdrawal of LST'),
        # Ordinal / regression (commented out until ready)
        # ('epilepsy_subtypes', 'Epilepsy Subtypes'),
        # ('mrs',               'Modified Rankin Score'),
        # ('nihss',             'NIH Stroke Scale'),
    ]

    grid = []
    labels = []
    model_types = []  # 'rule' or 'ml'
    for key, name in MODEL_ORDER:
        fp = MODELS_DIR / key / 'config.yaml'
        if not fp.exists():
            continue
        with open(fp) as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        schema = cfg.get('schema', {})
        row = [1 if k in schema else 0 for k in SCHEMA_KEYS]
        grid.append(row)
        labels.append(name)
        model_types.append('rule' if key in {'af', 'cad', 'dm', 'is_icd', 'mi'} else 'ml')

    grid = np.array(grid)
    n_rows, n_cols = grid.shape

    fig_h = max(5, n_rows * 0.42 + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    # Draw colored squares
    cmap_ml   = mcolors.to_rgba('#1565C0')
    cmap_rule = mcolors.to_rgba('#558B2F')
    cmap_empty = mcolors.to_rgba('#F5F5F5')

    for r in range(n_rows):
        for c in range(n_cols):
            color = (cmap_rule if model_types[r] == 'rule' else cmap_ml) if grid[r, c] else cmap_empty
            rect = mpatches.FancyBboxPatch(
                (c + 0.05, n_rows - r - 1 + 0.05), 0.9, 0.9,
                boxstyle='round,pad=0.05',
                facecolor=color, edgecolor='white', linewidth=1.5,
            )
            ax.add_patch(rect)

    # Grid lines
    for r in range(n_rows + 1):
        ax.axhline(r, color='white', lw=0.5)
    for c in range(n_cols + 1):
        ax.axvline(c, color='white', lw=0.5)

    # Dividing line between rule-based and ML
    n_rule = sum(1 for t in model_types if t == 'rule')
    ax.axhline(n_rows - n_rule, color='#424242', lw=1.5, ls='--')

    # Axes
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(DATA_TYPES, fontsize=10, fontweight='bold')
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.tick_params(axis='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    handles = [
        mpatches.Patch(facecolor='#1565C0', label='ML Model'),
        mpatches.Patch(facecolor='#558B2F', label='Rule-Based Model'),
        mpatches.Patch(facecolor='#F5F5F5', edgecolor='#BDBDBD', label='Not Used'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_title('EHR Data Sources per Phenotype', fontweight='bold', pad=12)

    out = FIGURES_DIR / 'ehr_inputs.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Model Performance Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_performance_heatmap():
    """Heatmap: phenotype × {Accuracy, Precision, Recall, F1, AUROC, AUPRC}."""
    perf = {
        **load_binary_performance(),
        **load_narcolepsy_performance(),
        **load_epilepsy_subtypes_performance(),
        # **load_mrs_performance(),  # uncomment when mRS results are ready
    }

    order = (
        sorted(BINARY_MODELS.values())
        + sorted(NARCOLEPSY_MODELS.values())
        + sorted(EPILEPSY_SYNDROME_LABELS.values())
        # + sorted(ORDINAL_MODELS.values())  # uncomment when ready
    )
    order = [k for k in order if k in perf]

    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc']
    COL_LABELS = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC']

    matrix = np.full((len(order), len(METRICS)), np.nan)
    for i, label in enumerate(order):
        for j, m in enumerate(METRICS):
            v = perf[label].get(m)
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(9, max(5, len(order) * 0.45 + 1.5)))

    # Mask NaN cells for coloring (show as light grey)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad('#EEEEEE')
    im = ax.imshow(masked, cmap=cmap, vmin=0.60, vmax=1.0, aspect='auto')

    # Annotations
    for i in range(len(order)):
        for j in range(len(METRICS)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='#9E9E9E')
            else:
                text_color = 'white' if val > 0.91 else '#212121'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8.5, color=text_color, fontweight='bold')

    # Vertical divider between standard metrics and AUROC/AUPRC
    ax.axvline(3.5, color='white', lw=2)

    # Axes
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(COL_LABELS, fontweight='bold', fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.tick_params(axis='both', length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Score', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title('Model Performance (5-Fold CV / Leave-One-Site-Out)', fontweight='bold', pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    out = FIGURES_DIR / 'performance_heatmap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — AUROC / AUPRC with Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────
def fig_roc_prc_bootstrap():
    """Forest plot: per-disease AUROC and AUPRC with 95 % bootstrap CI."""
    perf = {
        **load_binary_performance(),
        **load_narcolepsy_performance(),
        **load_epilepsy_subtypes_performance(),
    }

    order = (
        sorted(BINARY_MODELS.values())
        + sorted(NARCOLEPSY_MODELS.values())
        + sorted(EPILEPSY_SYNDROME_LABELS.values())
    )
    order = [k for k in order if k in perf][::-1]  # reverse for bottom-up

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(order) * 0.45 + 2)),
                              sharey=True)

    for ax, metric, color, title in [
        (axes[0], 'auroc', PALETTE['auroc'], 'AUROC'),
        (axes[1], 'auprc', PALETTE['auprc'], 'AUPRC'),
    ]:
        means, los, his = [], [], []
        for label in order:
            p = perf[label]
            folds = p[f'fold_{metric}']
            mean = p[metric]
            if len(folds) >= 2:
                _, lo, hi = bootstrap_ci(folds)
            else:
                lo = hi = mean
            means.append(mean)
            los.append(lo)
            his.append(hi)

        y = np.arange(len(order))
        xerr = np.array([
            [m - l for m, l in zip(means, los)],
            [h - m for m, h in zip(means, his)],
        ])

        ax.barh(y, means, xerr=xerr, height=0.6,
                color=color, alpha=0.75, capsize=3,
                error_kw={'elinewidth': 1.2, 'ecolor': '#424242'})
        ax.axvline(0.5, color='#9E9E9E', lw=1, ls='--', alpha=0.6, label='chance')
        ax.set_xlim(0.4, 1.02)
        ax.set_xlabel(title, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=8)
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(axis='x', color=PALETTE['grid'], lw=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(order, fontsize=9)
    axes[0].tick_params(axis='y', length=0)

    fig.suptitle('Model Performance with 95% Bootstrap CI', fontweight='bold',
                 fontsize=13, y=1.01)
    fig.tight_layout()

    out = FIGURES_DIR / 'roc_prc_bootstrap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Swimmer Plots  (from swimmer_plots.ipynb, swimmer_df_updated)
# ─────────────────────────────────────────────────────────────────────────────

# Color map from swimmer_plots.ipynb (Cell 44)
PHENOTYPE_COLOR_MAP = {
    'cardiac_arrest':            '#8B0000',
    'myocardial_infarction':     '#DC143C',
    'congestive_heart_failure':  '#FF6347',
    'atrial_fibrillation':       '#FF7F50',
    'coronary_artery_disease':   '#CD5C5C',
    'ischemic_stroke':           '#4169E1',
    'intracranial_hemorrhage':   '#6A5ACD',
    'subarachnoid_hemorrhage':   '#8A2BE2',
    'subdural_hematoma':         '#9370DB',
    'traumatic_brain_injury':    '#483D8B',
    'brain_tumor':               '#2F4F4F',
    'parkinsons_disease':        '#228B22',
    'mild_cognitive_impairment': '#32CD32',
    'epilepsy':                  '#00CED1',
    'diabetes_mellitus':         '#FFD700',
    'hospital_record':           '#F5F5F5',
    'death':                     '#C5C5C5',
}


def create_adaptive_timeline_plot_optimized(
    df, figsize=(12, 8), dpi=100,
    color_map=None, event_order=None,
    align_on=None, sort_by=None, group_by=None,
    ax=None, fontsize=None, use_calendar_years=False,
    show_progress=True, show_legend=True, legend_position='right',
):
    """Swimmer plot engine — ported from swimmer_plots.ipynb (Cell 43)."""
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.patches import Rectangle
    from datetime import datetime
    import random, time
    from tqdm.auto import tqdm

    def _p(msg):
        if show_progress:
            print(f'  {msg}')

    t0 = time.time()
    required = ['id', 'state', 'start_date', 'end_date']
    if not all(c in df.columns for c in required):
        raise ValueError(f'DataFrame must contain: {required}')

    _p(f'Input: {len(df):,} rows, {df["id"].n_unique():,} patients')

    if align_on:
        ids = df.filter(pl.col('state') == align_on)['id'].unique().to_list()
        df = df.filter(pl.col('id').is_in(ids))

    grouped = df.group_by('id').agg(
        pl.col('start_date').min().alias('min_date'),
        pl.col('start_date').max().alias('max_date'),
        pl.col('state').unique().alias('states_list'),
    )

    if group_by:
        group_states = [group_by] if isinstance(group_by, str) else group_by
        for state in group_states:
            state_ids = set(df.filter(pl.col('state') == state)['id'].unique().to_list())
            grouped = grouped.with_columns(
                pl.col('id').map_elements(
                    lambda pid: pid in state_ids, return_dtype=pl.Boolean
                ).alias(f'has_{state}')
            )

    ref_date = grouped['min_date'].min()

    if align_on:
        align_dates = df.filter(pl.col('state') == align_on).group_by('id').agg(
            pl.col('start_date').min().alias('align_date')
        )
        grouped = grouped.join(align_dates, on='id', how='left')
        grouped = grouped.with_columns([
            (pl.col('min_date') - pl.col('align_date')).alias('start_days'),
            (pl.col('max_date') - pl.col('align_date')).alias('end_days'),
        ])
    else:
        grouped = grouped.with_columns([
            (pl.col('min_date') - ref_date).alias('start_days'),
            (pl.col('max_date') - ref_date).alias('end_days'),
        ])

    if sort_by:
        sort_dates = df.filter(pl.col('state') == sort_by).group_by('id').agg(
            pl.col('start_date').min().alias('sort_date')
        )
        if align_on:
            grouped = grouped.join(sort_dates, on='id', how='left')
            grouped = grouped.with_columns(
                (pl.col('sort_date') - pl.col('align_date')).alias('relative_sort_days')
            ).with_columns(
                pl.when(pl.col('sort_date').is_null())
                .then(float('inf')).otherwise(pl.col('relative_sort_days'))
                .alias('final_sort_key')
            )
        else:
            earliest = df.group_by('id').agg(pl.col('start_date').min().alias('earliest_date'))
            sorting_df = earliest.join(sort_dates, on='id', how='left').with_columns(
                (pl.col('sort_date') - pl.col('earliest_date')).alias('sort_distance')
            )
            grouped = grouped.join(
                sorting_df.select(['id', 'sort_date', 'sort_distance']), on='id', how='left'
            ).with_columns(
                pl.when(pl.col('sort_date').is_null())
                .then(float('inf')).otherwise(pl.col('sort_distance'))
                .alias('final_sort_key')
            )
        if group_by:
            gcols = [f'has_{s}' for s in group_states]
            grouped = grouped.sort(gcols + ['final_sort_key'],
                                   descending=[True] * len(gcols) + [False])
        else:
            grouped = grouped.sort('final_sort_key')
    else:
        if group_by:
            gcols = [f'has_{s}' for s in group_states]
            grouped = grouped.sort(gcols, descending=[True] * len(gcols))
        else:
            grouped = grouped.sort('start_days')

    patient_to_idx = {pid: i for i, pid in enumerate(grouped['id'].to_list())}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def rand_color():
        return '#' + ''.join(random.choice('0123456789ABCDEF') for _ in range(6))

    if event_order is None:
        event_order = sorted(df['state'].unique().to_list())

    state_colors = {
        s: (color_map.get(s, rand_color()) if color_map else rand_color())
        for s in event_order
    }

    min_days = grouped['start_days'].min()
    max_days = grouped['end_days'].max()

    align_info = {}
    if align_on:
        for row in grouped.iter_rows(named=True):
            align_info[row['id']] = row.get('align_date')

    all_bars, all_lines = {}, {}
    for state in tqdm(event_order, desc='States', disable=not show_progress):
        edata = df.filter(pl.col('state') == state)
        if len(edata) == 0:
            continue
        bars, lines = [], []
        for ev in edata.iter_rows(named=True):
            pid = ev['id']
            if pid not in patient_to_idx:
                continue
            y = patient_to_idx[pid]
            if align_on and pid in align_info and align_info[pid]:
                sx = (ev['start_date'] - align_info[pid]).days
                ex = (ev['end_date'] - align_info[pid]).days if ev['end_date'] else max_days.days
            else:
                sx = (ev['start_date'] - ref_date).days
                ex = (ev['end_date'] - ref_date).days if ev['end_date'] else max_days.days
            if sx == ex:
                lines.append([(sx, y), (sx, y + 1)])
            else:
                bars.append([(sx, y), (ex, y), (ex, y + 1), (sx, y + 1)])
        if bars:
            all_bars[state] = bars
        if lines:
            all_lines[state] = lines

    for state in event_order:
        color = state_colors[state]
        if state in all_bars:
            ax.add_collection(PolyCollection(all_bars[state], facecolors=color, edgecolors='none'))
        if state in all_lines:
            ax.add_collection(LineCollection(all_lines[state], colors=color, linewidths=1))

    if group_by:
        sep_styles = [
            {'color': 'black', 'linewidth': 2.0, 'alpha': 0.8},
            {'color': 'gray',  'linewidth': 1.5, 'alpha': 0.6},
        ]
        prev_groups = None
        for i, row in enumerate(grouped.iter_rows(named=True)):
            cur = tuple(row[f'has_{s}'] for s in group_states)
            if i > 0 and cur != prev_groups:
                for lvl in range(len(group_states)):
                    if cur[:lvl+1] != prev_groups[:lvl+1]:
                        st = sep_styles[min(lvl, len(sep_styles)-1)]
                        ax.axhline(y=i, color=st['color'], linestyle='--',
                                   linewidth=st['linewidth'], alpha=st['alpha'], zorder=5)
                        break
            prev_groups = cur

    ax.set_ylim(0, len(grouped))

    if align_on:
        yb = abs(min_days.days) // 365.25
        ya = max_days.days // 365.25
        xt = np.arange(-yb * 365.25, (ya + 1) * 365.25, 365.25)
        ax.set_xticks(xt, [int(x // 365.25) for x in xt],
                      fontsize=fontsize or None)
        ax.set_xlim(min_days.days, max_days.days)
        ax.axvline(0, color='#9c0000', lw=1, zorder=4)
        ax.set_xlabel(f'Years Relative to First {align_on}', fontsize=fontsize or None)
    elif use_calendar_years:
        min_dt = grouped['min_date'].min()
        max_dt = grouped['max_date'].max()
        years = range(min_dt.year, max_dt.year + 2)
        tick_dates = [datetime(y, 1, 1) for y in years]
        xt = [(d - ref_date).days for d in tick_dates]
        ax.set_xticks(xt, [str(y) for y in years], fontsize=fontsize or None)
        ax.set_xlim(min_days.days, max_days.days)
        ax.set_xlabel('Calendar Year', fontsize=fontsize or None)
    else:
        yt = max_days.days // 365.25
        xt = np.arange(0, (yt + 1) * 365.25, 365.25)
        ax.set_xticks(xt, [int(x // 365.25) for x in xt], fontsize=fontsize or None)
        ax.set_xlim(min_days.days, max_days.days)
        ax.set_xlabel('Years Since First Event', fontsize=fontsize or None)

    title = f'Patient Timeline (n = {len(grouped):,})'
    if sort_by:
        title += f'\nSorted by Time to First {sort_by}'
    if group_by:
        title += f'\nGrouped by {group_by if isinstance(group_by, str) else " → ".join(group_by)}'
    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.grid(True, axis='x', color='white', lw=0.5, zorder=3)
    ax.set_yticks([])
    ax.set_axisbelow(False)

    if show_legend:
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=state_colors[s], edgecolor='none')
            for s in event_order if s in all_bars or s in all_lines
        ]
        lbls = [s for s in event_order if s in all_bars or s in all_lines]
        bbox = {'right': (1.05, 1), 'left': (-0.05, 1),
                'top': (0.5, 1.15), 'bottom': (0.5, -0.15)}.get(legend_position, (1.05, 1))
        loc  = {'right': 'upper left', 'left': 'upper right',
                'top': 'upper center', 'bottom': 'upper center'}.get(legend_position, 'upper left')
        kw = dict(handles=handles, labels=lbls, bbox_to_anchor=bbox, loc=loc,
                  frameon=True, fancybox=True, shadow=True)
        if legend_position in ('top', 'bottom'):
            kw['ncol'] = min(len(handles), 4)
        if fontsize:
            kw['fontsize'] = fontsize
        ax.legend(**kw)

    _p(f'Done in {time.time()-t0:.1f}s')
    return ax


def _build_swimmer_df():
    """
    Construct swimmer_df_updated following swimmer_plots.ipynb (Cells 42, 50).
    Uses censor_dates.parquet (Dropbox) for hospital_record/death spans,
    and outcomes_all.parquet for disease prediction first-event dates.
    """
    CENSOR_GLOB = '/home/niels/cdac Dropbox/Niels Turley/BDSP_deID/*/data_Outcomes/censor_dates.parquet'

    print('  Loading censor_dates…', end=' ', flush=True)
    filter_dates = pl.read_parquet(glob(CENSOR_GLOB))
    print(f'{filter_dates["bdsp_patient_id"].n_unique():,} patients')

    print('  Loading outcomes…', end=' ', flush=True)
    outcomes_raw = pl.read_parquet(OUTCOMES_DIR / 'outcomes_all.parquet')
    # Keep ML models only (rule-based have null prob_YES) and positive predictions
    outcomes = (
        outcomes_raw
        .filter(pl.col('prediction') == 1, pl.col('prob_YES').is_not_null())
        .sort(['bdsp_patient_id', 'date'])
    )
    print(f'{len(outcomes):,} positive ML predictions')

    # First positive prediction date per patient × disease  (Cell 38)
    filtered_outcomes = outcomes.group_by(['bdsp_patient_id', 'disease']).agg(
        pl.col('date').first().alias('first_event_date')
    )

    # Cell 42 — build swimmer_df
    death_events = (
        filter_dates.filter(pl.col('date_of_death').is_not_null())
        .select(
            pl.col('bdsp_patient_id').alias('id'),
            pl.lit('death').alias('state'),
            pl.col('date_of_death').cast(pl.Datetime).alias('start_date'),
            pl.lit(None).cast(pl.Datetime).alias('end_date'),
        )
    )
    hospital_events = (
        filter_dates.filter(
            pl.col('earliest_record').is_not_null(),
            pl.col('latest_record').is_not_null(),
        )
        .select(
            pl.col('bdsp_patient_id').alias('id'),
            pl.lit('hospital_record').alias('state'),
            pl.col('earliest_record').cast(pl.Datetime).alias('start_date'),
            pl.col('latest_record').cast(pl.Datetime).alias('end_date'),
        )
    )
    disease_events = filtered_outcomes.select(
        pl.col('bdsp_patient_id').alias('id'),
        pl.col('disease').alias('state'),
        pl.col('first_event_date').cast(pl.Datetime).alias('start_date'),
        pl.col('first_event_date').cast(pl.Datetime).alias('end_date'),
    )
    swimmer_df = pl.concat([death_events, hospital_events, disease_events],
                            how='diagonal_relaxed')

    # Cell 50 — set disease end_dates to next-event / death / hospital-end
    hospital_end = (
        swimmer_df.filter(pl.col('state') == 'hospital_record')
        .select('id', pl.col('end_date').alias('hospital_end_date'))
    )
    disease_df = (
        swimmer_df.filter(~pl.col('state').is_in(['hospital_record', 'death']))
        .sort(['id', 'start_date'])
    )
    next_ev = disease_df.with_columns(
        pl.col('start_date').shift(-1).over('id').alias('next_event_date')
    )
    death_dates = (
        swimmer_df.filter(pl.col('state') == 'death')
        .select('id', pl.col('start_date').alias('death_date'))
    )
    updated_disease = (
        next_ev
        .join(death_dates, on='id', how='left')
        .join(hospital_end, on='id', how='left')
        .with_columns(
            pl.min_horizontal(['next_event_date', 'death_date', 'hospital_end_date'])
            .alias('end_date')
        )
        .select(['id', 'state', 'start_date', 'end_date'])
    )
    swimmer_df_updated = pl.concat([
        swimmer_df.filter(pl.col('state').is_in(['hospital_record', 'death'])),
        updated_disease,
    ], how='diagonal_relaxed')

    return swimmer_df_updated


def fig_swimmer_plots():
    """
    Swimmer plots following swimmer_plots.ipynb.
    Generates:
      • swimmer_plot_all.png          — full cohort (Cell 51/53)
      • swimmer_plot_per_site.png     — per-site subplots (Cell 59)
    """
    CENSOR_GLOB = '/home/niels/cdac Dropbox/Niels Turley/BDSP_deID/*/data_Outcomes/censor_dates.parquet'
    if not glob(CENSOR_GLOB):
        print('  [SKIP] censor_dates not found')
        return

    swimmer_df_updated = _build_swimmer_df()

    # Filter obviously bad dates
    swimmer_df_updated = swimmer_df_updated.filter(
        pl.col('start_date').dt.year() > 1990
    )

    # Only keep patients who have at least one disease prediction (not just hospital_record/death)
    disease_patient_ids = (
        swimmer_df_updated
        .filter(~pl.col('state').is_in(['hospital_record', 'death']))
        ['id'].unique().to_list()
    )
    swimmer_df_updated = swimmer_df_updated.filter(
        pl.col('id').is_in(disease_patient_ids)
    )
    print(f'  Patients with ≥1 disease prediction: {len(disease_patient_ids):,}')

    # ── Full cohort plot (Cell 51 / 53) ──────────────────────────────────────
    print('  Rendering full-cohort swimmer plot…')
    ax = create_adaptive_timeline_plot_optimized(
        swimmer_df_updated,
        figsize=(15, 20),
        color_map=PHENOTYPE_COLOR_MAP,
        use_calendar_years=True,
        show_progress=True,
        show_legend=True,
        legend_position='right',
    )
    ax.tick_params(axis='x', rotation=90)
    fig = ax.get_figure()
    out = FIGURES_DIR / 'swimmer_plot_all.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')

    # ── Per-site subplots (Cell 59) ───────────────────────────────────────────
    filter_dates = pl.read_parquet(glob(CENSOR_GLOB))
    sources = sorted([s for s in filter_dates['cohort'].unique().to_list() if s])
    SOURCE_LABELS = {
        'I0001': 'MGB', 'I0002': 'BIDMC', 'I0003': 'BCH',
        'I0004': 'Stanford', 'I0006': 'Emory', 'I0007': 'KP',
    }

    print(f'  Rendering per-site swimmer plots ({len(sources)} sites)…')
    n_cols = 2
    # +1 for legend panel; ceil to full rows
    n_panels = len(sources) + 1
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10 * n_rows))
    axes = axes.flatten()

    for idx, s in enumerate(sources):
        site_ids = filter_dates.filter(pl.col('cohort') == s)['bdsp_patient_id'].unique().to_list()
        site_df = swimmer_df_updated.filter(pl.col('id').is_in(site_ids))
        if len(site_df) == 0:
            axes[idx].axis('off')
            continue
        create_adaptive_timeline_plot_optimized(
            site_df,
            color_map=PHENOTYPE_COLOR_MAP,
            use_calendar_years=True,
            show_progress=False,
            show_legend=False,
            ax=axes[idx],
        )
        axes[idx].tick_params(axis='x', rotation=90)
        axes[idx].set_title(SOURCE_LABELS.get(s, s), fontsize=28, fontweight='bold')

    # Shared legend in next unused panel
    from matplotlib.patches import Rectangle as Rect
    legend_ax = axes[len(sources)]
    legend_ax.axis('off')
    handles = [Rect((0, 0), 1, 1, facecolor=c, edgecolor='none')
               for c in PHENOTYPE_COLOR_MAP.values()]
    labels  = list(PHENOTYPE_COLOR_MAP.keys())
    legend_ax.legend(handles, labels, loc='center', fontsize=14,
                     frameon=True, fancybox=True, shadow=True, ncol=2)

    for idx in range(len(sources) + 1, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    out = FIGURES_DIR / 'swimmer_plot_per_site.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — mRS Classification Performance  [COMMENTED OUT]
# ─────────────────────────────────────────────────────────────────────────────
# def fig_mrs_classification():
#     """Per-class classification performance for Modified Rankin Score."""
#     # Uncomment and update results path when mRS model is retrained.
#     #
#     # Expected data: results/modified_rankin_score/kfold_results.csv
#     # with columns: [class_label, precision, recall, f1, support, fold, model]
#     #
#     # RESULTS_FP = RESULTS_DIR / 'modified_rankin_score' / 'kfold_results.csv'
#     # if not RESULTS_FP.exists():
#     #     print('  [SKIP] mRS results not found'); return
#     # df = pl.read_csv(RESULTS_FP)
#     # ... (plotting code here)
#     pass


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — NIHSS Regression / Bin Performance
# # ─────────────────────────────────────────────────────────────────────────────
# def fig_nihss_performance():
#     """
#     Scatter (predicted vs actual NIHSS) and severity-bin classification.
#     Loads the trained model and dataset from the Dropbox path used in
#     train_nihss.py. Skips gracefully if data is unavailable.
#     """
#     import joblib
#     from sklearn.metrics import mean_squared_error
#     from scipy.stats import spearmanr

#     DATA_DIR  = Path('/home/niels/cdac Dropbox/Niels Turley/prophet_dataset/nih_stroke_scale')
#     MODEL_PATH = MODELS_DIR / 'nihss' / 'nihss_model.joblib'

#     if not DATA_DIR.exists():
#         print('  [SKIP] NIHSS Dropbox data not found')
#         return
#     if not MODEL_PATH.exists():
#         print('  [SKIP] NIHSS model not found (run train_nihss.py first)')
#         return

#     sys.path.insert(0, str(ROOT / 'src'))
#     import pandas as pd

#     feat  = pl.read_parquet(DATA_DIR / 'feat.parquet')
#     annot = pl.read_parquet(DATA_DIR / 'annot.parquet').select(['note_idx', 'annot'])

#     df = (
#         feat
#         .join(annot, on='note_idx', how='left')
#         .to_pandas()
#     )

#     model = joblib.load(MODEL_PATH)

#     # Use columns that the model was trained on (single source of truth in predictor)
#     from prophet.models.nihss.predictor import NIHSS_FEATURES as FEATURE_COLS
#     X = df[FEATURE_COLS].values
#     y = df['annot'].values
#     y_pred = np.clip(model.predict(X), 0, 42)

#     rmse = np.sqrt(mean_squared_error(y, y_pred))
#     rho, pval = spearmanr(y, y_pred)

#     # Severity bins: 0=no stroke, 1-4=minor, 5-15=moderate, 16-20=mod-severe, 21-42=severe
#     bins  = [0, 1, 5, 16, 21, 43]
#     labels_bin = ['Normal\n(0)', 'Minor\n(1–4)', 'Moderate\n(5–15)',
#                   'Mod-Severe\n(16–20)', 'Severe\n(21–42)']
#     y_bin      = np.digitize(y,      bins, right=False) - 1
#     y_pred_bin = np.digitize(y_pred, bins, right=False) - 1
#     n_bins = len(labels_bin)

#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Panel A: scatter predicted vs actual
#     ax = axes[0]
#     ax.scatter(y, y_pred, alpha=0.25, s=10, color=PALETTE['auroc'], rasterized=True)
#     ax.plot([0, 42], [0, 42], 'k--', lw=1, label='perfect prediction')
#     ax.set_xlabel('Actual NIHSS', fontweight='bold')
#     ax.set_ylabel('Predicted NIHSS', fontweight='bold')
#     ax.set_title(f'NIHSS Prediction\nRMSE={rmse:.2f}, ρ={rho:.3f} (p={pval:.1e})',
#                  fontweight='bold')
#     ax.legend(fontsize=9)
#     ax.set_xlim(-1, 43); ax.set_ylim(-1, 43)

#     # Panel B: confusion-style bar chart across severity bins
#     ax = axes[1]
#     bin_acc = []
#     for b in range(n_bins):
#         mask = y_bin == b
#         if mask.sum() == 0:
#             bin_acc.append(0)
#         else:
#             bin_acc.append((y_pred_bin[mask] == b).mean())
#     bars = ax.bar(labels_bin, bin_acc, color=PALETTE['auprc'], alpha=0.8, edgecolor='white')
#     ax.set_ylim(0, 1)
#     ax.set_ylabel('Within-bin Accuracy', fontweight='bold')
#     ax.set_title('Per-Severity-Bin Accuracy', fontweight='bold')
#     for bar, val in zip(bars, bin_acc):
#         ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
#                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)
#     ax.grid(axis='y', color=PALETTE['grid'], lw=0.8, zorder=0)
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     fig.suptitle('NIH Stroke Scale Performance', fontweight='bold', fontsize=13)
#     fig.tight_layout()

#     out = FIGURES_DIR / 'nihss_performance.png'
#     fig.savefig(out)
#     plt.close(fig)
#     print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Consort Diagram
# ─────────────────────────────────────────────────────────────────────────────
def fig_consort_diagram():
    """Generate consort diagram showing cohort composition by phenotype and hospital."""

    # Data structure: each phenotype with total notes and sources by hospital/dataset
    data = {
        'Brain Tumor': {
            'total_notes': 2000,
            'sources': [
                {'hospital': 'BIDMC', 'dataset': 'No Dataset', 'pts': 470, 'notes': 1000},
                {'hospital': 'MGB', 'dataset': 'No Dataset', 'pts': 500, 'notes': 1000}
            ]
        },
        'Cardiac Arrest': {
            'total_notes': 3000,
            'sources': [
                {'hospital': 'BIDMC', 'dataset': 'MIMIC', 'pts': 1000, 'notes': 1000},
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 2000, 'notes': 2000}
            ]
        },
        'Congestive Heart Failure': {
            'total_notes': 2800,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 1393, 'notes': 1400},
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 428, 'notes': 1400}
            ]
        },
        'Epilepsy': {
            'total_notes': 8402,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 3903, 'notes': 8402}
            ]
        },
        'Epilepsy Subtypes': {
            'total_notes': 13334,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 2196, 'notes': 5985},
                {'hospital': 'BCH', 'dataset': 'BDSP', 'pts': 874, 'notes': 3578},
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 1143, 'notes': 3771}
            ]
        },
        'Intracerebral Hemorrhage': {
            'total_notes': 2594,
            'sources': [
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 1403, 'notes': 1509},
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 296, 'notes': 1085}
            ]
        },
        'Ischemic Stroke': {
            'total_notes': 3404,
            'sources': [
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 248, 'notes': 1945},
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 477, 'notes': 1459}
            ]
        },
        'MCI / Alzheimer\'s Disease': {
            'total_notes': 2332,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 1964, 'notes': 2332}
            ]
        },
        'Modified Rankin Score': {
            'total_notes': 5530,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'No Dataset', 'pts': 5337, 'notes': 5530}
            ]
        },
        'Narcolepsy': {
            'total_notes': 9356,
            'sources': [
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 1549, 'notes': 2110},
                {'hospital': 'Emory', 'dataset': 'BDSP', 'pts': 1292, 'notes': 1841},
                {'hospital': 'Stanford', 'dataset': 'BDSP', 'pts': 1454, 'notes': 1563},
                {'hospital': 'BCH', 'dataset': 'BDSP', 'pts': 1138, 'notes': 1881},
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 1059, 'notes': 1961}
            ]
        },
        'Neuroinfectious Diseases': {
            'total_notes': 3000,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 2469, 'notes': 3000}
            ]
        },
        'NIH Stroke Scale': {
            'total_notes': 4007,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'No Dataset', 'pts': 3876, 'notes': 4007}
            ]
        },
        'Parkinson\'s Disease': {
            'total_notes': 3827,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 1964, 'notes': 1964},
                {'hospital': 'Stanford', 'dataset': 'BDSP', 'pts': 863, 'notes': 863},
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 1000, 'notes': 1000}
            ]
        },
        'Subarachnoid Hemorrhage': {
            'total_notes': 1548,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 1041, 'notes': 1041},
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 507, 'notes': 507}
            ]
        },
        'Subdural Hematoma': {
            'total_notes': 2999,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 1499, 'notes': 1499},
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 1500, 'notes': 1500}
            ]
        },
        'Traumatic Brain Injury': {
            'total_notes': 3000,
            'sources': [
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 2000, 'notes': 2000},
                {'hospital': 'BIDMC', 'dataset': 'BDSP', 'pts': 1000, 'notes': 1000}
            ]
        },
        'Withdrawal of LST': {
            'total_notes': 3000,
            'sources': [
                {'hospital': 'BIDMC', 'dataset': 'MIMIC', 'pts': 1000, 'notes': 1000},
                {'hospital': 'MGB', 'dataset': 'BDSP', 'pts': 2000, 'notes': 2000}
            ]
        }
    }

    # Get unique hospitals sorted
    hospitals = set()
    for condition_data in data.values():
        for source in condition_data['sources']:
            hospitals.add(source['hospital'])
    hospitals = sorted(list(hospitals))

    # Color scheme by dataset (from global PALETTE)
    color_bdsp = PALETTE['cell_bdsp']
    color_mimic = PALETTE['cell_mimic']
    color_none = PALETTE['cell_no_dataset']
    color_header = PALETTE['cell_header']
    color_row_label = '#F5F5F5'  # Matches cmap_empty in fig_ehr_inputs

    # Grid dimensions
    cell_width = 1.4
    cell_height = 0.8
    start_x = 3.5
    start_y = len(data) + 1

    # Create figure with appropriate xlim
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.set_xlim(0, start_x + (len(hospitals) + 1) * cell_width + 0.5)
    ax.set_ylim(0, len(data) + 2)
    ax.axis('off')

    # Helper function to draw cell
    def draw_cell(ax, x, y, width, height, text, color, fontsize=11, fontweight='normal', edgecolor='white'):
        box = mpatches.FancyBboxPatch((x, y), width, height,
                                      boxstyle="round,pad=0.02",
                                      edgecolor=edgecolor, facecolor=color,
                                      linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, wrap=True)

    # Draw header row with hospital names
    for i, hospital in enumerate(hospitals):
        draw_cell(ax, start_x + i * cell_width, start_y, cell_width, cell_height,
                  hospital, color_header, fontsize=11, fontweight='bold', edgecolor='white')

    # Draw TOTAL column header
    draw_cell(ax, start_x + len(hospitals) * cell_width, start_y, cell_width, cell_height,
              'TOTAL', color_header, fontsize=11, fontweight='bold', edgecolor='white')

    # Draw grid with data
    for row_idx, (condition, info) in enumerate(data.items()):
        y_pos = start_y - (row_idx + 1) * cell_height

        # Draw condition label on the left
        draw_cell(ax, 0.1, y_pos, start_x - 0.2, cell_height,
                  condition, color_row_label, fontsize=10, fontweight='bold')

        # Create lookup for this condition's sources
        source_lookup = {}
        for source in info['sources']:
            source_lookup[source['hospital']] = source

        # Calculate totals for this condition
        total_pts_cond = sum(s['pts'] for s in info['sources'])
        total_notes_cond = info['total_notes']

        # Draw cells for each hospital
        for col_idx, hospital in enumerate(hospitals):
            x_pos = start_x + col_idx * cell_width

            if hospital in source_lookup:
                source = source_lookup[hospital]
                # Determine color based on dataset
                if source['dataset'] == 'BDSP':
                    cell_color = color_bdsp
                elif source['dataset'] == 'MIMIC':
                    cell_color = color_mimic
                else:
                    cell_color = color_none

                cell_text = f"{source['pts']:,}\n{source['notes']:,}"
                draw_cell(ax, x_pos, y_pos, cell_width, cell_height,
                         cell_text, cell_color, fontsize=9)
            else:
                # Empty cell
                draw_cell(ax, x_pos, y_pos, cell_width, cell_height,
                         '', 'white', fontsize=9)

        # Draw TOTAL cell
        total_x = start_x + len(hospitals) * cell_width
        total_text = f"{total_pts_cond:,}\n{total_notes_cond:,}"
        draw_cell(ax, total_x, y_pos, cell_width, cell_height,
                 total_text, color_header, fontsize=9, fontweight='bold')

    # Add legend with row labels
    legend_y_start = start_y - (len(data) + 0.5) * cell_height
    ax.text(start_x, legend_y_start - 0.8, 'First row: patients  |  Second row: notes',
            fontsize=10, style='italic')

    legend_elements = [
        mpatches.Patch(color=color_bdsp, label='BDSP'),
        mpatches.Patch(color=color_mimic, label='MIMIC'),
        mpatches.Patch(color=color_none, label='No Dataset')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, title='Dataset')

    fig.suptitle('Consort Diagram by Phenotype and Hospital',
                 fontsize=13, fontweight='bold', y=0.9)

    out = FIGURES_DIR / 'consort_diagram.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate PROPHET paper figures')
    parser.add_argument('--figs', nargs='*', default=None,
                        help='Subset of figures to run: ehr perf roc swim nihss consort')
    args = parser.parse_args()

    run = set(args.figs) if args.figs else {'ehr', 'perf', 'roc', 'swim', 'nihss', 'consort'}

    if 'ehr' in run:
        print('\n[1] EHR input figure…')
        fig_ehr_inputs()

    if 'perf' in run:
        print('\n[2] Performance heatmap…')
        fig_performance_heatmap()

    if 'roc' in run:
        print('\n[3] ROC/PRC bootstrap forest plot…')
        fig_roc_prc_bootstrap()

    if 'swim' in run:
        print('\n[4] Swimmer plots…')
        fig_swimmer_plots()

    # if 'nihss' in run:
    #     print('\n[5] NIHSS performance…')
    #     fig_nihss_performance()

    if 'consort' in run:
        print('\n[6] CONSORT diagram…')
        fig_consort_diagram()

    print('\nDone. Figures saved to:', FIGURES_DIR)
