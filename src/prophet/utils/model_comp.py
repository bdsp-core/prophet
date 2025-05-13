import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, average_precision_score, precision_recall_curve,
                            roc_curve, confusion_matrix)
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.base import clone
import argparse

def build_adaptive_pipeline(X, model_class):
    """
    Build a pipeline that selectively applies scaling only to non-binary features
    """
    # Identify binary and continuous features
    binary_cols = []
    continuous_cols = []
    
    for col in X.columns:
        # Check if column contains only 0s and 1s
        if set(X[col].unique()).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
    
    print(f"Detected {len(binary_cols)} binary features and {len(continuous_cols)} continuous features")
    
    # Create preprocessing steps based on feature types
    if len(continuous_cols) > 0:
        # We have mixed feature types - use ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', StandardScaler(), continuous_cols),
                ('binary', FunctionTransformer(func=None), binary_cols)  # Identity transformation
            ],
            remainder='passthrough'
        )
    else:
        # All features are binary - skip scaling entirely
        preprocessor = FunctionTransformer(func=None)  # Identity transformation
        
    # Create and return the pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_class())
    ])

def cross_source_model_comparison(df, source_col, target_col, models_config, output_dir='output', random_state=42):
    # Get unique sources
    sources = df[source_col].unique().to_list()
    print(f"Found {len(sources)} unique sources: {sources}")
    
    results = []
    curves_data = {}  # To store curve data for plotting
    
    # For each model type
    for model_name, (model_class, param_grid) in models_config.items():
        print(f"\n===== Evaluating {model_name} =====")
        curves_data[model_name] = {}
        
        # For each source as training
        for train_source in sources:
            # Filter data for training source
            train_data = df.filter(pl.col(source_col) == train_source)
            X_train, y_train = prepare_data(train_data, target_col, source_col)
            
            # Create pipeline with current model
            pipeline = build_adaptive_pipeline(X_train, model_class)
            
            # Add 'model__' prefix to param grid keys
            model_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
            
            # Train model
            print(f"\nTraining {model_name} on source: {train_source}")
            
            # Create a stratified cross-validation strategy to maintain class distribution
            from sklearn.model_selection import StratifiedKFold
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid=model_param_grid,
                cv=cv_strategy,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"Best parameters for {model_name} on {train_source}: {best_params}")
            
            # Test on each other source
            for test_source in sources:
                if test_source != train_source:
                    print(f"Testing {model_name} on source: {test_source}")
                    
                    # Filter data for test source
                    test_data = df.filter(pl.col(source_col) == test_source)
                    X_test, y_test = prepare_data(test_data, target_col, source_col)
                    
                    # Predict and evaluate
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = {
                        'model': model_name,
                        'train_source': train_source,
                        'test_source': test_source,
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_proba),
                        'auprc': average_precision_score(y_test, y_proba),  # Added AUPRC
                        'best_params': str(best_params)  # Convert to string for DataFrame compatibility
                    }
                    
                    # Store curve data for plotting
                    curves_data[model_name][f"{train_source}_{test_source}"] = {
                        'y_test': y_test,
                        'y_proba': y_proba,
                        'y_pred': y_pred  # Store predictions for confusion matrix
                    }
                    
                    results.append(metrics)
                    
                    print(f"{model_name} {train_source} → {test_source} Performance:")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1 Score: {metrics['f1']:.4f}")
                    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
                    print(f"AUPRC Score: {metrics['auprc']:.4f}")  # Print AUPRC
    
    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)
    return results_df, curves_data

def prepare_data(source_df, target_col, source_col):
    """
    Prepare data for model training/testing with proper checks for data quality.
    
    Args:
        source_df: Source DataFrame
        target_col: Target column name
        source_col: Source column name
        
    Returns:
        X, y: Features and target arrays
    """
    # Check if required columns exist
    if target_col not in source_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    if source_col not in source_df.columns:
        raise ValueError(f"Source column '{source_col}' not found in data")
    
    X = source_df.drop([target_col, source_col])
    y = source_df[target_col]
    
    # Check for class imbalance and warn if extreme
    class_counts = y.value_counts()
    if len(class_counts) > 1:
        minority_ratio = class_counts['count'].min() / class_counts['count'].sum()
        if minority_ratio < 0.1:
            print(f"WARNING: Severe class imbalance detected. Minority class represents only {minority_ratio:.1%} of the data.")
            print("Consider using stratified sampling and evaluating with AUPRC metrics.")
    
    # Check for missing values in features
    missing_cols = X.null_count().transpose(include_header=True).filter(pl.col('column_0')>0)['column'].to_list()
    if missing_cols:
        print(f"WARNING: Missing values detected in {len(missing_cols)} feature columns. Consider imputation strategy.")
    
    return X, y

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find the optimal threshold that maximizes a given metric.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'youdens_j', 'precision_recall_product')
        
    Returns:
        optimal_threshold: The threshold that maximizes the chosen metric
        metrics_at_threshold: Dictionary of metrics at the optimal threshold
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    # Generate a range of thresholds to evaluate
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # Store metrics for each threshold
    metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate various metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        youdens_j = recall + specificity - 1
        precision_recall_product = precision * recall
        
        metrics[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'youdens_j': youdens_j,
            'precision_recall_product': precision_recall_product
        }
    
    # Find threshold that maximizes the chosen metric
    if metric in ['f1', 'balanced_accuracy', 'youdens_j', 'precision_recall_product']:
        optimal_threshold = max(metrics.items(), key=lambda x: x[1][metric])[0]
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'f1', 'balanced_accuracy', 'youdens_j', 'precision_recall_product'")
    
    return optimal_threshold, metrics[optimal_threshold]

def train_best_production_model(df, source_col, target_col, results_df, output_dir='output', random_state=42):
    # Analyze which model performed best overall
    avg_metrics = results_df.group_by('model').agg(
        pl.mean('accuracy').alias('avg_accuracy'),
        pl.mean('precision').alias('avg_precision'),
        pl.mean('recall').alias('avg_recall'),
        pl.mean('f1').alias('avg_f1'),
        pl.mean('roc_auc').alias('avg_roc_auc'),
        pl.mean('auprc').alias('avg_auprc')  # Added AUPRC averaging
    )
    
    # Sort by roc_auc (or your preferred metric)
    best_models = avg_metrics.sort('avg_roc_auc', descending=True)
    best_model_name = best_models[0, 'model']
    
    print(f"\nBest overall model: {best_model_name}")
    print("Average metrics across all validations:")
    print(best_models.filter(pl.col('model') == best_model_name))
    
    # Get the configuration for the best model
    models_config = define_models_config()
    model_class, param_grid = models_config[best_model_name]
    
    # Train on all data with the best model type
    if source_col is not None:
        X = df.drop([target_col, source_col])
    else:
        X = df.drop([target_col])
    y = df[target_col]
    
    # Create pipeline with best model
    pipeline = build_adaptive_pipeline(X, model_class)
    
    # Add 'model__' prefix to param grid keys
    model_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
    
    print(f"\nTraining production {best_model_name} on all data...")
    
    # Use consistent stratified cross-validation for final model training
    from sklearn.model_selection import StratifiedKFold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid=model_param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    final_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters for production model: {best_params}")

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    final_model.fit(X_train, y_train)
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    y_val_pred = final_model.predict(X_val)

    # Find optimal threshold for F1 score (or change to your preferred metric)
    optimal_threshold, threshold_metrics = find_optimal_threshold(y_val, y_val_proba, metric='f1')

    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Metrics at optimal threshold:")
    for metric, value in threshold_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Create and save confusion matrix for the best model
    # Create and save confusion matrix for the validation set
    plot_confusion_matrix(y_val, y_val_pred, f"{best_model_name}_validation", output_dir)
    
    # Create and save confusion matrix for the entire dataset
    y_all_pred = final_model.predict(X)
    plot_confusion_matrix(y, y_all_pred, f"{best_model_name}_full_dataset", output_dir)

    y_all_proba = final_model.predict_proba(X)[:, 1]
    
    # Create a dataframe with the original data plus predictions
    predictions_df = df.clone()
    predictions_df = predictions_df.with_columns([
        pl.Series(name="predicted_label", values=y_all_pred),
        pl.Series(name="predicted_probability", values=y_all_proba)
    ])
    
    # Add a column to identify false positives and false negatives
    predictions_df = predictions_df.with_columns([
        pl.when((pl.col(target_col) == 0) & (pl.col("predicted_label") == 1))
        .then(pl.lit("false_positive"))
        .when((pl.col(target_col) == 1) & (pl.col("predicted_label") == 0))
        .then(pl.lit("false_negative"))
        .when(pl.col(target_col) == pl.col("predicted_label"))
        .then(pl.lit("correct"))
        .alias("prediction_type")
    ])
    
    # Save the predictions dataframe
    predictions_path = os.path.join(output_dir, f'final_predictions_{best_model_name}.csv')
    predictions_df.write_csv(predictions_path)
    print(f"Final predictions saved to {predictions_path}")

    # Save the threshold with the model
    model_info = {
        'threshold': optimal_threshold,
        'metrics': threshold_metrics
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and threshold information in the output directory
    joblib.dump(model_info, os.path.join(output_dir, f'threshold_metrics_{best_model_name}.pkl'))
    joblib.dump(final_model, os.path.join(output_dir, f'final_model_{best_model_name}.pkl'))
    
    return final_model, best_params, best_model_name

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir='output'):
    """
    Plot and save confusion matrix for the given model predictions.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        model_name: Name of the model for plot title
        output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a more visually appealing confusion matrix with seaborn
    plt.figure(figsize=(10, 8))
    
    # Plot with seaborn
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Set labels
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix - {model_name}')
    
    # Set tick labels
    ax.set_xticklabels(['Negative (0)', 'Positive (1)'])
    ax.set_yticklabels(['Negative (0)', 'Positive (1)'])
    
    # Calculate and display additional metrics on the plot
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add text with metrics below the heatmap
    plt.figtext(0.5, 0.01, 
                f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Specificity: {specificity:.4f}",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for text
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {os.path.join(output_dir, f'confusion_matrix_{model_name}.png')}")

def define_models_config():
    # Define models and their parameter grids
    models_config = {
        'LogisticRegression': (
            LogisticRegression,
            {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs'],  # Remove liblinear for simplicity
                'penalty': ['l2', None],  # Compatible with lbfgs
                'class_weight': [None, 'balanced'],
                'random_state': [42],
                'max_iter': [2000]  # Increased iterations
            }
        ),
        'LogisticRegression_Liblinear': (
            LogisticRegression,
            {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear'],
                'penalty': ['l1', 'l2'],  # Only these work with liblinear
                'class_weight': [None, 'balanced'],
                'random_state': [42],
                'max_iter': [2000]
            }
        ),
        'RandomForest': (
            RandomForestClassifier,
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'class_weight': [None, 'balanced'],
                'random_state': [42]
            }
        ),
        'GradientBoosting': (
            GradientBoostingClassifier,
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'random_state': [42]
            }
        ),
        'XGBoost': (
            XGBClassifier,
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                # 'scale_pos_weight': [1, sum(y_train == 0) / sum(y_train == 1)],  # For imbalanced datasets
                'random_state': [42]
            }
        ),
        # 'LightGBM': (
        #     LGBMClassifier,
        #     {
        #         'n_estimators': [100, 200],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'max_depth': [3, 5, 7],
        #         'subsample': [0.8, 1.0],
        #         'colsample_bytree': [0.8, 1.0],
        #         'class_weight': [None, 'balanced'],
        #         'random_state': [42]
        #     }
        # ),
        'SVM': (
            SVC,
            {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'probability': [True],
                'class_weight': [None, 'balanced'],
                'random_state': [42]
            }
        )
    }
    
    return models_config

def plot_model_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot ROC and PR curves for all models, with the best model highlighted.
    
    Args:
        curves_data: Dictionary containing curve data for each model
        results_df: DataFrame with performance metrics
        output_dir: Directory to save plot images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best model based on average ROC AUC
    avg_metrics = results_df.group_by('model').agg(
        pl.mean('roc_auc').alias('avg_roc_auc'),
        pl.mean('auprc').alias('avg_auprc')
    )
    best_model = avg_metrics.sort('avg_roc_auc', descending=True)[0, 'model']
    
    # Get unique train-test source combinations
    train_test_pairs = results_df.select(
        pl.col('train_source'), 
        pl.col('test_source')
    ).unique()
    
    # For each train-test source pair
    for row in train_test_pairs.iter_rows(named=True):
        train_source = row['train_source']
        test_source = row['test_source']
        pair_key = f"{train_source}_{test_source}"
        
        # Create ROC curve plot
        plt.figure(figsize=(10, 8))
        
        # Plot for each model
        for model_name in curves_data.keys():
            if pair_key in curves_data[model_name]:
                data = curves_data[model_name][pair_key]
                y_test = data['y_test']
                y_proba = data['y_proba']
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                
                # Determine line style based on if it's the best model
                if model_name == best_model:
                    plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                             label=f"{model_name} (Best) - AUC: {roc_auc_score(y_test, y_proba):.4f}")
                else:
                    plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                             label=f"{model_name} - AUC: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Add diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        # Configure plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: Train on {train_source}, Test on {test_source}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"roc_curve_{train_source}_{test_source}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create PR curve plot
        plt.figure(figsize=(10, 8))
        
        # Plot for each model
        for model_name in curves_data.keys():
            if pair_key in curves_data[model_name]:
                data = curves_data[model_name][pair_key]
                y_test = data['y_test']
                y_proba = data['y_proba']
                
                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                auprc = average_precision_score(y_test, y_proba)
                
                # Determine line style based on if it's the best model
                if model_name == best_model:
                    plt.plot(recall, precision, linewidth=2, linestyle='-', 
                             label=f"{model_name} (Best) - AUPRC: {auprc:.4f}")
                else:
                    plt.plot(recall, precision, linewidth=1, linestyle='--', 
                             label=f"{model_name} - AUPRC: {auprc:.4f}")
        
        # Add baseline (ratio of positive samples)
        baseline = sum(y_test) / len(y_test)
        plt.axhline(y=baseline, color='r', linestyle='-', alpha=0.3, 
                    label=f'Baseline: {baseline:.4f}')
        
        # Configure plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves: Train on {train_source}, Test on {test_source}')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"pr_curve_{train_source}_{test_source}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create combined performance plot
    model_colors = {
        'LogisticRegression': 'blue',
        'RandomForest': 'green', 
        'GradientBoosting': 'red',
        'XGBoost': 'purple',
        'SVM': 'orange',
        'LightGBM': 'brown'
    }
    
    # Average metrics per model
    avg_model_metrics = results_df.group_by('model').agg(
        pl.mean('roc_auc').alias('avg_roc_auc'),
        pl.mean('auprc').alias('avg_auprc'),
        pl.mean('f1').alias('avg_f1')
    )
    
    # Plot summary performance
    plt.figure(figsize=(12, 10))
    
    # Sort by ROC AUC
    avg_model_metrics = avg_model_metrics.sort(by='avg_roc_auc', descending=True)
    
    for row in avg_model_metrics.iter_rows(named=True):
        model_name = row['model']
        x = [row['avg_roc_auc'], row['avg_auprc'], row['avg_f1']]
        
        # Determine marker style based on if it's the best model
        marker_size = 100 if model_name == best_model else 60
        alpha = 1.0 if model_name == best_model else 0.7
        
        plt.scatter([1, 2, 3], x, s=marker_size, label=model_name, 
                    color=model_colors.get(model_name, 'gray'), 
                    alpha=alpha, edgecolors='black')
        
        # Connect points with lines
        plt.plot([1, 2, 3], x, color=model_colors.get(model_name, 'gray'), 
                 alpha=alpha, linestyle='-' if model_name == best_model else '--')
    
    plt.xticks([1, 2, 3], ['ROC AUC', 'AUPRC', 'F1 Score'])
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots saved to directory: {output_dir}")

def leave_one_source_out_validation(df, source_col, target_col, models_config, output_dir='output', random_state=42):
    """
    Implement true leave-one-source-out validation:
    - For each source S:
      - Train on all sources EXCEPT S
      - Test on S only
    """
    # Get unique sources
    sources = df[source_col].unique().to_list()
    print(f"Found {len(sources)} unique sources: {sources}")
    
    results = []
    curves_data = {}  # To store curve data for plotting
    
    # For each model type
    for model_name, (model_class, param_grid) in models_config.items():
        print(f"\n===== Evaluating {model_name} =====")
        curves_data[model_name] = {}
        
        # For each source as test set (leave one out)
        for held_out_source in sources:
            # Filter data for training (all sources except the held-out one)
            train_data = df.filter(pl.col(source_col) != held_out_source)
            X_train, y_train = prepare_data(train_data, target_col, source_col)
            
            # Filter data for testing (only the held-out source)
            test_data = df.filter(pl.col(source_col) == held_out_source)
            X_test, y_test = prepare_data(test_data, target_col, source_col)
            
            # Create pipeline with current model
            pipeline = build_adaptive_pipeline(X_train, model_class)
            
            # Add 'model__' prefix to param grid keys
            model_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
            
            # Train model
            print(f"\nTraining {model_name} on all sources except: {held_out_source}")
            
            # Create a stratified cross-validation strategy to maintain class distribution
            from sklearn.model_selection import StratifiedKFold
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid=model_param_grid,
                cv=cv_strategy,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            print(f"Best parameters for {model_name} excluding {held_out_source}: {best_params}")
            
            # Test on the held-out source
            print(f"Testing {model_name} on held-out source: {held_out_source}")
            
            # Predict and evaluate
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'model': model_name,
                'train_sources': f"all_except_{held_out_source}",
                'test_source': held_out_source,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'auprc': average_precision_score(y_test, y_proba),
                'best_params': str(best_params)
            }
            
            # Store curve data for plotting
            curves_data[model_name][f"all_except_{held_out_source}_{held_out_source}"] = {
                'y_test': y_test,
                'y_proba': y_proba,
                'y_pred': y_pred
            }
            
            results.append(metrics)
            
            print(f"{model_name} trained on all except {held_out_source} → tested on {held_out_source} Performance:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
            print(f"AUPRC Score: {metrics['auprc']:.4f}")
    
    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)
    return results_df, curves_data

# Modified plot_model_curves to work with the leave-one-out format
def plot_leave_one_out_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot ROC and PR curves for all models, with the best model highlighted.
    Adapted for leave-one-source-out validation.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best model based on average ROC AUC
    avg_metrics = results_df.group_by('model').agg(
        pl.mean('roc_auc').alias('avg_roc_auc'),
        pl.mean('auprc').alias('avg_auprc')
    )
    best_model = avg_metrics.sort('avg_roc_auc', descending=True)[0, 'model']
    
    # Get unique test sources (held-out sources)
    test_sources = results_df.select(pl.col('test_source')).unique()
    
    # For each held-out source
    for row in test_sources.iter_rows(named=True):
        held_out_source = row['test_source']
        pair_key = f"all_except_{held_out_source}_{held_out_source}"
        
        # Create ROC curve plot
        plt.figure(figsize=(10, 8))
        
        # Plot for each model
        for model_name in curves_data.keys():
            if pair_key in curves_data[model_name]:
                data = curves_data[model_name][pair_key]
                y_test = data['y_test']
                y_proba = data['y_proba']
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                
                # Determine line style based on if it's the best model
                if model_name == best_model:
                    plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                             label=f"{model_name} (Best) - AUC: {roc_auc_score(y_test, y_proba):.4f}")
                else:
                    plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                             label=f"{model_name} - AUC: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Add diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        # Configure plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: Train on all except {held_out_source}, Test on {held_out_source}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"roc_curve_held_out_{held_out_source}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create PR curve plot
        plt.figure(figsize=(10, 8))
        
        # Plot for each model
        for model_name in curves_data.keys():
            if pair_key in curves_data[model_name]:
                data = curves_data[model_name][pair_key]
                y_test = data['y_test']
                y_proba = data['y_proba']
                
                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                auprc = average_precision_score(y_test, y_proba)
                
                # Determine line style based on if it's the best model
                if model_name == best_model:
                    plt.plot(recall, precision, linewidth=2, linestyle='-', 
                             label=f"{model_name} (Best) - AUPRC: {auprc:.4f}")
                else:
                    plt.plot(recall, precision, linewidth=1, linestyle='--', 
                             label=f"{model_name} - AUPRC: {auprc:.4f}")
        
        # Add baseline (ratio of positive samples)
        baseline = sum(y_test) / len(y_test)
        plt.axhline(y=baseline, color='r', linestyle='-', alpha=0.3, 
                    label=f'Baseline: {baseline:.4f}')
        
        # Configure plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves: Train on all except {held_out_source}, Test on {held_out_source}')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"pr_curve_held_out_{held_out_source}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    
    print(f"All plots saved to directory: {output_dir}")

def regular_kfold_validation(df, target_col, models_config, output_dir='output', random_state=42, n_folds=5):
    """
    Implement regular k-fold cross-validation when no source column is available.
    """
    print(f"Performing {n_folds}-fold cross-validation")
    
    # Prepare data
    X = df.drop(target_col)
    y = df[target_col]
    
    # Create k-fold cross-validator
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    results = []
    curves_data = {}  # To store curve data for plotting
    
    # For each model type
    for model_name, (model_class, param_grid) in models_config.items():
        print(f"\n===== Evaluating {model_name} =====")
        curves_data[model_name] = {}
        
        # Create pipeline with current model
        pipeline = build_adaptive_pipeline(X, model_class)
        
        # Add 'model__' prefix to param grid keys
        model_param_grid = {f"model__{k}": v for k, v in param_grid.items()}
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid=model_param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {best_params}")
        
        # Evaluate using cross-validation
        fold_idx = 0
        for train_index, test_index in cv.split(X, y):
            fold_idx += 1
            
            # Split data for this fold
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train model on this fold
            fold_model = clone(best_model)
            fold_model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'model': model_name,
                'fold': fold_idx,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'auprc': average_precision_score(y_test, y_proba),
                'best_params': str(best_params)
            }
            
            # Store curve data for plotting
            curves_data[model_name][f"fold_{fold_idx}"] = {
                'y_test': y_test,
                'y_proba': y_proba,
                'y_pred': y_pred
            }
            
            results.append(metrics)
            
            print(f"{model_name} Fold {fold_idx} Performance:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
            print(f"AUPRC Score: {metrics['auprc']:.4f}")
    
    # Convert results to Polars DataFrame
    results_df = pl.DataFrame(results)
    return results_df, curves_data

def plot_kfold_curves(curves_data, results_df, output_dir='model_curves'):
    """
    Plot ROC and PR curves for all models in k-fold cross-validation.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best model based on average ROC AUC
    avg_metrics = results_df.group_by('model').agg(
        pl.mean('roc_auc').alias('avg_roc_auc'),
        pl.mean('auprc').alias('avg_auprc')
    )
    best_model = avg_metrics.sort('avg_roc_auc', descending=True)[0, 'model']
    
    # Get unique fold indices
    folds = results_df.select(pl.col('fold')).unique().to_series().sort().to_list()
    
    # For each fold
    for fold in folds:
        # Create ROC curve plot
        plt.figure(figsize=(10, 8))
        
        # Plot for each model
        for model_name in curves_data.keys():
            fold_key = f"fold_{fold}"
            if fold_key in curves_data[model_name]:
                data = curves_data[model_name][fold_key]
                y_test = data['y_test']
                y_proba = data['y_proba']
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                
                # Determine line style based on if it's the best model
                if model_name == best_model:
                    plt.plot(fpr, tpr, linewidth=2, linestyle='-', 
                             label=f"{model_name} (Best) - AUC: {roc_auc_score(y_test, y_proba):.4f}")
                else:
                    plt.plot(fpr, tpr, linewidth=1, linestyle='--', 
                             label=f"{model_name} - AUC: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Add diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        # Configure plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: Fold {fold}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"roc_curve_fold_{fold}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create PR curve plot
        plt.figure(figsize=(10, 8))
        
        # Plot for each model
        for model_name in curves_data.keys():
            fold_key = f"fold_{fold}"
            if fold_key in curves_data[model_name]:
                data = curves_data[model_name][fold_key]
                y_test = data['y_test']
                y_proba = data['y_proba']
                
                # Calculate PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                auprc = average_precision_score(y_test, y_proba)
                
                # Determine line style based on if it's the best model
                if model_name == best_model:
                    plt.plot(recall, precision, linewidth=2, linestyle='-', 
                             label=f"{model_name} (Best) - AUPRC: {auprc:.4f}")
                else:
                    plt.plot(recall, precision, linewidth=1, linestyle='--', 
                             label=f"{model_name} - AUPRC: {auprc:.4f}")
        
        # Add baseline (ratio of positive samples)
        baseline = sum(y_test) / len(y_test)
        plt.axhline(y=baseline, color='r', linestyle='-', alpha=0.3, 
                    label=f'Baseline: {baseline:.4f}')
        
        # Configure plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves: Fold {fold}')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"pr_curve_fold_{fold}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create average performance plot across all folds
    plot_model_avg_performance(results_df, output_dir)
    
    print(f"All plots saved to directory: {output_dir}")

def plot_model_avg_performance(results_df, output_dir):
    """Plot average performance metrics for each model across all folds."""
    
    # Calculate average metrics by model
    avg_performance = results_df.group_by('model').agg(
        pl.mean('accuracy').alias('avg_accuracy'),
        pl.mean('precision').alias('avg_precision'),
        pl.mean('recall').alias('avg_recall'),
        pl.mean('f1').alias('avg_f1'),
        pl.mean('roc_auc').alias('avg_roc_auc'),
        pl.mean('auprc').alias('avg_auprc'),
        pl.std('roc_auc').alias('std_roc_auc'),
        pl.std('auprc').alias('std_auprc')
    ).sort('avg_roc_auc', descending=True)
    
    # Create a bar plot for ROC AUC and AUPRC
    plt.figure(figsize=(12, 8))
    
    models = avg_performance['model'].to_list()
    x = np.arange(len(models))
    width = 0.35
    
    # Plot ROC AUC
    roc_auc = avg_performance['avg_roc_auc'].to_list()
    roc_std = avg_performance['std_roc_auc'].to_list()
    plt.bar(x - width/2, roc_auc, width, label='ROC AUC', alpha=0.7)
    
    # Plot AUPRC
    auprc = avg_performance['avg_auprc'].to_list()
    auprc_std = avg_performance['std_auprc'].to_list()
    plt.bar(x + width/2, auprc, width, label='AUPRC', alpha=0.7)
    
    # Add error bars
    plt.errorbar(x - width/2, roc_auc, yerr=roc_std, fmt='o', color='black', capsize=5)
    plt.errorbar(x + width/2, auprc, yerr=auprc_std, fmt='o', color='black', capsize=5)
    
    # Configure plot
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Average Model Performance Across All Folds')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    
    # Add values on top of bars
    for i, v in enumerate(roc_auc):
        plt.text(i - width/2, v + 0.03, f"{v:.3f}", ha='center')
    
    for i, v in enumerate(auprc):
        plt.text(i + width/2, v + 0.03, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, "avg_model_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run model comparison pipeline')
    parser.add_argument('--input', required=True, help='Path to the input parquet file')
    parser.add_argument('--output_dir', required=True, help='Directory to save output files')
    parser.add_argument('--source_column', default='source', help='Name of the source column')
    parser.add_argument('--target_column', default='annot', help='Name of the target column')
    parser.add_argument('--drop_columns', help='Comma-separated columns to drop')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set consistent random seed for reproducibility
    random_seed = args.random_seed
    np.random.seed(random_seed)
    
    # Define output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse columns to drop
    drop_cols = args.drop_columns.split(',') if args.drop_columns else []
    
    # Load the data using Polars
    df = pl.read_parquet(args.input)
    if drop_cols:
        df = df.drop(drop_cols)
    
    # Specify column names
    source_column = args.source_column
    target_column = args.target_column
    
    # Define models to compare
    models_config = define_models_config()
    
    # Check if source column exists in the dataframe
    has_source_column = source_column in df.columns
    
    # Check data before proceeding
    print("\n===== Data Quality Check =====")
    
    # Check class distribution overall
    class_counts = df[target_column].value_counts()
    print(f"Overall class distribution:\n{class_counts}")
    
    if has_source_column:
        # Check class distribution by source
        class_by_source = df.group_by([source_column, target_column]).agg(
            pl.len().alias('count')
        ).pivot(
            index=source_column,
            on=target_column,
            values='count'
        )
        print("\nClass distribution by source:")
        print(class_by_source)
        
        # Run leave-one-source-out validation
        results, curves_data = leave_one_source_out_validation(
            df, source_column, target_column, models_config, 
            output_dir=output_dir, random_state=random_seed
        )
        
        # Save detailed results
        results.write_csv(os.path.join(output_dir, "leave_one_out_results.csv"))
        
        # Create summary table by model and test source
        summary = results.group_by(['model', 'test_source']).agg(
            pl.mean('accuracy').alias('avg_accuracy'),
            pl.mean('f1').alias('avg_f1'),
            pl.mean('roc_auc').alias('avg_roc_auc'),
            pl.mean('auprc').alias('avg_auprc')
        ).sort(['model', 'test_source'])
        
        summary.write_csv(os.path.join(output_dir, "leave_one_out_summary.csv"))
        
        # Generate and save plots
        plot_leave_one_out_curves(curves_data, results, output_dir=output_dir)
    else:
        # No source column, perform regular k-fold cross-validation
        print("\nNo source column found. Performing regular k-fold cross-validation.")
        
        # Run k-fold cross-validation
        results, curves_data = regular_kfold_validation(
            df, target_column, models_config, 
            output_dir=output_dir, random_state=random_seed,
            n_folds=args.n_folds
        )
        
        # Save detailed results
        results.write_csv(os.path.join(output_dir, "kfold_results.csv"))
        
        # Create summary table by model
        summary = results.group_by(['model']).agg(
            pl.mean('accuracy').alias('avg_accuracy'),
            pl.mean('f1').alias('avg_f1'),
            pl.mean('roc_auc').alias('avg_roc_auc'),
            pl.mean('auprc').alias('avg_auprc')
        ).sort(['model'])
        
        summary.write_csv(os.path.join(output_dir, "kfold_summary.csv"))
        
        # Generate and save plots
        plot_kfold_curves(curves_data, results, output_dir=output_dir)
    
    # Train the best overall model on all data with consistent random state
    best_model, best_params, best_model_name = train_best_production_model(
        df, source_column if has_source_column else None, target_column, 
        results, output_dir=output_dir, random_state=random_seed
    )
    
    print(f"\nFinal model trained and saved as '{os.path.join(output_dir, f'final_model_{best_model_name}.pkl')}'")
    print(f"Production model parameters: {best_params}")
    print(f"All results and visualizations have been saved to the '{output_dir}' directory")
