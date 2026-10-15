"""Project StreamGuard - Feature Selection and Validation Setup Module.
Executes RFECV to determine optimal feature subset, saves ranking results,
and defines the global StratifiedKFold cross-validation strategy.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import prepare_data

def get_cv_split(n_splits=5, random_state=42):
    """Returns the standardized StratifiedKFold object to be reused across all models."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def run_feature_selection(data_dict=None, results_dir="outputs/results", figures_dir="outputs/figures", random_state=42):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    if data_dict is None:
        data_dict = prepare_data(random_state=random_state)
        
    X_train = data_dict["X_train"]
    y_train = data_dict["y_churn_train"]
    cv = get_cv_split(n_splits=5, random_state=random_state)
    
    # Verify Fold Stratification
    print("=== Cross-Validation Stratification Proof ===")
    overall_churn_ratio = y_train.mean()
    print(f"Overall Training Churn Ratio: {overall_churn_ratio:.2%}")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        fold_y_train = y_train.iloc[train_idx]
        fold_y_val = y_train.iloc[val_idx]
        print(f"Fold {fold_idx}: Train Churn = {fold_y_train.mean():.2%} ({fold_y_train.sum()}/{len(fold_y_train)}) | "
              f"Val Churn = {fold_y_val.mean():.2%} ({fold_y_val.sum()}/{len(fold_y_val)})")
        
    # RFECV with Random Forest estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=3,
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)
    
    # Compile ranking table
    feature_names = X_train.columns.tolist()
    ranking_df = pd.DataFrame({
        "Feature": feature_names,
        "Ranking": rfecv.ranking_,
        "Selected": rfecv.support_
    }).sort_values(by=["Ranking", "Feature"]).reset_index(drop=True)
    
    ranking_file = os.path.join(results_dir, "feature_ranking.csv")
    ranking_df.to_csv(ranking_file, index=False)
    print(f"\nFeature ranking saved to {ranking_file}")
    print("\nFeature Ranking Table:")
    print(ranking_df)
    
    selected_features = [f for f, s in zip(feature_names, rfecv.support_) if s]
    dropped_features = [f for f, s in zip(feature_names, rfecv.support_) if not s]
    print(f"\nOptimal number of features: {rfecv.n_features_}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Dropped features ({len(dropped_features)}): {dropped_features}")
    
    # Plot RFECV performance vs number of features
    plt.figure(figsize=(9, 5))
    num_features_range = range(rfecv.min_features_to_select, len(feature_names) + 1)
    # rfecv.cv_results_['mean_test_score']
    mean_scores = rfecv.cv_results_["mean_test_score"]
    std_scores = rfecv.cv_results_["std_test_score"]
    
    plt.plot(num_features_range, mean_scores, marker="o", color="royalblue", lw=2, label="Mean CV ROC-AUC")
    plt.fill_between(num_features_range, mean_scores - std_scores, mean_scores + std_scores, alpha=0.15, color="royalblue")
    plt.axvline(x=rfecv.n_features_, color="firebrick", linestyle="--", label=f"Optimal N = {rfecv.n_features_}")
    plt.title("Recursive Feature Elimination with CV (RFECV)")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Cross-Validated ROC-AUC Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    rfecv_fig_path = os.path.join(figures_dir, "rfecv_feature_selection.png")
    plt.savefig(rfecv_fig_path, dpi=300)
    plt.close()
    print(f"Saved RFECV score plot to {rfecv_fig_path}")
    
    return {
        "selected_features": selected_features,
        "dropped_features": dropped_features,
        "ranking_df": ranking_df,
        "cv": cv,
        "rfecv": rfecv
    }

if __name__ == "__main__":
    run_feature_selection()
