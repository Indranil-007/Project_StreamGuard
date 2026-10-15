"""Project StreamGuard - Baseline Models and Hyperparameter Tuning Module.
Trains baseline classifiers (Logistic Regression, KNN, SVM) and performs
budget-constrained RandomizedSearchCV for Random Forest and AdaBoost.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import prepare_data
from src.selection import get_cv_split, run_feature_selection

def evaluate_baseline_models(X_train, y_train, cv):
    """Trains and benchmarks baseline classifiers with stratified 5-fold CV."""
    baselines = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(class_weight="balanced", probability=True, random_state=42)
    }
    
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    baseline_results = {}
    
    print("=== Baseline Classifiers Stratified 5-Fold CV Performance ===")
    for name, model in baselines.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        res = {
            "Accuracy": np.mean(scores["test_accuracy"]),
            "Precision": np.mean(scores["test_precision"]),
            "Recall": np.mean(scores["test_recall"]),
            "F1": np.mean(scores["test_f1"]),
            "ROC-AUC": np.mean(scores["test_roc_auc"])
        }
        baseline_results[name] = res
        print(f"\n[{name}]")
        for metric, val in res.items():
            print(f"  {metric}: {val:.4f}")
            
    return baseline_results, baselines

def tune_ensemble_models(X_train, y_train, cv, n_iter=12, random_state=42):
    """Tunes Random Forest and AdaBoost using RandomizedSearchCV with a strict budget."""
    print(f"\n=== Hyperparameter Tuning with RandomizedSearchCV (n_iter={n_iter}) ===")
    
    # 1. Random Forest Tuning
    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    rf_param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 4, 6, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }
    
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    print("\n[Random Forest Tuned]")
    print(f"  Best ROC-AUC: {rf_search.best_score_:.4f}")
    print(f"  Best Parameters: {rf_search.best_params_}")
    
    # 2. AdaBoost Tuning with DecisionTreeClassifier base estimator
    # Accommodate both `estimator` (sklearn >= 1.2) and `base_estimator`
    base_tree = DecisionTreeClassifier(random_state=random_state)
    try:
        ada = AdaBoostClassifier(estimator=base_tree, random_state=random_state)
        param_prefix = "estimator__"
    except TypeError:
        ada = AdaBoostClassifier(base_estimator=base_tree, random_state=random_state)
        param_prefix = "base_estimator__"
        
    ada_param_dist = {
        f"{param_prefix}max_depth": [1, 2, 3],
        "n_estimators": [50, 80, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
    }
    
    ada_search = RandomizedSearchCV(
        estimator=ada,
        param_distributions=ada_param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1
    )
    ada_search.fit(X_train, y_train)
    print("\n[AdaBoost Tuned]")
    print(f"  Best ROC-AUC: {ada_search.best_score_:.4f}")
    print(f"  Best Parameters: {ada_search.best_params_}")
    
    return {
        "rf_search": rf_search,
        "ada_search": ada_search,
        "best_rf": rf_search.best_estimator_,
        "best_ada": ada_search.best_estimator_
    }

def run_models_pipeline():
    data_dict = prepare_data()
    sel_res = run_feature_selection(data_dict)
    selected_features = sel_res["selected_features"]
    cv = sel_res["cv"]
    
    X_train_sel = data_dict["X_train"][selected_features]
    y_train = data_dict["y_churn_train"]
    
    baseline_results, baselines = evaluate_baseline_models(X_train_sel, y_train, cv)
    tuning_results = tune_ensemble_models(X_train_sel, y_train, cv, n_iter=12)
    
    return {
        "baseline_results": baseline_results,
        "baselines": baselines,
        "tuning_results": tuning_results,
        "selected_features": selected_features,
        "cv": cv
    }

if __name__ == "__main__":
    run_models_pipeline()
