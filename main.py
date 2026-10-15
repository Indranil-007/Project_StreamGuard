"""Project StreamGuard - End-to-End Orchestrator Pipeline.
Runs Phase 0 through Phase 6 in order, verifying all checkpoints and persisting
all figures, evaluation tables, model artifacts, and reports.
"""

import os
import sys
import time

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.generate_data import generate_streamguard_data
from src.cleaning import clean_data
from src.features import prepare_data
from src.selection import run_feature_selection
from src.models import evaluate_baseline_models, tune_ensemble_models
from src.evaluate import run_evaluation
from src.regression import run_clv_regression

def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("STARTING PROJECT STREAMGUARD: OTT CHURN & REVENUE INTELLIGENCE")
    print("=" * 70)
    
    # Phase 0: Setup and Dataset Generation
    print("\n>>> [PHASE 0] Generating Synthetic Telemetry Dataset...")
    raw_df = generate_streamguard_data()
    print(f"[+] Phase 0 Complete: Raw data shape = {raw_df.shape}")
    
    # Phase 1: Data Cleaning and Wrangling
    print("\n>>> [PHASE 1] Cleaning Telemetry Data, Imputing & Outlier Capping...")
    clean_df = clean_data()
    print(f"[+] Phase 1 Complete: Cleaned data shape = {clean_df.shape}")
    
    # Phase 2: Feature Engineering & Preprocessing
    print("\n>>> [PHASE 2] Feature Engineering, OHE & Leak-Free Scaling...")
    data_dict = prepare_data()
    print(f"[+] Phase 2 Complete: X_train = {data_dict['X_train'].shape}, X_test = {data_dict['X_test'].shape}")
    
    # Phase 3: Feature Selection & Validation Setup
    print("\n>>> [PHASE 3] Recursive Feature Elimination (RFECV) & Stratified K-Fold...")
    selection_dict = run_feature_selection(data_dict)
    print(f"[+] Phase 3 Complete: Selected {len(selection_dict['selected_features'])} features.")
    
    # Phase 4 & 5: Baseline Modeling, Hyperparameter Tuning & Final Ensemble Evaluation
    print("\n>>> [PHASE 4 & 5] Training Baselines, Tuning Ensembles & Held-Out Testing...")
    cv = selection_dict["cv"]
    selected_features = selection_dict["selected_features"]
    X_train_sel = data_dict["X_train"][selected_features]
    y_train = data_dict["y_churn_train"]
    
    baseline_results, baselines = evaluate_baseline_models(X_train_sel, y_train, cv)
    tuning_results = tune_ensemble_models(X_train_sel, y_train, cv, n_iter=12)
    models_dict = {"baselines": baselines, "tuning_results": tuning_results}
    
    eval_res = run_evaluation(data_dict, selection_dict, models_dict)
    print(f"[+] Phase 5 Complete: Champion Model = {eval_res['best_model_name']} "
          f"(Test ROC-AUC: {eval_res['test_metrics']['roc_auc']:.4f}, Recall: {eval_res['test_metrics']['recall']:.4f})")
    
    # Phase 6: Regression Track (CLV Forecasting)
    print("\n>>> [PHASE 6] Customer Lifetime Value (CLV) Regression Modeling...")
    reg_res = run_clv_regression(data_dict)
    print(f"[+] Phase 6 Complete: Champion Regressor = {reg_res['champion_name']} "
          f"(Test R2: {reg_res['test_metrics']['r2']:.4f}, MAE: ${reg_res['test_metrics']['mae']:.2f})")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
