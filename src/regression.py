"""Project StreamGuard - Regression Track (Customer Lifetime Value Forecasting).
Predicts continuous Lifetime_Value_USD using preprocessed subscriber telemetry.
Benchmarks Linear Regression against tuned Random Forest Regressor and Gradient Boosting Regressor.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import prepare_data

def run_clv_regression(data_dict=None, results_dir="outputs/results", figures_dir="outputs/figures", models_dir="outputs/models", random_state=42):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    if data_dict is None:
        data_dict = prepare_data(random_state=random_state)
        
    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_train = data_dict["y_clv_train"]
    y_test = data_dict["y_clv_test"]
    
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    print("\n=== CLV Regression Track: Model Benchmarking & Tuning ===")
    
    # 1. Baseline: Linear Regression
    lin_reg = LinearRegression()
    
    # 2. Tuned Random Forest Regressor
    rf_reg = RandomForestRegressor(random_state=random_state)
    rf_param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 6, 10, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    rf_search = RandomizedSearchCV(
        rf_reg, rf_param_dist, n_iter=12, scoring="neg_root_mean_squared_error",
        cv=cv, random_state=random_state, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf_reg = rf_search.best_estimator_
    print(f"Random Forest Regressor Best Params: {rf_search.best_params_}")
    
    # 3. Tuned Gradient Boosting Regressor
    gb_reg = GradientBoostingRegressor(random_state=random_state)
    gb_param_dist = {
        "n_estimators": [50, 80, 120],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4]
    }
    gb_search = RandomizedSearchCV(
        gb_reg, gb_param_dist, n_iter=12, scoring="neg_root_mean_squared_error",
        cv=cv, random_state=random_state, n_jobs=-1
    )
    gb_search.fit(X_train, y_train)
    best_gb_reg = gb_search.best_estimator_
    print(f"Gradient Boosting Regressor Best Params: {gb_search.best_params_}")
    
    models = {
        "Linear Regression": lin_reg,
        "Random Forest Regressor (Tuned)": best_rf_reg,
        "Gradient Boosting Regressor (Tuned)": best_gb_reg
    }
    
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
    records = []
    
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        records.append({
            "Model": name,
            "CV_MAE": -np.mean(scores["test_neg_mean_absolute_error"]),
            "CV_RMSE": -np.mean(scores["test_neg_root_mean_squared_error"]),
            "CV_R2": np.mean(scores["test_r2"])
        })
        
    comp_df = pd.DataFrame(records).sort_values(by="CV_R2", ascending=False)
    csv_path = os.path.join(results_dir, "clv_regression_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    print("\nCLV Regression CV Comparison:")
    print(comp_df.to_string(index=False))
    
    # Champion Regressor
    champion_name = comp_df.iloc[0]["Model"]
    champion_reg = models[champion_name]
    print(f"\nChampion Regressor: {champion_name}")
    
    # Fit on full training set and evaluate on test set
    champion_reg.fit(X_train, y_train)
    y_test_pred = champion_reg.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n=== Held-Out Test Set CLV Performance ===")
    print(f"MAE:  ${test_mae:.2f}")
    print(f"RMSE: ${test_rmse:.2f}")
    print(f"R2:   {test_r2:.4f}")
    
    # Plot Predicted vs Actual
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_test_pred, alpha=0.6, color="royalblue", edgecolors="k")
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction (y = x)")
    plt.title(f"CLV Forecasting: Predicted vs. Actual ({champion_name})")
    plt.xlabel("Actual Lifetime Value (USD)")
    plt.ylabel("Predicted Lifetime Value (USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(figures_dir, "clv_predicted_vs_actual.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Predicted vs. actual plot saved to {plot_path}")
    
    # Save marketing acquisition note
    marketing_note = f"""### Guiding Marketing Customer Acquisition Budgets (CAC) Using CLV Forecasts

1. **Strategic Budget Allocation**:
   - The CLV model achieves an R2 of {test_r2:.2f} and an MAE of ${test_mae:.2f}, accurately forecasting the future cash flows of incoming subscriber profiles.
   - Growth marketing can establish upper-bound Customer Acquisition Cost (CAC) thresholds based on segment-level predicted CLV to enforce the industry standard 3:1 CLV-to-CAC ratio.
2. **Channel Optimization**:
   - Telemetry-based CLV predictions allow marketing channels (social ads, Smart TV pre-installs, mobile app store bundles) to be evaluated not just on conversion volume, but on expected lifetime revenue yield.
   - High-value subscriber personas (e.g. Standard/Premium tiers on Smart TVs with high projected engagement) justify elevated paid-search bids, whereas lower-CLV segments receive cost-efficient organic onboarding funnels.
"""
    guidance_path = os.path.join(results_dir, "clv_marketing_guidance.txt")
    with open(guidance_path, "w") as f:
        f.write(marketing_note)
    print(f"Marketing guidance note saved to {guidance_path}")
    
    # Save model artifact
    model_artifact_path = os.path.join(models_dir, "best_clv_regressor.joblib")
    joblib.dump({
        "model": champion_reg,
        "model_name": champion_name,
        "test_metrics": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2}
    }, model_artifact_path)
    print(f"Best CLV regressor saved to {model_artifact_path}")
    
    return {
        "comp_df": comp_df,
        "champion_name": champion_name,
        "champion_reg": champion_reg,
        "test_metrics": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2}
    }

if __name__ == "__main__":
    run_clv_regression()
