# Project StreamGuard - Progress Tracking

| Phase | Status | Notes & Assumptions |
|-------|--------|---------------------|
| Phase 0: Setup and Dataset | Completed | Generated 600 rows with synthetic OTT telemetry, ~8% missingness, corrupted fees ($99,999), negative watch hours, ~32.7% churn rate. |
| Phase 1: Exploration, Wrangling, Cleaning | Completed | Converted negative watch hours to NaN, imputed medians/modes, winsorized IQR outliers (capped $99,999 fee at $24.99), saved figures. |
| Phase 2: Feature Engineering, Encoding, Scaling | Completed | Engineered Engagement_Efficiency_Index & Engagement_Index; OHE drop_first; ColumnTransformer scaler fit strictly on train (80/20 stratified split). |
| Phase 3: Feature Selection and Validation Setup | Completed | RFE with RF estimator selected 10 optimal features; StratifiedKFold(5) verified with stable ~32.3-33.3% churn ratio across all folds. |
| Phase 4: Baseline Models and Hyperparameter Tuning | Completed | Evaluated Logistic Regression, KNN, SVM on Stratified 5-Fold CV; tuned RF and AdaBoost via budget RandomizedSearchCV (n_iter=12). |
| Phase 5: Ensembles and Final Evaluation | Completed | Benchmarked 5 classifiers across 5-fold CV; champion evaluated on test set (ROC-AUC: 0.841, Recall: 87.2%); saved plots, models, recommendations. |
| Phase 6: Regression Track (CLV Forecasting) | Completed | Benchmarked Linear Regression, Tuned RF & Gradient Boosting Regressors across 5-fold CV; Champion GB Regressor scored test R2 = 0.882, MAE = $54.85; saved plots and CAC guidance. |
| Phase 7: Documentation and Deliverables | Completed | Authored README.md with resume summary, requirements.txt, and assembled executable notebooks/streamguard.ipynb. |








