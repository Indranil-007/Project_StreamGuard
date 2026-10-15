"""Project StreamGuard - Feature Engineering, Encoding, and Scaling Module.
Constructs domain metrics, creates leak-free scikit-learn preprocessing pipelines,
and executes stratified train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class OTTFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers OTT-specific domain metrics without data leakage.
    - Engagement_Efficiency_Index: Watch_Hours_Monthly / Monthly_Fee_USD (Watch-to-Fee Ratio)
    - Engagement_Index: Average of z-score normalized watch hours and login frequency,
      using training set mean and std.
    """
    def __init__(self):
        self.watch_mean_ = None
        self.watch_std_ = None
        self.login_mean_ = None
        self.login_std_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        self.watch_mean_ = df["Watch_Hours_Monthly"].mean()
        self.watch_std_ = df["Watch_Hours_Monthly"].std()
        self.login_mean_ = df["Login_Frequency_Monthly"].mean()
        self.login_std_ = df["Login_Frequency_Monthly"].std()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        
        # 1. Engagement_Efficiency_Index (guarding against 0 or negative fee)
        df["Engagement_Efficiency_Index"] = df["Watch_Hours_Monthly"] / np.maximum(df["Monthly_Fee_USD"], 1e-4)
        
        # 2. Engagement_Index: Composite metric = mean of standardized watch hours and login frequency
        # Formula: 0.5 * [(Watch_Hours - mean_train) / std_train + (Login_Freq - mean_train) / std_train]
        watch_z = (df["Watch_Hours_Monthly"] - self.watch_mean_) / (self.watch_std_ + 1e-8)
        login_z = (df["Login_Frequency_Monthly"] - self.login_mean_) / (self.login_std_ + 1e-8)
        df["Engagement_Index"] = (watch_z + login_z) / 2.0
        
        return df

def get_preprocessor(continuous_cols, categorical_cols):
    """Builds a leak-free scikit-learn ColumnTransformer.
    StandardScaler for continuous features, OneHotEncoder (drop_first) for categoricals.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_data(data_path="data/processed/streamguard_clean.csv", random_state=42):
    """Loads clean data, splits 80/20 stratified on Churned,
    applies feature engineering and ColumnTransformer strictly fit on training set.
    """
    df = pd.read_csv(data_path)
    
    # Separate targets and features
    X = df.drop(columns=["Customer_ID", "Churned", "Lifetime_Value_USD"])
    y_churn = df["Churned"]
    y_clv = df["Lifetime_Value_USD"]
    
    # Stratified 80/20 train/test split on Churned
    X_train_raw, X_test_raw, y_churn_train, y_churn_test, y_clv_train, y_clv_test = train_test_split(
        X, y_churn, y_clv, test_size=0.20, random_state=random_state, stratify=y_churn
    )
    
    # Apply Domain Feature Engineering
    engineer = OTTFeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train_raw)
    X_test_eng = engineer.transform(X_test_raw)
    
    continuous_cols = [
        "Monthly_Fee_USD", "Watch_Hours_Monthly", "Login_Frequency_Monthly",
        "Support_Interactions", "Tenure_Months",
        "Engagement_Efficiency_Index", "Engagement_Index"
    ]
    categorical_cols = ["Subscription_Type", "Preferred_Device"]
    
    preprocessor = get_preprocessor(continuous_cols, categorical_cols)
    
    # Fit transformer strictly on training features
    X_train_proc = preprocessor.fit_transform(X_train_eng)
    X_test_proc = preprocessor.transform(X_test_eng)
    
    # Retrieve clean feature names from preprocessor
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
    all_feature_names = continuous_cols + cat_feature_names
    
    X_train_df = pd.DataFrame(X_train_proc, columns=all_feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=all_feature_names, index=X_test_raw.index)
    
    print("=== Feature Engineering & Preprocessing Verification ===")
    print(f"X_train shape: {X_train_df.shape}, X_test shape: {X_test_df.shape}")
    print(f"Features: {all_feature_names}")
    
    # Confirm scaled columns mean ~ 0, std ~ 1 on training set
    print("\nTraining Set Continuous Feature Means (Target ~ 0):")
    print(X_train_df[continuous_cols].mean().round(4).to_dict())
    print("\nTraining Set Continuous Feature Stds (Target ~ 1):")
    print(X_train_df[continuous_cols].std(ddof=0).round(4).to_dict())
    
    return {
        "X_train": X_train_df,
        "X_test": X_test_df,
        "y_churn_train": y_churn_train,
        "y_churn_test": y_churn_test,
        "y_clv_train": y_clv_train,
        "y_clv_test": y_clv_test,
        "preprocessor": preprocessor,
        "engineer": engineer,
        "feature_names": all_feature_names
    }

if __name__ == "__main__":
    prepare_data()
