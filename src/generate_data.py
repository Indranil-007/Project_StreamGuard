"""Project StreamGuard - Synthetic Data Generator.
Generates realistic OTT platform telemetry data with intentional anomalies and missing values.
"""

import numpy as np
import pandas as pd
import os

def generate_streamguard_data(n_samples=600, random_state=42, output_path="data/raw/streamguard_raw.csv", verbose=False):
    np.random.seed(random_state)
    
    # 1. Customer IDs
    customer_ids = [f"SG-{1000 + i}" for i in range(n_samples)]
    
    # 2. Subscription Types & Preferred Devices
    sub_types = np.random.choice(["Basic", "Standard", "Premium"], size=n_samples, p=[0.45, 0.35, 0.20])
    devices = np.random.choice(["Mobile", "Smart TV", "Laptop", "Tablet"], size=n_samples, p=[0.35, 0.30, 0.20, 0.15])
    
    # 3. Monthly Fee USD (dependent on tier + slight variation)
    base_fees = {"Basic": 8.99, "Standard": 14.99, "Premium": 19.99}
    monthly_fees = np.array([base_fees[st] + np.random.uniform(-0.5, 0.5) for st in sub_types])
    monthly_fees = np.round(monthly_fees, 2)
    
    # 4. Tenure in months (1 to 48 months)
    tenure_months = np.random.randint(1, 49, size=n_samples)
    
    # 5. Engagement telemetry: Watch hours & Login frequency
    # Basic users might watch 15-40h, Premium might watch 30-70h
    tier_watch_bonus = {"Basic": 0, "Standard": 15, "Premium": 30}
    watch_hours = np.array([
        np.random.normal(loc=25 + tier_watch_bonus[st], scale=10) for st in sub_types
    ])
    watch_hours = np.clip(watch_hours, 2.0, 120.0).round(1)
    
    # Login frequency roughly correlated with watch hours
    login_freq = np.clip(np.random.normal(loc=watch_hours / 2.5, scale=4), 1, 45).round(0).astype(int)
    
    # Support interactions (Poisson distribution: mean 2)
    support_interactions = np.random.poisson(lam=2.2, size=n_samples)
    
    # 6. Churn probability modeling
    # Lower watch hours, lower logins, higher support interactions, higher fees increase churn risk
    # Standardize factors to logit scale
    z = (
        -0.8 
        - 0.05 * (watch_hours - 35)
        - 0.08 * (login_freq - 14)
        + 0.45 * (support_interactions - 2)
        + 0.06 * (monthly_fees - 13)
        - 0.03 * (tenure_months - 20)
    )
    churn_probs = 1 / (1 + np.exp(-z))
    churned = (np.random.rand(n_samples) < churn_probs).astype(int)
    
    # Ensure churn rate is approx 25-30%
    current_churn_rate = churned.mean()
    
    # 7. Customer Lifetime Value (CLV in USD)
    # Historic value + expected future value based on tenure, fee, and churn status
    # Churners realized lifetime is limited by their tenure; active users have higher expected CLV
    clv = monthly_fees * tenure_months * (1.0 - 0.4 * churned) + np.random.normal(0, 20, size=n_samples)
    clv = np.clip(np.round(clv, 2), 15.0, None)
    
    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Subscription_Type": sub_types,
        "Preferred_Device": devices,
        "Monthly_Fee_USD": monthly_fees,
        "Watch_Hours_Monthly": watch_hours,
        "Login_Frequency_Monthly": login_freq.astype(float),
        "Support_Interactions": support_interactions.astype(float),
        "Tenure_Months": tenure_months,
        "Churned": churned,
        "Lifetime_Value_USD": clv
    })
    
    # 8. Inject mess
    # a. ~8% missing values in numeric telemetry
    mask_watch = np.random.rand(n_samples) < 0.08
    mask_login = np.random.rand(n_samples) < 0.08
    mask_support = np.random.rand(n_samples) < 0.08
    df.loc[mask_watch, "Watch_Hours_Monthly"] = np.nan
    df.loc[mask_login, "Login_Frequency_Monthly"] = np.nan
    df.loc[mask_support, "Support_Interactions"] = np.nan
    
    # b. Missing values in categorical columns (~5%)
    mask_sub = np.random.rand(n_samples) < 0.05
    mask_dev = np.random.rand(n_samples) < 0.05
    df.loc[mask_sub, "Subscription_Type"] = np.nan
    df.loc[mask_dev, "Preferred_Device"] = np.nan
    
    # c. Corrupted fees (e.g. 99999.0)
    corrupt_fee_idx = np.random.choice(n_samples, size=4, replace=False)
    df.loc[corrupt_fee_idx, "Monthly_Fee_USD"] = 99999.00
    
    # d. Negative watch hours
    neg_watch_idx = np.random.choice(np.where(~df["Watch_Hours_Monthly"].isna())[0], size=5, replace=False)
    df.loc[neg_watch_idx, "Watch_Hours_Monthly"] = -1 * np.abs(df.loc[neg_watch_idx, "Watch_Hours_Monthly"])
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Dataset generated with shape {df.shape} saved to {output_path}")
        print(f"Churn rate: {df['Churned'].mean():.2%}")
        print("\nMissing values:")
        print(df.isna().sum())
        print("\nDescribe summary:")
        print(df.describe())
    
    return df

if __name__ == "__main__":
    generate_streamguard_data(verbose=True)

