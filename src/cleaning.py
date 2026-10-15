"""Project StreamGuard - Data Exploration, Wrangling, and Cleaning Module.
Handles inspection, impossible value sanitization, median/mode imputation,
IQR outlier detection & capping (winsorization), and visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(
    raw_path="data/raw/streamguard_raw.csv",
    clean_path="data/processed/streamguard_clean.csv",
    figures_dir="outputs/figures"
):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    
    # 1. Load and Inspect
    df = pd.read_csv(raw_path)
    print("=== Raw Data Inspection ===")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pcts})
    print("\nMissing Values Table:")
    print(missing_df[missing_df["Missing Count"] > 0])
    
    # Plot missing values bar chart
    plt.figure(figsize=(10, 5))
    missing_cols = missing_df[missing_df["Missing Count"] > 0]
    sns.barplot(x=missing_cols.index, y=missing_cols["Missing Count"], hue=missing_cols.index, palette="viridis", legend=False)
    plt.title("Missing Value Counts per Telemetry Column (Raw Data)")
    plt.xlabel("Features")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=25)
    plt.tight_layout()
    missing_fig_path = os.path.join(figures_dir, "missing_values_raw.png")
    plt.savefig(missing_fig_path, dpi=300)
    plt.close()
    print(f"Saved missing values chart to {missing_fig_path}")

    # 2. Handle impossible values: negative watch hours -> NaN
    neg_watch_count = (df["Watch_Hours_Monthly"] < 0).sum()
    print(f"\nFound {neg_watch_count} negative Watch_Hours_Monthly values. Converting to NaN.")
    df.loc[df["Watch_Hours_Monthly"] < 0, "Watch_Hours_Monthly"] = np.nan
    
    # 3. Impute: Median for numerical columns, Mode for categorical columns
    num_cols = ["Monthly_Fee_USD", "Watch_Hours_Monthly", "Login_Frequency_Monthly", 
                "Support_Interactions", "Tenure_Months", "Lifetime_Value_USD"]
    cat_cols = ["Subscription_Type", "Preferred_Device"]
    
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed numerical column '{col}' with median: {median_val:.2f}")
        
    for col in cat_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed categorical column '{col}' with mode: '{mode_val}'")

    # 4. Outlier Detection with IQR method and Before/After comparison
    # Track outliers count per column
    outlier_counts = {}
    iqr_bounds = {}
    
    # Snapshot before outlier treatment for plotting
    df_before_capping = df.copy()
    
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_bounds[col] = (lower_bound, upper_bound)
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_counts[col] = len(outliers)
        
        # 5. Treat corrupted values / extreme outliers by capping (Winsorization)
        # Retaining records prevents loss of valuable churn signals while suppressing extreme distortion
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    print("\n=== Outlier Detection (IQR Method) ===")
    for col, count in outlier_counts.items():
        lb, ub = iqr_bounds[col]
        print(f"Feature: {col} | Bounds: [{lb:.2f}, {ub:.2f}] | Outliers Capped: {count}")

    # Generate Before vs. After Outlier Boxplots for key continuous features
    plot_cols = ["Monthly_Fee_USD", "Watch_Hours_Monthly", "Login_Frequency_Monthly", "Support_Interactions"]
    fig, axes = plt.subplots(len(plot_cols), 2, figsize=(12, 10))
    fig.suptitle("Telemetry Features: Before vs. After IQR Outlier Capping (Winsorization)", fontsize=14)
    
    for i, col in enumerate(plot_cols):
        sns.boxplot(x=df_before_capping[col], ax=axes[i, 0], color="salmon")
        axes[i, 0].set_title(f"{col} (Before Capping)")
        
        sns.boxplot(x=df[col], ax=axes[i, 1], color="mediumseagreen")
        axes[i, 1].set_title(f"{col} (After Capping)")
        
    plt.tight_layout()
    outlier_fig_path = os.path.join(figures_dir, "outliers_before_after.png")
    plt.savefig(outlier_fig_path, dpi=300)
    plt.close()
    print(f"Saved outlier comparison boxplot to {outlier_fig_path}")

    # 6. Save cleaned data
    df.to_csv(clean_path, index=False)
    print(f"\nCleaned dataset saved to {clean_path} with shape {df.shape}")
    
    # Confirmation checks
    assert df.isna().sum().sum() == 0, "Missing values still remain!"
    assert (df["Watch_Hours_Monthly"] < 0).sum() == 0, "Negative watch hours detected!"
    fee_upper = iqr_bounds["Monthly_Fee_USD"][1]
    assert df["Monthly_Fee_USD"].max() <= fee_upper + 1e-4, "Monthly fee exceeds IQR upper bound!"
    print("DONE WHEN check passed: 0 missing values, valid watch hours, fee capped within bounds.")
    
    return df

if __name__ == "__main__":
    clean_data()
