import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score
)
import shap
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

print("Starting Data Processing & Modeling pipeline...")

# 1. LOAD DATA

print("Loading dataset...")

file_path = "online_retail_II.xlsx"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found.")

df = pd.read_excel(file_path)

print(f"Initial shape: {df.shape}")

# 2. CLEAN DATA

print("Cleaning data...")

df = df.dropna(subset=["Customer ID"])

df["Invoice"] = df["Invoice"].astype(str)

# Remove cancelled orders
df = df[~df["Invoice"].str.startswith("C")]

# Remove invalid values
df = df[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Revenue
df["Revenue"] = df["Quantity"] * df["Price"]

print(f"Cleaned shape: {df.shape}")

# 3. TRAIN / TARGET

max_date = df["InvoiceDate"].max()

# Last 90 days used to determine churn
cutoff_date = max_date - pd.Timedelta(days=90)

train_df = df[df["InvoiceDate"] < cutoff_date]
target_df = df[df["InvoiceDate"] >= cutoff_date]

# 4. CUSTOMER FEATURES

def create_customer_features(transaction_df, reference_date):

    group = transaction_df.groupby("Customer ID")

    features = pd.DataFrame(index=group.groups.keys())

    # Recency
    last_purchase = group["InvoiceDate"].max()

    features["Recency"] = (
        reference_date - last_purchase
    ).dt.days

    # Frequency
    features["Frequency"] = (
        group["Invoice"].nunique()
    )

    # Monetary
    features["Monetary"] = (
        group["Revenue"].sum()
    )

    # Total items
    features["Total_Items"] = (
        group["Quantity"].sum()
    )

    # Average order value
    features["Average_Order_Value"] = (
        features["Monetary"] /
        features["Frequency"].replace(0, 1)
    )

    # Tenure
    first_purchase = group["InvoiceDate"].min()

    features["Tenure"] = (
        reference_date - first_purchase
    ).dt.days

    # Product diversity
    features["Product_Diversity"] = (
        group["StockCode"].nunique()
    )

    # Purchase frequency
    features["Purchase_Freq_Rate"] = (
        features["Frequency"] /
        (features["Tenure"] + 1)
    )

    # Recency relative to tenure
    features["Recency_vs_Tenure"] = (
        features["Recency"] /
        (features["Tenure"] + 1)
    )

    # Average items per order
    features["Average_Items_Per_Order"] = (
        features["Total_Items"] /
        features["Frequency"].replace(0, 1)
    )

    # Revenue per day
    features["Revenue_Per_Day"] = (
        features["Monetary"] /
        (features["Tenure"] + 1)
    )

    return features.fillna(0)


print("Engineering features...")

X = create_customer_features(
    train_df,
    cutoff_date
)

# 5. CREATE TARGETS

print("Creating targets...")

# Churn

target_customers = target_df["Customer ID"].unique()

# 1 = Churned
# 0 = Active

X["Churn"] = 1

X.loc[
    X.index.isin(target_customers),
    "Churn"
] = 0

# CLV

target_revenue = (
    target_df
    .groupby("Customer ID")["Revenue"]
    .sum()
)

X["Target_CLV"] = (
    X.index
    .map(target_revenue)
    .fillna(0)
)

# 6. SELECT FEATURES

features_cols = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Items",
    "Average_Order_Value",
    "Tenure",
    "Product_Diversity",
    "Purchase_Freq_Rate",
    "Recency_vs_Tenure",
    "Average_Items_Per_Order",
    "Revenue_Per_Day"
]

X_model = X[features_cols]

y_churn = X["Churn"]

y_clv = X["Target_CLV"]

# 7. CHURN MODEL

print("Training Churn Model...")

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_churn,
    test_size=0.20,
    random_state=42,
    stratify=y_churn
)

# Handle class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_weight = negative / positive


# Improved XGBoost parameters
churn_model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    reg_alpha=0.1,
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric="auc",
    n_jobs=-1
)

churn_model.fit(
    X_train,
    y_train
)


# Predictions
churn_probs = (
    churn_model
    .predict_proba(X_test)[:, 1]
)


# Default threshold
churn_preds = (
    churn_probs >= 0.50
).astype(int)


# Metrics
auc = roc_auc_score(
    y_test,
    churn_probs
)

accuracy = accuracy_score(
    y_test,
    churn_preds
)

precision = precision_score(
    y_test,
    churn_preds,
    zero_division=0
)

recall = recall_score(
    y_test,
    churn_preds,
    zero_division=0
)

f1 = f1_score(
    y_test,
    churn_preds,
    zero_division=0
)


print("\nChurn Model Results")
print("-------------------")
print(f"AUC:       {auc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# 8. CLV MODEL

print("\nTraining CLV Model...")

X_train_clv, X_test_clv, y_train_clv, y_test_clv = (
    train_test_split(
        X_model,
        y_clv,
        test_size=0.20,
        random_state=42
    )
)


# Log transform CLV
y_train_clv_log = np.log1p(
    y_train_clv
)


clv_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    reg_alpha=0.1,
    random_state=42,
    n_jobs=-1
)


clv_model.fit(
    X_train_clv,
    y_train_clv_log
)


# Predict
clv_preds_log = clv_model.predict(
    X_test_clv
)

clv_preds = np.expm1(
    clv_preds_log
)

# Prevent negative predictions
clv_preds = np.maximum(
    clv_preds,
    0
)


clv_mae = mean_absolute_error(
    y_test_clv,
    clv_preds
)

clv_r2 = r2_score(
    y_test_clv,
    clv_preds
)


print("\nCLV Model Results")
print("-----------------")
print(f"MAE: {clv_mae:.4f}")
print(f"R2:  {clv_r2:.4f}")

# 9. CURRENT CUSTOMER DATA

print(
    "\nGenerating features for current customers..."
)

current_customers = create_customer_features(
    df,
    max_date
)

current_customers = current_customers[
    features_cols
]

# 10. PREDICTIONS

print("Predicting for current customers...")

current_customers[
    "Churn_Probability"
] = churn_model.predict_proba(
    current_customers[features_cols]
)[:, 1]


current_customers[
    "Predicted_CLV"
] = np.expm1(
    clv_model.predict(
        current_customers[features_cols]
    )
)

current_customers[
    "Predicted_CLV"
] = np.maximum(
    current_customers["Predicted_CLV"],
    0
)

# 11. CUSTOMER SEGMENTATION

high_clv = (
    current_customers["Predicted_CLV"]
    .median()
)

def segment_customer(row):

    high_value = (
        row["Predicted_CLV"] > high_clv
    )

    high_risk = (
        row["Churn_Probability"] > 0.5
    )

    if high_value and not high_risk:
        return "Loyal Customer"

    elif high_value and high_risk:
        return "High-Value At Risk"

    elif not high_value and not high_risk:
        return "Regular Customer"

    else:
        return "At Risk Customer"


current_customers[
    "Segment"
] = current_customers.apply(
    segment_customer,
    axis=1
)

# 12. SHAP

print("Computing SHAP values...")

explainer_churn = shap.TreeExplainer(
    churn_model
)

shap_values = explainer_churn.shap_values(
    current_customers[features_cols]
)

shap_df = pd.DataFrame(
    shap_values,
    index=current_customers.index,
    columns=[
        f"{c}_SHAP"
        for c in features_cols
    ]
)

current_customers = pd.concat(
    [
        current_customers,
        shap_df
    ],
    axis=1
)

current_customers[
    "SHAP_Base_Value"
] = explainer_churn.expected_value

# 13. SAVE MODELS

print("Saving artifacts...")

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    churn_model,
    "models/churn_model.joblib"
)

joblib.dump(
    clv_model,
    "models/clv_model.joblib"
)

current_customers.to_csv(
    "models/current_customers.csv"
)

# 14. FINISHED

print("Pipeline completed successfully!")
print("===================================")

print(f"Churn AUC:      {auc:.4f}")
print(f"Churn Accuracy: {accuracy:.4f}")
print(f"Churn F1:       {f1:.4f}")
print(f"CLV MAE:        {clv_mae:.4f}")
print(f"CLV R2:         {clv_r2:.4f}")

print("Models saved successfully.")
