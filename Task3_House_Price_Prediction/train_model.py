"""
train_model.py
Trains a Linear Regression model on the California Housing dataset,
generates required visualizations for the web app, and saves the model.
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Force UTF-8 output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

def save_plot(fig, filename):
    filepath = os.path.join(IMAGE_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] Saved plot: {filename}")

print("==================================================")
print("1. LOADING DATASET")
print("==================================================")
california = fetch_california_housing(as_frame=True)
df = california.frame

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n==================================================")
print("2. EXPLORING DATA DISTRIBUTIONS & PLOTTING")
print("==================================================")

# 1. Price Distribution Histogram
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["MedHouseVal"], bins=50, kde=True, color="#3498db", ax=ax)
ax.set_title("House Price Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Median House Value ($100,000s)")
ax.set_ylabel("Frequency")
save_plot(fig, "price_distribution.png")

# 2. House Age Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["HouseAge"], bins=30, kde=True, color="#9b59b6", ax=ax)
ax.set_title("House Age Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("House Age (Years)")
save_plot(fig, "age_distribution.png")

# 3. House Area / Rooms Distribution
fig, ax = plt.subplots(figsize=(8, 5))
# AveRooms has outliers, so let's clip for visualization
sns.histplot(df["AveRooms"].clip(upper=10), bins=30, kde=True, color="#2ecc71", ax=ax)
ax.set_title("Average Rooms Distribution (Clipped at 10)", fontsize=14, fontweight="bold")
ax.set_xlabel("Average Rooms per Household")
save_plot(fig, "rooms_distribution.png")

# 3. Bedroom Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["AveBedrms"].clip(upper=3), bins=30, kde=True, color="#e74c3c", ax=ax)
ax.set_title("Average Bedrooms Distribution (Clipped at 3)", fontsize=14, fontweight="bold")
ax.set_xlabel("Average Bedrooms per Household")
save_plot(fig, "bedrooms_distribution.png")

# 4. Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
save_plot(fig, "correlation_heatmap.png")


print("\n==================================================")
print("3. DATA PREPROCESSING (Handling missing, Normalization)")
print("==================================================")
# Check missing
missing = df.isnull().sum().sum()
print(f"Missing values: {missing}")

# Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Normalization
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Data normalized using StandardScaler.")

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("[OK] Saved scaler.pkl")

print("\n==================================================")
print("4. TRAIN-TEST SPLIT")
print("==================================================")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples (80%)")
print(f"Testing set : {X_test.shape[0]} samples (20%)")

print("\n==================================================")
print("5. TRAIN REGRESSION MODEL (Linear Regression)")
print("==================================================")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully.")

joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
print("[OK] Saved model.pkl")

print("\n==================================================")
print("6. EVALUATION")
print("==================================================")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Save metrics for app to read
metrics = {"MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f)
print("[OK] Saved metrics.json")

# 5. Actual vs Predicted Graph
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.3, color="#9b59b6")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
ax.set_xlabel("Actual Median House Value ($100k)")
ax.set_ylabel("Predicted Median House Value ($100k)")
ax.set_title("Actual vs Predicted Prices", fontsize=14, fontweight="bold")
save_plot(fig, "actual_vs_predicted.png")

# 6. Feature Importance Graph (Coefficients)
fig, ax = plt.subplots(figsize=(10, 6))
coefs = pd.Series(model.coef_, index=X.columns).sort_values()
coefs.plot(kind="barh", color="#f1c40f", ax=ax)
ax.set_title("Feature Importance (Linear Regression Coefficients)", fontsize=14, fontweight="bold")
ax.set_xlabel("Coefficient Value")
save_plot(fig, "feature_importance.png")

print("\nDone! Pipeline complete.")
