# train_models.py
import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Path to your dataset (change if needed)
DATA_CSV = os.path.join(ROOT, "karnataka_real_dataset_2020_2024.csv")

# Output model files (used by your Streamlit app)
SOLAR_MODEL_PATH = os.path.join(DATA_DIR, "model_solar.pkl")
WIND_MODEL_PATH  = os.path.join(DATA_DIR, "model_wind.pkl")
COMBINED_SCORES_CSV = os.path.join(DATA_DIR, "combined_scores.csv")

# ---------------- Utility functions ----------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

def ensure_cols(df):
    # canonical column names we expect
    required = [
        "datetime", "district", "latitude", "longitude",
        "temperature_C", "pressure_mb", "humidity_pct",
        "wind_speed_ms", "solar_radiation_Wm2", "wind_power_kw"
    ]
    # try to map common alternative names
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}
    if "temp" in col_lower and "temperature_C" not in df.columns:
        rename_map[col_lower["temp"]] = "temperature_C"
    if "pressure" in col_lower and "pressure_mb" not in df.columns:
        rename_map[col_lower["pressure"]] = "pressure_mb"
    if "humidity" in col_lower and "humidity_pct" not in df.columns:
        rename_map[col_lower["humidity"]] = "humidity_pct"
    if "wind_speed" in col_lower and "wind_speed_ms" not in df.columns:
        rename_map[col_lower["wind_speed"]] = "wind_speed_ms"
    if "solar_radiation" in col_lower and "solar_radiation_Wm2" not in df.columns:
        rename_map[col_lower["solar_radiation"]] = "solar_radiation_Wm2"
    if "wind_power" in col_lower and "wind_power_kw" not in df.columns:
        rename_map[col_lower["wind_power"]] = "wind_power_kw"
    if rename_map:
        df = df.rename(columns=rename_map)
    # Add missing columns if essential (fill with reasonable defaults)
    if "wind_direction" not in df.columns:
        df["wind_direction"] = 180.0  # default (calm/neutral direction)
    # If air_density not present, estimate roughly using standard 1.225
    if "air_density" not in df.columns:
        df["air_density"] = 1.225
    return df

def save_matplotlib_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

# ---------------- Load dataset ----------------
print("Loading dataset:", DATA_CSV)
df = safe_read_csv(DATA_CSV)
df = ensure_cols(df)

# Keep only necessary columns for modelling
model_df = df.copy()

# Basic cleaning: drop rows with NaNs in target columns
model_df = model_df.dropna(subset=["solar_radiation_Wm2", "wind_power_kw"])

print("Total rows after cleaning:", len(model_df))

# ---------------- Feature / Target ----------------
X_solar = model_df[["temperature_C", "pressure_mb", "humidity_pct", "wind_speed_ms"]].astype(float)
y_solar = model_df["solar_radiation_Wm2"].astype(float)

X_wind = model_df[["wind_speed_ms", "wind_direction", "air_density"]].astype(float)
y_wind = model_df["wind_power_kw"].astype(float)

# ---------------- Train/test split ----------------
RSEED = 42
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_solar, y_solar, test_size=0.2, random_state=RSEED)
Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_wind, y_wind, test_size=0.2, random_state=RSEED)

print("Solar train/test:", Xs_train.shape, Xs_test.shape)
print("Wind  train/test:", Xw_train.shape, Xw_test.shape)

# ---------------- Pipelines & models ----------------
solar_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=RSEED))
])

wind_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=RSEED))
])

print("Training solar model...")
solar_pipeline.fit(Xs_train, ys_train)
print("Training wind model...")
wind_pipeline.fit(Xw_train, yw_train)

# ---------------- Evaluate ----------------
def metrics(true, pred):
    # compute RMSE in a version-robust way (works with all sklearn versions)
    mse = mean_squared_error(true, pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(true, pred))
    return rmse, r2

ys_pred = solar_pipeline.predict(Xs_test)
yw_pred = wind_pipeline.predict(Xw_test)

rmse_s, r2_s = metrics(ys_test, ys_pred)
rmse_w, r2_w = metrics(yw_test, yw_pred)

print("\n=== SOLAR MODEL METRICS ===")
print(f"RMSE: {rmse_s:.4f}")
print(f"R2   : {r2_s:.4f}")

print("\n=== WIND MODEL METRICS ===")
print(f"RMSE: {rmse_w:.4f}")
print(f"R2   : {r2_w:.4f}")

# ---------------- Save models ----------------
joblib.dump(solar_pipeline, SOLAR_MODEL_PATH)
joblib.dump(wind_pipeline, WIND_MODEL_PATH)
print("\nSaved models to:", SOLAR_MODEL_PATH, "and", WIND_MODEL_PATH)

# ---------------- Create and save plots for Streamlit app ----------------
# 1) Solar feature importance
try:
    rf_s = solar_pipeline.named_steps["rf"]
    importances_s = rf_s.feature_importances_
    features_s = Xs_train.columns.tolist()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(features_s, importances_s)
    ax.set_xlabel("Importance")
    ax.set_title("Solar feature importances")
    figpath = os.path.join(PLOTS_DIR, "solar_feature_importance.png")
    save_matplotlib_fig(fig, figpath)
    print("Saved:", figpath)
except Exception as e:
    print("Could not create solar feature importance:", e)

# 2) Solar predicted vs actual (test)
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(ys_test, ys_pred, s=10, alpha=0.6)
mn = min(ys_test.min(), ys_pred.min())
mx = max(ys_test.max(), ys_pred.max())
ax.plot([mn,mx],[mn,mx], color="red", linewidth=1)
ax.set_xlabel("Actual Solar (W/m2)")
ax.set_ylabel("Predicted Solar (W/m2)")
ax.set_title("Solar: Predicted vs Actual")
figpath = os.path.join(PLOTS_DIR, "solar_pred_vs_actual.png")
save_matplotlib_fig(fig, figpath)
print("Saved:", figpath)

# 3) Wind feature importance
try:
    rf_w = wind_pipeline.named_steps["rf"]
    importances_w = rf_w.feature_importances_
    features_w = Xw_train.columns.tolist()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(features_w, importances_w)
    ax.set_xlabel("Importance")
    ax.set_title("Wind feature importances")
    figpath = os.path.join(PLOTS_DIR, "wind_feature_importance.png")
    save_matplotlib_fig(fig, figpath)
    print("Saved:", figpath)
except Exception as e:
    print("Could not create wind feature importance:", e)

# 4) Wind predicted vs actual
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(yw_test, yw_pred, s=10, alpha=0.6)
mn = min(yw_test.min(), yw_pred.min())
mx = max(yw_test.max(), yw_pred.max())
ax.plot([mn,mx],[mn,mx], color="red", linewidth=1)
ax.set_xlabel("Actual Wind Power (kW)")
ax.set_ylabel("Predicted Wind Power (kW)")
ax.set_title("Wind: Predicted vs Actual")
figpath = os.path.join(PLOTS_DIR, "wind_pred_vs_actual.png")
save_matplotlib_fig(fig, figpath)
print("Saved:", figpath)

# 5) Combined scores histogram (use test set)
def norm_solar_val(val): return np.clip((val - 0.0) / (1200.0 - 0.0 + 1e-9), 0.0, 1.0)
def norm_wind_val(val):  return np.clip((val - 0.0) / (4000.0 - 0.0 + 1e-9), 0.0, 1.0)

solar_norm_test = norm_solar_val(ys_pred)
wind_norm_test  = norm_wind_val(yw_pred)
combined_scores = 0.6 * solar_norm_test + 0.4 * wind_norm_test
combined_scores_percent = combined_scores * 100.0

fig, ax = plt.subplots(figsize=(6,3))
ax.hist(combined_scores_percent, bins=30, color="#3b82f6", alpha=0.8)
ax.set_xlabel("Combined Suitability Score (%)")
ax.set_ylabel("Count")
ax.set_title("Combined Suitability Distribution (test predictions)")
figpath = os.path.join(PLOTS_DIR, "combined_score_hist.png")
save_matplotlib_fig(fig, figpath)
print("Saved:", figpath)

# ---------------- Save combined scores CSV (test rows)
n = min(len(ys_test), len(yw_test))
results_df = pd.DataFrame({
    "index": np.arange(n),
    "solar_actual": ys_test.iloc[:n].values,
    "solar_pred": ys_pred[:n],
    "wind_actual": yw_test.iloc[:n].values,
    "wind_pred": yw_pred[:n],
    "solar_norm": solar_norm_test[:n],
    "wind_norm": wind_norm_test[:n],
    "combined_score": combined_scores_percent[:n]
})

results_df.to_csv(COMBINED_SCORES_CSV, index=False)
print("Saved combined scores CSV:", COMBINED_SCORES_CSV)

print("\nAll done. Plots & models saved. You can now run your Streamlit app (geo_ai_ultimate.py).")
