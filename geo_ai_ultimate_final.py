# geo_ai_ultimate_final.py
import os
import io
import time
import json
import requests
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
from streamlit_lottie import st_lottie
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -----------------------------
# Config & Paths
# -----------------------------
st.set_page_config(page_title="Geo-AI Ultimate — Shashi Teja", layout="wide", initial_sidebar_state="expanded")
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Helper: Load Lottie animation
# -----------------------------
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# -----------------------------
# Theme CSS (A..E)
# -----------------------------
THEMES = {
    "A": {"name":"Cyberpunk (A)", "css": """
        <style>
        .stApp { background: linear-gradient(180deg,#020617,#07102a); color:#cfefff; }
        .title { font-size:30px; font-weight:800; color:#cffcff; text-shadow:0 2px 10px rgba(0,255,255,0.08); }
        .card { background: rgba(6,10,20,0.6); padding:14px; border-radius:12px; border: 1px solid rgba(0,255,255,0.06);}
        </style>
        """},
    "B": {"name":"Glass (B)", "css": """
        <style>
        .stApp { background: linear-gradient(180deg,#f8fafc,#ffffff); color:#0f172a; }
        .title { font-size:30px; font-weight:700; color:#0b1220; }
        .card { background: rgba(255,255,255,0.9); padding:12px; border-radius:10px; box-shadow:0 6px 18px rgba(2,6,23,0.06);}
        </style>
        """},
    "C": {"name":"Material (C)", "css": """
        <style>
        .stApp { background: linear-gradient(180deg,#f1f5f9,#e2e8f0); color:#0f172a; }
        .title { font-size:28px; font-weight:700; color:#0b1220; }
        .card { background:#ffffff;padding:12px;border-radius:8px;border-left:6px solid #3b82f6;}
        </style>
        """},
    "D": {"name":"Space (D)", "css": """
        <style>
        .stApp { background: radial-gradient(circle at 10% 20%, #05021b 0, #020617 50%); color:#dfe7ff; }
        .title { font-size:30px; font-weight:800; color:#dff; }
        .card { background: rgba(10,10,20,0.5); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); }
        </style>
        """},
    "E": {"name":"Corporate Green (E)", "css": """
        <style>
        .stApp { background: linear-gradient(180deg,#f7fdf7,#eef7f0); color:#07281a; }
        .title { font-size:30px; font-weight:800; color:#0b4530; }
        .card { background:#ffffff;padding:12px;border-radius:10px;border:1px solid rgba(11,69,48,0.06); }
        </style>
        """}
}

# -----------------------------
# Utilities & plotting helpers
# -----------------------------
def try_load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def normalize(val, vmin, vmax):
    if vmax - vmin == 0:
        return 0.0
    return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))

# Matplotlib images for robust PDF embedding (no kaleido)
def mpl_solar_hours_plot(hours, rad):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.fill_between(hours, rad, color='#fbbf24', alpha=0.4)
    ax.plot(hours, rad, color='#d97706', linewidth=2)
    ax.set_xlabel("Hour")
    ax.set_ylabel("W/m²")
    ax.grid(alpha=0.4, linestyle='--')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def mpl_wind_curve(ws_range, power_curve, cur_speed, cur_power):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(ws_range, power_curve, color='#0ea5e9', lw=2)
    ax.axvline(cur_speed, color='red', linestyle='--')
    ax.scatter([cur_speed],[cur_power], color='red', zorder=5)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.grid(alpha=0.4, linestyle='--')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def mpl_score_bar(score):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.barh([0],[100], color='#e5e7eb')
    color = '#ef4444' if score<40 else '#f59e0b' if score<75 else '#10b981'
    ax.barh([0],[score], color=color)
    ax.set_xlim(0,100)
    ax.set_yticks([])
    ax.set_xlabel("Suitability (%)")
    ax.set_title(f"Final Suitability Score: {score}/100")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def safe_plotly_download(fig, name):
    """Try to export PNG via plotly; fallback to JSON"""
    try:
        img_bytes = fig.to_image(format="png", width=1000, height=600)
        return ("image/png", img_bytes, f"{name}.png")
    except Exception:
        j = fig.to_json().encode()
        return ("application/json", j, f"{name}.json")

def load_plot_png(name):
    """Return bytes of a PNG in PLOTS_DIR if present, else None"""
    path = os.path.join(PLOTS_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

# -----------------------------
# PDF generator embedding plots (reportlab)
# -----------------------------
def make_pdf_report(data, fig_hours_buf=None, fig_wind_buf=None, fig_score_buf=None, existing_plot_bytes=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    # Header
    c.setFillColorRGB(0.05,0.1,0.2)
    c.rect(0, H-80, W, 80, fill=1)
    c.setFillColorRGB(1,1,1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H-52, "Geo-AI Suitability Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, H-70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # --- ADDED NAME IN PDF ---
    c.drawString(40, H-85, "Author: Shashi Teja") 

    # Summary block
    c.setFillColorRGB(0,0,0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, H-110, "Summary")
    c.setFont("Helvetica", 10)
    lines = [
        f"Final score: {data['score']} / 100",
        f"Solar pred: {data['solar_pred']:.2f} W/m²",
        f"Wind pred: {data['wind_pred']:.2f} kW",
        f"Temperature: {data['temp']} °C",
        f"Pressure: {data['pressure']} mb",
        f"Humidity: {data['humidity']} %",
        f"Hub wind speed: {data['wind_speed']} m/s",
        f"Air density: {data['air_density']} kg/m³"
    ]
    y = H-130
    for ln in lines:
        c.drawString(40, y, ln)
        y -= 14

    # embed either provided Matplotlib buffers or existing plot bytes
    try:
        if fig_hours_buf:
            c.drawImage(ImageReader(fig_hours_buf), 40, 330, width=260, height=150)
        elif existing_plot_bytes and "solar" in existing_plot_bytes:
            c.drawImage(ImageReader(io.BytesIO(existing_plot_bytes["solar"])), 40, 330, width=260, height=150)

        if fig_wind_buf:
            c.drawImage(ImageReader(fig_wind_buf), 320, 330, width=260, height=150)
        elif existing_plot_bytes and "wind" in existing_plot_bytes:
            c.drawImage(ImageReader(io.BytesIO(existing_plot_bytes["wind"])), 320, 330, width=260, height=150)

        if fig_score_buf:
            c.drawImage(ImageReader(fig_score_buf), 100, 140, width=400, height=120)
        elif existing_plot_bytes and "hist" in existing_plot_bytes:
            c.drawImage(ImageReader(io.BytesIO(existing_plot_bytes["hist"])), 100, 140, width=400, height=120)
    except Exception as e:
        c.setFont("Helvetica", 10)
        c.drawString(40, 300, "Could not embed charts in PDF: " + str(e))

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# -----------------------------
# Load saved models (if present)
# -----------------------------
model_solar = try_load_model(os.path.join(DATA_DIR, "model_solar.pkl"))
model_wind  = try_load_model(os.path.join(DATA_DIR, "model_wind.pkl"))

# -----------------------------
# Session state defaults
# -----------------------------
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "theme" not in st.session_state:
    st.session_state.theme = "A"
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = True

# -----------------------------
# Sidebar: controls & inputs
# -----------------------------
st.sidebar.markdown("## Geo-AI Ultimate · Controls")
# --- ADDED NAME IN SIDEBAR ---
st.sidebar.markdown("**Developed by Shashi Teja**")

theme_choice = st.sidebar.selectbox("Theme (A..E)", options=["A","B","C","D","E"], index=0)
st.session_state.theme = theme_choice
st.markdown(THEMES[theme_choice]["css"], unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_sample = st.sidebar.button("Load sample points (demo)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Modeling / Options")
st.session_state.ai_mode = st.sidebar.checkbox("AI Explanation Mode (text insights)", value=True)
reload_models = st.sidebar.button("Reload saved models (from data/)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Region / District")
district = st.sidebar.selectbox("Pick district (quick centroids)", options=["Custom","Bengaluru","Mysuru","Raichur"])
# centroid lookup
district_centroids = {
    "Bengaluru": (12.9716, 77.5946),
    "Mysuru":    (12.2958, 76.6394),
    "Raichur":   (16.2048, 77.3455)
}

st.sidebar.markdown("---")
st.sidebar.markdown("### Map & Visualization")
globe_enabled = st.sidebar.checkbox("Enable 3D Globe", True)
heatmap_enabled = st.sidebar.checkbox("Enable Heatmap overlay", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Export")
# placeholder shown later

# -----------------------------
# Main UI header
# -----------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown(f"<h1 class='title'>⚡ Geo-AI Ultimate — Renewable Site Suitability</h1>", unsafe_allow_html=True)
    st.write("Interactive globe, heatmap, model predictions, downloadable charts, and PDF reporting.")
    # --- ADDED NAME IN HEADER ---
    st.markdown("### Created by Shashi Teja")
with col2:
    lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_cizyir2r.json")
    if lottie:
        st_lottie(lottie, height=110, key="lottie")

st.markdown("---")

# -----------------------------
# Handle uploaded csv or sample
# -----------------------------
if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df_up
        st.success("CSV loaded — preview below.")
        st.dataframe(df_up.head(5))
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if use_sample:
    sample = pd.DataFrame({
        "lat":[12.9716,17.3850,15.9129,13.0827,14.4426],
        "lon":[77.5946,78.4867,79.7400,80.2707,78.3900],
        "temp":[28,30,27,29,26],
        "pressure":[1010,1008,1006,1012,1011],
        "humidity":[45,50,55,60,42],
        "wind_speed":[6,7,5,8,4],
        "air_density":[1.225,1.22,1.23,1.21,1.225]
    })
    st.session_state.uploaded_df = sample
    st.success("Sample points loaded (demo).")
    st.dataframe(sample)

# -----------------------------
# Single-site input sliders
# -----------------------------
st.markdown("### Manual Site Input (single-site quick test)")
colA, colB, colC = st.columns(3)
with colA:
    temp = st.slider("Temperature (°C)", -10, 50, 25)
    pressure = st.slider("Pressure (mb)", 800, 1100, 1013)
with colB:
    humidity = st.slider("Humidity (%)", 0, 100, 45)
    wind_speed = st.slider("Hub wind speed (m/s)", 0, 40, 8)
with colC:
    air_density = st.slider("Air density (kg/m³)", 1.0, 1.5, 1.225)
    if district != "Custom":
        lat, lon = district_centroids[district]
    else:
        lon = st.number_input("Longitude (e.g. 76.0)", value=75.0, format="%.6f")
        lat = st.number_input("Latitude (e.g. 15.0)", value=14.0, format="%.6f")

solar_wt = st.slider("Solar weight (combined score)", 0.0, 1.0, 0.6)

run_single = st.button("Run single-site prediction")
run_batch  = st.button("Run batch predictions (CSV)")

# -----------------------------
# Prediction functions
# -----------------------------
def predict_single_site(temp, pressure, humidity, wind_speed, air_density):
    if model_solar:
        try:
            X = np.array([[temp, pressure, humidity, wind_speed]])
            sp = float(model_solar.predict(X)[0])
        except Exception:
            sp = 800 * (1 - 0.3*humidity/100.0) * (1 - abs(temp-25)/100.0)
    else:
        sp = 800 * (1 - 0.3*humidity/100.0) * (1 - abs(temp-25)/100.0)

    if model_wind:
        try:
            Xw = np.array([[wind_speed, 0.0, air_density]])
            wp = float(model_wind.predict(Xw)[0])
        except Exception:
            wp = 0.25 * air_density * (wind_speed**3)
    else:
        wp = 0.25 * air_density * (wind_speed**3)

    s_norm = normalize(sp, 0, 1200)
    w_norm = normalize(wp, 0, 4000)
    combined = np.clip(solar_wt * s_norm + (1 - solar_wt) * w_norm, 0.0, 1.0)
    score = round(combined * 100, 2)
    return sp, wp, score

def batch_predict(df):
    results=[]
    for _, row in df.iterrows():
        sp, wp, sc = predict_single_site(
            float(row.get("temp", temp)),
            float(row.get("pressure", pressure)),
            float(row.get("humidity", humidity)),
            float(row.get("wind_speed", wind_speed)),
            float(row.get("air_density", air_density))
        )
        results.append({
            "lon": row.get("lon", lon),
            "lat": row.get("lat", lat),
            "temp": row.get("temp", temp),
            "pressure": row.get("pressure", pressure),
            "humidity": row.get("humidity", humidity),
            "wind_speed": row.get("wind_speed", wind_speed),
            "air_density": row.get("air_density", air_density),
            "solar_pred": sp, "wind_pred": wp, "score": sc
        })
    return pd.DataFrame(results)

# -----------------------------
# Run if requested
# -----------------------------
single_result=None
if run_single:
    sp, wp, sc = predict_single_site(temp, pressure, humidity, wind_speed, air_density)
    single_result = {"temp":temp,"pressure":pressure,"humidity":humidity,
                     "wind_speed":wind_speed,"air_density":air_density,
                     "lon":lon,"lat":lat,"solar_pred":sp,"wind_pred":wp,"score":sc}
    st.success(f"Single-site score: {sc} / 100 — Solar {sp:.2f} W/m² | Wind {wp:.2f} kW")

batch_df=None
if run_batch:
    if st.session_state.uploaded_df is None:
        st.error("Upload a CSV first (contains site points).")
    else:
        with st.spinner("Running batch predictions..."):
            batch_df = batch_predict(st.session_state.uploaded_df)
            st.session_state.predictions = batch_df
            st.success("Batch predictions completed — preview below.")
            st.dataframe(batch_df.head(10))

# -----------------------------
# Dashboard panels: summary + globe/map
# -----------------------------
st.markdown("---")
left_col, right_col = st.columns([1.2,2])

with left_col:
    st.subheader("Site Summary")
    if single_result:
        st.metric("Final score", f"{single_result['score']} / 100")
        st.write(f"Solar: {single_result['solar_pred']:.2f} W/m²")
        st.write(f"Wind: {single_result['wind_pred']:.2f} kW")
    elif batch_df is not None:
        st.metric("Batch sites", f"{len(batch_df)}")
        csv_bytes = batch_df.to_csv(index=False).encode()
        st.download_button("Download batch CSV", csv_bytes, "batch_predictions.csv", "text/csv")
    else:
        st.info("Run single-site or batch analysis to view summary.")

    st.subheader("AI Explanation")
    if st.session_state.ai_mode:
        if single_result:
            expl=[]
            sc=single_result['score']
            expl.append(f"Score {sc}: {'Excellent' if sc>=75 else 'Good' if sc>=50 else 'Low'}")
            expl.append(f"Humidity {single_result['humidity']}% → {'reduces' if single_result['humidity']>60 else 'ok'} solar output")
            expl.append(f"Wind {single_result['wind_speed']} m/s → {'good' if single_result['wind_speed']>8 else 'weak' } for small turbines")
            st.write("\n".join(expl))
        else:
            st.write("AI Explanation mode: run prediction to get tailored insights.")

with right_col:
    st.subheader("Globe & Map Visuals")
    if globe_enabled:
        st.markdown("**Interactive Globe (orthographic)**")
        if single_result:
            gl_lon = single_result['lon']; gl_lat = single_result['lat']
        elif batch_df is not None:
            gl_lon = float(batch_df['lon'].mean()); gl_lat = float(batch_df['lat'].mean())
        elif st.session_state.uploaded_df is not None and 'lon' in st.session_state.uploaded_df.columns:
            gl_lon = float(st.session_state.uploaded_df['lon'].mean()); gl_lat = float(st.session_state.uploaded_df['lat'].mean())
        else:
            gl_lon, gl_lat = lon, lat

        rot = st.slider("Rotate globe (lon rotation)", -180, 180, 0)
        globe_fig = go.Figure()
        if single_result:
            color = single_result['score'] / 100.0
            globe_fig.add_trace(go.Scattergeo(
                lon=[gl_lon], lat=[gl_lat],
                marker=dict(size=12, color=px.colors.sample_colorscale("Viridis",[color])[0]),
                text=[f"Score: {single_result['score']}"],
                hoverinfo='text'
            ))
        elif batch_df is not None:
            globe_fig.add_trace(go.Scattergeo(
                lon=batch_df['lon'], lat=batch_df['lat'],
                marker=dict(size=6, color=batch_df['score'], colorscale='Turbo', cmin=0, cmax=100, colorbar=dict(title="Score")),
                hovertemplate="Score: %{marker.color:.2f}<extra></extra>"
            ))
        else:
            globe_fig.add_trace(go.Scattergeo(lon=[gl_lon], lat=[gl_lat], marker=dict(size=8, color='cyan')))

        globe_fig.update_geos(projection_type="orthographic", projection_rotation=dict(lon=rot, lat=0))
        globe_fig.update_layout(height=450, margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(globe_fig, use_container_width=True)

    if heatmap_enabled:
        st.markdown("**Map heatmap (density)**")
        df_map = None
        if batch_df is not None:
            df_map = batch_df.copy()
        elif st.session_state.uploaded_df is not None and 'lat' in st.session_state.uploaded_df.columns:
            df_map = st.session_state.uploaded_df.copy()
            if 'score' not in df_map.columns:
                df_map = batch_predict(df_map)
        elif single_result:
            df_map = pd.DataFrame([single_result])

        if df_map is not None and 'lon' in df_map.columns and 'lat' in df_map.columns:
            try:
                fig_map = px.density_mapbox(df_map, lat='lat', lon='lon', z='score', radius=30,
                                            center=dict(lat=df_map['lat'].mean(), lon=df_map['lon'].mean()),
                                            zoom=6, mapbox_style="open-street-map", title="Suitability Heatmap")
                fig_map.update_layout(height=350, margin=dict(t=10,b=10,l=10,r=10))
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error("Could not create density map: " + str(e))
        else:
            st.info("Map requires latitude/longitude columns in the dataset.")

# -----------------------------
# Charts & downloads (use either built objects or existing PNG files)
# -----------------------------
st.markdown("---")
st.subheader("Charts & Downloads")

# Build interactive plotly charts
hours = np.arange(0,24,1)
if single_result:
    solar_val = single_result['solar_pred']
elif batch_df is not None:
    solar_val = batch_df['solar_pred'].mean()
else:
    solar_val = 700.0

rad = np.clip(np.sin((hours-6)/12 * np.pi) * solar_val, 0, None)
fig_hours = px.area(x=hours, y=rad, labels={"x":"Hour","y":"W/m²"}, title="Predicted Daily Irradiance")
fig_hours.update_traces(line_color='#fbbf24', fillcolor='rgba(251,191,36,0.4)')
st.plotly_chart(fig_hours, use_container_width=True)
mime, data_bytes, fname = safe_plotly_download(fig_hours, "daily_irradiance")
st.download_button(f"Download {fname}", data_bytes, file_name=fname, mime=mime)

# Wind curve
ws = np.linspace(0,30,50)
pw = np.clip(ws**3 * (air_density if 'air_density' in locals() else 1.225) * 0.01, 0, None)
fig_wind_curve = px.line(x=ws, y=pw, labels={"x":"Wind speed (m/s)","y":"Power (kW)"}, title="Turbine Power Curve (approx.)")
st.plotly_chart(fig_wind_curve, use_container_width=True)
mime, data_bytes, fname = safe_plotly_download(fig_wind_curve, "wind_power_curve")
st.download_button(f"Download {fname}", data_bytes, file_name=fname, mime=mime)

# Combined histogram
if single_result:
    center = single_result['score']/100.0
elif batch_df is not None:
    center = batch_df['score'].mean()/100.0
else:
    center = 0.5
samples = np.clip(np.random.normal(loc=center, scale=0.15, size=800), 0,1)
fig_hist = px.histogram(samples, nbins=20, title="Combined Suitability Distribution")
st.plotly_chart(fig_hist, use_container_width=True)
mime, data_bytes, fname = safe_plotly_download(fig_hist, "combined_hist")
st.download_button(f"Download {fname}", data_bytes, file_name=fname, mime=mime)

# Also show your existing training plots (if present in data/plots)
st.markdown("### Training / Model Plots (from data/plots)")
plot_files = {
    "combined":"combined_score_hist.png",
    "solar_feat":"solar_feature_importance.png",
    "solar_pred":"solar_pred_vs_actual.png",
    "wind_feat":"wind_feature_importance.png",
    "wind_pred":"wind_pred_vs_actual.png"
}
cols = st.columns(5)
for i,(k,fn) in enumerate(plot_files.items()):
    b = load_plot_png(fn)
    if b:
        with cols[i]:
            st.image(b, caption=fn, use_container_width=True)

            st.download_button(f"Download {fn}", b, file_name=fn, mime="image/png")
    else:
        with cols[i]:
            st.write(fn)
            st.info("Not found in data/plots/")

# -----------------------------
# PDF Export (Matplotlib images embedded or existing PNGs)
# -----------------------------
st.markdown("---")
st.subheader("Generate PDF Report (images embedded)")

if single_result:
    pdf_label = "Generate PDF for this site"
else:
    pdf_label = "Generate PDF (batch)"

if st.button(pdf_label):
    with st.spinner("Preparing PDF..."):
        # Prepare Matplotlib buffers
        fig_hours_buf = mpl_solar_hours_plot(hours, rad)
        fig_wind_buf  = mpl_wind_curve(ws, pw, wind_speed if 'wind_speed' in locals() else 8,
                                       (single_result['wind_pred'] if single_result else (batch_df['wind_pred'].mean() if batch_df is not None else 0)))
        score_val = single_result['score'] if single_result else (batch_df['score'].mean() if batch_df is not None else 0)
        fig_score_buf = mpl_score_bar(score_val)

        # also attempt to load existing png bytes to embed instead of Matplotlib ones
        existing_bytes = {}
        for tag,fn in plot_files.items():
            b = load_plot_png(fn)
            if b:
                # map keys we used earlier
                if "combined" in fn:
                    existing_bytes["hist"] = b
                if "solar_feature" in fn:
                    existing_bytes["solar"] = b
                if "solar_pred" in fn:
                    existing_bytes["solar_pred"] = b
                if "wind_feature" in fn:
                    existing_bytes["wind"] = b
                if "wind_pred" in fn:
                    existing_bytes["wind_pred"] = b

        # data dictionary for PDF
        data_for_pdf = {
            "temp": single_result['temp'] if single_result else (batch_df['temp'].mean() if batch_df is not None else temp),
            "pressure": single_result['pressure'] if single_result else (batch_df['pressure'].mean() if batch_df is not None else pressure),
            "humidity": single_result['humidity'] if single_result else (batch_df['humidity'].mean() if batch_df is not None else humidity),
            "wind_speed": single_result['wind_speed'] if single_result else (batch_df['wind_speed'].mean() if batch_df is not None else wind_speed),
            "air_density": single_result['air_density'] if single_result else (batch_df['air_density'].mean() if batch_df is not None else air_density),
            "solar_pred": single_result['solar_pred'] if single_result else (batch_df['solar_pred'].mean() if batch_df is not None else solar_val),
            "wind_pred": single_result['wind_pred'] if single_result else (batch_df['wind_pred'].mean() if batch_df is not None else (pw.mean())),
            "score": score_val
        }

        pdf_buf = make_pdf_report(data_for_pdf, fig_hours_buf, fig_wind_buf, fig_score_buf, existing_plot_bytes=existing_bytes if existing_bytes else None)
        st.success("PDF ready!")
        st.download_button("⬇️ Download PDF Report", pdf_buf.getvalue(), file_name="GeoAI_Suitability_Report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Geo-AI Ultimate — Advanced interactive exploration with robust PDF reporting. Ask me to add Mapbox tiles or live raster overlays next.")