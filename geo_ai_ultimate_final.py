import os
import io
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

# ===== INSANE UI STYLING =====
st.set_page_config(
    page_title="🚀 GEO-AI ULTIMATE",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# AGGRESSIVE DARK CYBERPUNK THEME
st.markdown("""
<style>

    /* GLOBAL APP BACKGROUND */
    .stApp {
        background: #f4f6f9;
        font-family: 'Inter', sans-serif;
        color: #2b2b2b;
    }

    /* HEADINGS */
    h1, h2, h3 {
        font-weight: 700;
        color: #333;
        letter-spacing: 0.5px;
    }

    h1 { font-size: 2.2rem; }
    h2 { font-size: 1.7rem; }
    h3 { font-size: 1.3rem; }

    /* METRIC / INFO CARDS */
    .info-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e3e6eb;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
        margin: 15px 0;
    }

    /* PASTEL COLOR CARDS */
    .pastel-blue {
        background: #e7f0ff;
        border-left: 6px solid #5b8def;
    }

    .pastel-green {
        background: #e9f9ee;
        border-left: 6px solid #5dcf75;
    }

    .pastel-yellow {
        background: #fff8df;
        border-left: 6px solid #f4c542;
    }

    .pastel-red {
        background: #ffecec;
        border-left: 6px solid #e05a5a;
    }

    /* BUTTONS */
    .stButton > button {
        background: #5b8def;
        color: white;
        border: none;
        padding: 10px 22px;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.2s ease;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
    }

    .stButton > button:hover {
        background: #4b7ad3;
        transform: translateY(-2px);
        box-shadow: 0 5px 14px rgba(0, 0, 0, 0.12);
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0e3e8;
    }

    /* SIDEBAR HEADERS */
    [data-testid="stSidebar"] h2 {
        color: #444;
        font-weight: 700;
    }

    /* INPUT FIELDS */
    input, textarea, select {
        background: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #d0d5db !important;
        padding: 8px 10px !important;
        color: #333 !important;
        box-shadow: none !important;
    }

    input:focus, textarea:focus, select:focus {
        border-color: #5b8def !important;
        box-shadow: 0 0 0 2px rgba(91, 141, 239, 0.2) !important;
    }

    /* DATA TABLE */
    .dataframe {
        border: 1px solid #d4d7dd !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

</style>
""", unsafe_allow_html=True)


# ===== PATHS =====
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ===== HELPER FUNCTIONS =====
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

def try_load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def normalize(val, vmin, vmax):
    if vmax - vmin == 0:
        return 0.0
    return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))

def mpl_solar_hours_plot(hours, rad):
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0a0e27')
    ax.set_facecolor('#1a1a3e')
    ax.fill_between(hours, rad, color='#ff6600', alpha=0.6)
    ax.plot(hours, rad, color='#ffff00', linewidth=3)
    ax.set_xlabel("Hour", color='#00ff88', fontweight='bold')
    ax.set_ylabel("W/m²", color='#00ff88', fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--', color='#00ff88')
    ax.tick_params(colors='#00ff88')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='#0a0e27')
    plt.close(fig)
    buf.seek(0)
    return buf

def mpl_wind_curve(ws_range, power_curve, cur_speed, cur_power):
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='#0a0e27')
    ax.set_facecolor('#1a1a3e')
    ax.plot(ws_range, power_curve, color='#00ffff', lw=3)
    ax.axvline(cur_speed, color='#ff00ff', linestyle='--', linewidth=2)
    ax.scatter([cur_speed], [cur_power], color='#ff6600', s=200, zorder=5, edgecolors='#ffff00', linewidth=2)
    ax.set_xlabel("Wind speed (m/s)", color='#00ff88', fontweight='bold')
    ax.set_ylabel("Power (kW)", color='#00ff88', fontweight='bold')
    ax.grid(alpha=0.2, linestyle='--', color='#00ff88')
    ax.tick_params(colors='#00ff88')
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='#0a0e27')
    plt.close(fig)
    buf.seek(0)
    return buf

def mpl_score_bar(score):
    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#0a0e27')
    ax.set_facecolor('#1a1a3e')
    
    # Background bar
    ax.barh([0], [100], color='rgba(0, 255, 136, 0.1)', height=0.5, edgecolor='#00ff88', linewidth=2)
    
    # Score bar with gradient color
    if score < 40:
        color = '#ff0000'
    elif score < 75:
        color = '#ff6600'
    else:
        color = '#00ff88'
    
    ax.barh([0], [score], color=color, height=0.5, edgecolor='#ffff00', linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Suitability (%)", color='#00ff88', fontweight='bold', fontsize=12)
    ax.set_title(f"FINAL SUITABILITY: {score}/100", color='#00ff88', fontweight='bold', fontsize=14)
    ax.tick_params(colors='#00ff88')
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor='#0a0e27')
    plt.close(fig)
    buf.seek(0)
    return buf

def safe_plotly_download(fig, name):
    try:
        img_bytes = fig.to_image(format="png", width=1000, height=600)
        return ("image/png", img_bytes, f"{name}.png")
    except Exception:
        j = fig.to_json().encode()
        return ("application/json", j, f"{name}.json")

def load_plot_png(name):
    path = os.path.join(PLOTS_DIR, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    return None

def make_pdf_report(data, fig_hours_buf=None, fig_wind_buf=None,
                    fig_score_buf=None, existing_plot_bytes=None):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    W, H = letter

    c.setFillColorRGB(0.05, 0.08, 0.2)
    c.rect(0, H-90, W, 90, fill=1)
    c.setFillColorRGB(0, 1, 0.5)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, H-50, "🚀 GEO-AI ULTIMATE REPORT")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 1, 1)
    c.drawString(40, H-70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFillColorRGB(1, 0, 1)
    c.drawString(40, H-85, "Author: Shashi Teja")

    c.setFillColorRGB(0, 1, 0.5)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, H-115, "ANALYSIS SUMMARY")
    c.setFont("Helvetica", 9)
    lines = [
        f"⚡ Final Score: {data['score']} / 100",
        f"☀️  Solar Prediction: {data['solar_pred']:.2f} W/m²",
        f"💨 Wind Prediction: {data['wind_pred']:.2f} kW",
        f"🌡️  Temperature: {data['temp']} °C",
        f"📊 Pressure: {data['pressure']} mb",
        f"💧 Humidity: {data['humidity']} %",
        f"🌪️  Hub Wind Speed: {data['wind_speed']} m/s",
        f"🔬 Air Density: {data['air_density']} kg/m³"
    ]
    y = H-135
    for ln in lines:
        c.drawString(40, y, ln)
        y -= 14

    try:
        if fig_hours_buf:
            c.drawImage(ImageReader(fig_hours_buf), 40, 330, width=260, height=150)
        if fig_wind_buf:
            c.drawImage(ImageReader(fig_wind_buf), 320, 330, width=260, height=150)
        if fig_score_buf:
            c.drawImage(ImageReader(fig_score_buf), 100, 140, width=400, height=120)
    except Exception as e:
        c.setFont("Helvetica", 9)
        c.drawString(40, 300, f"Note: Charts embedded: {str(e)[:40]}")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf

# ===== LOAD MODELS =====
model_solar = try_load_model(os.path.join(DATA_DIR, "model_solar.pkl"))
model_wind = try_load_model(os.path.join(DATA_DIR, "model_wind.pkl"))

# ===== SESSION STATE =====
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = True

# ===== INSANE SIDEBAR =====
st.sidebar.markdown("<h2 style='color: #ff00ff; text-shadow: 0 0 20px #ff00ff;'>⚙️ GEO-AI CONTROL CENTER</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #00ff88; font-size: 0.9em;'>🔧 Developed by Shashi Teja</p>", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border-color: #ff00ff;'>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("📤 Upload CSV Dataset", type=["csv"])
use_sample = st.sidebar.button("🎯 Load Demo Sample", use_container_width=True)

st.sidebar.markdown("<hr style='border-color: #00ffff;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #00ffff;'>🎛️ Settings</h3>", unsafe_allow_html=True)

st.session_state.ai_mode = st.sidebar.checkbox("🤖 AI Insights Mode", value=True)
globe_enabled = st.sidebar.checkbox("🌍 3D Globe", True)
heatmap_enabled = st.sidebar.checkbox("🔥 Heatmap Overlay", True)

st.sidebar.markdown("<hr style='border-color: #ff6600;'>", unsafe_allow_html=True)

district = st.sidebar.selectbox(
    "📍 Select District",
    options=["Custom", "Bengaluru", "Mysuru", "Raichur"],
    format_func=lambda x: f"📍 {x}"
)

district_centroids = {
    "Bengaluru": (12.9716, 77.5946),
    "Mysuru": (12.2958, 76.6394),
    "Raichur": (16.2048, 77.3455)
}

# ===== HEADER SECTION =====
st.markdown("""
<div style='text-align: center; padding: 30px 0;'>
    <h1 style='font-size: 3.5em; background: linear-gradient(135deg, #00ff88, #00ffff, #ff00ff); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: none;'>
    🚀 GEO-AI ULTIMATE
    </h1>
    <p style='color: #00ffff; font-size: 1.3em; text-shadow: 0 0 20px #00ffff;'>
    RENEWABLE ENERGY SITE SUITABILITY ANALYSIS
    </p>
    <p style='color: #ff00ff; font-size: 0.95em;'>≈≈≈ Created by Shashi Teja ≈≈≈</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 2px solid #00ff88; margin: 30px 0;'>", unsafe_allow_html=True)

# ===== INPUT SECTION =====
st.markdown("<div class='glow-card'>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #00ffff;'>⚡ SITE INPUT PARAMETERS</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p style='color: #ff00ff; font-weight: bold;'>🌡️ Temperature (°C)</p>", unsafe_allow_html=True)
    temp = st.slider("Temp", -10, 50, 25, label_visibility="collapsed")
    
    st.markdown("<p style='color: #ff00ff; font-weight: bold;'>📊 Pressure (mb)</p>", unsafe_allow_html=True)
    pressure = st.slider("Pressure", 800, 1100, 1013, label_visibility="collapsed")

with col2:
    st.markdown("<p style='color: #00ffff; font-weight: bold;'>💧 Humidity (%)</p>", unsafe_allow_html=True)
    humidity = st.slider("Humidity", 0, 100, 45, label_visibility="collapsed")
    
    st.markdown("<p style='color: #00ffff; font-weight: bold;'>🌪️ Wind Speed (m/s)</p>", unsafe_allow_html=True)
    wind_speed = st.slider("Wind", 0, 40, 8, label_visibility="collapsed")

with col3:
    st.markdown("<p style='color: #ff6600; font-weight: bold;'>🔬 Air Density (kg/m³)</p>", unsafe_allow_html=True)
    air_density = st.slider("Density", 1.0, 1.5, 1.225, label_visibility="collapsed")
    
    if district != "Custom":
        lat, lon = district_centroids[district]
        st.markdown(f"<p style='color: #00ff88;'>📍 {district}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #ff00ff; font-weight: bold;'>🧭 Coordinates</p>", unsafe_allow_html=True)
        lon = st.number_input("Longitude", value=75.0, format="%.6f", label_visibility="collapsed")
        lat = st.number_input("Latitude", value=14.0, format="%.6f", label_visibility="collapsed")

st.markdown("<p style='color: #00ffff; font-weight: bold; margin-top: 15px;'>⚖️ Solar Weight (0.0-1.0)</p>", unsafe_allow_html=True)
solar_wt = st.slider("Weight", 0.0, 1.0, 0.6, label_visibility="collapsed")

col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    run_single = st.button("🔥 RUN SINGLE-SITE", use_container_width=True)
with col_btn2:
    run_batch = st.button("💥 RUN BATCH ANALYSIS", use_container_width=True)
with col_btn3:
    pass

st.markdown("</div>", unsafe_allow_html=True)

# ===== PREDICTION FUNCTIONS =====
def predict_single_site(temp, pressure, humidity, wind_speed, air_density, solar_weight):
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
    combined = np.clip(solar_weight * s_norm + (1 - solar_weight) * w_norm, 0.0, 1.0)
    score = round(combined * 100, 2)
    return sp, wp, score

def batch_predict(df, temp_default, pressure_default, humidity_default,
                  wind_speed_default, air_density_default, solar_weight):
    results = []
    for _, row in df.iterrows():
        sp, wp, sc = predict_single_site(
            float(row.get("temp", temp_default)),
            float(row.get("pressure", pressure_default)),
            float(row.get("humidity", humidity_default)),
            float(row.get("wind_speed", wind_speed_default)),
            float(row.get("air_density", air_density_default)),
            solar_weight
        )
        results.append({
            "lon": row.get("lon", lon),
            "lat": row.get("lat", lat),
            "temp": row.get("temp", temp_default),
            "pressure": row.get("pressure", pressure_default),
            "humidity": row.get("humidity", humidity_default),
            "wind_speed": row.get("wind_speed", wind_speed_default),
            "air_density": row.get("air_density", air_density_default),
            "solar_pred": sp, "wind_pred": wp, "score": sc
        })
    return pd.DataFrame(results)

# ===== FILE UPLOAD =====
if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df_up
        st.markdown("<div class='cyan-card'>✅ CSV Successfully Loaded!</div>", unsafe_allow_html=True)
        st.dataframe(df_up.head(5), use_container_width=True)
    except Exception as e:
        st.markdown(f"<div class='orange-card'>❌ Error: {e}</div>", unsafe_allow_html=True)

if use_sample:
    sample = pd.DataFrame({
        "lat": [12.9716, 17.3850, 15.9129, 13.0827, 14.4426],
        "lon": [77.5946, 78.4867, 79.7400, 80.2707, 78.3900],
        "temp": [28, 30, 27, 29, 26],
        "pressure": [1010, 1008, 1006, 1012, 1011],
        "humidity": [45, 50, 55, 60, 42],
        "wind_speed": [6, 7, 5, 8, 4],
        "air_density": [1.225, 1.22, 1.23, 1.21, 1.225]
    })
    st.session_state.uploaded_df = sample
    st.markdown("<div class='cyan-card'>🎯 Demo Sample Loaded!</div>", unsafe_allow_html=True)
    st.dataframe(sample, use_container_width=True)

# ===== RUN PREDICTIONS =====
single_result = None
batch_df = None

if run_single:
    sp, wp, sc = predict_single_site(
        temp, pressure, humidity, wind_speed, air_density, solar_wt
    )
    single_result = {
        "temp": temp, "pressure": pressure, "humidity": humidity,
        "wind_speed": wind_speed, "air_density": air_density,
        "lon": lon, "lat": lat, "solar_pred": sp, "wind_pred": wp, "score": sc
    }
    st.balloons()
    st.markdown(f"""
    <div class='purple-card'>
        <h2 style='color: #00ff88; text-align: center;'>✨ SINGLE-SITE ANALYSIS COMPLETE ✨</h2>
        <p style='color: #00ffff; font-size: 1.2em; text-align: center;'>
        Score: <span style='color: #ffff00; font-size: 1.5em;'>{sc} / 100</span>
        </p>
        <p style='color: #ff6600; text-align: center;'>☀️ Solar: {sp:.2f} W/m² | 💨 Wind: {wp:.2f} kW</p>
    </div>
    """, unsafe_allow_html=True)

if run_batch:
    if st.session_state.uploaded_df is None:
        st.markdown("<div class='orange-card'>❌ Upload CSV first!</div>", unsafe_allow_html=True)
    else:
        with st.spinner("🔄 Running batch predictions..."):
            batch_df = batch_predict(
                st.session_state.uploaded_df,
                temp, pressure, humidity, wind_speed, air_density, solar_wt
            )
            st.session_state.predictions = batch_df
            st.balloons()
            st.markdown("<div class='cyan-card'>✅ Batch Predictions Complete!</div>", unsafe_allow_html=True)
            st.dataframe(batch_df.head(10), use_container_width=True)

if batch_df is None and st.session_state.predictions is not None:
    batch_df = st.session_state.predictions

# ===== RESULTS DISPLAY =====
st.markdown("<hr style='border: 2px solid #ff00ff; margin: 30px 0;'>", unsafe_allow_html=True)

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.markdown("<div class='purple-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00ff88;'>📊 RESULTS</h3>", unsafe_allow_html=True)
    
    if single_result:
        st.markdown(f"""
        <div class='metric-glow'>
        🎯 Score: {single_result['score']} / 100
        </div>
        <div class='metric-glow'>
        ☀️ Solar: {single_result['solar_pred']:.2f} W/m²
        </div>
        <div class='metric-glow'>
        💨 Wind: {single_result['wind_pred']:.2f} kW
        </div>
        """, unsafe_allow_html=True)
    elif batch_df is not None:
        st.markdown(f"""
        <div class='metric-glow'>
        📈 Sites: {len(batch_df)}
        </div>
        <div class='metric-glow'>
        ⭐ Avg Score: {batch_df['score'].mean():.2f}
        </div>
        """, unsafe_allow_html=True)
        csv_bytes = batch_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes, "predictions.csv", "text/csv", use_container_width=True)
    else:
        st.markdown("<p style='color: #00ffff;'>Run analysis to see results...</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with res_col2:
    st.markdown("<div class='glow-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00ffff;'>🤖 AI INSIGHTS</h3>", unsafe_allow_html=True)
    
    if st.session_state.ai_mode:
        if single_result:
            sc = single_result['score']
            rating = "🔥 EXCELLENT" if sc >= 75 else "⭐ GOOD" if sc >= 50 else "❌ LOW"
            st.markdown(f"<p style='color: #00ff88; font-size: 1.1em;'>{rating}</p>", unsafe_allow_html=True)
            st.markdown(f"""
            <p style='color: #00ffff;'>
            💧 Humidity {single_result['humidity']}% {'reduces' if single_result['humidity']>60 else 'optimizes'} solar output.
            </p>
            <p style='color: #ff00ff;'>
            🌪️ Wind {single_result['wind_speed']} m/s is {'excellent' if single_result['wind_speed']>8 else 'moderate'} for turbines.
            </p>
            """, unsafe_allow_html=True)
        elif batch_df is not None:
            avg_sc = batch_df['score'].mean()
            st.markdown(f"<p style='color: #00ff88;'>⭐ Average: {avg_sc:.2f} / 100</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #00ffff;'>Run a prediction for insights...</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #ff00ff;'>Enable AI Mode for insights</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ===== VISUALIZATIONS =====
st.markdown("<hr style='border: 2px solid #00ffff; margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #00ff88; text-shadow: 0 0 20px #00ff88;'>📊 VISUALIZATIONS</h2>", unsafe_allow_html=True)

# Solar Irradiance
hours = np.arange(0, 24, 1)
if single_result:
    solar_val = single_result['solar_pred']
elif batch_df is not None:
    solar_val = batch_df['solar_pred'].mean()
else:
    solar_val = 700.0

rad = np.clip(np.sin((hours - 6) / 12 * np.pi) * solar_val, 0, None)

fig_hours = px.area(
    x=hours, y=rad,
    labels={"x": "Hour", "y": "W/m²"},
    title="☀️ DAILY SOLAR IRRADIANCE PREDICTION"
)
fig_hours.update_traces(line_color='#ffff00', fillcolor='rgba(255, 102, 0, 0.4)')
fig_hours.update_layout(
    paper_bgcolor="#0a0e27", plot_bgcolor="#1a1a3e",
    font=dict(color='#00ff88', family='Courier New'),
    title_font_size=16,
    hovermode='x unified'
)
st.plotly_chart(fig_hours, use_container_width=True)

# Wind Power Curve
ws = np.linspace(0, 30, 50)
pw = np.clip(ws**3 * (air_density if 'air_density' in locals() else 1.225) * 0.01, 0, None)

fig_wind = px.line(
    x=ws, y=pw,
    labels={"x": "Wind speed (m/s)", "y": "Power (kW)"},
    title="💨 TURBINE POWER CURVE"
)
fig_wind.update_traces(line_color='#00ffff', line_width=3)
fig_wind.update_layout(
    paper_bgcolor="#0a0e27", plot_bgcolor="#1a1a3e",
    font=dict(color='#00ff88', family='Courier New'),
    title_font_size=16
)
st.plotly_chart(fig_wind, use_container_width=True)

# Distribution
if single_result:
    center = single_result['score'] / 100.0
elif batch_df is not None:
    center = batch_df['score'].mean() / 100.0
else:
    center = 0.5

samples = np.clip(np.random.normal(loc=center, scale=0.15, size=800), 0, 1)
fig_hist = px.histogram(
    samples, nbins=20,
    title="📈 SUITABILITY DISTRIBUTION",
    labels={"value": "Score", "count": "Frequency"}
)
fig_hist.update_traces(marker_color='#00ff88', marker_line_color='#ffff00', marker_line_width=2)
fig_hist.update_layout(
    paper_bgcolor="#0a0e27", plot_bgcolor="#1a1a3e",
    font=dict(color='#00ff88', family='Courier New'),
    title_font_size=16
)
st.plotly_chart(fig_hist, use_container_width=True)

# Globe & Maps
st.markdown("<div class='glow-card'>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #00ffff;'>🌍 GEO-SPATIAL ANALYSIS</h3>", unsafe_allow_html=True)

if globe_enabled:
    if single_result:
        gl_lon, gl_lat = single_result['lon'], single_result['lat']
    elif batch_df is not None:
        gl_lon, gl_lat = float(batch_df['lon'].mean()), float(batch_df['lat'].mean())
    elif st.session_state.uploaded_df is not None and 'lon' in st.session_state.uploaded_df.columns:
        gl_lon = float(st.session_state.uploaded_df['lon'].mean())
        gl_lat = float(st.session_state.uploaded_df['lat'].mean())
    else:
        gl_lon, gl_lat = lon, lat

    rot = st.slider("Rotate Globe", -180, 180, 0, step=10)
    
    globe_fig = go.Figure()
    if single_result:
        color = single_result['score'] / 100.0
        globe_fig.add_trace(go.Scattergeo(
            lon=[gl_lon], lat=[gl_lat],
            marker=dict(size=15, color=px.colors.sample_colorscale("Viridis", [color])[0]),
            text=[f"🎯 Score: {single_result['score']}"],
            hoverinfo='text'
        ))
    elif batch_df is not None:
        globe_fig.add_trace(go.Scattergeo(
            lon=batch_df['lon'], lat=batch_df['lat'],
            marker=dict(size=8, color=batch_df['score'], colorscale='Turbo', cmin=0, cmax=100,
                       colorbar=dict(title="Score")),
            hovertemplate="Score: %{marker.color:.2f}<extra></extra>"
        ))

    globe_fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=rot, lat=0),
        showcountries=True, showland=True, showocean=True,
        landcolor="rgb(20, 30, 60)", oceancolor="rgb(10, 20, 50)"
    )
    globe_fig.update_layout(height=450, paper_bgcolor="#0a0e27", font=dict(color='#00ff88'))
    st.plotly_chart(globe_fig, use_container_width=True)

if heatmap_enabled:
    st.markdown("<p style='color: #00ffff; font-weight: bold;'>🔥 Generating suitability heatmap...</p>", unsafe_allow_html=True)
    df_map = None
    if batch_df is not None:
        df_map = batch_df.copy()
    elif st.session_state.uploaded_df is not None and 'lat' in st.session_state.uploaded_df.columns:
        df_map = batch_predict(st.session_state.uploaded_df, temp, pressure, humidity, wind_speed, air_density, solar_wt)

    if df_map is not None and 'lon' in df_map.columns:
        try:
            fig_map = px.density_mapbox(
                df_map, lat='lat', lon='lon', z='score', radius=30,
                center=dict(lat=df_map['lat'].mean(), lon=df_map['lon'].mean()),
                zoom=6, mapbox_style="open-street-map", title="🔥 SUITABILITY HEATMAP"
            )
            fig_map.update_layout(height=400, paper_bgcolor="#0a0e27", font=dict(color='#00ff88'))
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.markdown(f"<p style='color: #ff0000;'>Map error: {str(e)[:50]}</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ===== PDF EXPORT =====
st.markdown("<hr style='border: 2px solid #ff00ff; margin: 30px 0;'>", unsafe_allow_html=True)
st.markdown("<h2 style='color: #ff00ff; text-shadow: 0 0 20px #ff00ff;'>📄 PDF REPORT GENERATOR</h2>", unsafe_allow_html=True)

st.markdown("<div class='orange-card'>", unsafe_allow_html=True)

if st.button("🚀 GENERATE PDF REPORT", use_container_width=True):
    with st.spinner("⏳ Generating PDF..."):
        fig_hours_buf = mpl_solar_hours_plot(hours, rad)
        fig_wind_buf = mpl_wind_curve(
            ws, pw,
            wind_speed if 'wind_speed' in locals() else 8,
            (single_result['wind_pred'] if single_result else
             (batch_df['wind_pred'].mean() if batch_df is not None else 0))
        )
        score_val = single_result['score'] if single_result else (
            batch_df['score'].mean() if batch_df is not None else 0
        )
        fig_score_buf = mpl_score_bar(score_val)

        data_for_pdf = {
            "temp": single_result['temp'] if single_result else (batch_df['temp'].mean() if batch_df is not None else temp),
            "pressure": single_result['pressure'] if single_result else (batch_df['pressure'].mean() if batch_df is not None else pressure),
            "humidity": single_result['humidity'] if single_result else (batch_df['humidity'].mean() if batch_df is not None else humidity),
            "wind_speed": single_result['wind_speed'] if single_result else (batch_df['wind_speed'].mean() if batch_df is not None else wind_speed),
            "air_density": single_result['air_density'] if single_result else (batch_df['air_density'].mean() if batch_df is not None else air_density),
            "solar_pred": single_result['solar_pred'] if single_result else (batch_df['solar_pred'].mean() if batch_df is not None else solar_val),
            "wind_pred": single_result['wind_pred'] if single_result else (batch_df['wind_pred'].mean() if batch_df is not None else pw.mean()),
            "score": score_val
        }

        pdf_buf = make_pdf_report(data_for_pdf, fig_hours_buf, fig_wind_buf, fig_score_buf)
        st.success("✅ PDF Ready!")
        st.download_button(
            "⬇️ DOWNLOAD PDF REPORT",
            pdf_buf.getvalue(),
            file_name="GeoAI_Report_Shashi_Teja.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
<hr style='border: 2px solid #00ff88; margin: 40px 0;'>
<div style='text-align: center; padding: 20px;'>
    <p style='color: #00ff88; font-size: 1.1em; text-shadow: 0 0 15px #00ff88;'>
    🚀 GEO-AI ULTIMATE | Renewable Energy Analysis Platform
    </p>
    <p style='color: #ff00ff;'>Created by Shashi Teja | 2025</p>
</div>
""", unsafe_allow_html=True)