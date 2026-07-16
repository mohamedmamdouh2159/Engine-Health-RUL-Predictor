import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import time
from datetime import datetime

st.set_page_config(
    layout="wide",
    page_title="NASA Engine Health Dashboard",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

COLORS = {
    "indigo": "#818CF8", "violet": "#A78BFA", "cyan": "#22D3EE",
    "success": "#34D399", "warning": "#FBBF24", "danger": "#FB7185",
    "text_dim": "#8B96AC",
}

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
MULTI_CONDITION_DATASETS = ["FD002", "FD004"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
DEAD_SENSORS = ["sensor_1", "sensor_5", "sensor_18", "sensor_19"]
ACTIVE_SENSORS = [s for s in SENSOR_COLS if s not in DEAD_SENSORS]
RUL_CAP = 125
DEGRADING_THRESHOLD = 30
PCA_N_COMPONENTS = 11
PCA_COLS = [f"pc{i+1}" for i in range(PCA_N_COMPONENTS)]

MODEL_DIR = "models"

SENSOR_META = {
    1: ("T2", "Total Temperature at Fan Inlet", "°R"),
    2: ("T24", "LPC Outlet Temperature", "°R"),
    3: ("T30", "HPC Outlet Temperature", "°R"),
    4: ("T50", "LPT Outlet Temperature", "°R"),
    5: ("P2", "Fan Inlet Pressure", "psia"),
    6: ("P15", "Bypass-Duct Pressure", "psia"),
    7: ("P30", "HPC Outlet Pressure", "psia"),
    8: ("Nf", "Physical Fan Speed", "rpm"),
    9: ("Nc", "Physical Core Speed", "rpm"),
    10: ("epr", "Engine Pressure Ratio", "ratio"),
    11: ("Ps30", "Static Pressure HPC Outlet", "psia"),
    12: ("phi", "Fuel Flow / Ps30", "pps/psi"),
    13: ("NRf", "Corrected Fan Speed", "rpm"),
    14: ("NRc", "Corrected Core Speed", "rpm"),
    15: ("BPR", "Bypass Ratio", "ratio"),
    16: ("farB", "Burner Fuel-Air Ratio", "ratio"),
    17: ("htBleed", "Bleed Enthalpy", "—"),
    18: ("Nf_dmd", "Demanded Fan Speed", "rpm"),
    19: ("PCNfR_dmd", "Demanded Corrected Fan Speed", "rpm"),
    20: ("W31", "HPT Coolant Bleed", "lbm/s"),
    21: ("W32", "LPT Coolant Bleed", "lbm/s"),
}

SENSOR_GROUPS = {
    "🌡️ Temperature": [1, 2, 3, 4],
    "💨 Pressure": [5, 6, 7, 11],
    "🌀 Speed": [8, 9, 13, 14, 18, 19],
    "⛽ Fuel & Flow": [10, 12, 15, 16],
    "📡 Other": [17, 20, 21],
}

OP_META = [
    ("op_setting_1", "Altitude", "ft"),
    ("op_setting_2", "Mach Number", "M"),
    ("op_setting_3", "Throttle Resolver Angle", "%"),
]

ALL_SENSOR_INPUT_KEYS = [f"sensor_{i}" for i in range(1, 22)]
ALL_OP_KEYS = [k for k, _, _ in OP_META]
ALL_INPUT_KEYS = ALL_OP_KEYS + ALL_SENSOR_INPUT_KEYS


class DemoFallbackModels:
    class _DemoKMeans:
        def predict(self, X):
            return np.array([abs(hash(tuple(np.round(X[0], 3)))) % 6])

    class _DemoScaler:
        def transform(self, X):
            return X

    class _DemoPCA:
        def transform(self, X):
            rng = np.random.default_rng(abs(hash(tuple(np.round(X[0], 3)))) % (2**32))
            return rng.normal(size=(1, PCA_N_COMPONENTS))

    class _DemoStage1:
        feature_importances_ = None
        def predict(self, X):
            seed = abs(hash(tuple(np.round(X[0], 3)))) % (2**32)
            return np.array([np.random.default_rng(seed).integers(0, 2)])
        def predict_proba(self, X):
            seed = abs(hash(tuple(np.round(X[0], 3)))) % (2**32)
            p = np.random.default_rng(seed).uniform(0.55, 0.97)
            return np.array([[1 - p, p]])

    class _DemoStage2:
        feature_importances_ = np.abs(np.random.default_rng(0).normal(size=len(ACTIVE_SENSORS)))
        def predict(self, X):
            seed = abs(hash(tuple(np.round(X[0], 3)))) % (2**32)
            return np.array([np.random.default_rng(seed).integers(1, DEGRADING_THRESHOLD)])

    def __init__(self):
        self.kmeans = self._DemoKMeans()
        self.pca = self._DemoPCA()
        self.stage1 = self._DemoStage1()
        self.stage2 = self._DemoStage2()
        self.scalers = {}

    def get_scaler(self, dataset_name, op_condition):
        return self._DemoScaler()


@st.cache_resource
def load_pipeline():
    required = ["kmeans_model.pkl", "scalers.pkl", "pca_model.pkl", "stage1_model.pkl", "stage2_model.pkl"]
    paths = {name: os.path.join(MODEL_DIR, name) for name in required}

    if all(os.path.exists(p) for p in paths.values()):
        kmeans = joblib.load(paths["kmeans_model.pkl"])
        scalers = joblib.load(paths["scalers.pkl"])
        pca = joblib.load(paths["pca_model.pkl"])
        stage1 = joblib.load(paths["stage1_model.pkl"])
        stage2 = joblib.load(paths["stage2_model.pkl"])

        class RealPipeline:
            def get_scaler(self, dataset_name, op_condition):
                return scalers.get((dataset_name, op_condition))

        real = RealPipeline()
        real.kmeans, real.pca, real.stage1, real.stage2, real.scalers = kmeans, pca, stage1, stage2, scalers
        return real, True

    return DemoFallbackModels(), False


def get_op_condition(dataset_name, op_raw, models):
    if dataset_name in MULTI_CONDITION_DATASETS:
        arr = np.array([[op_raw["op_setting_1"], op_raw["op_setting_2"], op_raw["op_setting_3"]]])
        return int(models.kmeans.predict(arr)[0])
    return 0


def scale_active_sensors(dataset_name, op_condition, sensor_raw, models):
    scaler = models.get_scaler(dataset_name, op_condition)
    raw_vec = np.array([[sensor_raw[c] for c in SENSOR_COLS]])
    if scaler is None:
        scaled_vec = raw_vec[0]
    else:
        scaled_vec = scaler.transform(raw_vec)[0]
    scaled_dict = dict(zip(SENSOR_COLS, scaled_vec))
    return np.array([[scaled_dict[c] for c in ACTIVE_SENSORS]]), scaled_dict


def run_pipeline(dataset_name, op_raw, sensor_raw, models):
    op_condition = get_op_condition(dataset_name, op_raw, models)
    active_vec, scaled_dict = scale_active_sensors(dataset_name, op_condition, sensor_raw, models)

    pca_vec = models.pca.transform(active_vec)
    is_degrading = int(models.stage1.predict(pca_vec)[0])
    proba = models.stage1.predict_proba(pca_vec)[0]
    confidence = float(proba[is_degrading]) * 100

    rul_pred = None
    if is_degrading == 1:
        raw_rul = float(models.stage2.predict(active_vec)[0])
        rul_pred = int(round(max(0, min(RUL_CAP, raw_rul))))

    return {
        "op_condition": op_condition,
        "is_degrading": is_degrading,
        "confidence": confidence,
        "rul": rul_pred,
        "active_vec": active_vec[0],
    }


if "history" not in st.session_state:
    st.session_state.history = []
if "last_run_ms" not in st.session_state:
    st.session_state.last_run_ms = None
if "csv_df" not in st.session_state:
    st.session_state.csv_df = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None


def reset_inputs():
    for k in ALL_INPUT_KEYS + ["unit_number", "time_cycles"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.history = []
    st.session_state.last_result = None
    st.session_state.last_inputs = None


def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(129,140,248,0.10) 0%, transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(34,211,238,0.08) 0%, transparent 40%),
                #0A0E1A;
            color: #F1F5F9;
        }
        #MainMenu, header, footer {visibility: hidden;}
        div.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px;}
        section[data-testid="stSidebar"] { background: #0D1220; border-right: 1px solid rgba(255,255,255,0.06); }
        .side-brand { display: flex; align-items: center; gap: 0.7rem; padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 1.2rem; }
        .side-brand .logo { width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, #818CF8, #A78BFA 55%, #22D3EE); display: flex; align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 4px 18px rgba(129,140,248,0.35); }
        .side-brand .title {font-weight: 800; font-size: 1.02rem; color: #F1F5F9; line-height: 1.15;}
        .side-brand .subtitle {font-size: 0.72rem; color: #6B7690; font-weight: 500;}
        .side-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7690; margin: 1.1rem 0 0.5rem 0; }
        .about-card { background: #12182A; border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 1rem 1.1rem; font-size: 0.83rem; color: #8B96AC; line-height: 1.55; }
        .dev-card { display: flex; align-items: center; gap: 0.7rem; background: linear-gradient(135deg, rgba(129,140,248,0.10), rgba(34,211,238,0.06)); border: 1px solid rgba(129,140,248,0.20); border-radius: 14px; padding: 0.85rem 1rem; margin-top: 0.6rem; }
        .dev-card .avatar { width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, #818CF8, #22D3EE); display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; color: #0A0E1A; flex-shrink: 0; }
        .dev-card .name {font-weight: 700; font-size: 0.85rem; color: #F1F5F9;}
        .dev-card .role {font-size: 0.72rem; color: #8B96AC;}
        .side-links {margin-top: 0.6rem; display: flex; gap: 0.5rem;}
        .side-links a { flex: 1; text-align: center; font-size: 0.75rem; font-weight: 600; color: #C7D2FE; background: rgba(129,140,248,0.10); border: 1px solid rgba(129,140,248,0.22); border-radius: 9px; padding: 0.45rem 0; text-decoration: none; }
        .csv-badge { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.35); color: #34D399; border-radius: 10px; padding: 0.5rem 0.8rem; font-size: 0.78rem; font-weight: 600; margin-top: 0.5rem; }
        .model-status { border-radius: 10px; padding: 0.55rem 0.8rem; font-size: 0.76rem; font-weight: 600; margin-top: 0.5rem; }
        .hero-card { position: relative; background: linear-gradient(160deg, #12182A 0%, #0D1220 100%); border: 1px solid rgba(129,140,248,0.18); border-radius: 28px; padding: 2.8rem 2rem 2.4rem 2rem; text-align: center; margin-bottom: 1.4rem; overflow: hidden; }
        .hero-card::before { content: ""; position: absolute; inset: 0; background: radial-gradient(circle at 50% -10%, rgba(129,140,248,0.20), transparent 60%); pointer-events: none; }
        .hero-eyebrow { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #A5B4FC; background: rgba(129,140,248,0.12); border: 1px solid rgba(129,140,248,0.3); padding: 0.35rem 0.9rem; border-radius: 999px; margin-bottom: 1.1rem; position: relative; z-index: 1; }
        .hero-card h1 { font-size: 2.65rem; font-weight: 900; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 0.7rem; position: relative; z-index: 1; background: linear-gradient(100deg, #F1F5F9 20%, #C7D2FE 55%, #67E8F9 90%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero-card p { color: #8B96AC; font-size: 1.03rem; max-width: 560px; margin: 0 auto; position: relative; z-index: 1; }
        .badge-row {margin-top: 1.3rem; position: relative; z-index: 1;}
        .badge { display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.09); color: #C7D2FE; padding: 0.4rem 0.95rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin: 0.25rem; }
        .stat-card { background: #12182A; border: 1px solid rgba(255,255,255,0.07); border-radius: 18px; padding: 1.25rem 1.1rem; transition: all 0.25s cubic-bezier(0.16,1,0.3,1); overflow: hidden; }
        .stat-card:hover { transform: translateY(-5px); border-color: rgba(129,140,248,0.45); box-shadow: 0 14px 34px rgba(76,29,149,0.25); }
        .stat-card .icon-wrap { width: 38px; height: 38px; border-radius: 11px; background: linear-gradient(135deg, rgba(129,140,248,0.18), rgba(34,211,238,0.12)); display: flex; align-items: center; justify-content: center; font-size: 1.15rem; margin-bottom: 0.8rem; }
        .stat-card .value {font-size: 1.32rem; font-weight: 800; color: #F1F5F9; letter-spacing: -0.01em;}
        .stat-card .label {color: #6B7690; font-size: 0.78rem; font-weight: 500; margin-top: 0.15rem;}
        .glass-card { background: rgba(18,24,42,0.55); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.07); border-radius: 22px; padding: 1.7rem 1.7rem 1.4rem 1.7rem; margin-bottom: 1.5rem; }
        .section-title { font-size: 1.06rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.9rem; display: flex; align-items: center; gap: 0.5rem; }
        div[data-testid="stNumberInput"] label { color: #9AA4BC !important; font-size: 0.8rem !important; font-weight: 600 !important; }
        div[data-testid="stNumberInput"] input { background-color: #171F35 !important; color: #F1F5F9 !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 11px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.86rem !important; padding: 0.55rem 0.7rem !important; }
        div[data-testid="stNumberInput"] input:focus { border-color: #818CF8 !important; box-shadow: 0 0 0 3px rgba(129,140,248,0.18) !important; }
        div[data-testid="stSelectbox"] > div > div { background-color: #171F35 !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 11px !important; color: #F1F5F9 !important; }
        section[data-testid="stFileUploader"] > div { background: #171F35 !important; border: 1.5px dashed rgba(129,140,248,0.3) !important; border-radius: 14px !important; }
        div.stButton > button { background: linear-gradient(135deg, #6366F1, #818CF8 45%, #22D3EE); color: #0A0E1A; border: none; border-radius: 16px; font-size: 1.15rem; font-weight: 800; box-shadow: 0 8px 28px rgba(99,102,241,0.4); padding: 1.1rem 0; }
        div.stButton > button p {color: #0A0E1A !important; font-weight: 800 !important; font-size: 1.15rem !important; text-align: center !important;}
        div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 14px 36px rgba(34,211,238,0.45); filter: brightness(1.06); }
        section[data-testid="stSidebar"] div.stButton > button { background: #171F35; color: #C7D2FE !important; border: 1px solid rgba(255,255,255,0.09); box-shadow: none; font-weight: 600; font-size: 0.95rem; padding: 0.7rem 0; }
        section[data-testid="stSidebar"] div.stButton > button p {color: #C7D2FE !important; font-weight: 600 !important; font-size: 0.95rem !important;}

        /* توسيط زرار Predict داخل منطقة المحتوى فعليًا (من غير أرقام تخمينية) */
        .predict-wrap {
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }
        .predict-wrap .stButton { width: 340px; }

        .stTabs [data-baseweb="tab-list"] { gap: 6px; background: #10152A; padding: 6px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
        .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 10px; padding: 8px 18px; color: #6B7690; font-weight: 600; font-size: 0.87rem; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #6366F1, #818CF8) !important; color: #0A0E1A !important; box-shadow: 0 4px 14px rgba(99,102,241,0.35); }
        .stTabs [aria-selected="true"] p {color: #0A0E1A !important; font-weight: 700 !important;}
        .metric-card { background: #12182A; border: 1px solid rgba(255,255,255,0.07); border-radius: 18px; padding: 1.15rem 1.3rem; height: 100%; }
        .metric-card:hover {border-color: rgba(129,140,248,0.4); transform: translateY(-3px);}
        .metric-card .m-label { font-size: 0.73rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #6B7690; margin-bottom: 0.5rem; }
        .metric-card .m-value { font-size: 1.65rem; font-weight: 800; color: #F1F5F9; letter-spacing: -0.01em; font-family: 'JetBrains Mono', monospace; }
        .metric-card .m-unit {font-size: 0.85rem; color: #6B7690; font-weight: 500; margin-left: 0.2rem;}
        .m-delta { font-size: 0.78rem; font-weight: 700; margin-top: 0.35rem; }
        .status-pill { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.5rem 1.1rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; }
        .alert-banner { display: flex; align-items: center; gap: 0.8rem; border-radius: 14px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.9rem; font-weight: 600; }
        .empty-state { text-align: center; padding: 3rem 1.5rem; border: 1.5px dashed rgba(255,255,255,0.1); border-radius: 20px; color: #5B657C; }
        .empty-state .icon { font-size: 2.2rem; margin-bottom: 0.8rem; opacity: 0.7; }
        .empty-state .title { color: #9AA4BC; font-weight: 700; font-size: 1rem; margin-bottom: 0.3rem; }
        .footer-box { text-align: center; padding: 2rem 0 0.5rem 0; color: #5B657C; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.06); margin-top: 2rem; line-height: 1.8; }
        .footer-box b {color: #8B96AC;}

        /* كروت الـ Gauge والـ Feature Importance عن طريق st.container(border=True, key=...) */
        .st-key-gauge_card, .st-key-feature_card {
            background: rgba(18,24,42,0.55) !important;
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 22px !important;
            padding: 1.7rem 1.7rem 1.4rem 1.7rem !important;
        }
        .st-key-history_card {
            background: rgba(18,24,42,0.55) !important;
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 22px !important;
            padding: 1.7rem 1.7rem 1.4rem 1.7rem !important;
        }
    </style>
    """, unsafe_allow_html=True)


def hero_section():
    st.markdown("""
    <div class="hero-card">
        <div class="hero-eyebrow"><span>🛰️</span> TWO-STAGE MODEL</div>
        <h1>Engine Health Intelligence</h1>
        <p>Stage 1 flags Healthy vs Degrading engines. Stage 2 estimates Remaining Useful Life — only when the engine is actually degrading.</p>
        <div class="badge-row">
            <span class="badge">🧠 RandomForest Classifier + Regressor</span>
            <span class="badge">🛰️ NASA C-MAPSS Dataset</span>
            <span class="badge">📉 PCA + Per-Condition Scaling</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def quick_stats():
    cols = st.columns(4)
    stats = [
        ("📊", "21", "Raw Sensors Collected"),
        ("🧬", "17", "Active Sensors Used"),
        ("🎯", "11", "PCA Components (Stage 1)"),
        ("🔗", "125", "RUL Cap (cycles)"),
    ]
    for col, (icon, value, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="icon-wrap">{icon}</div>
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('<div style="height: 2.2rem;"></div>', unsafe_allow_html=True)


def sidebar(models_loaded):
    with st.sidebar:
        st.markdown("""
        <div class="side-brand">
            <div class="logo">🛰️</div>
            <div>
                <div class="title">RUL Predictor</div>
                <div class="subtitle">NASA C-MAPSS Suite</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if models_loaded:
            st.markdown('<div class="model-status" style="background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.35); color:#34D399;">✓ Trained pipeline loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="model-status" style="background:rgba(251,191,36,0.12); border:1px solid rgba(251,191,36,0.35); color:#FBBF24;">⚠ Put your .pkl files in /{MODEL_DIR}</div>', unsafe_allow_html=True)

        st.markdown('<div class="side-label">Batch Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed",
                                     help="Upload a CSV with matching column names to auto-fill inputs from a row")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.csv_df = df
                st.markdown(f'<div class="csv-badge">✓ Loaded {len(df)} rows</div>', unsafe_allow_html=True)
                row_idx = st.selectbox("Row to load", options=list(range(len(df))), key="csv_row_pick")
                if st.button("⬆️  Apply Row to Inputs"):
                    row = df.iloc[row_idx]
                    for k in ALL_INPUT_KEYS + ["unit_number", "time_cycles"]:
                        if k in row.index:
                            st.session_state[k] = float(row[k])
                    if "dataset" in row.index:
                        st.session_state["dataset_select"] = row["dataset"]
                    st.rerun()
            except Exception:
                st.error("Couldn't parse this CSV. Check the format and try again.")

        st.write("")
        if st.button("↻  Reset Inputs"):
            reset_inputs()
            st.rerun()

        st.markdown('<div class="side-label">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-card">
            Two-stage engine health system: a classifier flags Healthy vs
            Degrading, then a regressor estimates Remaining Useful Life (RUL)
            only for degrading engines. Built on the NASA C-MAPSS dataset.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="side-label">Developer</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="dev-card">
            <div class="avatar">MM</div>
            <div>
                <div class="name">Mohamed Mamdouh</div>
                <div class="role">ML Engineer</div>
            </div>
        </div>
        <div class="side-links">
            <a href="https://github.com" target="_blank">GitHub</a>
            <a href="https://linkedin.com" target="_blank">LinkedIn</a>
        </div>
        """, unsafe_allow_html=True)


def identifiers_and_dataset():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏷️ Engine Identification</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        dataset_name = st.selectbox("Sub-dataset", DATASETS, key="dataset_select")
    with c2:
        unit_number = st.number_input("Unit Number (engine ID)", value=1, step=1, key="unit_number")
    with c3:
        time_cycles = st.number_input("Time Cycles", value=1, step=1, key="time_cycles")
    st.markdown('</div>', unsafe_allow_html=True)
    return dataset_name, unit_number, time_cycles


def operational_settings():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Operational Settings</div>', unsafe_allow_html=True)
    op_settings = {}
    cols = st.columns(3)
    for col, (key, label, unit) in zip(cols, OP_META):
        with col:
            op_settings[key] = st.number_input(f"{label} ({unit})", value=0.0, step=0.01, format="%.4f", key=key)
    st.markdown('</div>', unsafe_allow_html=True)
    return op_settings


def sensor_inputs():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📡 Sensor Measurements</div>', unsafe_allow_html=True)

    sensor_values = {}
    tabs = st.tabs(list(SENSOR_GROUPS.keys()))
    for tab, (group_name, sensor_ids) in zip(tabs, SENSOR_GROUPS.items()):
        with tab:
            st.write("")
            for i in range(0, len(sensor_ids), 2):
                cols = st.columns(2)
                for col, sid in zip(cols, sensor_ids[i:i+2]):
                    code, full_name, unit = SENSOR_META[sid]
                    with col:
                        sensor_values[f"sensor_{sid}"] = st.number_input(
                            f"{code} — {full_name}", value=0.0, step=0.01, format="%.4f",
                            key=f"sensor_{sid}", help=unit
                        )

    st.markdown('</div>', unsafe_allow_html=True)
    return sensor_values


def prediction_section():
    st.markdown('<div class="predict-wrap">', unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict Engine Health")
    st.markdown('</div>', unsafe_allow_html=True)
    return predict_clicked


def empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🛰️</div>
        <div class="title">No prediction yet</div>
    </div>
    """, unsafe_allow_html=True)


def gauge_chart(confidence, is_degrading):
    color = COLORS["danger"] if is_degrading else COLORS["success"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={'suffix': "%", 'font': {'color': "#F1F5F9", 'size': 42, 'family': "Inter"}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#5B657C", 'tickfont': {'color': '#5B657C', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': "#171F35",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(251,113,133,0.15)'},
                {'range': [50, 80], 'color': 'rgba(251,191,36,0.15)'},
                {'range': [80, 100], 'color': 'rgba(52,211,153,0.15)'},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={'color': "#F1F5F9", 'family': "Inter"}, height=280,
                       margin=dict(l=25, r=25, t=35, b=15))
    return fig


def feature_importance_chart(models, is_degrading):
    if is_degrading and getattr(models.stage2, "feature_importances_", None) is not None:
        importances = np.array(models.stage2.feature_importances_)
        features = [SENSOR_META[int(s.split("_")[1])][0] for s in ACTIVE_SENSORS]
    elif getattr(models.stage1, "feature_importances_", None) is not None:
        importances = np.array(models.stage1.feature_importances_)
        features = PCA_COLS
    else:
        rng = np.random.default_rng(0)
        importances = rng.random(len(ACTIVE_SENSORS))
        features = [SENSOR_META[int(s.split("_")[1])][0] for s in ACTIVE_SENSORS]

    order = np.argsort(importances)[-8:]
    features = [features[i] for i in order]
    importances = importances[order]

    fig = px.bar(x=importances, y=features, orientation='h')
    fig.update_traces(marker=dict(color=importances, colorscale=[[0, "#6366F1"], [1, "#22D3EE"]], line=dict(width=0)),
                       marker_cornerradius=6,
                       hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={'color': "#8B96AC", 'family': "Inter", 'size': 12},
                       showlegend=False, coloraxis_showscale=False, height=280,
                       margin=dict(l=10, r=20, t=25, b=10),
                       xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, title=None),
                       yaxis=dict(showgrid=False, title=None, tickfont=dict(color="#C7D2FE", size=12)),
                       bargap=0.35)
    return fig


def history_chart():
    hist = st.session_state.history
    runs = list(range(1, len(hist) + 1))
    rul_vals = [h["rul"] if h["rul"] is not None else RUL_CAP for h in hist]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=runs, y=rul_vals, mode="lines+markers",
        line=dict(color="#22D3EE", width=2.5, shape="spline"),
        marker=dict(size=7, color="#0A0E1A", line=dict(color="#22D3EE", width=2)),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.08)",
        hovertemplate="Run %{x}<br>RUL: %{y} cycles<extra></extra>"
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={'color': "#8B96AC", 'family': "Inter", 'size': 11},
                       height=220, margin=dict(l=10, r=10, t=15, b=10), showlegend=False,
                       xaxis=dict(showgrid=False, title=None, dtick=1),
                       yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=None))
    return fig


def compute_and_store_result(dataset_name, unit_number, time_cycles, op_settings, sensor_values, models):
    with st.spinner("Running Stage 1 → Stage 2 pipeline..."):
        result = run_pipeline(dataset_name, op_settings, sensor_values, models)

    st.session_state.history.append({
        "unit_number": unit_number, "dataset": dataset_name,
        "is_degrading": result["is_degrading"], "confidence": result["confidence"], "rul": result["rul"],
        "ts": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.last_result = result
    st.session_state.last_inputs = {
        "dataset_name": dataset_name, "unit_number": unit_number, "time_cycles": time_cycles,
        "op_settings": op_settings, "sensor_values": sensor_values,
    }


def render_results(models):
    result = st.session_state.last_result
    inputs = st.session_state.last_inputs
    is_degrading = result["is_degrading"]
    confidence = result["confidence"]
    rul_pred = result["rul"]

    hist = st.session_state.history
    prev = hist[-2] if len(hist) > 1 else None

    # ===== Output: Health status + Confidence دايمًا، RUL بس لو Degrading =====
    if is_degrading:
        st.markdown(f"""
        <div class="alert-banner" style="background:{COLORS['danger']}14; border:1px solid {COLORS['danger']}55; color:{COLORS['danger']};">
            ⚠️ Engine flagged <b>Degrading</b> — {confidence:.0f}% confident this engine is degrading. Estimated RUL: ≈{rul_pred} cycles remaining.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-banner" style="background:{COLORS['success']}14; border:1px solid {COLORS['success']}55; color:{COLORS['success']};">
            ✅ Engine flagged <b>Healthy</b> — {confidence:.0f}% confident this engine is healthy. No immediate concern (RUL cap: {RUL_CAP} cycles).
        </div>
        """, unsafe_allow_html=True)

    def delta_html(curr, prev_val, fmt="{:.0f}"):
        if prev_val is None:
            return '<div class="m-delta" style="color:#6B7690;">First run</div>'
        diff = curr - prev_val
        if diff == 0:
            return '<div class="m-delta" style="color:#6B7690;">No change</div>'
        arrow = "▲" if diff > 0 else "▼"
        clr = COLORS["success"] if diff > 0 else COLORS["danger"]
        return f'<div class="m-delta" style="color:{clr};">{arrow} {fmt.format(abs(diff))} vs last run</div>'

    m1, m2, m3 = st.columns(3)
    with m1:
        status_color = COLORS["danger"] if is_degrading else COLORS["success"]
        status_label = "Degrading" if is_degrading else "Healthy"
        status_icon = "⚠" if is_degrading else "✓"
        st.markdown(f"""
        <div class="metric-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center;">
            <div class="m-label">Health Status</div>
            <div class="status-pill" style="background:{status_color}1F; color:{status_color}; border:1px solid {status_color}55;">
                {status_icon} {status_label}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="m-label">Confidence</div>
            <div class="m-value">{confidence:.0f}<span class="m-unit">%</span></div>
            {delta_html(confidence, prev["confidence"] if prev else None)}
        </div>
        """, unsafe_allow_html=True)
    with m3:
        if is_degrading:
            st.markdown(f"""
            <div class="metric-card">
                <div class="m-label">Predicted RUL</div>
                <div class="m-value">{rul_pred}<span class="m-unit">cycles</span></div>
                {delta_html(rul_pred, prev["rul"] if (prev and prev["rul"] is not None) else None)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="m-label">Predicted RUL</div>
                <div class="m-value">—<span class="m-unit">n/a</span></div>
                <div class="m-delta" style="color:#6B7690;">Cap value: {RUL_CAP} cycles</div>
            </div>
            """, unsafe_allow_html=True)

    # ===== كروت الـ Gauge والـ Feature Importance باستخدام st.container(border=True) =====
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True, key="gauge_card"):
            st.markdown('<div class="section-title">🎯 Stage 1 Confidence</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(confidence, is_degrading), use_container_width=True, config={'displayModeBar': False})
    with c2:
        with st.container(border=True, key="feature_card"):
            st.markdown('<div class="section-title">📊 Feature Importance</div>', unsafe_allow_html=True)
            st.plotly_chart(feature_importance_chart(models, is_degrading), use_container_width=True, config={'displayModeBar': False})

    if len(st.session_state.history) > 1:
        with st.container(border=True, key="history_card"):
            hcol1, hcol2 = st.columns([5, 1])
            with hcol1:
                st.markdown('<div class="section-title">📈 Prediction History</div>', unsafe_allow_html=True)
            with hcol2:
                if st.button("Clear", key="clear_history"):
                    st.session_state.history = []
                    st.session_state.last_result = None
                    st.rerun()
            st.plotly_chart(history_chart(), use_container_width=True, config={'displayModeBar': False})

    with st.expander("📋 View Raw Input Snapshot"):
        input_data = {"dataset": inputs["dataset_name"], "unit_number": inputs["unit_number"],
                       "time_cycles": inputs["time_cycles"], **inputs["op_settings"], **inputs["sensor_values"],
                       "op_condition (derived)": result["op_condition"]}
        st.dataframe(pd.DataFrame([input_data]), use_container_width=True)


def footer():
    st.markdown("""
    <div class="footer-box">
        NASA C-MAPSS Dataset · Two-Stage Health Classification & RUL Regression<br>
        Built with using Streamlit · 
    </div>
    """, unsafe_allow_html=True)


def main():
    load_css()
    models, models_loaded = load_pipeline()
    sidebar(models_loaded)
    hero_section()
    quick_stats()

    dataset_name, unit_number, time_cycles = identifiers_and_dataset()
    op_settings = operational_settings()
    sensor_values = sensor_inputs()
    predict_clicked = prediction_section()

    if predict_clicked:
        compute_and_store_result(dataset_name, unit_number, time_cycles, op_settings, sensor_values, models)

    if st.session_state.last_result is not None:
        render_results(models)
    else:
        empty_state()

    footer()


if __name__ == "__main__":
    main()