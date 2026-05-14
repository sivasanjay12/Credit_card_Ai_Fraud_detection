import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from feature_engineering import engineer_features

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark glassmorphism theme) ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    min-height: 100vh;
}

/* ── hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeInDown 0.8s ease;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff3cac);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin: 0;
}
.hero p {
    color: #8899aa;
    font-size: 1.1rem;
    margin-top: .4rem;
}

/* ── glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
    animation: fadeIn 0.6s ease;
}

/* ── result banners ── */
.result-fraud {
    background: linear-gradient(135deg, rgba(255,45,85,0.18), rgba(180,0,50,0.1));
    border: 1px solid rgba(255,45,85,0.5);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite, slideUp 0.5s ease;
}
.result-safe {
    background: linear-gradient(135deg, rgba(0,212,100,0.18), rgba(0,150,70,0.1));
    border: 1px solid rgba(0,212,100,0.5);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite, slideUp 0.5s ease;
}
.result-title { font-size: 2rem; font-weight: 800; margin: 0; }
.result-sub   { font-size: 1rem; color: #ccc; margin-top: .4rem; }

/* ── metric tiles ── */
.metric-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.metric-tile .val { font-size: 1.8rem; font-weight: 700; }
.metric-tile .lbl { font-size: .78rem; color: #778899; text-transform: uppercase; letter-spacing: 1px; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,14,26,0.85);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* ── sliders & inputs ── */
.stSlider > div > div { background: rgba(123,47,247,0.3); }

/* ── button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #7b2ff7, #00d4ff);
    color: white;
    border: none;
    border-radius: 12px;
    padding: .85rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 700;
    cursor: pointer;
    transition: transform .15s, box-shadow .15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(123,47,247,0.5);
}

/* ── tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: .3rem;
}
.stTabs [data-baseweb="tab"] {
    color: #778899;
    border-radius: 9px;
    padding: .5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#7b2ff7,#00d4ff);
    color: white !important;
}

/* ── animations ── */
@keyframes fadeInDown {
    from { opacity:0; transform:translateY(-24px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeIn {
    from { opacity:0; } to { opacity:1; }
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,45,85,0); }
    50%      { box-shadow: 0 0 30px 6px rgba(255,45,85,0.25); }
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,212,100,0); }
    50%      { box-shadow: 0 0 30px 6px rgba(0,212,100,0.2); }
}
</style>
""", unsafe_allow_html=True)

# ── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model     = joblib.load(PROJECT_ROOT / "models" / "fraud_model.pkl")
    threshold = joblib.load(PROJECT_ROOT / "models" / "optimal_threshold.pkl")
    scaler    = joblib.load(PROJECT_ROOT / "data" / "processed" / "scaler.joblib")
    return model, threshold, scaler

def run_prediction(df: pd.DataFrame):
    model, threshold, scaler = load_artifacts()
    df_fe    = engineer_features(df.copy(), "Amount", "Time")
    X_scaled = scaler.transform(df_fe)
    prob     = model.predict_proba(X_scaled)[:, 1]
    return float(prob[0]), bool(prob[0] >= threshold)

# ── Gauge chart ──────────────────────────────────────────────────────────────
def fraud_gauge(prob: float) -> go.Figure:
    color = "#ff2d55" if prob >= 0.5 else "#00d464"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 2),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": "Fraud Probability", "font": {"size": 15, "color": "#aabbcc"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#445566",
                     "tickfont": {"color": "#778899"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  40], "color": "rgba(0,212,100,0.10)"},
                {"range": [40, 70], "color": "rgba(255,165,0,0.10)"},
                {"range": [70,100], "color": "rgba(255,45,85,0.15)"},
            ],
            "threshold": {
                "line":  {"color": color, "width": 4},
                "thickness": 0.80,
                "value": round(prob * 100, 2),
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccddee",
    )
    return fig

# ── Feature-importance bar chart ─────────────────────────────────────────────
def feature_importance_chart() -> go.Figure:
    model, _, _ = load_artifacts()
    feat_names = (
        [f"V{i}" for i in range(1, 29)] + ["LogAmount", "Hour"]
    )
    importances = model.feature_importances_
    top_n = 15
    idx   = np.argsort(importances)[::-1][:top_n]
    names = [feat_names[i] if i < len(feat_names) else f"F{i}" for i in idx]
    vals  = importances[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0,"#7b2ff7"],[1,"#00d4ff"]],
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#aabbcc", size=11),
    ))
    fig.update_layout(
        title=dict(text="Top Feature Importances", font=dict(color="#ccddee", size=15)),
        xaxis=dict(showgrid=False, color="#556677"),
        yaxis=dict(autorange="reversed", color="#aabbcc"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=30, t=40, b=10),
        font_color="#ccddee",
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🛡️ Credit Card Fraud Detector</h1>
  <p>AI-powered real-time transaction analysis · Powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)

# ── Load model once & show status ────────────────────────────────────────────
with st.spinner("Loading model artifacts…"):
    try:
        load_artifacts()
        st.success("✅ Model loaded successfully", icon="🤖")
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 Predict Transaction", "📊 Model Insights", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💳 Transaction Details")

        time_val   = st.number_input("Time (seconds since first transaction)",
                                     min_value=0.0, max_value=172800.0,
                                     value=50000.0, step=100.0)
        amount_val = st.number_input("Transaction Amount (USD)",
                                     min_value=0.0, max_value=25000.0,
                                     value=120.0, step=1.0)

        st.markdown("#### 🔢 PCA Features (V1 – V28)")
        st.caption("These are anonymised principal components from the original dataset.")

        v_vals = {}
        cols_a, cols_b = st.columns(2)
        for i in range(1, 29):
            col = cols_a if i % 2 == 1 else cols_b
            v_vals[f"V{i}"] = col.number_input(
                f"V{i}", value=0.0, format="%.4f", key=f"v{i}"
            )

        predict_btn = st.button("🚀 Analyse Transaction", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown("### 📈 Result")

        if predict_btn:
            input_dict = {"Time": time_val, "Amount": amount_val, **v_vals}
            df_input   = pd.DataFrame([input_dict])

            with st.spinner("Analysing transaction…"):
                progress = st.progress(0)
                for p in range(0, 101, 10):
                    time.sleep(0.04)
                    progress.progress(p)
                prob, is_fraud = run_prediction(df_input)
                progress.empty()

            # Result banner
            if is_fraud:
                st.markdown(f"""
                <div class="result-fraud">
                  <div class="result-title" style="color:#ff2d55;">⚠️ FRAUD DETECTED</div>
                  <div class="result-sub">This transaction shows strong signs of fraud.</div>
                  <div style="font-size:2.4rem;font-weight:900;color:#ff6080;margin-top:.8rem;">
                    {prob*100:.2f}% probability
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                  <div class="result-title" style="color:#00d464;">✅ LEGITIMATE</div>
                  <div class="result-sub">Transaction appears to be safe.</div>
                  <div style="font-size:2.4rem;font-weight:900;color:#00d464;margin-top:.8rem;">
                    {prob*100:.2f}% fraud probability
                  </div>
                </div>""", unsafe_allow_html=True)

            # Gauge
            st.plotly_chart(fraud_gauge(prob), use_container_width=True)

            # Risk level
            risk = ("🟢 LOW"   if prob < 0.3
                    else "🟡 MEDIUM" if prob < 0.6
                    else "🔴 HIGH")
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-tile"><div class="val">{prob*100:.1f}%</div>'
                        f'<div class="lbl">Fraud Prob</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-tile"><div class="val">{risk}</div>'
                        f'<div class="lbl">Risk Level</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-tile"><div class="val">'
                        f'{"Fraud" if is_fraud else "Safe"}</div>'
                        f'<div class="lbl">Decision</div></div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="glass-card" style="text-align:center;padding:3rem;">'
                        '<div style="font-size:4rem;">🛡️</div>'
                        '<p style="color:#556677;margin-top:1rem;">'
                        'Fill in the transaction details and click <b>Analyse Transaction</b>.</p>'
                        '</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Model Insights
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🧠 Model Overview")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("XGBoost", "Algorithm"),
        ("30", "Input Features"),
        ("≤25ms", "Inference Time"),
        ("Optimised", "Threshold"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f'<div class="metric-tile"><div class="val" style="color:#00d4ff;">{val}</div>'
                     f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    try:
        st.plotly_chart(feature_importance_chart(), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render feature importance: {e}")

    # Simulated class distribution donut
    st.markdown("### 📊 Typical Dataset Class Distribution")
    fig_donut = go.Figure(go.Pie(
        labels=["Legitimate (99.8%)", "Fraudulent (0.2%)"],
        values=[99.8, 0.2],
        hole=0.6,
        marker=dict(colors=["#00d464", "#ff2d55"],
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textfont=dict(color="#ffffff"),
    ))
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccddee",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(font=dict(color="#aabbcc")),
        annotations=[dict(text="Class<br>Split", x=0.5, y=0.5,
                          font_size=16, font_color="#aabbcc", showarrow=False)],
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – About
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""
### 🛡️ About This Application

This **Credit Card Fraud Detection** system uses a trained **XGBoost** classifier to identify
potentially fraudulent credit card transactions in real time.

#### How It Works
1. **Input** — Enter transaction `Time`, `Amount`, and 28 anonymised PCA features (V1–V28).
2. **Feature Engineering** — `LogAmount` and `Hour` are derived automatically.
3. **Scaling** — Features are normalised using a pre-fitted scaler.
4. **Prediction** — The model outputs a fraud probability against an optimised decision threshold.

#### Dataset
- Based on the **[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** dataset.
- Contains transactions from European cardholders (September 2013).
- Highly imbalanced: only **0.17%** of transactions are fraudulent.

#### Model Pipeline
| Step | Detail |
|------|--------|
| Algorithm | XGBoost Classifier |
| Imbalance handling | SMOTE / class weighting |
| Threshold | Optimised for F1 / cost sensitivity |
| Features | 30 (V1–V28 + LogAmount + Hour) |

> **Note:** V1–V28 are principal components; original features are confidential.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#445566;font-size:.82rem;">
  Credit Card Fraud Detection · Capstone Project · Built with Streamlit & XGBoost
</div>
""", unsafe_allow_html=True)
