import pickle
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Page config
st.set_page_config(
    page_title="WSN Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme CSS
st.markdown(
    """
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Page background ── */
  .stApp { background-color: #F7F3EB; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B4332 0%, #2D6A4F 60%, #40916C 100%);
    color: #D8F3DC;
  }
  [data-testid="stSidebar"] * { color: #D8F3DC !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label,
  [data-testid="stSidebar"] .stRadio label { color: #B7E4C7 !important; font-size: 0.82rem !important; }
  [data-testid="stSidebar"] hr { border-color: #52B788 !important; }
  [data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none; }

  /* ── Sidebar inputs ── */
  [data-testid="stSidebar"] .stSelectbox > div > div,
  [data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid #52B788 !important;
    border-radius: 6px !important;
    color: #fff !important;
  }
  [data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] { background: #95D5B2 !important; }
  [data-testid="stSidebar"] .stSlider [data-testid="stSliderTrackActive"] { background: #52B788 !important; }

  /* ── Analyse button ── */
  [data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #52B788, #2D6A4F) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.65rem 1rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25) !important;
    transition: opacity .2s !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover { opacity: 0.88 !important; }

  /* ── Cards ── */
  .ids-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(27,67,50,0.08);
    border-left: 4px solid #40916C;
    margin-bottom: 1rem;
  }
  .ids-card.attack { border-left-color: #C1121F; }
  .ids-card.normal { border-left-color: #40916C; }

  /* ── Result badge ── */
  .badge {
    display: inline-block;
    padding: .35rem 1rem;
    border-radius: 999px;
    font-size: .85rem;
    font-weight: 600;
    letter-spacing: .05em;
  }
  .badge-attack { background: #FFEEF0; color: #C1121F; border: 1px solid #F4A0A0; }
  .badge-normal { background: #D8F3DC; color: #1B4332; border: 1px solid #95D5B2; }

  /* ── Metric chips ── */
  .metric-row { display: flex; gap: .8rem; flex-wrap: wrap; margin-top: .6rem; }
  .metric-chip {
    background: #F0EDE5;
    border-radius: 8px;
    padding: .4rem .9rem;
    font-size: .82rem;
    font-weight: 500;
    color: #2D3A2E;
  }
  .metric-chip span { font-weight: 700; color: #2D6A4F; }

  /* ── Section headers ── */
  .section-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #6B7C6E;
    margin-bottom: .5rem;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.8rem !important; padding-bottom: 2rem !important; }
</style>
""",
    unsafe_allow_html=True,
)


# Artifact loader
@st.cache_resource(show_spinner=False)
def load_artifacts():
    base = os.path.dirname(__file__)
    model_dir = os.path.join(base, "models")

    def _load(name):
        with open(os.path.join(model_dir, name), "rb") as f:
            return pickle.load(f)

    scaler_app = _load("scaler_app.pkl")
    encoder = _load("label_encoder.pkl")
    sel_feats = _load("selected_features.pkl")
    bg_data = _load("background_data.pkl")

    model_names = [
        "LR_baseline",
        "LR_SMOTETomek",
        "LR_SMOTEENN",
        "LR_tuned",
        "GaussianNB",
        "MultinomialNB",
        "ComplementNB",
    ]
    loaded_models = {}
    for mn in model_names:
        path = os.path.join(model_dir, f"{mn}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded_models[mn] = pickle.load(f)

    return scaler_app, encoder, sel_feats, bg_data, loaded_models


# LIME helper
def run_lime(model, input_scaled, bg_data, feature_names, class_names, pred_class_idx):
    try:
        import lime.lime_tabular as lt
    except ImportError:
        return None, "lime not installed — run: pip install lime"

    explainer = lt.LimeTabularExplainer(
        training_data=bg_data.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    exp = explainer.explain_instance(
        input_scaled[0],
        model.predict_proba,
        num_features=10,
        top_labels=1,
        labels=[pred_class_idx],
    )
    fig = exp.as_pyplot_figure(label=pred_class_idx)
    fig.patch.set_facecolor("#FFFFFF")
    fig.axes[0].set_facecolor("#F7F3EB")
    fig.tight_layout()
    return fig, None


# SHAP helper
def run_shap(model, input_scaled, bg_data, feature_names, model_name, pred_class_idx):
    try:
        import shap
    except ImportError:
        return None, "shap not installed — run: pip install shap"

    model_type = model_name.split("_")[0]
    try:
        if model_type == "LR":
            explainer = shap.LinearExplainer(
                model, bg_data.values, feature_perturbation="interventional"
            )
            shap_vals = explainer.shap_values(input_scaled)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba, shap.sample(bg_data.values, 50)
            )
            shap_vals = explainer.shap_values(input_scaled, nsamples=100)

        # Normalise to 1-D array of shape (n_features,) for the predicted class
        if isinstance(shap_vals, list):
            # list of n_classes arrays, each (n_samples, n_features)
            vals = np.array(shap_vals[pred_class_idx]).flatten()
        elif shap_vals.ndim == 3:
            # (n_samples, n_features, n_classes)
            vals = shap_vals[0, :, pred_class_idx]
        else:
            # (n_samples, n_features) — binary / single output
            vals = shap_vals[0]
    except Exception as e:
        return None, f"SHAP error: {e}"

    vals = np.array(vals).flatten()
    sorted_idx = np.argsort(np.abs(vals))

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7F3EB")

    colors = ["#C1121F" if v < 0 else "#2D6A4F" for v in vals[sorted_idx]]
    ax.barh(
        [feature_names[i].strip() for i in sorted_idx],
        vals[sorted_idx],
        color=colors,
        height=0.6,
    )
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title("Feature Contributions (SHAP)", fontsize=10, pad=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig, None


# Confidence bar chart
def conf_bar_chart(proba, class_names, pred_idx):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F7F3EB")

    colors = [
        "#2D6A4F" if i == pred_idx else "#95D5B2" for i in range(len(class_names))
    ]
    bars = ax.barh(class_names, proba * 100, color=colors, height=0.55)
    for bar, p in zip(bars, proba):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{p*100:.1f}%",
            va="center",
            fontsize=8,
            color="#2D3A2E",
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


#  MAIN
# Header
st.markdown(
    """
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.2rem;">
  <div style="font-size:2rem">🛡️</div>
  <div>
    <div style="font-size:1.4rem;font-weight:700;color:#1B4332;line-height:1.2">
      Intrusion Detection System
    </div>
    <div style="font-size:0.82rem;color:#6B7C6E;font-weight:400">
      Wireless Sensor Network · Real-time threat classification
    </div>
  </div>
</div>
<hr style="border:none;border-top:1px solid #D5CFC4;margin-bottom:1.4rem;">
""",
    unsafe_allow_html=True,
)

# Load artifacts
models_dir_exists = os.path.isdir(os.path.join(os.path.dirname(__file__), "models"))
if not models_dir_exists:
    st.error(
        "⚠️ `models/` directory not found. Run all cells in **ids-in-wsn.ipynb** first to generate pkl files."
    )
    st.stop()

try:
    scaler_app, encoder, sel_feats, bg_data, loaded_models = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"⚠️ Missing artifact: {e}\n\nRun the pkl export cell in the notebook first."
    )
    st.stop()

class_names = encoder.classes_.tolist()
feat_display = {
    " Time": "Time (s)",
    " Is_CH": "Is Cluster Head",
    " Dist_To_CH": "Distance to CH",
    " JOIN_S": "JOIN Sent",
    " SCH_R": "SCH Received",
    "Rank": "Rank",
    " DATA_S": "DATA Sent",
    " DATA_R": "DATA Received",
    " dist_CH_To_BS": "CH Distance to BS",
    " send_code ": "Send Code",
}
feat_ranges = {
    " Time": (50, 3600, 800, 1),
    " Is_CH": (0, 1, 0, 1),
    " Dist_To_CH": (0.0, 214.0, 18.4, 0.01),
    " JOIN_S": (0, 1, 1, 1),
    " SCH_R": (0, 1, 1, 1),
    "Rank": (0, 99, 3, 1),
    " DATA_S": (0, 241, 35, 1),
    " DATA_R": (0, 1496, 0, 1),
    " dist_CH_To_BS": (0.0, 202.0, 0.0, 0.01),
    " send_code ": (0, 15, 2, 1),
}

# Sidebar
with st.sidebar:
    st.markdown(
        """
    <div style="text-align:center;padding:.6rem 0 1rem;">
      <div style="font-size:1.8rem">🛡️</div>
      <div style="font-size:1rem;font-weight:700;letter-spacing:.06em;">IDS · WSN</div>
      <div style="font-size:.72rem;opacity:.7;margin-top:.2rem;">Threat Analyser</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.divider()

    selected_model = st.selectbox(
        "Model",
        options=list(loaded_models.keys()),
        index=(
            list(loaded_models.keys()).index("LR_tuned")
            if "LR_tuned" in loaded_models
            else 0
        ),
    )

    xai_method = st.radio(
        "Explainability",
        options=["LIME", "SHAP"],
        horizontal=True,
    )

    st.divider()
    st.markdown(
        '<div style="font-size:.72rem;opacity:.7;letter-spacing:.1em;text-transform:uppercase;">Node Parameters</div>',
        unsafe_allow_html=True,
    )

    feat_help = {
        " Time":
            "Time elapsed in the sensor network round (seconds). "
            "Higher values = later in the network lifecycle.",
        " Is_CH":
            "Is this node acting as a Cluster Head? "
            "A Cluster Head collects data from nearby nodes and forwards it to the base station. (1 = Yes, 0 = No)",
        " Dist_To_CH":
            "How far this node is from its Cluster Head, in meters. "
            "Nodes far from their CH spend more energy to communicate.",
        " JOIN_S":
            "Did this node send a JOIN request to join a cluster this round? "
            "Normal nodes send exactly one. Flooding attackers may send many. (1 = Yes, 0 = No)",
        " SCH_R":
            "Did this node receive a time-slot schedule from its Cluster Head? "
            "Without a schedule, the node cannot send data in an orderly way. (1 = Yes, 0 = No)",
        "Rank":
            "The node's priority rank inside its cluster (0 = highest priority, 99 = lowest). "
            "An unusually high rank may indicate a large or abnormal cluster.",
        " DATA_S":
            "Total number of data packets this node has sent this round. "
            "Attackers often send far more packets than normal nodes.",
        " DATA_R":
            "Total data packets this node has received. "
            "A very high count can signal a flooding or blackhole attack nearby.",
        " dist_CH_To_BS":
            "Distance from this node's Cluster Head to the central Base Station (meters). "
            "Longer distances require more energy per transmission round.",
        " send_code ":
            "Protocol message type being sent (0–15). "
            "Different codes represent JOIN, scheduling, data, and other control messages.",
    }

    inputs = {}
    for feat in sel_feats:
        lo, hi, default, step = feat_ranges[feat]
        label = feat_display.get(feat, feat.strip())
        help_txt = feat_help.get(feat, "")
        if hi == 1 and lo == 0 and isinstance(step, int):
            inputs[feat] = st.selectbox(label, [0, 1], index=int(default), help=help_txt)
        elif isinstance(step, float):
            inputs[feat] = st.number_input(
                label,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=step,
                help=help_txt,
            )
        else:
            inputs[feat] = st.slider(
                label, int(lo), int(hi), int(default), step=int(step), help=help_txt
            )

    st.divider()
    analyse_btn = st.button("🔍  Analyse Threat", width="stretch")


# Main content
if not analyse_btn:
    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
        st.markdown(
            """
        <div class="ids-card normal" style="color:#1C1C1C;">
          <b style="color:#1B4332;">How to use</b><br>
          <ol style="margin:.6rem 0 0 1rem;padding:0;font-size:.88rem;line-height:1.8;color:#1C1C1C;">
            <li>Select a model from the sidebar</li>
            <li>Choose LIME or SHAP explainability</li>
            <li>Set node parameters (feature values)</li>
            <li>Click <b>Analyse Threat</b></li>
          </ol>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            '<div class="section-title">Models Available</div>', unsafe_allow_html=True
        )
        model_info = {
            "LR_tuned": ("Logistic Regression", "GridSearch-tuned · Best overall"),
            "GaussianNB": ("Gaussian Naive Bayes", "Probabilistic · Fast"),
            "LR_baseline": ("Logistic Regression", "SMOTE baseline"),
            "LR_SMOTETomek": ("Logistic Regression", "SMOTE-Tomek balanced"),
            "LR_SMOTEENN": ("Logistic Regression", "SMOTE-ENN balanced"),
            "MultinomialNB": ("Multinomial NB", "Count-based features"),
            "ComplementNB": ("Complement NB", "Imbalance-robust"),
        }
        for mn in loaded_models:
            algo, note = model_info.get(mn, (mn, ""))
            active = "✦ " if mn == selected_model else "  "
            st.markdown(
                f'<div style="font-size:.82rem;padding:.25rem .1rem;color:#2D3A2E;">'
                f'{active}<b>{mn}</b> · <span style="color:#6B7C6E">{note}</span></div>',
                unsafe_allow_html=True,
            )
    st.stop()


# Run prediction
model = loaded_models[selected_model]
input_df = pd.DataFrame([inputs], columns=sel_feats)
input_scaled = scaler_app.transform(input_df)

y_pred_label = model.predict(input_scaled)[0]          # string label directly
proba        = model.predict_proba(input_scaled)[0]
pred_class_idx = list(model.classes_).index(y_pred_label)
confidence   = proba[pred_class_idx] * 100
is_attack    = y_pred_label != "Normal"

card_cls = "attack" if is_attack else "normal"
badge_cls = "badge-attack" if is_attack else "badge-normal"
badge_txt = "⚠ ATTACK DETECTED" if is_attack else "✔ NORMAL TRAFFIC"

# Result + confidence
col_res, col_conf = st.columns([1.1, 1], gap="large")

with col_res:
    st.markdown(
        '<div class="section-title">Detection Result</div>', unsafe_allow_html=True
    )
    st.markdown(
        f"""
    <div class="ids-card {card_cls}">
      <span class="badge {badge_cls}">{badge_txt}</span>
      <div style="margin-top:.9rem;font-size:1.65rem;font-weight:700;color:#1B4332;">
        {y_pred_label}
      </div>
      <div class="metric-row">
        <div class="metric-chip">Model <span>{selected_model}</span></div>
        <div class="metric-chip">Confidence <span>{confidence:.1f}%</span></div>
        <div class="metric-chip">XAI <span>{xai_method}</span></div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_conf:
    st.markdown(
        '<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True
    )
    fig_conf = conf_bar_chart(proba, class_names, pred_class_idx)
    st.pyplot(fig_conf, width="stretch")
    plt.close(fig_conf)

st.divider()

# Explainability
st.markdown(
    f'<div class="section-title">{xai_method} Explanation — "{y_pred_label}"</div>',
    unsafe_allow_html=True,
)

with st.spinner(f"Computing {xai_method} explanation…"):
    if xai_method == "LIME":
        xai_fig, xai_err = run_lime(
            model,
            input_scaled,
            bg_data,
            [f.strip() for f in sel_feats],
            class_names,
            int(pred_class_idx),
        )
    else:
        xai_fig, xai_err = run_shap(
            model,
            input_scaled,
            bg_data,
            sel_feats,
            selected_model,
            pred_class_idx,
        )

if xai_err:
    st.warning(f"⚠ {xai_err}")
elif xai_fig:
    col_xai, col_spacer = st.columns([1.4, 0.6])
    with col_xai:
        st.markdown('<div class="ids-card">', unsafe_allow_html=True)
        st.pyplot(xai_fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
        plt.close(xai_fig)
    with col_spacer:
        st.markdown(
            f"""
        <div class="ids-card" style="height:100%;">
          <div style="font-size:.8rem;color:#6B7C6E;line-height:1.7;">
            <b>Reading the chart</b><br>
            <span style="color:#2D6A4F;font-weight:600;">■ Green bars</span> push toward
            <em>{y_pred_label}</em>.<br>
            <span style="color:#C1121F;font-weight:600;">■ Red bars</span> push away.<br><br>
            Longer bar = stronger influence on this prediction.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Input summary
with st.expander("📋 Input Parameters", expanded=False):
    rows = []
    for feat in sel_feats:
        rows.append(
            {
                "Feature": feat_display.get(feat, feat.strip()),
                "Raw Value": inputs[feat],
                "Scaled Value": round(float(input_scaled[0][sel_feats.index(feat)]), 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
