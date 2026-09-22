# ============================================================
# app.py
# Final R-trained XGBoost -> pure Python Streamlit deployment
# Target displayed: POOR WOUND HEALING
# Stable static red/blue SHAP force plot (no JS/HTML rendering)
# ============================================================

from pathlib import Path
import io
import json
import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import streamlit as st
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "preprocess_config.json"
MODEL_FILE = BASE_DIR / "xgb_final_model.json"

st.set_page_config(
    page_title="Poor Wound Healing Prediction",
    page_icon="🏥",
    layout="wide",
)
st.title("🏥 Poor Wound Healing Prediction")
st.caption(
    "Enter a new patient's clinical characteristics to estimate the probability "
    "of poor wound healing and obtain an individualized SHAP explanation."
)

@st.cache_resource
def load_assets():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError("preprocess_config.json is missing.")
    if not MODEL_FILE.exists():
        raise FileNotFoundError("xgb_final_model.json is missing.")

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(str(MODEL_FILE))
    return config, booster

try:
    CONFIG, BOOSTER = load_assets()
except Exception as exc:
    st.error(f"Deployment files could not be loaded: {exc}")
    st.stop()

PREDICTORS = CONFIG["predictors"]
MODEL_FEATURES = CONFIG["model_features"]
BASE_VECTOR = np.asarray(CONFIG["baseline_vector"], dtype=float)

# Backward-compatible with the previous exporter.
if "threshold_poor" in CONFIG:
    THRESHOLD_POOR = float(CONFIG["threshold_poor"])
else:
    THRESHOLD_POOR = 1.0 - float(CONFIG["threshold"])

if BASE_VECTOR.shape[0] != len(MODEL_FEATURES):
    st.error("preprocess_config.json has inconsistent model dimensions.")
    st.stop()


def clean_label(name):
    return (
        str(name)
        .replace("_pct", " (%)")
        .replace("_num", " (#)")
        .replace("_", " ")
    )


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def preprocess_new_patient(raw_values):
    """Reproduce the exact preprocessing exported from the R pipeline."""
    vec = BASE_VECTOR.copy()

    for variable in PREDICTORS:
        kind = CONFIG["predictor_class"][variable]

        if kind == "numeric":
            value = float(raw_values[variable])

            if variable in CONFIG["cap_lower"]:
                value = max(value, float(CONFIG["cap_lower"][variable]))
            if variable in CONFIG["cap_upper"]:
                value = min(value, float(CONFIG["cap_upper"][variable]))

            if variable in CONFIG["scale_means"]:
                mean = float(CONFIG["scale_means"][variable])
                sd = float(CONFIG["scale_sds"][variable])
                value = (value - mean) / sd

            delta = np.asarray(CONFIG["numeric_deltas"][variable], dtype=float)
            vec += value * delta

        else:
            level = str(raw_values[variable])
            level_map = CONFIG["categorical_deltas"][variable]
            if level not in level_map:
                raise ValueError(f"Invalid category for {variable}: {level}")
            vec += np.asarray(level_map[level], dtype=float)

    if not np.all(np.isfinite(vec)):
        raise ValueError("Preprocessing generated non-finite values.")

    return vec


def predict_and_explain(raw_values):
    x_vector = preprocess_new_patient(raw_values)
    x_matrix = x_vector.reshape(1, -1)

    dnew = xgb.DMatrix(x_matrix, feature_names=MODEL_FEATURES)

    # The final trained model predicts P(healing=yes).
    probability_yes = float(BOOSTER.predict(dnew)[0])
    probability_poor = 1.0 - probability_yes

    # Exact XGBoost TreeSHAP contributions for the original yes-margin.
    contribution = BOOSTER.predict(
        dnew,
        pred_contribs=True,
        approx_contribs=False,
    )[0]

    shap_yes = np.asarray(contribution[:-1], dtype=float)
    base_yes = float(contribution[-1])

    # For the complementary poor-healing event:
    # margin_poor = -margin_yes, so SHAP and base value are negated.
    shap_poor = -shap_yes
    base_poor = -base_yes

    grouped = {}
    mapping = CONFIG["model_to_original"]

    for model_feature, shap_value in zip(MODEL_FEATURES, shap_poor):
        group = mapping.get(model_feature, model_feature)
        grouped[group] = grouped.get(group, 0.0) + float(shap_value)

    groups = [v for v in PREDICTORS if v in grouped]
    groups.extend([v for v in grouped if v not in groups])
    grouped_values = np.asarray([grouped[v] for v in groups], dtype=float)

    reconstructed_margin = base_poor + float(grouped_values.sum())
    reconstructed_probability = sigmoid(reconstructed_margin)

    if abs(reconstructed_probability - probability_poor) > 1e-5:
        raise ValueError("SHAP additivity check failed for poor wound healing.")

    return {
        "probability_poor": probability_poor,
        "probability_yes": probability_yes,
        "base_poor": base_poor,
        "groups": groups,
        "grouped_shap": grouped_values,
    }


def make_force_plot(result, raw_values):
    """
    Stable static SHAP force plot.
    Red  = increases poor wound-healing risk.
    Blue = decreases poor wound-healing risk.
    No JavaScript/HTML is used, so Streamlit Cloud rendering is stable.
    """
    names = result["groups"]
    values = np.asarray(result["grouped_shap"], dtype=float)

    order = np.argsort(np.abs(values))[::-1]
    n_show = min(8, len(order))
    keep = order[:n_show]
    rest = order[n_show:]

    shown_names = [clean_label(names[i]) for i in keep]
    shown_values = [float(values[i]) for i in keep]
    shown_raw = [str(raw_values.get(names[i], "")) for i in keep]

    if len(rest) > 0:
        other = float(values[rest].sum())
        if abs(other) > 1e-12:
            shown_names.append("Other features")
            shown_values.append(other)
            shown_raw.append("")

    # Put largest forces first in the cumulative sequence.
    local_order = np.argsort(np.abs(np.asarray(shown_values)))[::-1]
    shown_names = [shown_names[i] for i in local_order]
    shown_values = [shown_values[i] for i in local_order]
    shown_raw = [shown_raw[i] for i in local_order]

    base = float(result["base_poor"])
    final_margin = base + sum(shown_values)

    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    y = 0.52
    current = base
    positions = [base]

    for feature, raw_value, shap_value in zip(
        shown_names, shown_raw, shown_values
    ):
        next_value = current + shap_value
        color = "#ff0051" if shap_value > 0 else "#008bfb"

        arrow = FancyArrowPatch(
            (current, y),
            (next_value, y),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=7,
            color=color,
            alpha=0.90,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

        midpoint = (current + next_value) / 2.0
        label = feature
        if raw_value and feature != "Other features":
            label = f"{feature}={raw_value}"

        ax.text(
            midpoint,
            y + 0.12,
            label,
            ha="center",
            va="bottom",
            fontsize=8.3,
        )

        current = next_value
        positions.append(current)

    ax.axvline(base, color="#666666", linestyle="--", linewidth=1.1)
    ax.axvline(final_margin, color="#111111", linewidth=1.3)

    ax.text(
        base,
        0.22,
        f"Base value\n{sigmoid(base):.1%}",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.text(
        final_margin,
        0.22,
        f"Prediction\n{result['probability_poor']:.1%}",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    xmin, xmax = min(positions), max(positions)
    span = max(xmax - xmin, 0.5)
    ax.set_xlim(xmin - 0.15 * span, xmax + 0.15 * span)
    ax.set_ylim(0.05, 1.00)
    ax.set_yticks([])

    ax.set_xlabel(
        "SHAP contribution to poor wound-healing risk (XGBoost log-odds scale)"
    )
    ax.set_title("Individual SHAP Force Plot", fontsize=13, fontweight="bold", pad=15)

    ax.text(
        0.01,
        0.98,
        "Red = increases poor wound-healing risk",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#ff0051",
        fontweight="bold",
    )
    ax.text(
        0.99,
        0.98,
        "Blue = decreases poor wound-healing risk",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#008bfb",
        fontweight="bold",
    )

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_alpha(0.35)

    fig.tight_layout()
    return fig


# ============================================================
# New-patient form
# ============================================================

with st.form("new_patient_form"):
    st.subheader("New Patient")
    left, right = st.columns(2)
    raw_values = {}

    for i, variable in enumerate(PREDICTORS):
        meta = CONFIG["ui_metadata"][variable]
        target_col = left if i % 2 == 0 else right

        with target_col:
            if meta["type"] == "categorical":
                levels = [str(x) for x in meta["levels"]]
                default = str(meta.get("default", levels[0]))
                index = levels.index(default) if default in levels else 0
                raw_values[variable] = st.selectbox(
                    clean_label(variable),
                    options=levels,
                    index=index,
                )
            else:
                raw_values[variable] = st.number_input(
                    clean_label(variable),
                    value=float(meta["default"]),
                    step=float(meta["step"]),
                )
                st.caption(
                    f"Training range: {float(meta['observed_min']):.2f} – "
                    f"{float(meta['observed_max']):.2f}"
                )

    submitted = st.form_submit_button(
        "Predict Risk",
        type="primary",
        width="stretch",
    )


# ============================================================
# Result
# ============================================================

if submitted:
    try:
        result = predict_and_explain(raw_values)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        probability = float(result["probability_poor"])
        st.subheader("Prediction Result")
        st.metric("Probability of poor wound healing", f"{probability:.1%}")
        st.caption(f"Fixed XGBoost threshold for poor wound healing: {THRESHOLD_POOR:.3f}")

        if probability >= THRESHOLD_POOR:
            st.error("Predicted outcome: **Poor wound healing**")
        else:
            st.success("Predicted outcome: **Wound healing**")

    with col2:
        st.subheader("Individualized SHAP Force Plot")
        fig = make_force_plot(result, raw_values)
        st.pyplot(fig, width="stretch", clear_figure=False)
        st.caption(
            "Red features increase the predicted risk of poor wound healing; "
            "blue features decrease the predicted risk."
        )

        png = io.BytesIO()
        fig.savefig(
            png,
            format="png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        png.seek(0)
        st.download_button(
            "Download force plot (600 dpi PNG)",
            data=png,
            file_name="Poor_Wound_Healing_SHAP_ForcePlot.png",
            mime="image/png",
        )
else:
    st.info("Enter the patient's values and click Predict Risk.")
