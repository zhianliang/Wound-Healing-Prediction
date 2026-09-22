# ============================================================
# app.py
#
# Final R-trained XGBoost -> pure Python Streamlit deployment
#
# Target shown by website:
#   POOR WOUND HEALING
#
# IMPORTANT:
# Original final XGBoost predicts:
#   P(Wound_Healing = yes) = favorable wound healing
#
# Therefore website uses:
#   P(poor wound healing) = 1 - P(favorable wound healing)
#
# SHAP direction MUST also be reversed:
#   SHAP_poor = -SHAP_favorable
#   base_poor = -base_favorable
#
# Red  = increases poor wound-healing probability
# Blue = decreases poor wound-healing probability
#
# Required files:
#   app.py
#   xgb_final_model.json
#   preprocess_config.json
# ============================================================

from pathlib import Path
import json
import math
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import streamlit as st
import xgboost as xgb

warnings.filterwarnings("ignore")


# ============================================================
# 1. Files
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "preprocess_config.json"
MODEL_FILE = BASE_DIR / "xgb_final_model.json"


# ============================================================
# 2. Page
# ============================================================

st.set_page_config(
    page_title="Poor Wound Healing Prediction",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Poor Wound Healing Prediction")
st.caption(
    "Enter a new patient's clinical characteristics to estimate "
    "the probability of poor wound healing."
)


# ============================================================
# 3. Load final model and preprocessing
# ============================================================

@st.cache_resource
def load_assets():

    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            "preprocess_config.json is missing."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "xgb_final_model.json is missing."
        )

    config = json.loads(
        CONFIG_FILE.read_text(
            encoding="utf-8"
        )
    )

    booster = xgb.Booster()
    booster.load_model(
        str(MODEL_FILE)
    )

    return config, booster


try:
    CONFIG, BOOSTER = load_assets()
except Exception as exc:
    st.error(
        f"Model files could not be loaded: {exc}"
    )
    st.stop()


PREDICTORS = CONFIG["predictors"]
MODEL_FEATURES = CONFIG["model_features"]

BASE_VECTOR = np.asarray(
    CONFIG["baseline_vector"],
    dtype=float,
)

if len(BASE_VECTOR) != len(MODEL_FEATURES):
    st.error(
        "The preprocessing configuration has inconsistent dimensions."
    )
    st.stop()


# ============================================================
# 4. Helpers
# ============================================================

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


def format_value(value):

    try:
        f = float(value)

        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))

        return f"{f:.2f}".rstrip("0").rstrip(".")

    except Exception:
        return str(value)


# ============================================================
# 5. Exact preprocessing exported from R
# ============================================================

def preprocess_new_patient(raw_values):

    vec = BASE_VECTOR.copy()

    for variable in PREDICTORS:

        kind = CONFIG[
            "predictor_class"
        ][variable]

        if kind == "numeric":

            value = float(
                raw_values[variable]
            )

            # Training-derived capping
            if variable in CONFIG.get(
                "cap_lower",
                {}
            ):
                value = max(
                    value,
                    float(
                        CONFIG[
                            "cap_lower"
                        ][variable]
                    )
                )

            if variable in CONFIG.get(
                "cap_upper",
                {}
            ):
                value = min(
                    value,
                    float(
                        CONFIG[
                            "cap_upper"
                        ][variable]
                    )
                )

            # Training-derived standardization
            if variable in CONFIG.get(
                "scale_means",
                {}
            ):
                mean = float(
                    CONFIG[
                        "scale_means"
                    ][variable]
                )

                sd = float(
                    CONFIG[
                        "scale_sds"
                    ][variable]
                )

                value = (
                    value - mean
                ) / sd

            delta = np.asarray(
                CONFIG[
                    "numeric_deltas"
                ][variable],
                dtype=float,
            )

            vec += value * delta

        else:

            level = str(
                raw_values[variable]
            )

            level_map = CONFIG[
                "categorical_deltas"
            ][variable]

            if level not in level_map:
                raise ValueError(
                    f"Invalid category for {variable}: {level}"
                )

            vec += np.asarray(
                level_map[level],
                dtype=float,
            )

    if not np.all(np.isfinite(vec)):
        raise ValueError(
            "Preprocessing generated a non-finite model value."
        )

    return vec


# ============================================================
# 6. Prediction + TreeSHAP converted to POOR wound healing
# ============================================================

def predict_and_explain(raw_values):

    model_vector = preprocess_new_patient(
        raw_values
    )

    matrix = model_vector.reshape(
        1,
        -1,
    )

    dnew = xgb.DMatrix(
        matrix,
        feature_names=MODEL_FEATURES,
    )

    # --------------------------------------------------------
    # Original final XGBoost:
    # probability of favorable wound healing (yes)
    # --------------------------------------------------------

    probability_favorable = float(
        BOOSTER.predict(
            dnew
        )[0]
    )

    # --------------------------------------------------------
    # Website target:
    # probability of POOR wound healing
    # --------------------------------------------------------

    probability_poor = (
        1.0 - probability_favorable
    )

    # --------------------------------------------------------
    # Native TreeSHAP from XGBoost is for favorable=yes margin
    # --------------------------------------------------------

    contribution = BOOSTER.predict(
        dnew,
        pred_contribs=True,
        approx_contribs=False,
    )[0]

    shap_favorable = np.asarray(
        contribution[:-1],
        dtype=float,
    )

    base_favorable = float(
        contribution[-1]
    )

    # --------------------------------------------------------
    # CRITICAL DIRECTION CONVERSION
    #
    # p_favorable = sigmoid(margin)
    # p_poor      = 1 - sigmoid(margin)
    #             = sigmoid(-margin)
    #
    # So every component of the additive explanation must flip:
    #
    # margin_poor = -margin_favorable
    # base_poor   = -base_favorable
    # SHAP_poor   = -SHAP_favorable
    # --------------------------------------------------------

    shap_poor = -shap_favorable
    base_poor = -base_favorable

    # --------------------------------------------------------
    # Group encoded columns back to original clinical variables
    # --------------------------------------------------------

    grouped = {}

    mapping = CONFIG[
        "model_to_original"
    ]

    for model_feature, shap_value in zip(
        MODEL_FEATURES,
        shap_poor,
    ):

        original = mapping.get(
            model_feature,
            model_feature,
        )

        grouped[original] = (
            grouped.get(
                original,
                0.0,
            )
            + float(shap_value)
        )

    groups = [
        variable
        for variable in PREDICTORS
        if variable in grouped
    ]

    extras = [
        variable
        for variable in grouped
        if variable not in groups
    ]

    groups.extend(extras)

    grouped_shap = np.asarray(
        [
            grouped[variable]
            for variable in groups
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Additivity check on the POOR-wound-healing scale
    # --------------------------------------------------------

    margin_poor = (
        base_poor
        + float(
            grouped_shap.sum()
        )
    )

    reconstructed_probability_poor = sigmoid(
        margin_poor
    )

    if abs(
        reconstructed_probability_poor
        - probability_poor
    ) > 1e-5:

        raise ValueError(
            "Poor-wound-healing SHAP conversion failed additivity check."
        )

    return {
        "probability_poor": probability_poor,
        "base_poor": base_poor,
        "groups": groups,
        "grouped_shap": grouped_shap,
    }


# ============================================================
# 7. Stable classic SHAP-style force plot
#
# IMPORTANT DIRECTION:
#   Positive SHAP -> RED  -> increases poor wound-healing risk
#   Negative SHAP -> BLUE -> decreases poor wound-healing risk
# ============================================================

def make_classic_force_plot(
    result,
    raw_values,
):

    groups = result["groups"]

    shap_values = np.asarray(
        result["grouped_shap"],
        dtype=float,
    )

    order = np.argsort(
        np.abs(shap_values)
    )[::-1]

    max_display = min(
        8,
        len(order)
    )

    keep = order[:max_display]

    display_names = []
    display_values = []
    display_raw = []

    for i in keep:

        display_names.append(
            clean_label(
                groups[i]
            )
        )

        display_values.append(
            float(
                shap_values[i]
            )
        )

        display_raw.append(
            format_value(
                raw_values.get(
                    groups[i],
                    ""
                )
            )
        )

    if len(order) > max_display:

        other_value = float(
            shap_values[
                order[max_display:]
            ].sum()
        )

        if abs(other_value) > 1e-12:

            display_names.append(
                "Other features"
            )

            display_values.append(
                other_value
            )

            display_raw.append(
                ""
            )

    positive_items = []
    negative_items = []

    for name, raw, value in zip(
        display_names,
        display_raw,
        display_values,
    ):

        item = {
            "name": name,
            "raw": raw,
            "value": value,
        }

        # Positive poor-wound-healing SHAP = RED
        if value >= 0:
            positive_items.append(
                item
            )

        # Negative poor-wound-healing SHAP = BLUE
        else:
            negative_items.append(
                item
            )

    positive_items.sort(
        key=lambda x: abs(
            x["value"]
        )
    )

    negative_items.sort(
        key=lambda x: abs(
            x["value"]
        )
    )

    base_margin = float(
        result[
            "base_poor"
        ]
    )

    final_margin = (
        base_margin
        + float(
            np.sum(
                shap_values
            )
        )
    )

    base_prob = sigmoid(
        base_margin
    )

    final_prob = float(
        result[
            "probability_poor"
        ]
    )

    # --------------------------------------------------------
    # Build cumulative probability-space segments
    # --------------------------------------------------------

    pos_segments = []
    neg_segments = []

    current = final_margin

    for item in positive_items:

        previous = (
            current
            - item["value"]
        )

        pos_segments.append(
            {
                **item,
                "x0": sigmoid(
                    previous
                ),
                "x1": sigmoid(
                    current
                ),
            }
        )

        current = previous

    current = final_margin

    for item in negative_items:

        previous = (
            current
            - item["value"]
        )

        neg_segments.append(
            {
                **item,
                "x0": sigmoid(
                    current
                ),
                "x1": sigmoid(
                    previous
                ),
            }
        )

        current = previous

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            18,
            3.2
        )
    )

    fig.patch.set_facecolor(
        "white"
    )

    ax.set_facecolor(
        "white"
    )

    y_top = 0.62
    y_bottom = 0.42
    notch = 0.010

    RED = "#ff0051"
    BLUE = "#1e88e5"

    # --------------------------------------------------------
    # RED: increases POOR wound-healing probability
    # --------------------------------------------------------

    for k, item in enumerate(
        pos_segments
    ):

        left = min(
            item["x0"],
            item["x1"]
        )

        right = max(
            item["x0"],
            item["x1"]
        )

        if right - left < 1e-6:
            continue

        local_notch = min(
            notch,
            max(
                (right - left) * 0.25,
                0.001
            )
        )

        polygon = Polygon(
            [
                (left, y_bottom),
                (right - local_notch, y_bottom),
                (
                    right,
                    (y_bottom + y_top) / 2
                ),
                (right - local_notch, y_top),
                (left, y_top),
                (
                    left + local_notch,
                    (y_bottom + y_top) / 2
                ),
            ],
            closed=True,
            facecolor=RED,
            edgecolor=RED,
            linewidth=0,
            alpha=0.98,
        )

        ax.add_patch(
            polygon
        )

        label = item["name"]

        if (
            item["raw"] != ""
            and item["name"] != "Other features"
        ):
            label += (
                " = "
                + item["raw"]
            )

        midpoint = (
            left + right
        ) / 2

        ax.text(
            midpoint,
            y_bottom - 0.055 - 0.055 * (k % 2),
            label,
            color=RED,
            fontsize=10,
            ha="center",
            va="top",
        )

        ax.plot(
            [
                midpoint,
                midpoint,
            ],
            [
                y_bottom,
                y_bottom - 0.035,
            ],
            color=RED,
            linewidth=0.8,
            alpha=0.5,
        )

    # --------------------------------------------------------
    # BLUE: decreases POOR wound-healing probability
    # --------------------------------------------------------

    for k, item in enumerate(
        neg_segments
    ):

        left = min(
            item["x0"],
            item["x1"]
        )

        right = max(
            item["x0"],
            item["x1"]
        )

        if right - left < 1e-6:
            continue

        local_notch = min(
            notch,
            max(
                (right - left) * 0.25,
                0.001
            )
        )

        polygon = Polygon(
            [
                (left + local_notch, y_bottom),
                (right, y_bottom),
                (
                    right - local_notch,
                    (y_bottom + y_top) / 2
                ),
                (right, y_top),
                (left + local_notch, y_top),
                (
                    left,
                    (y_bottom + y_top) / 2
                ),
            ],
            closed=True,
            facecolor=BLUE,
            edgecolor=BLUE,
            linewidth=0,
            alpha=0.98,
        )

        ax.add_patch(
            polygon
        )

        label = item["name"]

        if (
            item["raw"] != ""
            and item["name"] != "Other features"
        ):
            label += (
                " = "
                + item["raw"]
            )

        midpoint = (
            left + right
        ) / 2

        ax.text(
            midpoint,
            y_bottom - 0.055 - 0.055 * (k % 2),
            label,
            color=BLUE,
            fontsize=10,
            ha="center",
            va="top",
        )

        ax.plot(
            [
                midpoint,
                midpoint,
            ],
            [
                y_bottom,
                y_bottom - 0.035,
            ],
            color=BLUE,
            linewidth=0.8,
            alpha=0.5,
        )

    # --------------------------------------------------------
    # Classic SHAP top line
    # --------------------------------------------------------

    ax.axhline(
        y=y_top + 0.055,
        color="#888888",
        linewidth=0.8,
    )

    # f(x) = current poor wound-healing probability
    ax.plot(
        [
            final_prob,
            final_prob,
        ],
        [
            y_top + 0.02,
            y_top + 0.09,
        ],
        color="#555555",
        linewidth=1.0,
    )

    ax.text(
        final_prob,
        y_top + 0.11,
        "f(x)",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#777777",
    )

    ax.text(
        final_prob,
        y_top + 0.072,
        f"{final_prob:.3f}",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="black",
    )

    # Base value on poor-wound-healing probability scale
    ax.plot(
        [
            base_prob,
            base_prob,
        ],
        [
            y_top + 0.02,
            y_top + 0.09,
        ],
        color="#888888",
        linewidth=0.9,
        alpha=0.8,
    )

    ax.text(
        base_prob,
        y_top + 0.11,
        "base value",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#777777",
    )

    ax.text(
        base_prob,
        y_top + 0.072,
        f"{base_prob:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#777777",
    )

    # Explicit meaning:
    # higher = higher POOR wound-healing probability
    # lower  = lower POOR wound-healing probability
    center = (
        final_prob + base_prob
    ) / 2

    ax.text(
        center - 0.025,
        0.955,
        "higher",
        transform=ax.get_xaxis_transform(),
        color=RED,
        fontsize=11,
        ha="right",
        va="top",
    )

    ax.text(
        center,
        0.955,
        "↔",
        transform=ax.get_xaxis_transform(),
        color="#777777",
        fontsize=11,
        ha="center",
        va="top",
    )

    ax.text(
        center + 0.025,
        0.955,
        "lower",
        transform=ax.get_xaxis_transform(),
        color=BLUE,
        fontsize=11,
        ha="left",
        va="top",
    )

    # --------------------------------------------------------
    # Probability axis
    # --------------------------------------------------------

    all_x = [
        final_prob,
        base_prob,
    ]

    for item in (
        pos_segments
        + neg_segments
    ):
        all_x.extend(
            [
                item["x0"],
                item["x1"],
            ]
        )

    xmin = max(
        0.0,
        min(all_x) - 0.08
    )

    xmax = min(
        1.0,
        max(all_x) + 0.08
    )

    if xmax - xmin < 0.35:

        mid = (
            xmin + xmax
        ) / 2

        xmin = max(
            0.0,
            mid - 0.175
        )

        xmax = min(
            1.0,
            mid + 0.175
        )

    ticks = np.linspace(
        xmin,
        xmax,
        9
    )

    ax.set_xticks(
        ticks
    )

    ax.set_xticklabels(
        [
            f"{x:.3f}".rstrip("0").rstrip(".")
            for x in ticks
        ],
        fontsize=9,
        color="#777777",
    )

    ax.tick_params(
        axis="x",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=3,
        color="#999999",
        pad=2,
    )

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_ylim(
        0.02,
        1.0
    )

    ax.set_yticks(
        []
    )

    for spine in ax.spines.values():
        spine.set_visible(
            False
        )

    fig.tight_layout(
        pad=0.6
    )

    return fig


# ============================================================
# 8. New-patient input
# ============================================================

st.sidebar.header(
    "📊 Patient Clinical Parameters"
)

st.sidebar.caption(
    "Enter the new patient's values."
)

user_inputs = {}

for variable in PREDICTORS:

    meta = CONFIG[
        "ui_metadata"
    ][variable]

    label = clean_label(
        variable
    )

    if meta["type"] == "categorical":

        levels = [
            str(x)
            for x in meta[
                "levels"
            ]
        ]

        default = str(
            meta.get(
                "default",
                levels[0]
            )
        )

        index = (
            levels.index(
                default
            )
            if default in levels
            else 0
        )

        user_inputs[
            variable
        ] = st.sidebar.selectbox(
            label=label,
            options=levels,
            index=index,
        )

    else:

        user_inputs[
            variable
        ] = st.sidebar.number_input(
            label=label,
            value=float(
                meta["default"]
            ),
            step=float(
                meta["step"]
            ),
        )


clicked = st.sidebar.button(
    "🚀 Predict",
    type="primary",
    width="stretch",
)


# ============================================================
# 9. Result
# ============================================================

if clicked:

    try:

        result = predict_and_explain(
            user_inputs
        )

        fig = make_classic_force_plot(
            result,
            user_inputs,
        )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )

        st.stop()


    st.subheader(
        "Prediction Result"
    )

    st.metric(
        "Probability of poor wound healing",
        f"{result['probability_poor']:.1%}",
    )

    st.subheader(
        "Individualized SHAP Force Plot"
    )

    st.pyplot(
        fig,
        width="stretch",
        clear_figure=True,
    )

else:

    st.info(
        "👈 Enter the patient's clinical parameters "
        "and click [Predict]."
    )
