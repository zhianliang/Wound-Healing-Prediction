# ============================================================
# app.py
#
# Final R-trained XGBoost -> pure Python Streamlit deployment
#
# Web target:
#   Probability of POOR WOUND HEALING
#
# Display:
#   - Prediction Result
#   - Probability of poor wound healing
#   - Classic SHAP force plot
#
# Required files in the same folder:
#   app.py
#   xgb_final_model.json
#   preprocess_config.json
# ============================================================

from pathlib import Path
import json
import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import shap
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
# 3. Load model and preprocessing configuration
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

            # Training-derived 1% / 99% capping
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

    if not np.all(
        np.isfinite(vec)
    ):
        raise ValueError(
            "Preprocessing generated a non-finite model value."
        )

    return vec


# ============================================================
# 6. Prediction + exact TreeSHAP
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

    # Original model predicts P(Wound_Healing = yes)
    probability_yes = float(
        BOOSTER.predict(
            dnew
        )[0]
    )

    # Website target = poor wound healing
    probability_poor = (
        1.0 - probability_yes
    )

    # Exact XGBoost TreeSHAP for original yes class
    contribution = BOOSTER.predict(
        dnew,
        pred_contribs=True,
        approx_contribs=False,
    )[0]

    shap_yes = np.asarray(
        contribution[:-1],
        dtype=float,
    )

    base_yes = float(
        contribution[-1]
    )

    # Convert explanation to poor wound healing.
    #
    # p_poor = 1 - sigmoid(margin_yes)
    #        = sigmoid(-margin_yes)
    #
    # Hence:
    # base_poor = -base_yes
    # shap_poor = -shap_yes

    shap_poor = -shap_yes
    base_poor = -base_yes

    # Group encoded columns back into original clinical variables
    grouped = {}

    mapping = CONFIG[
        "model_to_original"
    ]

    for model_feature, shap_value in zip(
        MODEL_FEATURES,
        shap_poor,
    ):

        original_feature = mapping.get(
            model_feature,
            model_feature,
        )

        grouped[
            original_feature
        ] = (
            grouped.get(
                original_feature,
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

    # Additivity check
    margin_poor = (
        base_poor
        + float(
            grouped_shap.sum()
        )
    )

    reconstructed_probability = sigmoid(
        margin_poor
    )

    if abs(
        reconstructed_probability
        - probability_poor
    ) > 1e-5:
        raise ValueError(
            "SHAP additivity check failed."
        )

    return {
        "probability_poor": probability_poor,
        "base_poor": base_poor,
        "groups": groups,
        "grouped_shap": grouped_shap,
    }


# ============================================================
# 7. Classic SHAP force plot
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

    feature_names = [
        clean_label(
            variable
        )
        for variable in groups
    ]

    feature_values = np.asarray(
        [
            raw_values.get(
                variable,
                ""
            )
            for variable in groups
        ],
        dtype=object,
    )

    # Remove exact-zero contributions from display only.
    keep = np.where(
        np.abs(
            shap_values
        ) > 1e-12
    )[0]

    if len(keep) == 0:
        keep = np.arange(
            len(
                shap_values
            )
        )

    show_shap = shap_values[
        keep
    ]

    show_values = feature_values[
        keep
    ]

    show_names = [
        feature_names[i]
        for i in keep
    ]

    plt.close("all")

    # This is SHAP's classic matplotlib force plot.
    # link="logit" displays the horizontal scale in probability,
    # matching the classic visual form in the reference image.
    force_output = shap.force_plot(
        base_value=float(
            result["base_poor"]
        ),
        shap_values=show_shap,
        features=show_values,
        feature_names=show_names,
        link="logit",
        matplotlib=True,
        show=False,
        figsize=(18, 3.0),
        contribution_threshold=0.02,
        text_rotation=0,
    )

    if hasattr(
        force_output,
        "savefig"
    ):
        fig = force_output

    else:
        fig = plt.gcf()

    fig.set_size_inches(
        18,
        3.0,
        forward=True,
    )

    fig.patch.set_facecolor(
        "white"
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

    if meta[
        "type"
    ] == "categorical":

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
                meta[
                    "default"
                ]
            ),
            step=float(
                meta[
                    "step"
                ]
            ),
        )


clicked = st.sidebar.button(
    "🚀 Predict Risk",
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


    # --------------------------------------------------------
    # Prediction Result
    # --------------------------------------------------------

    st.subheader(
        "Prediction Result"
    )

    st.metric(
        "Probability of poor wound healing",
        f"{result['probability_poor']:.1%}",
    )


    # --------------------------------------------------------
    # Classic force plot
    # --------------------------------------------------------

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
        "and click [Predict Risk]."
    )
