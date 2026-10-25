
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ExoHunter", page_icon="🪐", layout="wide")

FEATURES = {
    "koi_period": "Orbital period (days)",
    "koi_duration": "Transit duration (hours)",
    "koi_depth": "Transit depth (ppm)",
    "koi_prad": "Planet radius (Earth radii)",
    "koi_teq": "Equilibrium temperature (K)",
    "koi_insol": "Insolation (Earth flux)",
    "koi_model_snr": "Signal-to-noise ratio",
    "koi_impact": "Impact parameter",
    "koi_steff": "Star temperature (K)",
    "koi_slogg": "Star surface gravity (log g)",
    "koi_srad": "Star radius (Solar radii)",
}


@st.cache_data
def load_test_data():
    return pd.read_csv("data/processed/test.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.joblib")


test = load_test_data()
model = load_model()

st.title("🪐 ExoHunter")
st.write(
    "Classifying NASA Kepler observations as exoplanet candidates "
    "or likely false positives using machine learning."
)

st.subheader("Select an observation")
st.caption(
    "These observations come from the held-out test set, "
    "which the model never saw during training."
)

name = st.selectbox("Kepler Object of Interest (KOI)", test["kepoi_name"])
row = test[test["kepoi_name"] == name].iloc[0]

st.subheader("Measurements")
cols = st.columns(3)
for i, (col_name, label) in enumerate(FEATURES.items()):
    value = row[col_name]
    text = "N/A" if pd.isna(value) else f"{value:,.2f}"
    cols[i % 3].metric(label, text)

st.subheader("Prediction")

X_one = test[test["kepoi_name"] == name][list(FEATURES.keys())]
prob = model.predict_proba(X_one)[0, 1]

if prob >= 0.5:
    st.success("Model prediction: planet-like signal (exoplanet candidate)")
else:
    st.error("Model prediction: likely false positive")

st.metric("Model score for 'planet'", f"{prob:.1%}")

actual = "CONFIRMED planet" if row["label"] == 1 else "FALSE POSITIVE"
st.write(f"NASA catalog label (shown for comparison only): **{actual}**")

st.caption(
    "Limitations: this score is the share of Random Forest trees voting "
    "'planet', not a calibrated probability. The model only sees 11 catalog "
    "measurements, not the light curve. It classifies existing NASA catalog "
    "entries and does not discover new planets."
)


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

st.divider()
st.subheader("Model performance (Random Forest)")

X_test = test[list(FEATURES.keys())]
y_test = test["label"]
y_pred = model.predict(X_test)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
m2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
m3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
m4.metric("F1-score", f"{f1_score(y_test, y_pred):.3f}")

st.caption(
    f"Computed live on {len(test):,} held-out test observations "
    "(CONFIRMED planets vs FALSE POSITIVES)."
)

c1, c2 = st.columns(2)
c1.image("reports/figures/confusion_matrices.png", caption="Confusion matrices on the test set")
c2.image("reports/figures/rf_feature_importance.png", caption="Random Forest feature importance")


import plotly.express as px

st.divider()
st.subheader("Where does this observation sit?")

feature_label = st.selectbox("Feature to compare", list(FEATURES.values()))
feature = [k for k, v in FEATURES.items() if v == feature_label][0]
use_log = st.checkbox("Log scale", value=False)

plot_df = test.copy()
plot_df["Class"] = plot_df["label"].map(
    {1: "CONFIRMED planet", 0: "FALSE POSITIVE"}
)
plot_df = plot_df.dropna(subset=[feature])
if use_log:
    plot_df = plot_df[plot_df[feature] > 0]

fig = px.box(
    plot_df,
    x="Class",
    y=feature,
    color="Class",
    points=False,
    log_y=use_log,
    template="plotly_dark",
    labels={feature: feature_label},
)

selected = row[feature]
if pd.notna(selected) and (not use_log or selected > 0):
    fig.add_scatter(
        x=["FALSE POSITIVE", "CONFIRMED planet"],
        y=[selected, selected],
        mode="lines",
        line=dict(dash="dash", color="white"),
        name=f"{name}: {selected:,.2f}",
    )
else:
    st.info("This observation has no usable value for the selected feature.")

st.plotly_chart(fig, width="stretch")
st.caption(
    "Boxes show the spread of each class in the test set "
    "(median, quartiles and range). The dashed line is the selected observation."
)

st.divider()
st.subheader("NASA dataset overview")


@st.cache_data
def load_overview():
    labeled = pd.read_csv("data/processed/koi_labeled.csv")
    candidates = pd.read_csv("data/processed/koi_candidates.csv")
    return labeled, candidates


labeled, candidates = load_overview()

d1, d2, d3, d4 = st.columns(4)
d1.metric("Labeled observations", f"{len(labeled):,}")
d2.metric("CONFIRMED planets", f"{int(labeled['label'].sum()):,}")
d3.metric("FALSE POSITIVES", f"{int((labeled['label'] == 0).sum()):,}")
d4.metric("CANDIDATES (not used for training)", f"{len(candidates):,}")

st.write(
    "Data source: NASA Exoplanet Archive, Kepler Objects of Interest "
    "Cumulative Table (DOI: 10.26133/NEA4). Downloaded on 25 October 2026. "
    "The table is updated periodically, so newer downloads may differ."
)
st.write(
    "The model is trained on CONFIRMED planets (label 1) and FALSE POSITIVES "
    "(label 0). Disposition and vetting columns such as koi_score and the "
    "koi_fpflag columns are excluded to avoid data leakage."
)

st.divider()
st.subheader("Which features matter?")

st.image(
    "reports/figures/permutation_importance.png",
    caption="Permutation importance on the held-out test set (drop in F1-score)",
)

st.write(
    "Planet radius ranks first under both the Random Forest's built-in "
    "importance and permutation importance. The order of the other features "
    "differs between the two methods, so only the broad grouping should be "
    "trusted: planet size, signal quality and transit shape matter most, "
    "while stellar surface gravity has almost no measurable effect."
)
st.caption(
    "Several features are correlated (for example transit depth and planet "
    "radius), which can lower the score of each one individually. "
    "Importance describes what the model relies on, not physical causation."
)

st.divider()
st.subheader("Model comparison (held-out test set)")

MODEL_FILES = {
    "Random Forest": "models/random_forest.joblib",
    "Gradient Boosting": "models/gradient_boosting.joblib",
}

comparison_rows = []
for model_name, path in MODEL_FILES.items():
    m = joblib.load(path)
    pred = m.predict(X_test)
    comparison_rows.append(
        {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1-score": f1_score(y_test, pred),
        }
    )

st.dataframe(pd.DataFrame(comparison_rows).set_index("Model").round(3))

st.caption(
    "Both models use the same 11 features and the same train/test split. "
    "The differences are small and come from a single split, so they should "
    "not be read as one model clearly beating the other. Gradient Boosting "
    "tends to catch more planets (higher recall), while Random Forest raises "
    "fewer false alarms (higher precision)."
)

st.divider()
st.subheader("Limitations and honest notes")

st.markdown(
    """
- **Classification, not discovery.** ExoHunter classifies entries that already
  exist in NASA's Kepler catalog. It does not discover new exoplanets.
- **Labels come from NASA's vetting process.** The CONFIRMED and FALSE POSITIVE
  labels were produced by NASA and the Kepler team, so the model can only be as
  reliable as those labels. The catalog is updated periodically.
- **Limited input.** The model sees 11 summary measurements per observation,
  not the light curve itself.
- **CANDIDATE entries are not evaluated.** Their true status is not yet known,
  so there is no ground truth to measure the model against.
- **Errors are not random.** In the Random Forest's test-set mistakes, the
  median signal-to-noise ratio and transit depth were lower than for correct
  predictions. This is an observation from a small group of errors, not proof
  of the cause.
- **Scores are not calibrated probabilities.** The planet score is the share of
  trees voting for "planet".
- **Evaluation scope.** Results come from one train/test split plus 5-fold
  cross-validation on the training data, on a single dataset version.
"""
)