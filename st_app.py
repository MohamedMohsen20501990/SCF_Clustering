import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.inference import InferencePipeline

MODEL_PATH = "artifacts/model.pkl"
TRAIN_DATA_PATH = "artifacts/train.csv"
MAX_PLOT_POINTS = 4000

st.set_page_config(
    page_title="Household Cluster Explorer",
    page_icon="🏠",
    layout="wide",
)


# ----------------------------------------------------------------------
# Loading and reference data (cached so it runs once per server session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and reference data...")
def load_reference() -> dict:
    """Load the inference pipeline and build everything needed for plotting."""
    pipeline = InferencePipeline(model_path=MODEL_PATH)
    cols = list(pipeline.high_var_cols)

    raw = pd.read_csv(TRAIN_DATA_PATH)[cols].dropna().reset_index(drop=True)
    scaled = pd.DataFrame(
        pipeline.preprocessor.transform(raw), columns=cols, index=raw.index
    )
    labels = pipeline.model.predict(scaled)

    pca = PCA(n_components=2, random_state=42).fit(scaled)
    coords = pca.transform(scaled)

    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    plot_df["Cluster"] = [f"Cluster {l}" for l in labels]
    if len(plot_df) > MAX_PLOT_POINTS:
        plot_df = plot_df.sample(MAX_PLOT_POINTS, random_state=42)

    profile = raw.groupby(labels).mean()
    profile.insert(0, "Households", pd.Series(labels).value_counts().sort_index())
    profile.index = [f"Cluster {i}" for i in profile.index]
    profile.index.name = "Cluster"

    return {
        "pipeline": pipeline,
        "cols": cols,
        "raw": raw,
        "labels": labels,
        "pca": pca,
        "plot_df": plot_df,
        "profile": profile,
        "centers": pipeline.model.cluster_centers_,
    }


def analyze(ref: dict, values: dict) -> dict:
    """Run the prediction for one set of inputs and collect plotting info."""
    cols = ref["cols"]
    pipeline = ref["pipeline"]

    prediction = pipeline.predict(input_data={c: [v] for c, v in values.items()})
    # Works whether predict returns an int, a list, or an array.
    cluster = int(np.asarray(prediction).ravel()[0])

    user_raw = pd.DataFrame([values], columns=cols, dtype=float)
    user_scaled = pd.DataFrame(
        np.asarray(pipeline.preprocessor.transform(user_raw), dtype=float),
        columns=cols,
    )
    user_xy = ref["pca"].transform(user_scaled)[0]
    user_vec = user_scaled.to_numpy(dtype=float).reshape(-1)
    distances = np.linalg.norm(ref["centers"] - user_vec, axis=1)

    return {
        "cluster": cluster,
        "user_scaled": user_scaled.iloc[0],
        "user_xy": user_xy,
        "distances": distances,
    }


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def cluster_map(ref: dict, result: dict) -> alt.Chart:
    """Scatter of the training households (PCA 2D) with the user's point on top."""
    points = (
        alt.Chart(ref["plot_df"])
        .mark_circle(size=35, opacity=0.35)
        .encode(
            x=alt.X("PC1:Q", title="Principal component 1"),
            y=alt.Y("PC2:Q", title="Principal component 2"),
            color=alt.Color("Cluster:N", scale=alt.Scale(scheme="category10")),
            tooltip=["Cluster:N"],
        )
    )

    user_df = pd.DataFrame(
        {
            "PC1": [result["user_xy"][0]],
            "PC2": [result["user_xy"][1]],
            "Label": ["You"],
        }
    )
    marker = (
        alt.Chart(user_df)
        .mark_point(shape="diamond", size=450, filled=True, color="black", stroke="white", strokeWidth=2)
        .encode(x="PC1:Q", y="PC2:Q", tooltip=["Label:N"])
    )
    label = (
        alt.Chart(user_df)
        .mark_text(dy=-18, fontSize=14, fontWeight="bold")
        .encode(x="PC1:Q", y="PC2:Q", text="Label:N")
    )
    return (points + marker + label).properties(height=480).interactive()


def comparison_chart(ref: dict, result: dict) -> alt.Chart:
    """Grouped bars: the user's standardized values vs the cluster average."""
    cols = ref["cols"]
    center = ref["centers"][result["cluster"]]
    data = pd.DataFrame(
        {
            "Feature": cols * 2,
            "Series": ["You"] * len(cols) + ["Cluster average"] * len(cols),
            "Z-score": list(result["user_scaled"].to_numpy()) + list(center),
        }
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Feature:N", axis=alt.Axis(labelAngle=0), title=None),
            xOffset="Series:N",
            y=alt.Y("Z-score:Q", title="Standardized value (z-score)"),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#9aa5b1", "#1f77b4"])),
            tooltip=["Feature:N", "Series:N", alt.Tooltip("Z-score:Q", format=".2f")],
        )
        .properties(height=380)
    )


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏠 Household Cluster Explorer")
st.caption(
    "Enter a household's financial values in their original units. "
    "The app standardizes them, assigns a cluster with the trained KMeans "
    "model, and shows where the household sits among the training data."
)

try:
    ref = load_reference()
except Exception as exc:
    st.error(f"Could not load the model or reference data: {exc}")
    st.stop()

cols = ref["cols"]
raw = ref["raw"]

with st.sidebar:
    st.header("Household inputs")
    with st.form("input_form"):
        values = {}
        for col in cols:
            std = float(raw[col].std()) or 1.0
            values[col] = st.number_input(
                col,
                value=float(raw[col].median()),
                step=max(std / 20, 0.01),
                format="%.2f",
                help=f"Median in training data: {raw[col].median():,.2f}",
            )
        submitted = st.form_submit_button("Predict cluster", use_container_width=True, type="primary")

if submitted:
    try:
        st.session_state["result"] = analyze(ref, values)
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"Prediction failed: {exc}")

result = st.session_state.get("result")

if result is None:
    st.info("Fill in the values in the sidebar and click **Predict cluster**.")
    st.subheader("Training data overview")
    st.altair_chart(
        alt.Chart(ref["plot_df"])
        .mark_circle(size=35, opacity=0.35)
        .encode(
            x=alt.X("PC1:Q", title="Principal component 1"),
            y=alt.Y("PC2:Q", title="Principal component 2"),
            color=alt.Color("Cluster:N", scale=alt.Scale(scheme="category10")),
        )
        .properties(height=480)
        .interactive(),
        use_container_width=True,
    )
    st.stop()

cluster = result["cluster"]
size = int((ref["labels"] == cluster).sum())
share = size / len(ref["labels"])

m1, m2, m3 = st.columns(3)
m1.metric("Predicted cluster", f"Cluster {cluster}")
m2.metric("Cluster size", f"{size:,}", f"{share:.1%} of households", delta_color="off")
m3.metric("Distance to cluster center", f"{float(result['distances'][cluster]):.2f}")

tab_map, tab_compare, tab_profiles = st.tabs(
    ["📍 Cluster map", "📊 You vs. cluster", "🗂️ Cluster profiles"]
)

with tab_map:
    st.altair_chart(cluster_map(ref, result), use_container_width=True)
    ev = ref["pca"].explained_variance_ratio_
    st.caption(
        f"The {len(cols)} features are projected onto 2 dimensions (PCA), which "
        f"captures {ev.sum():.0%} of the variance. The black diamond is your "
        f"household. Drag to pan and scroll to zoom."
    )

with tab_compare:
    st.altair_chart(comparison_chart(ref, result), use_container_width=True)
    st.caption(
        "A z-score of 0 is the average household; positive values are above "
        "average and negative values are below."
    )
    dist_df = (
        pd.DataFrame(
            {
                "Cluster": [f"Cluster {i}" for i in range(len(result["distances"]))],
                "Distance": result["distances"],
            }
        )
        .sort_values("Distance")
        .reset_index(drop=True)
    )
    st.markdown("**Distance to each cluster center** (smaller means more similar)")
    st.dataframe(dist_df, hide_index=True, use_container_width=True)

with tab_profiles:
    st.markdown("**Average values per cluster (original units)**")
    st.dataframe(
        ref["profile"].style.format("{:,.2f}", subset=cols).format("{:,}", subset=["Households"]),
        use_container_width=True,
    )
    st.markdown(f"**Your input**")
    st.dataframe(pd.DataFrame([values]).style.format("{:,.2f}"), hide_index=True, use_container_width=True)