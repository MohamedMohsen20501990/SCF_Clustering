import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import DataPipeline
from src.logging_config import logger
from src.preprocess import Preprocessor
from src.train import ModelTrainer

st.set_page_config(
    page_title="Household Credit-Fear Clustering",
    page_icon="📊",
    layout="wide",
)

DEFAULT_DATA_PATH = Path("data") / "raw" / "SCFP2019.csv"


@st.cache_data
def load_training_data(csv_path: str, fearful_only: bool) -> pd.DataFrame:
    """Load and optionally filter the training dataset."""
    pipeline = DataPipeline()
    return pipeline.read_csv_data(csv_path, fearful=fearful_only)


def fit_pipeline(
    df: pd.DataFrame,
    trim_var: bool,
    n_features: int,
    n_clusters: int | None,
    max_clusters: int,
) -> tuple[Preprocessor, ModelTrainer, pd.DataFrame, pd.Series]:
    """Run preprocessing and KMeans training, returning fitted objects and labels."""
    preprocessor = Preprocessor()
    df_scaled = preprocessor.start_preprocessing(
        data=df,
        trim_var=trim_var,
        n_features=n_features,
    )

    trainer = ModelTrainer()
    best_model = trainer.train_model(
        train_data=df_scaled,
        n_clusters=n_clusters,
        max_clusters=max_clusters,
    )
    labels = pd.Series(best_model.labels_, index=df.index, name="cluster")
    return preprocessor, trainer, df_scaled, labels


def build_cluster_profile(
    raw_df: pd.DataFrame,
    selected_features: list[str],
    labels: pd.Series,
) -> pd.DataFrame:
    """Summarize cluster sizes and original-scale feature averages."""
    profile_source = raw_df[selected_features].copy()
    profile_source["cluster"] = labels.astype(str)

    summary = (
        profile_source.groupby("cluster", as_index=False)[selected_features]
        .mean(numeric_only=True)
    )
    counts = labels.astype(str).value_counts().rename("household_count")
    summary = summary.merge(counts, left_on="cluster", right_index=True)
    return summary.sort_values("cluster")


def render_metrics_section(trainer: ModelTrainer) -> None:
    """Display model selection metrics and k-search scores if available."""
    if not trainer.report:
        return

    st.subheader("Model Evaluation")
    if isinstance(trainer.report, dict) and "n_clusters" in trainer.report:
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Clusters (k)", trainer.report["n_clusters"])
        col2.metric("Inertia", f"{trainer.report['inertia score']:.2f}")
        col3.metric("Silhouette Score", f"{trainer.report['silhouette_score']:.4f}")
    elif isinstance(trainer.report, dict):
        rows = [
            {
                "model": key,
                "inertia_score": vals["inertia_score"],
                "silhouette_score": vals["silhouette_score"],
            }
            for key, vals in trainer.report.items()
        ]
        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True)

        fig_sil = px.line(
            metrics_df,
            x="model",
            y="silhouette_score",
            markers=True,
            title="Silhouette Score Across Candidate Cluster Counts",
        )
        st.plotly_chart(fig_sil, use_container_width=True)


def main() -> None:
    st.title("KMeans Clustering & Test Prediction App")
    st.caption(
        "Configure cluster count, inspect cluster characteristics, and score new test records."
    )

    # --- Sidebar Configuration ---
    st.sidebar.header("1. Data & Feature Settings")
    data_path = st.sidebar.text_input("Training CSV Path", value=str(DEFAULT_DATA_PATH))
    fearful_only = st.sidebar.checkbox("Filter fearful households (TURNFEAR == 1)", value=True)
    trim_var = st.sidebar.checkbox("Select top features via trimmed variance", value=True)
    n_features = st.sidebar.slider("Number of features", min_value=2, max_value=15, value=5)

    st.sidebar.header("2. Cluster Selection")
    selection_mode = st.sidebar.radio(
        "Cluster count mode",
        options=["Choose n_clusters", "Auto-select best k (Silhouette)"],
    )
    if selection_mode == "Choose n_clusters":
        n_clusters: int | None = st.sidebar.slider(
            "Number of clusters (n_clusters)", min_value=2, max_value=12, value=4
        )
        max_clusters = 10
    else:
        n_clusters = None
        max_clusters = st.sidebar.slider(
            "Max clusters to evaluate", min_value=3, max_value=12, value=8
        )

    run_button = st.sidebar.button("Train & Cluster Data", type="primary")

    # --- Load Training Data ---
    try:
        train_df = load_training_data(data_path, fearful_only)
    except Exception as exc:
        logger.exception("Failed to load training data in Streamlit app.")
        st.error(f"Could not load training data from `{data_path}`: {exc}")
        return

    st.info(f"Loaded **{len(train_df):,}** rows and **{train_df.shape[1]}** columns from `{data_path}`.")

    # --- Train & Cache Session Results ---
    if run_button or "cluster_state" not in st.session_state:
        try:
            with st.spinner("Preprocessing features and fitting KMeans..."):
                preprocessor, trainer, df_scaled, labels = fit_pipeline(
                    df=train_df,
                    trim_var=trim_var,
                    n_features=n_features,
                    n_clusters=n_clusters,
                    max_clusters=max_clusters,
                )
            st.session_state["cluster_state"] = {
                "preprocessor": preprocessor,
                "trainer": trainer,
                "df_scaled": df_scaled,
                "labels": labels,
                "train_df": train_df,
            }
        except Exception as exc:
            logger.exception("Training failed in Streamlit app.")
            st.error(f"Clustering failed: {exc}")
            return

    state = st.session_state["cluster_state"]
    preprocessor: Preprocessor = state["preprocessor"]
    trainer: ModelTrainer = state["trainer"]
    df_scaled: pd.DataFrame = state["df_scaled"]
    labels: pd.Series = state["labels"]
    cached_train_df: pd.DataFrame = state["train_df"]

    selected_features = preprocessor.high_var_columns or list(df_scaled.columns)

    # --- Tabs for Exploration & Test Prediction ---
    tab_viz, tab_predict = st.tabs(["📈 Cluster Characteristics", "🧪 Predict Test Data"])

    with tab_viz:
        render_metrics_section(trainer)

        st.subheader("Cluster Profile & Feature Characteristics")
        profile_df = build_cluster_profile(cached_train_df, selected_features, labels)
        st.dataframe(profile_df, use_container_width=True)

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_counts = px.bar(
                profile_df,
                x="cluster",
                y="household_count",
                color="cluster",
                title="Records per Cluster",
                labels={"household_count": "Count", "cluster": "Cluster"},
            )
            st.plotly_chart(fig_counts, use_container_width=True)

        with col_chart2:
            melted = profile_df.melt(
                id_vars=["cluster", "household_count"],
                value_vars=selected_features,
                var_name="feature",
                value_name="mean_value",
            )
            fig_means = px.bar(
                melted,
                x="feature",
                y="mean_value",
                color="cluster",
                barmode="group",
                title="Mean Feature Values by Cluster (Original Scale)",
            )
            st.plotly_chart(fig_means, use_container_width=True)

        if len(selected_features) >= 2:
            st.subheader("Cluster Scatter View")
            x_col = st.selectbox("X-axis feature", options=selected_features, index=0)
            y_col = st.selectbox(
                "Y-axis feature",
                options=selected_features,
                index=min(1, len(selected_features) - 1),
            )
            scatter_df = df_scaled[[x_col, y_col]].copy()
            scatter_df["cluster"] = labels.astype(str)
            fig_scatter = px.scatter(
                scatter_df,
                x=x_col,
                y=y_col,
                color="cluster",
                opacity=0.65,
                title=f"Standardized Clusters: {x_col} vs. {y_col}",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_predict:
        st.subheader("Cluster New / Test Data")
        st.write(
            "Upload a CSV file containing the same feature columns used during training. "
            "The app applies your fitted `Preprocessor` and assigns clusters using the trained `KMeans` model."
        )

        uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                test_scaled = preprocessor.transform(test_df)
                best_model = trainer.best_model
                if best_model is None:
                    st.error("No trained KMeans model available.")
                    return

                predicted_clusters = best_model.predict(test_scaled)
                results_df = test_df.copy()
                results_df["predicted_cluster"] = predicted_clusters

                st.success(f"Assigned clusters to **{len(results_df):,}** test rows.")
                st.dataframe(results_df.head(50), use_container_width=True)

                test_counts = (
                    results_df["predicted_cluster"]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )
                test_counts.columns = ["cluster", "count"]
                fig_test = px.bar(
                    test_counts,
                    x="cluster",
                    y="count",
                    color="cluster",
                    title="Predicted Cluster Distribution on Test Data",
                )
                st.plotly_chart(fig_test, use_container_width=True)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Clustered Test Data (CSV)",
                    data=csv_bytes,
                    file_name="clustered_test_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                logger.exception("Test data prediction failed in Streamlit app.")
                st.error(f"Could not score test data: {exc}")


if __name__ == "__main__":
    main()