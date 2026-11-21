# Customer Segmentation in the US

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mohamedmohsen20501990-scf-clustering-st-app-oinyom.streamlit.app/)

An end-to-end unsupervised machine learning project that segments U.S. households by financial profile using the **2019 Survey of Consumer Finances (SCF)**. The project covers the full lifecycle: exploratory analysis, preprocessing, K-Means modeling, a validated inference pipeline, and an interactive **Streamlit** app deployed to the cloud.

**Live demo:** https://mohamedmohsen20501990-scf-clustering-st-app-oinyom.streamlit.app/

---

## Table of Contents

- [Overview](#overview)
- [What I Built](#what-i-built)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [The Streamlit App](#the-streamlit-app)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Technologies](#technologies)
- [Engineering Practices](#engineering-practices)
- [Potential Applications](#potential-applications)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

Households differ widely in debt, wealth, and asset holdings, and averages hide those differences. This project uses **K-Means clustering** to group households into distinct financial segments, focusing on **credit-fearful households** (`TURNFEAR == 1`), and makes the model usable through a web app where anyone can enter a household's values and see which segment it belongs to.

## What I Built

- **Data pipeline:** loads the raw SCF data, filters it to credit-fearful households, and saves the training dataset.
- **Preprocessing module:** validates the data (type, emptiness, numeric columns, missing values), selects the most informative features by **trimmed variance**, and standardizes them with `StandardScaler`. The fitted preprocessor and the selected feature names are saved as artifacts.
- **Training module:** trains a **K-Means** model either with a user-defined number of clusters or by evaluating every `k` from 2 to a maximum and selecting the one with the highest **silhouette score**. The best model is saved.
- **Inference pipeline:** validates incoming data with **Pydantic** schemas, arranges the features in the training order, applies the saved scaler, and returns the predicted cluster.
- **Interactive web app:** a Streamlit dashboard that predicts a household's cluster and visualizes where it falls among the training data.
- **Deployment:** the app is deployed on Streamlit Community Cloud with pinned dependencies, so the saved models load reliably.

## Dataset

The project uses the **2019 Survey of Consumer Finances (SCF)**, sponsored by the Federal Reserve Board in cooperation with the U.S. Department of the Treasury and conducted by NORC at the University of Chicago. It is one of the most comprehensive sources of U.S. household finance data, covering assets, liabilities, income, net worth, retirement accounts, debt and credit, and demographics.

After filtering to credit-fearful households, the model uses the five highest-variance features:

| Feature | Description |
|---|---|
| `DEBT` | Total household debt |
| `NETWORTH` | Total household net worth |
| `HOUSES` | Value of owned houses |
| `NFIN` | Non-financial assets |
| `ASSET` | Total assets |

## Methodology

1. **Data cleaning and filtering:** load the SCF data and keep households with `TURNFEAR == 1`.
2. **Exploratory data analysis (EDA):** explore distributions and compare groups side by side (see the notebook).
3. **Feature selection:** rank features by **trimmed variance** (10% trimmed from each tail), which is robust to the extreme outliers typical of wealth data, and keep the top 5.
4. **Feature scaling:** standardize the selected features with `StandardScaler`, so no variable dominates the distance calculations.
5. **Dimensionality reduction:** use **PCA** to project the data to two dimensions for visualization.
6. **K-Means clustering:** fit the model and choose the number of clusters using inertia and **silhouette score**.
7. **Cluster interpretation:** profile each segment using the average original-unit values of its households.

## The Streamlit App

Live at https://mohamedmohsen20501990-scf-clustering-st-app-oinyom.streamlit.app/

Enter a household's values in their original units (for example, dollar amounts) and the app returns:

- **Predicted cluster**, the cluster's size and share of households, and the distance to the cluster center
- **Cluster map:** a 2D PCA projection of all training households colored by cluster, with the user's household highlighted
- **You vs. cluster:** the household's standardized values compared with the cluster average, plus its distance to every cluster center
- **Cluster profiles:** a table of the average values for each cluster

## Project Structure

```text
SCF_Clustering/
│
├── artifacts/                  # Saved model, preprocessor, selected features, training data
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── selected_features.pkl
│   ├── train.csv
│   └── transformed_data.parquet
│
├── data/
│   └── raw/SCFP2019.csv        # Raw SCF data
│
├── notebooks/
│   └── main.ipynb              # EDA and experiments
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data.py                 # Data loading, filtering, and end-to-end training run
│   ├── preprocess.py           # Validation, feature selection, and scaling
│   ├── train.py                # K-Means training and model selection
│   ├── inference.py            # Validated prediction pipeline
│   ├── schemas.py              # Pydantic input schemas
│   ├── utils.py                # Save and load helpers
│   └── logging_config.py       # Logger configuration
│
├── st_app.py                   # Streamlit application
├── requirements.txt
└── README.md
```

## Getting Started

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/MohamedMohsen20501990/SCF_Clustering.git
cd SCF_Clustering
pip install -r requirements.txt
```

**2. Train the model** (filters the data, preprocesses it, trains K-Means, and saves the artifacts)

```bash
python src/data.py
```

**3. Run the app locally**

```bash
streamlit run st_app.py
```

**4. Use the inference pipeline directly**

```python
from src.inference import InferencePipeline

pipeline = InferencePipeline(model_path="artifacts/model.pkl")
cluster = pipeline.predict(input_data={
    "DEBT": [250000],
    "NETWORTH": [400000],
    "HOUSES": [300000],
    "NFIN": [350000],
    "ASSET": [650000],
})
print(cluster)
```

Inputs must be in the original units of the dataset. The pipeline standardizes them using the saved scaler.

## Technologies

- **Language:** Python
- **Data and ML:** Pandas, NumPy, SciPy, Scikit-learn
- **Validation:** Pydantic
- **App and visualization:** Streamlit, Altair, Matplotlib, Seaborn
- **Development:** Jupyter Notebook, Git, GitHub, Streamlit Community Cloud

## Engineering Practices

- **Modular design:** separate modules for data, preprocessing, training, and inference
- **Input validation:** DataFrame checks during training and Pydantic schemas at inference
- **Consistent preprocessing:** the same fitted scaler is used for training and prediction
- **Logging:** centralized logging across the pipeline for traceability
- **Reproducibility:** fixed random seeds and saved artifacts
- **Dependency pinning:** versions are pinned in `requirements.txt`, since pickled models are sensitive to library versions

## Potential Applications

- Personalized marketing campaigns
- Customer profiling
- Financial product recommendations
- Market research
- Consumer behavior analysis
- Socioeconomic and wealth distribution studies

## Future Improvements

- Support batch predictions (multiple households per request)
- Compare additional algorithms, such as Gaussian Mixture Models and hierarchical clustering
- Add automated tests and a CI workflow
- Expose the model through a REST API

## Acknowledgements

This project uses data from the **2019 Survey of Consumer Finances (SCF)** provided by the **Federal Reserve Board**. The SCF is conducted in collaboration with the **U.S. Department of the Treasury** and the **National Opinion Research Center (NORC) at the University of Chicago**. Their commitment to collecting high-quality, nationally representative financial data makes this analysis possible.

## License

This project is intended for educational and research purposes. Please refer to the original SCF data documentation and licensing information before using the dataset for commercial applications.