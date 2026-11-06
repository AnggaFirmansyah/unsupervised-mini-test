# Unsupervised Learning Test

An end-to-end unsupervised learning pipeline on a synthetic dataset: exploratory data analysis (EDA), model selection, clustering, dimensionality reduction, anomaly detection, and a held-out evaluation. All results are displayed with pandas, seaborn, and matplotlib.

## Contents

- `unsupervised.py` — the full pipeline (see Pipeline below)
- `clustered_data.csv` — clustered dataset with an `id`, `cluster`, and `anomaly` column per row
- `model.pkl` — pickled fitted models (scaler, cluster model, anomaly model)
- Plot outputs:
  - `eda.png` — distributions, scatter plots, correlation heatmap
  - `model_selection.png` — bar chart of all 12 candidates ranked by silhouette score
  - `silhouette_curve.png` — silhouette score vs k for KMeans and GMM
  - `clusters.png` — final clusters in PCA projection
  - `cluster_stats.png` — box plots of each feature by cluster
  - `anomalies.png` — anomalies flagged by the Isolation Forest

## Requirements

- Python 3
- scikit-learn, pandas, numpy, matplotlib, seaborn

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## Usage

```bash
python unsupervised.py
```

The script runs unattended: it generates data, prints pandas tables to the console, saves every plot as a PNG, writes `clustered_data.csv`, and pickles the fitted models to `model.pkl`.

## Dataset

Synthetic data generated with `sklearn.datasets.make_blobs`:

- 900 samples, 4 features (`feat_0` to `feat_3`)
- 3 Gaussian blobs with increasing spread (`cluster_std=[0.9, 1.3, 1.8]`)
- 1 engineered noise feature (`noise_feat = feat_0 * 0.5 + N(0, 1)`)
- 15 injected outliers shifted by +8 to +12 on `feat_0` and `feat_1`

## Pipeline

1. **EDA**
   - Pandas: `head`, `dtypes`, `describe`, missing-value counts, correlation matrix
   - Seaborn: histograms of 3 features, pairwise scatter plots, annotated correlation heatmap (`eda.png`)
2. **Model selection**
   - 12 candidates evaluated: KMeans and GaussianMixture for k = 2, 3, 4, 5, 6, 7
   - Each candidate scored with the silhouette coefficient
   - Results shown in a pandas table and ranked in `model_selection.png`; score curves in `silhouette_curve.png`
3. **Final clustering model**
   - The winning configuration is refit on standardized features
   - Per-cluster statistics (size and feature means) shown with `pandas.groupby`
   - PCA projects the data to 2D for visualization (`clusters.png`), with per-cluster box plots in `cluster_stats.png`
4. **Anomaly detection**
   - Isolation Forest with 2% contamination
   - Detected anomalies plotted over the PCA projection (`anomalies.png`)
   - Anomaly summary shown with `pandas.groupby`
5. **Held-out evaluation**
   - 200 random rows held out from the training data
   - Cluster labels are arbitrary, so predictions are aligned to the true blobs by majority vote per blob before scoring
6. **Persistence**
   - Fitted scaler, cluster model, and anomaly model saved to `model.pkl`
   - Enriched dataset saved to `clustered_data.csv`

## Results

| Metric | Value |
|---|---|
| Winning model | KMeans, k = 3 |
| Silhouette score | 0.684 |
| Runner-up | GMM, k = 3 (0.684) |
| Cluster sizes | 300 / 303 / 297 |
| PCA 2D variance | 92.5% |
| Held-out accuracy | 99.0% |
| Anomalies flagged | 18 |
| Injected outliers caught | 8 / 15 |

Note on anomaly detection: 8 of the 15 injected outliers were recovered. The injected points were shifted along only two features before standardization, so after scaling some remained statistically similar to genuine cluster-edge points. Raising the contamination parameter or injecting outliers in the full feature space would improve recall.

## Outputs Explained

### Console (pandas)

- `df.head()` — first 5 rows of the raw dataset
- `df.dtypes` — column types
- `df.describe()` — count, mean, std, min, quartiles, max per feature
- `df.isna().sum()` — missing values per column (all zero)
- `df.corr()` — Pearson correlation matrix
- Model-selection table — all 12 candidates sorted by silhouette score
- `groupby("cluster")` — cluster sizes and per-feature means
- `groupby("anomaly")` — anomaly vs normal counts and feature ranges
- Final `pd.Series` summary of the run

### Plots

- `eda.png` — EDA dashboard: 3 histograms, 2 scatter plots, correlation heatmap
- `model_selection.png` — every candidate's silhouette score, winner marked
- `silhouette_curve.png` — silhouette vs k for KMeans and GMM
- `clusters.png` — final 3 clusters in PCA space
- `cluster_stats.png` — box plots of `feat_0`, `feat_1`, `feat_2` by cluster
- `anomalies.png` — Isolation Forest anomalies in red over PCA projection

## Saved Model

`model.pkl` is a pickle containing:

- `scaler` — the fitted `StandardScaler`
- `cluster_model` — the winning fitted cluster model (KMeans)
- `anomaly_model` — the fitted `IsolationForest`
- `winner` — tuple of (model name, k)

## Limitations

- Blob centers from `make_blobs` are randomly placed per seed; generating a second dataset with a different seed produces blobs in different locations, so a model trained on one dataset cannot be evaluated against a freshly generated one. The held-out test therefore uses a split of the training data.
- All 12 candidates tie or nearly tie in silhouette at k = 3 (KMeans and GMM both 0.684); the winner is chosen by sort order, so interpret the margin as negligible.
- Anomaly injection affected only 2 of 5 features, which compresses the outliers' distance from the bulk after standardization.
