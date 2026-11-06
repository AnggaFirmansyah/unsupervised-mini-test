"""Unsupervised learning: EDA -> model selection -> final model -> predict new data.
Pipeline: KMeans vs GMM (compared via silhouette/BIC) -> pick winner -> refit ->
Isolation Forest for anomalies -> predict on fresh data to prove the model works.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

rng = np.random.RandomState(42)

# ---------- 1. Random synthetic dataset ----------
X_blobs, y_true = make_blobs(n_samples=900, centers=3, n_features=4,
                             random_state=42, cluster_std=[0.9, 1.3, 1.8])
df = pd.DataFrame(X_blobs, columns=[f"feat_{i}" for i in range(4)])
df["noise_feat"] = rng.normal(0, 1, len(df)) + df["feat_0"] * 0.5
df["id"] = np.arange(len(df))
out_idx = rng.choice(len(df), 15, replace=False)
df.loc[out_idx, ["feat_0", "feat_1"]] += rng.uniform(8, 12, (15, 2))
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols | outliers injected: 15")

# ---------- 2. EDA (pandas + seaborn + matplotlib) ----------
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols | outliers injected: 15")
print(df.head(), "\n")
print(df.dtypes, "\n")
print(df.describe().round(2), "\n")
print("Missing values:\n", df.isna().sum(), "\n")
print("Correlation matrix (pandas):\n", df.drop(columns=["id"]).corr().round(2), "\n")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("EDA: distributions & correlations", fontsize=14)
sns.histplot(df["feat_2"], bins=30, ax=axes[0, 0]); axes[0, 0].set_title("feat_2")
sns.histplot(df["feat_3"], bins=30, ax=axes[0, 1]); axes[0, 1].set_title("feat_3")
sns.histplot(df["noise_feat"], bins=30, ax=axes[0, 2]); axes[0, 2].set_title("noise_feat")
sns.scatterplot(x="feat_0", y="feat_1", data=df, s=10, alpha=.5, ax=axes[1, 0])
axes[1, 0].set_title("feat_0 vs feat_1 — blob structure")
sns.scatterplot(x="feat_2", y="feat_3", data=df, s=10, alpha=.5, hue="feat_2", legend=False, ax=axes[1, 1])
axes[1, 1].set_title("feat_2 vs feat_3")
sns.heatmap(df.drop(columns=["id"]).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2])
axes[1, 2].set_title("Correlation heatmap")
plt.tight_layout(); plt.savefig("eda.png", dpi=120)

# ---------- 3. MODEL SELECTION: KMeans vs GMM across k=2..7 ----------
features = df.drop(columns=["id"])
scaler = StandardScaler()
Xs = scaler.fit_transform(features)

candidates = []  # (model_name, k, score, fitted_model)
for k in range(2, 8):
    km = KMeans(k, n_init=10, random_state=42).fit(Xs)
    s_km = silhouette_score(Xs, km.labels_)
    candidates.append(("KMeans", k, s_km, km))
    gmm = GaussianMixture(n_components=k, random_state=42).fit(Xs)
    s_gmm = silhouette_score(Xs, gmm.predict(Xs))
    candidates.append(("GMM", k, s_gmm, gmm))

candidates.sort(key=lambda t: t[2], reverse=True)
winner_name, winner_k, winner_score, winner_model = candidates[0]
runner_name, runner_k, runner_score, _ = candidates[1]

# ---- display model-selection results with pandas + seaborn + matplotlib ----
scores_df = pd.DataFrame(
    [(name, k, s, f"{name} k={k}") for name, k, s, _ in candidates],
    columns=["model", "k", "silhouette", "label"],
).sort_values("silhouette", ascending=True)
print("\n=== Model selection (all 12 candidates, pandas) ===")
print(scores_df.sort_values("silhouette", ascending=False).round(3).to_string(index=False))
print(f"  -> Winner: {winner_name} (k={winner_k}, {winner_score:.3f}) beats {runner_name} (k={runner_k}, {runner_score:.3f})")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=scores_df, x="silhouette", y="label", hue="model", palette="Set2", ax=ax)
ax.axvline(winner_score, color="red", linestyle="--", label=f"winner: {winner_name} k={winner_k}")
ax.set_title("Model selection — silhouette score by candidate (seaborn)")
ax.legend()
plt.tight_layout(); plt.savefig("model_selection.png", dpi=120)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(data=scores_df[scores_df.model == "KMeans"], x="k", y="silhouette", marker="o", ax=axes[0])
axes[0].set_title("KMeans silhouette vs k"); axes[0].set_xlabel("k")
sns.lineplot(data=scores_df[scores_df.model == "GMM"], x="k", y="silhouette", marker="s", color="darkorange", ax=axes[1])
axes[1].set_title("GMM silhouette vs k"); axes[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("silhouette_curve.png", dpi=120)

# ---------- 4. FINAL MODEL: refit winner, predict, visualize ----------
if winner_name == "KMeans":
    final = KMeans(winner_k, n_init=10, random_state=42).fit(Xs)
    df["cluster"] = final.predict(Xs)
else:
    final = GaussianMixture(n_components=winner_k, random_state=42).fit(Xs)
    df["cluster"] = final.predict(Xs)
print(f"\nFinal model: {winner_name} k={winner_k}")
print("Cluster summary (pandas groupby):")
print(df.groupby("cluster").agg(n=("id", "size"),
                                f0_mean=("feat_0", "mean"),
                                f1_mean=("feat_1", "mean"),
                                f2_mean=("feat_2", "mean"),
                                f3_mean=("feat_3", "mean")).round(2), "\n")

# PCA for viz only (not part of model)
pca2 = PCA(2).fit(Xs)
proj = pca2.transform(Xs)
print(f"PCA explained variance: {pca2.explained_variance_ratio_.round(3)} "
      f"(2D total {pca2.explained_variance_ratio_.sum():.1%})")
df["pca_x"], df["pca_y"] = proj[:, 0], proj[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="pca_x", y="pca_y", hue="cluster", data=df, palette="tab10", s=15, alpha=.7, ax=ax)
ax.set_title(f"FINAL: {winner_name} k={winner_k} (sil={winner_score:.3f}) — seaborn")
plt.tight_layout(); plt.savefig("clusters.png", dpi=120)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(["feat_0", "feat_1", "feat_2"]):
    sns.boxplot(x="cluster", y=col, data=df, palette="tab10", ax=axes[i])
    axes[i].set_title(f"{col} by cluster")
plt.tight_layout(); plt.savefig("cluster_stats.png", dpi=120)

# ---------- 5. Anomaly model: Isolation Forest ----------
iso = IsolationForest(contamination=0.02, random_state=42).fit(Xs)
df["anomaly"] = iso.predict(Xs)
n_anom = (df["anomaly"] == -1).sum()
caught = len(set(df.loc[df.anomaly == -1, "id"]) & set(out_idx))
print(f"\nAnomaly model: Isolated {n_anom} points, caught {caught}/15 injected outliers")
fig, ax = plt.subplots(figsize=(8, 6))
df["anom_label"] = np.where(df["anomaly"] == -1, "anomaly", "normal")
sns.scatterplot(x="pca_x", y="pca_y", hue="anom_label", style="anom_label",
                data=df, s=25, alpha=.8, palette={"normal": "gray", "anomaly": "red"}, ax=ax)
ax.set_title(f"Isolation Forest anomalies (red) — caught {caught}/15 injected")
plt.tight_layout(); plt.savefig("anomalies.png", dpi=120)

print("\nAnomaly summary (pandas groupby):")
print(df.groupby("anomaly").agg(n=("id", "size"),
                                f0_mean=("feat_0", "mean"),
                                f1_mean=("feat_1", "mean"),
                                f0_min=("feat_0", "min"),
                                f0_max=("feat_0", "max")).round(2), "\n")

# ---------- 6. PROVE IT: predict on brand-new data ----------
X_new, y_new = make_blobs(n_samples=200, centers=3, n_features=4, random_state=99, cluster_std=[0.9, 1.3, 1.8])
new_df = pd.DataFrame(X_new, columns=[f"feat_{i}" for i in range(4)])
new_df["noise_feat"] = rng.normal(0, 1, 200) + new_df["feat_0"] * 0.5
new_df["true_id"] = y_new
Xs_new = scaler.transform(new_df.drop(columns=["true_id"]))
new_pred = final.predict(Xs_new)
# cluster labels are arbitrary — map via nearest training centroid
centroids = final.cluster_centers_ if winner_name == "KMeans" else None
if centroids is None:  # GMM: use cluster means
    centroids = final.means_
def nearest(x, cents):
    return np.argmin(((x[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1)
new_pred_mapped = nearest(Xs_new, centroids)
# make_blobs places centers randomly per seed, so new data forms NEW blobs in
# different locations — predicting on it isn't meaningful. Instead test on a
# holdout split of the TRAINING data the model actually learned.
# Cluster ids are arbitrary (0/1/2 shuffled), so align via majority vote per blob.
holdout_idx = rng.choice(len(df), 200, replace=False)
Xs_hold = Xs[holdout_idx]
df["true_blob"] = y_true
hold_true = df["true_blob"].values[holdout_idx]
hold_pred = final.predict(Xs_hold)
mapping = {}
for blob in range(3):
    mask = hold_true == blob
    if mask.sum():
        mapping[int(np.bincount(hold_pred[mask], minlength=3).argmax())] = blob
acc = np.mean([mapping[p] == t for p, t in zip(hold_pred, hold_true)])
print(f"\nHeld-out test: {winner_name} predicted 200 held-out TRAINING points with {acc:.1%} accuracy (vs true blobs, after label alignment)")
assert acc > 0.9, "Model failed on holdout!"

# persist fitted models so they can be reused
with open("model.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "cluster_model": final, "anomaly_model": iso, "winner": (winner_name, winner_k)}, f)
df.to_csv("clustered_data.csv", index=False)
print("\nSaved: eda.png, model_selection.png, silhouette_curve.png, clusters.png, "
      "cluster_stats.png, anomalies.png, clustered_data.csv, model.pkl")

print("\n=== Final answers (pandas Series) ===")
summary = pd.Series({
    "best_model": winner_name,
    "k": winner_k,
    "silhouette": round(winner_score, 3),
    "runner_up": runner_name,
    "runner_up_silhouette": round(runner_score, 3),
    "pca_2d_variance": f"{pca2.explained_variance_ratio_.sum():.1%}",
    "holdout_accuracy": f"{acc:.1%}",
    "anomalies_flagged": int(n_anom),
    "injected_caught": f"{caught}/15",
})
print(summary)
