import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

os.makedirs("results-img", exist_ok=True)
rng = np.random.RandomState(42)

# data: 3 blobs + a noise column + 15 planted outliers
X, y_true = make_blobs(n_samples=900, centers=3, n_features=4,
                       random_state=42, cluster_std=[0.9, 1.3, 1.8])
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
df["noise_feat"] = rng.normal(0, 1, len(df)) + df["feat_0"] * 0.5
out_idx = rng.choice(len(df), 15, replace=False)
df.loc[out_idx, ["feat_0", "feat_1"]] += rng.uniform(8, 12, (15, 2))

print(df.describe().round(2), "\n")
print("corr:\n", df.corr().round(2), "\n")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("EDA: distributions & correlations", fontsize=14)
sns.histplot(df["feat_2"], bins=30, ax=axes[0, 0]); axes[0, 0].set_title("feat_2")
sns.histplot(df["feat_3"], bins=30, ax=axes[0, 1]); axes[0, 1].set_title("feat_3")
sns.histplot(df["noise_feat"], bins=30, ax=axes[0, 2]); axes[0, 2].set_title("noise_feat")
sns.scatterplot(x="feat_0", y="feat_1", data=df, s=10, alpha=.5, ax=axes[1, 0])
axes[1, 0].set_title("feat_0 vs feat_1 — blob structure")
sns.scatterplot(x="feat_2", y="feat_3", data=df, s=10, alpha=.5, hue="feat_2", legend=False, ax=axes[1, 1])
axes[1, 1].set_title("feat_2 vs feat_3")
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2])
axes[1, 2].set_title("Correlation heatmap")
plt.tight_layout(); plt.savefig("results-img/eda.png", dpi=120)

# scale, then try KMeans vs GMM at k=2..7, keep best by silhouette
Xs = StandardScaler().fit_transform(df)
cands = []
for k in range(2, 8):
    km = KMeans(k, n_init=10, random_state=42).fit(Xs)
    cands.append(("KMeans", k, silhouette_score(Xs, km.labels_), km))
    gmm = GaussianMixture(n_components=k, random_state=42).fit(Xs)
    cands.append(("GMM", k, silhouette_score(Xs, gmm.predict(Xs)), gmm))
cands.sort(key=lambda t: t[2], reverse=True)
name, k, sil, model = cands[0]
print(f"winner: {name} k={k} silhouette={sil:.3f} (runner-up {cands[1][0]} k={cands[1][1]}: {cands[1][2]:.3f})")

scores = pd.DataFrame([(n, kk, s, f"{n} k={kk}") for n, kk, s, _ in cands],
                      columns=["model", "k", "silhouette", "label"]).sort_values("silhouette")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=scores, x="silhouette", y="label", hue="model", palette="Set2", ax=ax)
ax.axvline(sil, color="red", linestyle="--", label=f"winner: {name} k={k}")
ax.set_title("Model selection — silhouette by candidate")
ax.legend()
plt.tight_layout(); plt.savefig("results-img/model_selection.png", dpi=120)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for m, ax in zip(["KMeans", "GMM"], axes):
    sns.lineplot(data=scores[scores.model == m], x="k", y="silhouette", marker="o", ax=ax)
    ax.set_title(f"{m} silhouette vs k"); ax.set_xlabel("k")
plt.tight_layout(); plt.savefig("results-img/silhouette_curve.png", dpi=120)

# refit winner, cluster, show where the clusters sit in PCA space
if name == "KMeans":
    final = KMeans(k, n_init=10, random_state=42).fit(Xs)
else:
    final = GaussianMixture(n_components=k, random_state=42).fit(Xs)
df["cluster"] = final.predict(Xs)
print(df.groupby("cluster").mean().round(2), "\n")

pca = PCA(2).fit(Xs)
df["pca_x"], df["pca_y"] = pca.transform(Xs).T
print(f"PCA 2D variance: {pca.explained_variance_ratio_.sum():.1%}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="pca_x", y="pca_y", hue="cluster", data=df, palette="tab10", s=15, alpha=.7, ax=ax)
ax.set_title(f"FINAL: {name} k={k} (sil={sil:.3f})")
plt.tight_layout(); plt.savefig("results-img/clusters.png", dpi=120)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(["feat_0", "feat_1", "feat_2"]):
    sns.boxplot(x="cluster", y=col, data=df, ax=axes[i])
    axes[i].set_title(f"{col} by cluster")
plt.tight_layout(); plt.savefig("results-img/cluster_stats.png", dpi=120)

# anomalies
iso = IsolationForest(contamination=0.02, random_state=42).fit(Xs)
df["anomaly"] = iso.predict(Xs)
caught = len(set(df.loc[df.anomaly == -1].index) & set(out_idx))
df["anom_label"] = np.where(df["anomaly"] == -1, "anomaly", "normal")
print(f"flagged {(df.anomaly == -1).sum()} anomalies, caught {caught}/15 planted")

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="pca_x", y="pca_y", hue="anom_label", style="anom_label",
                data=df, s=25, alpha=.8, palette={"normal": "gray", "anomaly": "red"}, ax=ax)
ax.set_title(f"Isolation Forest anomalies (red) — caught {caught}/15")
plt.tight_layout(); plt.savefig("results-img/anomalies.png", dpi=120)

# sanity check on 200 held-out rows (cluster labels are arbitrary, align by majority vote)
hold = rng.choice(len(df), 200, replace=False)
pred = final.predict(Xs[hold])
mapping = {int(np.bincount(pred[y_true[hold] == b], minlength=3).argmax()): b
           for b in range(3) if (y_true[hold] == b).any()}
acc = np.mean([mapping[p] == t for p, t in zip(pred, y_true[hold])])
print(f"held-out accuracy: {acc:.1%}")
assert acc > 0.9

pickle.dump({"cluster_model": final, "anomaly_model": iso}, open("model.pkl", "wb"))
df.to_csv("clustered_data.csv", index=False)
print(pd.Series({"best_model": name, "k": k, "silhouette": round(sil, 3),
                 "pca_2d": f"{pca.explained_variance_ratio_.sum():.1%}",
                 "accuracy": f"{acc:.1%}", "caught": f"{caught}/15"}))
