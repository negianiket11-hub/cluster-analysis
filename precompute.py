"""
Run this script once locally to generate pre-computed data files.
Output goes to the data/ folder and is committed to GitHub.
Render then loads these small files instead of reprocessing the raw CSV.

Usage:  python precompute.py
"""

import json, gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score

print("Loading raw data...")
df_raw = pd.read_csv(
    "Online Retail(1).csv",
    encoding="utf-8-sig",
    parse_dates=["InvoiceDate"],
    dayfirst=True,
    dtype={"Quantity": "int32", "UnitPrice": "float32"},
)
df_raw["IsCancelled"] = df_raw["InvoiceNo"].astype(str).str.startswith("C")

# ── df_raw_dedup (for cancellation rate callback) ──────────
df_raw_dedup = df_raw.drop_duplicates().copy()
df_raw_dedup["Month"] = df_raw_dedup["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
df_raw_dedup = df_raw_dedup[["InvoiceNo", "InvoiceDate", "IsCancelled", "Country", "Month"]]

# ── Clean ──────────────────────────────────────────────────
df = df_raw.drop_duplicates()
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df = df.dropna(subset=["Description"])
del df_raw; gc.collect()

# ── df_txn (transaction-level, includes missing CustomerID) ─
df_txn = df.reset_index(drop=True).copy()
df_txn["Revenue"]    = df_txn["Quantity"] * df_txn["UnitPrice"]
df_txn["DayOfWeek"]  = df_txn["InvoiceDate"].dt.day_name()
df_txn["Hour"]       = df_txn["InvoiceDate"].dt.hour.astype("int8")
df_txn["DayCode"]    = df_txn["InvoiceDate"].dt.dayofweek.astype("int8")
df_txn["Month"]      = df_txn["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

# ── df_cust (customer-level, CustomerID required) ──────────
df_cust = df.dropna(subset=["CustomerID"]).reset_index(drop=True).copy()
df_cust["Revenue"]   = df_cust["Quantity"] * df_cust["UnitPrice"]
df_cust["DayOfWeek"] = df_cust["InvoiceDate"].dt.day_name()
df_cust["Hour"]      = df_cust["InvoiceDate"].dt.hour.astype("int8")
df_cust["Month"]     = df_cust["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
del df; gc.collect()

# ── ABC Analysis ───────────────────────────────────────────
print("ABC analysis...")
abc_all = (df_cust.groupby("Description")["Revenue"].sum()
           .sort_values(ascending=False).reset_index())
abc_all["CumPct"]      = abc_all["Revenue"].cumsum() / abc_all["Revenue"].sum() * 100
abc_all["ABCClass"]    = pd.cut(abc_all["CumPct"], bins=[0, 80, 95, 100], labels=["A", "B", "C"])
abc_all["ProductRank"] = range(1, len(abc_all) + 1)
abc_summary = abc_all.groupby("ABCClass", observed=True).agg(
    Products=("Description", "count"),
    Revenue =("Revenue",     "sum"),
).reset_index()
abc_summary["RevPct"] = (abc_summary["Revenue"] / abc_summary["Revenue"].sum() * 100).round(1)

# ── Missing CustomerID by Country ──────────────────────────
missing_by_country = df_txn.groupby("Country").agg(
    Total  =("CustomerID", "count"),
    Missing=("CustomerID", lambda x: x.isna().sum()),
).reset_index()
missing_by_country["MissingPct"] = (
    missing_by_country["Missing"] / missing_by_country["Total"] * 100
).round(1)
missing_by_country = (missing_by_country[missing_by_country["Total"] >= 50]
                      .sort_values("MissingPct", ascending=False).head(15))

# ── Extended RFM Clustering ────────────────────────────────
print("RFM clustering...")
snapshot = df_cust["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm_full = df_cust.groupby("CustomerID").agg(
    Recency   =("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency =("InvoiceNo",   "nunique"),
    Monetary  =("Revenue",     "sum"),
    BasketSize=("Quantity",    "mean"),
    Tenure    =("InvoiceDate", lambda x: (x.max() - x.min()).days),
).reset_index()
rfm_full["AOV"] = rfm_full["Monetary"] / rfm_full["Frequency"]

features = ["Recency", "Frequency", "Monetary", "AOV", "BasketSize", "Tenure"]
for col in features:
    rfm_full[f"{col}_capped"] = rfm_full[col].clip(upper=rfm_full[col].quantile(0.99))
rfm_log = rfm_full[[f"{col}_capped" for col in features]].copy()
for col in ["Frequency_capped", "Monetary_capped", "AOV_capped",
            "BasketSize_capped", "Tenure_capped"]:
    rfm_log[col] = np.log1p(rfm_log[col])
X_rfm = StandardScaler().fit_transform(rfm_log)
del rfm_log; gc.collect()

K_RANGE     = list(range(2, 9))
inertia_rfm, sil_rfm = [], []
for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = km.fit_predict(X_rfm)
    inertia_rfm.append(float(km.inertia_))
    sil_rfm.append(float(silhouette_score(X_rfm, labels)))

best_k   = 4
best_sil = sil_rfm[K_RANGE.index(best_k)]

km_rfm              = KMeans(n_clusters=best_k, random_state=42, n_init=3)
rfm_full["Cluster"] = km_rfm.fit_predict(X_rfm)
db_score  = float(davies_bouldin_score(X_rfm, rfm_full["Cluster"]))
ari_score = float(adjusted_rand_score(
    rfm_full["Cluster"],
    KMeans(n_clusters=best_k, random_state=99, n_init=3).fit_predict(X_rfm)
))

profile  = rfm_full.groupby("Cluster")[features].mean().round(1)
rfm_rank = (profile["Recency"].rank(ascending=True) +
            profile["Frequency"].rank(ascending=False) +
            profile["Monetary"].rank(ascending=False))

def _seg_label(rank):
    t = np.percentile(rfm_rank, [25, 50, 75])
    if rank <= t[0]:   return "Champions"
    elif rank <= t[1]: return "Loyal Customers"
    elif rank <= t[2]: return "At-Risk"
    else:              return "Lapsed / Low-Value"

profile["Segment"]  = [_seg_label(rfm_rank[i]) for i in profile.index]
rfm_full["Segment"] = rfm_full["Cluster"].map(profile["Segment"])

rfm_full["Rev_Contribution"] = rfm_full["Monetary"]
seg_revenue = rfm_full.groupby("Segment")["Monetary"].sum().reset_index()
seg_revenue.columns = ["Segment", "TotalRevenue"]
total_rev   = float(seg_revenue["TotalRevenue"].sum())
seg_revenue["RevPct"] = (seg_revenue["TotalRevenue"] / total_rev * 100).round(1)

pca        = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_rfm)
rfm_full["PC1"] = pca_coords[:, 0]
rfm_full["PC2"] = pca_coords[:, 1]
pca_var    = [float(v) for v in pca.explained_variance_ratio_ * 100]

rfm_full["DBSCAN"]  = DBSCAN(eps=0.8, min_samples=5).fit_predict(X_rfm).astype(str)
n_dbscan_clusters   = int(len(set(rfm_full["DBSCAN"].unique()) - {"-1"}))
n_dbscan_noise      = int((rfm_full["DBSCAN"] == "-1").sum())

# Drop capped cols — not needed in callbacks
rfm_full.drop(columns=[c for c in rfm_full.columns if c.endswith("_capped")], inplace=True)
del X_rfm, pca_coords; gc.collect()

# ── Transaction Clustering ─────────────────────────────────
print("Transaction clustering...")
X_txn_raw = np.column_stack([
    np.log1p(df_txn["Quantity"]),
    np.log1p(df_txn["UnitPrice"]),
    df_txn["Hour"] / 23.0,
    df_txn["DayCode"] / 6.0,
])
X_txn_scaled = StandardScaler().fit_transform(X_txn_raw)
del X_txn_raw; gc.collect()

np.random.seed(42)
idx_s = np.random.choice(len(X_txn_scaled), size=min(30_000, len(X_txn_scaled)), replace=False)
sil_t = [float(silhouette_score(X_txn_scaled[idx_s],
                                KMeans(n_clusters=k, random_state=42, n_init=3)
                                .fit_predict(X_txn_scaled[idx_s])))
         for k in range(2, 8)]
best_k_txn = list(range(2, 8))[int(np.argmax(sil_t))]
df_txn["TxnCluster"] = (KMeans(n_clusters=best_k_txn, random_state=42, n_init=3)
                        .fit_predict(X_txn_scaled).astype(str))
del X_txn_scaled, idx_s; gc.collect()

# ── Save everything ────────────────────────────────────────
print("Saving parquet files...")

df_raw_dedup.to_parquet("data/df_raw_dedup.parquet", index=False)
df_txn.to_parquet("data/df_txn.parquet", index=False)
df_cust.to_parquet("data/df_cust.parquet", index=False)
rfm_full.to_parquet("data/rfm_full.parquet", index=False)
abc_all.to_parquet("data/abc_all.parquet", index=False)
abc_summary.to_parquet("data/abc_summary.parquet", index=False)
missing_by_country.to_parquet("data/missing_by_country.parquet", index=False)
seg_revenue.to_parquet("data/seg_revenue.parquet", index=False)

# Cluster profile needs special handling (index = cluster ID)
profile.reset_index().to_parquet("data/profile.parquet", index=False)

meta = {
    "best_k": best_k, "best_sil": best_sil,
    "db_score": db_score, "ari_score": ari_score,
    "best_k_txn": best_k_txn,
    "n_dbscan_clusters": n_dbscan_clusters, "n_dbscan_noise": n_dbscan_noise,
    "pca_var": pca_var, "total_rev": total_rev,
    "inertia_rfm": inertia_rfm, "sil_rfm": sil_rfm,
    "K_RANGE": K_RANGE,
    "ALL_COUNTRIES": sorted(df_cust["Country"].unique().tolist()),
    "MIN_DATE": str(df_cust["InvoiceDate"].min().date()),
    "MAX_DATE": str(df_cust["InvoiceDate"].max().date()),
}
with open("data/meta.json", "w") as f:
    json.dump(meta, f)

print("\nDone! Files saved to data/:")
import os
for fn in sorted(os.listdir("data")):
    size = os.path.getsize(f"data/{fn}") / 1024 / 1024
    print(f"  {fn:35s}  {size:.1f} MB")
