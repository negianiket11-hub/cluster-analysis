# Retail Customer Cluster Analysis

An end-to-end customer segmentation project built on the [UCI Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail). Includes a Jupyter notebook for full analysis and an interactive Dash dashboard for exploration.

---

## Project Structure

```
cluster-analysis/
├── Online Retail(1).csv   # Raw dataset (541,909 transactions)
├── main.ipynb             # Analysis notebook
├── dashboard.py           # Interactive Dash dashboard
├── precompute.py          # Run once locally to generate data/ files
├── requirements.txt       # Python dependencies
├── data/                  # Pre-computed parquet files (~8MB total)
│   ├── df_cust.parquet
│   ├── df_txn.parquet
│   ├── df_raw_dedup.parquet
│   ├── rfm_full.parquet
│   ├── profile.parquet
│   ├── seg_revenue.parquet
│   ├── abc_all.parquet
│   ├── abc_summary.parquet
│   ├── missing_by_country.parquet
│   └── meta.json
└── README.md
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — Online Retail II
- **Period:** December 2010 – December 2011
- **Scope:** UK-based online retailer selling gift/homeware items
- **Raw rows:** 541,909 transaction line items
- **Cleaned rows:** ~400,000 (after removing cancellations, nulls, bad prices)

---

## Notebook (`main.ipynb`)

Runs a full analysis pipeline across 37 code cells:

### Data Cleaning
- Remove duplicate rows, cancelled invoices (`InvoiceNo` starting with `C`), zero/negative Quantity or UnitPrice, and rows with missing Description or CustomerID
- Transaction-level dataset (`df_transactions`) is saved before dropping missing CustomerIDs so anonymous transactions are still analysed

### EDA
- Distribution plots for Quantity, UnitPrice, Revenue
- Box plots with 99th percentile capping
- Top 15 products by revenue
- ABC / Pareto analysis (A = top 80% revenue, B = next 15%, C = bottom 5%)
- Top 10 countries by revenue
- Monthly revenue trend
- AOV and basket size over time
- Revenue by day of week and hour of day
- Customer behaviour distributions (basket size, revenue per invoice, customer lifetime)
- Monthly cancellation rate

### RFM Feature Engineering
Six customer-level features computed from `df_clean`:

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of unique invoices |
| Monetary | Total spend (GBP) |
| AOV | Average order value (Monetary / Frequency) |
| BasketSize | Average units per transaction line item |
| Tenure | Days between first and last purchase |

All features are 99th-percentile capped, log-transformed (except Recency), and StandardScaled before clustering.

### Customer Clustering (K-Means)
- Silhouette, Inertia, and Davies-Bouldin scores evaluated for k = 2–10
- **k = 4 forced** (silhouette auto-selects k = 2 which is too coarse for CRM use)
- Cluster stability tested via Adjusted Rand Index across 10 random seeds
- Segments labelled using quartile thresholds on composite RFM rank:

| Segment | Profile |
|---|---|
| **Champions** | Low recency, high frequency, high spend |
| **Loyal Customers** | Regular buyers, solid spend, headroom to grow |
| **At-Risk** | Previously active, recency rising, intervention needed |
| **Lapsed / Low-Value** | Long inactive, low historical engagement |

### Visualisations
- Cluster profiles (bar charts for all 6 features)
- Cluster stability (ARI bar chart)
- PCA 2D projection
- DBSCAN density clustering (eps = 0.8, validated)
- Hierarchical dendrogram (Ward linkage, 300-customer sample)
- Recency vs Monetary and Frequency vs Monetary scatter plots
- **Radar chart** — all 4 segments on 6 normalised axes (scaled to [0.1, 0.9] so weakest cluster remains visible)

### Transaction Clustering
- Features: log(Quantity), log(UnitPrice), normalised Hour, normalised DayCode
- Includes all transactions (with and without CustomerID)
- Best k selected by silhouette score on a 30,000-row sample
- Profiles show avg quantity, price, revenue, hour, and % missing CustomerID per cluster

---

## Dashboard (`dashboard.py`)

Built with **Dash + Plotly + dash-bootstrap-components**.

### Running Locally

```bash
pip install -r requirements.txt
python dashboard.py
```

Then open **http://127.0.0.1:8050** in your browser.

> The `data/` folder is already committed — no need to run `precompute.py` unless you change the raw data.

### Tab 1 — Overview
- Monthly revenue trend
- Revenue by country (top 10)
- Top 15 products by revenue
- Revenue heatmap (day of week × hour)
- ABC / Pareto chart (dual axis: bar + cumulative %)
- Monthly cancellation rate

### Tab 2 — Customer Segments
- Validation KPIs: k, Silhouette score, Davies-Bouldin score, ARI
- Segment filter dropdown (ALL / Champions / Loyal Customers / At-Risk / Lapsed)
- Cluster profiles table
- Radar chart (6-axis, all 4 segments)
- Scatter plots: Recency vs Monetary, Frequency vs Monetary
- PCA 2D projection
- DBSCAN cluster view
- RFM feature distributions
- Revenue contribution donut by segment
- Segment size vs revenue bubble chart
- **Segment Deep Dive card** — per-segment KPI strip (8 metrics vs store average), behavioural profile, and 5 specific CRM actions

### Tab 3 — Transaction Segments
- Transaction scatter (Quantity vs Unit Price, log scale)
- Transaction cluster profiles
- Missing CustomerID analysis by cluster
- Revenue donut (known vs anonymous)
- Timing by cluster

### Tab 4 — EDA Deep Dive
- Box plots (Quantity, Unit Price, Revenue)
- AOV trend over time
- Basket size trend over time
- Customer behaviour distributions
- Missing CustomerID by country

---

## Deploying on Render (Free Tier)

1. Go to **render.com** → New → Web Service → connect this GitHub repo
2. Set **Build Command:** `pip install -r requirements.txt`
3. Set **Start Command:** `python dashboard.py`
4. Set **Instance Type:** Free
5. Click Deploy

The `data/` folder is pre-committed so Render loads ~8MB of parquet files at startup instead of reprocessing the raw CSV. This keeps startup RAM well under Render's 512MB free tier limit.

> If you update the raw data, run `python precompute.py` locally and push the new `data/` files to GitHub before redeploying.

---

## `precompute.py`

Runs the full data pipeline (cleaning, RFM, K-Means, DBSCAN, PCA, transaction clustering) and saves results to `data/` as parquet files. Only needs to be re-run if the raw dataset changes.

```bash
python precompute.py
```

---

## Key Technical Decisions

| Decision | Reason |
|---|---|
| Force k=4 | Silhouette peaks at k=2 (too coarse); k=4 gives standard retail CRM model |
| 99th pct capping + log transform | Wholesale bulk orders dominate cluster geometry without this |
| Radar scaled to [0.1, 0.9] | Weakest cluster collapses to invisible point at origin with [0, 1] scaling |
| ABC at 80%/95% | Standard Pareto principle; consistent between notebook and dashboard |
| DBSCAN eps=0.8 fixed | Auto percentile on k-distance graph was unreliable for this dataset |
| Transaction clustering excludes Revenue | Revenue = Quantity × Price introduces multicollinearity |
| Pre-computed parquet files | Avoids 700MB RAM spike from processing raw CSV on deployment |

---

## Requirements

```
pandas
numpy
scikit-learn
scipy
plotly
dash
dash-bootstrap-components
pyarrow
```
