"""
Retail Customer Cluster Analysis  —  Streamlit Dashboard
Run:  streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Retail Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
PALETTE   = ["#2563EB","#059669","#D97706","#7C3AED","#DB2777","#0891B2","#65A30D","#EA580C"]
BG        = "#F1F5F9"
CARD_BG   = "#FFFFFF"
BORDER    = "#E2E8F0"
TEXT_DARK = "#0F172A"
TEXT_MID  = "#475569"
TEXT_LITE = "#94A3B8"
ACCENT    = "#2563EB"
GREEN     = "#059669"
AMBER     = "#D97706"
RED       = "#DC2626"

BASE_LAYOUT = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=12),
    margin=dict(l=12, r=12, t=44, b=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font_size=11),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER,
                    font=dict(family="'Inter','Segoe UI',sans-serif", size=12, color=TEXT_DARK)),
)
AXIS_STYLE = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
                  linecolor=BORDER, tickfont=dict(size=11))

def _apply_layout(fig, xtitle="", ytitle="", title=""):
    fig.update_layout(**BASE_LAYOUT,
                      title=dict(text=title, font_size=13, x=0, xanchor="left"))
    fig.update_xaxes(**AXIS_STYLE, title_text=xtitle, title_font_size=11)
    fig.update_yaxes(**AXIS_STYLE, title_text=ytitle, title_font_size=11)
    return fig

def _rgba(hex_col, alpha):
    h = hex_col.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Load pre-computed data (cached across reruns) ───────────────────────────────
@st.cache_data
def load_data():
    df_raw_dedup       = pd.read_parquet("data/df_raw_dedup.parquet")
    df_txn             = pd.read_parquet("data/df_txn.parquet")
    df_cust            = pd.read_parquet("data/df_cust.parquet")
    rfm_full           = pd.read_parquet("data/rfm_full.parquet")
    abc_all            = pd.read_parquet("data/abc_all.parquet")
    abc_summary        = pd.read_parquet("data/abc_summary.parquet")
    missing_by_country = pd.read_parquet("data/missing_by_country.parquet")
    seg_revenue        = pd.read_parquet("data/seg_revenue.parquet")
    profile_df         = pd.read_parquet("data/profile.parquet")
    profile            = profile_df.set_index("Cluster")
    with open("data/meta.json") as f:
        meta = json.load(f)
    return (df_raw_dedup, df_txn, df_cust, rfm_full, abc_all, abc_summary,
            missing_by_country, seg_revenue, profile, meta)

(df_raw_dedup, df_txn, df_cust, rfm_full, abc_all, abc_summary,
 missing_by_country, seg_revenue, profile, _meta) = load_data()

best_k            = _meta["best_k"]
best_sil          = _meta["best_sil"]
db_score          = _meta["db_score"]
ari_score         = _meta["ari_score"]
best_k_txn        = _meta["best_k_txn"]
n_dbscan_clusters = _meta["n_dbscan_clusters"]
n_dbscan_noise    = _meta["n_dbscan_noise"]
pca_var           = _meta["pca_var"]
total_rev         = _meta["total_rev"]
inertia_rfm       = _meta["inertia_rfm"]
sil_rfm           = _meta["sil_rfm"]
K_RANGE           = _meta["K_RANGE"]
ALL_COUNTRIES     = _meta["ALL_COUNTRIES"]
MIN_DATE          = pd.to_datetime(_meta["MIN_DATE"]).date()
MAX_DATE          = pd.to_datetime(_meta["MAX_DATE"]).date()
ALL_SEGMENTS      = ["Champions", "Loyal Customers", "At-Risk", "Lapsed / Low-Value"]

rfm_rank = (profile["Recency"].rank(ascending=True) +
            profile["Frequency"].rank(ascending=False) +
            profile["Monetary"].rank(ascending=False))

# ── CRM strategy content ───────────────────────────────────────────────────────
_SEG_STRATEGY = {
    "Champions": {
        "icon": "★", "color": PALETTE[0],
        "behaviour": (
            "Champions are your highest-value customers. They purchased very recently, "
            "order frequently, and spend significantly more per transaction. Their AOV and "
            "basket size exceed the store average. Tenure is long — they have been loyal for "
            "over a year and show no signs of lapsing."
        ),
        "crm": [
            "Enrol in an exclusive VIP loyalty tier with early-access product drops and private sales.",
            "Invite to a referral or brand-ambassador programme — Champions' word-of-mouth is high-value.",
            "Use personalised 'thank you' comms (handwritten notes, personalised packaging) to deepen emotional loyalty.",
            "Test premium upsell: curated bundles, gift-wrapping, or premium postage at checkout.",
            "Monitor closely — any rise in Recency is an early churn signal; trigger a re-engagement touch before 60 days.",
        ],
    },
    "Loyal Customers": {
        "icon": "◆", "color": PALETTE[1],
        "behaviour": (
            "Loyal Customers are consistent, reliable buyers with above-average frequency. "
            "Their Recency is moderate — they return regularly but not as quickly as Champions. "
            "Spend per transaction is solid but there is headroom to grow AOV. "
            "They represent the most stable segment for baseline revenue forecasting."
        ),
        "crm": [
            "Run a 'graduation to Champions' programme: offer a points multiplier or bonus reward once they hit one more purchase threshold.",
            "Cross-sell complementary product categories to grow basket size — they already trust the brand.",
            "Deploy category-specific email campaigns based on past purchase history to increase AOV.",
            "Introduce a loyalty points scheme with visible progress tracking to drive repeat visits.",
            "Avoid heavy discounting — these customers already buy at full price; margin leakage is unnecessary.",
        ],
    },
    "At-Risk": {
        "icon": "▲", "color": PALETTE[2],
        "behaviour": (
            "At-Risk customers were once active buyers but their Recency has grown — they are drifting away. "
            "Their historical frequency and spend are moderate, indicating they had genuine intent to buy. "
            "The window to re-engage is narrow: the longer Recency grows, the harder re-activation becomes. "
            "AOV tends to be lower than Loyal, suggesting price sensitivity or browsing behaviour."
        ),
        "crm": [
            "Send a time-boxed 'We miss you' campaign within 90 days of last purchase with a meaningful incentive (15-20% off or free shipping).",
            "Use browse/purchase history to personalise the re-engagement offer — generic discounts underperform.",
            "Trigger a satisfaction survey to diagnose if churn was caused by product, delivery, or pricing issues.",
            "If no response after two touches, move to a lower-cost nurture sequence (monthly newsletter) rather than burning budget on non-responders.",
            "Suppress from expensive paid channels until re-engaged — focus budget on email and SMS.",
        ],
    },
    "Lapsed / Low-Value": {
        "icon": "●", "color": PALETTE[3],
        "behaviour": (
            "Lapsed / Low-Value customers last purchased a long time ago and had low frequency and spend even "
            "when active. Many may be one-time buyers or opportunistic purchasers responding to a past promotion. "
            "They are the largest segment by headcount but contribute the smallest share of revenue. "
            "Re-activation cost is high relative to expected lifetime value."
        ),
        "crm": [
            "Run one time-boxed win-back email sequence (2-3 touches max) before sunsetting the segment.",
            "If they respond, treat them as At-Risk and move to the appropriate nurture track.",
            "If no response, suppress from all paid and email channels to reduce cost-per-contact.",
            "Focus any re-engagement creative on your best-selling or highest-rated products — give the best first impression.",
            "Analyse what originally acquired them (source, campaign) to avoid re-acquiring similar profiles in future paid campaigns.",
        ],
    },
}

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 26px; font-weight: 800; }
[data-testid="stMetricDelta"] { font-size: 12px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 14px; }
div[data-testid="column"] > div { height: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛒 Retail Analytics")
    st.caption("UK Online Retailer  |  Dec 2010 – Dec 2011")
    st.markdown("---")
    st.subheader("Global Filters")
    country = st.selectbox("Country", ["ALL"] + ALL_COUNTRIES, index=0)
    date_range = st.date_input(
        "Date Range",
        value=[MIN_DATE, MAX_DATE],
        min_value=MIN_DATE,
        max_value=MAX_DATE,
    )
    start_date = date_range[0] if len(date_range) >= 1 else MIN_DATE
    end_date   = date_range[1] if len(date_range) == 2 else MAX_DATE
    st.markdown("---")
    st.markdown(f"**Model Summary**")
    st.caption(f"RFM clusters: k = {best_k}")
    st.caption(f"Silhouette score: {best_sil:.4f}")
    st.caption(f"Davies-Bouldin: {db_score:.4f}")
    st.caption(f"ARI stability: {ari_score:.4f}")
    st.caption(f"Txn clusters: k = {best_k_txn}")
    st.caption(f"DBSCAN clusters: {n_dbscan_clusters}  |  noise: {n_dbscan_noise}")

# ── Filter helpers ─────────────────────────────────────────────────────────────
def filter_cust():
    mc = df_cust["InvoiceDate"].dt.date.between(start_date, end_date)
    if country != "ALL":
        mc &= df_cust["Country"] == country
    return df_cust[mc].copy()

def filter_raw():
    mr = df_raw_dedup["InvoiceDate"].dt.date.between(start_date, end_date)
    if country != "ALL":
        mr &= df_raw_dedup["Country"] == country
    return df_raw_dedup[mr].copy()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "👤  Customer Segments",
    "🛍  Transaction Segments",
    "🔬  EDA Deep Dive",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    dc = filter_cust()
    dr = filter_raw()

    # KPI strip
    rev    = dc["Revenue"].sum()
    orders = dc["InvoiceNo"].nunique()
    custs  = int(dc["CustomerID"].nunique())
    aov    = rev / orders if orders else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue",    f"£{rev:,.0f}",   "Gross sales (GBP)")
    k2.metric("Unique Orders",    f"{orders:,}",     "Invoices placed")
    k3.metric("Known Customers",  f"{custs:,}",      "With CustomerID")
    k4.metric("Avg. Order Value", f"£{aov:,.2f}",    "Revenue per order")

    st.markdown("---")

    # Monthly revenue | Revenue by country
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly Revenue Trend")
        m = dc.groupby("Month")["Revenue"].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=m["Month"], y=m["Revenue"], mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
            marker=dict(size=6, color=ACCENT),
            hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: £%{y:,.0f}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Month", ytitle="Total Revenue (GBP £)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Revenue by Country (Top 10)")
        top = dc.groupby("Country")["Revenue"].sum().nlargest(10).reset_index()
        fig = go.Figure(go.Bar(
            x=top["Revenue"], y=top["Country"], orientation="h",
            marker=dict(color=top["Revenue"], colorscale="Blues", showscale=False),
            hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.0f}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Total Revenue (GBP £)")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Top products | Day × Hour heatmap
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top 15 Products by Revenue")
        top = dc.groupby("Description")["Revenue"].sum().nlargest(15).reset_index()
        fig = go.Figure(go.Bar(
            x=top["Revenue"], y=top["Description"], orientation="h",
            marker=dict(color=top["Revenue"], colorscale="Teal", showscale=False),
            hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.0f}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Total Revenue (GBP £)", ytitle="Product")
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont_size=10)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Revenue Heatmap — Day × Hour")
        days  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = dc.pivot_table(values="Revenue", index="DayOfWeek",
                               columns="Hour", aggfunc="sum")
        pivot = pivot.reindex(days).reindex(columns=list(range(24))).fillna(0)
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=[f"{h:02d}:00" for h in pivot.columns],
            y=pivot.index.tolist(), colorscale="Blues",
            hovertemplate="<b>%{y}  %{x}</b><br>Revenue: £%{z:,.0f}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Hour of Day (24-hour clock)", ytitle="Day of Week")
        fig.update_xaxes(showgrid=False, tickangle=0, tickfont_size=10)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ABC Pareto | Cancellation rate
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("ABC / Pareto Analysis")
        a = (dc.groupby("Description")["Revenue"].sum()
             .sort_values(ascending=False).reset_index())
        a["CumPct"] = a["Revenue"].cumsum() / a["Revenue"].sum() * 100
        top50 = a.head(50).copy()
        top50["Rank"] = range(1, len(top50) + 1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=top50["Rank"], y=top50["Revenue"],
            marker=dict(color=top50["CumPct"],
                        colorscale=[[0, PALETTE[0]], [0.8, PALETTE[2]], [1, RED]],
                        showscale=False),
            name="Revenue (GBP £)",
            hovertemplate="<b>Rank %{x}</b><br>%{customdata[0]}<br>Revenue: £%{y:,.0f}<extra></extra>",
            customdata=top50[["Description"]].values,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=top50["Rank"], y=top50["CumPct"], mode="lines",
            name="Cumulative %", line=dict(color=RED, width=2, dash="dot"),
            hovertemplate="Rank %{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)
        for pct, lbl, dash in [(80, "A/B (80%)", "dash"), (95, "B/C (95%)", "dot")]:
            fig.add_trace(go.Scatter(
                x=[1, len(top50)], y=[pct, pct], mode="lines",
                line=dict(color=AMBER, width=1.5, dash=dash),
                name=lbl, showlegend=True,
            ), secondary_y=True)
        fig.update_layout(**BASE_LAYOUT, barmode="overlay", height=360)
        fig.update_xaxes(**AXIS_STYLE, title_text="Product Rank (sorted by Revenue)")
        fig.update_yaxes(title_text="Revenue (GBP £)", secondary_y=False, **AXIS_STYLE)
        fig.update_yaxes(title_text="Cumulative Revenue (%)", secondary_y=True,
                         range=[0, 105], showgrid=False, tickfont=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("Monthly Cancellation Rate")
        m = dr.groupby("Month").agg(
            Total    =("InvoiceNo",    "count"),
            Cancelled=("IsCancelled",  "sum"),
        ).reset_index()
        m["CancelRate"] = m["Cancelled"] / m["Total"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=m["Month"], y=m["CancelRate"],
            marker=dict(color=m["CancelRate"], colorscale="Reds", showscale=False),
            hovertemplate="<b>%{x|%b %Y}</b><br>Cancellation Rate: %{y:.1f}%<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Month", ytitle="Cancellation Rate (%)")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # Segment filter
    seg_options = ["ALL"] + ALL_SEGMENTS
    seg = st.selectbox("Filter by Segment", seg_options, index=0, key="seg_filter")

    # Validation KPIs
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Clusters (k)",       str(best_k))
    v2.metric("Silhouette Score",   f"{best_sil:.4f}")
    v3.metric("Davies-Bouldin",     f"{db_score:.4f}")
    v4.metric("ARI Stability",      f"{ari_score:.4f}")

    st.markdown("---")

    # Cluster profiles table | Radar chart
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Cluster Profiles")
        features = ["Recency", "Frequency", "Monetary", "AOV", "BasketSize", "Tenure"]
        disp = profile[features + ["Segment"]].copy().round(1)
        disp.index.name = "Cluster"
        st.dataframe(disp, use_container_width=True, height=210)

        st.subheader("Cluster Feature Profiles")
        metrics = [
            ("Recency",    PALETTE[0]),
            ("Frequency",  PALETTE[1]),
            ("Monetary",   PALETTE[2]),
            ("AOV",        PALETTE[3]),
            ("BasketSize", PALETTE[4]),
        ]
        x_labels = [f"Cluster {i}\n{profile.loc[i, 'Segment']}" for i in profile.index]
        fig = go.Figure()
        for col, color in metrics:
            opacities = [1.0 if (seg == "ALL" or profile.loc[i, "Segment"] == seg) else 0.15
                         for i in profile.index]
            fig.add_trace(go.Bar(
                name=col, x=x_labels, y=profile[col],
                marker=dict(color=color, opacity=opacities),
                hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.1f}}<extra></extra>",
            ))
        _apply_layout(fig, xtitle="Customer Segment",
                      ytitle="Mean Value (days | invoices | GBP | units)")
        fig.update_layout(barmode="group", height=320,
                          xaxis=dict(showgrid=False, tickfont_size=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Segment Radar Chart")
        cols = ["Recency", "Frequency", "Monetary", "AOV", "BasketSize", "Tenure"]
        cats = ["Recency (inv)", "Frequency", "Monetary", "AOV", "BasketSize", "Tenure"]
        radar = profile[cols].copy()
        radar["Recency"] = radar["Recency"].max() - radar["Recency"]
        rng = radar.max() - radar.min()
        rng[rng == 0] = 1
        radar_norm = ((radar - radar.min()) / rng) * 0.8 + 0.1
        fig = go.Figure()
        for ci, (idx, row) in enumerate(radar_norm.iterrows()):
            seg_name = str(profile.loc[idx, "Segment"])
            selected = (seg == "ALL" or seg_name == seg)
            color    = PALETTE[ci % len(PALETTE)]
            r_vals   = row.tolist() + [row.tolist()[0]]
            t_vals   = cats + [cats[0]]
            fig.add_trace(go.Scatterpolar(
                r=r_vals, theta=t_vals, fill="toself", name=seg_name,
                line=dict(color=color, width=3 if selected else 1),
                fillcolor=_rgba(color, 0.28 if selected else 0.05),
                hovertemplate="<b>" + seg_name + "</b><br>%{theta}: %{r:.2f}<extra></extra>",
            ))
        fig.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=11),
            polar=dict(
                bgcolor=CARD_BG,
                radialaxis=dict(visible=True, range=[0, 1],
                                showticklabels=False, gridcolor=BORDER),
                angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
            ),
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=55, r=55, t=30, b=60),
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Scatter plots
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Recency vs Monetary")
        fig = go.Figure()
        if seg == "ALL":
            for ci, s in enumerate(ALL_SEGMENTS):
                d = rfm_full[rfm_full["Segment"] == s]
                fig.add_trace(go.Scatter(
                    x=d["Recency"], y=d["Monetary"], mode="markers", name=s,
                    marker=dict(size=4, color=PALETTE[ci], opacity=0.65),
                    hovertemplate="<b>" + s + "</b><br>Recency: %{x}d<br>Spend: £%{y:,.0f}<extra></extra>",
                ))
        else:
            bg = rfm_full[rfm_full["Segment"] != seg]
            fg = rfm_full[rfm_full["Segment"] == seg]
            fig.add_trace(go.Scatter(x=bg["Recency"], y=bg["Monetary"], mode="markers",
                                     name="Other", marker=dict(size=3, color="rgba(148,163,184,0.3)"),
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fg["Recency"], y=fg["Monetary"], mode="markers",
                                     name=seg, marker=dict(size=5, color=ACCENT, opacity=0.75),
                                     hovertemplate="Recency: %{x}d<br>Spend: £%{y:,.0f}<extra></extra>"))
        _apply_layout(fig, xtitle="Recency (days)", ytitle="Total Spend (£)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Frequency vs Monetary")
        fig = go.Figure()
        if seg == "ALL":
            for ci, s in enumerate(ALL_SEGMENTS):
                d = rfm_full[rfm_full["Segment"] == s]
                fig.add_trace(go.Scatter(
                    x=d["Frequency"], y=d["Monetary"], mode="markers", name=s,
                    marker=dict(size=4, color=PALETTE[ci], opacity=0.65),
                    hovertemplate="<b>" + s + "</b><br>Freq: %{x}<br>Spend: £%{y:,.0f}<extra></extra>",
                ))
        else:
            bg = rfm_full[rfm_full["Segment"] != seg]
            fg = rfm_full[rfm_full["Segment"] == seg]
            fig.add_trace(go.Scatter(x=bg["Frequency"], y=bg["Monetary"], mode="markers",
                                     name="Other", marker=dict(size=3, color="rgba(148,163,184,0.3)"),
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fg["Frequency"], y=fg["Monetary"], mode="markers",
                                     name=seg, marker=dict(size=5, color=GREEN, opacity=0.75),
                                     hovertemplate="Frequency: %{x}<br>Spend: £%{y:,.0f}<extra></extra>"))
        _apply_layout(fig, xtitle="Frequency (unique invoices)", ytitle="Total Spend (£)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    # PCA | DBSCAN
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("PCA 2D Projection")
        fig = go.Figure()
        if seg == "ALL":
            for ci, s in enumerate(ALL_SEGMENTS):
                d = rfm_full[rfm_full["Segment"] == s]
                fig.add_trace(go.Scatter(
                    x=d["PC1"], y=d["PC2"], mode="markers", name=s,
                    marker=dict(size=4, color=PALETTE[ci], opacity=0.65),
                    hovertemplate="<b>" + s + "</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                ))
        else:
            bg = rfm_full[rfm_full["Segment"] != seg]
            fg = rfm_full[rfm_full["Segment"] == seg]
            fig.add_trace(go.Scatter(x=bg["PC1"], y=bg["PC2"], mode="markers",
                                     name="Other", marker=dict(size=3, color="rgba(148,163,184,0.3)"),
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fg["PC1"], y=fg["PC2"], mode="markers",
                                     name=seg, marker=dict(size=5, color=ACCENT, opacity=0.75),
                                     hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"))
        _apply_layout(fig,
                      xtitle=f"PC1 ({pca_var[0]:.1f}% variance)",
                      ytitle=f"PC2 ({pca_var[1]:.1f}% variance)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("DBSCAN Density Clustering")
        fig = go.Figure()
        cluster_ids = sorted(rfm_full["DBSCAN"].unique(), key=lambda x: int(x))
        for ci, cl in enumerate(cluster_ids):
            d        = rfm_full[rfm_full["DBSCAN"] == cl]
            is_noise = (cl == "-1")
            fig.add_trace(go.Scatter(
                x=d["PC1"], y=d["PC2"], mode="markers",
                name="Noise" if is_noise else f"DBSCAN {cl}",
                marker=dict(size=3 if is_noise else 4,
                            color="#94A3B8" if is_noise else PALETTE[ci % len(PALETTE)],
                            opacity=0.4 if is_noise else 0.7),
                hovertemplate=("Noise<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>" if is_noise else
                               f"Cluster {cl}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"),
            ))
        _apply_layout(fig,
                      xtitle=f"PC1 ({pca_var[0]:.1f}% variance)",
                      ytitle=f"PC2 ({pca_var[1]:.1f}% variance)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Revenue donut | Bubble chart
    c7, c8 = st.columns(2)
    with c7:
        st.subheader("Revenue Contribution by Segment")
        df_sr = seg_revenue.copy()
        pull  = [0.06 if r["Segment"] == seg else 0 for _, r in df_sr.iterrows()]
        fig   = go.Figure(go.Pie(
            labels=df_sr["Segment"], values=df_sr["TotalRevenue"], pull=pull,
            marker=dict(colors=PALETTE[:len(df_sr)], line=dict(color=BG, width=2)),
            texttemplate="%{label}<br><b>%{percent}</b>",
            hovertemplate="<b>%{label}</b><br>Revenue: £%{value:,.0f}<br>Share: %{percent}<extra></extra>",
            hole=0.52,
        ))
        fig.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=11),
            margin=dict(l=12, r=12, t=12, b=12),
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08,
                        font_size=10, bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(text=f"£{total_rev/1e6:.1f}M<br>Total",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=13, color=TEXT_DARK))],
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        st.subheader("Segment Size vs Revenue")
        seg_counts = rfm_full.groupby("Segment").agg(
            CustomerCount=("Monetary", "count"),
            TotalRevenue =("Monetary", "sum"),
            AvgRecency   =("Recency",  "mean"),
            AvgFrequency =("Frequency","mean"),
            AvgAOV       =("AOV",      "mean"),
        ).reset_index().round(1)
        seg_counts = seg_counts.merge(seg_revenue[["Segment","RevPct"]], on="Segment", how="left")
        fig = go.Figure()
        for ci, row in seg_counts.iterrows():
            is_sel = (row["Segment"] == seg)
            fig.add_trace(go.Scatter(
                x=[row["CustomerCount"]], y=[row["TotalRevenue"]],
                mode="markers+text",
                marker=dict(size=max(18, row["RevPct"] * 2.2),
                            color=PALETTE[ci % len(PALETTE)],
                            opacity=1.0 if is_sel else 0.55,
                            line=dict(width=2 if is_sel else 0, color=TEXT_DARK)),
                text=[row["Segment"]], textposition="top center",
                textfont=dict(size=10, color=TEXT_DARK),
                name=row["Segment"],
                hovertemplate=(
                    f"<b>{row['Segment']}</b><br>"
                    f"Customers: {row['CustomerCount']:,}<br>"
                    f"Revenue: £{row['TotalRevenue']:,.0f}<br>"
                    f"Rev share: {row['RevPct']}%<br>"
                    f"Avg Recency: {row['AvgRecency']:.0f}d<br>"
                    f"Avg AOV: £{row['AvgAOV']:.2f}<extra></extra>"
                ),
            ))
        _apply_layout(fig, xtitle="Number of Customers", ytitle="Total Revenue (£)")
        fig.update_layout(showlegend=False, margin=dict(l=12, r=12, t=12, b=48), height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # RFM distributions
    st.subheader("RFM Feature Distributions")
    d = rfm_full if seg == "ALL" else rfm_full[rfm_full["Segment"] == seg]
    cfg_dist = [
        ("Recency",   d["Recency"],             ACCENT, "Recency (days since last purchase)"),
        ("Frequency", np.log1p(d["Frequency"]), GREEN,  "log(1 + Frequency)  [invoices]"),
        ("Monetary",  np.log1p(d["Monetary"]),  AMBER,  "log(1 + Monetary)  [GBP £]"),
    ]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[c[3] for c in cfg_dist],
                        horizontal_spacing=0.09)
    for i, (name, vals, color, xt) in enumerate(cfg_dist, 1):
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=40, marker_color=color, opacity=0.8,
                         name=name, showlegend=False,
                         hovertemplate=f"Value: %{{x:.2f}}<br>Count: %{{y}}<extra>{name}</extra>"),
            row=1, col=i)
        fig.update_xaxes(title_text=xt, title_font_size=10,
                         showgrid=True, gridcolor=BORDER, zeroline=False,
                         linecolor=BORDER, tickfont_size=10, row=1, col=i)
        fig.update_yaxes(title_text="No. of Customers" if i == 1 else "",
                         showgrid=True, gridcolor=BORDER, zeroline=False,
                         linecolor=BORDER, tickfont_size=10, row=1, col=i)
    fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=11),
                      margin=dict(l=12, r=12, t=44, b=12), height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Segment deep dive card
    st.subheader("Segment Deep Dive")
    if seg == "ALL":
        # Summary comparison table
        rows = []
        for s in ALL_SEGMENTS:
            sd       = rfm_full[rfm_full["Segment"] == s]
            rev_row  = seg_revenue[seg_revenue["Segment"] == s]
            rev_val  = rev_row["TotalRevenue"].values[0] if len(rev_row) else 0
            rev_pct  = rev_row["RevPct"].values[0] if len(rev_row) else 0
            info     = _SEG_STRATEGY[s]
            rows.append({
                "Segment":       f"{info['icon']} {s}",
                "Customers":     f"{len(sd):,}",
                "Revenue":       f"£{rev_val/1e3:,.0f}K",
                "Rev %":         f"{rev_pct}%",
                "Avg Recency":   f"{sd['Recency'].mean():.0f}d",
                "Avg Frequency": f"{sd['Frequency'].mean():.1f}",
                "Avg AOV":       f"£{sd['AOV'].mean():.2f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Segment"), use_container_width=True)
    else:
        info    = _SEG_STRATEGY[seg]
        sd      = rfm_full[rfm_full["Segment"] == seg]
        rev_row = seg_revenue[seg_revenue["Segment"] == seg]
        rev_val = rev_row["TotalRevenue"].values[0] if len(rev_row) else 0
        rev_pct = rev_row["RevPct"].values[0] if len(rev_row) else 0

        avg = rfm_full[["Recency","Frequency","Monetary","AOV","BasketSize","Tenure"]].mean()

        def delta_str(val, avg_val, lower_better=False):
            diff = val - avg_val
            pct  = diff / avg_val * 100 if avg_val else 0
            better = (diff < 0) if lower_better else (diff > 0)
            arrow  = "▲" if diff > 0 else "▼"
            sign   = "+" if diff > 0 else ""
            return f"{arrow} {sign}{pct:.0f}% vs avg", "normal" if not better else "normal"

        st.markdown(f"### {info['icon']} {seg}")
        m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
        m1.metric("Customers",    f"{len(sd):,}",
                  f"{len(sd)/len(rfm_full)*100:.1f}% of base")
        m2.metric("Revenue",      f"£{rev_val/1e3:,.0f}K",
                  f"{rev_pct}% of total")
        rec_d, _ = delta_str(sd["Recency"].mean(), avg["Recency"], lower_better=True)
        m3.metric("Avg Recency",  f"{sd['Recency'].mean():.0f}d",    rec_d)
        frq_d, _ = delta_str(sd["Frequency"].mean(), avg["Frequency"])
        m4.metric("Avg Frequency",f"{sd['Frequency'].mean():.1f}",   frq_d)
        mon_d, _ = delta_str(sd["Monetary"].mean(), avg["Monetary"])
        m5.metric("Avg Spend",    f"£{sd['Monetary'].mean():,.0f}",  mon_d)
        aov_d, _ = delta_str(sd["AOV"].mean(), avg["AOV"])
        m6.metric("Avg AOV",      f"£{sd['AOV'].mean():.2f}",        aov_d)
        bsk_d, _ = delta_str(sd["BasketSize"].mean(), avg["BasketSize"])
        m7.metric("Basket Size",  f"{sd['BasketSize'].mean():.1f}",  bsk_d)
        ten_d, _ = delta_str(sd["Tenure"].mean(), avg["Tenure"])
        m8.metric("Avg Tenure",   f"{sd['Tenure'].mean():.0f}d",     ten_d)

        st.markdown("**Behavioural Profile**")
        st.info(info["behaviour"])
        st.markdown("**CRM Action Plan**")
        for i, action in enumerate(info["crm"], 1):
            st.markdown(f"{i}. {action}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRANSACTION SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Transaction Scatter — Quantity vs Unit Price")
    st.caption("Sample of 8,000 transactions, log scale on both axes")
    sample = df_txn.sample(min(8_000, len(df_txn)), random_state=42)
    fig = go.Figure()
    for ci, cl in enumerate(sorted(sample["TxnCluster"].unique())):
        d = sample[sample["TxnCluster"] == cl]
        fig.add_trace(go.Scatter(
            x=d["Quantity"], y=d["UnitPrice"], mode="markers",
            name=f"Cluster {cl}",
            marker=dict(size=4, color=PALETTE[ci % len(PALETTE)], opacity=0.55),
            hovertemplate=f"<b>Cluster {cl}</b><br>Qty: %{{x}}<br>Price: £%{{y:.2f}}<br>"
                          "Revenue: £%{customdata[0]:,.2f}<extra></extra>",
            customdata=d[["Revenue"]].values,
        ))
    _apply_layout(fig, xtitle="Quantity (units)", ytitle="Unit Price (£)")
    fig.update_xaxes(type="log", dtick=1)
    fig.update_yaxes(type="log", dtick=1)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transaction Cluster Profiles")
        prof = df_txn.groupby("TxnCluster").agg(
            Avg_Quantity  =("Quantity",  "mean"),
            Avg_UnitPrice =("UnitPrice", "mean"),
            Avg_Revenue   =("Revenue",   "mean"),
        ).reset_index().round(2)
        metrics_t = [
            ("Avg_Quantity",  "Avg. Quantity (units)", PALETTE[0]),
            ("Avg_UnitPrice", "Avg. Unit Price (£)",   PALETTE[1]),
            ("Avg_Revenue",   "Avg. Revenue (£)",      PALETTE[2]),
        ]
        fig = go.Figure()
        for col, name, color in metrics_t:
            fig.add_trace(go.Bar(
                name=name, x=[f"Cluster {c}" for c in prof["TxnCluster"]],
                y=prof[col], marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2f}}<extra></extra>",
            ))
        _apply_layout(fig, xtitle="Transaction Cluster", ytitle="Average Value")
        fig.update_layout(barmode="group", height=340,
                          xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Missing CustomerID by Cluster")
        grp = df_txn.groupby("TxnCluster").agg(
            Known  =("CustomerID", lambda x: x.notna().sum()),
            Missing=("CustomerID", lambda x: x.isna().sum()),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Known CustomerID",
            x=[f"Cluster {c}" for c in grp["TxnCluster"]],
            y=grp["Known"], marker_color=ACCENT,
            hovertemplate="<b>%{x}</b><br>Known: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Missing CustomerID",
            x=[f"Cluster {c}" for c in grp["TxnCluster"]],
            y=grp["Missing"], marker_color="#F87171",
            hovertemplate="<b>%{x}</b><br>Missing: %{y:,}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Transaction Cluster", ytitle="Transaction Lines")
        fig.update_layout(barmode="stack", height=340,
                          xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Revenue Share by Cluster")
        rev = df_txn.groupby("TxnCluster")["Revenue"].sum().reset_index()
        fig = go.Figure(go.Pie(
            labels=[f"Cluster {c}" for c in rev["TxnCluster"]],
            values=rev["Revenue"], hole=0.48,
            marker=dict(colors=PALETTE[:len(rev)], line=dict(color=CARD_BG, width=2)),
            textinfo="percent+label", textfont_size=12,
            hovertemplate="<b>Cluster %{label}</b><br>Revenue: £%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=12),
            margin=dict(l=12, r=12, t=30, b=12), height=340,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="v", font_size=11),
            annotations=[dict(text="Revenue<br>Share", x=0.5, y=0.5,
                              font_size=12, font_color=TEXT_MID, showarrow=False)],
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Transaction Timing by Cluster")
        timing = df_txn.groupby("TxnCluster").agg(
            Avg_Hour   =("Hour",    "mean"),
            Avg_DayCode=("DayCode", "mean"),
        ).reset_index().round(2)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Avg. Hour of Day", "Avg. Day of Week"])
        fig.add_trace(go.Bar(
            x=[f"Cluster {c}" for c in timing["TxnCluster"]],
            y=timing["Avg_Hour"], marker_color=PALETTE[0],
            hovertemplate="<b>%{x}</b><br>Avg. Hour: %{y:.1f}:00<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[f"Cluster {c}" for c in timing["TxnCluster"]],
            y=timing["Avg_DayCode"], marker_color=PALETTE[1],
            hovertemplate="<b>%{x}</b><br>Avg. DayCode: %{y:.1f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                          font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=11),
                          margin=dict(l=12, r=12, t=44, b=12), height=340)
        for col_i in [1, 2]:
            fig.update_xaxes(showgrid=False, tickfont_size=10, row=1, col=col_i)
            fig.update_yaxes(showgrid=True, gridcolor=BORDER, tickfont_size=10, row=1, col=col_i)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    dc4 = filter_cust()

    # Box plots
    st.subheader("Distribution Box Plots")
    cfg_box = [
        ("Quantity",  "Quantity (units per transaction line)", PALETTE[0]),
        ("UnitPrice", "Unit Price (GBP £ per unit)",          PALETTE[1]),
        ("Revenue",   "Revenue (GBP £ per transaction line)", PALETTE[2]),
    ]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[c[1] for c in cfg_box],
                        horizontal_spacing=0.08)
    for i, (col, label, color) in enumerate(cfg_box, 1):
        cap  = dc4[col].quantile(0.99)
        vals = dc4[col].clip(upper=cap)
        fig.add_trace(go.Box(
            y=vals, name=label, marker_color=color,
            boxmean="sd", showlegend=False,
            hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=i)
        fig.update_yaxes(title_text=label, title_font_size=10,
                         showgrid=True, gridcolor=BORDER, zeroline=False,
                         linecolor=BORDER, tickfont_size=10, row=1, col=i)
        fig.update_xaxes(showgrid=False, tickfont_size=10, row=1, col=i)
    fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=11),
                      margin=dict(l=12, r=12, t=44, b=12), height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # AOV trend | Basket size trend
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("AOV Trend Over Time")
        m = dc4.groupby("Month").agg(
            Revenue=("Revenue",  "sum"),
            Orders =("InvoiceNo","nunique"),
        ).reset_index()
        m["AOV"] = m["Revenue"] / m["Orders"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=m["Month"], y=m["AOV"], mode="lines+markers",
            line=dict(color=GREEN, width=2.5),
            fill="tozeroy", fillcolor="rgba(5,150,105,0.07)",
            marker=dict(size=6, color=GREEN),
            hovertemplate="<b>%{x|%b %Y}</b><br>AOV: £%{y:,.2f}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Month", ytitle="Average Order Value (£)")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Basket Size Trend Over Time")
        inv_qty = dc4.groupby(["Month", "InvoiceNo"])["Quantity"].sum().reset_index()
        m = inv_qty.groupby("Month")["Quantity"].mean().reset_index()
        m.columns = ["Month", "AvgBasket"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=m["Month"], y=m["AvgBasket"], mode="lines+markers",
            line=dict(color=AMBER, width=2.5),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.07)",
            marker=dict(size=6, color=AMBER),
            hovertemplate="<b>%{x|%b %Y}</b><br>Avg. Basket: %{y:.1f} units<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Month", ytitle="Avg. Basket Size (units per order)")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Customer behaviour distributions
    st.subheader("Customer Behaviour Distributions")
    rfm_seg_filter = st.selectbox(
        "Filter segment", ["ALL"] + ALL_SEGMENTS, key="eda_seg_filter"
    )
    d_eda = rfm_full if rfm_seg_filter == "ALL" else rfm_full[rfm_full["Segment"] == rfm_seg_filter]
    cfg_beh = [
        ("BasketSize", d_eda["BasketSize"].clip(upper=d_eda["BasketSize"].quantile(0.99)),
         PALETTE[0], "Avg. Basket Size (units per transaction line)"),
        ("AOV",        d_eda["AOV"].clip(upper=d_eda["AOV"].quantile(0.99)),
         PALETTE[1], "Avg. Order Value (GBP £)"),
        ("Tenure",     d_eda["Tenure"],
         PALETTE[2], "Customer Tenure (days between first & last purchase)"),
    ]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[c[3] for c in cfg_beh],
                        horizontal_spacing=0.09)
    for i, (name, vals, color, xt) in enumerate(cfg_beh, 1):
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=40, marker_color=color, opacity=0.8,
                         name=name, showlegend=False,
                         hovertemplate=f"Value: %{{x:.2f}}<br>Count: %{{y}}<extra>{name}</extra>"),
            row=1, col=i)
        fig.update_xaxes(title_text=xt, title_font_size=10,
                         showgrid=True, gridcolor=BORDER, zeroline=False,
                         linecolor=BORDER, tickfont_size=10, row=1, col=i)
        fig.update_yaxes(title_text="No. of Customers" if i == 1 else "",
                         showgrid=True, gridcolor=BORDER, zeroline=False,
                         linecolor=BORDER, tickfont_size=10, row=1, col=i)
    fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                      font=dict(family="'Inter','Segoe UI',sans-serif", color=TEXT_DARK, size=11),
                      margin=dict(l=12, r=12, t=44, b=12), height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Missing CustomerID by country
    st.subheader("Missing CustomerID by Country (top 15 countries by volume)")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=missing_by_country["MissingPct"],
        y=missing_by_country["Country"],
        orientation="h",
        marker=dict(color=missing_by_country["MissingPct"],
                    colorscale="Oranges", showscale=False),
        hovertemplate="<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>",
    ))
    _apply_layout(fig, xtitle="Missing CustomerID (%)", ytitle="Country")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
