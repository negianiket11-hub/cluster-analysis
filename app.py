"""
Retail Customer Cluster Analysis  —  Streamlit Dashboard
Run:  python -m streamlit run app.py
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

# ── Insight panel helper ────────────────────────────────────────────────────────
def insight_panel(points, actions=None):
    pts = "".join(
        f'<li style="font-size:12px;color:{TEXT_DARK};margin-bottom:3px;line-height:1.55">{p}</li>'
        for p in points
    )
    html = (
        f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;'
        f'border-left:4px solid #2563EB;border-radius:10px;'
        f'padding:14px 16px;margin-top:10px;margin-bottom:8px">'
        f'<div style="margin-bottom:8px">'
        f'<span style="font-size:15px">💡</span>'
        f'<span style="font-weight:700;font-size:13px;color:#1D4ED8;margin-left:7px">Key Insights</span>'
        f'</div>'
        f'<ul style="padding-left:18px;margin:0">{pts}</ul>'
        f'</div>'
    )
    if actions:
        acts = "".join(
            f'<li style="font-size:12px;color:{TEXT_DARK};margin-bottom:3px;line-height:1.55">{a}</li>'
            for a in actions
        )
        html += (
            f'<div style="background:#ECFDF5;border:1px solid #A7F3D0;'
            f'border-left:4px solid #059669;border-radius:10px;'
            f'padding:14px 16px;margin-bottom:10px">'
            f'<div style="margin-bottom:8px">'
            f'<span style="font-size:15px">⚡</span>'
            f'<span style="font-weight:700;font-size:13px;color:#065F46;margin-left:7px">Actionable Recommendations</span>'
            f'</div>'
            f'<ul style="padding-left:18px;margin:0">{acts}</ul>'
            f'</div>'
        )
    return html

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

# ── Static insight content ──────────────────────────────────────────────────────
INS = {
    "monthly": insight_panel(
        points=[
            "Revenue grows steadily from Jan–Oct 2011 then surges sharply in November — the single largest revenue month — driven by early Christmas gift purchasing from the B2B wholesale customer base.",
            "January and February record the lowest revenue of the year, a predictable post-holiday trough as retailers have already stocked up for Christmas and are now clearing unsold inventory.",
            "December 2011 appears artificially low because the dataset ends on 09 Dec 2011, not at month end — full-month December would likely exceed November given typical seasonal momentum.",
            "The year-on-year growth trajectory (Dec 2010 vs. Dec 2011) suggests the business was in an active growth phase, not a steady-state operation.",
        ],
        actions=[
            "Place bulk inventory orders by September so stock is warehoused before the November demand spike — late procurement risks stockouts on top SKUs during the highest-revenue window.",
            "Launch a 'Beat the Rush' email campaign in early October targeting existing customers, offering early ordering incentives to smooth demand and reduce November logistics pressure.",
            "Use January as a strategic re-engagement window: send lapsed customers a 'New Year, New Stock' offer — the psychological freshness of January increases open rates for re-engagement emails.",
            "Budget marketing spend seasonally, not equally: allocate 40–50% of annual promotional spend to the Sep–Nov window when ROI on spend is highest.",
        ]
    ),
    "country": insight_panel(
        points=[
            "The United Kingdom dominates with ~85–90% of total revenue — this business is fundamentally UK-centric, making international revenue a secondary but growing stream.",
            "The top 5 international markets (Netherlands, Germany, France, Ireland, Australia) each represent meaningful pockets of demand, likely from established wholesale relationships.",
            "The long tail of 30+ countries contributes negligible individual revenue — many are likely one-off orders or test transactions rather than established customer relationships.",
            "The strong European presence (NL, DE, FR, IE) suggests the product range has natural cross-border appeal for home décor and gifting, aligning with EU consumer tastes.",
        ],
        actions=[
            "Invest in dedicated account management for the top 4 international markets (NL, DE, FR, IE) — these are established wholesale relationships worth protecting with service-level agreements.",
            "Audit the bottom 20 countries by revenue: calculate shipping cost-to-serve vs. revenue generated — countries where shipping exceeds 30% of order value are likely loss-making and should be discontinued.",
            "Launch a localised German and French catalogue with translated product names and EU-compliant packaging — removing language barriers can materially increase conversion in these markets.",
            "Explore a Netherlands-based distribution hub to service NL, DE, FR, BE simultaneously — reducing delivery times and cost for the second-most-important regional market cluster.",
        ]
    ),
    "products": insight_panel(
        points=[
            "The top 15 products by revenue are overwhelmingly decorative, seasonal, and giftable — confirming this is a gift and home décor wholesaler, not a commodity goods supplier.",
            "High-revenue products are not always high-volume — premium items with higher unit prices can rank in the top 15 on revenue despite selling fewer units, indicating a mixed pricing strategy.",
            "Many top products share thematic families (e.g. 'JUMBO BAG', 'LUNCH BAG', 'RETROSPOT') — suggesting customers order in product families, which is exploitable for bundle promotions.",
            "Seasonal items dominate the top revenue list, reinforcing the Q4 concentration risk — if a top seasonal product goes out of stock in October, there is no equivalent substitute.",
        ],
        actions=[
            "Create a 'Never Out of Stock' policy for the top 20 revenue products — set minimum stock thresholds at 150% of prior-year Q4 peak demand to buffer against supply chain delays.",
            "Design themed product bundles around the top-selling families (e.g. 'Retrospot Collection', 'Jumbo Bag Bundle') — bundling increases AOV without discounting individual items.",
            "Develop a 'New Season Drop' communication series for the top seasonal products each September — pre-selling creates committed orders and cash flow ahead of peak production.",
            "Analyse purchase co-occurrence: identify which products are frequently bought together and use this to power 'Frequently Bought Together' recommendations on the order portal.",
        ]
    ),
    "heatmap": insight_panel(
        points=[
            "Tuesday, Wednesday, and Thursday are the highest-revenue days — strongly B2B purchase behaviour, with buyers placing orders mid-week to receive deliveries before the weekend.",
            "Saturday generates near-zero revenue, and Sunday is completely absent — this confirms the customer base is wholesale/trade buyers, not end consumers who shop on weekends.",
            "The 10:00–14:00 window accounts for the vast majority of daily revenue — consistent with office-hours purchasing by buyers who check email and place orders during the working morning.",
            "Orders virtually cease after 17:00, reinforcing the business-hours-only purchasing pattern — automated order processing and notifications outside these hours is low-priority.",
        ],
        actions=[
            "Time all marketing emails (promotions, new stock alerts, seasonal campaigns) to land in inboxes by 08:30 on Tuesday or Wednesday — this maximises the chance of being seen during the morning ordering window.",
            "Staff the customer service and order processing team at full capacity Mon–Thu 09:00–15:00; allow reduced staffing on Fridays and close on weekends without service impact.",
            "Schedule replenishment reminders and low-stock alerts to fire automatically at 09:00 Mon–Thu — buyers are most likely to act on these during the working morning.",
            "Consider piloting a small B2C channel (weekend flash sales, consumer-direct website) to monetise the completely unused Saturday and Sunday time window with a different customer base.",
        ]
    ),
    "abc": insight_panel(
        points=[
            "The Pareto principle holds strongly: a small fraction of SKUs (~20% of products, Class A) generates approximately 80% of total revenue — the business is highly SKU-concentrated.",
            "Class B products (the next 15% of revenue, roughly products ranked 21–50% of catalogue) are the 'support tier' — important in aggregate but individually replaceable if they go out of stock.",
            "Class C products (the long tail, ~5% of revenue despite potentially comprising 60–70% of SKUs) exist but their management cost — procurement, storage, cataloguing — may exceed their revenue contribution.",
            "The steep initial drop in the Pareto curve shows revenue is not just concentrated but hyper-concentrated: the top 5–10 products alone may account for 30–40% of total revenue.",
        ],
        actions=[
            "Implement a 'Class A Priority Protocol': weekly stock checks, dedicated supplier relationships, and automatic reorder triggers for Class A SKUs — a 1-week stockout on a Class A product can wipe out a month of margin.",
            "Apply demand forecasting only to Class A and B products; use a simple safety-stock formula (average demand × 2 weeks) for the long tail instead.",
            "Conduct an annual Class C product review: any SKU that has generated less than £500 in the past 12 months and is not a strategic listing should be evaluated for discontinuation.",
            "Use ABC tiering to negotiate supplier terms: Class A SKUs justify dedicated supplier SLAs with guaranteed lead times, while Class C SKUs can be sourced opportunistically at spot prices.",
        ]
    ),
    "cancel_rate": insight_panel(
        points=[
            "The cancellation rate chart reveals the month-by-month proportion of orders that were cancelled — a key operational health metric that is invisible in the standard revenue view.",
            "Months where the cancellation rate spikes above the average warrant investigation: common causes include overselling a stockout item, pricing errors, or a supplier delivery failure causing bulk order cancellations.",
            "A rising cancellation trend across multiple consecutive months is an early warning signal of systemic operational problems — it will show up in cancellation rate before it visibly impacts revenue.",
            "Seasonally elevated cancellations in Q4 may reflect customers over-ordering 'just in case' and then cancelling excess quantities — a common pattern in gift-trade wholesale buying.",
        ],
        actions=[
            "Set a cancellation rate alert threshold (e.g. > 8% in any month) and trigger an automatic root-cause analysis report when it is breached — do not wait for the quarterly review.",
            "Cross-reference cancellation spikes against stock levels: if high-cancellation months align with stockouts on Class A products, the fix is inventory management, not CRM.",
            "Implement a cancellation reason capture field in the order management system — without knowing why customers cancel, operational fixes will be guesswork rather than targeted interventions.",
            "Analyse cancellations by customer segment: if Champions have elevated cancellations in a given month, this is far more urgent than the same rate among Lapsed customers.",
        ]
    ),
    "rfm_profile": insight_panel(
        points=[
            f"K-Means clustering on {len(rfm_full):,} customers using 6 behavioural features identified {best_k} segments: Champions, Loyal Customers, At-Risk, and Lapsed / Low-Value — the standard 4-tier retail CRM model.",
            "CHAMPIONS: Purchased very recently, order frequently, and have the highest lifetime spend and AOV. Long tenure confirms these are established relationships. They generate a disproportionate share of total revenue despite being a minority of customers.",
            "LOYAL CUSTOMERS: Regular buyers with solid spend and moderate recency. More valuable than At-Risk but not yet at Champion level. They have proven repeat-purchase behaviour and respond well to cross-sell and upsell campaigns. The goal is graduation to Champions.",
            "AT-RISK: Previously active customers whose recency is rising — they are starting to drift. Frequency and spend are moderate, but the gap since their last order is growing. Without intervention, these customers migrate to Lapsed within 60–90 days.",
            "LAPSED / LOW-VALUE: Last purchased a long time ago, ordered infrequently (often once or twice), low total spend. Short tenure — they made early purchases and then disengaged. The majority require a strong incentive to return; a subset will not respond regardless.",
        ],
        actions=[
            "CHAMPIONS: Enrol in a VIP loyalty tier (free priority shipping, early access to new ranges, dedicated account manager). Never use generic acquisition discounts — it devalues their loyalty. Launch a referral programme: offer account credit for every new customer they introduce.",
            "LOYAL CUSTOMERS: Focus on graduation to Champions. Introduce tiered volume incentives (e.g. '10 orders = next-tier pricing'). Send personalised cross-sell recommendations based on their purchase history. Monitor recency — if it starts rising, move them to the At-Risk workflow immediately.",
            "AT-RISK: The intervention window is narrow. Send a personalised re-engagement campaign within 30 days of their recency crossing the at-risk threshold, featuring the specific product categories they previously bought.",
            "LAPSED / LOW-VALUE: Run one time-boxed win-back offer (15% off, valid 30 days). If response rate < 5%, accept churn and remove from active marketing. Before writing them off, analyse what they initially bought — if it was a discontinued category, the issue is product, not customer.",
        ]
    ),
    "scatter": insight_panel(
        points=[
            "The Recency vs. Spend scatter reveals the fundamental customer lifecycle shape: Champions occupy the bottom-right quadrant (low recency = purchased recently, high spend), while Lapsed customers are in the top-left.",
            "The Frequency vs. Spend scatter shows a positive correlation within Champions — customers who order more frequently also spend more in total, confirming that frequency is a leading indicator of lifetime value.",
            "Extreme outliers in the top-right corner (very high spend and high frequency) are candidates for wholesale account status — they behave qualitatively differently from standard retail buyers and should be managed as key accounts.",
            "Cluster overlap in the scatter plots indicates customers near segment boundaries — these boundary customers are the highest-priority for proactive outreach as a small nudge can retain a customer who would otherwise migrate to a lower-value segment.",
        ],
        actions=[
            "Export the top-right outliers (highest Recency-Spend and Frequency-Spend scores) to a 'Key Account' list and assign dedicated account managers — wholesale buyers at this scale warrant bespoke relationship management.",
            "Set up a quarterly segment migration report: track how many customers moved from At-Risk → Champions vs. At-Risk → Lapsed. A worsening migration ratio is the earliest warning that CRM interventions are underperforming.",
            "For customers in the At-Risk cluster who sit closest to the Champions cluster in the scatter (boundary customers), trigger a personalised outreach call — these are the highest-ROI re-engagement targets because they need the least persuasion.",
            "Use the Frequency vs. Spend chart to identify 'Frequent but Low-Value' customers: these buyers order often but always in small quantities. An AOV uplift campaign (minimum order thresholds for free delivery) could significantly increase their revenue contribution.",
        ]
    ),
    "pca": insight_panel(
        points=[
            f"PCA compresses the 6 RFM features into 2 dimensions while retaining {pca_var[0]+pca_var[1]:.1f}% of the total statistical variance — meaning this 2D view preserves most of the meaningful differences between customers.",
            f"PC1 (explaining {pca_var[0]:.1f}% of variance) typically captures the primary driver of customer differentiation — in RFM contexts this is usually a combined 'overall engagement' axis where high values = highly engaged (recent, frequent, high-spend).",
            f"PC2 (explaining {pca_var[1]:.1f}% of variance) typically captures a secondary contrast — often distinguishing between customers who are frequent-but-low-spend vs. infrequent-but-high-spend.",
            "Tight, well-separated cluster blobs in the PCA plot provide geometric confirmation that the K-Means segments are not arbitrary — they represent genuinely different regions of the customer behaviour space.",
        ],
        actions=[
            "Run this PCA scatter every quarter using freshly recomputed RFM features. If the cluster blobs begin overlapping more than the previous quarter, the segmentation has degraded and the model needs retraining.",
            "Customers in the inter-cluster overlap zones should be flagged in the CRM as 'At Boundary' — assign them to the higher-value adjacent segment for treatment purposes.",
            "Use the PC1 axis as a single composite 'Customer Health Score': customers with PC1 below the median are at elevated churn risk and should enter an automated re-engagement nurture sequence.",
            f"If PC1 and PC2 together explain less than 60% of variance, consider adding the 4th and 5th principal components — the 2D view may be missing an important dimension of customer behaviour.",
        ]
    ),
    "dbscan": insight_panel(
        points=[
            f"DBSCAN identified {n_dbscan_clusters} high-density customer clusters and flagged {n_dbscan_noise} customers as noise points — customers who do not belong to any dense cluster.",
            "Unlike K-Means, DBSCAN does not force every customer into a cluster — noise points (-1) are genuine statistical outliers whose behaviour is too unusual or too sparse to form a group. This is a feature, not a flaw.",
            f"If DBSCAN's {n_dbscan_clusters} clusters align structurally with K-Means' {best_k} clusters, this cross-validation significantly increases confidence that the segmentation reflects real behavioural patterns rather than algorithmic artefacts.",
            "DBSCAN noise points in a customer dataset almost always represent wholesale/reseller accounts with atypical order patterns, customers who made a single anomalous bulk purchase, or data quality issues.",
        ],
        actions=[
            f"Manually review a sample of the {min(n_dbscan_noise, 50)} noise point customers: if more than 30% have lifetime spend > £10,000, they are likely wholesale key accounts and should be moved to a separate 'Key Account' segment.",
            "If noise points cluster tightly in the PCA plot, reduce DBSCAN's eps parameter — these customers may form a valid micro-segment (e.g. 'Occasional Bulk Buyers') worth targeting separately.",
            "Use DBSCAN as a data quality filter: noise points with implausibly high or low feature values are strong candidates for data investigation before inclusion in any marketing segmentation.",
            "Run DBSCAN quarterly alongside K-Means: a growing noise point count over time indicates the customer base is becoming more heterogeneous, which may eventually warrant adding a new K-Means cluster.",
        ]
    ),
    "rfm_dist": insight_panel(
        points=[
            "The Recency distribution is broad and approximately uniform across most of its range, with a spike at very low recency (recent purchasers) — revealing a two-speed customer base: active repeat buyers and a long dormant tail.",
            "Frequency distribution is extremely right-skewed: after log-transformation, the peak is still at low values, meaning the majority of customers have purchased only 1–3 times in the entire observation period.",
            "Monetary distribution is also heavily right-skewed even after log-transformation — the typical customer contributes a modest amount, while a small elite (the top 5–10% by spend) contributes an outsized share of total revenue.",
            "The shapes of these distributions are diagnostic: a healthy, growing customer base would show Frequency shifting right (more repeat buyers) and Recency shifting left (more recent purchasers) over time.",
        ],
        actions=[
            "The one-time buyer problem is the most impactful addressable opportunity: if even 20% of single-purchase customers can be converted to two purchases, Frequency skewness reduces and lifetime value for the cohort doubles. Launch a 'Second Purchase' automated email sequence triggered 30 days after a first-time buyer's initial order.",
            "Identify the Monetary top 5% and place them in a 'Platinum Account' watch list: set alerts if any of these customers have Recency > 60 days without a new order.",
            "For customers with Recency > 180 days, the probability of organic return is very low. Use a final automated win-back campaign, then mark as 'churned' and remove from active marketing.",
            "Track the median (not mean) of each RFM distribution quarterly — medians are resistant to outlier distortion and give a more accurate picture of the 'typical' customer's trajectory.",
        ]
    ),
    "boxplots": insight_panel(
        points=[
            "The Quantity box plot shows the median transaction line is 2–6 units, but the distribution has a very long upper tail — a small number of bulk order lines inflate the mean significantly, which is why median is a more honest measure of typical purchase size.",
            "The Unit Price box plot reveals most transactions fall in a modest price band (£1–£5 per unit, consistent with gift/décor wholesale), with outliers above £10–£20 representing either premium products or possible pricing errors.",
            "The Revenue per line box plot combines Quantity and Unit Price variance — its upper tail is the most extreme of the three, because both a high Quantity and a high Unit Price independently stretch the upper bound.",
            "The presence of extreme outliers in all three box plots is the reason log-transformation and 99th percentile capping are applied before clustering — without this preprocessing, a handful of wholesale bulk orders would dominate the cluster geometry.",
        ],
        actions=[
            "Define operational categories based on the box plot ranges: transactions below the 75th percentile Quantity are 'Standard Retail', above the 95th percentile are 'Bulk Orders' — route Bulk Orders to a specialist fulfilment team.",
            "Audit Unit Price outliers (transactions > 99th percentile Unit Price): verify these are legitimate product prices, not data entry errors (e.g. £500 entered when £5.00 was intended).",
            "Set dynamic stock alerts based on the Quantity distribution: if a customer's order contains a Quantity line item above the 90th percentile for that SKU, trigger a stock availability check before confirming the order.",
            "Use the Revenue per line distribution to inform trade discount thresholds: customers whose typical Revenue per line sits in the top quartile are natural candidates for a volume rebate scheme.",
        ]
    ),
    "aov_trend": insight_panel(
        points=[
            "AOV (Average Order Value = Total Revenue ÷ Number of Orders) trends upward from mid-2011 through Q4, peaking in November — buyers are not just placing more orders in peak season, they are also placing larger orders.",
            "The basket size trend (avg units per order) mirrors the AOV trend, confirming that AOV growth is driven by volume (more items per order), not by price increases — an important distinction when interpreting revenue growth.",
            "Months where AOV dips while order count holds steady indicate customers are splitting larger orders into smaller ones — a potential signal of cash-flow pressure among buyers, or a response to shipping cost structures.",
            "A rising AOV over time (year-on-year comparison) is a positive signal indicating customers are deepening their relationship with the supplier — ordering broader product ranges or moving to committed stocking orders.",
        ],
        actions=[
            "Set the free-delivery threshold just above the median AOV: if median AOV is £180, set the free delivery minimum at £200. This 'AOV nudge' encourages buyers to add one or two more items to qualify, lifting revenue per order without discounting.",
            "Track AOV by customer segment separately: Champions should show consistently higher AOV than the overall average. If Champions' AOV begins declining while order frequency holds, they may be splitting orders.",
            "In months where AOV drops significantly, cross-reference against any pricing changes, competitor promotions, or changes to shipping policy — AOV is a sensitive leading indicator of buyer response to commercial changes.",
            "Use basket size data to identify under-served product categories: if buyers consistently order 10+ units of Category A but only 1–2 units of Category B in the same order, B may have an availability or pricing barrier worth addressing.",
        ]
    ),
    "cust_dist": insight_panel(
        points=[
            "Basket Size distribution is right-skewed: most customers order in small quantities per line (1–5 units), but a tail of customers consistently order 10–50+ units per line — these are the wholesale bulk buyers who need separate commercial treatment.",
            "AOV distribution reveals the typical customer spends £100–£400 per order, but a meaningful tail of customers regularly places £1,000+ orders. These high-AOV customers are disproportionately valuable and should be treated as key accounts.",
            "Tenure distribution is bimodal: a large spike near 0 days (customers who made one purchase and never returned) and a second distribution of longer-tenure customers who have been active for 6–18 months.",
            "The gap between the one-and-done spike and the loyal customer distribution is the 'conversion valley' — customers who make a second purchase overwhelmingly go on to make a third, fourth, and more.",
        ],
        actions=[
            "Design a tiered wholesale account structure based on the Basket Size distribution: Standard Account (1–5 units/line), Trade Account (5–20 units/line), Wholesale Account (20+ units/line) — each tier gets different pricing, payment terms, and service levels.",
            "For the high-AOV tail customers: introduce a 'Preferred Buyer' programme with dedicated stock reservations, extended payment terms (net-30 or net-60), and a named account contact.",
            "Attack the one-and-done problem directly: identify all customers whose Tenure is < 30 days and who have made exactly one order. Send a personalised follow-up at day 21 with a 'Here's what other buyers like you ordered next' email.",
            "Track the percentage of customers crossing from Tenure = 0 (one-time buyer) to Tenure > 30 days (second purchase made) as a monthly KPI — this 'first-to-second conversion rate' is the most actionable metric for improving customer lifetime value.",
        ]
    ),
    "missing_country": insight_panel(
        points=[
            "Missing CustomerID rates vary dramatically by country — some markets capture customer identity on nearly every transaction, while others have rates above 50%, meaning the majority of their transactions are invisible to customer-level analytics.",
            "A high missing rate in a specific country almost always indicates a channel or system problem: that country may be served through a marketplace, a local reseller, or an older POS system that does not capture CustomerID.",
            "Countries with high missing rates have artificially deflated customer counts in the RFM analysis — their true customer base is larger than what the clustering captured, making customer-level insights for those markets statistically less reliable.",
            "The impact of missing CustomerIDs is not symmetric: a 30% missing rate means 30% of that market's revenue is unattributable to any customer, so lifetime value calculations, churn rates, and segment distributions are all understated.",
        ],
        actions=[
            "For the top 3 countries by missing CustomerID rate: raise a formal data quality incident with the responsible sales or IT team — this is a revenue attribution problem, not just an analytical inconvenience.",
            "Investigate whether high-missing-rate countries are served through intermediary channels (agents, marketplaces, distributors): if so, negotiate a data-sharing agreement that captures at minimum a hashed buyer identifier.",
            "Implement a lightweight customer registration incentive for markets with high missing rates: '5% off your next order when you create an account' — even a 20% uptake will substantially improve data quality within one quarter.",
            "Until the missing data problem is resolved, exclude high-missing-rate countries from the RFM segmentation results when making country-specific marketing decisions.",
        ]
    ),
    "txn_scatter": insight_panel(
        points=[
            f"Transaction-level K-Means clustering on {len(df_txn):,} individual order line items (including the ~25% with missing CustomerID) identified {best_k_txn} distinct purchase archetypes based on Quantity, Unit Price, Hour of Day, and Day of Week.",
            "High-Quantity / Low-Price clusters represent wholesale bulk ordering behaviour — large quantities of low-cost items (e.g. 48 units of a £1.25 gift bag). These have high operational impact (large pick-and-pack volumes) but moderate revenue per line.",
            "Low-Quantity / High-Price clusters represent premium retail purchasing behaviour — single or small quantities of higher-value items. These are often individual retail buyers making selective range additions, not bulk replenishment.",
            "The addition of temporal features (Hour and Day of Week) allows transaction archetypes to be defined not just by what was bought but when — enabling identification of time-specific purchase patterns.",
        ],
        actions=[
            "Develop cluster-specific fulfilment workflows: High-Quantity clusters require bulk pick-and-pack processes with pallet handling, while Low-Quantity / High-Price clusters warrant individual item inspection and premium packaging.",
            "Use transaction cluster membership as a real-time order routing signal: when an incoming order is classified as a Bulk cluster transaction, automatically route it to the trade fulfilment team.",
            "Analyse whether specific customers (CustomerIDs) consistently fall in the same transaction cluster across their order history — customers whose transaction cluster changes over time may be diversifying their purchasing behaviour.",
            "Cross-reference the temporal features of each cluster with delivery capacity: if the highest-volume transaction cluster is concentrated in Tuesday mornings, ensure logistics capacity is confirmed for Tuesday afternoon despatch.",
        ]
    ),
    "txn_missing": insight_panel(
        points=[
            "The stacked bar reveals which transaction clusters have the highest proportion of anonymous (missing CustomerID) transactions — a non-uniform distribution would indicate that certain purchase types are more likely to be made by unidentified buyers.",
            "If anonymous transactions concentrate disproportionately in the High-Quantity / Low-Price cluster, it suggests that bulk buyers (potentially trade customers using one-off purchase orders) are the primary source of the missing data problem.",
            "If anonymous transactions are spread uniformly across all clusters, the missing CustomerID is a systemic data capture failure at the order entry point (e.g. a checkout flow that allows guest purchasing without login).",
            "The ~25% overall missing CustomerID rate means a significant portion of the business's revenue base is completely invisible to CRM systems — these customers cannot be re-marketed to or tracked for churn.",
        ],
        actions=[
            "If anonymous transactions are concentrated in one or two clusters, prioritise CustomerID capture fixes for the product categories or order channels that generate those cluster types — a targeted fix is more cost-effective than a blanket system change.",
            "Introduce a post-purchase email capture for anonymous transactions: capturing an email address at despatch ('Enter your email for shipping updates') creates a retroactive link that can be used for future identification.",
            "Estimate the 'CRM blind spot' revenue: multiply the missing transaction count by the average revenue per transaction in that cluster — this £ figure should be used in the business case for fixing the data capture system.",
            "For anonymous transactions that do have a Country, use country + product combination as a probabilistic matching signal to attribute unidentified orders to known wholesale customers with reasonable confidence.",
        ]
    ),
}

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
    st.markdown("**Model Summary**")
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
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Revenue Trend")
        st.caption("Monthly gross revenue over the observation period")
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
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["monthly"], unsafe_allow_html=True)

    with c2:
        st.subheader("Revenue by Country")
        st.caption("Top 10 markets by gross revenue")
        top = dc.groupby("Country")["Revenue"].sum().nlargest(10).reset_index()
        fig = go.Figure(go.Bar(
            x=top["Revenue"], y=top["Country"], orientation="h",
            marker=dict(color=top["Revenue"], colorscale="Blues", showscale=False),
            hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.0f}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Total Revenue (GBP £)")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["country"], unsafe_allow_html=True)

    st.markdown("---")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Top 15 Products")
        st.caption("Products ranked by total revenue contribution")
        top = dc.groupby("Description")["Revenue"].sum().nlargest(15).reset_index()
        fig = go.Figure(go.Bar(
            x=top["Revenue"], y=top["Description"], orientation="h",
            marker=dict(color=top["Revenue"], colorscale="Teal", showscale=False),
            hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.0f}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Total Revenue (GBP £)", ytitle="Product")
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont_size=10)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["products"], unsafe_allow_html=True)

    with c4:
        st.subheader("Purchase Timing Heatmap")
        st.caption("Revenue intensity by day of week and hour of day")
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
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["heatmap"], unsafe_allow_html=True)

    st.markdown("---")

    c5, c6 = st.columns([7, 5])
    with c5:
        st.subheader("ABC / Pareto Analysis")
        st.caption("Pareto curve — cumulative revenue % by product rank")
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
        fig.update_layout(**BASE_LAYOUT, barmode="overlay", height=340)
        fig.update_xaxes(**AXIS_STYLE, title_text="Product Rank (sorted by Revenue)")
        fig.update_yaxes(title_text="Revenue (GBP £)", secondary_y=False, **AXIS_STYLE)
        fig.update_yaxes(title_text="Cumulative Revenue (%)", secondary_y=True,
                         range=[0, 105], showgrid=False, tickfont=dict(size=11))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["abc"], unsafe_allow_html=True)

    with c6:
        st.subheader("Monthly Cancellation Rate")
        st.caption("% of orders that were cancellations each month")
        m = dr.groupby("Month").agg(
            Total    =("InvoiceNo",   "count"),
            Cancelled=("IsCancelled", "sum"),
        ).reset_index()
        m["CancelRate"] = m["Cancelled"] / m["Total"] * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=m["Month"], y=m["CancelRate"],
            marker=dict(color=m["CancelRate"], colorscale="Reds", showscale=False),
            hovertemplate="<b>%{x|%b %Y}</b><br>Cancellation Rate: %{y:.1f}%<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Month", ytitle="Cancellation Rate (%)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["cancel_rate"], unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    seg_options = ["ALL"] + ALL_SEGMENTS
    seg = st.selectbox("Filter by Segment", seg_options, index=0, key="seg_filter")

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Clusters (k)",     str(best_k))
    v2.metric("Silhouette Score", f"{best_sil:.4f}")
    v3.metric("Davies-Bouldin",   f"{db_score:.4f}")
    v4.metric("ARI Stability",    f"{ari_score:.4f}")

    st.markdown("---")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Cluster Profiles")
        features = ["Recency", "Frequency", "Monetary", "AOV", "BasketSize", "Tenure"]
        disp = profile[features + ["Segment"]].copy().round(1)
        disp.index.name = "Cluster"
        st.dataframe(disp, use_container_width=True, height=210)

        st.subheader("Cluster Feature Bar Chart")
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
        st.markdown(INS["rfm_profile"], unsafe_allow_html=True)

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

    st.markdown(INS["scatter"], unsafe_allow_html=True)

    st.markdown("---")

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
        _apply_layout(fig, xtitle=f"PC1 ({pca_var[0]:.1f}% variance)",
                      ytitle=f"PC2 ({pca_var[1]:.1f}% variance)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["pca"], unsafe_allow_html=True)

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
        _apply_layout(fig, xtitle=f"PC1 ({pca_var[0]:.1f}% variance)",
                      ytitle=f"PC2 ({pca_var[1]:.1f}% variance)")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(INS["dbscan"], unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Revenue Contribution by Segment  &  Segment Size vs Revenue")
    c7, c8 = st.columns(2)
    with c7:
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

    st.subheader("RFM Feature Distributions")
    d = rfm_full if seg == "ALL" else rfm_full[rfm_full["Segment"] == seg]
    cfg_dist = [
        ("Recency",   d["Recency"],             ACCENT, "Recency (days since last purchase)"),
        ("Frequency", np.log1p(d["Frequency"]), GREEN,  "log(1 + Frequency)  [invoices]"),
        ("Monetary",  np.log1p(d["Monetary"]),  AMBER,  "log(1 + Monetary)  [GBP £]"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[c[3] for c in cfg_dist],
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
    st.markdown(INS["rfm_dist"], unsafe_allow_html=True)

    st.markdown("---")

    # Segment deep dive
    st.subheader("Segment Deep Dive")
    if seg == "ALL":
        rows = []
        for s in ALL_SEGMENTS:
            sd      = rfm_full[rfm_full["Segment"] == s]
            rev_row = seg_revenue[seg_revenue["Segment"] == s]
            rev_val = rev_row["TotalRevenue"].values[0] if len(rev_row) else 0
            rev_pct = rev_row["RevPct"].values[0] if len(rev_row) else 0
            info    = _SEG_STRATEGY[s]
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
        avg     = rfm_full[["Recency","Frequency","Monetary","AOV","BasketSize","Tenure"]].mean()

        def delta_str(val, avg_val, lower_better=False):
            diff  = val - avg_val
            pct   = diff / avg_val * 100 if avg_val else 0
            arrow = "▲" if diff > 0 else "▼"
            sign  = "+" if diff > 0 else ""
            return f"{arrow} {sign}{pct:.0f}% vs avg"

        st.markdown(f"### {info['icon']} {seg}")
        m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
        m1.metric("Customers",    f"{len(sd):,}",
                  f"{len(sd)/len(rfm_full)*100:.1f}% of base")
        m2.metric("Revenue",      f"£{rev_val/1e3:,.0f}K",
                  f"{rev_pct}% of total")
        m3.metric("Avg Recency",  f"{sd['Recency'].mean():.0f}d",
                  delta_str(sd["Recency"].mean(), avg["Recency"], lower_better=True))
        m4.metric("Avg Frequency",f"{sd['Frequency'].mean():.1f}",
                  delta_str(sd["Frequency"].mean(), avg["Frequency"]))
        m5.metric("Avg Spend",    f"£{sd['Monetary'].mean():,.0f}",
                  delta_str(sd["Monetary"].mean(), avg["Monetary"]))
        m6.metric("Avg AOV",      f"£{sd['AOV'].mean():.2f}",
                  delta_str(sd["AOV"].mean(), avg["AOV"]))
        m7.metric("Basket Size",  f"{sd['BasketSize'].mean():.1f}",
                  delta_str(sd["BasketSize"].mean(), avg["BasketSize"]))
        m8.metric("Avg Tenure",   f"{sd['Tenure'].mean():.0f}d",
                  delta_str(sd["Tenure"].mean(), avg["Tenure"]))

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
    st.caption(f"Sample of 8,000 from {len(df_txn):,} transaction lines, log scale on both axes")
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
    st.markdown(INS["txn_scatter"], unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transaction Cluster Profiles")
        prof = df_txn.groupby("TxnCluster").agg(
            Avg_Quantity  =("Quantity",  "mean"),
            Avg_UnitPrice =("UnitPrice", "mean"),
            Avg_Revenue   =("Revenue",   "mean"),
        ).reset_index().round(2)
        fig = go.Figure()
        for col, name, color in [
            ("Avg_Quantity",  "Avg. Quantity (units)", PALETTE[0]),
            ("Avg_UnitPrice", "Avg. Unit Price (£)",   PALETTE[1]),
            ("Avg_Revenue",   "Avg. Revenue (£)",      PALETTE[2]),
        ]:
            fig.add_trace(go.Bar(
                name=name, x=[f"Cluster {c}" for c in prof["TxnCluster"]],
                y=prof[col], marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2f}}<extra></extra>",
            ))
        _apply_layout(fig, xtitle="Transaction Cluster", ytitle="Average Value")
        fig.update_layout(barmode="group", height=320, xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Missing CustomerID by Cluster")
        grp = df_txn.groupby("TxnCluster").agg(
            Known  =("CustomerID", lambda x: x.notna().sum()),
            Missing=("CustomerID", lambda x: x.isna().sum()),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Known", x=[f"Cluster {c}" for c in grp["TxnCluster"]],
            y=grp["Known"], marker_color=ACCENT,
            hovertemplate="<b>%{x}</b><br>Known: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Missing", x=[f"Cluster {c}" for c in grp["TxnCluster"]],
            y=grp["Missing"], marker_color="#F87171",
            hovertemplate="<b>%{x}</b><br>Missing: %{y:,}<extra></extra>",
        ))
        _apply_layout(fig, xtitle="Transaction Cluster", ytitle="Transaction Lines")
        fig.update_layout(barmode="stack", height=320, xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(INS["txn_missing"], unsafe_allow_html=True)

    st.markdown("---")

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
            margin=dict(l=12, r=12, t=30, b=12), height=320,
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
                          margin=dict(l=12, r=12, t=44, b=12), height=320)
        for col_i in [1, 2]:
            fig.update_xaxes(showgrid=False, tickfont_size=10, row=1, col=col_i)
            fig.update_yaxes(showgrid=True, gridcolor=BORDER, tickfont_size=10, row=1, col=col_i)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    dc4 = filter_cust()

    st.subheader("Distribution Box Plots")
    st.caption("99th-percentile capped to suppress extreme outliers")
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
    st.markdown(INS["boxplots"], unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("AOV Trend Over Time")
        m = dc4.groupby("Month").agg(
            Revenue=("Revenue",   "sum"),
            Orders =("InvoiceNo", "nunique"),
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
        fig.update_layout(height=300)
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
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(INS["aov_trend"], unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Customer Behaviour Distributions")
    rfm_seg_filter = st.selectbox(
        "Filter by segment", ["ALL"] + ALL_SEGMENTS, key="eda_seg_filter"
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
    fig = make_subplots(rows=1, cols=3, subplot_titles=[c[3] for c in cfg_beh],
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
    st.markdown(INS["cust_dist"], unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Missing CustomerID by Country")
    st.caption("Top 15 countries by transaction volume")
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
    st.markdown(INS["missing_country"], unsafe_allow_html=True)
