# =============================================================================
# myntra_analysis.py — Myntra E-Commerce Comprehensive Analysis
# Tools: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn), SQL (SQLite)
# Author: Sharat Laha
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import os
os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS  = ["#1A5276","#C0392B","#1D8348","#D68910","#7D3C98","#0E6655","#BA4A00"]
PALETTE = "Blues_d"

print("=" * 65)
print("  MYNTRA E-COMMERCE ANALYSIS")
print("=" * 65)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
customers = pd.read_csv("myntra_analysis/customers.csv", parse_dates=["signup_date"])
orders    = pd.read_csv("myntra_analysis/orders.csv",    parse_dates=["order_date"])
merged    = orders.merge(customers, on="customer_id", how="left")

print(f"    Customers : {len(customers):,}")
print(f"    Orders    : {len(orders):,}")
print(f"    Total Revenue: ₹{orders['price_paid'].sum():,.0f}")
print(f"    Avg Order Value: ₹{orders['price_paid'].mean():,.0f}")
print(f"    Return Rate: {orders['returned'].mean()*100:.1f}%")
print(f"    Avg Rating: {orders['rating'].mean():.2f} / 5.0")

# ── 2. SQL ANALYSIS ───────────────────────────────────────────────────────────
print("\n[2] SQL analysis...")
conn = sqlite3.connect(":memory:")
orders.to_sql("orders", conn, index=False, if_exists="replace")
customers.to_sql("customers", conn, index=False, if_exists="replace")

q_revenue = pd.read_sql("""
    SELECT category,
           COUNT(*)                          AS total_orders,
           ROUND(SUM(price_paid),0)          AS total_revenue,
           ROUND(AVG(price_paid),0)          AS avg_order_value,
           ROUND(AVG(discount_pct),1)        AS avg_discount,
           ROUND(AVG(rating),2)              AS avg_rating,
           ROUND(SUM(CASE WHEN returned=1 THEN 1.0 ELSE 0 END)*100/COUNT(*),1) AS return_rate
    FROM orders GROUP BY category ORDER BY total_revenue DESC
""", conn)

q_monthly = pd.read_sql("""
    SELECT strftime('%Y-%m', order_date) AS month,
           COUNT(*)                      AS orders,
           ROUND(SUM(price_paid),0)      AS revenue
    FROM orders GROUP BY month ORDER BY month
""", conn)

q_city = pd.read_sql("""
    SELECT city, COUNT(*) AS orders,
           ROUND(SUM(price_paid),0) AS revenue
    FROM orders GROUP BY city ORDER BY revenue DESC LIMIT 10
""", conn)

q_yoy = pd.read_sql("""
    SELECT strftime('%Y', order_date) AS year,
           COUNT(*)                   AS orders,
           ROUND(SUM(price_paid),0)   AS revenue
    FROM orders GROUP BY year ORDER BY year
""", conn)

print("\n    --- Revenue by Category ---")
print(q_revenue[["category","total_orders","total_revenue","avg_discount","return_rate"]].to_string(index=False))
print("\n    --- YoY Performance ---")
print(q_yoy.to_string(index=False))
conn.close()

# ── 3. EDA VISUALISATIONS ─────────────────────────────────────────────────────
print("\n[3] Generating visualisations...")

# Chart 1: Revenue by Category
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(q_revenue["category"], q_revenue["total_revenue"]/1e6,
               color=COLORS[:len(q_revenue)])
ax.bar_label(bars, labels=[f"₹{v:.1f}M" for v in q_revenue["total_revenue"]/1e6],
             padding=4, fontsize=9)
ax.set_title("Total Revenue by Category (2022–2024)", fontsize=13, fontweight="bold")
ax.set_xlabel("Revenue (₹ Millions)")
plt.tight_layout()
plt.savefig("outputs/01_revenue_by_category.png", dpi=150)
plt.close()
print("    Saved: 01_revenue_by_category.png")

# Chart 2: Monthly Revenue Trend
q_monthly["month_dt"] = pd.to_datetime(q_monthly["month"])
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(q_monthly["month_dt"], q_monthly["revenue"]/1e5,
        color=COLORS[0], linewidth=2)
ax.fill_between(q_monthly["month_dt"], q_monthly["revenue"]/1e5,
                alpha=0.15, color=COLORS[0])
ax.set_title("Monthly Revenue Trend (2022–2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("Revenue (₹ Lakhs)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/02_monthly_revenue_trend.png", dpi=150)
plt.close()
print("    Saved: 02_monthly_revenue_trend.png")

# Chart 3: Discount vs Revenue heatmap
disc_cat = orders.groupby(["category","discount_pct"])["price_paid"].sum().reset_index()
disc_pivot = disc_cat.pivot_table(index="category", columns="discount_pct",
                                   values="price_paid", aggfunc="sum")
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(disc_pivot/1e5, cmap="YlOrRd", ax=ax, linewidths=0.3,
            cbar_kws={"label": "Revenue (₹ Lakhs)"})
ax.set_title("Revenue Heatmap: Category vs Discount % (₹ Lakhs)", fontsize=13, fontweight="bold")
ax.set_xlabel("Discount %")
plt.tight_layout()
plt.savefig("outputs/03_discount_revenue_heatmap.png", dpi=150)
plt.close()
print("    Saved: 03_discount_revenue_heatmap.png")

# Chart 4: Rating distribution by category
fig, ax = plt.subplots(figsize=(10, 5))
rating_cat = orders.groupby("category")["rating"].mean().sort_values()
bars = ax.barh(rating_cat.index, rating_cat.values, color=COLORS)
ax.bar_label(bars, labels=[f"{v:.2f}" for v in rating_cat.values],
             padding=4, fontsize=9)
ax.set_xlim(0, 5.5)
ax.axvline(4.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax.set_title("Average Customer Rating by Category", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Rating (/ 5.0)")
plt.tight_layout()
plt.savefig("outputs/04_ratings_by_category.png", dpi=150)
plt.close()
print("    Saved: 04_ratings_by_category.png")

# Chart 5: Top 10 cities by revenue
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(q_city["city"], q_city["revenue"]/1e5,
              color=COLORS[0], edgecolor="white")
ax.bar_label(bars, labels=[f"₹{v:.0f}L" for v in q_city["revenue"]/1e5],
             padding=3, fontsize=8)
ax.set_title("Top 10 Cities by Revenue", fontsize=13, fontweight="bold")
ax.set_ylabel("Revenue (₹ Lakhs)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/05_revenue_by_city.png", dpi=150)
plt.close()
print("    Saved: 05_revenue_by_city.png")

# Chart 6: Gender split by category
gender_cat = merged.groupby(["category","gender"]).size().unstack(fill_value=0)
gender_pct = gender_cat.div(gender_cat.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 5))
gender_pct.plot(kind="barh", ax=ax, color=["#1A5276","#C0392B","#1D8348"],
                edgecolor="white")
ax.set_title("Gender Split by Category (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Percentage (%)")
ax.legend(title="Gender", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig("outputs/06_gender_split_by_category.png", dpi=150)
plt.close()
print("    Saved: 06_gender_split_by_category.png")

# Chart 7: Return rate by category
fig, ax = plt.subplots(figsize=(10, 4))
return_cat = orders.groupby("category")["returned"].mean().sort_values(ascending=False) * 100
bars = ax.bar(return_cat.index, return_cat.values,
              color=[COLORS[1] if v > 12 else COLORS[0] for v in return_cat.values])
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in return_cat.values],
             padding=3, fontsize=9)
ax.axhline(return_cat.mean(), color="red", linestyle="--", linewidth=1.5,
           label=f"Avg {return_cat.mean():.1f}%")
ax.set_title("Return Rate by Category", fontsize=13, fontweight="bold")
ax.set_ylabel("Return Rate (%)")
plt.xticks(rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/07_return_rate_by_category.png", dpi=150)
plt.close()
print("    Saved: 07_return_rate_by_category.png")

# ── 4. RFM ANALYSIS ───────────────────────────────────────────────────────────
print("\n[4] RFM Analysis...")
snapshot_date = pd.Timestamp("2025-01-01")
rfm = orders.groupby("customer_id").agg(
    recency  =("order_date",  lambda x: (snapshot_date - x.max()).days),
    frequency=("order_id",    "count"),
    monetary =("price_paid",  "sum")
).reset_index()

# Score 1–5
for col, ascending in [("recency", False), ("frequency", True), ("monetary", True)]:
    label = col[0].upper()
    rfm[f"{label}_score"] = pd.qcut(rfm[col], q=5,
                                     labels=[1,2,3,4,5],
                                     duplicates="drop").astype(int)
    if not ascending:
        rfm[f"{label}_score"] = 6 - rfm[f"{label}_score"]

rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
rfm["segment"]   = pd.cut(rfm["RFM_score"],
                           bins=[0, 5, 8, 11, 15],
                           labels=["At Risk","Needs Attention","Loyal","Champions"])

seg_counts = rfm["segment"].value_counts()
print(f"    Champions       : {seg_counts.get('Champions', 0):,}")
print(f"    Loyal           : {seg_counts.get('Loyal', 0):,}")
print(f"    Needs Attention : {seg_counts.get('Needs Attention', 0):,}")
print(f"    At Risk         : {seg_counts.get('At Risk', 0):,}")

# Chart 8: RFM segments
fig, ax = plt.subplots(figsize=(8, 5))
seg_rev = rfm.groupby("segment")["monetary"].sum().sort_values(ascending=False)
bars = ax.bar(seg_rev.index, seg_rev.values/1e5,
              color=[COLORS[1],COLORS[3],COLORS[2],COLORS[0]])
ax.bar_label(bars, labels=[f"₹{v:.0f}L" for v in seg_rev.values/1e5],
             padding=3, fontsize=9)
ax.set_title("Revenue by RFM Customer Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("Revenue (₹ Lakhs)")
plt.tight_layout()
plt.savefig("outputs/08_rfm_segments.png", dpi=150)
plt.close()
print("    Saved: 08_rfm_segments.png")

# ── 5. K-MEANS CUSTOMER SEGMENTATION ─────────────────────────────────────────
print("\n[5] K-Means Customer Segmentation...")
features = rfm[["recency","frequency","monetary"]].copy()
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Find optimal k using silhouette
sil_scores = {}
for k in range(2, 7):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, lbl)

best_k = max(sil_scores, key=sil_scores.get)
print(f"    Optimal clusters (silhouette): k={best_k} → score={sil_scores[best_k]:.4f}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
rfm["cluster"] = km_final.fit_predict(X_scaled)

cluster_summary = rfm.groupby("cluster").agg(
    count     =("customer_id","count"),
    avg_recency=("recency","mean"),
    avg_freq  =("frequency","mean"),
    avg_spend =("monetary","mean")
).round(1)
print(f"\n    Cluster Summary:\n{cluster_summary.to_string()}")

# Chart 9: Cluster scatter
fig, ax = plt.subplots(figsize=(9, 5))
for c in sorted(rfm["cluster"].unique()):
    sub = rfm[rfm["cluster"] == c]
    ax.scatter(sub["frequency"], sub["monetary"]/1e3,
               alpha=0.5, s=20, color=COLORS[c], label=f"Cluster {c}")
ax.set_title(f"K-Means Customer Segmentation (k={best_k})", fontsize=13, fontweight="bold")
ax.set_xlabel("Purchase Frequency")
ax.set_ylabel("Total Spend (₹ Thousands)")
ax.legend(title="Cluster")
plt.tight_layout()
plt.savefig("outputs/09_kmeans_clusters.png", dpi=150)
plt.close()
print("    Saved: 09_kmeans_clusters.png")

# Chart 10: Festive season analysis
orders["month"] = orders["order_date"].dt.month
orders["year"]  = orders["order_date"].dt.year
festive = orders.groupby(["year","month"])["price_paid"].sum().reset_index()
festive["is_festive"] = festive["month"].isin([10,11,12])
festive_avg = festive.groupby("is_festive")["price_paid"].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
monthly_avg = orders.groupby("month")["price_paid"].sum() / orders["year"].nunique()
axes[0].bar(range(1, 13), monthly_avg.values/1e5,
            color=[COLORS[1] if m in [10,11,12] else COLORS[0] for m in range(1,13)])
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
axes[0].set_title("Avg Monthly Revenue (Festive = Red)", fontweight="bold")
axes[0].set_ylabel("Revenue (₹ Lakhs)")

axes[1].bar(["Non-Festive\n(Jan–Sep)", "Festive\n(Oct–Dec)"],
            [festive_avg[False]/1e5, festive_avg[True]/1e5],
            color=[COLORS[0], COLORS[1]])
uplift = (festive_avg[True] - festive_avg[False]) / festive_avg[False] * 100
axes[1].set_title(f"Festive vs Non-Festive Revenue\n(+{uplift:.1f}% uplift)", fontweight="bold")
axes[1].set_ylabel("Avg Monthly Revenue (₹ Lakhs)")
plt.suptitle("Festive Season Impact Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/10_festive_season_analysis.png", dpi=150)
plt.close()
print("    Saved: 10_festive_season_analysis.png")

# ── 6. EXPORT POWER BI CSVs ───────────────────────────────────────────────────
print("\n[6] Exporting Power BI-ready CSVs...")
q_revenue.to_csv("outputs/powerbi_category_summary.csv", index=False)
q_monthly.to_csv("outputs/powerbi_monthly_revenue.csv", index=False)
rfm[["customer_id","recency","frequency","monetary","segment","cluster"]]\
    .to_csv("outputs/powerbi_rfm_segments.csv", index=False)
q_city.to_csv("outputs/powerbi_city_revenue.csv", index=False)
print("    Saved: 4 Power BI CSVs")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
total_rev    = orders["price_paid"].sum()
festive_rev  = orders[orders["month"].isin([10,11,12])]["price_paid"].sum()
festive_pct  = festive_rev / total_rev * 100
top_cat      = q_revenue.iloc[0]["category"]
top_city     = q_city.iloc[0]["city"]
return_rate  = orders["returned"].mean() * 100
champ_count  = seg_counts.get("Champions", 0)

print(f"\n{'=' * 65}")
print("  KEY BUSINESS INSIGHTS")
print(f"{'=' * 65}")
print(f"  Total Revenue      : ₹{total_rev/1e6:.2f}M across {len(orders):,} orders")
print(f"  Top Category       : {top_cat}")
print(f"  Top City           : {top_city}")
print(f"  Festive Season Rev : {festive_pct:.1f}% of annual revenue (Oct–Dec)")
print(f"  Festive Uplift     : +{uplift:.1f}% vs non-festive months")
print(f"  Overall Return Rate: {return_rate:.1f}%")
print(f"  Champion Customers : {champ_count:,} ({champ_count/len(rfm)*100:.1f}% of base)")
print(f"  Optimal Segments   : {best_k} clusters (silhouette: {sil_scores[best_k]:.4f})")
print(f"  Visualisations     : 10 charts + 4 Power BI CSVs")
print(f"{'=' * 65}")
