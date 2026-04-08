# =============================================================================
# marketing_analysis.py — Marketing Campaign Effectiveness Analysis
# Tools: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy), SQL
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

from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os
os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
BLUE   = "#1A5276"
RED    = "#C0392B"
GREEN  = "#1D8348"
AMBER  = "#D68910"
PURPLE = "#7D3C98"
COLORS = [BLUE, RED, GREEN, AMBER, PURPLE, "#0E6655", "#BA4A00"]

print("=" * 65)
print("  MARKETING CAMPAIGN EFFECTIVENESS ANALYSIS")
print("=" * 65)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv("campaign_data.csv", parse_dates=["start_date","end_date"])
print(f"    Campaigns      : {len(df):,}")
print(f"    Date range     : {df['start_date'].min().date()} → {df['end_date'].max().date()}")
print(f"    Total Budget   : ₹{df['budget'].sum()/1e7:.1f} Cr")
print(f"    Total Revenue  : ₹{df['revenue'].sum()/1e7:.1f} Cr")
print(f"    Avg ROI        : {df['roi'].mean():.1f}%")
print(f"    Avg CTR        : {df['ctr'].mean():.2f}%")
print(f"    Avg CVR        : {df['cvr'].mean():.2f}%")

# ── 2. SQL ────────────────────────────────────────────────────────────────────
print("\n[2] SQL analysis...")
conn = sqlite3.connect(":memory:")
df.to_sql("campaigns", conn, index=False, if_exists="replace")

q_channel = pd.read_sql("""
    SELECT channel,
           COUNT(*)                            AS campaigns,
           ROUND(SUM(budget)/1e6,1)           AS total_budget_M,
           ROUND(SUM(revenue)/1e6,1)          AS total_revenue_M,
           ROUND(AVG(roi),1)                  AS avg_roi,
           ROUND(AVG(ctr),2)                  AS avg_ctr,
           ROUND(AVG(cvr),2)                  AS avg_cvr,
           ROUND(AVG(cpa),0)                  AS avg_cpa
    FROM campaigns GROUP BY channel ORDER BY avg_roi DESC
""", conn)

q_segment = pd.read_sql("""
    SELECT segment,
           COUNT(*) AS campaigns,
           ROUND(AVG(roi),1) AS avg_roi,
           ROUND(AVG(cvr),2) AS avg_cvr,
           ROUND(SUM(revenue)/1e6,1) AS total_revenue_M
    FROM campaigns GROUP BY segment ORDER BY avg_roi DESC
""", conn)

q_objective = pd.read_sql("""
    SELECT objective,
           COUNT(*) AS campaigns,
           ROUND(AVG(roi),1) AS avg_roi,
           ROUND(AVG(ctr),2) AS avg_ctr,
           ROUND(AVG(cvr),2) AS avg_cvr
    FROM campaigns GROUP BY objective ORDER BY avg_roi DESC
""", conn)

q_monthly = pd.read_sql("""
    SELECT strftime('%Y-%m', start_date) AS month,
           COUNT(*) AS campaigns,
           ROUND(SUM(budget)/1e6,2) AS budget_M,
           ROUND(SUM(revenue)/1e6,2) AS revenue_M
    FROM campaigns GROUP BY month ORDER BY month
""", conn)

q_ab = pd.read_sql("""
    SELECT ab_group,
           COUNT(*) AS campaigns,
           ROUND(AVG(ctr),3) AS avg_ctr,
           ROUND(AVG(cvr),3) AS avg_cvr,
           ROUND(AVG(roi),2) AS avg_roi,
           ROUND(AVG(conversions),1) AS avg_conversions
    FROM campaigns GROUP BY ab_group
""", conn)

print("\n    --- Channel ROI Ranking ---")
print(q_channel[["channel","avg_roi","avg_ctr","avg_cvr","avg_cpa"]].to_string(index=False))
print("\n    --- A/B Test Summary ---")
print(q_ab.to_string(index=False))
conn.close()

# ── 3. A/B TESTING ───────────────────────────────────────────────────────────
print("\n[3] Statistical A/B Testing...")
grp_A = df[df["ab_group"] == "A"]
grp_B = df[df["ab_group"] == "B"]

# CTR t-test
t_ctr, p_ctr = stats.ttest_ind(grp_A["ctr"], grp_B["ctr"])
# CVR t-test
t_cvr, p_cvr = stats.ttest_ind(grp_A["cvr"], grp_B["cvr"])
# ROI t-test
t_roi, p_roi = stats.ttest_ind(grp_A["roi"], grp_B["roi"])

# Effect sizes (Cohen's d)
def cohens_d(a, b):
    return (b.mean() - a.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)

d_ctr = cohens_d(grp_A["ctr"], grp_B["ctr"])
d_cvr = cohens_d(grp_A["cvr"], grp_B["cvr"])

print(f"    CTR  — A: {grp_A['ctr'].mean():.3f}% | B: {grp_B['ctr'].mean():.3f}% | p={p_ctr:.4f} | Cohen's d={d_ctr:.3f} | {'SIGNIFICANT ✓' if p_ctr < 0.05 else 'Not significant'}")
print(f"    CVR  — A: {grp_A['cvr'].mean():.3f}% | B: {grp_B['cvr'].mean():.3f}% | p={p_cvr:.4f} | Cohen's d={d_cvr:.3f} | {'SIGNIFICANT ✓' if p_cvr < 0.05 else 'Not significant'}")
print(f"    ROI  — A: {grp_A['roi'].mean():.2f}%  | B: {grp_B['roi'].mean():.2f}%  | p={p_roi:.4f} | {'SIGNIFICANT ✓' if p_roi < 0.05 else 'Not significant'}")

# ── 4. VISUALISATIONS ─────────────────────────────────────────────────────────
print("\n[4] Generating visualisations...")

# Chart 1: Channel ROI comparison
fig, ax = plt.subplots(figsize=(10, 5))
sorted_ch = q_channel.sort_values("avg_roi", ascending=True)
bars = ax.barh(sorted_ch["channel"], sorted_ch["avg_roi"],
               color=[GREEN if v > q_channel["avg_roi"].mean() else BLUE
                      for v in sorted_ch["avg_roi"]])
ax.axvline(q_channel["avg_roi"].mean(), color=RED, linestyle="--",
           linewidth=1.5, label=f"Avg ROI {q_channel['avg_roi'].mean():.0f}%")
ax.bar_label(bars, labels=[f"{v:.0f}%" for v in sorted_ch["avg_roi"]],
             padding=4, fontsize=9)
ax.set_title("Average ROI by Marketing Channel", fontsize=13, fontweight="bold")
ax.set_xlabel("Avg ROI (%)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/01_channel_roi.png", dpi=150)
plt.close()
print("    Saved: 01_channel_roi.png")

# Chart 2: Budget vs Revenue by channel
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(q_channel))
w = 0.35
ax.bar(x - w/2, q_channel["total_budget_M"],  w, label="Budget (₹M)",  color=BLUE, edgecolor="white")
ax.bar(x + w/2, q_channel["total_revenue_M"], w, label="Revenue (₹M)", color=GREEN, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(q_channel["channel"], rotation=30, ha="right")
ax.set_title("Budget vs Revenue by Channel (₹ Millions)", fontsize=13, fontweight="bold")
ax.set_ylabel("Amount (₹ Millions)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/02_budget_vs_revenue.png", dpi=150)
plt.close()
print("    Saved: 02_budget_vs_revenue.png")

# Chart 3: A/B Test results
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics = [("CTR (%)", "ctr", p_ctr), ("CVR (%)", "cvr", p_cvr), ("ROI (%)", "roi", p_roi)]
for ax, (title, col, pval) in zip(axes, metrics):
    means = [grp_A[col].mean(), grp_B[col].mean()]
    errs  = [grp_A[col].std()/np.sqrt(len(grp_A)), grp_B[col].std()/np.sqrt(len(grp_B))]
    bars = ax.bar(["Group A\n(Control)","Group B\n(Treatment)"],
                  means, color=[BLUE, RED], edgecolor="white",
                  yerr=errs, capsize=5)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=8, fontsize=9)
    sig = "p<0.05 ✓" if pval < 0.05 else f"p={pval:.3f}"
    ax.set_title(f"{title}\n({sig})", fontweight="bold")
    ax.set_ylabel(title)
plt.suptitle("A/B Test Results — Group A (Control) vs Group B (Treatment)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_ab_test_results.png", dpi=150)
plt.close()
print("    Saved: 03_ab_test_results.png")

# Chart 4: CTR vs CVR scatter by channel
channel_perf = df.groupby("channel").agg(
    avg_ctr=("ctr","mean"), avg_cvr=("cvr","mean"),
    total_rev=("revenue","sum"), campaigns=("campaign_id","count")
).reset_index()
fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(channel_perf["avg_ctr"], channel_perf["avg_cvr"],
                     s=channel_perf["total_rev"]/1e5,
                     c=range(len(channel_perf)), cmap="Blues",
                     alpha=0.8, edgecolors="white", linewidths=1)
for _, row in channel_perf.iterrows():
    ax.annotate(row["channel"], (row["avg_ctr"], row["avg_cvr"]),
                fontsize=9, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("Avg CTR (%)")
ax.set_ylabel("Avg CVR (%)")
ax.set_title("Channel Performance: CTR vs CVR\n(bubble size = total revenue)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/04_ctr_vs_cvr.png", dpi=150)
plt.close()
print("    Saved: 04_ctr_vs_cvr.png")

# Chart 5: ROI by customer segment
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(q_segment["segment"], q_segment["avg_roi"],
              color=COLORS[:len(q_segment)], edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.0f}%" for v in q_segment["avg_roi"]],
             padding=3, fontsize=9)
ax.set_title("Average ROI by Customer Segment", fontsize=13, fontweight="bold")
ax.set_ylabel("Avg ROI (%)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("outputs/05_roi_by_segment.png", dpi=150)
plt.close()
print("    Saved: 05_roi_by_segment.png")

# Chart 6: Monthly budget & revenue trend
q_monthly["month_dt"] = pd.to_datetime(q_monthly["month"])
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(q_monthly["month_dt"], q_monthly["revenue_M"], color=GREEN,
        linewidth=2, label="Revenue (₹M)")
ax.plot(q_monthly["month_dt"], q_monthly["budget_M"],  color=BLUE,
        linewidth=2, linestyle="--", label="Budget (₹M)")
ax.fill_between(q_monthly["month_dt"], q_monthly["budget_M"],
                q_monthly["revenue_M"], alpha=0.1, color=GREEN, label="Profit zone")
ax.set_title("Monthly Budget vs Revenue (2022–2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("Amount (₹ Millions)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/06_monthly_trend.png", dpi=150)
plt.close()
print("    Saved: 06_monthly_trend.png")

# Chart 7: CPA heatmap — channel vs segment
cpa_pivot = df.groupby(["channel","segment"])["cpa"].mean().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(cpa_pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            ax=ax, linewidths=0.5, cbar_kws={"label": "Avg CPA (₹)"})
ax.set_title("Cost Per Acquisition Heatmap: Channel vs Segment (₹)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/07_cpa_heatmap.png", dpi=150)
plt.close()
print("    Saved: 07_cpa_heatmap.png")

# Chart 8: ROI distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["roi"], bins=40, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].axvline(df["roi"].mean(), color=RED, linestyle="--",
                label=f"Avg: {df['roi'].mean():.0f}%")
axes[0].axvline(0, color=AMBER, linestyle="-", linewidth=1.5, label="Break-even")
axes[0].set_title("ROI Distribution (all campaigns)", fontweight="bold")
axes[0].set_xlabel("ROI (%)")
axes[0].legend()
roi_obj = df.groupby("objective")["roi"].mean().sort_values()
axes[1].barh(roi_obj.index, roi_obj.values, color=GREEN, edgecolor="white")
axes[1].axvline(roi_obj.mean(), color=RED, linestyle="--", linewidth=1.5)
axes[1].set_title("Avg ROI by Campaign Objective", fontweight="bold")
axes[1].set_xlabel("Avg ROI (%)")
plt.suptitle("ROI Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/08_roi_analysis.png", dpi=150)
plt.close()
print("    Saved: 08_roi_analysis.png")

# ── 5. ML — CAMPAIGN SUCCESS PREDICTION ──────────────────────────────────────
print("\n[5] ML — Campaign Success Prediction...")
ml = df.copy()
le = LabelEncoder()
for col in ["channel","segment","objective","product","region","ab_group"]:
    ml[col] = le.fit_transform(ml[col].astype(str))

features = ["channel","segment","objective","product","region","ab_group",
            "budget","duration_days","impressions","ctr","cvr","cpc"]
X = ml[features]
y = ml["success"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_s)[:,1])

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                 max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:,1])

print(f"    Logistic Regression  → Accuracy: {lr_acc*100:.2f}% | AUC-ROC: {lr_auc:.4f}")
print(f"    Gradient Boosting    → Accuracy: {gb_acc*100:.2f}% | AUC-ROC: {gb_auc:.4f}")

# Chart 9: Feature importance (GB)
fi = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(fi.index, fi.values, color=COLORS[:len(fi)], edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.3f}" for v in fi.values], padding=3, fontsize=8)
ax.set_title(f"Campaign Success Prediction — Feature Importances\n(GB Acc: {gb_acc*100:.2f}% | AUC: {gb_auc:.4f})",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Importance Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/09_ml_feature_importance.png", dpi=150)
plt.close()
print("    Saved: 09_ml_feature_importance.png")

# Chart 10: Channel efficiency quadrant (CTR vs ROI)
ch_quad = df.groupby("channel").agg(avg_ctr=("ctr","mean"), avg_roi=("roi","mean")).reset_index()
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(ch_quad["avg_ctr"], ch_quad["avg_roi"],
           s=200, color=COLORS[:len(ch_quad)], edgecolors="white", linewidths=1.5, zorder=3)
for _, row in ch_quad.iterrows():
    ax.annotate(row["channel"], (row["avg_ctr"], row["avg_roi"]),
                fontsize=9, fontweight="500",
                xytext=(6, 6), textcoords="offset points")
ax.axvline(ch_quad["avg_ctr"].mean(), color=AMBER, linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(ch_quad["avg_roi"].mean(), color=AMBER, linestyle="--", linewidth=1, alpha=0.7)
ax.text(ch_quad["avg_ctr"].max()*0.85, ch_quad["avg_roi"].max()*0.95,
        "High CTR\nHigh ROI", fontsize=8, color=GREEN, fontweight="bold")
ax.text(ch_quad["avg_ctr"].min()*1.02, ch_quad["avg_roi"].max()*0.95,
        "Low CTR\nHigh ROI", fontsize=8, color=BLUE, fontweight="bold")
ax.set_xlabel("Avg CTR (%)")
ax.set_ylabel("Avg ROI (%)")
ax.set_title("Channel Efficiency Quadrant\n(CTR vs ROI)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/10_channel_quadrant.png", dpi=150)
plt.close()
print("    Saved: 10_channel_quadrant.png")

# ── 6. POWER BI EXPORT ────────────────────────────────────────────────────────
print("\n[6] Exporting Power BI CSVs...")
q_channel.to_csv("outputs/powerbi_channel_performance.csv", index=False)
q_segment.to_csv("outputs/powerbi_segment_performance.csv", index=False)
q_monthly.to_csv("outputs/powerbi_monthly_trend.csv", index=False)
q_ab.to_csv("outputs/powerbi_ab_test_results.csv", index=False)
print("    Saved: 4 Power BI CSVs")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
best_channel = q_channel.iloc[0]["channel"]
print(f"\n{'=' * 65}")
print("  KEY INSIGHTS")
print(f"{'=' * 65}")
print(f"  Total Campaigns  : {len(df):,} | Budget: ₹{df['budget'].sum()/1e7:.1f} Cr")
print(f"  Total Revenue    : ₹{df['revenue'].sum()/1e7:.1f} Cr")
print(f"  Best Channel     : {best_channel} (ROI: {q_channel.iloc[0]['avg_roi']:.0f}%)")
print(f"  A/B Test CTR     : B > A by {((grp_B['ctr'].mean()-grp_A['ctr'].mean())/grp_A['ctr'].mean()*100):.1f}% (p={p_ctr:.4f})")
print(f"  A/B Test CVR     : B > A by {((grp_B['cvr'].mean()-grp_A['cvr'].mean())/grp_A['cvr'].mean()*100):.1f}% (p={p_cvr:.4f})")
print(f"  Best ML Model    : Gradient Boosting → {gb_acc*100:.2f}% | AUC: {gb_auc:.4f}")
print(f"  Visualisations   : 10 charts + 4 Power BI CSVs")
print(f"{'=' * 65}")
