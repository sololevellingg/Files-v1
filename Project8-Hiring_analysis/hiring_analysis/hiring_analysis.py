# =============================================================================
# hiring_analysis.py — Hiring Process Analysis
# Tools: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn), SQL (SQLite)
# Author: Sharat Laha
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder

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
print("  HIRING PROCESS ANALYSIS")
print("=" * 65)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_csv("hiring_analysis/hiring_data.csv", parse_dates=["application_date"])
print(f"    Records        : {len(df):,}")
print(f"    Date range     : {df['application_date'].min().date()} → {df['application_date'].max().date()}")
print(f"    Departments    : {df['department'].nunique()}")
print(f"    Offer rate     : {df['offered'].mean()*100:.1f}%")
print(f"    Acceptance rate: {df['accepted'].mean()*100:.1f}%")
print(f"    Join rate      : {df['joined'].mean()*100:.1f}%")

# ── 2. SQL ────────────────────────────────────────────────────────────────────
print("\n[2] SQL analysis...")
conn = sqlite3.connect(":memory:")
df.to_sql("hiring", conn, index=False, if_exists="replace")

q_funnel = pd.read_sql("""
    SELECT
        COUNT(*)                                          AS total_applications,
        SUM(CASE WHEN screened=1   THEN 1 ELSE 0 END)   AS screened,
        SUM(CASE WHEN interviewed=1 THEN 1 ELSE 0 END)  AS interviewed,
        SUM(CASE WHEN offered=1    THEN 1 ELSE 0 END)   AS offered,
        SUM(CASE WHEN accepted=1   THEN 1 ELSE 0 END)   AS accepted,
        SUM(CASE WHEN joined=1     THEN 1 ELSE 0 END)   AS joined
    FROM hiring
""", conn)

q_dept = pd.read_sql("""
    SELECT department,
           COUNT(*)                                        AS applications,
           SUM(CASE WHEN joined=1 THEN 1 ELSE 0 END)     AS hires,
           ROUND(SUM(CASE WHEN joined=1 THEN 1.0 ELSE 0 END)*100/COUNT(*),1) AS hire_rate,
           ROUND(AVG(CASE WHEN days_to_hire IS NOT NULL THEN days_to_hire END),1) AS avg_days_to_hire
    FROM hiring GROUP BY department ORDER BY hires DESC
""", conn)

q_source = pd.read_sql("""
    SELECT source,
           COUNT(*) AS applications,
           SUM(CASE WHEN joined=1 THEN 1 ELSE 0 END) AS hires,
           ROUND(SUM(CASE WHEN joined=1 THEN 1.0 ELSE 0 END)*100/COUNT(*),1) AS hire_rate,
           ROUND(AVG(CASE WHEN offered_salary IS NOT NULL THEN offered_salary END),0) AS avg_salary_offered
    FROM hiring GROUP BY source ORDER BY hires DESC
""", conn)

q_gender = pd.read_sql("""
    SELECT gender, department,
           COUNT(*) AS applications,
           SUM(CASE WHEN joined=1 THEN 1 ELSE 0 END) AS hires
    FROM hiring GROUP BY gender, department
""", conn)

q_salary = pd.read_sql("""
    SELECT role_level,
           ROUND(AVG(offered_salary),0)  AS avg_offered,
           ROUND(AVG(expected_salary),0) AS avg_expected,
           ROUND(AVG(offered_salary)-AVG(expected_salary),0) AS salary_gap,
           COUNT(*) AS offers
    FROM hiring WHERE offered=1
    GROUP BY role_level ORDER BY avg_offered DESC
""", conn)

print("\n    --- Hiring Funnel ---")
print(q_funnel.T.to_string())
print("\n    --- Dept Hire Rate ---")
print(q_dept[["department","applications","hires","hire_rate"]].to_string(index=False))
conn.close()

# ── 3. VISUALISATIONS ─────────────────────────────────────────────────────────
print("\n[3] Generating visualisations...")

# Chart 1: Hiring funnel
funnel_stages  = ["Applications","Screened","Interviewed","Offered","Accepted","Joined"]
funnel_values  = [q_funnel["total_applications"].values[0],
                  q_funnel["screened"].values[0],
                  q_funnel["interviewed"].values[0],
                  q_funnel["offered"].values[0],
                  q_funnel["accepted"].values[0],
                  q_funnel["joined"].values[0]]
funnel_pcts = [f"{v/funnel_values[0]*100:.1f}%" for v in funnel_values]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(funnel_stages[::-1], funnel_values[::-1],
               color=[COLORS[i % len(COLORS)] for i in range(len(funnel_stages))],
               edgecolor="white")
for bar, val, pct in zip(bars, funnel_values[::-1], funnel_pcts[::-1]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
            f"{val:,}  ({pct})", va="center", fontsize=9)
ax.set_title("Hiring Funnel Analysis (2022–2024)", fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Candidates")
ax.set_xlim(0, funnel_values[0] * 1.25)
plt.tight_layout()
plt.savefig("outputs/01_hiring_funnel.png", dpi=150)
plt.close()
print("    Saved: 01_hiring_funnel.png")

# Chart 2: Hire rate by department
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(q_dept["department"], q_dept["hire_rate"],
              color=COLORS[:len(q_dept)], edgecolor="white")
ax.bar_label(bars, labels=[f"{v}%" for v in q_dept["hire_rate"]], padding=3, fontsize=9)
ax.axhline(q_dept["hire_rate"].mean(), color=RED, linestyle="--", linewidth=1.5,
           label=f"Avg {q_dept['hire_rate'].mean():.1f}%")
ax.set_title("Hire Rate by Department", fontsize=13, fontweight="bold")
ax.set_ylabel("Hire Rate (%)")
plt.xticks(rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/02_hire_rate_by_dept.png", dpi=150)
plt.close()
print("    Saved: 02_hire_rate_by_dept.png")

# Chart 3: Source effectiveness
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(q_source["source"][::-1], q_source["applications"][::-1], color=BLUE)
axes[0].set_title("Applications by Source", fontweight="bold")
axes[0].set_xlabel("Applications")
axes[1].barh(q_source["source"][::-1], q_source["hire_rate"][::-1], color=GREEN)
axes[1].set_title("Hire Rate by Source (%)", fontweight="bold")
axes[1].set_xlabel("Hire Rate (%)")
plt.suptitle("Recruitment Source Effectiveness", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_source_effectiveness.png", dpi=150)
plt.close()
print("    Saved: 03_source_effectiveness.png")

# Chart 4: Time to hire by department
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(q_dept["department"], q_dept["avg_days_to_hire"],
              color=AMBER, edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.0f}d" for v in q_dept["avg_days_to_hire"]],
             padding=3, fontsize=9)
ax.axhline(q_dept["avg_days_to_hire"].mean(), color=RED, linestyle="--",
           linewidth=1.5, label=f"Avg {q_dept['avg_days_to_hire'].mean():.0f} days")
ax.set_title("Average Time-to-Hire by Department (Days)", fontsize=13, fontweight="bold")
ax.set_ylabel("Days")
plt.xticks(rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/04_time_to_hire.png", dpi=150)
plt.close()
print("    Saved: 04_time_to_hire.png")

# Chart 5: Salary offered vs expected by role
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(q_salary))
w = 0.35
ax.bar(x - w/2, q_salary["avg_offered"]/1e5,  w, label="Avg Offered",  color=BLUE, edgecolor="white")
ax.bar(x + w/2, q_salary["avg_expected"]/1e5, w, label="Avg Expected", color=RED,  edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(q_salary["role_level"])
ax.set_title("Offered vs Expected Salary by Role Level (₹ Lakhs)", fontsize=13, fontweight="bold")
ax.set_ylabel("Salary (₹ Lakhs)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/05_salary_analysis.png", dpi=150)
plt.close()
print("    Saved: 05_salary_analysis.png")

# Chart 6: Gender diversity by department
gender_pivot = q_gender.pivot_table(index="department", columns="gender",
                                     values="applications", fill_value=0)
gender_pct = gender_pivot.div(gender_pivot.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(11, 5))
gender_pct.plot(kind="bar", ax=ax, color=["#1A5276","#C0392B","#1D8348"],
                edgecolor="white", stacked=False)
ax.set_title("Gender Diversity by Department (%)", fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Gender")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/06_gender_diversity.png", dpi=150)
plt.close()
print("    Saved: 06_gender_diversity.png")

# Chart 7: Interview score distribution
scored = df[df["interview_score"].notna()]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(scored["interview_score"], bins=20, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].axvline(scored["interview_score"].mean(), color=RED, linestyle="--",
                label=f"Avg: {scored['interview_score'].mean():.2f}")
axes[0].set_title("Interview Score Distribution", fontweight="bold")
axes[0].set_xlabel("Score (/ 10)")
axes[0].legend()
# Offered vs not by score
axes[1].boxplot([scored[scored["offered"]==True]["interview_score"].dropna(),
                 scored[scored["offered"]==False]["interview_score"].dropna()],
                labels=["Offered", "Not Offered"],
                patch_artist=True,
                boxprops=dict(facecolor=BLUE, alpha=0.6))
axes[1].set_title("Interview Score: Offered vs Not Offered", fontweight="bold")
axes[1].set_ylabel("Interview Score")
plt.suptitle("Interview Performance Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/07_interview_scores.png", dpi=150)
plt.close()
print("    Saved: 07_interview_scores.png")

# Chart 8: Monthly application trend
df["month"] = df["application_date"].dt.to_period("M")
monthly = df.groupby("month").size().reset_index(name="applications")
monthly["month_dt"] = monthly["month"].dt.to_timestamp()
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(monthly["month_dt"], monthly["applications"], color=BLUE, linewidth=2)
ax.fill_between(monthly["month_dt"], monthly["applications"], alpha=0.15, color=BLUE)
ax.set_title("Monthly Application Volume (2022–2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("Applications")
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/08_monthly_applications.png", dpi=150)
plt.close()
print("    Saved: 08_monthly_applications.png")

# ── 4. ML — OFFER ACCEPTANCE PREDICTION ──────────────────────────────────────
print("\n[4] ML — Offer Acceptance Prediction...")
ml = df[df["offered"] == True].copy()
le = LabelEncoder()
for col in ["department","role_level","source","gender","education","city"]:
    ml[col] = le.fit_transform(ml[col].astype(str))

ml["salary_ratio"] = ml["offered_salary"] / ml["expected_salary"].replace(0, np.nan)
ml = ml.dropna(subset=["salary_ratio","interview_score"])

features = ["department","role_level","source","gender","education",
            "experience_years","interview_score","salary_ratio"]
X = ml[features]
y = ml["accepted"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:,1])

print(f"    Random Forest    → Accuracy: {rf_acc*100:.2f}% | AUC-ROC: {rf_auc:.4f}")
print(f"    Gradient Boosting→ Accuracy: {gb_acc*100:.2f}% | AUC-ROC: {gb_auc:.4f}")

# Chart 9: Feature importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(fi.index, fi.values, color=COLORS[:len(fi)], edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.3f}" for v in fi.values], padding=3, fontsize=9)
ax.set_title(f"Offer Acceptance Prediction — Feature Importances\n(RF Acc: {rf_acc*100:.2f}% | AUC: {rf_auc:.4f})",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Importance Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/09_ml_feature_importance.png", dpi=150)
plt.close()
print("    Saved: 09_ml_feature_importance.png")

# Chart 10: Experience vs hire rate
exp_bins = pd.cut(df["experience_years"], bins=[0,2,5,8,12,16],
                  labels=["0-2 yrs","2-5 yrs","5-8 yrs","8-12 yrs","12+ yrs"])
exp_hire = df.groupby(exp_bins)["joined"].mean() * 100
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(exp_hire.index, exp_hire.values, color=PURPLE, edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in exp_hire.values], padding=3, fontsize=10)
ax.set_title("Hire Rate by Experience Band", fontsize=13, fontweight="bold")
ax.set_ylabel("Hire Rate (%)")
ax.set_xlabel("Experience")
plt.tight_layout()
plt.savefig("outputs/10_experience_vs_hire_rate.png", dpi=150)
plt.close()
print("    Saved: 10_experience_vs_hire_rate.png")

# ── 5. POWER BI EXPORT ────────────────────────────────────────────────────────
print("\n[5] Exporting Power BI CSVs...")
q_funnel.to_csv("outputs/powerbi_funnel.csv", index=False)
q_dept.to_csv("outputs/powerbi_dept_stats.csv", index=False)
q_source.to_csv("outputs/powerbi_source_stats.csv", index=False)
q_salary.to_csv("outputs/powerbi_salary_analysis.csv", index=False)
print("    Saved: 4 Power BI CSVs")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
top_source = q_source.iloc[0]["source"]
avg_days   = q_dept["avg_days_to_hire"].mean()
print(f"\n{'=' * 65}")
print("  KEY INSIGHTS")
print(f"{'=' * 65}")
print(f"  Total Applications : {len(df):,}")
print(f"  Overall Hire Rate  : {df['joined'].mean()*100:.1f}%")
print(f"  Best Source        : {top_source} ({q_source.iloc[0]['hire_rate']}% hire rate)")
print(f"  Avg Time-to-Hire   : {avg_days:.0f} days")
print(f"  RF Acceptance Model: {rf_acc*100:.2f}% accuracy | AUC: {rf_auc:.4f}")
print(f"  GB Acceptance Model: {gb_acc*100:.2f}% accuracy | AUC: {gb_auc:.4f}")
print(f"  Salary ratio = top predictor of offer acceptance")
print(f"  Visualisations     : 10 charts + 4 Power BI CSVs")
print(f"{'=' * 65}")
