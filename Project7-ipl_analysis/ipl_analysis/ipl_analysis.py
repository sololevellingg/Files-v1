# =============================================================================
# ipl_analysis.py — IPL Data Analysis (2008–2024)
# Tools: Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn), SQL (SQLite)
# Author: Sharat Laha
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sqlite3
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import os
os.makedirs("outputs", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
BLUE   = "#1A5276"
RED    = "#C0392B"
GREEN  = "#1D8348"
AMBER  = "#D68910"
PURPLE = "#7D3C98"
COLORS = [BLUE, RED, GREEN, AMBER, PURPLE, "#0E6655", "#BA4A00", "#1A237E", "#4A235A", "#1B5E20"]

print("=" * 65)
print("  IPL DATA ANALYSIS — 2008 to 2024")
print("=" * 65)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
matches = pd.read_csv("ipl_analysis/ipl_analysis/matches.csv", parse_dates=["date"])
batting = pd.read_csv("ipl_analysis/ipl_analysis/batting.csv")
bowling = pd.read_csv("ipl_analysis/ipl_analysis/bowling.csv")

print(f"    Matches : {len(matches):,} | Seasons: {matches['season'].nunique()}")
print(f"    Batting : {len(batting):,} rows")
print(f"    Bowling : {len(bowling):,} rows")
print(f"    Date range: {matches['date'].min().date()} → {matches['date'].max().date()}")

# ── 2. SQL ANALYSIS ───────────────────────────────────────────────────────────
print("\n[2] SQL analysis...")
conn = sqlite3.connect(":memory:")
matches.to_sql("matches", conn, index=False, if_exists="replace")
batting.to_sql("batting",  conn, index=False, if_exists="replace")
bowling.to_sql("bowling",  conn, index=False, if_exists="replace")

# Team wins
q_wins = pd.read_sql("""
    SELECT winner AS team,
           COUNT(*) AS wins,
           ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM matches),1) AS win_pct
    FROM matches WHERE winner IS NOT NULL
    GROUP BY winner ORDER BY wins DESC
""", conn)

# Top batsmen
q_bat = pd.read_sql("""
    SELECT batsman,
           COUNT(DISTINCT match_id)          AS matches,
           SUM(runs)                          AS total_runs,
           MAX(runs)                          AS highest_score,
           ROUND(AVG(runs),1)                AS avg_runs,
           ROUND(AVG(strike_rate),1)         AS avg_sr,
           SUM(sixes)                         AS total_sixes,
           SUM(fours)                         AS total_fours
    FROM batting GROUP BY batsman
    HAVING matches >= 20
    ORDER BY total_runs DESC LIMIT 10
""", conn)

# Top bowlers
q_bowl = pd.read_sql("""
    SELECT bowler,
           COUNT(DISTINCT match_id)       AS matches,
           SUM(wickets)                    AS total_wickets,
           ROUND(AVG(economy),2)          AS avg_economy,
           ROUND(SUM(runs_given)*1.0/NULLIF(SUM(wickets),0),2) AS bowling_avg
    FROM bowling GROUP BY bowler
    HAVING matches >= 20
    ORDER BY total_wickets DESC LIMIT 10
""", conn)

# Toss analysis
q_toss = pd.read_sql("""
    SELECT toss_decision,
           COUNT(*) AS total,
           SUM(CASE WHEN toss_winner=winner THEN 1 ELSE 0 END) AS toss_wins_match,
           ROUND(SUM(CASE WHEN toss_winner=winner THEN 1.0 ELSE 0 END)*100/COUNT(*),1) AS win_pct
    FROM matches GROUP BY toss_decision
""", conn)

# Season stats
q_season = pd.read_sql("""
    SELECT season,
           COUNT(*) AS matches,
           ROUND(AVG(team1_score),1) AS avg_score,
           ROUND(AVG(team1_score+team2_score),1) AS avg_total_runs
    FROM matches GROUP BY season ORDER BY season
""", conn)

print("\n    --- Top 5 Teams by Wins ---")
print(q_wins.head().to_string(index=False))
print("\n    --- Toss Impact ---")
print(q_toss.to_string(index=False))
conn.close()

# ── 3. VISUALISATIONS ─────────────────────────────────────────────────────────
print("\n[3] Generating visualisations...")

# Chart 1: Team wins
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(q_wins["team"], q_wins["wins"],
              color=[COLORS[i % len(COLORS)] for i in range(len(q_wins))],
              edgecolor="white")
ax.bar_label(bars, labels=[f"{v}" for v in q_wins["wins"]], padding=3, fontsize=9)
ax.set_title("IPL Total Wins by Team (2008–2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Wins")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/01_team_wins.png", dpi=150)
plt.close()
print("    Saved: 01_team_wins.png")

# Chart 2: Season avg scores
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(q_season["season"], q_season["avg_score"],
        "o-", color=BLUE, linewidth=2, markersize=6, label="Avg 1st innings score")
ax.plot(q_season["season"], q_season["avg_total_runs"]/2,
        "s--", color=RED, linewidth=2, markersize=5, label="Avg per innings (both)")
ax.fill_between(q_season["season"], q_season["avg_score"], alpha=0.1, color=BLUE)
ax.set_title("IPL Average Scores per Season (2008–2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("Average Runs")
ax.set_xlabel("Season")
ax.set_xticks(q_season["season"])
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/02_season_avg_scores.png", dpi=150)
plt.close()
print("    Saved: 02_season_avg_scores.png")

# Chart 3: Top 10 batsmen
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(q_bat["batsman"][::-1], q_bat["total_runs"][::-1], color=BLUE)
ax.bar_label(bars, labels=[f"{v:,}" for v in q_bat["total_runs"][::-1]],
             padding=4, fontsize=9)
ax.set_title("Top 10 IPL Run Scorers (all time)", fontsize=13, fontweight="bold")
ax.set_xlabel("Total Runs")
plt.tight_layout()
plt.savefig("outputs/03_top_batsmen.png", dpi=150)
plt.close()
print("    Saved: 03_top_batsmen.png")

# Chart 4: Top 10 bowlers
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(q_bowl["bowler"][::-1], q_bowl["total_wickets"][::-1], color=RED)
ax.bar_label(bars, labels=[f"{v}" for v in q_bowl["total_wickets"][::-1]],
             padding=4, fontsize=9)
ax.set_title("Top 10 IPL Wicket Takers (all time)", fontsize=13, fontweight="bold")
ax.set_xlabel("Total Wickets")
plt.tight_layout()
plt.savefig("outputs/04_top_bowlers.png", dpi=150)
plt.close()
print("    Saved: 04_top_bowlers.png")

# Chart 5: Toss decision impact
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
toss_labels = q_toss["toss_decision"].str.capitalize()
axes[0].bar(toss_labels, q_toss["total"], color=[BLUE, RED], edgecolor="white")
axes[0].set_title("Toss Decision Distribution", fontweight="bold")
axes[0].set_ylabel("Number of Matches")
axes[1].bar(toss_labels, q_toss["win_pct"], color=[GREEN, AMBER], edgecolor="white")
axes[1].axhline(50, color="red", linestyle="--", linewidth=1, label="50% baseline")
axes[1].set_title("Win % After Toss Decision", fontweight="bold")
axes[1].set_ylabel("Win %")
axes[1].legend()
plt.suptitle("Toss Impact Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/05_toss_analysis.png", dpi=150)
plt.close()
print("    Saved: 05_toss_analysis.png")

# Chart 6: Venue win rates
venue_wins = matches.groupby("venue").agg(
    matches=("match_id","count"),
    avg_score=("team1_score","mean")
).reset_index().sort_values("avg_score", ascending=False).head(10)
venue_wins["venue_short"] = venue_wins["venue"].str.split(",").str[0]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(venue_wins["venue_short"][::-1], venue_wins["avg_score"][::-1], color=PURPLE)
ax.bar_label(bars, labels=[f"{v:.0f}" for v in venue_wins["avg_score"][::-1]],
             padding=4, fontsize=9)
ax.set_title("Avg 1st Innings Score by Venue (Top 10)", fontsize=13, fontweight="bold")
ax.set_xlabel("Avg Runs")
plt.tight_layout()
plt.savefig("outputs/06_venue_analysis.png", dpi=150)
plt.close()
print("    Saved: 06_venue_analysis.png")

# Chart 7: Win margin distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
run_wins  = matches[matches["win_by_runs"] > 0]["win_by_runs"]
wkt_wins  = matches[matches["win_by_wickets"] > 0]["win_by_wickets"]
axes[0].hist(run_wins, bins=20, color=BLUE, edgecolor="white", alpha=0.85)
axes[0].axvline(run_wins.mean(), color=RED, linestyle="--", linewidth=1.5,
                label=f"Avg: {run_wins.mean():.0f} runs")
axes[0].set_title("Win by Runs — Distribution", fontweight="bold")
axes[0].set_xlabel("Runs")
axes[0].legend()
axes[1].hist(wkt_wins, bins=10, color=RED, edgecolor="white", alpha=0.85)
axes[1].axvline(wkt_wins.mean(), color=BLUE, linestyle="--", linewidth=1.5,
                label=f"Avg: {wkt_wins.mean():.1f} wkts")
axes[1].set_title("Win by Wickets — Distribution", fontweight="bold")
axes[1].set_xlabel("Wickets")
axes[1].legend()
plt.suptitle("Win Margin Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/07_win_margins.png", dpi=150)
plt.close()
print("    Saved: 07_win_margins.png")

# Chart 8: Strike rate vs avg runs (scatter)
bat_qual = batting.groupby("batsman").agg(
    total_runs=("runs","sum"),
    avg_runs=("runs","mean"),
    avg_sr=("strike_rate","mean"),
    matches=("match_id","count")
).reset_index()
bat_qual = bat_qual[bat_qual["matches"] >= 20]

fig, ax = plt.subplots(figsize=(9, 5))
scatter = ax.scatter(bat_qual["avg_runs"], bat_qual["avg_sr"],
                     c=bat_qual["total_runs"], cmap="Blues",
                     alpha=0.7, s=60, edgecolors="white", linewidths=0.5)
plt.colorbar(scatter, ax=ax, label="Total Runs")
ax.axvline(bat_qual["avg_runs"].mean(), color=RED, linestyle="--", linewidth=1, alpha=0.6)
ax.axhline(bat_qual["avg_sr"].mean(),  color=RED, linestyle="--", linewidth=1, alpha=0.6)
ax.set_xlabel("Avg Runs per Innings")
ax.set_ylabel("Avg Strike Rate")
ax.set_title("Batsman Quality: Avg Runs vs Strike Rate", fontsize=13, fontweight="bold")
# Label top 5
top5 = bat_qual.nlargest(5, "total_runs")
for _, row in top5.iterrows():
    ax.annotate(row["batsman"], (row["avg_runs"], row["avg_sr"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
plt.tight_layout()
plt.savefig("outputs/08_batsman_quality.png", dpi=150)
plt.close()
print("    Saved: 08_batsman_quality.png")

# ── 4. ML — MATCH OUTCOME PREDICTION ─────────────────────────────────────────
print("\n[4] ML — Match Outcome Prediction (Random Forest)...")

ml = matches.copy()
le = LabelEncoder()
for col in ["team1","team2","venue","toss_winner","toss_decision","winner"]:
    ml[col] = le.fit_transform(ml[col].astype(str))

features = ["team1","team2","venue","toss_winner","toss_decision",
            "team1_score","season"]
X = ml[features]
y = (ml["winner"] == ml["team1"]).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"    Random Forest Accuracy: {acc*100:.2f}%")

# Chart 9: Feature importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(fi.index, fi.values, color=COLORS[:len(fi)], edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.3f}" for v in fi.values], padding=3, fontsize=9)
ax.set_title(f"Match Outcome Prediction — Feature Importances\n(Random Forest Accuracy: {acc*100:.2f}%)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Importance Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("outputs/09_ml_feature_importance.png", dpi=150)
plt.close()
print("    Saved: 09_ml_feature_importance.png")

# Chart 10: Team win rate heatmap (head-to-head)
teams_top6 = q_wins.head(6)["team"].tolist()
h2h = np.zeros((6, 6))
for _, row in matches.iterrows():
    if row["team1"] in teams_top6 and row["team2"] in teams_top6:
        i = teams_top6.index(row["team1"])
        j = teams_top6.index(row["team2"])
        if row["winner"] == row["team1"]:
            h2h[i][j] += 1
        else:
            h2h[j][i] += 1

team_short = [t.split()[0] + " " + t.split()[1][0] if len(t.split()) > 1 else t[:10]
              for t in teams_top6]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(h2h, annot=True, fmt=".0f", cmap="Blues",
            xticklabels=team_short, yticklabels=team_short, ax=ax,
            linewidths=0.5, cbar_kws={"label": "Wins"})
ax.set_title("Head-to-Head Wins (Top 6 Teams)", fontsize=13, fontweight="bold")
ax.set_xlabel("Lost against →")
ax.set_ylabel("← Won against")
plt.tight_layout()
plt.savefig("outputs/10_head_to_head.png", dpi=150)
plt.close()
print("    Saved: 10_head_to_head.png")

# ── 5. EXPORT POWER BI CSVs ───────────────────────────────────────────────────
print("\n[5] Exporting Power BI-ready CSVs...")
q_wins.to_csv("outputs/powerbi_team_wins.csv", index=False)
q_bat.to_csv("outputs/powerbi_top_batsmen.csv", index=False)
q_bowl.to_csv("outputs/powerbi_top_bowlers.csv", index=False)
q_season.to_csv("outputs/powerbi_season_stats.csv", index=False)
print("    Saved: 4 Power BI CSVs")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
top_team   = q_wins.iloc[0]["team"]
top_bat    = q_bat.iloc[0]["batsman"]
top_bowl   = q_bowl.iloc[0]["bowler"]
toss_field = q_toss[q_toss["toss_decision"] == "field"]["win_pct"].values[0]

print(f"\n{'=' * 65}")
print("  KEY INSIGHTS")
print(f"{'=' * 65}")
print(f"  Total Matches     : {len(matches):,} | Seasons: 2008–2024")
print(f"  Most Successful   : {top_team} ({q_wins.iloc[0]['wins']} wins)")
print(f"  Top Run Scorer    : {top_bat} ({q_bat.iloc[0]['total_runs']:,} runs)")
print(f"  Top Wicket Taker  : {top_bowl} ({q_bowl.iloc[0]['total_wickets']} wickets)")
print(f"  Field after toss  : {toss_field}% win rate (teams prefer fielding)")
print(f"  ML Accuracy       : {acc*100:.2f}% (Random Forest, match outcome)")
print(f"  Visualisations    : 10 charts + 4 Power BI CSVs")
print(f"{'=' * 65}")
