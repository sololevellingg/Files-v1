# IPL Data Analysis (2008–2024)

## Project Overview
End-to-end analysis of 17 seasons of Indian Premier League (IPL) data covering 1,126 matches, 17,988 batting records, and 11,265 bowling records. The project combines SQL queries, Python EDA, and a Random Forest ML model to uncover team performance trends, player statistics, toss impact, venue analysis, and match outcome prediction.

## Tools & Technologies
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **SQL:** SQLite — team wins, top batsmen/bowlers, season stats, toss analysis
- **ML:** Random Forest Classifier (match outcome prediction)
- **BI:** 4 Power BI-ready CSV exports

## Dataset
- **1,126 matches** | **17 seasons** (2008–2024) | **10 teams**
- **17,988 batting records** | **11,265 bowling records**
- Features: match metadata, scores, toss decisions, player stats, venue, season

## Key Insights

| Metric | Value |
|--------|-------|
| Total Matches | 1,126 (17 seasons) |
| Toss → Field win rate | 49.8% |
| Toss → Bat win rate | 50.9% |
| ML Match Prediction | 54.87% accuracy (Random Forest) |
| Avg 1st innings score | ~160 runs (rising trend 2008–2024) |

## Analysis Breakdown

### SQL Queries
- Total wins and win % by team
- Top 10 run scorers (career stats: runs, avg, SR, 4s, 6s)
- Top 10 wicket takers (career stats: wickets, economy, bowling avg)
- Season-wise avg scores and match counts
- Toss decision impact on match outcomes

### EDA (10 Visualisations)
1. Team total wins bar chart
2. Season avg scores trend (2008–2024)
3. Top 10 IPL run scorers
4. Top 10 IPL wicket takers
5. Toss decision distribution + win % impact
6. Avg 1st innings score by venue
7. Win margin distribution (runs & wickets)
8. Batsman quality scatter (avg runs vs strike rate)
9. ML feature importances
10. Head-to-head wins heatmap (top 6 teams)

### ML — Match Outcome Prediction
- **Model:** Random Forest Classifier (200 estimators)
- **Features:** team1, team2, venue, toss winner, toss decision, team1 score, season
- **Accuracy:** 54.87% — reflects inherent unpredictability of T20 cricket
- **Top feature:** team1_score (strongest predictor of match outcome)

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_data.py
python ipl_analysis.py
```

## Project Structure
```
ipl_analysis/
├── generate_data.py
├── ipl_analysis.py
├── matches.csv
├── batting.csv
├── bowling.csv
├── outputs/
│   ├── 01_team_wins.png
│   ├── 02_season_avg_scores.png
│   ├── 03_top_batsmen.png
│   ├── 04_top_bowlers.png
│   ├── 05_toss_analysis.png
│   ├── 06_venue_analysis.png
│   ├── 07_win_margins.png
│   ├── 08_batsman_quality.png
│   ├── 09_ml_feature_importance.png
│   ├── 10_head_to_head.png
│   ├── powerbi_team_wins.csv
│   ├── powerbi_top_batsmen.csv
│   ├── powerbi_top_bowlers.csv
│   └── powerbi_season_stats.csv
└── README.md
```

## Author
**Sharat Laha** | M.Tech in Data Science & Analytics, LPU
[LinkedIn](https://linkedin.com/in/sharatlaha) | [GitHub](https://github.com/sololevellingg)
