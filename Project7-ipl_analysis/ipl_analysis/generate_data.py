# generate_data.py — IPL Dataset Generator (2008–2024)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
    "Sunrisers Hyderabad", "Punjab Kings", "Lucknow Super Giants", "Gujarat Titans"
]

VENUES = [
    "Wankhede Stadium, Mumbai", "M. A. Chidambaram Stadium, Chennai",
    "Eden Gardens, Kolkata", "M. Chinnaswamy Stadium, Bangalore",
    "Arun Jaitley Stadium, Delhi", "Sawai Mansingh Stadium, Jaipur",
    "Rajiv Gandhi Stadium, Hyderabad", "Punjab Cricket Association Stadium, Mohali",
    "Narendra Modi Stadium, Ahmedabad", "BRSABV Ekana Stadium, Lucknow"
]

BATSMEN = [
    "V. Kohli", "R. Sharma", "S. Dhawan", "D. Warner", "K. Williamson",
    "A. Russell", "H. Pandya", "S. Iyer", "K. Rahul", "F. du Plessis",
    "J. Roy", "Q. de Kock", "S. Samson", "R. Pant", "A. Finch",
    "S. Yadav", "T. Head", "R. Garg", "Y. Jaiswal", "R. Tewatia"
]

BOWLERS = [
    "J. Bumrah", "D. Chahar", "Y. Chahal", "R. Ashwin", "S. Nadeem",
    "M. Shami", "T. Boult", "P. Chawla", "K. Rabada", "A. Mishra",
    "B. Kumar", "M. Markande", "W. Sundar", "K. Yadav", "H. Rauf",
    "M. Pathirana", "A. Nortje", "J. Hazlewood", "T. Natarajan", "D. Kuldekar"
]

# ── Matches ───────────────────────────────────────────────────────────────────
match_rows = []
match_id = 1
for year in range(2008, 2025):
    n_matches = np.random.randint(58, 74)
    start_date = datetime(year, 3, 25)
    for m in range(n_matches):
        team1, team2 = np.random.choice(TEAMS, 2, replace=False)
        venue       = np.random.choice(VENUES)
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(["bat", "field"], p=[0.35, 0.65])
        # toss winner wins ~52% of time
        winner = toss_winner if np.random.random() < 0.52 else (team2 if toss_winner == team1 else team1)
        win_by_runs = np.random.randint(1, 80)  if np.random.random() < 0.5 else 0
        win_by_wkts = np.random.randint(1, 10)  if win_by_runs == 0 else 0
        t1_score = np.random.randint(120, 230)
        t2_score = t1_score - win_by_runs if win_by_runs > 0 else t1_score + np.random.randint(1, 20)

        match_rows.append({
            "match_id":      match_id,
            "season":        year,
            "date":          (start_date + timedelta(days=m*2)).strftime("%Y-%m-%d"),
            "venue":         venue,
            "team1":         team1,
            "team2":         team2,
            "toss_winner":   toss_winner,
            "toss_decision": toss_decision,
            "winner":        winner,
            "win_by_runs":   win_by_runs,
            "win_by_wickets":win_by_wkts,
            "team1_score":   t1_score,
            "team2_score":   t2_score,
            "player_of_match": np.random.choice(BATSMEN + BOWLERS),
        })
        match_id += 1

matches = pd.DataFrame(match_rows)

# ── Batting ───────────────────────────────────────────────────────────────────
batting_rows = []
for _, match in matches.iterrows():
    for team in [match["team1"], match["team2"]]:
        n_batsmen = np.random.randint(6, 11)
        runs_left = match["team1_score"] if team == match["team1"] else match["team2_score"]
        for i, bat in enumerate(np.random.choice(BATSMEN, n_batsmen, replace=False)):
            runs  = max(0, int(np.random.exponential(25) if i < 4 else np.random.exponential(12)))
            runs  = min(runs, runs_left)
            runs_left -= runs
            balls = max(runs, np.random.randint(max(1, runs-10), max(runs+15, 2)))
            fours = int(runs * np.random.uniform(0.1, 0.3) / 4)
            sixes = int(runs * np.random.uniform(0.05, 0.2) / 6)
            batting_rows.append({
                "match_id":  match["match_id"],
                "season":    match["season"],
                "batsman":   bat,
                "team":      team,
                "runs":      runs,
                "balls":     balls,
                "fours":     fours,
                "sixes":     sixes,
                "strike_rate": round(runs / balls * 100, 2) if balls > 0 else 0,
                "dismissed": np.random.choice([True, False], p=[0.75, 0.25]),
            })

batting = pd.DataFrame(batting_rows)

# ── Bowling ───────────────────────────────────────────────────────────────────
bowling_rows = []
for _, match in matches.iterrows():
    for team in [match["team1"], match["team2"]]:
        n_bowlers = np.random.randint(4, 7)
        for bowl in np.random.choice(BOWLERS, n_bowlers, replace=False):
            overs   = round(np.random.uniform(1, 4), 1)
            wickets = np.random.choice([0,1,2,3,4,5], p=[0.40,0.30,0.17,0.08,0.04,0.01])
            runs_given = int(overs * np.random.uniform(6, 12))
            economy = round(runs_given / overs, 2) if overs > 0 else 0
            bowling_rows.append({
                "match_id": match["match_id"],
                "season":   match["season"],
                "bowler":   bowl,
                "team":     team,
                "overs":    overs,
                "wickets":  wickets,
                "runs_given": runs_given,
                "economy":  economy,
            })

bowling = pd.DataFrame(bowling_rows)

# ── Save ──────────────────────────────────────────────────────────────────────
matches.to_csv("matches.csv", index=False)
batting.to_csv("batting.csv", index=False)
bowling.to_csv("bowling.csv", index=False)

print(f"Matches : {len(matches):,} | Seasons: 2008–2024")
print(f"Batting : {len(batting):,} rows")
print(f"Bowling : {len(bowling):,} rows")
print(f"Teams   : {len(TEAMS)}")
