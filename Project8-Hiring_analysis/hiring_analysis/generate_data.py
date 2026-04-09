# generate_data.py — Hiring Process Dataset Generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N = 5000
START = datetime(2022, 1, 1)
END   = datetime(2024, 12, 31)

DEPARTMENTS   = ["Engineering","Data Science","Marketing","Sales","Finance","HR","Operations","Product"]
SOURCES       = ["LinkedIn","Naukri","Employee Referral","Campus","Indeed","Company Website","Recruitment Agency"]
ROLES         = ["Junior","Mid-Level","Senior","Lead","Manager"]
GENDERS       = ["Male","Female","Other"]
EDUCATION     = ["B.Tech","MBA","MCA","M.Tech","B.Sc","BBA","PhD"]
CITIES        = ["Mumbai","Delhi","Bangalore","Hyderabad","Pune","Chennai","Kolkata","Ahmedabad"]

dept_probs    = [0.25,0.20,0.10,0.15,0.08,0.07,0.08,0.07]
source_probs  = [0.30,0.25,0.18,0.10,0.08,0.05,0.04]

rows = []
for i in range(N):
    dept   = np.random.choice(DEPARTMENTS, p=dept_probs)
    source = np.random.choice(SOURCES, p=source_probs)
    role   = np.random.choice(ROLES, p=[0.25,0.30,0.25,0.12,0.08])
    gender = np.random.choice(GENDERS, p=[0.52,0.45,0.03])
    edu    = np.random.choice(EDUCATION)
    city   = np.random.choice(CITIES)
    exp    = round(np.random.uniform(0, 15), 1)

    app_date  = START + timedelta(days=int(np.random.uniform(0,(END-START).days)))

    # Screening pass rate varies by source
    screen_pass_rate = {"LinkedIn":0.65,"Naukri":0.55,"Employee Referral":0.80,
                        "Campus":0.60,"Indeed":0.50,"Company Website":0.58,"Recruitment Agency":0.62}
    screened = np.random.random() < screen_pass_rate[source]

    interviewed, offered, accepted, joined = False, False, False, False
    interview_score, offered_salary, expected_salary = None, None, None
    days_to_hire = None

    if screened:
        interviewed = np.random.random() < 0.70
    if interviewed:
        interview_score = round(np.random.uniform(4, 10), 1)
        # Higher score → more likely offer
        offer_prob = 0.30 + (interview_score - 4) / 6 * 0.50
        offered = np.random.random() < offer_prob
    if offered:
        base = {"Junior":400000,"Mid-Level":700000,"Senior":1200000,
                "Lead":1800000,"Manager":2500000}[role]
        offered_salary  = int(base * np.random.uniform(0.9, 1.2))
        expected_salary = int(base * np.random.uniform(0.85, 1.3))
        # Accept if offered >= 90% of expected
        # Accept probability based on multiple factors
        accept_prob = 0.5 + (offered_salary - expected_salary) / (expected_salary + 1) * 2
        accept_prob += (interview_score - 7) * 0.05 if interview_score else 0
        accept_prob = max(0.1, min(0.95, accept_prob))
        accepted = np.random.random() < accept_prob
    if accepted:
        joined = np.random.random() < 0.88
        days_to_hire = np.random.randint(15, 75)

    rows.append({
        "candidate_id":    f"CAND{str(i+1).zfill(5)}",
        "application_date": app_date.strftime("%Y-%m-%d"),
        "department":       dept,
        "role_level":       role,
        "source":           source,
        "gender":           gender,
        "education":        edu,
        "city":             city,
        "experience_years": exp,
        "screened":         screened,
        "interviewed":      interviewed,
        "interview_score":  interview_score,
        "offered":          offered,
        "offered_salary":   offered_salary,
        "expected_salary":  expected_salary,
        "accepted":         accepted,
        "joined":           joined,
        "days_to_hire":     days_to_hire,
    })

df = pd.DataFrame(rows)
df.to_csv("hiring_data.csv", index=False)
print(f"Total Applications : {len(df):,}")
print(f"Screened           : {df['screened'].sum():,} ({df['screened'].mean()*100:.1f}%)")
print(f"Interviewed        : {df['interviewed'].sum():,} ({df['interviewed'].mean()*100:.1f}%)")
print(f"Offered            : {df['offered'].sum():,} ({df['offered'].mean()*100:.1f}%)")
print(f"Accepted           : {df['accepted'].sum():,} ({df['accepted'].mean()*100:.1f}%)")
print(f"Joined             : {df['joined'].sum():,} ({df['joined'].mean()*100:.1f}%)")
