# Hiring Process Analysis

## Project Overview
End-to-end HR analytics project analysing 5,000 recruitment records across 8 departments (2022–2024). Covers the full hiring funnel from application to joining, with SQL-driven insights, EDA, diversity analysis, salary benchmarking, and an ML model predicting offer acceptance using Random Forest and Gradient Boosting.

## Tools & Technologies
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **SQL:** SQLite — funnel analysis, department stats, source effectiveness, salary analysis
- **ML:** Random Forest + Gradient Boosting (offer acceptance prediction)
- **BI:** 4 Power BI-ready CSV exports

## Dataset
- **5,000 candidates** | Jan 2022 – Dec 2024
- **8 departments:** Engineering, Data Science, Sales, Marketing, Finance, HR, Operations, Product
- **7 recruitment sources:** LinkedIn, Naukri, Employee Referral, Campus, Indeed, Company Website, Agency
- Features: application date, department, role level, source, gender, education, experience, interview score, salary offered/expected, hiring outcome

## Key Business Insights

| Metric | Value |
|--------|-------|
| Total Applications | 5,000 |
| Overall Hire Rate | 10.8% |
| Avg Time-to-Hire | 43 days |
| Best Source (hire rate) | Employee Referral |
| Top Predictor of Acceptance | Salary ratio (offered/expected) |
| GB Model Accuracy | 69.17% |
| GB AUC-ROC | 0.7747 |

## Hiring Funnel

| Stage | Count | Conversion |
|-------|-------|-----------|
| Applications | 5,000 | 100% |
| Screened | 3,194 | 63.9% |
| Interviewed | 2,288 | 45.8% |
| Offered | 1,263 | 25.3% |
| Accepted | 617 | 12.3% |
| Joined | 539 | 10.8% |

## Analysis Breakdown

### SQL Queries
- Full hiring funnel conversion rates
- Hire rate and avg time-to-hire by department
- Source effectiveness (applications vs hire rate)
- Offered vs expected salary gap by role level

### EDA (10 Visualisations)
1. Hiring funnel (horizontal bar with conversion %)
2. Hire rate by department
3. Source effectiveness (applications + hire rate)
4. Time-to-hire by department
5. Offered vs expected salary by role level
6. Gender diversity by department
7. Interview score distribution + offered vs not offered
8. Monthly application volume trend
9. ML feature importances
10. Hire rate by experience band

### ML — Offer Acceptance Prediction
- **Models:** Random Forest (200 trees) + Gradient Boosting (200 estimators)
- **Features:** department, role, source, gender, education, experience, interview score, salary ratio
- **Best Model:** Gradient Boosting → **69.17% accuracy | AUC-ROC: 0.7747**
- **Top predictor:** Salary ratio (offered/expected) — biggest driver of acceptance decisions

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_data.py
python hiring_analysis.py
```

## Project Structure
```
hiring_analysis/
├── generate_data.py
├── hiring_analysis.py
├── hiring_data.csv
├── outputs/
│   ├── 01_hiring_funnel.png ... 10_experience_vs_hire_rate.png
│   ├── powerbi_funnel.csv
│   ├── powerbi_dept_stats.csv
│   ├── powerbi_source_stats.csv
│   └── powerbi_salary_analysis.csv
└── README.md
```

## Author
**Sharat Laha** | M.Tech in Data Science & Analytics, LPU
[LinkedIn](https://linkedin.com/in/sharatlaha) | [GitHub](https://github.com/sololevellingg)
