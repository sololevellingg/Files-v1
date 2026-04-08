# Marketing Campaign Effectiveness Analysis

## Project Overview
Comprehensive marketing analytics project analysing 8,000 campaigns across 7 channels (2022–2024). Covers ROI analysis, A/B testing with statistical significance testing, CPA optimisation, customer segment performance, and an ML model predicting campaign success using Logistic Regression and Gradient Boosting.

## Tools & Technologies
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy
- **SQL:** SQLite — channel ROI, segment performance, monthly trends, A/B summary
- **Statistics:** Two-sample t-tests, Cohen's d effect size, p-value significance testing
- **ML:** Logistic Regression + Gradient Boosting (campaign success prediction)
- **BI:** 4 Power BI-ready CSV exports

## Dataset
- **8,000 campaigns** | Jan 2022 – Dec 2024
- **7 channels:** Email, Social Media, PPC, SEO, Influencer, Content Marketing, SMS
- **5 customer segments:** New, Returning, Premium, At-Risk, Dormant
- **5 objectives:** Brand Awareness, Lead Generation, Conversion, Retention, Upsell
- Features: budget, impressions, clicks, CTR, conversions, CVR, revenue, ROI, CPC, CPA, A/B group

## Key Business Insights

| Metric | Value |
|--------|-------|
| Total Budget | ₹202.7 Cr |
| Total Revenue | ₹7,244.9 Cr |
| Avg CTR | 24.85% |
| Avg CVR | 4.63% |
| Best Channel (ROI) | SEO (9,689% ROI) |
| A/B Test CTR lift | +14.0% (p<0.0001) |
| A/B Test CVR lift | +14.0% (p<0.0001) |
| GB Model Accuracy | 98.25% |
| GB AUC-ROC | 0.9685 |

## Analysis Breakdown

### SQL Queries
- Channel performance (ROI, CTR, CVR, CPA)
- Segment revenue and ROI comparison
- Monthly budget vs revenue trends
- A/B group summary statistics

### A/B Testing (Statistical)
- Two-sample t-tests for CTR, CVR, and ROI
- Cohen's d effect size calculation
- All metrics statistically significant (p<0.0001)
- Treatment group (B) outperforms control (A) by 14% on CTR and CVR

### EDA (10 Visualisations)
1. Channel ROI ranking (horizontal bar)
2. Budget vs Revenue by channel
3. A/B test results with error bars and significance
4. CTR vs CVR scatter (bubble = revenue)
5. ROI by customer segment
6. Monthly budget vs revenue trend
7. CPA heatmap — channel vs segment
8. ROI distribution + by objective
9. ML feature importances
10. Channel efficiency quadrant (CTR vs ROI)

### ML — Campaign Success Prediction
- **Logistic Regression:** 98.25% accuracy | AUC: 0.9625
- **Gradient Boosting:** 98.25% accuracy | AUC: 0.9685
- **Top predictors:** CVR, CTR, impressions, budget

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
python generate_data.py
python marketing_analysis.py
```

## Author
**Sharat Laha** | M.Tech in Data Science & Analytics, LPU
[LinkedIn](https://linkedin.com/in/sharatlaha) | [GitHub](https://github.com/sololevellingg)
