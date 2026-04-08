# Myntra E-Commerce Analysis

## Project Overview
Comprehensive e-commerce analytics project on Myntra fashion platform data — covering revenue analysis, customer segmentation, RFM modelling, K-Means clustering, festive season impact, and return rate analysis across 12,000 orders and 2,000 customers (2022–2024).

## Tools & Technologies
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **SQL:** SQLite — revenue queries, YoY analysis, city/category breakdowns
- **ML:** K-Means Clustering, RFM Segmentation, Silhouette Score optimisation
- **BI:** 4 Power BI-ready CSV exports

## Dataset
- **12,000 orders** | **2,000 customers** | Jan 2022 – Dec 2024
- 7 product categories: Women's Clothing, Footwear, Men's Clothing, Sports & Fitness, Accessories, Beauty, Kids
- Features: order date, category, brand, MRP, discount %, price paid, rating, return status, delivery days, city

## Key Business Insights

| Metric | Value |
|--------|-------|
| Total Revenue | ₹21.49M |
| Avg Order Value | ₹1,791 |
| Overall Return Rate | 11.4% |
| Avg Customer Rating | 3.98 / 5.0 |
| Top Category | Women's Clothing (₹6.45M) |
| Festive Season Share | 25.6% of annual revenue |
| Champion Customers | 473 (23.7% of base) |

## Analysis Breakdown

### SQL Analysis
- Revenue, AOV, return rate, and avg discount by category
- YoY performance (2022–2024)
- Top 10 cities by revenue

### EDA (10 Visualisations)
1. Revenue by category (horizontal bar)
2. Monthly revenue trend (area chart)
3. Discount vs revenue heatmap
4. Avg rating by category
5. Top 10 cities by revenue
6. Gender split by category
7. Return rate by category
8. RFM segment revenue
9. K-Means customer clusters (scatter)
10. Festive season impact analysis

### RFM Segmentation
Customers scored on Recency, Frequency, Monetary value (1–5 each) and segmented into:
- **Champions** (473) — high value, recent, frequent buyers
- **Loyal** (572) — consistent buyers
- **Needs Attention** (528) — declining engagement
- **At Risk** (422) — low recency, low spend

### K-Means Clustering
- Optimal k=2 identified via Silhouette Score (0.3711)
- **Cluster 0:** High-value active customers (avg spend ₹14,713, freq 7.9)
- **Cluster 1:** Low-engagement customers (avg spend ₹6,783, freq 4.1)

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_data.py
python myntra_analysis.py
```

## Project Structure
```
myntra_analysis/
├── generate_data.py
├── myntra_analysis.py
├── customers.csv
├── orders.csv
├── outputs/
│   ├── 01_revenue_by_category.png
│   ├── 02_monthly_revenue_trend.png
│   ├── 03_discount_revenue_heatmap.png
│   ├── 04_ratings_by_category.png
│   ├── 05_revenue_by_city.png
│   ├── 06_gender_split_by_category.png
│   ├── 07_return_rate_by_category.png
│   ├── 08_rfm_segments.png
│   ├── 09_kmeans_clusters.png
│   ├── 10_festive_season_analysis.png
│   ├── powerbi_category_summary.csv
│   ├── powerbi_monthly_revenue.csv
│   ├── powerbi_rfm_segments.csv
│   └── powerbi_city_revenue.csv
└── README.md
```

## Author
**Sharat Laha** | M.Tech in Data Science & Analytics, LPU
[LinkedIn](https://linkedin.com/in/sharatlaha) | [GitHub](https://github.com/sololevellingg)
