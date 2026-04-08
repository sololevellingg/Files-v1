# generate_data.py — Marketing Campaign Dataset Generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N = 8000
START = datetime(2022, 1, 1)
END   = datetime(2024, 12, 31)

CHANNELS    = ["Email","Social Media","PPC","SEO","Influencer","Content Marketing","SMS"]
SEGMENTS    = ["New Customers","Returning","Premium","At-Risk","Dormant"]
OBJECTIVES  = ["Brand Awareness","Lead Generation","Conversion","Retention","Upsell"]
PRODUCTS    = ["Electronics","Fashion","Beauty","Home & Living","Sports","Food & Beverage"]
REGIONS     = ["North","South","East","West","Central"]
AB_GROUPS   = ["A","B"]  # A=control, B=treatment

channel_probs   = [0.25,0.22,0.18,0.12,0.10,0.08,0.05]
segment_probs   = [0.30,0.28,0.18,0.14,0.10]
objective_probs = [0.20,0.25,0.30,0.15,0.10]

# Channel performance profiles
channel_profiles = {
    "Email":              {"base_ctr":0.22, "base_cvr":0.045, "cost_per_k":800,   "roi_mult":3.2},
    "Social Media":       {"base_ctr":0.18, "base_cvr":0.032, "cost_per_k":1200,  "roi_mult":2.8},
    "PPC":                {"base_ctr":0.35, "base_cvr":0.065, "cost_per_k":3500,  "roi_mult":2.2},
    "SEO":                {"base_ctr":0.28, "base_cvr":0.055, "cost_per_k":500,   "roi_mult":4.5},
    "Influencer":         {"base_ctr":0.15, "base_cvr":0.028, "cost_per_k":2500,  "roi_mult":2.0},
    "Content Marketing":  {"base_ctr":0.12, "base_cvr":0.022, "cost_per_k":600,   "roi_mult":3.8},
    "SMS":                {"base_ctr":0.30, "base_cvr":0.038, "cost_per_k":400,   "roi_mult":2.5},
}

rows = []
for i in range(N):
    channel   = np.random.choice(CHANNELS, p=channel_probs)
    segment   = np.random.choice(SEGMENTS, p=segment_probs)
    objective = np.random.choice(OBJECTIVES, p=objective_probs)
    product   = np.random.choice(PRODUCTS)
    region    = np.random.choice(REGIONS)
    ab_group  = np.random.choice(AB_GROUPS, p=[0.5, 0.5])

    start_dt  = START + timedelta(days=int(np.random.uniform(0,(END-START).days)))
    duration  = np.random.randint(7, 60)
    end_dt    = start_dt + timedelta(days=duration)

    profile    = channel_profiles[channel]
    budget     = int(np.random.uniform(10000, 500000))
    impressions = int(budget / profile["cost_per_k"] * 1000 * np.random.uniform(0.8, 1.2))

    # B group gets 15% boost (treatment effect)
    ab_boost  = 1.15 if ab_group == "B" else 1.0
    noise     = np.random.uniform(0.85, 1.15)

    ctr       = min(profile["base_ctr"] * ab_boost * noise, 0.8)
    clicks    = int(impressions * ctr)
    cvr       = min(profile["base_cvr"] * ab_boost * noise, 0.5)
    conversions = int(clicks * cvr)
    revenue   = int(conversions * np.random.uniform(500, 5000))
    roi       = round((revenue - budget) / budget * 100, 2)
    cpc       = round(budget / clicks, 2) if clicks > 0 else 0
    cpa       = round(budget / conversions, 2) if conversions > 0 else 0

    # Campaign success: ROI > 100% and conversions > 50
    success   = (roi > 100) and (conversions > 50)

    rows.append({
        "campaign_id":   f"CAMP{str(i+1).zfill(5)}",
        "start_date":    start_dt.strftime("%Y-%m-%d"),
        "end_date":      end_dt.strftime("%Y-%m-%d"),
        "duration_days": duration,
        "channel":       channel,
        "segment":       segment,
        "objective":     objective,
        "product":       product,
        "region":        region,
        "ab_group":      ab_group,
        "budget":        budget,
        "impressions":   impressions,
        "clicks":        clicks,
        "ctr":           round(ctr * 100, 2),
        "conversions":   conversions,
        "cvr":           round(cvr * 100, 2),
        "revenue":       revenue,
        "roi":           roi,
        "cpc":           cpc,
        "cpa":           cpa,
        "success":       success,
    })

df = pd.DataFrame(rows)
df.to_csv("campaign_data.csv", index=False)
print(f"Campaigns      : {len(df):,}")
print(f"Total Budget   : ₹{df['budget'].sum()/1e7:.1f} Cr")
print(f"Total Revenue  : ₹{df['revenue'].sum()/1e7:.1f} Cr")
print(f"Avg ROI        : {df['roi'].mean():.1f}%")
print(f"Success Rate   : {df['success'].mean()*100:.1f}%")
print(f"A/B Split      : A={len(df[df['ab_group']=='A']):,} | B={len(df[df['ab_group']=='B']):,}")
