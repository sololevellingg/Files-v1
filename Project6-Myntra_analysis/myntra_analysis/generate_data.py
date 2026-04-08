# generate_data.py — Myntra E-Commerce Dataset Generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 2000
N_ORDERS    = 12000
START_DATE  = datetime(2022, 1, 1)
END_DATE    = datetime(2024, 12, 31)

CATEGORIES = {
    "Men's Clothing":    {"brands": ["H&M","Zara","Roadster","Jack & Jones","US Polo"], "price_range": (499, 4999),  "return_rate": 0.12},
    "Women's Clothing":  {"brands": ["W","Biba","AND","Aurelia","Vero Moda"],           "price_range": (599, 5999),  "return_rate": 0.18},
    "Footwear":          {"brands": ["Nike","Adidas","Puma","Bata","Metro"],             "price_range": (799, 8999),  "return_rate": 0.10},
    "Accessories":       {"brands": ["Titan","Fossil","Fastrack","Baggit","Lavie"],      "price_range": (299, 3999),  "return_rate": 0.08},
    "Sports & Fitness":  {"brands": ["Nike","Adidas","Reebok","Decathlon","Puma"],       "price_range": (399, 6999),  "return_rate": 0.07},
    "Beauty":            {"brands": ["Lakme","Maybelline","L'Oreal","Nykaa","MAC"],      "price_range": (199, 2999),  "return_rate": 0.05},
    "Kids":              {"brands": ["Mothercare","H&M Kids","Carter's","Marks & Spencer","Hopscotch"], "price_range": (299, 2499), "return_rate": 0.09},
}

CITIES   = ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad","Jaipur","Lucknow"]
GENDERS  = ["Male","Female","Other"]
AGE_DIST = {"18-24":0.28, "25-34":0.35, "35-44":0.20, "45-54":0.12, "55+":0.05}

# ── Customers ─────────────────────────────────────────────────────────────────
age_groups = list(AGE_DIST.keys())
age_probs  = list(AGE_DIST.values())

customers = pd.DataFrame({
    "customer_id":   [f"C{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS+1)],
    "gender":        np.random.choice(GENDERS, N_CUSTOMERS, p=[0.45, 0.52, 0.03]),
    "age_group":     np.random.choice(age_groups, N_CUSTOMERS, p=age_probs),
    "city":          np.random.choice(CITIES, N_CUSTOMERS),
    "is_premium":    np.random.choice([True, False], N_CUSTOMERS, p=[0.25, 0.75]),
    "signup_date":   [START_DATE - timedelta(days=int(np.random.uniform(0, 730)))
                      for _ in range(N_CUSTOMERS)],
})

# ── Orders ────────────────────────────────────────────────────────────────────
order_rows = []
for i in range(N_ORDERS):
    cat     = np.random.choice(list(CATEGORIES.keys()),
                               p=[0.22,0.28,0.15,0.12,0.08,0.10,0.05])
    info    = CATEGORIES[cat]
    brand   = np.random.choice(info["brands"])
    cust_id = f"C{str(np.random.randint(1, N_CUSTOMERS+1)).zfill(5)}"

    # Order date with seasonality (festival boost Oct-Dec)
    rand_days = int(np.random.uniform(0, (END_DATE - START_DATE).days))
    order_dt  = START_DATE + timedelta(days=rand_days)
    if order_dt.month in [10, 11, 12]:
        if np.random.random() < 0.3:
            order_dt = order_dt  # keep more orders in festive season

    mrp       = int(np.random.uniform(*info["price_range"]))
    discount  = round(np.random.choice([0.1,0.2,0.3,0.4,0.5,0.6,0.7],
                      p=[0.05,0.10,0.20,0.25,0.20,0.15,0.05]), 2)
    price_paid = int(mrp * (1 - discount))
    rating     = round(np.random.choice([1,2,3,4,5],
                       p=[0.03,0.07,0.15,0.40,0.35]), 1)
    returned   = np.random.random() < info["return_rate"]
    delivery   = np.random.randint(2, 8)

    order_rows.append({
        "order_id":      f"ORD{str(i+1).zfill(6)}",
        "customer_id":   cust_id,
        "order_date":    order_dt.strftime("%Y-%m-%d"),
        "category":      cat,
        "brand":         brand,
        "mrp":           mrp,
        "discount_pct":  int(discount * 100),
        "price_paid":    price_paid,
        "rating":        rating,
        "returned":      returned,
        "delivery_days": delivery,
        "city":          np.random.choice(CITIES),
    })

orders = pd.DataFrame(order_rows)

# ── Save ──────────────────────────────────────────────────────────────────────
customers.to_csv("customers.csv", index=False)
orders.to_csv("orders.csv", index=False)
print(f"Customers : {len(customers):,} rows")
print(f"Orders    : {len(orders):,} rows")
print(f"Date range: {orders['order_date'].min()} → {orders['order_date'].max()}")
print(f"Revenue   : ₹{orders['price_paid'].sum():,.0f}")
