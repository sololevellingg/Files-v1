# =============================================================================
# analysis.py  —  Natural Gas Price Estimator: Full EDA & Model Evaluation
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from model import (RAW_DATA, T0, DATA_END, FORECAST_END,
                   BETA, R2, MAE, RMSE, CI_95, AMPLITUDE,
                   TREND_YR, PHASE_DEG, _ts, _prices,
                   X_train, y_train, y_pred_train, _resid,
                   to_years, build_row, predict, N)

os.makedirs("outputs", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
BLUE  = "#1A5276"
RED   = "#C0392B"
GREEN = "#1D8348"
AMBER = "#D68910"
GREY  = "#7F8C8D"

# ── Build DataFrame ───────────────────────────────────────────────────────────
dates  = [datetime.strptime(d, "%m/%d/%y") for d, _ in RAW_DATA]
df     = pd.DataFrame({"date": dates, "price": _prices,
                        "t": _ts, "fitted": y_pred_train,
                        "residual": _resid})
df["month"] = df["date"].dt.month
df["year"]  = df["date"].dt.year
df["season"] = df["month"].map({
    12:"Winter", 1:"Winter", 2:"Winter",
    3:"Spring",  4:"Spring", 5:"Spring",
    6:"Summer",  7:"Summer", 8:"Summer",
    9:"Autumn", 10:"Autumn", 11:"Autumn"
})

print("=" * 60)
print("  NATURAL GAS PRICE ESTIMATOR — Analysis Report")
print("=" * 60)
print(f"\n  Dataset    : {N} monthly observations")
print(f"  Period     : {dates[0].strftime('%b %Y')} → {dates[-1].strftime('%b %Y')}")
print(f"  Price range: ${min(_prices):.2f} – ${max(_prices):.2f} / MMBtu")
print(f"\n  MODEL COEFFICIENTS")
print(f"  Intercept (a)   : {BETA[0]:.4f}")
print(f"  Trend (b/yr)    : +{TREND_YR:.4f}  → prices rise ~${TREND_YR:.2f}/yr")
print(f"  Seasonal amp    : {AMPLITUDE:.4f}  → ±${AMPLITUDE:.2f} seasonal swing")
print(f"  Phase shift     : {PHASE_DEG:.1f}°")
print(f"\n  MODEL PERFORMANCE")
print(f"  R²   : {R2:.4f}  ({R2*100:.1f}% variance explained)")
print(f"  MAE  : ${MAE:.4f}")
print(f"  RMSE : ${RMSE:.4f}")
print(f"  95% CI width: ±${CI_95:.4f}")

# ── Chart 1: Actual vs Fitted ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["price"],  "o-", color=BLUE,  linewidth=2,
        markersize=4, label="Actual price", zorder=3)
ax.plot(df["date"], df["fitted"], "--",  color=RED,   linewidth=2,
        label=f"OLS fitted (R²={R2:.3f})")
ax.fill_between(df["date"],
                df["fitted"] - CI_95, df["fitted"] + CI_95,
                alpha=0.12, color=RED, label="95% confidence band")
ax.set_title("Natural Gas Price — Actual vs OLS Fitted Model", fontsize=13, fontweight="bold")
ax.set_ylabel("Price ($/MMBtu)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.xticks(rotation=45)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/01_actual_vs_fitted.png", dpi=150)
plt.close()
print("\n  Saved: 01_actual_vs_fitted.png")

# ── Chart 2: Forecast ─────────────────────────────────────────────────────────
forecast_dates = pd.date_range(DATA_END, FORECAST_END, freq="MS")
f_prices, f_lo, f_hi = [], [], []
for fd in forecast_dates:
    p, lo, hi, _ = predict(fd)
    f_prices.append(p); f_lo.append(lo); f_hi.append(hi)

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(df["date"], df["price"], "o-", color=BLUE, linewidth=2,
        markersize=4, label="Historical (actual)", zorder=3)
ax.plot(df["date"], df["fitted"], "--", color=BLUE, linewidth=1.5,
        alpha=0.5, label="OLS fit (historical)")
ax.plot(forecast_dates, f_prices, "o--", color=RED, linewidth=2,
        markersize=5, label="Forecast (Oct 2024 – Sep 2025)")
ax.fill_between(forecast_dates, f_lo, f_hi,
                alpha=0.15, color=RED, label="95% CI (widening)")
ax.axvline(DATA_END, color=GREY, linestyle=":", linewidth=1.5)
ax.text(DATA_END + timedelta(days=5), ax.get_ylim()[0] + 0.1,
        "Forecast start", fontsize=8, color=GREY)
ax.set_title("Natural Gas Price Forecast — Oct 2024 to Sep 2025", fontsize=13, fontweight="bold")
ax.set_ylabel("Price ($/MMBtu)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/02_forecast.png", dpi=150)
plt.close()
print("  Saved: 02_forecast.png")

# ── Chart 3: Residuals ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(df["date"], df["residual"], color=[RED if r < 0 else BLUE for r in df["residual"]], width=20)
axes[0].axhline(0, color=GREY, linewidth=1)
axes[0].set_title("Residuals over Time", fontweight="bold")
axes[0].set_ylabel("Residual ($/MMBtu)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
axes[1].hist(df["residual"], bins=12, color=BLUE, edgecolor="white", alpha=0.8)
axes[1].axvline(0, color=RED, linewidth=1.5, linestyle="--")
axes[1].set_title("Residual Distribution", fontweight="bold")
axes[1].set_xlabel("Residual ($/MMBtu)")
axes[1].set_ylabel("Frequency")
plt.suptitle("Model Residual Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/03_residuals.png", dpi=150)
plt.close()
print("  Saved: 03_residuals.png")

# ── Chart 4: Seasonal pattern ─────────────────────────────────────────────────
monthly_avg = df.groupby("month")["price"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(month_names, monthly_avg.values, color=BLUE, edgecolor="white")
ax.bar_label(bars, labels=[f"${v:.2f}" for v in monthly_avg.values],
             padding=3, fontsize=8)
ax.axhline(monthly_avg.mean(), color=RED, linestyle="--", linewidth=1.5,
           label=f"Annual avg ${monthly_avg.mean():.2f}")
ax.set_title("Average Natural Gas Price by Month (2020–2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("Avg Price ($/MMBtu)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/04_seasonal_pattern.png", dpi=150)
plt.close()
print("  Saved: 04_seasonal_pattern.png")

# ── Chart 5: YoY comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
colors = [BLUE, GREEN, AMBER, RED]
for i, yr in enumerate([2021, 2022, 2023, 2024]):
    yr_df = df[df["year"] == yr]
    ax.plot(yr_df["month"], yr_df["price"], "o-",
            color=colors[i], linewidth=2, markersize=5, label=str(yr))
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title("Year-on-Year Price Comparison by Month", fontsize=13, fontweight="bold")
ax.set_ylabel("Price ($/MMBtu)")
ax.legend(title="Year")
plt.tight_layout()
plt.savefig("outputs/05_yoy_comparison.png", dpi=150)
plt.close()
print("  Saved: 05_yoy_comparison.png")

# ── Chart 6: Model metrics summary card ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")
metrics = [
    ["Metric",        "Value",          "Interpretation"],
    ["R²",            f"{R2:.4f}",      f"{R2*100:.1f}% variance explained"],
    ["MAE",           f"${MAE:.4f}",    "Avg absolute error per month"],
    ["RMSE",          f"${RMSE:.4f}",   "Root mean squared error"],
    ["95% CI",        f"±${CI_95:.4f}", "Confidence band on training data"],
    ["Trend",         f"+${TREND_YR:.4f}/yr", "Annual price increase"],
    ["Seasonality",   f"±${AMPLITUDE:.4f}", "Peak-to-trough seasonal swing"],
]
tbl = ax.table(cellText=metrics[1:], colLabels=metrics[0],
               cellLoc="center", loc="center",
               colWidths=[0.2, 0.2, 0.5])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#EBF5FB")
    cell.set_edgecolor("white")
ax.set_title("Model Performance Summary", fontsize=13,
             fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/06_model_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 06_model_metrics.png")

# ── Export CSV ────────────────────────────────────────────────────────────────
export_df = df[["date","price","fitted","residual"]].copy()
export_df.columns = ["date","actual_price","fitted_price","residual"]
forecast_df = pd.DataFrame({
    "date":          forecast_dates,
    "actual_price":  None,
    "fitted_price":  f_prices,
    "residual":      None,
    "lower_95":      f_lo,
    "upper_95":      f_hi,
    "is_forecast":   True,
})
export_df["is_forecast"] = False
final_export = pd.concat([export_df, forecast_df], ignore_index=True)
final_export.to_csv("outputs/nat_gas_data_and_forecast.csv", index=False)
print("  Saved: nat_gas_data_and_forecast.csv")

print(f"\n{'=' * 60}")
print("  ANALYSIS COMPLETE")
print(f"{'=' * 60}")
print(f"  R²: {R2:.4f}  |  MAE: ${MAE:.4f}  |  RMSE: ${RMSE:.4f}")
print(f"  Trend: +${TREND_YR:.4f}/yr  |  Seasonal amplitude: ±${AMPLITUDE:.4f}")
print(f"  6 charts + 1 CSV saved to /outputs")
print(f"{'=' * 60}")
