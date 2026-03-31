# =============================================================================
# model.py  —  Natural Gas Price Model (shared by analysis.py and gui.py)
# OLS regression: price = a + b*t + A*sin(2πt) + B*cos(2πt)
# where t = years elapsed since Oct 2020
# =============================================================================

import math
import numpy as np
from datetime import datetime

# ── Raw monthly price data ($/MMBtu) ─────────────────────────────────────────
RAW_DATA = [
    ("10/31/20", 10.10), ("11/30/20", 10.30), ("12/31/20", 11.00),
    ("01/31/21", 10.90), ("02/28/21", 10.90), ("03/31/21", 10.90),
    ("04/30/21", 10.40), ("05/31/21",  9.84), ("06/30/21", 10.00),
    ("07/31/21", 10.10), ("08/31/21", 10.30), ("09/30/21", 10.20),
    ("10/31/21", 10.10), ("11/30/21", 11.20), ("12/31/21", 11.40),
    ("01/31/22", 11.50), ("02/28/22", 11.80), ("03/31/22", 11.50),
    ("04/30/22", 10.70), ("05/31/22", 10.70), ("06/30/22", 10.40),
    ("07/31/22", 10.50), ("08/31/22", 10.40), ("09/30/22", 10.80),
    ("10/31/22", 11.00), ("11/30/22", 11.60), ("12/31/22", 11.60),
    ("01/31/23", 12.10), ("02/28/23", 11.70), ("03/31/23", 12.00),
    ("04/30/23", 11.50), ("05/31/23", 11.20), ("06/30/23", 10.90),
    ("07/31/23", 11.40), ("08/31/23", 11.10), ("09/30/23", 11.50),
    ("10/31/23", 11.80), ("11/30/23", 12.20), ("12/31/23", 12.80),
    ("01/31/24", 12.60), ("02/29/24", 12.40), ("03/31/24", 12.70),
    ("04/30/24", 12.10), ("05/31/24", 11.40), ("06/30/24", 11.50),
    ("07/31/24", 11.60), ("08/31/24", 11.50), ("09/30/24", 11.80),
]

T0           = datetime(2020, 10, 31)   # reference date (t = 0)
DATA_END     = datetime(2024,  9, 30)   # last training observation
FORECAST_END = datetime(2025,  9, 30)   # maximum forecast horizon

def to_years(dt: datetime) -> float:
    """Convert datetime to fractional years since T0."""
    return (dt - T0).days / 365.25

def build_row(t: float) -> list:
    """Build design-matrix row: [1, t, sin(2πt), cos(2πt)]."""
    w = 2 * math.pi * t
    return [1.0, t, math.sin(w), math.cos(w)]

# ── Fit OLS model ─────────────────────────────────────────────────────────────
_ts     = [to_years(datetime.strptime(d, "%m/%d/%y")) for d, _ in RAW_DATA]
_prices = [p for _, p in RAW_DATA]

X_train = np.array([build_row(t) for t in _ts])
y_train = np.array(_prices)

BETA, residuals, rank, sv = np.linalg.lstsq(X_train, y_train, rcond=None)

# ── Model metrics ─────────────────────────────────────────────────────────────
y_pred_train = X_train @ BETA
_resid       = y_train - y_pred_train
SS_res       = float(np.sum(_resid ** 2))
SS_tot       = float(np.sum((y_train - np.mean(y_train)) ** 2))

R2   = 1 - SS_res / SS_tot
MAE  = float(np.mean(np.abs(_resid)))
RMSE = float(np.sqrt(np.mean(_resid ** 2)))
N    = len(y_train)

# 95% confidence interval width (±1.96 * residual std)
RESID_STD = float(np.std(_resid))
CI_95     = 1.96 * RESID_STD

# Derived model properties
AMPLITUDE  = float(np.sqrt(BETA[2]**2 + BETA[3]**2))
TREND_YR   = float(BETA[1])
PHASE_DEG  = float(math.degrees(math.atan2(BETA[2], BETA[3])))

def predict(dt: datetime) -> tuple:
    """
    Predict price for a given datetime.
    Returns (price, lower_95, upper_95, is_forecast).
    Raises ValueError for out-of-range dates.
    """
    if dt < T0:
        raise ValueError(f"Date is before data start ({T0.date()})")
    if dt > FORECAST_END:
        raise ValueError(f"Beyond forecast limit ({FORECAST_END.date()})")
    t     = to_years(dt)
    price = float(np.array(build_row(t)) @ BETA)
    is_fc = dt > DATA_END
    # CI widens linearly beyond training data
    extra = max(0, to_years(dt) - to_years(DATA_END))
    ci    = CI_95 * (1 + extra * 0.5)
    return price, price - ci, price + ci, is_fc
