# Natural Gas Price Estimator

## Project Overview
An end-to-end data science project that models and forecasts natural gas prices using **OLS regression with trend and Fourier seasonality components**. The project includes a full EDA & model evaluation pipeline and a production-grade interactive desktop GUI application built with Tkinter.

## Tools & Technologies
- **Python:** NumPy, Pandas, Matplotlib, Tkinter
- **Statistical Modelling:** OLS Regression (numpy.linalg.lstsq), Fourier Seasonality
- **Model Evaluation:** R², MAE, RMSE, 95% Confidence Intervals
- **Visualisation:** 6 EDA/evaluation charts + interactive GUI spark chart
- **Export:** CSV data export from both analysis script and GUI

## Model
The price model takes the form:

```
price(t) = a + b·t + A·sin(2πt) + B·cos(2πt)
```

Where:
- `t` = fractional years since Oct 2020
- `a` = intercept (base price level)
- `b` = linear trend (annual price drift)
- `A`, `B` = Fourier coefficients capturing annual seasonality

Fitted using Ordinary Least Squares on 48 monthly observations (Oct 2020 – Sep 2024).

## Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² | **0.9283** | 92.8% of price variance explained |
| MAE | $0.1512 | Average error of ~15 cents/MMBtu |
| RMSE | $0.2008 | Root mean squared error |
| 95% CI | ±$0.3935 | Confidence band on training data |
| Trend | +$0.55/yr | Annual price increase |
| Seasonality | ±$0.69 | Peak-to-trough seasonal swing |

## Project Structure
```
nat_gas_estimator/
├── model.py          # Shared OLS model, coefficients, predict()
├── analysis.py       # Full EDA + model evaluation + chart generation
├── gui.py            # Interactive desktop GUI application
├── outputs/
│   ├── 01_actual_vs_fitted.png
│   ├── 02_forecast.png
│   ├── 03_residuals.png
│   ├── 04_seasonal_pattern.png
│   ├── 05_yoy_comparison.png
│   ├── 06_model_metrics.png
│   └── nat_gas_data_and_forecast.csv
└── README.md
```

## Key Features

### analysis.py
- Full EDA pipeline with 6 publication-quality charts
- Model evaluation metrics (R², MAE, RMSE, 95% CI)
- YoY price comparison, seasonal pattern analysis
- Residual distribution analysis
- CSV export of historical + forecast data

### gui.py (Desktop Application)
- Interactive spark chart with fitted curve, forecast, and CI bands
- Date spinbox input supporting historical estimates and future forecasts
- Real-time price estimation with 95% confidence interval display
- Widening CI beyond training data (uncertainty grows with forecast horizon)
- One-click CSV export of results
- Model coefficient dashboard (R², MAE, RMSE, trend, amplitude)

## Key Insights
- Natural gas prices show a clear **annual seasonality** — peaking in winter (Nov–Jan) and dipping in summer (May–Jun)
- Prices have risen at **~$0.55/yr** over the 2020–2024 period
- The model achieves **R² = 0.9283**, explaining 92.8% of price variance with just 4 parameters
- Forecast confidence intervals widen linearly beyond the training window (Sep 2024)

## How to Run
```bash
pip install numpy pandas matplotlib
# Run EDA + model evaluation
python analysis.py
# Launch interactive GUI
python gui.py
```

## Author
**Sharat Laha** | M.Tech in Data Science & Analytics, LPU
[LinkedIn](https://linkedin.com/in/sharatlaha) | [GitHub](https://github.com/sololevellingg)
