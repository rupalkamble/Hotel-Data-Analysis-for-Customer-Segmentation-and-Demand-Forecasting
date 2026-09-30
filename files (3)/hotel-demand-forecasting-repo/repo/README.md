# Hotel Customer Segmentation & Demand Forecasting

Segments hotel guests into actionable marketing personas using clustering, and forecasts weekly room-booking demand using regression and time-series models. Built as part of a data science internship project.

## Overview

Hotels need to know **who their guests are** and **how much demand to expect** to price rooms, staff appropriately, and run targeted marketing. This project tackles both:

1. **Customer segmentation** — K-Means and DBSCAN clustering on booking behavior (lead time, stay length, spend, loyalty) to identify distinct guest personas.
2. **Demand forecasting** — Linear Regression and ARIMA models to predict weekly booking volume, accounting for seasonality.

**Dataset:** 6,000 synthetic hotel bookings spanning Jan 2023–Dec 2024, generated to reflect realistic booking-system fields (lead time, stay duration, room type, booking channel, guest country, ADR, cancellations, loyalty status). See [`data/hotel_bookings.csv`](data/hotel_bookings.csv).

## Repo Structure

```
├── data/
│   └── hotel_bookings.csv                          # synthetic booking dataset
├── notebooks/
│   └── Hotel_Segmentation_Demand_Forecasting.ipynb  # full analysis, executed end-to-end
├── reports/
│   └── Hotel_Analysis_Documentation_Report.docx     # write-up: methodology, findings, evaluation
└── images/                                          # charts exported from the notebook
```

## Methods

| Task | Techniques |
|---|---|
| Preprocessing | Missing-value imputation, IQR outlier capping, feature engineering (month, weekend flag, total guests), standardization |
| Segmentation | K-Means (k selected via elbow method + silhouette score), DBSCAN (density-based comparison) |
| Forecasting | Linear Regression (calendar features), ARIMA (weekly time series), seasonal decomposition |
| Evaluation | Silhouette score for clustering; MAE/RMSE for forecasting, on a held-out 10-week test period |

## Key Results

**Three customer segments identified:**

| Segment | Size | Avg ADR | Loyalty % | Description |
|---|---|---|---|---|
| Loyal Mid-Spend | 26% | $136 | 100% | Existing repeat-customer base |
| Budget Majority | 54% | $125 | 0% | Core acquisition target |
| High-Value Suite | 19% | $315 | 20% | Premium upsell candidates |

![K-Means clusters](images/kmeans_clusters_pca.png)

**Demand forecasting** (final 10-week holdout, weekly bookings):

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 5.97 | 6.96 |
| ARIMA(2,1,1) | 6.71 | 7.78 |

![Forecast comparison](images/forecast_comparison.png)

Linear Regression's explicit month-of-year features captured the seasonal pattern slightly better than ARIMA on this ~2-year history. Full discussion of trade-offs, challenges, and next steps (SARIMA, external regressors, XGBoost) is in the [documentation report](reports/Hotel_Analysis_Documentation_Report.docx).

## Running It

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter
jupyter notebook notebooks/Hotel_Segmentation_Demand_Forecasting.ipynb
```

## Tools

Python · Pandas · Scikit-learn (K-Means, DBSCAN, Linear Regression) · Statsmodels (ARIMA, seasonal decomposition) · Matplotlib · Seaborn
