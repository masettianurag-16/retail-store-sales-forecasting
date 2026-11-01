# 🛒 Retail Store Sales Forecasting

An end-to-end time series analysis and forecasting project built with Python to predict weekly retail store sales. This repository covers data ingestion and merging, exploratory data analysis (EDA), time series decomposition, stationarity testing (ADF test), and model comparison between Seasonal Naive, ARIMA, and SARIMA.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Technologies Used](#-technologies-used)
- [Data Cleaning and Exploratory Data Analysis (EDA)](#-data-cleaning-and-exploratory-data-analysis-eda)
- [Time Series Modeling & Evaluation Metrics](#-time-series-modeling--evaluation-metrics)
- [Project Results](#-project-results)
- [How to Run the Jupyter Notebook](#-how-to-run-the-jupyter-notebook)
- [Future Improvements](#-future-improvements)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📖 Project Overview
Accurate sales forecasting is critical for large retail organizations. Predicting weekly demand helps store managers optimize inventory levels, prevent stockouts, schedule staffing effectively, and plan seasonal promotions. 

This project explores historical weekly sales data across multiple stores and departments, analyzes trends and seasonal patterns, and develops forecasting models to predict aggregate weekly sales for an upcoming 13-week quarter.

---

## 🎯 Problem Statement
Retail sales fluctuate significantly due to seasonal holidays (Thanksgiving, Christmas, Super Bowl), store promotions (markdowns), and macroeconomic indicators (unemployment, CPI, fuel prices). 

The goal of this project is to:
1. Identify historical sales trends and yearly seasonal cycles across stores and departments.
2. Check and enforce stationarity in time series data.
3. Train and compare baseline and statistical time series models (**Seasonal Naive**, **ARIMA**, and **SARIMA**) to find the most accurate predictor of weekly retail sales.

---

## 📊 Dataset Description
The analysis integrates data from three primary relational files:

1. **`sales.csv`**:
   - `store_id`: Store identifier.
   - `department`: Department identifier within the store.
   - `date`: Weekly sales date.
   - `weekly_sales`: Total sales for the given department in the given store for that week.
   - `is_holiday`: Boolean flag indicating whether the week contains a major promotional holiday.

2. **`stores.csv`**:
   - `store_id`: Unique identifier for each store.
   - `store_type`: Categorical classification of the store (e.g., Type A, B, C).
   - `size`: Physical floor space / square footage of the store.

3. **`features.csv`**:
   - `store_id`, `date`: Keys for merging.
   - `temperature`: Regional temperature during the week.
   - `fuel_price`: Regional cost of fuel.
   - `mark_down1` to `mark_down5`: Anonymized promotional markdowns.
   - `cpi`: Consumer Price Index.
   - `unemployment`: Regional unemployment rate.
   - `is_holiday`: Holiday status indicator.

The files are combined on `store_id` and `date` to generate `retail_sales_merged.csv`.

---

## 🛠️ Technologies Used
- **Programming Language:** Python
- **Data Manipulation & Processing:**
  - [Pandas](https://pandas.pydata.org/) – Data merging, aggregation, datetime indexing, and resampling (`W-SAT`)
  - [NumPy](https://numpy.org/) – Numerical operations and array transformations
- **Data Visualization:**
  - [Matplotlib](https://matplotlib.org/) – Line plots, bar charts, error residual plots
  - [Seaborn](https://seaborn.pydata.org/) – Statistical visualizations and styling
- **Time Series & Statistical Modeling:**
  - [Statsmodels](https://www.statsmodels.org/) – Time series decomposition (`seasonal_decompose`), stationarity testing (`adfuller`), autocorrelation functions (`plot_acf`, `plot_pacf`), and models (`ARIMA`, `SARIMAX`, `ExponentialSmoothing`)
- **Evaluation:**
  - [Scikit-learn](https://scikit-learn.org/) – Regression and time series error metrics (MAE, RMSE, MAPE)
- **Environment:**
  - [Jupyter Notebook](https://jupyter.org/) – Interactive execution and reporting

---

## 🔍 Data Cleaning and Exploratory Data Analysis (EDA)

### 1. Data Preprocessing & Cleaning
- **Merging Datasets:** Combined sales records with store metadata and regional economic features.
- **Date Handling:** Converted date strings to `pd.to_datetime` and set weekly frequency to `W-SAT` (ending on Saturdays).
- **Integrity Checks:** Verified missing values, investigated negative and zero sales records, and removed duplicate rows.
- **Time Series Aggregation:** Aggregated total weekly sales across all stores and departments to build the master time series.

### 2. Exploratory Data Analysis (EDA) Insights
- **Top Stores & Departments:** Identified the highest and lowest revenue-generating stores and departments via bar chart rankings.
- **Holiday vs. Non-Holiday Performance:** Holiday weeks exhibited noticeable spikes in average weekly sales compared to non-holiday weeks.
- **Store Type Analysis:** Significant variation in average sales based on store format and physical store size.
- **13-Week Rolling Average:** Smoothed out weekly volatility to reveal underlying multi-year sales trends.

### 3. Time Series Diagnostics
- **Classical Decomposition:** Decomposed sales into **Trend**, **Seasonal (52-week annual cycle)**, and **Residual** components using an additive model.
- **Stationarity Testing (Augmented Dickey-Fuller Test):**
  - Raw Series: Evaluated $p$-value for unit roots.
  - First-Order Differencing ($\Delta y_t = y_t - y_{t-1}$): Successfully removed trend and confirmed stationarity ($p < 0.05$).
- **Autocorrelation Analysis:** Inspected ACF and PACF plots up to 60 lags to guide order selection for autoregressive ($p$) and moving average ($q$) terms.

---

## 🤖 Time Series Modeling & Evaluation Metrics

The aggregate weekly sales series was partitioned chronologically:
- **Training Set:** Historical data up to the final 13 weeks.
- **Test Set:** The last 13 weeks (representing a 1-quarter out-of-sample forecast horizon).

### Models Evaluated:
1. **Seasonal Naive Baseline:** Uses the sales value from the corresponding week in the previous year ($t - 52$) as the forecast.
2. **ARIMA(1, 1, 1):** Autoregressive Integrated Moving Average model capturing linear trends and short-term autoregressive dynamics.
3. **SARIMA(1, 1, 1)(0, 1, 1, 52):** Seasonal ARIMA adding a 52-week seasonal differencing and seasonal moving-average parameter.

### Evaluation Metrics:
- **Mean Absolute Error (MAE):** Measures average magnitude of forecast errors in dollars.
- **Root Mean Squared Error (RMSE):** Penalizes larger forecasting mistakes heavily.
- **Mean Absolute Percentage Error (MAPE):** Expresses error as a percentage of actual sales for business interpretability.

---

## 📈 Project Results

The models were evaluated against the 13-week holdout test set:

| Model | MAE ($) | RMSE ($) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| 🥇 **Seasonal Naive** | **4,999,066.30** | **14,067,578.21** | **5.41%** |
| 🥈 **SARIMA(1, 1, 1)(0, 1, 1, 52)** | 9,641,851.27 | 16,942,877.94 | 11.13% |
| 🥉 **ARIMA(1, 1, 1)** | 11,620,019.03 | 22,416,552.92 | 11.97% |

### Key Takeaways:
- **Strong Annual Seasonality:** The **Seasonal Naive benchmark** achieved the lowest error (**5.41% MAPE**), demonstrating that year-over-year weekly seasonality is the single most dominant driver of aggregate retail sales.
- **SARIMA vs. Standard ARIMA:** SARIMA outperformed non-seasonal ARIMA significantly (reducing MAPE from 11.97% to 11.13%), confirming the necessity of seasonal parameters.
- **Residual Diagnostics:** Error analysis on the top-performing model revealed zero-centered residuals with normal distribution characteristics, with peak errors concentrated around major holiday shifts.

---

## 🚀 How to Run the Jupyter Notebook

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/retail-store-sales-forecasting.git
cd retail-store-sales-forecasting
```

### 2. Set Up a Virtual Environment (Recommended)
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn notebook
```

### 4. Ensure Data Files Are Present
Make sure the following CSV files are placed in the project directory:
- `sales.csv`
- `stores.csv`
- `features.csv`

### 5. Launch the Notebook
```bash
jupyter notebook "Retail Store Sales Forecasting.ipynb"
```
Execute the notebook cells sequentially from top to bottom (**Cell > Run All**).

---

## 🔮 Future Improvements
- [ ] **Incorporate Exogenous Regressors (SARIMAX):** Integrate promotional markdowns (`mark_down1-5`), fuel price, CPI, and temperature to evaluate incremental predictive power.
- [ ] **Department & Store-Level Forecasting:** Scale from aggregate national sales to hierarchical forecasting (store-by-store and department-by-department).
- [ ] **Machine Learning & Deep Learning:** Implement and benchmark machine learning models (LightGBM, XGBoost, CatBoost with lag features) and neural approaches (Prophet, N-BEATS, or LSTM).
- [ ] **Interactive Dashboard:** Build a [Streamlit](https://streamlit.io/) web dashboard allowing store managers to select individual stores and visualize forecast intervals.

---

