#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Forecasting models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Model evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

print("Libraries imported successfully")


# In[2]:


sales = pd.read_csv("sales.csv")
stores = pd.read_csv("stores.csv")
features = pd.read_csv("features.csv")

print("Data loaded successfully")


# In[3]:


print("Sales dataset:", sales.shape)
print("Stores dataset:", stores.shape)
print("Features dataset:", features.shape)

display(sales.head())
display(stores.head())
display(features.head())


# In[4]:


print("Sales information:")
sales.info()

print("\nStores information:")
stores.info()

print("\nFeatures information:")
features.info()


# In[5]:


print("Missing values in sales:")
print(sales.isnull().sum())

print("\nMissing values in stores:")
print(stores.isnull().sum())

print("\nMissing values in features:")
print(features.isnull().sum())


# In[6]:


print("Duplicate rows in sales:", sales.duplicated().sum())
print("Duplicate rows in stores:", stores.duplicated().sum())
print("Duplicate rows in features:", features.duplicated().sum())


# In[7]:


sales["date"] = pd.to_datetime(sales["date"])
features["date"] = pd.to_datetime(features["date"])

print("Sales date range:", sales["date"].min(), "to", sales["date"].max())
print("Features date range:", features["date"].min(), "to", features["date"].max())


# In[8]:


print("Unique stores in sales:", sales["store_id"].nunique())
print("Unique departments:", sales["department"].nunique())
print("Unique sales dates:", sales["date"].nunique())

print("Unique stores in store metadata:", stores["store_id"].nunique())
print("Unique stores in features:", features["store_id"].nunique())


# In[9]:


print("Minimum weekly sales:", sales["weekly_sales"].min())
print("Maximum weekly sales:", sales["weekly_sales"].max())
print("Average weekly sales:", sales["weekly_sales"].mean())

print("Negative sales records:", (sales["weekly_sales"] < 0).sum())
print("Zero sales records:", (sales["weekly_sales"] == 0).sum())


# In[10]:


# Merge using store_id and date
df = pd.merge(
    sales,
    features,
    on=["store_id", "date"],
    how="left",
    suffixes=("_sales", "_features")
)

print("Shape after merging sales and features:", df.shape)
display(df.head())


# In[11]:


df = pd.merge(
    df,
    stores,
    on="store_id",
    how="left"
)

print("Final dataset shape:", df.shape)
display(df.head())


# In[12]:


print("Missing values after merging:")
print(df.isnull().sum())

print("\nUnique stores:", df["store_id"].nunique())
print("Unique departments:", df["department"].nunique())
print("Unique dates:", df["date"].nunique())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())


# In[13]:


df.to_csv("retail_sales_merged.csv", index=False)

print("Merged dataset saved successfully")


# In[14]:


print("Sales Summary:")
display(df["weekly_sales"].describe())

print("\nTotal Sales:", df["weekly_sales"].sum())
print("Average Sales:", df["weekly_sales"].mean())
print("Median Sales:", df["weekly_sales"].median())


# In[15]:


store_sales = (
    df.groupby("store_id")["weekly_sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(14, 5))
store_sales.plot(kind="bar")

plt.title("Total Sales by Store")
plt.xlabel("Store ID")
plt.ylabel("Total Weekly Sales")
plt.xticks(rotation=90)
plt.show()


# In[16]:


department_sales = (
    df.groupby("department")["weekly_sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 5))
department_sales.plot(kind="bar")

plt.title("Total Sales by Department")
plt.xlabel("Department")
plt.ylabel("Total Weekly Sales")
plt.xticks(rotation=0)
plt.show()


# In[17]:


holiday_sales = df.groupby("is_holiday_sales")["weekly_sales"].mean()

print(holiday_sales)

holiday_sales.plot(kind="bar", figsize=(8, 5))
plt.title("Average Sales: Holiday vs Non-Holiday")
plt.xlabel("Holiday Indicator")
plt.ylabel("Average Weekly Sales")
plt.xticks([0, 1], ["Non-Holiday", "Holiday"], rotation=0)
plt.show()


# In[18]:


store_type_sales = df.groupby("store_type")["weekly_sales"].mean()

print(store_type_sales)

store_type_sales.plot(kind="bar", figsize=(8, 5))
plt.title("Average Sales by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Average Weekly Sales")
plt.show()


# In[19]:


# Aggregate sales across all stores and departments
weekly_sales = (
    df.groupby("date")["weekly_sales"]
    .sum()
    .sort_index()
)

# Set weekly frequency
weekly_sales = weekly_sales.asfreq("W-SAT")

print("Number of weekly observations:", len(weekly_sales))
print("Start date:", weekly_sales.index.min())
print("End date:", weekly_sales.index.max())

display(weekly_sales.head())


# In[20]:


plt.figure(figsize=(15, 6))

plt.plot(
    weekly_sales.index,
    weekly_sales.values,
    marker="o",
    markersize=3
)

plt.title("Weekly Retail Sales Across All Stores")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.grid(True)

plt.show()


# In[21]:


rolling_mean = weekly_sales.rolling(window=13).mean()

plt.figure(figsize=(15, 6))

plt.plot(
    weekly_sales.index,
    weekly_sales,
    label="Actual Weekly Sales",
    alpha=0.5
)

plt.plot(
    rolling_mean.index,
    rolling_mean,
    label="13-Week Rolling Average",
    color="red"
)

plt.title("Weekly Sales with 13-Week Rolling Average")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.grid(True)

plt.show()


# In[22]:


print("Mean weekly sales:", weekly_sales.mean())
print("Median weekly sales:", weekly_sales.median())
print("Standard deviation:", weekly_sales.std())
print("Minimum weekly sales:", weekly_sales.min())
print("Maximum weekly sales:", weekly_sales.max())


# In[23]:


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(
    weekly_sales,
    model="additive",
    period=52
)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


# In[24]:


fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(weekly_sales)
axes[0].set_title("Original Weekly Sales")

axes[1].plot(trend)
axes[1].set_title("Trend Component")

axes[2].plot(seasonal)
axes[2].set_title("Seasonal Component")

axes[3].plot(residual)
axes[3].set_title("Residual Component")

plt.tight_layout()
plt.show()


# In[25]:


print("Trend component:")
print("Start:", trend.dropna().iloc[0])
print("End:", trend.dropna().iloc[-1])

print("\nSeasonal component:")
print("Minimum:", seasonal.min())
print("Maximum:", seasonal.max())

print("\nResidual component:")
print("Mean:", residual.mean())
print("Standard deviation:", residual.std())


# In[26]:


from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(weekly_sales)

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:")

for key, value in adf_result[4].items():
    print(f"{key}: {value}")


# In[28]:


weekly_diff = weekly_sales.diff().dropna()

plt.figure(figsize=(14, 5))

plt.plot(weekly_diff)

plt.title("First-Order Differenced Weekly Sales")
plt.xlabel("Date")
plt.ylabel("Differenced Sales")
plt.grid(True)

plt.show()


# In[29]:


adf_diff = adfuller(weekly_diff)

print("ADF Statistic:", adf_diff[0])
print("p-value:", adf_diff[1])

for key, value in adf_diff[4].items():
    print(f"{key}: {value}")


# In[30]:


from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize=(12, 5))

plot_acf(
    weekly_diff,
    lags=60,
    alpha=0.05
)

plt.title("ACF of Differenced Weekly Sales")
plt.show()


# In[31]:


from statsmodels.graphics.tsaplots import plot_pacf

plt.figure(figsize=(12, 5))

plot_pacf(
    weekly_diff,
    lags=60,
    method="ywm",
    alpha=0.05
)

plt.title("PACF of Differenced Weekly Sales")
plt.show()


# In[32]:


# Forecast horizon: 13 weeks
test_size = 13

train = weekly_sales.iloc[:-test_size]
test = weekly_sales.iloc[-test_size:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

print("\nTraining period:")
print(train.index.min(), "to", train.index.max())

print("\nTesting period:")
print(test.index.min(), "to", test.index.max())


# In[33]:


plt.figure(figsize=(14, 5))

plt.plot(train.index, train, label="Training Data")
plt.plot(test.index, test, label="Testing Data", color="red")

plt.title("Train-Test Split of Weekly Retail Sales")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.grid(True)

plt.show()


# In[34]:


# Seasonal naive forecast using the same week last year
seasonal_naive_forecast = weekly_sales.iloc[-test_size-52:-52].values

seasonal_naive_forecast = pd.Series(
    seasonal_naive_forecast,
    index=test.index
)

print("Seasonal naive forecast:")
display(seasonal_naive_forecast)


# In[35]:


plt.figure(figsize=(14, 5))

plt.plot(test.index, test, label="Actual Sales", marker="o")
plt.plot(
    seasonal_naive_forecast.index,
    seasonal_naive_forecast,
    label="Seasonal Naive Forecast",
    marker="o"
)

plt.title("Seasonal Naive Forecast vs Actual Sales")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.grid(True)

plt.show()


# In[36]:


mae_naive = mean_absolute_error(
    test,
    seasonal_naive_forecast
)

rmse_naive = np.sqrt(
    mean_squared_error(test, seasonal_naive_forecast)
)

mape_naive = mean_absolute_percentage_error(
    test,
    seasonal_naive_forecast
) * 100

print("Seasonal Naive Model")
print("MAE:", mae_naive)
print("RMSE:", rmse_naive)
print("MAPE:", mape_naive)


# In[37]:


from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model using training data
arima_model = ARIMA(
    train,
    order=(1, 1, 1)
)

arima_fit = arima_model.fit()

print(arima_fit.summary())


# In[38]:


arima_forecast = arima_fit.forecast(steps=len(test))

arima_forecast = pd.Series(
    arima_forecast.values,
    index=test.index
)

display(arima_forecast)


# In[39]:


plt.figure(figsize=(14, 5))

plt.plot(train.index, train, label="Training Data", alpha=0.5)
plt.plot(test.index, test, label="Actual Sales", marker="o")
plt.plot(
    arima_forecast.index,
    arima_forecast,
    label="ARIMA Forecast",
    marker="o"
)

plt.title("ARIMA Forecast vs Actual Sales")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.grid(True)

plt.show()


# In[40]:


mae_arima = mean_absolute_error(test, arima_forecast)

rmse_arima = np.sqrt(
    mean_squared_error(test, arima_forecast)
)

mape_arima = mean_absolute_percentage_error(
    test, arima_forecast
) * 100

print("ARIMA Model")
print("MAE:", mae_arima)
print("RMSE:", rmse_arima)
print("MAPE:", mape_arima)


# In[45]:


print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train date range:", train.index.min(), "to", train.index.max())

print("Starting SARIMA fit...")


# In[46]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

print("Fitting SARIMA...")
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 52),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=True, maxiter=50)
print("Fit complete.")
print(sarima_fit.summary())


# In[47]:


sarima_forecast = pd.Series(
    sarima_fit.forecast(steps=len(test)).values,
    index=test.index
)

mae = mean_absolute_error(test, sarima_forecast)
rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
mape = (np.abs((test - sarima_forecast) / test)).mean() * 100

print(f"SARIMA MAE:  {mae:,.2f}")
print(f"SARIMA RMSE: {rmse:,.2f}")
print(f"SARIMA MAPE: {mape:.2f}%")

plt.figure(figsize=(12, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual test")
plt.plot(sarima_forecast.index, sarima_forecast, label="SARIMA forecast")
plt.title("SARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Weekly sales")
plt.legend()
plt.show()


# In[51]:


comparison = pd.DataFrame({
    "Model": ["Seasonal naive", "ARIMA(1,1,1)", "SARIMA(1,1,1)(0,1,1,52)"],
    "MAE": [4_999_066.30, 11_620_019.03, 9_641_851.27],
    "RMSE": [14_067_578.21, 22_416_552.92, 16_942_877.94],
    "MAPE (%)": [5.41, 11.97, 11.13]
})
comparison = comparison.sort_values("MAPE (%)").reset_index(drop=True)
comparison


# In[52]:


# Seasonal-naive forecast errors on the test period
seasonal_naive_errors = test - seasonal_naive_forecast

print("Mean error:", seasonal_naive_errors.mean())
print("Error standard deviation:", seasonal_naive_errors.std())

plt.figure(figsize=(12, 4))
plt.plot(seasonal_naive_errors.index, seasonal_naive_errors)
plt.axhline(0, linestyle="--")
plt.title("Seasonal-Naive Forecast Errors")
plt.xlabel("Date")
plt.ylabel("Actual - Forecast")
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(seasonal_naive_errors, bins=10, edgecolor="black")
plt.title("Distribution of Seasonal-Naive Forecast Errors")
plt.xlabel("Forecast error")
plt.ylabel("Frequency")
plt.show()


# In[53]:


error_check = pd.DataFrame({
    "actual": test,
    "forecast": seasonal_naive_forecast,
    "error": seasonal_naive_errors
})

error_check["abs_error"] = error_check["error"].abs()
error_check.sort_values("abs_error", ascending=False).head(5)

