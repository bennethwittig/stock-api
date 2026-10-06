"""
Step 1: train a model on past prices and SAVE it to a file.

Data:  Tesla daily closing prices, June 2010 - Dec 2022
       (TidyTuesday big-tech stock prices, the data from the time series lecture).
Model: predicts tomorrow's close from the last two closes (lag_1, lag_2).

Run:   python train.py
Makes: model.joblib      (the trained model)
       model_info.json   (facts about the model that the API shows)
"""
import json

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["lag_1", "lag_2"]

# 1. Load the data and build the lag features (as in notebook 6)
df = pd.read_csv("data/tsla.csv", parse_dates=["date"]).sort_values("date")
df["lag_1"] = df["close"].shift(1)  # yesterday's close
df["lag_2"] = df["close"].shift(2)  # the day before
df = df.dropna()

# 2. Train on the past, test on the future: 80/20, never shuffled
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

model = LinearRegression()
model.fit(train[FEATURES], train["close"])
pred = model.predict(test[FEATURES])
naive = test["lag_1"]  # naive baseline: tomorrow = today


def rmse(actual, forecast):
    return mean_squared_error(actual, forecast) ** 0.5


print(f"Test period: {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Model           RMSE {rmse(test['close'], pred):6.2f}   MAE {mean_absolute_error(test['close'], pred):6.2f}")
print(f"Naive baseline  RMSE {rmse(test['close'], naive):6.2f}   MAE {mean_absolute_error(test['close'], naive):6.2f}")

# 3. Refit on ALL the data, so the saved model has seen the most recent prices
model.fit(df[FEATURES], df["close"])

# 4. Save the model and a few facts about it
joblib.dump(model, "model.joblib")
info = {
    "model": "LinearRegression on lag_1 and lag_2",
    "trained_on": "TSLA daily close",
    "data_from": str(df["date"].min().date()),
    "data_to": str(df["date"].max().date()),
    "test_rmse_model": round(rmse(test["close"], pred), 2),
    "test_rmse_naive": round(rmse(test["close"], naive), 2),
}
with open("model_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("Saved model.joblib and model_info.json")
