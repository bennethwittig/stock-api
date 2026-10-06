"""
Step 2: a small backend that serves the saved model as an API.

Run on your own computer:   uvicorn main:app --reload
Then open in a browser:     http://127.0.0.1:8000/docs

Endpoints
  GET /                          is the API running? what model is loaded?
  GET /predict?lag_1=..&lag_2=.. predict from two prices you type in
  GET /predict/live?symbol=TSLA  fetch the latest prices from Alpha Vantage, then predict
"""
import json
import os

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load the saved model ONCE, when the server starts
model = joblib.load("model.joblib")
with open("model_info.json") as f:
    info = json.load(f)

app = FastAPI(
    title="Stock price prediction API",
    description="Predicts the next closing price from the last two closes. "
    "A teaching example, not investment advice.",
)

# Allow web pages on other addresses (any frontend) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def predict_next(lag_1: float, lag_2: float) -> float:
    X = pd.DataFrame({"lag_1": [lag_1], "lag_2": [lag_2]})
    return round(float(model.predict(X)[0]), 2)


@app.get("/")
def home():
    return {"message": "The API is running. Open /docs to try it.", **info}


@app.get("/predict")
def predict(lag_1: float, lag_2: float):
    """Predict tomorrow's close. lag_1 = latest close, lag_2 = the close before it."""
    return {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "predicted_next_close": predict_next(lag_1, lag_2),
        "naive_forecast": lag_1,
    }


@app.get("/predict/live")
def predict_live(symbol: str = "TSLA"):
    """Get the latest two closes from Alpha Vantage, then predict the next one."""
    key = os.environ.get("ALPHAVANTAGE_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="ALPHAVANTAGE_KEY is not set on the server.")

    reply = requests.get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key},
        timeout=15,
    ).json()

    quote = reply.get("Global Quote")
    if not quote:
        # Alpha Vantage answered with a message (bad key, daily limit, unknown symbol...).
        # We pass it on with a proper error status code.
        message = reply if "Global Quote" not in reply else f"No quote found for symbol '{symbol}'."
        raise HTTPException(status_code=502, detail=message)

    last_close = float(quote["05. price"])
    previous_close = float(quote["08. previous close"])
    return {
        "symbol": quote["01. symbol"],
        "latest_trading_day": quote["07. latest trading day"],
        "last_close": last_close,
        "previous_close": previous_close,
        "predicted_next_close": predict_next(last_close, previous_close),
        "naive_forecast": last_close,
        "model": info["model"],
        "trained_on": info["trained_on"],
    }
