# Stock price prediction API (teaching example)

Train a model on past Tesla prices, save it, and host it as your own API on Render.
The API fetches the latest prices from Alpha Vantage and predicts the next close.

> A classroom exercise, not investment advice. Daily stock prices are very hard to predict:
> always compare the model with the naive guess "tomorrow = today".

## What's in this folder

| File | What it does |
|---|---|
| `data/tsla.csv` | Tesla daily closes, June 2010 - Dec 2022 (TidyTuesday big-tech stock prices) |
| `train.py` | Trains the model on lag_1 and lag_2 and saves it to `model.joblib` |
| `model.joblib`, `model_info.json` | The saved model and facts about it |
| `main.py` | The API (FastAPI): loads the model and answers requests |
| `requirements.txt` | The Python packages Render must install |
| `.python-version` | The Python version Render should use |
| `frontend/index.html` | Bonus: a small web page that calls your API |

## 1. Train and save the model (on your computer)

```bash
pip install -r requirements.txt
python train.py
```

This prints the model's test error next to the naive baseline and writes `model.joblib`.
Train with the versions in `requirements.txt`, so your computer and Render use the same scikit-learn.

## 2. Try the API on your computer (optional)

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs and try `/predict` with `lag_1=121.82` and `lag_2=112.71`.
For `/predict/live`, first set your key: `export ALPHAVANTAGE_KEY=yourkey` (Mac/Linux) or `set ALPHAVANTAGE_KEY=yourkey` (Windows).

## 3. Put the project on GitHub

Create a new repository on github.com and upload every file in this folder (Add file > Upload files).
Do **not** upload your API key anywhere.

## 4. Deploy on Render

1. Sign in at render.com (you can use your GitHub account).
2. Click **New > Web Service** and connect your repository.
3. Fill in:
   - **Language:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free
4. Under **Environment Variables**, add `ALPHAVANTAGE_KEY` = your Alpha Vantage key.
5. Click **Deploy Web Service** and wait for the log to say the service is live.

Your API is now at `https://<your-service-name>.onrender.com`.

## 5. Test it

- Browser: `https://<your-service-name>.onrender.com/docs`
- Postman: `GET https://<your-service-name>.onrender.com/predict/live?symbol=TSLA`
- Frontend: put your address in `API_URL` in `frontend/index.html` and open the file.

## Good to know

- **Free servers sleep.** After 15 minutes without traffic the service spins down; the next request takes about a minute.
- **Errors have real status codes.** If Alpha Vantage refuses (bad key, daily limit of 25 requests, unknown symbol), the API answers `502` with Alpha Vantage's message.
- **Version error when the model loads?** Change the Build Command to
  `pip install -r requirements.txt && python train.py` so Render trains the model itself.
- **Want to use your Decision Tree or XGBoost?** Swap the model in `train.py`. Remember the lecture:
  tree models can't predict above the highest price they saw in training.
