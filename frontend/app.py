import streamlit as st
import requests

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")

st.title("📈 Stock Price Predictor")
st.write("Enter a stock ticker to get the model's prediction for the next closing price.")

# 1. Base URL for your deployed Render API (replace with your actual Render URL)
# Note: Ensure there are no spaces or extra dashes in the URL
api_url = "https://your-service.onrender.com"

# 2. User Input
symbol = st.text_input("Stock symbol", "TSLA").upper().strip()

# 3. Action Button
if st.button("Predict next close"):
    if not symbol:
        st.warning("Please enter a valid stock symbol.")
    else:
        with st.spinner(f"Fetching prediction for {symbol}..."):
            try:
                # Send GET request to /predict/live with ?symbol=...
                response = requests.get(
                    f"{api_url}/predict/live",
                    params={"symbol": symbol},
                    timeout=10
                )

                # Check if request succeeded
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract prediction value from response dictionary
                    # Adjust key name if your API returns a different key (e.g. "prediction")
                    prediction = data.get("predicted_price") or data.get("prediction") or data.get("predicted_next_close")
                    
                    if prediction is not None:
                        st.metric(
                            label=f"Predicted Next Close for {symbol}",
                            value=f"${float(prediction):,.2f}"
                        )
                    else:
                        st.success("API Response received:")
                        st.json(data)

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API at {api_url}.")
                st.info("Check if your Render service is live and that the URL is correct.")
