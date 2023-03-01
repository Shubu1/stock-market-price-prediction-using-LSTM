import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved LSTM model
lstm_model = load_model("NICA_LSTM.h5")

# Define function to get stock market data and preprocess it for the LSTM model
def get_stock_data(stock_name):
    dataset = pd.read_csv(stock_name)
    dataset["Date"] = pd.to_datetime(dataset.Date, format="%m/%d/%Y")
    dataset.index = dataset["Date"]
    data = dataset.sort_index(ascending=True, axis=0)
    new_dataset = pd.DataFrame(index=range(0, len(dataset)), columns=["Date", "Close"])
    for i in range(0, len(data)):
        new_dataset["Date"][i] = data["Date"][i]
        new_dataset["Close"][i] = data["Close"][i]
    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)
    final_dataset = new_dataset.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(final_dataset)
    return scaler, scaled_data, new_dataset

# Define function to predict future stock prices using the LSTM model
def predict_stock_prices(model, scaler, data):
    inputs_data = data[len(data)-60:].values
    inputs_data = inputs_data.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)
    x_test = []
    for i in range(60, inputs_data.shape[0]):
        x_test.append(inputs_data[i-60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted_closing_price = model.predict(x_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
    return predicted_closing_price

# Define the options for the stock markets
stock_options = {
    "Nepal Stock Exchange": "NEPSE.csv",
    "New York Stock Exchange": "NYSE.csv",
    "London Stock Exchange": "LSE.csv"
}

# Set the default stock market to Nepal Stock Exchange
default_stock = "NEPSE.csv"

# Create the Streamlit app
st.title("Stock Price Prediction")

# Create a dropdown menu for the stock market selection
selected_stock = st.selectbox("Select a stock market", list(stock_options.keys()), index=list(stock_options.keys()).index("Nepal Stock Exchange"))

# Get the stock data for the selected stock market
scaler, scaled_data, data = get_stock_data(stock_options[selected_stock])

# Use the LSTM model to predict future stock prices
predicted_prices = predict_stock_prices(lstm_model, scaler, data)

# Plot the actual and predicted stock prices
fig, ax = plt.subplots()
ax.plot(data.index, data["Close"], label="Actual Price")
ax.plot(data.index[-len(predicted_prices):], predicted_prices, label="Predicted Price")
ax.set_title(f"{selected_stock} Stock Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
plt.xticks(rotation=15)
plt.subplots_adjust(top=0.9, bottom=0.17)

# Show the plot in the Streamlit app
st.pyplot(fig)
