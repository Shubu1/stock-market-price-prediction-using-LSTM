from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Read Six Flags Stock History CSV into DataFrame
dataset = pd.read_csv("ADBL(2013-2023).csv")
print(dataset.head())
print()
# Change Data column into Python Datetime
dataset["Date"] = pd.to_datetime(dataset.Date, format="%Y-%m-%d")
dataset.index = dataset["Date"]
# Create new dataset with only Date and Close price
data = dataset.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(dataset)), columns=["Date", "Close", "Volume"])
for i in range(0, len(data)):
    new_dataset["Date"][i] = data["Date"][i]
    new_dataset["Close"][i] = data["Close"][i]
    new_dataset["Volume"][i] = data["Volume"][i]
new_dataset.head()
# Change dataframe index identifier to data values
new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)
# Split data using an 80:20 split
final_dataset = new_dataset.values
train_data = final_dataset[0:203, :]
valid_data = final_dataset[203:, :]
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Create lstm neural network
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(units=50, return_sequences=True))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))

lstm_model.add(Dense(units=1))

lstm_model.summary()
# Train the model
lstm_model.compile(optimizer="adam", loss="mean_squared_error")
history=lstm_model.fit(x_train, y_train, epochs=50, batch_size=4)
inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)
x_test = []

for i in range(60, inputs_data.shape[0]):
    x_test.append(inputs_data[i-60:i, 0])

x_test = np.array(x_test)
# Using the model to predict x_test
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_closing_price = lstm_model.predict(x_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
# Save the model to disk for later use
# Uncomment to save model
lstm_model.save("ADBL(2013-2023).h5")
check_data = new_dataset[203:]
check_data["Predictions"] = predicted_closing_price
# Plot the result of the model predictions
plt.plot(check_data["Close"], color="Green", label="Actual Price")
plt.plot(check_data["Predictions"], color="Red", label="Predicted Price")
plt.title("stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.xticks(rotation=15)
plt.subplots_adjust(top=0.9, bottom=0.17)
plt.legend()
plt.show()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

