import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2011-01-01'
end = '2021-12-31'


st.title('Stock Price Inclination Prediction')
st.caption("This is a web app that allows users to analyze and predict the stock price trend")


from datetime import datetime
import pandas as pd
import yfinance as yf


# Get the stock ticker from the user
user_input = st.text_input('Enter Stock Ticker', 'NFLX')


# Convert the start and end dates to datetime objects
start_date = datetime.strptime(start, '%Y-%m-%d')
end_date = datetime.strptime(end, '%Y-%m-%d')


# Use the yfinance module to get the stock data
data = yf.download(user_input, start=start_date, end=end_date)


# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)


# Describing Data Visually
st.subheader('Data From Year 2011 - 2021')
st.write(df.describe())


st.subheader('Volume Chart Vs Time Chart')
st.bar_chart(df)


# Chart Visualisation
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


# Splitting the Data into Two Parts Training & Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


data_training_array = scaler.fit_transform(data_training)


# Loading Machine Learning Model
model = load_model('keras_model.h5')


# Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# Final Prediction Graph
st.subheader('Prediction vs Orignal')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Orignal Price')
plt.plot(y_predicted, 'r', label = ' Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)











