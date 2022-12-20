# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import datetime
import pandas as pd
import yfinance as yf

# %%
start = '2011-01-01'
end = '2021-12-31'

start_date = datetime.strptime(start, '%Y-%m-%d')
end_date = datetime.strptime(end, '%Y-%m-%d')


# Use the yfinance module to get the stock data
data = yf.download('NFLX', start=start_date, end=end_date)


# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data)
df.head()

# %%
df.tail()

# %%
df = df.reset_index()
df.head()

# %%
df = df.drop(['Date','Adj Close'], axis = 1)
df.head()

# %%
plt.plot(df.Close)

# %%
df

# %%
ma50 = df.Close.rolling(50).mean()
ma50

# %%
plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma50, 'r')

# %%
ma100 = df.Close.rolling(100).mean()
ma100

# %%
plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')

# %%
ma200 = df.Close.rolling(200).mean()
ma200

# %%
plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

# %%
df.shape

# %%
# Splitting the Data into Two Parts Training & Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

# %%
data_training.head()

# %%
data_testing.head()

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# %%
data_training_array = scaler.fit_transform(data_training)
data_training_array

# %%
data_training_array.shape

# %%
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

# %%
x_train.shape

# %%
# Machine Learning Model Creation

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# %%
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
              input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))



model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))



model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))



model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))


model.add(Dense(units = 1))

# %%
model.summary()

# %%
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)

# %%
model.save('keras_model.h5')

# %%
data_testing.head()

# %%
data_training.tail(100)

# %%
past_100_days = data_training.tail(100)

# %%
final_df = past_100_days.append(data_testing, ignore_index = True)

# %%
final_df.head()

# %%
input_data = scaler.fit_transform(final_df)
input_data

# %%
input_data.shape

# %%
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

# %%
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

# %%
# Making Stock Price Prediction

y_predicted = model.predict(x_test)

# %%
y_predicted.shape

# %%
y_test

# %%
y_predicted

# %%
scaler.scale_

# %%
scale_factor = 1/0.00218431
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

# %%
plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Orignal Price')
plt.plot(y_predicted, 'r', label = ' Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# %%
plt.figure(figsize = (12,6))
plt.plot(y_predicted, 'r', label = ' Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# %%
plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Orignal Price')
plt.plot(y_predicted, 'r', label = ' Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


