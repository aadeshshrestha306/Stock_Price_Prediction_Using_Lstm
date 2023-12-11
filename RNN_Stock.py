# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import mplfinance as mpf
import requests
from sklearn.metrics import mean_squared_error

# %%
#API Key

key = "XTL1OOAD2EJD1V6Y"

# %%
#API

base_url = "https://www.alphavantage.co/query"
function = "TIME_SERIES_DAILY"
symbol = "TSLA"
outputsize = "full"  
interval = "daily"  

# %%
#API Parameters

params = {
    "function": function,
    "symbol": symbol,
    "outputsize": outputsize,
    "apikey": key,
    #"interval": interval,  
}

# %%
#Calling API

response = requests.get(base_url, params=params)

# %%
#Converting fetched data into json format

data = response.json()

# %%
data

# %%
ts_data = data["Time Series (Daily)"]

# %%
#json to dataframe

df = pd.DataFrame.from_dict(ts_data, orient="index")  

# %%
df.head()

# %%
#renaming dataframe columns 

columns = ['open', 'high', 'low', 'close', 'volume']  

df.columns = columns 

# %%
df.head()

# %%
df.info()

# %%
#datatype conversion from object(string) to float/int(numeric)

df = df.apply(pd.to_numeric, errors='coerce')

# %%
df.info()

# %%
sns.set_theme(style="dark")

# %%
#creating new column of "datetime" datatype

df['Date'] = pd.to_datetime(df.index)  

# %%
df1 = df.sort_values('Date')

# %%
df1.head()

# %%
#lineplotting closing value of last 100 days

plt.figure(figsize=(18,9))
sns.lineplot(data=df1, x='Date', y='close')
plt.xlabel('')
plt.ylabel('Closing Value')
plt.title("Tesla Stock Price")
plt.tight_layout()
plt.show()

# %%
df1['Date']=pd.to_datetime(df1['Date'])

# %%
df1.index=df1['Date']

# %%
df1 = df1.drop(columns='Date')

# %%
mpf.plot(df1, type='candle', style='yahoo', mav=(5,20), figratio=(18,9))  #candlestick plotting using mpl finance lib
plt.show()

# %% [markdown]
# # Machine Learning

# %%
total_rows = len(df1)

total_rows

# %% [markdown]
# ### Seperating training_size and test_size 
# 

# %%
train_size = int(0.7*total_rows)

# %%
test_size = int(0.3*total_rows)

# %%
train_size

# %%
test_size

# %% [markdown]
# ### Spliting data 

# %%
train_data = df1[:train_size]

# %%
test_data = df1[train_size:(train_size+test_size)]

# %%
test_data.head()

# %%
train_data.head()

# %%
len(train_data)

# %%
len(test_data)

# %%
#from tensorflow.keras.callbacks import EarlyStopping

# %%
train_data.info()

# %%
test_data.info()

# %% [markdown]
# ## Moving Average Calculation and Visualization

# %%
close_price = train_data['close'].values

# %%
close_price

# %%
window = 20

ma = df1['close'].ewm(span=window, adjust=True).mean()

# %%
ma

# %%
ma.info()

# %%
ind = ma.index

# %%
ind

# %%
fil_d = df1[df1.index.isin(ind)]

# %%
fil_d

# %%
avg = ma.values

# %%
avg

# %%
mav = pd.DataFrame(avg, columns=['price'], index=ma.index)

# %%
mav

# %%
plt.figure(figsize=(18,9))
sns.lineplot(data=mav, x='Date', y='price', color='blue', label='EMA')
sns.lineplot(data=fil_d, x='Date', y='close', color='red', label='Actual Closing Price')
plt.legend()
plt.tight_layout()
plt.show()

# %%
#normalising data 

scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# %%
scaled_train_data

# %%
scaled_test_data

# %%
x_train = []
y_train = []

# %%
sequence_length = 10

# %%
for i in range(len(scaled_train_data)-sequence_length):
    x_train.append(scaled_train_data[i:i+sequence_length])
    y_train.append(scaled_train_data[i+sequence_length])

# %%
x_test = []
y_test = []

# %%
for i in range(len(scaled_test_data)-sequence_length):
    x_test.append(scaled_test_data[i:i+sequence_length])
    y_test.append(scaled_test_data[i+sequence_length])

# %%
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

# %%
x_train

# %%
y_train

# %%
x_train.shape

# %%
y_train.shape

# %%
x_test.shape

# %%
y_test.shape

# %%
len(df1.columns)

# %%
#Neural Network model

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(100, activation='relu', input_shape=(sequence_length, 5)),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(5)
])

# %%
#compile the model

model.compile(optimizer='adam', loss='mean_squared_error')

# %%
#train the model

model.fit(x_train, y_train, epochs=50, batch_size=16)

# %%
test_loss = model.evaluate(x_test)

test_loss

# %%
history = model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.3)

# %%
# Plot the training loss and validation loss over epochs

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
#predict using the created neural network

predicted_prices = model.predict(x_test)

# %%
predicted_prices

# %%
#retransforming the scaled data into real data

predictions_x = scaler.inverse_transform(predicted_prices)

# %%
predictions_y = scaler.inverse_transform(y_test)

# %%
#cost function and error

rmse = np.sqrt(mean_squared_error(y_test, predicted_prices))

# %%
rmse

# %%
scaled_train_data.shape

# %%
predictions_x.shape

# %%
predicted_prices.shape[0]

# %%
train = df1[:train_size+1]
valid = df1[train_size:train_size + len(predicted_prices)]
valid.loc[:, 'Predictions'] = predictions_x[:, 3]


# %%
train

# %%
valid

# %%
#visualising real stock value with the predicted values

plt.figure(figsize=(18,9))
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(valid['close'])
plt.plot(valid['Predictions'])
plt.legend([ 'Actual','Predictions'])
plt.show()

# %%




