# %%
# import
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime as dt
from tensorflow.keras.utils import plot_model
print(tf.__version__)
print(tf.keras.__version__)

# %matplotlib inline

# %%
# load data
raw = pd.read_csv('/Users/ryan/Documents/Python/ML-Homework-practice/Hw6/nyse/prices-split-adjusted.csv')
raw.head()

# %%
# preprosseing
# choose one stock and remove text colume

def preprossing(raw, ticker):
    temp = raw[raw['symbol'] == ticker]
    temp = temp.drop(['symbol', 'date'], axis=1)
    return np.array(temp)
# there are on average 20 trading days in a month
# I use all features from the first 15 day of the month as label to predict the close price of the last 5 days
# Drop surplus row
# randomly and respectively select 10% of total data to be testing set and validation set


# %%
# create train/test data
temp = preprossing(raw, 'MSFT')
min = temp.min(0)
max = temp.max(0)
temp = (temp - min) / (max - min)
temp.shape
x = []
y = []
for day in range(len(temp) - 19):
    x.append(temp[day:day + 15, :])
    y.append(temp[day + 15:day + 20, 1])
x = np.array(x).reshape((-1, 15, 5))
y = np.array(y).reshape((-1, 5, 1))
x_test = x[-150:]
y_test = y[-150:]
x_val = x[-320:-170]
y_val = y[-320:-170]
x_train = x[:-340]
y_train = y[:-340]

x_train.shape
# %%
# build LSTM model
inputs = layers.Input(shape=(15, 5))
x = layers.LSTM(16, dropout=0.3, recurrent_dropout=0.0, return_sequences=False)(inputs)
x = layers.RepeatVector(5)(x)
x = layers.LSTM(16, dropout=0.3, recurrent_dropout=0.0, return_sequences=True)(x)
outputs = layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()
plot_model(model, show_shapes=True)
# %%
# fit model
model.compile(loss='mse', optimizer='Adam')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5, verbose=1)
model.fit(x_train, y_train, batch_size=100, epochs=1000, validation_data=(x_val, y_val), callbacks=[callback])
# %%
# predict and plot
prep_test = model.predict(x_test)
prep_val = model.predict(x_val)
plt.plot(np.arange(y_test.shape[0]), y_val[:, 1, :], color='b')
plt.plot(np.arange(y_test.shape[0]), prep_val[:, 1, :], color='r')

'''
the predicted price is much more fluctuating than the real price.
which is more likely to be a moving sum of the real price instead of a index that could predict
the result is not surprising since "no one could really predict stock price"
'''
