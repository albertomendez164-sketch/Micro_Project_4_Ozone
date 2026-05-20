# ============================================
# ANA500 MICRO PROJECT 4
# Deep Learning Regression Using LSTM
# Predicting Ozone Levels
# ============================================

# STEP 1: Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# STEP 2: Load dataset

url="https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/airquality.csv"

df=pd.read_csv(url)

print(df.head())

# STEP 3: Prepare data

df=df.dropna()

features=df[['Solar.R','Wind','Temp']]
target=df['Ozone']

# scale

scalerX=MinMaxScaler()
X_scaled=scalerX.fit_transform(features)

scalery=MinMaxScaler()
y_scaled=scalery.fit_transform(
target.values.reshape(-1,1))

# reshape for LSTM

X=X_scaled.reshape(
X_scaled.shape[0],
1,
X_scaled.shape[1]
)

# split data

split=int(.8*len(X))

X_train=X[:split]
X_test=X[split:]

y_train=y_scaled[:split]
y_test=y_scaled[split:]

# STEP 4: Build model

model=Sequential()

model.add(
LSTM(
50,
activation='relu',
input_shape=(1,3)
)
)

model.add(Dense(1))

model.compile(
optimizer='adam',
loss='mse'
)

# train

history=model.fit(
X_train,
y_train,
epochs=100,
batch_size=8,
validation_split=.2
)

# predict

pred=model.predict(X_test)

pred=scalery.inverse_transform(pred)

y_actual=scalery.inverse_transform(y_test)

# metrics

rmse=np.sqrt(
mean_squared_error(
y_actual,
pred
)
)

r2=r2_score(
y_actual,
pred
)

print("RMSE:",rmse)

print("R2:",r2)

# GRAPH 1

plt.figure(figsize=(10,6))

plt.plot(
y_actual,
label="Actual"
)

plt.plot(
pred,
label="Predicted"
)

plt.legend()

plt.title(
'Actual vs Predicted Ozone'
)

plt.savefig(
'prediction.png'
)

plt.show()

# GRAPH 2

plt.figure(figsize=(10,6))

plt.plot(
history.history['loss']
)

plt.plot(
history.history['val_loss']
)

plt.legend(
['Training',
'Validation']
)

plt.title(
'LSTM Training Loss'
)

plt.savefig(
'loss.png'
)

plt.show()

print(
"Analysis complete"
)