#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import math
import datetime as dlt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, explained_variance_score, r2_score,
    mean_poisson_deviance, mean_gamma_deviance, accuracy_score
)
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import cycle
import plotly.express as px
from plotly.subplots import make_subplots
df = pd.read_csv(r"C:\Users\hplap\Downloads\Telegram Desktop\btc.csv")
df.head()
print('Total number of days present in the dataset',df.shape[0])
print('Total number of fields present in the dataset',df.shape[1])
df.shape
df.tail()
df.describe()
print('NULL VALUES',df.isnull().values.sum())
closedf = df [['Date','Close']]
print("Shape of the close Dataframe",closedf.shape)
fig = px.line(closedf, x = closedf.Date, y=closedf.Close, labels = {'date':'Date','close':"Close Stock"})
fig.update_traces(marker_line_width = 2,opacity = 0.8,marker_line_color = 'orange')
fig.update_layout(title_text = 'Whole period timeframe of Bitcoin close price 2014-2022',plot_bgcolor='white'
                 ,font_size = 15,font_color = 'black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
fig.show()

# Deleting the column and normalizing using MinMax Scaler

del closedf['Date']
scaler = MinMaxScaler(feature_range = (0,1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)

# Now we training set as 60% and test 40% of the datasets
training_size = int(len(closedf)*0.60)
test_size = len(closedf)-training_size
train_data,test_data = closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ",train_data.shape)
print("test_data :", test_data.shape)


def create_dataset(dataset, time_step = 1):
    dataX , dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]  # i=0 , 0,1,2,3--------99  100
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train,y_train = create_dataset(train_data, time_step)
X_test , y_test = create_dataset(test_data, time_step)


print('X_train :',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:',X_test.shape)
print('y_test' ,y_test.shape)
# reshape input to be [sample,time steps, feature] which is requires for LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], 1)

print('X_train:', X_train.shape)
print('X_test:',X_test.shape)



model = Sequential()
model.add(LSTM(10,input_shape=(None,1),activation = 'relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

history = model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs=150,batch_size = 32,verbose = 1)



# prediction and perform metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict.shape , test_predict.shape


train_predict = scaler.inverse_transform(train_predict)
test_predict  = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

# METRICS RMSE
print('Train data RMSE:', math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("-----------------------------------------------------------------------------")
print("Test data RMSE:", math.sqrt(mean_squared_error(original_ytest,test_predict)))


# In[ ]:




