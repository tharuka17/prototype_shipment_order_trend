import pandas as pd
import numpy as np

# Import Data Set
df=pd.read_csv(r"delivery.csv",header = 0, parse_dates = [0])

# Drop Misssing values
df=df.dropna()

print('Shape of data',df.shape)
df.head()

# Plot Data
df['CNT'].plot(figsize=(12,5))

# Check for Stationaity of the dataset
from statsmodels.tsa.stattools import adfuller
def adf_test(dataset):
  dftest = adfuller(dataset, autolag = 'AIC')
  print("1. ADF : ",dftest[0])
  print("2. P-Value : ", dftest[1])
  print("3. Num Of Lags : ", dftest[2])
  print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
  print("5. Critical Values :")
  for key, val in dftest[4].items():
      print("\t",key, ": ", val)
      
adf_test(df['CNT'])

# Figure Out Order for ARIMA Model
from pmdarima import auto_arima
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(df['CNT'], suppress_warnings=True)
stepwise_fit.summary()

from statsmodels.tsa.arima_model import ARIMA

# Train Test Split

print(df.shape)
# Rest of data as training data
train=df.iloc[:-30]

#  Last 30 values as testing data
test=df.iloc[-30:] 
print(train.shape,test.shape)
print(test.iloc[0],test.iloc[-1])

# Train the model
from statsmodels.tsa.arima_model import ARIMA
# model=ARIMA(train['CNT'],order=(1,0,5))
model=model.fit()
model.summary()

# Make Predictions On Test Set
start=len(train)
end=len(train)+len(test)-1

pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA predictions')

pred.plot(legend=True)
test['CNT'].plot(legend=True)

pred.plot(legend='ARIMA Predictions')
test['CNT'].plot(legend=True)

test['CNT'].mean()

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred,test['CNT']))
print(rmse)

# Find the order
# model2=ARIMA(df['CNT'],order=(1,0,5))
model2=model2.fit()
df.tail()

# Predicions for the Future Dates

# Fix the date range
index_future_dates=pd.date_range(start='2018-12-30',end='2019-01-29')
#print(index_future_dates)

pred=model2.predict(start=len(df),end=len(df)+30,typ='levels').rename('ARIMA Predictions')
#print(comp_pred)

pred.index=index_future_dates
print(pred)

pred.plot(figsize=(12,5),legend=True)