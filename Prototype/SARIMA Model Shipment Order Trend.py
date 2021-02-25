import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX

#Import dataset
df = pd.read_csv(r"delivery.csv", header = 0, parse_dates = [0])

df.head()
df.tail()

df.index = df['Month']

from statsmodels.tsa.seasonal import seasonal_decompose
result_a = seasonal_decompose(df['CNT'], model='multiplicative')
result_a.plot()

model = SARIMAX(df['CNT'],order = (5,1,3), seasonal_order = (1,1,1,12))

model_fit = model.fit()

residuals = model_fit.resid

residuals.plot()

output = model_fit.forecast()
output

# Forecast for 12 days

model_fit.forecast(12)

yhat = model_fit.predict()

yhat.head()

from matplotlib import pyplot
pyplot.plot(df['CNT'])
pyplot.plot(yhat, color='red')

# Blue lines are the original values, 
# Red lines are the predicted values

# use walk forward validation