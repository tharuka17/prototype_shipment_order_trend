import pandas as pd

#Import dataset
df = pd.read_csv(r"delivery.csv", header = 0, parse_dates = [0])

df['t'] = df['CNT'].shift(1)
df.head()

# Train Test Split
train_size = int(df.shape[0]*0.8)
train_size
train, test = df[0:train_size], df[train_size:]
train.head()
train_x,train_y = train['t'],train['CNT']
test_x,test_y = test['t'],test['CNT']

df['Resid'] = df['CNT'] - df['t']

df.head()

# Building the mode for Residuals
# For Moving average, run autoregression for residuals

train_size = int(df.shape[0]*0.8)
train, test = df[0:train_size], df[train_size:]

train.head()

from statsmodels.tsa.ar_model import AR
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)

model = AR(train)
model_fit = model.fit()
model_fit.k_ar
model_fit.params
pred_resid = model_fit.predict(start = len(train), end = len(train) + len(test)-1)
pred_resid
df.t[df.shape[0]-7:]
predictions = df.t[df.shape[0]-7:] + pred_resid

predictions

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_y, predictions)
mse

from matplotlib import pyplot
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')