import pandas as pd

df = pd.read_csv(r"C:\Users\Tharuka\Downloads\Forecasting Resources (ST Academy)\Data + Code\delivery.csv", header = 0, parse_dates = [0])
df.head()

# Creating a Lag Value

df['t'] = df['CNT'].shift(7)
df.head()
df.tail()
df.shape
df.shape[0]

# Train Test Split

train_size = int(df.shape[0]*0.8)
train_size
train, test = df[0:train_size], df[train_size:]
train.shape
test.shape
train.head()
train_x,train_y = train['t'],train['CNT']
test_x,test_y = test['t'],test['CNT']

# Walk Foward Validation

predictions = test_x.copy()
print(predictions)
print(test_y)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_y,predictions)
mse
from matplotlib import pyplot
pyplot.plot(test_y)
pyplot.plot(predictions, color = 'red')