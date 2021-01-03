import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.ar_model import AR

plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

df = pd.read_csv(r"delivery.csv")
df['date']=pd.to_datetime(df['date'])
# df.set_index('date', inplace=True)

df.plot()

pd.plotting.lag_plot(df['deliveries'])

pd.plotting.autocorrelation_plot(df['deliveries'])

#create train/test datasets
X = df['deliveries'].dropna()
# print(len(X))
train_data, test_data = X[1:len(X)-12], X[len(X)-12:]
# train_data = X[1:len(X)-7]
# test_data = X[X[len(X)-7:]]

#train the autoregression model
model = AR(train_data)
model_fitted = model.fit()

print('The lag value chose is: %s' % model_fitted.k_ar)
print('The coefficients of the model are:\n %s' % model_fitted.params)

# make predictions 
predictions = model_fitted.predict(
    start=len(train_data), 
    end=len(train_data) + len(test_data)-1, 
    dynamic=False)

# create a comparison dataframe
compare_df = pd.concat(
    [df['deliveries'].tail(12),
    predictions], axis=1).rename(
    columns={'deliveries': 'actual', 0:'predicted'})

#plot the two values
compare_df.plot()

from sklearn.metrics import r2_score

r2 = r2_score(df['deliveries'].tail(12), predictions)
print(r2)

# try:
#     model_fitted.forecast('2020-06-01')
# except KeyError as e:
#     print(e)