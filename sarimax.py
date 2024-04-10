from base import *


# Init the best SARIMAX model
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(
    train_y,
    exog=train_X,
    order=(0,1,1),
    seasonal_order =(0, 0, 1, 12),
    enforce_invertibility=False,
    enforce_stationarity=False
)

# training the model
results = model.fit()

# get predictions
predictions = results.predict(start =train_size, end=train_size+test_size-2,exog=test_X)


# setting up for plots
act = pd.DataFrame(scaler_output.iloc[train_size:, 0])
predictions=pd.DataFrame(predictions)
predictions.reset_index(drop=True, inplace=True)
predictions.index=test_X.index
predictions['Actual'] = act['BTC Price next day']
predictions.rename(columns={0:'Pred', 'predicted_mean':'Pred'}, inplace=True)


# post-processing inverting normalization
testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])

# prediction plots
plt.figure(figsize=(20,10))
plt.plot(predictions.index, testActual, label='Pred', color='blue')
plt.plot(predictions.index, testPredict, label='Actual', color='red')
plt.legend()
plt.show()

# print RMSE
from statsmodels.tools.eval_measures import rmse
print("RMSE:",rmse(testActual, testPredict))