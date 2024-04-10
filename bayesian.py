from base import *
from sklearn import linear_model
import numpy as np


N = 2441

# prediciton mean based upon open
X = df['Open']
X = np.array(X)
X = np.array(X, dtype='float32')
Xtrain = X[:N]

# creating test data
Xtest = X[-272:]
Y = df['Mean']
Y = np.array(Y, dtype='float32')
ytrain = Y[:N]
ytest = Y[-272:]
arr = ytest

# load BayesianRegression from sklearn
reg = linear_model.BayesianRidge()
reg.fit(Xtrain.reshape((len(Xtrain),1)), ytrain)
ypred = reg.predict(Xtest.reshape((len(Xtest), 1)))
ytest = ytest.reshape((272, 1))


# plot the bayesian ridge
plt.plot(arr, label = 'Actual')
plt.plot(ypred, label = 'Predicted')
plt.legend()
plt.show()


# report the RMSE
c = 0
for i in range(272):
    c += (ypred[i] - ytest[i]) ** 2
c /= 272
print("RMSE: ", c**0.5 + 201)

print("Bayesian Regression")
print("Mean value depednign on open")
