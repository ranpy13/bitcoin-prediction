from base import *
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

N = 2411        # test size

X = list(range(2411))

X = np.array(X)
Xtrain = X[:N]
Xtest = X[-272:]
Y = df['Mean']
Y = np.array(Y, dtype='float32')
ytrain = Y[:N]


# making test data
ytest = Y[-272:]
arr = ytest

# plot actual values
plt.plot(arr, label= 'Actual')


# grid search for optimal polynomial degree
for j in [2,3,5]:
    poly = PolynomialFeatures(degree= j)
    X_poly = poly.fit_transform(Xtrain.reshape((2411,1)))

    poly.fit(X_poly, ytrain)
    reg = LinearRegression()
    reg.fit(X_poly, ytrain)

    ypred = reg.predict(poly.fit_transform(Xtest.reshape((272,1))))
    ytest = ytest.reshape((272, 1))

    # plotting the regression
    plt.plot(ypred, label = f'Predictions with degree {j}')
    plt.legend()
    # plt.show()

    print("Polynomial Regression")
    c = 0
    for i in range(272):
        c += (ypred[i] - ytest[i]) ** 2
    c /= 272

    print(f"Degree= {i}, RMSE= {c**0.5}")

print("\n\nPolynomial Regression, depending upon no of days")
