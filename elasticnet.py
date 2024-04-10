from base import *
from sklearn.linear_model import ElasticNet
import numpy as np

# N --> train size
N = 2411

# prediciton mean based upon open
X=df['Open']
X=np.array(X)
X=np.array(X,dtype='float32')
Xtrain=X[:N]

#creating test data
Xtest=X[-272:]
Y=df['Mean']
Y=np.array(Y,dtype='float32')
ytrain=Y[:N]
ytest=Y[-272:]
arr=ytest



# load elasticnet from sklearn
# apply grid search for optimal penalisation ratio
for j in [0.1, 0.5, 0.9]:
    reg = ElasticNet(l1_ratio= j, random_state= None)
    reg.fit(Xtrain.reshape((len(Xtrain), 1)), ytrain)
    ypred = reg.predict(Xtest.reshape((len(Xtest), 1)))
    ytest = ytest.reshape((272, 1))


    plt.plot(arr, label = 'Actual')
    plt.plot(ypred, label= 'Predicted')
    plt.legend()
    plt.show()


    # report the RMSE
    c = 0
    for i in range(272):
        c += (ypred[i] - ytest[i]) ** 2
    c /= 272

    print("RMSE: ", c**0.5 + 201)
    
    print("Linear Regression with Elastic Net")
    print("Mean value depending on open")
