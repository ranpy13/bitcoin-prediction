from base import *
import numpy as np

from statsmodels.tsa.vector_ar.var_model import VAR

# prediction
data = df[['Mean', 'Close']]
data = np.array(data, dtype='float32')
data = data[:2500]

# exogeous varibales
exo = df[['Open']]
exo = np.array(exo, dtype='float32')
exo = exo[:2500,:]
model = VAR(data, exog= exo)
x = np.array(df['Date'])
model.index = x[:2500]
result = model.fit()
arr = np.array(df['Mean'])

N = 200     # test data
ap = arr[-N:]
z = exo[-N:, :]
a2 = result.forecast(model.endog, N, z)
act = a2[:, 1:]

# VAR model calls
print("VAR")
plt.plot(act, color= 'cyan', label= 'Predicted')
plt.plot(ap, label = 'Actual')

c = 0
for i in range(N):
    c += (act[i] - ap[i]) ** 2
c /= N

# print RMSE
print(c ** 0.5)
plt.xlabel('Days')
plt.ylabel('Value')
plt.legend()
plt.show()