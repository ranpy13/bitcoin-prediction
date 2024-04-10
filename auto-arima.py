from base import *
from pmdarima.arima import auto_arima

# running auto-arima grid search to find the best model
step_wise = auto_arima(
    train_y,
    exogenous = train_X,
    start_p= 1,
    start_q= 1,
    max_p= 7,
    max_q= 7,
    d= 1,
    max_d= 7,
    trace= True,
    m= 12,
    error_action= 'ignore',
    suppress_warnings= True,
    stepwise= True
)

# print final results
print(step_wise.summary())
