import pandas as pd
import matplotlib.pyplot as plt
import requests 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import warnings
warnings.filterwarnings("ignore")



#fetching data from the server
url = "https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
param = {
            "covnert":"USD",
            "slug": "bitcoin",
            "time_end": "1601410400",
            "time_start": "1367107200"
         }
content = requests.get(url=url, params=param).json()

df = pd.json_normalize(content['data']['quotes'])


# extracting and renaming the important variables
df['Date'] = pd.to_datetime(df['quote.USD.timestamp']).dt.tz_localize(None)
df['Low'] = df['quote.USD.low']
df['High'] = df['quote.USD.high']
df['Open'] = df['quote.USD.open']
df['Close'] = df['quote.USD.close']
df['Volume'] = df['quote.USD.volume']


# drop original and redundant features
df = df.drop(columns=[
                        'time_close', 
                        'time_open', 
                        'time_high', 
                        'time_low', 
                        'quote.USD.low', 
                        'quote.USD.high', 
                        'quote.USD.open',
                        'quote.USD.close',
                        'quote.USD.volume',
                        'quote.USD.market_cap',
                        'quote.USD.tiemstamp'
                      ])

# creating a new feature for better represtentation of day-wise values
df['Mean'] = (df['Low'] + df['High'])

#cleaning the data for any NaN or Null fields
df = df.dropna()



# creating a copy for making small refinements
dataset_for_prediction = df.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['Mean'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()

# date time typecast
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']




# normalizing the exogeneous variables
from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range= (0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low','High','Open','Close','Volume','Mean']])
scaled_input = pd.DataFrame(scaled_input, index= dataset_for_prediction.index)
X = scaled_input
X.rename(columns={
    0: 'Low',
    1: 'High',
    2: 'Open',
    3: 'Close',
    4: 'Volume',
    5: 'Mean'
}, inplace= True)

print("Normalized X")
print(X.head())
