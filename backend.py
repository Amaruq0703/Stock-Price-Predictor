import requests
from config import APIkey
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


symbol = input('Enter Stock Ticker: ')

# Getting the historical stock data from an API

class StockData:
    def __init__(self, symbol):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&outputsize=full&apikey={APIkey}'
        r=requests.get(url)
        self.data = r.json()
        self.stockdate = self.data['Time Series (60min)']

        urlnews = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=1000&apikey={APIkey}'
        rnews = requests.get(urlnews)
        self.datanews = rnews.json()
        
# Creating stock price indicators 

class FeatureEngineering(StockData):
    def __init__(self, symbol):
        super().__init__(symbol)

    # Calculating RSI values for 14 period window

    def RSI(self):

        #Getting usable data from API
        datelist = self.stockdate.keys()
        stockhourlist = [self.stockdate.get(date) for date in datelist]
        stockopen = [stockhour.get('1. open') for stockhour in stockhourlist]
        stockopenfloat = [float(i) for i in stockopen]
        #Creating Dataframe to store values
        self.df = pd.DataFrame({'Prices': stockopenfloat, 'Stock time' : datelist} )
        window = 14

        #Calculating RSI and smoothing it with moving averages
        delta = self.df['Prices'].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        RolUp = pd.Series(dUp).rolling(window).mean()
        RolDown = pd.Series(dDown).rolling(window).mean().abs()
        RS = RolUp / RolDown
        rsi= 100.0 - (100.0 / (1.0 + RS))
        self.df['RSI'] = rsi
        
        return self.df
    
    #Getting News sentiment value for specific stock ticker at that current time

    def news_sentiment(self, symbol):

        #Getting Sentiment value from API
        self.df = feature.RSI()
        newsdata = self.datanews['feed']
        newstickers = [i.get('ticker_sentiment') for i in newsdata]
        for i in newstickers:
            newstickerlist = [list(l.values()) for l in i]
        tickersentidict = {x[0]:x[2] for x in newstickerlist}

        #Storing Sentiment value in dataframe
        if symbol in tickersentidict.keys():
            self.df['News Senti'] = tickersentidict[symbol]
        else:
            self.df['News Senti'] = 0
        
        return self.df
        


stockdata = StockData(symbol)
feature = FeatureEngineering(symbol)
df = feature.news_sentiment(symbol)

x = df['Prices']
y=['RSI', 'News Senti']


# Creating Train and Test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

