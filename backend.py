import requests
from config import APIkey, newsAPI
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib.pyplot as plt

symbol = input('Enter Stock Ticker: ')

# Getting the historical stock data from an API

class StockData:
    def __init__(self, symbol):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&outputsize=full&apikey={APIkey}'
        r=requests.get(url)
        self.data = r.json()
        self.stockdate = self.data['Time Series (60min)']
        
        date = dt.date.today()
        days_before = (dt.date.today()-dt.timedelta(days=30))   
        urlnews = f'https://eodhd.com/api/sentiments?s={symbol}&from={days_before}&to={date}&&api_token={newsAPI}'
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
        newsdata = self.datanews[f'{symbol}.US']
        newsdates = [i.get('date') for i in newsdata]
        senti = [i.get('normalized') for i in newsdata]

        for i in range(14, 252, 16):
            self.df.loc[i-14:i, 'News Senti'] = senti[round((i/14)-1)]

        return self.df
        


stockdata = StockData(symbol)
feature = FeatureEngineering(symbol)
df = feature.news_sentiment(symbol)

df.dropna(inplace=True)
y = df['Prices']
x=df[['RSI', 'News Senti']]
print(len(x))
# Creating Train and Test split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

model = linear_model.LinearRegression()
model.fit(x, y)

# Predicting stock price and findin r2 score of model
predictedprice= model.predict(X_test)
print(predictedprice)
print("R2 score =", model.score(x, y))
