import requests
from config import APIkey
import numpy as np
import pandas as pd

# Getting the historical stock data from an API

class StockData:
    def __init__(self, symbol):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=60min&outputsize=full&apikey={APIkey}'
        r=requests.get(url)
        self.data = r.json()
        self.stockdate = self.data['Time Series (60min)']

        urlnews = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={APIkey}'
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
    

    def news_sentiment(self):
        newsdata = self.datanews

        return print(newsdata)






stockdata = StockData('IBM')

feature = FeatureEngineering('IBM')
feature.news_sentiment()

