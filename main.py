from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
import os
import pandas as panda
import matplotlib.pyplot as plotter
import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer

finwiz_url = 'https://finviz.com/quote.ashx?t='

final_news_table = {}
tickers = ['TWTR','GOOGL']

for tck in tickers:
    final_url = finwiz_url + tck
    req = Request(url=final_url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
    response = urlopen(req)
    html_res = BeautifulSoup(response,features="html.parser")
    news_table = html_res.find(id='news-table')
    final_news_table[tck] = news_table

parsed_news =[]

for file_name, news_table in final_news_table.items():
    for name_tr in news_table.findAll('tr'):
        text = name_tr.a.get_text()
        total_date = name_tr.td.text.split()
        if len(total_date) == 1:
            time  = total_date[0]
        else:
            date = total_date[0]
            time = total_date[1]
            
        ticker = file_name.split('_')[0]
        parsed_news.append([ticker,date,time,text])

sentiment = SentimentIntensityAnalyzer()

columns = ['ticker','date','time','headline']

parsed_and_scored_news = panda.DataFrame(parsed_news,columns=columns)
scores = parsed_and_scored_news['headline'].apply(sentiment.polarity_scores).tolist()

scores_df = panda.DataFrame(scores)

parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

parsed_and_scored_news['date'] = panda.to_datetime(parsed_and_scored_news.date).dt.date

print(type(parsed_and_scored_news['date']))

plotter.rcParams['figure.figsize'] = [10,6]

mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()

mean_scores = mean_scores.unstack()

mean_scores = mean_scores.xs('compound', axis="columns").transpose()

mean_scores.plot.bar()
plotter.grid()
plotter.show()
