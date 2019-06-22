'''
Created on 2019年6月21日

@author: ch
'''
import pandas as pd
import datetime

def parse(x):
    return datetime.datetime.strptime(x, '%Y %m %d %H')
dataset = pd.read_csv('/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/train.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['result', 'name', 'cpn', 'rate', 'price', 'yield']
dataset.index.name = 'date'
dataset = dataset[24:]
print(dataset.head(5))
dataset.to_csv('/Users/ch/git/tf/bid.csv')