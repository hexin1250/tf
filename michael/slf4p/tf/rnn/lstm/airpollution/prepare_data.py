'''
Created on 2019年6月21日

@author: ch
'''
import pandas as pd
import datetime

def parse(x):
    return datetime.datetime.strptime(x, '%Y %m %d %H')
dataset = pd.read_csv('File:/Users/ch/git/tf/raw_data.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('/Users/ch/git/tf/pollution.csv')