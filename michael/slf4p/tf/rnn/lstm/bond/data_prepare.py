'''
Created on 2019年6月26日

@author: ch
'''
import pandas as pd
import datetime

def parse(x):
    return datetime.datetime.strptime(x, '%Y %m %d %H')

def split_data(filepath, topath):
    dataset = pd.read_csv(filepath, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    dataset.columns = ['result', 'name', 'cpn', 'rate', 'price', 'yield']
    dataset.index.name = 'date'
    dataset.to_csv(topath)

namelist = ["Michael", "Wendy", "Vicky", "Sam", "George", "Rose"]
for i in namelist:
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/train-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/data/' + i + '-bid-seq-train.csv'
    split_data(filepath, topath)
    
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/val-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/data/' + i + '-bid-seq-val.csv'
    split_data(filepath, topath)
    
    filepath = '/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/test-seq-' + i + '.data'
    topath = '/Users/ch/git/tf/resources/data/' + i + '-bid-seq-test.csv'
    split_data(filepath, topath)

