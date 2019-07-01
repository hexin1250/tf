'''
Created on 2019年6月21日

@author: ch
'''
import pandas as pd
import datetime

def parse(x):
    return datetime.datetime.strptime(x, '%Y %m %d %H')

def split_data(filepath, topath, splitpath=None, split_test=False):
    dataset = pd.read_csv(filepath, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    dataset.columns = ['result', 'name', 'cpn', 'rate', 'price', 'yield']
    dataset.index.name = 'date'
    if split_test:
        val_dataset = dataset[:9500]
        test_dataset = dataset[9500:]
        val_dataset.to_csv(topath)
        test_dataset.to_csv(splitpath)
    else:
        dataset.to_csv(topath)
# dataset = pd.read_csv('/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/train.data', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset = pd.read_csv('/Users/ch/eclipse-workspace/neural-rnn/src/main/resources/data/val-seq.data', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['result', 'name', 'cpn', 'rate', 'price', 'yield']
dataset.index.name = 'date'
dataset = dataset[24:]
ds1 = dataset[:24]
ds2 = dataset[24:]
print(len(ds1), len(ds2), len(dataset))
print(dataset.head(5))
dataset.to_csv('/Users/ch/git/tf/resources/data/bid-seq-val.csv')