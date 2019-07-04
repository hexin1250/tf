'''
Created on 2019年6月26日

@author: ch
'''
import pandas as pd
import numpy as np
import datetime

# convert series to supervised learning
def series_to_supervised(data, n_in=0, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(0))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(0))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis = 1)
    agg.columns = names;
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def get_data(dataset, scaler, fit=False):
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    if fit:
        scaler.fit(values)
    scaled = scaler.transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[7,8,9,10,11]], axis=1, inplace=True)
    
    # split into train and test sets
    values = reframed.values
    X = values[:, 2:6]
    Y = values[:, 0]
    return X, Y

def data_X_reshape(X):
    return X.reshape(X.shape[0], 1, X.shape[1])

def get_XY(data):
    # reshape input to be 3D [samples, timestamps, features]
    return data_X_reshape(data)

def convert_to_dataframe(arr):
    std_arr = ['', 'result','name','cpn','rate','price','yield']
    new_arr = []
    new_arr.append(std_arr)
    new_arr.append(arr)
    data = np.array(new_arr)
    return pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])

def consist_data(sys_args, order):
    arr = []
    arr.append(datetime.datetime.now())
#     arr.append(datetime.datetime.strptime('2016 07 16 01', '%Y %m %d %H'))
    arr.append(1)
    arr.append(order)
    for item in sys_args:
        arr.append(item)
    return convert_to_dataframe(arr)

def parse(x):
    return datetime.datetime.strptime(x, '%Y %m %d %H')

def split_data(filepath, topath):
    dataset = pd.read_csv(filepath, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    dataset.columns = ['result', 'name', 'cpn', 'rate', 'price', 'yield']
    dataset.index.name = 'date'
    dataset.to_csv(topath)
