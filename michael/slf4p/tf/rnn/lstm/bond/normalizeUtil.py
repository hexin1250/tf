'''
Created on 2019年6月26日

@author: ch
'''
import pandas as pd

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
