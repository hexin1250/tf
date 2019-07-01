'''
Created on 2019年6月22日

@author: ch
'''
import pandas as pd
import math as math
import numpy as np
import keras
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import load_model

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-1))
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

def get_data(dataset, scaler, encoder):
    values = dataset.values
#     values[:,1] = encoder.fit_transform(values[:, 1])
#     values[:,3] = encoder.fit_transform(values[:, 3])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[7,8,9,10,11]], axis=1, inplace=True)
    
    # split into train and test sets
    values = reframed.values
    return values

def data_X_reshape(X):
    return X.reshape(X.shape[0], 1, X.shape[1])

def get_XY(data):
    # reshape input to be 3D [samples, timestamps, features]
    return data_X_reshape(data[:, :-1]), data[:, -1]

model = load_model(filepath='/Users/ch/git/tf/resources/model/bond.md')
model.load_weights(filepath='/Users/ch/git/tf/resources/model/bond_weights.md')
scaler = MinMaxScaler(feature_range=(0,1))
# integer encode direction
encoder = LabelEncoder()
val_dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/bid-seq-test.csv', header=0, index_col=0)
val_values = get_data(val_dataset, scaler, encoder)
# split into input and outputs
val_X, val_y = get_XY(val_values)
actual_y = model.predict(val_X)
print("expected value\tpredict value")
for i in range(len(val_y)):
    print(val_y[i], actual_y[i])