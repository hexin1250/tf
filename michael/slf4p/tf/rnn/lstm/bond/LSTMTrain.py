'''
Created on 2019年6月26日

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
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_data
from michael.slf4p.tf.rnn.lstm.bond.normalizeUtil import get_XY

namelist = ["Michael", "Wendy", "Vicky", "Sam", "George", "Rose"]
for i in namelist:
    name = i
    dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/' + name + '-bid-seq-train.csv', header=0, index_col=0)
    scaler = MinMaxScaler(feature_range=(0,1))
    values = get_data(dataset, scaler, encoder=LabelEncoder())
    
    validation_ds = pd.read_csv('/Users/ch/git/tf/resources/data/' + name + '-bid-seq-val.csv', header=0, index_col=0)
    validation_values = get_data(validation_ds, scaler, encoder=LabelEncoder())
    
    train = values
    test = validation_values
    train_X, train_y = get_XY(train)
    test_X, test_y = get_XY(test)
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
#     model = load_model(filepath='/Users/ch/git/tf/resources/model/bond-' + name + '.md')
#     model.load_weights(filepath='/Users/ch/git/tf/resources/model/bond_weights-' + name + '.md')
    history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
#     pyplot.plot(history.history['loss'], label='train')
#     pyplot.plot(history.history['val_loss'], label='test')
#     pyplot.legend()
#     pyplot.show()
    
    # save model
    model.save(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond_weights.md')
    model.save(filepath='/Users/ch/git/tf/resources/model/' + name + '-bond.md')
    # val_dataset = pd.read_csv('File:/Users/ch/git/tf/test-seq.csv', header=0, index_col=0)
    val_dataset = pd.read_csv('File:/Users/ch/git/tf/resources/data/' + name + '-bid-seq-test.csv', header=0, index_col=0)
    val_values = get_data(val_dataset, scaler, encoder=LabelEncoder())
    # split into input and outputs
    val_X, val_y = get_XY(val_values)
    actual_y = model.predict(val_X)
    print(name, ":expected value\tpredict value")
    for i in range(len(val_y)):
        print(val_y[i], actual_y[i])